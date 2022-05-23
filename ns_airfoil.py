import os
path = os.path.dirname(os.path.abspath(__file__))
import sys
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.core import domain
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize
from natsort import natsorted
from configparser import ConfigParser

filename = path + '/nsvp_options.cfg'
config = ConfigParser()
config.read(str(filename))

# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64

restart = config.getboolean('parameters', 'restart')
run_name = str(config.get('parameters', 'run_name'))
if not os.path.exists(path + '/' + run_name):
    os.makedirs(path + '/' + run_name, exist_ok=True) 


Reynolds = config.getfloat('parameters', 'Reynolds')
nu = 1 / Reynolds
U0 = config.getfloat('parameters', 'U0')
tau = config.getfloat('parameters', 'tau')
delta = config.getfloat('parameters', 'delta')

scale = config.getfloat('parameters', 'scale')
rotation = config.getfloat('parameters', 'rotation')

max_timestep = config.getfloat('parameters', 'max_dt')
stop_sim_time = config.getfloat('parameters', 'T') + 0.1

# Bases
coords = d3.CartesianCoordinates('y', 'x')
coords.name = coords.names
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
bases = (ybasis, xbasis)
ey, ex = coords.unit_vector_fields(dist)
y, x = ybasis.global_grid(), xbasis.global_grid()
y_g = y * np.ones_like(x)
x_g = x * np.ones_like(y)
dy = lambda A: d3.Differentiate(A, coords.coords[0])
dx = lambda A: d3.Differentiate(A, coords.coords[1])

# Fields
u = dist.VectorField(coords, name='u', bases=bases)
p = dist.Field(name='p', bases=bases)
tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(ybasis))

U = dist.VectorField(coords, name='U', bases=bases)
U['g'][1] = U0
F = dist.VectorField(coords, name='F', bases=bases)

# Mask function (airfoil geometry)
#################################################################
domain = domain.Domain(dist, bases)
slices = dist.grid_layout.slices(domain, scales=1)
phi = dist.Field(name='phi', bases=bases)
if True:

    from phi_evp import construct_phi
    a0 = 0.0
    a = [a0, 1.0, (0.4)]

    rot_exp = np.exp(1j*(rotation / 180 * np.pi))
    a = [ak*scale*rot_exp for ak in a]

    phi_g, rs = construct_phi(a, delta, dist, coords, bases)
    phi['g'] = phi_g
    dist.comm.Barrier()
    phi.change_scales(1)
    phi_g_global = phi.allgather_data('g')

    # Plot phi
    if dist.comm.rank == 0:
        plt.figure(figsize=(6, 4))
        res = 8
        pc = plt.pcolormesh(x.ravel(), y.ravel(), phi_g_global, cmap='seismic', shading='gouraud', rasterized=True)
        plt.colorbar(pc)
        # plt.fill(rx, ry, color='black')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title("phi mask function")
        plt.tight_layout()
        plt.savefig(run_name + '/phi.png', dpi=200)
        plt.close()
    
logger.info('Done. Mask function phi constructed and plotted.')
rx, ry = zip(*rs)
dist.comm.Barrier()
#################################################################


lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + ex*lift(tau_u1,-1) # First-order reduction

problem = d3.IVP([u, p, F, tau_p, tau_u1, tau_u2], namespace=locals())

problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2, -1) = -u@grad(u) - phi*(u + U)/tau")
problem.add_equation("F = integ(phi*(u + U)/tau)")

problem.add_equation("(u @ ex)(x='left') = 0")
problem.add_equation("(u @ ey)(x='left') = 0")
problem.add_equation("(u @ ex)(x='right') = 0")
problem.add_equation("(u @ ey)(x='right') = 0")
# problem.add_equation("(dx(u) @ ey)(x='left') = 0")

problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

if (restart):
    # u['g'] = U['g'].copy()
    fhmode = 'overwrite'
else:
    load_dir = path + '/' + run_name + "/checkpoints/"
    checkpoint_names = [name for name in os.listdir(load_dir) if '.h5' in name]
    last_checkpoint = natsorted(checkpoint_names)[-1]
    solver.load_state(load_dir + last_checkpoint, -1)
    fhmode = 'append'

checkpoints = solver.evaluator.add_file_handler(path + '/' + run_name + "/checkpoints", max_writes=1, sim_dt=5, mode=fhmode)
checkpoints.add_tasks(solver.state, layout='g')

CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.8, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt((u@ex)**2 + (u@ey)**2), name='u_mag')

ts = []
max_us = []
dist.comm.Barrier()

while solver.proceed:
    # timestep = CFL.compute_timestep()
    timestep = max_timestep
    solver.step(timestep)
    if (solver.iteration-1) % 10 == 0:
        max_u_mag = flow.max('u_mag')
        max_us.append(max_u_mag)
        ts.append(solver.sim_time)
        Fd = (F @ ex).evaluate()['g'].flat[0]
        Fl = (F @ ey).evaluate()['g'].flat[0]
        # phi['g'] *= 1.01
        logger.info('Iteration=%i; Time=%e; dt=%e; max(|u|)=%f; lift=%f; drag=%f; tau=%f' %(solver.iteration, solver.sim_time, timestep, max_u_mag, Fl, Fd, tau))

if (dist.comm.rank == 0):
    plt.plot(ts, max_us)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$L-\infty [\vec{u}]$")
    plt.savefig(run_name + '/max_u.png')
    plt.show()
dist.comm.Barrier()

# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()

ux = ((u + U) @ ex).evaluate()
ux.change_scales(1)
ugx = ux.allgather_data('g')

uy = ((u + U) @ ey).evaluate()
uy.change_scales(1)
ugy = uy.allgather_data('g')
mag_u = np.sqrt(ugx**2 + ugy**2)

# Plot
if dist.comm.rank == 0:
    plt.figure(figsize=(6, 4))
    res = 8
    # plt.pcolormesh(x.ravel(), y.ravel(), phi_g_global, cmap='viridis', shading='gouraud', rasterized=True)
    pc = plt.pcolormesh(x.ravel(), y.ravel(), mag_u, cmap='seismic', shading='gouraud', rasterized=True)
    plt.colorbar(pc)
    plt.quiver(x_g.T[::res, ::res], y_g.T[::res, ::res], ugx.T[::res, ::res], ugy.T[::res, ::res])
    plt.fill(rx, ry, edgecolor='k', linewidth=1, fill=False)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Flow: t = {:.2f}".format(solver.sim_time))
    plt.tight_layout()
    plt.savefig(run_name + '/flow_final.png', dpi=200)