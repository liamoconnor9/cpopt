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

# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64
Reynolds = 1e2
nu = 1 / Reynolds
U0 = 1
tau = 2e0
max_timestep = 0.00025
stop_sim_time = 4e2
restart = False

# Bases
coords = d3.CartesianCoordinates('y', 'x')
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


# phi_g = np.exp(-(r / sigma)**2)
# domain = domain.Domain(dist, bases)
# slices = dist.grid_layout.slices(domain, scales=1)

from construct_phi_diff import construct_phi_diff

a0 = 0.0
a = [a0, 0.4, (-1.2), 0, 0]

rotation = 0
rot_exp = np.exp(1j*(rotation / 180 * np.pi))
scale = 0.5
a = [ak*scale*rot_exp for ak in a]

phi_g, rs = construct_phi_diff(a, 0.03, 100, dist, coords, bases)
dist.comm.Barrier()

phi = dist.Field(name='phi', bases=bases)
rx, ry = zip(*rs)
# phi['g'] = phi_g

lift_basis = xbasis.derivative_basis(1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + ex*lift(tau_u1,-1) # First-order reduction

problem = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())

problem.add_equation("trace(grad_u) + tau_p = 0")
# problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2, -1) =  ")
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2, -1) = -u@grad(u) - phi*(u)/tau")
# problem.add_equation("dt(u) + grad(p) - nu*div(grad_u) + lift(tau_u2, -1) = - phi*(u)/tau")

if False:
    problem.add_equation("u(x='left') = u(x='right')")
    problem.add_equation("dx(u)(x='left') = dx(u)(x='right')")
else:
    # problem.add_equation("u(x='left') = 0")
    # problem.add_equation("u(x='right') = 0")

    problem.add_equation("(u @ ex)(x='left') = U0")
    problem.add_equation("(u @ ey)(x='left') = 0")

    problem.add_equation("(u @ ex)(x='right') = U0")
    problem.add_equation("(u @ ey)(x='right') = 0")

problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

load_dir = "checkpoints_sdf/"
checkpoint_names = [name for name in os.listdir(load_dir) if '.h5' in name]
last_checkpoint = natsorted(checkpoint_names)[-1]
solver.load_state(load_dir + last_checkpoint, -1)
fhmode = 'append'

# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()
# u.change_scales(1)
ux = ((u - U) @ ex).evaluate()
ux.change_scales(1)
ugx = ux.allgather_data('g')

uy = ((u - U) @ ey).evaluate()
uy.change_scales(1)
ugy = uy.allgather_data('g')
mag_u = np.sqrt(ugx**2 + ugy**2)

# phi_g_global = phi.allgather_data('g')

# Plot
if dist.comm.rank == 0:
    plt.figure(figsize=(6, 4))
    res = 8
    # plt.pcolormesh(x.ravel(), y.ravel(), phi_g.T, cmap='viridis', shading='gouraud', rasterized=True)
    plt.pcolormesh(x.ravel(), y.ravel(), mag_u, cmap='seismic', shading='gouraud', rasterized=True)
    # plt.fill(rx, ry, color='black')
    # plt.pcolormesh(x.ravel(), y.ravel(), phi_g, cmap='seismic', shading='gouraud', rasterized=True)
    plt.quiver(x_g.T[::res, ::res], y_g.T[::res, ::res], ugx.T[::res, ::res], ugy.T[::res, ::res])
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Flow: t = {:.2f}".format(solver.sim_time))
    plt.tight_layout()
    # plt.savefig('stokes.pdf')
    plt.savefig('ns_flow_recent2.png', dpi=200)
    # plt.show()