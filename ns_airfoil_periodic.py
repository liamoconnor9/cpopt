import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize

# Parameters
Lx, Ly = 4*np.pi, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64
Reynolds = 1e4
nu = 1 / Reynolds
U0 = 1
tau = 0.01
max_timestep = 0.001
stop_sim_time = 0.01

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2))
bases = (xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
x, y = dist.local_grids(xbasis, ybasis)
x_g = x * np.ones_like(y)
y_g = y * np.ones_like(x)
dx = lambda A: d3.Differentiate(A, coords.coords[0])
dy = lambda A: d3.Differentiate(A, coords.coords[1])

# Fields
u = dist.VectorField(coords, name='u', bases=bases)
p = dist.Field(name='p', bases=bases)
tau_p = dist.Field(name='tau_p')
# tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis))
# tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis))

U = dist.VectorField(coords, name='U', bases=bases)
U['g'][0] = U0



# Mask function (airfoil geometry)
#################################################################
if False:
    from construct_phi import construct_phi
    phi_g = construct_phi(dist, coords, bases)
    with open('phi_g.npy', 'wb') as f:
        np.save(f, phi_g)
else:
    with open('phi_g.npy', 'rb') as f:
        phi_g = np.load(f)

# phi_g = np.exp(-(r / sigma)**2)
phi = dist.Field(name='phi', bases=bases)
phi['g'] = phi_g
logger.info('done solving SDF. Mask function phi constructed.')

#################################################################


problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = phi*(u - U)/tau")
problem.add_equation("integ(p) = 0") # Pressure gauge

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.1, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt((u@ex)**2 + (u@ey)**2), name='u_mag')

while solver.proceed:
    timestep = CFL.compute_timestep()
    solver.step(timestep)
    if (solver.iteration-1) % 10 == 0:
        max_u_mag = flow.max('u_mag')
        logger.info('Iteration=%i, Time=%e, dt=%e, max(|u|)=%f' %(solver.iteration, solver.sim_time, timestep, max_u_mag))


# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()
ugx = ((u + U) @ ex).evaluate().allgather_data('g')
ugy = ((u + U) @ ey).evaluate().allgather_data('g')
mag_u = np.sqrt(ugx**2 + ugy**2)

# Plot
if dist.comm.rank == 0:
    plt.figure(figsize=(6, 4))
    res = 8
    # plt.pcolormesh(x.ravel(), y.ravel(), phi_g.T, cmap='viridis', shading='gouraud', rasterized=True)
    plt.pcolormesh(x.ravel(), y.ravel(), mag_u.T, cmap='seismic', shading='gouraud', rasterized=True)
    plt.quiver(x_g.T[::res, ::res], y_g.T[::res, ::res], ugx.T[::res, ::res], ugy.T[::res, ::res])
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title("Flow Speed")
    plt.tight_layout()
    # plt.savefig('stokes.pdf')
    plt.savefig('stokes.png', dpi=200)
    plt.show()