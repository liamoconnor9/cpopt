import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize



def construct_phi_diff(a, T, Ndt, dist, coords, bases):

    xbasis, ybasis = bases[1], bases[0]
    ey, ex = coords.unit_vector_fields(dist)
    y, x = ybasis.global_grid(), xbasis.global_grid()
    y_g = y * np.ones_like(x)
    x_g = x * np.ones_like(y)
    Nx = max(x.shape)
    Ny = max(y.shape)
    dy = lambda A: d3.Differentiate(A, coords.coords[0])
    dx = lambda A: d3.Differentiate(A, coords.coords[1])

    n = len(a)
    ks = [0]
    for i in range(2, n + 1):
        ks.append((-1)**i * int(i / 2))

    thetas = np.linspace(0, 2*np.pi, 1000)
    r = np.zeros(thetas.shape, dtype=np.complex128)
    for i, theta in enumerate(thetas):
        for k, a_k in zip(ks, a):
            r[i] += a_k * np.exp(1j*k*theta)

    rx = r.real
    ry = r.imag
    rs = list(zip(rx, ry))

    # plt.scatter(rx, ry)
    # plt.show()

    logger.info('solving for the signed distance function. This might take a sec')
    from matplotlib import path
    curve = path.Path(rs) 
    # flags = p.contains_points(x_g, y_g)
    enclosed = np.zeros_like(x_g)
    for ix in range(Nx):
        for iy in range(Ny):
            if (curve.contains_points([(x_g[iy, ix], y_g[iy, ix])])):
                enclosed[iy, ix] = 1

    p = dist.Field(name='p', bases=bases)
    tau_p1 = dist.Field(name='tau_p1', bases=(ybasis))
    tau_p2 = dist.Field(name='tau_p2', bases=(ybasis))

    lift_basis = xbasis.derivative_basis(1) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    grad_p = d3.grad(p) + ex*lift(tau_p1,-1) # First-order reduction

    problem = d3.IVP([p, tau_p1, tau_p2], namespace=locals())

    problem.add_equation("dt(p) - div(grad_p) + lift(tau_p2, -1) = 0")
    problem.add_equation("p(x = 'left') = 0") 
    problem.add_equation("p(x = 'right') = 0") 
    # problem.add_equation("integ(p) = 0") 
    
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    p['g'] = enclosed.copy()

    dt = T / Ndt
    for i in range(Ndt):
        solver.step(dt)
        if (solver.iteration % round(Ndt / 10) == 0):
            logger.info('Iteration=%i, Time=%e' %(solver.iteration, solver.sim_time))
    
    p.change_scales(1)
    pdiff = p['g'].copy()
    pdiff -= np.min(pdiff)
    pdiff /= np.max(pdiff)
    return pdiff


# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64
Reynolds = 1e2
nu = 1 / Reynolds
U0 = 1
tau = 1e2
max_timestep = 0.001
stop_sim_time = 1e1

# Bases
coords = d3.CartesianCoordinates('y', 'x')
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=3/2)
xbasis = d3.ChebyshevT(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=3/2)
bases = (ybasis, xbasis)
ey, ex = coords.unit_vector_fields(dist)
y, x = dist.local_grids(ybasis, xbasis)
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

a0 = 0.0
a = [a0, 0.4, (-1.2), 0, 0]

rotation = 80
rot_exp = np.exp(1j*(rotation / 180 * np.pi))
scale = 0.75
a = [ak*scale*rot_exp for ak in a]

pg = construct_phi_diff(a, 0.01, 100, dist, coords, bases)

pc = plt.pcolormesh(x_g.T, y_g.T, pg.T, cmap='seismic', shading='gouraud', rasterized=True)
plt.colorbar(pc)
# plt.fill(rx, ry, edgecolor='k', linewidth=1, linestyle='--', fill=False)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title("SDF")
plt.tight_layout()
# plt.savefig('stokes.pdf')
plt.savefig('SDF.png', dpi=200)
plt.show()
