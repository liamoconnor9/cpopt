"""
Dedalus script solving the 2D Poisson equation with mixed boundary conditions.
This script demonstrates solving a 2D Cartesian linear boundary value problem
and produces a plot of the solution. It should take just a few seconds to run.

We use a Fourier(x) * Chebyshev(y) discretization to solve the LBVP:
    dx(dx(u)) + dy(dy(u)) = f
    u(y=0) = g
    dy(u)(y=Ly) = h

For a scalar Laplacian on a finite interval, we need two tau terms. Here we
choose to lift them to the natural output (second derivative) basis.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Ly = 2, 1
Nx, Ny = 128, 256
dtype = np.float64
Reynolds = 1e8
nu = 1 / Reynolds
U0 = 1

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2))
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
tau_p = dist.Field(name='tau_p', bases=(xbasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis))

U = dist.VectorField(coords, name='U', bases=bases)
U['g'][0] = U0
phi = dist.Field(name='phi', bases=bases)
sigma = 0.4
r = np.sqrt(y**2 + (x - Lx/2)**2)
phi_g = np.tanh(1000*(sigma - r)) + 0.5
# phi_g = np.exp(-(r / sigma)**2)
phi['g'] = phi_g

lift_basis = ybasis.clone_with(a=1/2, b=1/2) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
grad_u = d3.grad(u) + ey*lift(tau_u1,-1) # First-order reduction

# Problem
problem = d3.NLBVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
# problem.add_equation("grad(p) - nu*div(grad_u) + lift(tau_u2,-1) = - u @ grad(u)")
problem.add_equation("grad(p) - nu*div(grad_u) + lift(tau_u2,-1) = phi*(u - U)/tau")

problem.add_equation("integ(p) = 0") # Pressure gauge
# problem.add_equation("p(y='left') = 0")
if (False):
    problem.add_equation("u(y='left') = 0")
    problem.add_equation("u(y='right') = 0")
else:
    uy = grad_u @ ey
    problem.add_equation("(u @ ex)(y='left') = 0")
    problem.add_equation("(u @ ex)(y='right') = 0")
    
    problem.add_equation("(u @ ey)(y='left') = 0")
    problem.add_equation("(u @ ey)(y='right') = 0")


# Solver
solver = problem.build_solver()
# solver.solve()
for i in range(10):
    solver.newton_iteration()

# Gather global data
x = xbasis.global_grid()
y = ybasis.global_grid()
ugx = ((u) @ ex).evaluate().allgather_data('g')
ugy = ((u) @ ey).evaluate().allgather_data('g')
mag_u = np.sqrt(ugx**2 + ugy**2)

# Plot
if dist.comm.rank == 0:
    plt.figure(figsize=(6, 4))
    res = 8
    # plt.pcolormesh(x.ravel(), y.ravel(), phi_g.T, cmap='viridis', shading='gouraud', rasterized=True)
    plt.pcolormesh(x.ravel(), y.ravel(), mag_u.T, cmap='viridis', shading='gouraud', rasterized=True)
    plt.quiver(x_g.T[::res, ::res], y_g.T[::res, ::res], ugx.T[::res, ::res], ugy.T[::res, ::res])
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Stokes Equation")
    plt.tight_layout()
    # plt.savefig('stokes.pdf')
    plt.savefig('stokes.png', dpi=200)