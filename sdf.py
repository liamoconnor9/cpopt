import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize

# Parameters
Lx, Ly = 4*np.pi, 2*np.pi
Nx, Ny = 32, 16
# Nx, Ny = 256, 128
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

# a0 = Nx / 2 + 1j * Ny / 2
a0 = 0.0

#circle
a = [a0, 0, 1, 0, 0]

#ellipse
a = [a0, 0.4, 1.2-0.8j, 0, 0]
ks = [0, 1, -1, 2, -2]

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

def distance_to_curve(theta, x, y, a, ks):
    r = 0 + 0j
    for k, a_k in zip(ks, a):
        r += a_k * np.exp(1j*k*theta)

    difference = (x + 1j*y) - r
    return np.sqrt(difference * np.conj(difference)).real.flat[0]

def min_distance_to_curve(x, y, a, ks):
    minmin = 10000
    guess = np.arctan2(y, x)
    for guess in np.linspace(0, 2*np.pi, 3):
        minmin = min(minmin, minimize(distance_to_curve, guess, args=(x, y, a, ks)).fun)

    return minmin

from matplotlib import path
p = path.Path(rs) 
# flags = p.contains_points(x_g, y_g)
enclosed = np.zeros_like(x_g)
for ix in range(Nx):
    for iy in range(Ny):
        if (p.contains_points([(x_g[ix, iy], y_g[ix, iy])])):
            enclosed[ix, iy] = 1

DF = np.zeros_like(x_g)
for ix in range(Nx):
    for iy in range(Ny):
        x = x_g[ix, iy]
        y = y_g[ix, iy]
        DF[ix, iy] = min_distance_to_curve(x, y, a, ks)
        # print("(x, y) = ({}, {})".format(x, y))
SDF = 2.0*(enclosed-0.5)*DF
# SDF = 2.0*(enclosed-0.5)

plt.figure(figsize=(6, 4))
res = 8
# plt.pcolormesh(x.ravel(), y.ravel(), phi_g.T, cmap='viridis', shading='gouraud', rasterized=True)
plt.pcolormesh(x_g.T, y_g.T, SDF.T, cmap='seismic', shading='gouraud', rasterized=True)
plt.fill(rx, ry, edgecolor='k', linewidth=1, linestyle='--', fill=False)
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.title("SDF")
plt.tight_layout()
# plt.savefig('stokes.pdf')
plt.savefig('SDF.png', dpi=200)