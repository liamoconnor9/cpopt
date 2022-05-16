import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize

# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64
Reynolds = 1e4
nu = 1 / Reynolds
U0 = 1
tau = 1e-2
max_timestep = 0.0001
stop_sim_time = 1e-2

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

# a0 = Nx / 2 + 1j * Ny / 2
a0 = 0.0
rotation = 90
rot_exp = np.exp(1j*(rotation / 180 * np.pi))

#circle
scale = 0.5
a = [a0, 0.4, (-1.2), 0, 0]
# a = [a0, 0.4, (-1.2+0.8j), 0, 0]
ks = [0, 1, -1, 2, -2]

a = [ak*scale*rot_exp for ak in a]

#ellipse
# a = [0.244+0.175j, 0.434 - 0.769j, 0.234 + 0.168j]


thetas = np.linspace(0, 2*np.pi, 100)
r = np.zeros(thetas.shape, dtype=np.complex128)
for i, theta in enumerate(thetas):
    for k, a_k in zip(ks, a):
        # print(k)
        r[i] += a_k * np.exp(1j*k*theta)

rx = r.real
ry = r.imag

# plt.scatter(rx, ry)
plt.fill(rx, ry, edgecolor='k', linewidth=4, linestyle='--', fill=True)
plt.xlim(-Lx/2, Lx/2)
plt.ylim(-Ly/2, Ly/2)
plt.title('Airfoil Profile')
plt.savefig("airfoil.png")
plt.show()