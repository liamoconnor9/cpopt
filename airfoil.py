import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize

# Parameters
Lx, Ly = 4*np.pi, 2*np.pi
Nx, Ny = 64, 32
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
a = [a0, 0.2, 0.6-0.4j, 0, 0]
a = [a0, 0, 1, 0, 0]
ks = [0, 1, -1, 2, -2]
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

plt.fill(rx, ry, edgecolor='k', linewidth=4, linestyle='--', fill=True)
plt.title('Airfoil Profile')
plt.savefig("airfoil.png")
# plt.show()
