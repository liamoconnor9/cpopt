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
from min_dist import min_dist

def construct_phi(a, delta, dist, coords, bases):

    xbasis, ybasis = bases[1], bases[0]
    ey, ex = coords.unit_vector_fields(dist)
    y, x = ybasis.local_grid(), xbasis.local_grid()
    # y, x = ybasis.global_grid(), xbasis.global_grid()
    y_g = y * np.ones_like(x)
    x_g = x * np.ones_like(y)
    Nx = max(x.shape)
    Ny = max(y.shape)
    ks = [0]
    n = len(a)
    for i in range(2, n + 1):
        ks.append((-1)**i * int(i / 2))    

    thetas = np.linspace(0.0, 2*np.pi, 200)

    r = np.zeros(thetas.shape, dtype=np.complex128)
    for i, theta in enumerate(thetas):
        for k, a_k in zip(ks, a):
            r[i] += a_k * np.exp(1j*k*theta)

    rx = r.real
    ry = r.imag
    rs = list(zip(rx, ry))

    logger.info('solving for the signed distance function. This might take a sec')
    from matplotlib import path
    curve = path.Path(rs) 
    # flags = p.contains_points(x_g, y_g)
    enclosed = np.zeros_like(x_g)
    DF = np.zeros_like(x_g)
    for ix in range(Nx):
        for iy in range(Ny):
            DF[iy, ix] = min_dist(x_g[iy, ix], y_g[iy, ix], a)
            if (curve.contains_points([(x_g[iy, ix], y_g[iy, ix])])):
                enclosed[iy, ix] = 1
    
    SDF = enclosed * DF

    phi_g = (np.tanh(2*SDF / delta) + 1.0) / 2.0

    return DF, rs



filename = path + '/nsvp_options.cfg'
config = ConfigParser()
config.read(str(filename))

# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64
delta = 4.0

scale = 0.5
rotation = 0.0

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

phi = dist.Field(name='phi', bases=bases)

# Mask function (airfoil geometry)
#################################################################
domain = domain.Domain(dist, bases)
slices = dist.grid_layout.slices(domain, scales=1)
if True:

    a0 = 0.2
    # a = [a0, 1.0, 0.4]
    a = [a0, 1.0, 0.4, 1.4+0.64j, 0.24-1.44j, 0.124+3.8j, 0.1]

    rot_exp = np.exp(1j*(rotation / 180 * np.pi))
    a = [ak*scale*rot_exp for ak in a]

    phi_g, rs = construct_phi(a, delta, dist, coords, bases)
    [rx, ry] = zip(*rs)
    phi['g'] = phi_g
    dist.comm.Barrier()
    phi.change_scales(1)
    phi_g_global = phi.allgather_data('g')

    if (dist.comm.rank == 0):
        plt.pcolormesh(x_g, y_g, phi_g_global, cmap='seismic')
        plt.scatter(rx, ry, color='white')
        plt.savefig('sdf_evp.png')
    # plt.show()
