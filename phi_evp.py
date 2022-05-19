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


def construct_phi(a, delta, dist, coords, bases):

    xbasis, ybasis = bases[1], bases[0]
    ey, ex = coords.unit_vector_fields(dist)
    y, x = ybasis.local_grid(), xbasis.local_grid()
    # y, x = ybasis.global_grid(), xbasis.global_grid()
    y_g = y * np.ones_like(x)
    x_g = x * np.ones_like(y)
    Nx = max(x.shape)
    Ny = max(y.shape)

    a0 = -2.0
    pt = 1j
    a = [a0 - pt, 1.0, 0.4]
    # a = [a0 - pt, 1.0, 0.4, 1.4+0.64j, 0.24-1.44j, 0.124+3.8j, 0.9]

    #ellipse
    n = len(a)
    ks = [0]
    for i in range(2, n + 1):
        ks.append((-1)**i * int(i / 2))    

    # theta = np.pi
    def dist(thetas, a):
        n = len(a)
        
        dist_mat = np.zeros((n, n, len(thetas)), dtype=np.complex128)
        for row in range(n):
            for col in range(n):
                dist_mat[row, col, :] = (a[col] * np.conj(a[row]) * np.exp(1j*(ks[col]-ks[row])*thetas))

        return dist_mat.sum(axis=tuple(range(dist_mat.ndim - 1))).real

    kmax = round((n - 1) / 2)
    sin_coeff = np.zeros(4*kmax + 1)
    cos_coeff = np.zeros(4*kmax + 1)

    def dist_prime(thetas, a):
        n = len(a)
        dist_mat = np.zeros((n, n, len(thetas)), dtype=np.complex128)
        # for col in range(n):
        #     for row in range(n):
        for col in range(1, n):
            for row in range(col):
                # dist_mat[row, col, :] = -2*np.imag((ks[col]-ks[row])*(a[col] * np.conj(a[row]) * np.exp(1j*(ks[col]-ks[row])*thetas)))
                dist_mat[row, col, :] += -2*(ks[col]-ks[row]) * np.real(a[col] * np.conj(a[row])) * np.sin((ks[col]-ks[row])*thetas)
                dist_mat[row, col, :] += -2*(ks[col]-ks[row]) * np.imag(a[col] * np.conj(a[row])) * np.cos((ks[col]-ks[row])*thetas)

                sin_coeff[(ks[col]-ks[row])] += -2*(ks[col]-ks[row]) * np.real(a[col] * np.conj(a[row]))
                cos_coeff[(ks[col]-ks[row])] += -2*(ks[col]-ks[row]) * np.imag(a[col] * np.conj(a[row]))

        return dist_mat.sum(axis=tuple(range(dist_mat.ndim - 1))).real


    thetas = np.linspace(0.0, 2*np.pi, 200)
    dists = dist(thetas, a)
 
    r = np.zeros(thetas.shape, dtype=np.complex128)
    for i, theta in enumerate(thetas):
        for k, a_k in zip(ks, a):
            r[i] += a_k * np.exp(1j*k*theta)

    rx = r.real
    ry = r.imag
    rs = list(zip(rx, ry))

    thetas_adj = np.arctan2(ry, rx)
    plt.plot(thetas, dists)
    dist_primes = dist_prime(thetas, a)
    plt.plot(thetas, dist_primes)
    # plt.scatter(dists*np.cos(thetas_adj), dists*np.sin(thetas_adj))

    # plt.axvline(x=np.arctan(1/2)) 
    # plt.axhline(y=0) 

    sin_coeff_trunc = []
    cos_coeff_trunc = []
    for i in range(1, 2*kmax + 1):
        sin_coeff_trunc.append((sin_coeff[i] - sin_coeff[-i]))
        cos_coeff_trunc.append((cos_coeff[i] + cos_coeff[-i]))

    recon = np.zeros_like(thetas)
    for i in range(len(sin_coeff_trunc)):
        sin_c = sin_coeff_trunc[i]
        cos_c = cos_coeff_trunc[i]
        recon += sin_c * np.sin((i+1) * thetas)
        recon += cos_c * np.cos((i+1) * thetas)

    plt.plot(thetas, recon, linestyle='--')


    # plt.show()
    plt.savefig('dist_prime.png')
    plt.close()

    # https://math.stackexchange.com/questions/370996/roots-of-a-finite-fourier-series
    # computing the roots of the distance function's derivative to find extrema (this can be done explicitly)
    N = n - 1
    hvec = np.zeros(2*N + 1, dtype=np.complex128)

    for k in range(N):
        hvec[k] = cos_coeff_trunc[N - k - 1] + 1j * sin_coeff_trunc[N - k - 1]
    
    for k in range(N + 1, 2*N + 1):
        hvec[k] = cos_coeff_trunc[k - 1 - N] - 1j * sin_coeff_trunc[k - 1 - N]

    B = np.zeros((2*N, 2*N), dtype=np.complex128)
    for k in range(1, 2*N+1):
        for j in range(1, 2*N+1):
            if (j == 2*N):
                B[j - 1, k - 1] = -hvec[k - 1] / (cos_coeff_trunc[-1] - 1j * sin_coeff_trunc[-1])
            elif (j == k - 1):
                B[j - 1, k - 1] = 1.0

    w, v = np.linalg.eig(B)

    roots = -1j*np.log(w) 

    recon = np.zeros_like(roots)
    for i in range(len(sin_coeff_trunc)):
        sin_c = sin_coeff_trunc[i]
        cos_c = cos_coeff_trunc[i]
        recon += sin_c * np.sin((i+1) * roots)
        recon += cos_c * np.cos((i+1) * roots)
    print(recon)

    # print(roots)
    sys.exit()

    # dists_comp = dists * np.exp(1j*thetas_adj)

    rx2 = np.sqrt(dists) * np.cos(thetas_adj)
    ry2 = np.sqrt(dists) * np.sin(thetas_adj)
    # rx2 = [dist*np.cos(theta) for dist, theta in zip(dists, thetas)]
    # ry2 = [dist*np.sin(theta) for dist, theta in zip(dists, thetas)]

    plt.scatter(rx, ry)
    plt.scatter(rx2, ry2, marker='.')
    plt.show()
    sys.exit()

    logger.info('solving for the signed distance function. This might take a sec')
    from matplotlib import path
    curve = path.Path(rs) 
    # flags = p.contains_points(x_g, y_g)
    enclosed = np.zeros_like(x_g)
    for ix in range(Nx):
        for iy in range(Ny):
            if (curve.contains_points([(x_g[iy, ix], y_g[iy, ix])])):
                enclosed[iy, ix] = 1

    # phi_g = (np.tanh(2*SDF / delta) + 1.0) / 2.0

    return None, rs



filename = path + '/nsvp_options.cfg'
config = ConfigParser()
config.read(str(filename))

# Parameters
Lx, Ly = 10, 2*np.pi
Nx, Ny = 256, 128
dtype = np.float64

Reynolds = config.getfloat('parameters', 'Reynolds')
nu = 1 / Reynolds
U0 = config.getfloat('parameters', 'U0')
tau = config.getfloat('parameters', 'tau')
delta = config.getfloat('parameters', 'delta')

scale = 1.0
rotation = 0.0

max_timestep = config.getfloat('parameters', 'max_dt')
stop_sim_time = config.getfloat('parameters', 'T') + 0.1

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

# Mask function (airfoil geometry)
#################################################################
domain = domain.Domain(dist, bases)
slices = dist.grid_layout.slices(domain, scales=1)
phi = dist.Field(name='phi', bases=bases)
if True:

    a0 = -2.0
    a = [a0, 1.0, 0, 0]

    rot_exp = np.exp(1j*(rotation / 180 * np.pi))
    a = [ak*scale*rot_exp for ak in a]

    phi_g, rs = construct_phi(a, delta, dist, coords, bases)
    phi['g'] = phi_g
    dist.comm.Barrier()
    phi.change_scales(1)
    phi_g_global = phi.allgather_data('g')
