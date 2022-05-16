import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize

def construct_phi(dist, coords, bases):

    xbasis, ybasis = bases[0], bases[1]
    ex, ey = coords.unit_vector_fields(dist)
    x, y = dist.local_grids(xbasis, ybasis)
    x_g = x * np.ones_like(y)
    y_g = y * np.ones_like(x)
    Nx = max(x.shape)
    Ny = max(y.shape)

    #ellipse
    a0 = 0.0
    rotation = 15
    rot_exp = np.exp(1j*(rotation / 180 * np.pi))

    #circle
    scale = 0.75
    a = [a0, 0.4, (-1.2), 0, 0]
    ks = [0, 1, -1, 2, -2]

    a = [ak*scale*rot_exp for ak in a]

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

    logger.info('solving for the signed distance function. This might take a sec')
    from matplotlib import path
    curve = path.Path(rs) 
    # flags = p.contains_points(x_g, y_g)
    enclosed = np.zeros_like(x_g)
    for ix in range(Nx):
        for iy in range(Ny):
            if (curve.contains_points([(x_g[ix, iy], y_g[ix, iy])])):
                enclosed[ix, iy] = 1

    DF = np.zeros_like(x_g)
    counter = 0
    num_points = Nx * Ny
    for ix in range(Nx):
        for iy in range(Ny):
            x = x_g[ix, iy]
            y = y_g[ix, iy]
            DF[ix, iy] = min_distance_to_curve(x, y, a, ks)
            if (counter % 1000 == 0):
                logger.info("{:.0%} done solving..".format(counter / num_points))
            counter += 1
            # print("(x, y) = ({}, {})".format(x, y))
    SDF = 2.0*(enclosed-0.5)*DF

    phi_g = np.tanh(100*SDF) + 0.5
    return phi_g