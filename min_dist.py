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

def min_dist(x, y, a):

    anew = a.copy()
    pt = x + 1j*y
    anew[0] -= pt
    n = len(anew)
    
    ks = np.zeros(n) 
    ks[1::2] = range(1, n//2 + 1)
    ks[2::2] = range(-1, -n//2, -1)
    ks = ks.astype(int)

    N = n - 1
    sin_coeff_trunc = np.zeros(N)
    cos_coeff_trunc = np.zeros(N)

    for col in range(1, n):
        for row in range(col):
            sin_coeff_trunc[np.abs(ks[col]-ks[row]) - 1] += np.sign((ks[col]-ks[row]))*-2*(ks[col]-ks[row]) * np.real(anew[col] * np.conj(anew[row]))
            cos_coeff_trunc[np.abs(ks[col]-ks[row]) - 1] += -2*(ks[col]-ks[row]) * np.imag(anew[col] * np.conj(anew[row]))


    # https://math.stackexchange.com/questions/370996/roots-of-a-finite-fourier-series
    # computing the roots of the distance function's derivative to find extrema (this can be done explicitly)
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
    rrs = []
    for root in roots:
        if root.imag < 1e-8:
            rr = root.real
            if rr < 0:
                rr += 2*np.pi
            rrs.append(rr)
            # plt.axvline(x=rr, linestyle=':', color='black')

    rrs = np.array(rrs)
    root_dists = np.zeros_like(rrs, dtype=np.complex128)
    for k, a_k in zip(ks, anew):
        root_dists += a_k * np.exp(1j*k*rrs)

    rootdmagsq = (root_dists * root_dists.conj()).real
    max_ind = np.argmin(rootdmagsq)
    # plt.scatter([rrs[max_ind]], rootdmagsq[max_ind], marker='x', color='black')
    return np.sqrt(rootdmagsq[max_ind])
