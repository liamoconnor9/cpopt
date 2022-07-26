import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from scipy.optimize import minimize
from dedalus.core import domain


def construct_phi_diff(run_name, T, Ndt, dist, coords, bases):


    xbasis, ybasis = bases[1], bases[0]
    ey, ex = coords.unit_vector_fields(dist)
    y, x = ybasis.local_grid(), xbasis.local_grid()
    # y, x = ybasis.global_grid(), xbasis.global_grid()
    y_g = y * np.ones_like(x)
    x_g = x * np.ones_like(y)
    Nx = max(x.shape)
    Ny = max(y.shape)
    dy = lambda A: d3.Differentiate(A, coords.coords[0])
    dx = lambda A: d3.Differentiate(A, coords.coords[1])

    xscale = 1.0
    yscale = 0.5

    rs = [(0.0, 1.0), (1.0, 1.0), (1.0, -1.0), (0.0, -1.0)]

    if ('triangular' in run_name):
        rs.append((-1.0, 0.0))

    elif ('parabolic' in run_name):

        y = np.linspace(-1, 1, 1000)
        a = 0.794
        x = a*(1-y**2)
        for i in range(1000):
            rs.append((x[i],y[i]))

    elif ('ellipt' in run_name):
        logger.error('Not implemented')
        raise

    else:
        logger.error('Not implemented')
        raise



    for ind, (rx, ry) in enumerate(rs):
        rs[ind] = (rx * xscale, ry * yscale)

    # rs = list(zip(rx, ry))

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
    solver.stop_sim_time = T

    # domain = domain.Domain(dist, bases)
    # slices = dist.grid_layout.slices(domain, scales=1)
    p['g'] = enclosed.copy()

    dt = T / Ndt
    for i in range(Ndt):
        solver.step(dt)
        if (solver.iteration % round(Ndt / 10) == 0):
            logger.info('Iteration=%i, Time=%e' %(solver.iteration, solver.sim_time))
    
    p.change_scales(1)
    pdiff = p.allgather_data('g')
    pdiff -= np.min(pdiff)
    pdiff /= np.max(pdiff)

    # pc = plt.pcolormesh(x_g.T, y_g.T, pdiff.T, cmap='seismic', shading='gouraud', rasterized=True)
    # plt.colorbar(pc)
    # # plt.fill(rx, ry, edgecolor='k', linewidth=1, linestyle='--', fill=False)
    # plt.gca().set_aspect('equal')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title("Phi mask (diffused indicator)")
    # plt.tight_layout()
    # # plt.savefig('stokes.pdf')
    # plt.savefig('phi_diff.png', dpi=200)


    return pdiff, rs