"""Coordinate metric derivatives and generic Levi-Civita curvature.

The Riemann sign is that of docs/convenciones.md. No field equations or
reference-solution identities enter these contractions.
"""

import numpy as np
import sympy as sp


def metric_from_ansatz(r, theta, b, f, h, w):
    """Return the equal-spin metric in (t,r,theta,phi1,phi2), gauge g=r²."""
    s, c = sp.sin(theta)**2, sp.cos(theta)**2
    metric = sp.zeros(5)
    metric[0, 0] = -b+h*w**2
    metric[1, 1] = 1/sp.sympify(f)
    metric[2, 2] = r**2
    metric[0, 3] = metric[3, 0] = -h*w*s
    metric[0, 4] = metric[4, 0] = -h*w*c
    metric[3, 3] = h*s+(r**2-h)*s*c
    metric[4, 4] = h*c+(r**2-h)*s*c
    metric[3, 4] = metric[4, 3] = -(r**2-h)*s*c
    return metric


def compile_metric(metric, coordinates, parameters=()):
    """Compile scalar evaluation of g, dg[k,i,j], ddg[k,l,i,j].

    Arguments are coordinate values followed by parameter values. Symbolic
    differentiation precedes NumPy evaluation; only nonzero entries are
    compiled, with common subexpressions shared across derivative orders.
    """
    coordinates, parameters = tuple(coordinates), tuple(parameters)
    metric = sp.Matrix(metric)
    n = len(coordinates)
    if metric.shape != (n, n):
        raise ValueError('metric shape must match the coordinate count')
    shapes = ((n, n), (n, n, n), (n, n, n, n))
    positions, expressions = [], []

    def add(order, index, expression):
        if expression != 0:
            positions.append((order, index))
            expressions.append(expression)

    for i in range(n):
        for j in range(n):
            entry = metric[i, j]
            add(0, (i, j), entry)
            for k, coordinate in enumerate(coordinates):
                derivative = sp.diff(entry, coordinate)
                add(1, (k, i, j), derivative)
                if derivative != 0:
                    for l, other in enumerate(coordinates):
                        add(2, (k, l, i, j), sp.diff(derivative, other))
    numerical = sp.lambdify((*coordinates, *parameters), expressions,
                            modules='numpy', cse=True, docstring_limit=0)

    def evaluate(*values):
        arrays = tuple(np.zeros(shape, dtype=float) for shape in shapes)
        for (order, index), value in zip(positions, numerical(*values)):
            arrays[order][index] = value
        return arrays

    return evaluate


def curvature(g, dg, ddg):
    """Return covariant Ricci/Einstein, scalar and mixed Riemann.

    All coordinate derivatives and tensor components are contracted in full.
    Float64 evaluation requires a regular, reasonably conditioned chart;
    horizon/axis limits must be treated outside this evaluator. `riemann`
    is R^rho_{sigma mu nu}, indexed riemann[rho,sigma,mu,nu]; contracting
    rho with mu reproduces `ricci` (checked in tests), which is how the
    Hito 1/2 acceptance already exercises this same connection algebra.
    """
    g, dg, ddg = (np.asarray(value, dtype=float) for value in (g, dg, ddg))
    n = g.shape[0]
    if g.shape != (n, n) or dg.shape != (n, n, n) or ddg.shape != (n, n, n, n):
        raise ValueError('expected shapes (n,n), (n,n,n), (n,n,n,n)')
    inverse = np.linalg.inv(g)
    dinverse = -np.einsum('ia,kab,bj->kij', inverse, dg, inverse)
    # C[a,b,c] = d_b g_ac + d_c g_ab - d_a g_bc.
    connection_lower = dg.transpose(1, 0, 2)+dg.transpose(1, 2, 0)-dg
    dconnection_lower = ddg.transpose(0, 2, 1, 3)+ddg.transpose(0, 2, 3, 1)-ddg
    gamma = 0.5*np.einsum('ia,abc->ibc', inverse, connection_lower)
    dgamma = 0.5*(np.einsum('kia,abc->kibc', dinverse, connection_lower)
                  + np.einsum('ia,kabc->kibc', inverse, dconnection_lower))
    # R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - d_nu Gamma^rho_{mu sigma}
    #                       + Gamma^rho_{mu l}Gamma^l_{nu sigma} - Gamma^rho_{nu l}Gamma^l_{mu sigma}.
    riemann = (np.einsum('mrns->rsmn', dgamma)-np.einsum('nrms->rsmn', dgamma)
               + np.einsum('rml,lns->rsmn', gamma, gamma)
               - np.einsum('rnl,lms->rsmn', gamma, gamma))
    ricci = np.einsum('rsrn->sn', riemann)
    scalar = float(np.einsum('ij,ij->', inverse, ricci))
    return {'ricci': ricci, 'scalar': scalar, 'einstein': ricci-0.5*scalar*g,
            'riemann': riemann, 'inverse': inverse}


def gauss_bonnet(g, inverse, riemann, ricci, scalar):
    """Return the Lanczos--Lovelock tensor H and the L_GB density.

    `H_{mu nu}` and `L_GB` follow docs/convenciones.md; `G_{mu nu}+alpha_GB
    H_{mu nu}=0` are the EGB field equations. Inputs are the outputs of
    `curvature`; nothing here fixes an ansatz or a reference solution.
    """
    g, inverse, riemann, ricci = (np.asarray(v, dtype=float)
                                   for v in (g, inverse, riemann, ricci))
    riemann_lower = np.einsum('ra,asmn->rsmn', g, riemann)
    riemann_upper = np.einsum('ra,sb,mc,nd,abcd->rsmn',
                               inverse, inverse, inverse, inverse, riemann_lower)
    riemann_squared = float(np.einsum('abcd,abcd->', riemann_lower, riemann_upper))
    ricci_mixed = np.einsum('ab,bc->ac', inverse, ricci)
    ricci_squared = float(np.einsum('ab,ba->', ricci_mixed, ricci_mixed))
    lagrangian = scalar**2-4*ricci_squared+riemann_squared

    ricci_upper = np.einsum('ra,sb,ab->rs', inverse, inverse, ricci)
    riemann_nu_up = np.einsum('na,arst->nrst', g, riemann_upper)
    a_term = np.einsum('mrst,nrst->mn', riemann_lower, riemann_nu_up)
    b_term = np.einsum('mrns,rs->mn', riemann_lower, ricci_upper)
    c_term = np.einsum('ma,ab,bn->mn', ricci, inverse, ricci)
    h_tensor = 2*(a_term-2*b_term-2*c_term+scalar*ricci)-0.5*g*lagrangian
    return {'H': h_tensor, 'lagrangian': lagrangian}
