"""Neural seed for the rotating EGB solver (Hito 5).

A small tanh network proposes a starting guess; the existing spectral solver is
what actually solves. Nothing here touches the solver, the adaptive method, the
Hito 4A derivation or the generated equations -- the only contact point is that
`Seed` exposes the `evaluate(x, derivative)` interface `solve` expects for its
`previous` argument.

Three things shape the implementation.

**The dependency freeze.** No learning framework exists in this environment and
none is added, so the network, its derivatives and its gradient are written out
in numpy.

**The boundary conditions are built in, not learned.** In the compact variables
(`z = 1-x`, with `x=1` spatial infinity and `x=0` the horizon -- the opposite of
what `egb_rotating_bvp`'s docstring suggests) Myers-Perry is elementary and is
the exact `alpha_gb=0` solution. Anchoring on it with prefactors that vanish
exactly where each condition lives makes the four infinity rows and
`W(horizon)=omega_h` hold for *any* weights. With a zero output layer the
untrained network is Myers-Perry to machine precision, which makes the untrained
control a demanding one rather than a straw man.

**The gradient is analytic, and the generated equations are never
differentiated by hand.** `rhs` acts pointwise, so its Jacobian in the seven
field arguments is block diagonal and a complex step recovers it exactly in
seven vectorised evaluations. Backpropagation through the ansatz and the network
is then plain algebra. The complex step over all parameters survives only as the
verification oracle it should have been: `check_gradient` compares against it.
"""
import numpy as np
import sympy as sp

__all__ = ['LAYERS', 'n_parameters', 'MyersPerryAnchor', 'Problem', 'Seed',
           'TrivialSeed', 'initial_parameters', 'check_gradient', 'multipliers',
           'anchored_fields', 'network_input', 'to_x_derivatives']

LAYERS = (1, 12, 12, 4)


def n_parameters(layers=LAYERS):
    return sum(a*b + b for a, b in zip(layers[:-1], layers[1:]))


def _unpack(theta, layers=LAYERS):
    out, offset = [], 0
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        weight = theta[offset:offset+n_in*n_out].reshape(n_out, n_in)
        offset += n_in*n_out
        bias = theta[offset:offset+n_out]
        offset += n_out
        out.append((weight, bias))
    return out


def initial_parameters(rng, layers=LAYERS):
    """Random hidden weights, zero output layer: the start is Myers-Perry exactly."""
    theta = np.zeros(n_parameters(layers))
    hidden = n_parameters(layers) - (layers[-2]*layers[-1] + layers[-1])
    theta[:hidden] = rng.normal(0., 1., hidden)
    return theta


class MyersPerryAnchor:
    """Compact-variable Myers-Perry and its first two x-derivatives.

    b=(1-z^2)B, f=(1-z^2)F, h=(1+z^4 H)/z^2, w=z^4 W with z=1-x. Built once with
    sympy rather than differentiated by hand; `tests/test_neural_seed.py` checks
    it against the solver at alpha_gb=0.
    """

    def __init__(self, omega_h):
        if not np.isfinite(omega_h) or abs(omega_h) >= 1/np.sqrt(2):
            raise ValueError('omega_h must be finite with |omega_h| < 1/sqrt(2)')
        self.omega_h = float(omega_h)
        x, om = sp.symbols('x omega', real=True)
        z = 1-x
        q2 = om**2
        denominator = 1-q2
        F = 1 - q2*z**2/denominator
        squash = 1 + q2*z**4/denominator
        expressions = [F/squash, F, q2/denominator + 0*x, om/(denominator + q2*z**4)]
        self._levels = [sp.lambdify((x, om), [sp.diff(e, x, order) for e in expressions],
                                    'numpy') for order in range(3)]

    def __call__(self, x):
        ones = np.ones_like(np.asarray(x))
        return [np.array([term*ones for term in level(x, self.omega_h)])
                for level in self._levels]


def multipliers(x):
    """Prefactor, dp/dx and d2p/dx2 for each field, in terms of u = z^2.

    The network is a function of `u = (1-x)^2`, not of x. That single choice
    makes `dH/dx` and `dW/dx` vanish at infinity for free, because `du/dx = -2z`
    is zero there whatever the network does. An earlier version used prefactors
    in z instead and, in enforcing those two derivative conditions, also pinned
    `H` and `W` to their Myers-Perry *values* at infinity -- which the solver
    never imposes and which move strongly with the coupling (H(inf) runs
    0.1222 -> 0.0206 and W(inf) 0.3703 -> 0.6307 between alpha_gb 0 and 0.5).
    That over-constraint, not capacity, was what capped the fit.

    So: B and F carry a factor u, which keeps them at 1 at infinity; H carries no
    factor at all, leaving its value there free; and W carries (1-u), which frees
    its value at infinity while pinning it to omega_h at the horizon.
    """
    z = 1-np.asarray(x, dtype=float)
    u_x, u_xx = -2*z, np.full_like(z, 2.)
    u = z*z
    zeros = np.zeros_like(z)
    return [(u, u_x, u_xx),
            (u, u_x, u_xx),
            (np.ones_like(z), zeros, zeros),
            (1-u, -u_x, -u_xx)]


def network_input(x):
    """u = z^2 and its first two x-derivatives, with z = 1-x."""
    z = 1-np.asarray(x)
    return z*z, -2*z, 2.


def to_x_derivatives(value, first, second, u_x, u_xx):
    """Convert network derivatives in u into derivatives in x."""
    return value, first*u_x, second*u_x*u_x + first*u_xx


def anchored_fields(anchor_values, multiplier_set, value, first, second):
    """Combine Myers-Perry with the network correction, keeping derivatives."""
    m0, m1, m2 = anchor_values
    out = [np.empty_like(value) for _ in range(3)]
    for i, (p, dp, ddp) in enumerate(multiplier_set):
        n, dn, ddn = value[i], first[i], second[i]
        if i < 2:
            u = 1 + p*n
            du = dp*n + p*dn
            ddu = ddp*n + 2*dp*dn + p*ddn
            out[0][i] = m0[i]*u
            out[1][i] = m1[i]*u + m0[i]*du
            out[2][i] = m2[i]*u + 2*m1[i]*du + m0[i]*ddu
        else:
            out[0][i] = m0[i] + p*n
            out[1][i] = m1[i] + dp*n + p*dn
            out[2][i] = m2[i] + ddp*n + 2*dp*dn + p*ddn
    return out


def _network(theta, x, layers=LAYERS):
    """Forward pass keeping the first two x-derivatives, plus the activations
    the backward pass needs. Holomorphic in theta, so the same routine serves
    the complex-step oracle."""
    x = np.asarray(x)
    dtype = np.result_type(np.asarray(theta).dtype, x.dtype)
    a = x.reshape(1, -1).astype(dtype)
    da = np.ones_like(a)
    dda = np.zeros_like(a)
    tape = []
    weights = _unpack(theta, layers)
    for index, (weight, bias) in enumerate(weights):
        tape.append((a, da, dda))
        value = weight@a + bias[:, None]
        first = weight@da
        second = weight@dda
        if index == len(weights)-1:
            return value, first, second, tape
        a = np.tanh(value)
        da = (1-a*a)*first
        dda = (1-a*a)*second - 2*a*(1-a*a)*first*first


def _network_backward(theta, adjoints, tape, layers=LAYERS):
    """Reverse mode through the forward pass above.

    The hidden layers carry three coupled outputs (a, da, dda), so each needs
    the full set of partials; writing them out is the whole point of this
    function, since no framework is available to do it.
    """
    grad_n, grad_dn, grad_ddn = adjoints
    weights = _unpack(theta, layers)
    gradient = np.zeros_like(theta)
    offsets, offset = [], 0
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        offsets.append((offset, offset+n_in*n_out, offset+n_in*n_out+n_out))
        offset += n_in*n_out + n_out

    ga, gda, gdda = grad_n, grad_dn, grad_ddn
    for index in reversed(range(len(weights))):
        weight, _ = weights[index]
        a_prev, da_prev, dda_prev = tape[index]
        if index < len(weights)-1:
            # Re-evaluate this layer's activation to recover a, dz, ddz.
            value = weight@a_prev + _unpack(theta, layers)[index][1][:, None]
            first = weight@da_prev
            second = weight@dda_prev
            a = np.tanh(value)
            s = 1-a*a
            gz = (ga*s
                  + gda*(-2*a*s*first)
                  + gdda*(-2*a*s*second - 2*s*s*first*first + 4*a*a*s*first*first))
            gdz = gda*s + gdda*(-4*a*s*first)
            gddz = gdda*s
        else:
            gz, gdz, gddz = ga, gda, gdda

        start, split, end = offsets[index]
        gradient[start:split] = (gz@a_prev.T + gdz@da_prev.T
                                 + gddz@dda_prev.T).ravel()
        gradient[split:end] = gz.sum(axis=1)
        ga = weight.T@gz
        gda = weight.T@gdz
        gdda = weight.T@gddz
    return gradient


class Problem:
    """The solver's own residual system, as a differentiable objective.

    The radial scalings and the boundary rows are taken from
    `egb_rotating_bvp._residual` so this optimises the solver's system rather
    than a re-derivation of it; `tests/test_neural_seed.py` checks the two agree.
    """

    def __init__(self, omega_h, alpha_gb, points=41, horizon_weight=10., layers=LAYERS):
        from ._egb_rotating_compact_generated import rhs
        from ._egb_rotating_horizon_generated import horizon
        self._rhs, self._horizon = rhs, horizon
        self.omega_h = float(omega_h)
        self.alpha_gb = float(alpha_gb)
        self.layers = layers
        self.horizon_weight = float(horizon_weight)
        # Solver node ordering: index 0 is x=1 (infinity), index -1 is x=0.
        self.x = (1+np.cos(np.pi*np.arange(points)/(points-1)))/2
        self.anchor = MyersPerryAnchor(omega_h)
        # Both depend only on x, so they are computed once rather than on every
        # objective and gradient evaluation.
        self._multipliers = multipliers(self.x)
        self._anchor_values = self.anchor(self.x)
        self._u, self._u_x, self._u_xx = network_input(self.x)

    def fields(self, theta):
        """Anchored fields and their first two x-derivatives, plus the tape."""
        raw = _network(theta, self._u, self.layers)
        value, first, second = to_x_derivatives(*raw[:3], self._u_x, self._u_xx)
        out = anchored_fields(self._anchor_values, self._multipliers,
                              value, first, second)
        return out[0], out[1], out[2], (value, first, second, raw[3])

    def _bulk_and_horizon(self, value, first, second):
        B, F, H, W = value
        P, Fx, Q, V = first
        Bxx, _, Hxx, Wxx = second
        z = 1-self.x
        interior = np.array(self._rhs(z[1:-1], B[1:-1], F[1:-1], H[1:-1], W[1:-1],
                                      P[1:-1], Q[1:-1], V[1:-1], self.alpha_gb))
        represented = np.array([Bxx[1:-1], Fx[1:-1], Hxx[1:-1], Wxx[1:-1]])
        scales = np.array([z*z*(1-z)**2, z*(1-z), z*(1-z)**2, z*(1-z)**2])[:, 1:-1]
        bulk = (represented-interior)*scales
        horizon = np.array(list(self._horizon(B[-1], F[-1], H[-1], W[-1], P[-1],
                                              Q[-1], V[-1], Fx[-1], Wxx[-1],
                                              self.alpha_gb))[:3])
        return bulk, horizon, scales

    def objective(self, theta):
        value, first, second, _ = self.fields(np.asarray(theta))
        bulk, horizon, _ = self._bulk_and_horizon(value, first, second)
        # Squared without abs(), so a complex step stays valid.
        return np.sum(bulk*bulk) + self.horizon_weight*np.sum(horizon*horizon)

    def residual_norms(self, theta):
        value, first, second, _ = self.fields(np.asarray(theta, dtype=float))
        bulk, horizon, _ = self._bulk_and_horizon(value, first, second)
        return float(np.max(np.abs(bulk))), float(np.max(np.abs(horizon)))

    def residual_vector(self, theta):
        """The objective as a residual vector, so least-squares solvers can see
        its structure. `sum(residual_vector**2) == objective`."""
        value, first, second, _ = self.fields(np.asarray(theta, dtype=float))
        bulk, horizon, _ = self._bulk_and_horizon(value, first, second)
        return np.concatenate([bulk.ravel(),
                               np.sqrt(self.horizon_weight)*horizon])

    def residual_jacobian(self, theta, step=1e-20):
        """Jacobian of `residual_vector` in the parameters.

        Gauss-Newton needs the full Jacobian, not just the gradient of the sum,
        and getting it is what makes a least-squares method usable here. The
        expensive piece -- the generated equations' Jacobian -- is evaluated once
        and reused for every column; the per-parameter part is a complex step
        through the network alone, which is a few microseconds each.
        """
        theta = np.asarray(theta, dtype=float)
        value, first, second, _ = self.fields(theta)
        d_interior, d_horizon = self._generated_jacobians(value, first, second)
        _, _, scales = self._bulk_and_horizon(value, first, second)
        m0, m1, m2 = self._anchor_values
        points = self.x.size
        rows = 4*(points-2) + 3
        jacobian = np.empty((rows, theta.size))
        base = theta.astype(complex)

        for j in range(theta.size):
            trial = base.copy()
            trial[j] += step*1j
            raw = _network(trial, self._u, self.layers)
            n, dn, ddn = to_x_derivatives(*raw[:3], self._u_x, self._u_xx)
            dv = np.empty((4, points))
            df = np.empty((4, points))
            ds = np.empty((4, points))
            for i, (p, dp, ddp) in enumerate(self._multipliers):
                ni, dni, ddni = n[i].imag/step, dn[i].imag/step, ddn[i].imag/step
                if i < 2:
                    u, du = p*ni, dp*ni + p*dni
                    ddu = ddp*ni + 2*dp*dni + p*ddni
                    dv[i] = m0[i]*u
                    df[i] = m1[i]*u + m0[i]*du
                    ds[i] = m2[i]*u + 2*m1[i]*du + m0[i]*ddu
                else:
                    dv[i] = p*ni
                    df[i] = dp*ni + p*dni
                    ds[i] = ddp*ni + 2*dp*dni + p*ddni

            represented = np.array([ds[0, 1:-1], df[1, 1:-1], ds[2, 1:-1], ds[3, 1:-1]])
            args = [dv[0, 1:-1], dv[1, 1:-1], dv[2, 1:-1], dv[3, 1:-1],
                    df[0, 1:-1], df[2, 1:-1], df[3, 1:-1]]
            interior = sum(d_interior[k]*args[k] for k in range(7))
            jacobian[:4*(points-2), j] = ((represented-interior)*scales).ravel()

            horizon_args = [dv[0, -1], dv[1, -1], dv[2, -1], dv[3, -1],
                            df[0, -1], df[2, -1], df[3, -1], df[1, -1], ds[3, -1]]
            jacobian[4*(points-2):, j] = np.sqrt(self.horizon_weight)*sum(
                d_horizon[k]*horizon_args[k] for k in range(9))
        return jacobian

    def _generated_jacobians(self, value, first, second, step=1e-20):
        """Jacobians of the generated equations, by complex step.

        `rhs` is pointwise, so one complex evaluation per argument recovers a
        whole column across every collocation point at once: seven vectorised
        calls instead of one per network parameter. The generated sources are
        used exactly as they are and never differentiated by hand.
        """
        B, F, H, W = value
        P, Fx, Q, V = first
        Bxx, _, Hxx, Wxx = second
        z = (1-self.x)[1:-1]
        interior_args = [B[1:-1], F[1:-1], H[1:-1], W[1:-1], P[1:-1], Q[1:-1], V[1:-1]]
        # All seven columns in a single call: the perturbed copies are stacked
        # along the point axis, which `rhs` treats independently anyway. Seven
        # separate calls cost 3.7x more here, because evaluating the generated
        # expression is dominated by its per-call overhead at this point count.
        points = z.size
        stacked = [np.tile(np.asarray(arg, dtype=complex), 7) for arg in interior_args]
        for k in range(7):
            stacked[k][k*points:(k+1)*points] += step*1j
        batched = np.array(self._rhs(np.tile(z, 7), *stacked, self.alpha_gb)).imag/step
        d_interior = batched.reshape(4, 7, points).transpose(1, 0, 2)

        horizon_args = [B[-1], F[-1], H[-1], W[-1], P[-1], Q[-1], V[-1], Fx[-1], Wxx[-1]]
        stacked = [np.full(9, complex(arg)) for arg in horizon_args]
        for k in range(9):
            stacked[k][k] += step*1j
        d_horizon = np.array(list(self._horizon(*stacked, self.alpha_gb))[:3]).imag.T/step
        return d_interior, d_horizon

    def gradient(self, theta):
        theta = np.asarray(theta, dtype=float)
        value, first, second, (n, dn, ddn, tape) = self.fields(theta)
        bulk, horizon, scales = self._bulk_and_horizon(value, first, second)
        d_interior, d_horizon = self._generated_jacobians(value, first, second)

        # Adjoints of the field arrays. Columns: B,F,H,W values; then P,Q,V and
        # Fx as first derivatives; then Bxx,Hxx,Wxx as second derivatives.
        adj_value = np.zeros_like(value)
        adj_first = np.zeros_like(first)
        adj_second = np.zeros_like(second)

        weighted = 2*bulk*scales                      # dL/d(represented - interior)
        adj_second[0, 1:-1] += weighted[0]
        adj_first[1, 1:-1] += weighted[1]
        adj_second[2, 1:-1] += weighted[2]
        adj_second[3, 1:-1] += weighted[3]
        # interior enters with the opposite sign, through its seven arguments.
        targets = [(adj_value, 0), (adj_value, 1), (adj_value, 2), (adj_value, 3),
                   (adj_first, 0), (adj_first, 2), (adj_first, 3)]
        for k, (array, row) in enumerate(targets):
            array[row, 1:-1] -= np.einsum('ij,ij->j', weighted, d_interior[k])

        weighted_h = 2*self.horizon_weight*horizon
        horizon_targets = [(adj_value, 0), (adj_value, 1), (adj_value, 2), (adj_value, 3),
                           (adj_first, 0), (adj_first, 2), (adj_first, 3),
                           (adj_first, 1), (adj_second, 3)]
        for k, (array, row) in enumerate(horizon_targets):
            array[row, -1] += float(weighted_h@d_horizon[k])

        # Back through the anchored ansatz to the raw network outputs.
        m0, m1, m2 = self._anchor_values
        grad_n = np.zeros_like(n, dtype=float)
        grad_dn = np.zeros_like(dn, dtype=float)
        grad_ddn = np.zeros_like(ddn, dtype=float)
        for i, (p, dp, ddp) in enumerate(self._multipliers):
            a0, a1, a2 = adj_value[i], adj_first[i], adj_second[i]
            if i < 2:
                gu = a0*m0[i] + a1*m1[i] + a2*m2[i]
                gdu = a1*m0[i] + 2*a2*m1[i]
                gddu = a2*m0[i]
            else:
                gu, gdu, gddu = a0, a1, a2
            grad_n[i] = gu*p + gdu*dp + gddu*ddp
            grad_dn[i] = gdu*p + 2*gddu*dp
            grad_ddn[i] = gddu*p
        # The network differentiates in u, not x; carry the adjoints across.
        return _network_backward(theta,
                                 (grad_n,
                                  grad_dn*self._u_x + grad_ddn*self._u_xx,
                                  grad_ddn*self._u_x*self._u_x),
                                 tape, self.layers)


def check_gradient(problem, theta, step=1e-20):
    """Compare the analytic gradient against a complex step over every parameter.

    The complex step is exact to rounding, so this is a genuine oracle rather
    than a finite-difference approximation, and it covers all parameters rather
    than a sample.
    """
    theta = np.asarray(theta, dtype=float)
    analytic = problem.gradient(theta)
    oracle = np.empty_like(theta)
    base = theta.astype(complex)
    for j in range(theta.size):
        trial = base.copy()
        trial[j] += step*1j
        oracle[j] = problem.objective(trial).imag/step
    scale = max(1., float(np.max(np.abs(oracle))))
    return float(np.max(np.abs(analytic-oracle))/scale), analytic, oracle


class Seed:
    """A trained (or untrained) network, in the form `solve` accepts.

    Evaluated at arbitrary x rather than only at the training collocation
    points, so the solver can sample it on its own nodes.
    """

    def __init__(self, problem, theta):
        self.problem = problem
        self.theta = np.asarray(theta, dtype=float)

    def evaluate(self, x, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('derivative must be 0, 1 or 2')
        x = np.asarray(x, dtype=float)
        u, u_x, u_xx = network_input(x)
        raw = _network(self.theta, u, self.problem.layers)
        value, first, second = to_x_derivatives(*raw[:3], u_x, u_xx)
        return np.real(anchored_fields(self.problem.anchor(x), multipliers(x),
                                       value, first, second)[derivative])


class TrivialSeed:
    """The guess the solver falls back on with no `previous`; the baseline."""

    def __init__(self, omega_h):
        self.omega_h = float(omega_h)

    def evaluate(self, x, derivative=0):
        x = np.asarray(x, dtype=float)
        out = np.zeros((4, x.size))
        if derivative == 0:
            out[0] = 1.
            out[1] = 1.
            out[3] = self.omega_h
        return out
