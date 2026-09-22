"""A scale-aware reading of the same tensor residual (Hito 8).

Like `egb_rotating_predictor`, this sits beside the audited module rather than
inside it: `egb_rotating_validation.py` is a recorded input of every artifact
from Hito 2 on, and its sha256 has to keep matching the bytes those runs saw.
Nothing here changes how the residual is computed -- `physical_jets`, the
compiled metric jets and the sampling points are all imported unchanged.

What it adds is the *scale* the residual should be read against.
`tensor_residual` returns an absolute number, and the Hito 4 gate on it is
absolute too, which is the right instrument while the curvature is of order
one. Approaching extremality it stops being: with the temperature five decades
below the static scale, |G| and |alpha_GB H| each reach O(100) near the horizon
while their sum stays at 1e-6, so the absolute gate starts rejecting solutions
whose relative error is 1e-9. That looked like a wall in Hito 6 and it is not
one.

The scale cannot be |G| alone. On a vacuum solution the Einstein and Ricci
tensors vanish identically, so dividing by them turns a residual of 1e-12 into
a relative error of one -- which is what the first Hito 8 production run hit at
alpha_GB=0. Including the square root of the Kretschmann invariant
`R_{abcd} R^{abcd}`, a coordinate invariant of the same dimension as the
residual and nonzero on every black hole, fixes that.
"""
import numpy as np

from .egb_rotating_validation import _metric_jets, check_points, physical_jets
from .einstein import curvature, gauss_bonnet

__all__ = ['tensor_parts', 'scaled_diagnose']


def tensor_parts(solution, x, alpha_gb, theta=0.6):
    """G+alpha_GB H, its two halves, and the Kretschmann scale, at one point."""
    jets = physical_jets(solution, np.array([x]))[:, 0]
    r_, b, bp, bpp, f, fp, fpp, h, hp, hpp, w, wp, wpp = jets
    arrays = _metric_jets()(0, r_, theta, 0, 0, r_, b, bp, bpp, f, fp, fpp,
                            h, hp, hpp, w, wp, wpp)
    result = curvature(*arrays)
    gb = gauss_bonnet(arrays[0], result['inverse'], result['riemann'],
                      result['ricci'], result['scalar'])
    ruler = np.array([1, 1, r_, r_, r_])
    scales = np.outer(ruler, ruler)
    einstein = result['einstein']/scales
    lovelock = alpha_gb*gb['H']/scales
    lower = np.einsum('ra,asmn->rsmn', arrays[0], result['riemann'])
    upper = np.einsum('ra,sb,mc,nd,abcd->rsmn', result['inverse'], result['inverse'],
                      result['inverse'], result['inverse'], lower)
    kretschmann = float(np.einsum('abcd,abcd->', lower, upper))
    return (float(np.max(np.abs(einstein+lovelock))),
            float(np.max(np.abs(einstein))), float(np.max(np.abs(lovelock))),
            float(np.sqrt(abs(kretschmann))))


def scaled_diagnose(solution, alpha_gb, count=51):
    """Absolute and relative sampled residuals over the same fixed points.

    `max_tensor_residual` is bit-for-bit the number `diagnose` reports, so a
    state measured here stays comparable with every state of Hitos 4-6.
    """
    points = check_points(solution, count)
    sampled = [tensor_parts(solution, x, alpha_gb) for x in points]
    absolute = max(v[0] for v in sampled)
    scale = max(max(v[1], v[2], v[3]) for v in sampled)
    return dict(max_tensor_residual=float(absolute),
                max_tensor_scale=float(scale),
                max_relative_tensor_residual=(float(absolute/scale) if scale
                                              else float('inf')),
                count=count)
