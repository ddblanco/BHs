"""Continuation with explicit attempt logs and bounded step reduction."""
import numpy as np

from .vacuum_bvp import solve, SolverFailure


class ContinuationFailure(SolverFailure):
    def __init__(self,message,attempts):
        super().__init__(message)
        self.attempts=attempts


def continue_family(targets,*,method,resolution,tol,**solver_options):
    """Return destination solutions and all attempts, including rejected steps.

    Convergence here is numerical only; independent diagnosis decides whether
    a candidate is scientifically accepted. No analytic rotating seed is used.
    """
    targets=list(targets)
    if not targets or targets[0]!=0:
        raise ValueError('continuation must start at q=0')
    solutions=[];attempts=[];previous=None
    for target in targets:
        trial=float(target);failures=0
        while True:
            origin=None if previous is None else previous.q
            try:
                candidate=solve(trial,method=method,resolution=resolution,tol=tol,
                                previous=previous,**solver_options)
            except SolverFailure as error:
                attempts.append(dict(origin=origin,target=float(target),trial=trial,
                                     converged=False,reason=str(error)))
                failures+=1
                if previous is None or failures>=20 or abs(trial-previous.q)/2<1e-3:
                    raise ContinuationFailure(f'continuation failed towards q={target}',attempts) from error
                trial=previous.q+(trial-previous.q)/2
                continue
            attempts.append(dict(origin=origin,target=float(target),trial=trial,
                                 converged=True,nodes=len(candidate.nodes),iterations=candidate.iterations))
            previous=candidate
            if abs(trial-target)<1e-14:
                solutions.append(candidate)
                break
            trial=float(target)
    return solutions,attempts
