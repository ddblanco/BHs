"""Deterministic continuation, convergence, cutoff and independent checks."""
import numpy as np

from .vacuum_bvp import solve, SolverFailure
from .vacuum_continuation import continue_family
from .vacuum_validation import diagnose, physical_jets, tensor_residual

POSITIVE=[0,.1,.2,.33,.5,.6]
NEGATIVE=[0,-.1,-.2,-.33]


def run_benchmark():
    runs=[];logs=[];selected={};successive=[];last_profiles={}
    x=np.linspace(.01,.99,201)
    configs=[('spectral',n,1e-11) for n in (12,20,32,48)]
    configs += [('adaptive',32,tol) for tol in (1e-4,1e-6,1e-8)]
    for method,n,tol in configs:
        all_solutions=[];attempts=[]
        for branch in (POSITIVE,NEGATIVE):
            solutions,branch_log=continue_family(branch,method=method,resolution=n,tol=tol)
            all_solutions.extend(solutions if branch is POSITIVE else solutions[1:])
            attempts.extend(branch_log)
        logs.append(dict(method=method,resolution=n,tolerance=tol,attempts=attempts))
        for solution in all_solutions:
            runs.append(dict(method=method,resolution=n,tolerance=tol,q=solution.q,
                nodes=len(solution.nodes),iterations=solution.iterations,cutoff=solution.cutoff,
                **diagnose(solution)))
            if (method=='spectral' and n==32) or (method=='adaptive' and tol==1e-8):
                selected[(method,solution.q)]=solution
            if solution.q==.6:
                profile=solution.evaluate(x)
                if method in last_profiles:
                    old_n,old_tol,old_profile=last_profiles[method]
                    successive.append(dict(method=method,q=.6,previous_resolution=old_n,
                        resolution=n,previous_tolerance=old_tol,tolerance=tol,
                        profile_difference=float(np.max(abs(profile-old_profile)))))
                last_profiles[method]=(n,tol,profile)
    cutoff_study=[]
    for cutoff in (.01,.003,.001):
        solutions,attempts=continue_family(POSITIVE,method='adaptive',resolution=32,tol=1e-8,cutoff=cutoff)
        cutoff_study.append(dict(cutoff=cutoff,**diagnose(solutions[-1]),attempts=attempts))
    differences=[float(np.max(abs(selected[('spectral',q)].evaluate(x)
                                    -selected[('adaptive',q)].evaluate(x))))
                 for q in POSITIVE+NEGATIVE[1:]]
    local=physical_jets(selected[('spectral',.33)],1.3)
    local[:,3]*=1.01
    negative=tensor_residual(1.3,.7,local)
    failures=[]
    for method,options in [('spectral',dict(max_iterations=0)),('adaptive',dict(max_nodes=8))]:
        try:
            solve(.3,method=method,resolution=8,tol=1e-12,**options)
        except SolverFailure as error:
            failures.append(dict(method=method,converged=False,reason=str(error)))
        else:
            failures.append(dict(method=method,converged=True))
    fine=[r for r in runs if (r['method']=='spectral' and r['resolution']>=20)
          or (r['method']=='adaptive' and r['tolerance']==1e-8)]
    spectral=[r for r in runs if r['method']=='spectral' and r['q']==.6]
    adaptive=[r for r in runs if r['method']=='adaptive' and r['q']==.6]
    checks=[
        dict(name='fine_solutions_independently_accepted',passed=all(r['accepted'] for r in fine)),
        dict(name='spectral_convergence_before_roundoff',passed=spectral[1]['profile_error']<spectral[0]['profile_error']/100
             and spectral[2]['profile_error']<spectral[1]['profile_error']/100),
        dict(name='adaptive_ode_convergence',passed=adaptive[-1]['ode_residual']<adaptive[0]['ode_residual']/100),
        dict(name='cutoff_convergence',passed=cutoff_study[-1]['profile_error']<cutoff_study[0]['profile_error']/20
             and cutoff_study[-1]['accepted']),
        dict(name='methods_agree',passed=max(differences)<1e-6,value=max(differences)),
        dict(name='perturbation_detected',passed=negative>1e-5,value=negative),
        dict(name='real_resource_failures_reported',passed=all(not f['converged'] for f in failures)),
    ]
    profiles=[dict(method=method,q=.6,x=x.tolist(),fields=selected[(method,.6)].evaluate(x).tolist())
              for method in ('spectral','adaptive')]
    return dict(parameters=dict(r_h=1.,G5=1.,positive_targets=POSITIVE,negative_targets=NEGATIVE,
        spectral_resolutions=[12,20,32,48],spectral_tolerance=1e-11,
        adaptive_tolerances=[1e-4,1e-6,1e-8],adaptive_initial_resolution=32,
        default_cutoff=.003,cutoffs=[.01,.003,.001],min_continuation_step=1e-3,
        max_retries=20,check_points=401,profile_tolerance=1e-6,ode_tolerance=1e-6,
        tensor_tolerance=1e-6,constraint_tolerance=1e-6,boundary_tolerance=1e-8),
        runs=runs,continuation=logs,cutoff_study=cutoff_study,profiles=profiles,
        successive_differences=successive,
        method_differences=differences,negative_control_residual=negative,
        failure_controls=failures,checks=checks)
