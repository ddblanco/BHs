"""Reproducible Hito 4C family; run with PYTHONPATH=src from project/."""
import hashlib
import json
from pathlib import Path
import sys
import platform
import scipy
from scipy.fft import dct

import numpy as np
from rotating_bh.egb_rotating_bvp import solve
from rotating_bh.egb_rotating_adaptive import solve as adaptive_solve
from rotating_bh.egb_rotating_validation import diagnose, physical_jets
from rotating_bh.egb_rotating_observables import measure, ergosurface
from rotating_bh.static_egb import StaticEGB
from rotating_bh.egb_rotating_family_validation import checks as acceptance_checks

ROOT = Path(__file__).resolve().parents[1]


class StaticSeed:
    def __init__(self, alpha):
        self.alpha = alpha

    def evaluate(self, x):
        z = 1-np.asarray(x)
        a = self.alpha
        B = 2/(1+4*a*z*z+np.sqrt(1+8*a*(1+2*a)*z**4))
        return np.array([B,B,np.zeros_like(z),np.zeros_like(z)])


class Perturbed:
    def __init__(self, previous):
        self.previous = previous

    def evaluate(self, x):
        return 1.01*self.previous.evaluate(x)


def run():
    records = []
    def step(q,a,previous,label,n=24,max_iterations=50):
        print(f'{label}: q={q:g} alpha={a:g} N={n}', flush=True)
        try:
            s = solve(q,a,previous=previous,resolution=n,tol=1e-11,max_iterations=max_iterations)
            d = diagnose(s,a,count=51)
            accepted = d['max_tensor_residual'] < 1e-6 and s.boundary_residual < 1e-8
            record = dict(label=label,q=q,alpha_gb=a,N=n,iterations=s.iterations,
                          boundary_residual=s.boundary_residual,**d,accepted=bool(accepted),
                          observables=measure(s),ergo_radius=ergosurface(s))
            coefficients = dct(s.evaluate(s.nodes),type=1,axis=1)/(n-1)
            coefficients[:,[0,-1]] *= .5
            record['chebyshev_coefficients'] = coefficients.tolist()
            records.append(record)
            (ROOT/'.cache/hito-4c-progress.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            if not accepted:
                raise RuntimeError('independent tensor/boundary acceptance failed')
            return s,record
        except Exception as exc:
            records.append(dict(label=label,q=q,alpha_gb=a,N=n,max_iterations=max_iterations,
                                error=str(exc),accepted=False))
            (ROOT/'.cache/hito-4c-progress.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            raise

    family = {}
    refinement = []
    for a in [.05,.1]:
        previous = StaticSeed(a)
        # Check the regularized seed against the existing static oracle.
        x = np.linspace(.01,.99,101)
        oracle = StaticEGB(1,a).functions(1/(1-x))['b']
        seed_error = float(np.max(np.abs((1-(1-x)**2)*previous.evaluate(x)[0]-oracle)))
        assert seed_error < 1e-13
        for q in [0.,.1,.2,.3,.4]:
            previous,coarse = step(q,a,previous,'family')
            fine,record = step(q,a,Perturbed(previous),'resolution',n=32)
            family[a,q] = fine
            refinement.append(dict(q=q,alpha_gb=a,seed_oracle_error=seed_error,
                differences={k:abs(record['observables'][k]-coarse['observables'][k])
                             for k in ['E','J','T_H','S']}))

    previous = None
    for a in [0.,.02,.05,.1]:
        previous,_ = step(.3,a,previous,'alpha-route',n=32)
    x = np.linspace(0,1,301)
    route_difference = float(np.max(np.abs(previous.evaluate(x)-family[.1,.3].evaluate(x))))

    central = measure(family[.1,.3])
    law = []
    for h in [.02,.01,.005]:
        neighbors = []
        for q in [.3-h,.3+h]:
            _, rec = step(q,.1,family[.1,.3],'first-law',n=32)
            neighbors.append(rec['observables'])
        derivative = {k:(neighbors[1][k]-neighbors[0][k])/(2*h) for k in ['E','J','S']}
        rhs = central['T_H']*derivative['S']+2*.3*derivative['J']
        law.append(dict(step=h,derivatives=derivative,rhs=rhs,
                        residual=derivative['E']-rhs,
                        relative_residual=abs(derivative['E']-rhs)/abs(derivative['E'])))

    cutoff = []
    for c in [.003,.0015]:
        print(f'adaptive cutoff={c}',flush=True)
        s = adaptive_solve(.3,.1,previous=family[.1,.3],cutoff=c,tol=1e-7)
        d = diagnose(s,.1,count=51)
        obs = measure(s)
        cutoff.append(dict(cutoff=c,**d,boundary_residual=s.boundary_residual,
            observables=obs,differences={k:abs(obs[k]-central[k]) for k in ['E','J','T_H','S']}))

    previous = family[.1,.3]
    previous,_ = step(.33,.1,previous,'external')
    external = []
    for a in [.15,.2,.25,.3,.35,.4,.45,.5]:
        previous,rec = step(.33,a,previous,'external',n=32)
        if a in [.25,.5]:
            published,rounding = (1.08,.005) if a == .25 else (1.104,.0005)
            jets = physical_jets(previous,np.array([1-1/rec['ergo_radius'],1-1/published]))
            gtt = -jets[1]+jets[7]*jets[10]**2
            slope = -jets[2,0]+jets[8,0]*jets[10,0]**2+2*jets[7,0]*jets[10,0]*jets[11,0]
            external.append(dict(alpha_paper=4*a,alpha_gb=a,q=.33,r_h=1.,
                published_radius=published,rounding_half_unit=rounding,
                computed_radius=rec['ergo_radius'],
                gtt_at_computed=float(gtt[0]),gtt_at_published=float(gtt[1]),
                gtt_radial_slope=float(slope),
                absolute_difference=abs(rec['ergo_radius']-published)))

    # Independently investigate the discrepancy at alpha_paper=2, without
    # increasing the previously chosen comparison tolerance.
    fine,rec = step(.33,.5,Perturbed(previous),'external-refinement',n=40)
    external_validation = [dict(method='spectral-N40',radius=rec['ergo_radius'],
                               tensor=rec['max_tensor_residual'])]
    print('external adaptive cross-check',flush=True)
    independent = adaptive_solve(.33,.5,previous=fine,cutoff=.0015,tol=1e-7)
    external_validation.append(dict(method='adaptive',radius=ergosurface(independent),
        tensor=diagnose(independent,.5,count=51)['max_tensor_residual'],
        boundary_residual=independent.boundary_residual))
    profiles = []
    # The prior .5 -> .6 jump exhausted 50 iterations. Reduce the step
    # fourfold, with a bounded budget; preserve a failure without losing
    # the already validated family and external comparison.
    for a in [.525,.55,.575,.6,.625,.65,.675,.7,.725,.75]:
        try:
            previous,_ = step(.33,a,previous,'figure-1b',n=32,max_iterations=15)
        except RuntimeError:
            break
    high = previous
    previous = family[.05,.3]
    # The 0.05 -> 0.02 jump exhausted its 15 iterations at residual 0.0928415.
    # The budget was not the constraint: with the step subdivided, every rung
    # below converges in three to four iterations at a tensor residual near
    # 1e-13. The earlier failure measured the continuation step, not a physical
    # boundary of the branch, so the budget stays at 15 to keep saying so.
    for a in [.05,.04,.03,.02,.015,.01,.0075,.005,.0035,.0025]:
        try:
            previous,_ = step(.33,a,previous,'figure-1b',n=32,max_iterations=15)
        except RuntimeError:
            break
    for s in [previous,high]:
        r = np.logspace(0,1.25,201)
        z = 1/r
        B,F,H,W = s.evaluate(1-z)
        profiles.append(dict(alpha_gb=s.alpha_gb,q=.33,r=r.tolist(),
            b=((1-z*z)*B).tolist(),f=((1-z*z)*F).tolist(),
            h_over_r2=(1+z**4*H).tolist(),w=(z**4*W).tolist()))
    sources = [Path(__file__),ROOT/'src/rotating_bh/egb_rotating_observables.py',
               ROOT/'src/rotating_bh/egb_rotating_bvp.py',ROOT/'src/rotating_bh/egb_rotating_adaptive.py',
               ROOT/'src/rotating_bh/egb_rotating_validation.py',ROOT/'src/rotating_bh/einstein.py',
               ROOT/'src/rotating_bh/egb_rotating_family_validation.py',
               ROOT/'src/rotating_bh/egb_rotating_angular.py',ROOT/'src/rotating_bh/static_egb.py',
               ROOT/'environment/requirements-lock.txt']
    sources += list((ROOT/'src/rotating_bh').glob('_egb_rotating*_generated.py'))
    report = dict(schema=1,units='r_H=G5=1; J is each spin; alpha_paper=4 alpha_gb',
        external_source='https://arxiv.org/html/1010.0860v1#S4.SS1.SSS1',
        external_location='section 4.1.1, paragraph after Figure 1; approximate text values, no digitization',
        records=[r for r in records if 'error' not in r],
        failed_attempts=[r for r in records if 'error' in r],
        refinement=refinement,route_difference=route_difference,
        first_law=law,cutoff=cutoff,external=external,external_validation=external_validation,
        profiles=profiles,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        sources={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    report['checks'] = acceptance_checks(report)
    (ROOT/'results/egb-rotating-family.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['checks'],indent=2),flush=True)
    return report


if __name__ == '__main__':
    result = run()
    sys.exit(0 if all(result['checks'].values()) else 1)
