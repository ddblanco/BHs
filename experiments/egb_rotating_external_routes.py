"""Test branch dependence of the discrepant paper ergosurface datum."""
import hashlib
import json
from pathlib import Path
import numpy as np
from rotating_bh.egb_rotating_bvp import solve
from rotating_bh.egb_rotating_saved import SavedProfile
from rotating_bh.egb_rotating_observables import measure,ergosurface
from rotating_bh.egb_rotating_validation import diagnose

ROOT = Path(__file__).resolve().parents[1]


class StaticSeed:
    def __init__(self,alpha):
        self.alpha = alpha

    def evaluate(self,x):
        z = 1-np.asarray(x)
        a = self.alpha
        B = 2/(1+4*a*z*z+np.sqrt(1+8*a*(1+2*a)*z**4))
        return np.array([B,B,np.zeros_like(z),np.zeros_like(z)])


def run():
    data = json.loads((ROOT/'results/egb-rotating-family.json').read_text())
    target_row = next(r for r in data['records'] if r['label']=='external' and r['alpha_gb']==.5)
    target = SavedProfile(target_row)
    target_tensor = diagnose(target,.5,count=51)['max_tensor_residual']
    if target_tensor>=1e-6:
        raise ValueError('stored target fails independent tensor check')
    previous = StaticSeed(.5)
    rows = []
    for q in [0.,.05,.1,.15,.2,.25,.3,.33]:
        print(f'static route: alpha_GB=.5 q={q}',flush=True)
        try:
            previous = solve(q,.5,previous=previous,resolution=32,tol=1e-11,max_iterations=20)
            tensor = diagnose(previous,.5,count=51)['max_tensor_residual']
            rows.append(dict(q=q,tensor=tensor,boundary_residual=previous.boundary_residual,
                iterations=previous.iterations,observables=measure(previous),ergo_radius=ergosurface(previous)))
            if tensor>=1e-6 or previous.boundary_residual>=1e-8:
                raise RuntimeError('independent physical acceptance failed')
        except RuntimeError as exc:
            rows.append(dict(q=q,error=str(exc)))
            break
    complete = 'error' not in rows[-1] and rows[-1]['q']==.33
    delta = float(np.max(np.abs(previous.evaluate(np.linspace(0,1,301))-target.evaluate(np.linspace(0,1,301))))) if complete else None
    result = dict(alpha_gb=.5,target_q=.33,method='Static EGB seed; spin continuation independent of alpha route',
        target_tensor=target_tensor,rows=rows,profile_difference=delta,
        checks=dict(route_completed=complete,routes_agree=complete and delta<1e-8))
    files = ['experiments/egb_rotating_external_routes.py','experiments/egb_rotating_family.py',
             'src/rotating_bh/egb_rotating_saved.py','results/egb-rotating-family.json',
             *data['sources']]
    result['source_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in dict.fromkeys(files)}
    output = ROOT/'results/egb-rotating-external-routes.json'
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = dict(id='egb-rotating-external-routes',kind='result',path=str(output.relative_to(ROOT)).replace('\\','/'),
        command='python experiments/egb_rotating_external_routes.py',commit='working-tree',inputs=list(result['source_sha256']),
        parameters=dict(alpha_gb=.5,source_sha256=result['source_sha256']),environment='environment/requirements-lock.txt',
        agent='codex',prompt_refs=['prompts/prompt-log.md'],decisions=['Test continuation-route dependence without changing paper tolerance'],
        checks=[dict(name=k,passed=bool(v)) for k,v in result['checks'].items()],
        status='verified' if all(result['checks'].values()) else 'rejected',sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id']!=record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result['checks']),flush=True)
    print('Profile difference:',delta,flush=True)


if __name__=='__main__':
    run()
