import json,sys
from pathlib import Path
import numpy as np
from rotating_bh.near_horizon import extremal
from rotating_bh._near_horizon_generated import entropy_function
D=json.loads(Path('results/egb-extremality.json').read_text())
print('NEAR-HORIZON TEST TOLERANCE (existing test uses step=1e-6)')
for a in [.05,.3,.7]:
 e=extremal(a); v=[e['v1'],e['v2']*1.01,e['v3']]; step=1e-6; data=[]
 for i in range(3):
  p=v.copy();m=v.copy();p[i]+=step;m[i]-=step
  fm=entropy_function(*m,e['k'],a);fp=entropy_function(*p,e['k'],a)
  derivative=(fp-fm)/(2*step);tol=1e-6*max(1,abs(fm)/step)
  data.append([float(derivative),float(tol),bool(abs(derivative)<tol)])
 print(a,data)
print('PRIOR REVIEW ARTIFACTS (read this session; not rerun)')
for p in sorted(Path('work/review/out2').glob('*.json')):
 d=json.loads(p.read_text());good=[r for r in d['rows'] if 'mine' in r]; bad=[r for r in d['rows'] if 'failed' in r]
 print(p.name,'rows',len(d['rows']),'reconstructed',len(good),'failed',len(bad))
print('SIGN-LOCUS BRACKETS')
for r in D['sign_locus']:
 print(json.dumps(r))
