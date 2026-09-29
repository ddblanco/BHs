import sys,json,platform,hashlib
from pathlib import Path
import numpy as np
import sympy as s
sys.path.insert(0,str(Path('manuscript').resolve()))
from reanalysis import sensitivity
from rotating_bh.extremality import extremal_state
from rotating_bh.near_horizon import coupling_from_invariant,extremal
out={}; checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
r,a,J=s.symbols('r alpha J',positive=True)
M=3*s.pi*(r*r+2*a)/8; T=r/(2*s.pi*(r*r+4*a)); S=s.pi**2*r**3/2+6*s.pi**2*a*r
psi=s.simplify(s.diff(M,a)-T*s.diff(S,a))
check('static radial first law',s.simplify(s.diff(M,r)-T*s.diff(S,r))==0)
check('static constrained potential',s.simplify(psi-3*s.pi/4*(4*a-3*r*r)/(r*r+4*a))==0)
check('static Smarr',s.simplify(2*M-3*T*S-2*a*psi)==0)
check('static scaled temperature',s.simplify(4*s.sqrt(2*s.pi/3)*T*s.sqrt(M)-r*s.sqrt(r*r+2*a)/(r*r+4*a))==0)
check('static zero',s.simplify(psi.subs(a,3*r*r/4))==0)
mu=s.Function('mu'); y=a/J**s.Rational(2,3); mass=J**s.Rational(2,3)*mu(y)
check('fixed-J scale derivative',s.simplify(s.diff(mass,a)-s.Subs(s.diff(mu(s.Symbol('y')),s.Symbol('y')),s.Symbol('y'),y))==0)
u=s.symbols('u'); p=-s.pi*(u*u-14*u+9)/4
check('perturbative endpoints',p.subs(u,0)==-9*s.pi/4 and p.subs(u,1)==s.pi)
D,R,A=s.symbols('D R A',positive=True)
g=-(D-3)*2*s.pi**2/(16*s.pi*R**4*(A*A+R*R)**((5-D)/2))*(A**4-2*(2*D-3)*A*A*R*R+(D-2)**2*R**4)
check('Wu-Lu eq33 specialization',s.simplify(g.subs(D,5).subs(A,R*s.sqrt(u))-p)==0)
check('MP extremal mass normalization',s.simplify((3*s.pi*A*A/2)/(s.pi*A**3)**s.Rational(2,3)-3*s.pi**s.Rational(1,3)/2)==0)
out['symbolic_assertions']=checks.copy()
d=json.loads(Path('results/egb-extremality.json').read_text()); rows=[]
for e in d['extremals']:
 k=format(e['alpha_gb'],'.12g'); states=d['walks'][k]; re=extremal_state(states)
 for key in ('E','J','S','psi_gb'):
  check(k+' '+key+' cubic intercept',abs(re[key]['value']-e[key]['value'])<1e-10)
 se=sensitivity(states,e['psi_gb']['value']); env=max(se['envelope'],e['psi_gb']['spread'])
 # Apply exactly the potential model family jointly to M,J,S, then form invariants.
 fits={key:sensitivity([dict(v,psi_gb=v[key]) for v in states],e[key]['value'])['intercepts'] for key in ('E','J','S')}
 alt=[]
 for model in fits['J']:
  jf=fits['J'][model]
  assert jf>0
  alt.append((e['alpha_gb']/jf**(2/3),fits['E'][model]/jf**(2/3),fits['S'][model]/jf))
 nh=extremal(coupling_from_invariant(e['y']))['S_over_J']
 rows.append(dict(alpha=e['alpha_gb'],y=e['y'],mu=e['mu'],psi=e['psi_gb']['value'],envelope=env,psi_low=e['psi_gb']['value']-env,Tmin=min(v['T_H'] for v in states),tau_min=min(v['tau'] for v in states),mu_model_shift=max(abs(v[1]-e['mu']) for v in alt),y_model_shift=max(abs(v[0]-e['y']) for v in alt),sigma_model_shift=max(abs(v[2]-e['sigma']) for v in alt),nh_relative=abs(nh/e['sigma']-1)))
check('all sampled envelopes positive',all(v['psi_low']>0 for v in rows))
check('all central mass increments positive',all(v['mu']>u['mu'] for u,v in zip(rows,rows[1:])))
states=[v for group in d['walks'].values() for v in group]
check('356 states at 13 couplings',len(states)==356 and len(rows)==13)
check('all recorded tensor gates pass',max(v['max_relative_tensor_residual'] for v in states)<1e-8)
out.update(rows=rows,assertion_count=len(checks),state_count=len(states),min_psi_lower=min(v['psi_low'] for v in rows),max_smarr=max(v['smarr'] for v in states),max_relative_tensor=max(v['max_relative_tensor_residual'] for v in states),max_absolute_tensor=max(v['max_tensor_residual'] for v in states),failed_absolute_gate=sum(not v['absolute_gate_passed'] for v in states),max_nh_relative=max(v['nh_relative'] for v in rows),environment=dict(python=platform.python_version(),numpy=np.__version__,sympy=s.__version__),sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('manuscript/main.pdf'),Path('manuscript/main.tex'),Path('results/egb-extremality.json')]})
Path('work/critical-review/audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('ASSERTIONS:',len(checks),'passed (symbolic identities, saved-data fits and diagnostics; not a fresh 356-state solve)')
for key in ('state_count','min_psi_lower','max_smarr','max_relative_tensor','max_absolute_tensor','failed_absolute_gate','max_nh_relative'): print(key,out[key])
print('alpha   Tmin       Psi        envelope   delta_mu_models delta_y_models')
for v in rows: print(f"{v['alpha']:5g} {v['Tmin']:.3e} {v['psi']:.7f} {v['envelope']:.3e} {v['mu_model_shift']:.3e} {v['y_model_shift']:.3e}")
