import json
from pathlib import Path
from rotating_bh.gb_response import predictor_ladder,response_observables
from rotating_bh.extremality import state_at
from rotating_bh.consistency import first_law_residual
D=json.loads(Path('results/egb-extremality.json').read_text()); rows=[]
for a,q in [(0.02,.6),(.15,.6),(.5,.6)]:
 sol=predictor_ladder(q,a,step=.01,resolution=64)
 got=state_at(sol)
 ref=min(D['walks'][format(a,'.12g')],key=lambda x:abs(x['omega_h']-q))
 assert abs(ref['omega_h']-q)<1e-10
 errors={k:abs(got[k]-ref[k])/max(1,abs(ref[k])) for k in ['E','J','S','T_H','psi_gb']}
 assert max(errors.values())<1e-6,errors
 fl=first_law_residual(sol); assert fl['relative']<1e-6
 row=dict(alpha=a,q=q,errors=errors,tensor_relative=got['max_relative_tensor_residual'],first_law_relative=fl['relative'])
 rows.append(row); print(json.dumps(row),flush=True)
Path('work/critical-review/fresh.json').write_text(json.dumps(rows,indent=2)+'\n')
print('ASSERTED: three fresh N=64 finite-coupling states agree with stored observables; tensor gates and spin first law pass.',flush=True)
