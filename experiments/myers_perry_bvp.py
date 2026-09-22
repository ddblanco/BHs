"""Generate Hito 2 data, convergence figure and source-hashed provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT=Path(__file__).resolve().parents[1]
for name,relative in [('TEMP','.cache/tmp'),('TMP','.cache/tmp'),('MPLCONFIGDIR','.cache/matplotlib')]:
    directory=ROOT/relative;directory.mkdir(parents=True,exist_ok=True)
    os.environ[name]=str(directory)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sympy

from rotating_bh.vacuum_benchmark import run_benchmark
from rotating_bh.provenance import validate_manifest, validate_record


def main():
    manifest=ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    records=json.loads(manifest.read_text(encoding='utf-8'))
    result=run_benchmark()
    sources=['src/rotating_bh/'+name+'.py' for name in
        ('vacuum_equations','_vacuum_generated','vacuum_bvp','vacuum_continuation',
         'vacuum_validation','vacuum_benchmark','einstein','provenance')]
    sources += ['experiments/derive_vacuum.py','experiments/myers_perry_bvp.py',
                'environment/requirements-lock.txt','docs/vacuum-bvp.md','docs/convenciones.md']
    result['parameters']['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    result['versions']={'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'sympy':sympy.__version__,'matplotlib':matplotlib.__version__}
    result['scope']='Nonextremal GR; sampled coordinate norms, explicit adaptive cutoff; no EGB or global bounds'
    result['representation']='Chebyshev polynomials; adaptive integrated velocity splines for B,H,W and cubic F'
    passed=all(c['passed'] for c in result['checks'])
    data=ROOT/'results/myers-perry-bvp.json';figure=ROOT/'artifacts/myers-perry-bvp.png'
    data.parent.mkdir(parents=True,exist_ok=True);figure.parent.mkdir(parents=True,exist_ok=True)
    data.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for method,ax in zip(('spectral','adaptive'),axes):
        rows=[r for r in result['runs'] if r['method']==method and r['q']==.6]
        coordinate=[r['resolution'] if method=='spectral' else r['tolerance'] for r in rows]
        for key,label in [('profile_error','Perfil'),('ode_residual','ODE'),('einstein_residual','Einstein')]:
            ax.loglog(coordinate,[r[key] for r in rows],'o-',label=label)
        ax.set_xlabel('Nodos Lobatto' if method=='spectral' else 'Tolerancia adaptativa')
        ax.set_title('Espectral' if method=='spectral' else 'Adaptativo, corte 0.003')
        if method=='adaptive':ax.invert_xaxis()
        ax.axhline(1e-6,color='gray',ls=':',label='Umbral')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[0].set_ylabel('Máximo muestreado')
    rows=result['cutoff_study']
    axes[2].loglog([r['cutoff'] for r in rows],[r['profile_error'] for r in rows],'o-')
    axes[2].invert_xaxis();axes[2].set_xlabel('Corte epsilon')
    axes[2].set_title('Error de perfil frente al corte')
    axes[2].grid(alpha=.25)
    fig.suptitle('Myers–Perry recuperado por BVP · q=0.6 · r_H=1 · GR')
    fig.savefig(figure,dpi=160,metadata={'Software':'rotating-bh Hito 2'});plt.close(fig)
    ids={'myers-perry-bvp','myers-perry-bvp-convergence'}
    records=[r for r in records if r['id'] not in ids]
    for artifact_id,kind,path in [('myers-perry-bvp','result',data),('myers-perry-bvp-convergence','figure',figure)]:
        record=dict(id=artifact_id,kind=kind,path=path.relative_to(ROOT).as_posix(),
            command='.venv/Scripts/python experiments/myers_perry_bvp.py',commit='working-tree',
            inputs=sources,parameters=result['parameters'],environment='environment/requirements-lock.txt',
            agent='codex',prompt_refs=['prompts/prompt-log.md','plans/2026-09-05-hito-2.md'],
            decisions=['Generic action varied before radial gauge; E_f excluded from solve',
                'Perturbed static seed and numerical continuation; analytic oracle only in validation',
                result['representation'],result['scope'],
                'Coarse converged candidates retained with individual acceptance; no timing claims'],
            checks=result['checks'],status='verified' if passed else 'rejected',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        errors=validate_record(record)
        if errors:raise ValueError('; '.join(errors))
        records.append(record)
    manifest.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    validate_manifest(manifest)
    print(json.dumps(dict(status='verified' if passed else 'rejected',runs=len(result['runs']),
        checks=result['checks'],outputs=[data.relative_to(ROOT).as_posix(),figure.relative_to(ROOT).as_posix()])))
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
