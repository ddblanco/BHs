"""Generate Hito 3 data, convergence/GR-limit figure and hashed provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
for name, relative in [('TEMP', '.cache/tmp'), ('TMP', '.cache/tmp'), ('MPLCONFIGDIR', '.cache/matplotlib')]:
    directory = ROOT/relative; directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sympy

from rotating_bh.static_egb_benchmark import run_benchmark
from rotating_bh.provenance import validate_manifest, validate_record


def main():
    manifest = ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    records = json.loads(manifest.read_text(encoding='utf-8'))
    result = run_benchmark()
    sources = ['src/rotating_bh/'+name+'.py' for name in
               ('static_egb', '_static_egb_generated', 'static_egb_equations', 'static_egb_bvp',
                'static_egb_validation', 'static_egb_benchmark', 'einstein', 'provenance')]
    sources += ['experiments/derive_static_egb.py', 'experiments/static_egb_bvp.py',
                'environment/requirements-lock.txt', 'docs/static-egb.md', 'docs/convenciones.md']
    result['parameters']['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                              for p in sources}
    result['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'sympy': sympy.__version__,
                           'matplotlib': matplotlib.__version__}
    result['scope'] = ('Static (w=0,h=r^2) EGB, r_h=1, small alpha_hat sweep; sampled coordinate '
                        'norms; no rotation, no extremality, no interior')
    result['representation'] = ('Chebyshev polynomial (spectral, regular at both compact endpoints) '
                                 'and RK45 dense output (adaptive); only F(0)=0 imposed, F(1)~=1 checked')
    passed = all(c['passed'] for c in result['checks'])
    data = ROOT/'results/static-egb.json'; figure = ROOT/'artifacts/static-egb.png'
    data.parent.mkdir(parents=True, exist_ok=True); figure.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    rows = result['resolution_sweep']
    for key, label in [('profile_error', 'Perfil'), ('ode_residual', 'ODE'),
                        ('einstein_residual', 'Einstein+GB')]:
        axes[0].loglog([r['resolution'] for r in rows], [r[key] for r in rows], 'o-', label=label)
    axes[0].axhline(1e-6, color='gray', ls=':', label='Umbral')
    axes[0].set_xlabel('Nodos Chebyshev'); axes[0].set_title(f'Espectral, alpha_hat={result["parameters"]["hardest_alpha_hat"]}')
    axes[0].set_ylabel('Máximo muestreado'); axes[0].grid(alpha=.25); axes[0].legend(fontsize=8)

    rows = result['tolerance_sweep']
    for key, label in [('profile_error', 'Perfil'), ('ode_residual', 'ODE'),
                        ('einstein_residual', 'Einstein+GB')]:
        axes[1].loglog([r['tolerance'] for r in rows], [r[key] for r in rows], 'o-', label=label)
    axes[1].invert_xaxis(); axes[1].axhline(1e-6, color='gray', ls=':', label='Umbral')
    axes[1].set_xlabel('Tolerancia adaptativa'); axes[1].set_title('Adaptativo')
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=8)

    rows = result['gr_limit']
    alphas = [r['alpha_hat'] for r in rows if r['alpha_hat'] > 0]
    diffs = [r['max_difference'] for r in rows if r['alpha_hat'] > 0]
    axes[2].loglog(alphas, diffs, 'o-')
    axes[2].set_xlabel('alpha_hat'); axes[2].set_title('Distancia al límite GR (cerrada)')
    axes[2].grid(alpha=.25)
    fig.suptitle('EGB estático D=5 · r_H=1 · límite GR en alpha_hat->0')
    fig.savefig(figure, dpi=160, metadata={'Software': 'rotating-bh Hito 3'}); plt.close(fig)

    ids = {'static-egb', 'static-egb-convergence'}
    records = [r for r in records if r['id'] not in ids]
    for artifact_id, kind, path in [('static-egb', 'result', data), ('static-egb-convergence', 'figure', figure)]:
        record = dict(id=artifact_id, kind=kind, path=path.relative_to(ROOT).as_posix(),
                       command='.venv/Scripts/python experiments/static_egb_bvp.py', commit='working-tree',
                       inputs=sources, parameters=result['parameters'],
                       environment='environment/requirements-lock.txt', agent='claude',
                       prompt_refs=['prompts/prompt-log.md', 'plans/2026-09-09-hito-3.md'],
                       decisions=['Static field equations derived directly from the tensor (Riemann+GB), not a reduced action',
                                  'b=f and the theta-theta equation follow from tt+rr (Bianchi); both checked symbolically before solving',
                                  'Closed-form eq. (2.20) used only as an oracle in validation, never fed into either solver',
                                  result['representation'], result['scope'],
                                  'GR-limit distance measured against the closed form, not asserted from a fixed power'],
                       checks=result['checks'], status='verified' if passed else 'rejected',
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        errors = validate_record(record)
        if errors: raise ValueError('; '.join(errors))
        records.append(record)
    manifest.write_text(json.dumps(records, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    validate_manifest(manifest)
    print(json.dumps(dict(status='verified' if passed else 'rejected', runs=len(result['runs']),
                           checks=result['checks'],
                           outputs=[data.relative_to(ROOT).as_posix(), figure.relative_to(ROOT).as_posix()])))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
