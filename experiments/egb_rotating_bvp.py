"""Generate Hito 4B data and provenance (first accepted rotating EGB point).

Scope: see src/rotating_bh/egb_rotating_benchmark.py's module docstring --
one omega_h, a short alpha_GB continuation, a two-resolution check. Not the
Hito 4C family/comparison sweep.
"""
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

from rotating_bh.egb_rotating_benchmark import run_benchmark
from rotating_bh.provenance import validate_manifest, validate_record


def main():
    manifest = ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    records = json.loads(manifest.read_text(encoding='utf-8'))
    result = run_benchmark()
    sources = ['src/rotating_bh/'+name+'.py' for name in
               ('egb_rotating_bvp', 'egb_rotating_validation', 'egb_rotating_benchmark',
                '_egb_rotating_generated', '_egb_rotating_compact_generated',
                '_egb_rotating_horizon_generated', 'myers_perry', 'einstein', 'provenance')]
    sources += ['experiments/egb_rotating_bvp.py', 'experiments/derive_egb_rotating.py',
                'environment/requirements-lock.txt', 'docs/egb-rotating.md', 'docs/convenciones.md']
    result['parameters']['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                              for p in sources}
    result['versions'] = {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'sympy': sympy.__version__,
                           'matplotlib': matplotlib.__version__}
    result['scope'] = ('First accepted rotating EGB points (Hito 4B), one omega_h, short '
                        'alpha_GB continuation; not the Hito 4C family/comparison sweep')
    passed = all(c['passed'] for c in result['checks'])
    data = ROOT/'results/egb-rotating.json'; figure = ROOT/'artifacts/egb-rotating.png'
    data.parent.mkdir(parents=True, exist_ok=True); figure.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    runs = result['runs']
    alphas = [r['alpha_gb'] for r in runs]
    axes[0].semilogy(alphas, [r['max_tensor_residual'] for r in runs], 'o-', label='max')
    axes[0].semilogy(alphas, [r['mean_tensor_residual'] for r in runs], 's--', label='mean')
    axes[0].axhline(1e-6, color='gray', ls=':', label='umbral')
    axes[0].set_xlabel('alpha_GB'); axes[0].set_ylabel('Residuo tensorial')
    axes[0].set_title(f'omega_H={result["parameters"]["omega_h"]}'); axes[0].grid(alpha=.25)
    axes[0].legend(fontsize=8)

    rows = result['resolution_sweep']
    axes[1].semilogy([r['resolution'] for r in rows], [r['max_tensor_residual'] for r in rows], 'o-')
    axes[1].axhline(1e-6, color='gray', ls=':')
    axes[1].set_xlabel('Nodos Chebyshev'); axes[1].set_title(f'alpha_GB={alphas[-1]} (más difícil)')
    axes[1].grid(alpha=.25)
    fig.suptitle('EGB rotante D=5 · primer punto aceptado (Hito 4B)')
    fig.savefig(figure, dpi=160, metadata={'Software': 'rotating-bh Hito 4B'}); plt.close(fig)

    ids = {'egb-rotating', 'egb-rotating-convergence'}
    records = [r for r in records if r['id'] not in ids]
    for artifact_id, kind, path in [('egb-rotating', 'result', data), ('egb-rotating-convergence', 'figure', figure)]:
        record = dict(id=artifact_id, kind=kind, path=path.relative_to(ROOT).as_posix(),
                       command='.venv/Scripts/python experiments/egb_rotating_bvp.py', commit='working-tree',
                       inputs=sources, parameters=result['parameters'],
                       environment='environment/requirements-lock.txt', agent='claude',
                       prompt_refs=['prompts/prompt-log.md', 'docs/egb-rotating.md'],
                       decisions=['Spectral solver only (Hito 4B first pass); horizon rows from '
                                  'compact_horizon_conditions, infinity rows B=1,F=1,Hx=Wx=0 as in '
                                  'vacuum_bvp.py; W(horizon)=omega_h replaces the raw w-equation row',
                                  'Accepted via the independent tensor evaluator (G+alpha_GB H), not '
                                  'the action-derived equations used to derive/solve the system',
                                  result['scope']],
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
