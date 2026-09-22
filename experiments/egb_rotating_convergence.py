"""Generate a new resolution/tolerance artifact without changing old results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
for key, relative in [('TEMP', '.cache/tmp'), ('TMP', '.cache/tmp'),
                       ('MPLCONFIGDIR', '.cache/matplotlib')]:
    directory = ROOT/relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sympy

from rotating_bh.egb_rotating_convergence import run_study, closure_checks, TOLERANCES
from rotating_bh.provenance import validate_manifest, validate_record

DATA = 'results/egb-rotating-resolution.json'
FIGURE = 'artifacts/egb-rotating-resolution.png'
IDS = ('egb-rotating-resolution', 'egb-rotating-resolution-figure')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    """Verify stored acceptance and file/source hashes without regenerating equations."""
    manifest = root/'artifacts/manifest.json'
    validate_manifest(manifest)
    result = json.loads((root/DATA).read_text(encoding='utf-8'))
    computed = closure_checks(result)
    if result['checks'] != computed or not all(check['passed'] for check in computed):
        raise ValueError('resolution study does not pass recomputed closure gates')
    for name, digest in result['source_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError(f'changed numerical input: {name}')
    records = json.loads(manifest.read_text(encoding='utf-8'))
    cross_path = 'results/egb-rotating-cross-check.json'
    cross = json.loads((root/cross_path).read_text(encoding='utf-8'))
    matches = [r for r in records if r['id'] == 'egb-rotating-cross-check']
    if (len(matches) != 1 or matches[0]['status'] != 'verified'
            or matches[0]['path'] != cross_path or matches[0]['sha256'] != sha(root/cross_path)
            or matches[0]['checks'] != cross['checks']
            or not cross['checks'] or not all(check['passed'] for check in cross['checks'])
            or cross['parameters']['omega_h'] != .3 or cross['parameters']['alpha_gb'] != .1):
        raise ValueError('independent cross-check artifact is not verified')
    for name, digest in cross['source_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError(f'changed cross-check numerical input: {name}')
    for identifier, path in zip(IDS, (DATA, FIGURE)):
        matches = [r for r in records if r['id'] == identifier]
        if (len(matches) != 1 or matches[0]['status'] != 'verified'
                or matches[0]['checks'] != computed or matches[0]['path'] != path
                or matches[0]['sha256'] != sha(root/path)):
            raise ValueError(f'invalid resolution artifact: {identifier}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    arguments = parser.parse_args()
    if arguments.verify_only:
        verify()
        print('HITO_4B_RESOLUTION_VERIFIED')
        return 0
    manifest = ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    result = run_study(progress=lambda row: print(json.dumps(row), flush=True))
    sources = ['src/rotating_bh/'+name+'.py' for name in
               ('egb_rotating_convergence', 'egb_rotating_bvp', 'egb_rotating_adaptive',
                'egb_rotating_validation', '_egb_rotating_compact_generated',
                '_egb_rotating_horizon_generated', 'einstein', 'myers_perry', 'provenance')]
    sources += ['experiments/egb_rotating_convergence.py', 'environment/requirements-lock.txt']
    result['source_sha256'] = {p: sha(ROOT/p) for p in sources}
    result['versions'] = dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, sympy=sympy.__version__,
                              matplotlib=matplotlib.__version__)
    (ROOT/DATA).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for tolerance in TOLERANCES:
        rows = sorted([r for r in result['runs'] if r['tolerance'] == tolerance and r['converged']],
                      key=lambda r: r['resolution'])
        for ax, field in zip(axes, ('max_tensor_residual', 'adaptive_difference')):
            ax.semilogy([r['resolution'] for r in rows], [r[field] for r in rows],
                        'o-', label=f'tol={tolerance:g}')
    rejected = [r for r in result['runs'] if r['converged'] and not r['accepted']]
    for ax, field in zip(axes, ('max_tensor_residual', 'adaptive_difference')):
        ax.scatter([r['resolution'] for r in rejected], [r[field] for r in rejected],
                   marker='x', color='crimson', s=65, zorder=4, label='no aceptado')
    for ax, threshold, ylabel in zip(axes, (1e-6, 1e-5),
                                     ('Residuo tensorial máximo', 'Máximo |delta(B,F,H,W)|')):
        ax.axhline(threshold, color='gray', linestyle=':', label='umbral físico')
        ax.set_xlabel('Nodos espectrales')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    failures = sum(not r['converged'] for r in result['runs'])
    count = sum(r['accepted'] for r in result['runs'])
    fig.suptitle(f'EGB rotante: omega_H=0.3, alpha_GB=0.1 · {count}/15 aceptados · {failures} sin converger')
    fig.savefig(ROOT/FIGURE, dpi=160, metadata={'Software': 'rotating-bh Hito 4B convergence'})
    plt.close(fig)

    passed = all(check['passed'] for check in result['checks'])
    records = json.loads(manifest.read_text(encoding='utf-8'))
    records = [r for r in records if r['id'] not in IDS]
    for identifier, path, kind in zip(IDS, (DATA, FIGURE), ('result', 'figure')):
        record = dict(id=identifier, kind=kind, path=path,
                       command='python experiments/egb_rotating_convergence.py (PYTHONPATH=src)',
                       commit='working-tree', inputs=sources, parameters=result['parameters'],
                       environment='environment/requirements-lock.txt', agent='codex',
                       prompt_refs=['prompts/prompt-log.md', 'plans/2026-09-10-hito-4b-closure.md'],
                       decisions=['Same 1% perturbed coarse seed for every resolution/tolerance cell',
                                  'Tensor on represented derivatives and independent adaptive profile',
                                  'Preserve loose/coarse rejections and strict-tolerance failures; '
                                  'closure requires the specified fine nominal and strict cells'],
                       checks=result['checks'], status='verified' if passed else 'rejected',
                       sha256=sha(ROOT/path))
        errors = validate_record(record)
        if errors:
            raise ValueError('; '.join(errors))
        records.append(record)
    manifest.write_text(json.dumps(records, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    validate_manifest(manifest)
    if passed:
        verify()
    print(json.dumps(dict(status='verified' if passed else 'rejected', checks=result['checks'])))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
