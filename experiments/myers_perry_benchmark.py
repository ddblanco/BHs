"""Generate the Myers--Perry JSON, figure and provenance within this project."""

import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
# Set these before importing plotting/scientific packages; never use user caches.
for name,relative in (('TEMP','.cache/tmp'),('TMP','.cache/tmp'),
                      ('MPLCONFIGDIR','.cache/matplotlib')):
    directory = ROOT/relative
    directory.mkdir(parents=True,exist_ok=True)
    os.environ[name] = str(directory)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import sympy

from rotating_bh.myers_perry_benchmark import run_benchmark
from rotating_bh.provenance import validate_manifest, validate_record


def main():
    manifest = ROOT/'artifacts/manifest.json'
    validate_manifest(manifest)
    records = json.loads(manifest.read_text(encoding='utf-8'))
    result = run_benchmark()
    sources = ['src/rotating_bh/myers_perry.py','src/rotating_bh/einstein.py',
        'src/rotating_bh/myers_perry_benchmark.py','src/rotating_bh/provenance.py',
        'experiments/myers_perry_benchmark.py','environment/requirements-lock.txt',
        'docs/convenciones.md','docs/myers-perry.md','references/sources.json']
    hashes = {path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    result['parameters']['source_sha256'] = hashes
    result['versions'] = {'python':platform.python_version(),'numpy':np.__version__,
                          'sympy':sympy.__version__,'matplotlib':matplotlib.__version__}
    result['scope'] = 'Analytic GR; sampled coordinate residuals, not global bounds or EGB validation'
    result['residual_definition'] = 'r_h^2 max_ij |G_ij/(s_i s_j)|, s=(1,1,r,r,r)'
    verified = all(check['passed'] for check in result['checks'])
    result_path = ROOT/'results/myers-perry.json'
    figure_path = ROOT/'artifacts/myers-perry.png'
    result_path.parent.mkdir(parents=True,exist_ok=True)
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    fig,axes = plt.subplots(2,2,figsize=(9,6),sharex=True,layout='constrained')
    for ax,key,label in zip(axes.flat,('b','f','h_over_r2','r_h_w'),
                            ('$b$','$f$','$h/r^2$','$r_H w$')):
        for profile in result['profiles']:
            ax.plot(profile['rho'],profile[key],label=f"q = {profile['q']:g}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xlim(1,6)
    for ax in axes[-1]:
        ax.set_xlabel('$r/r_H$')
    axes[0,0].legend()
    fig.suptitle('Myers–Perry D=5 · spins iguales · GR\n'
                 '$q=r_H\\Omega_H$, $r_H=1$, $G_5=1$')
    fig.savefig(figure_path,dpi=160,metadata={'Software':'rotating-bh Hito 1'})
    plt.close(fig)

    python = Path(sys.executable).resolve()
    executable = python.relative_to(ROOT).as_posix() if python.is_relative_to(ROOT) else f'"{python}"'
    command = executable+' experiments/myers_perry_benchmark.py'
    new_records = []
    for artifact_id,kind,path in [('myers-perry-benchmark','result',result_path),
                                  ('myers-perry-profiles','figure',figure_path)]:
        record = {'id':artifact_id,'kind':kind,'path':path.relative_to(ROOT).as_posix(),
            'command':command,'commit':'working-tree','inputs':sources,
            'parameters':result['parameters'],'environment':'environment/requirements-lock.txt',
            'agent':'codex','prompt_refs':['prompts/prompt-log.md','plans/2026-09-05-hito-1.md'],
            'decisions':['NumPy radial formulas; SymPy metric derivatives; NumPy full tensor contractions',
                'Separate mu,a symbolic metric checked against r_h,Omega_H radial implementation',
                'All 25 Einstein components sampled away from chart singularities',
                result['residual_definition'],result['scope'],
                'No randomness; G5=1 declared; fixed output paths inside project'],
            'checks':result['checks'],'status':'verified' if verified else 'rejected',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        errors = validate_record(record)
        if errors:
            raise ValueError('; '.join(errors))
        new_records.append(record)
    ids = {record['id'] for record in new_records}
    records = [record for record in records if record['id'] not in ids]+new_records
    manifest.write_text(json.dumps(records,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    validate_manifest(manifest)
    print(json.dumps({'status':'verified' if verified else 'rejected',
        'max_einstein_residual':max(row['einstein_residual'] for row in result['runs']),
        'negative_control_residual':result['negative_control_residual'],
        'outputs':['results/myers-perry.json','artifacts/myers-perry.png']}))
    return 0 if verified else 1


if __name__ == '__main__':
    raise SystemExit(main())
