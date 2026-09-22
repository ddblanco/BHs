"""Plot/register or verify Hito 4C evidence, including rejected comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from rotating_bh.egb_rotating_family_validation import checks
from rotating_bh.provenance import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = 'results/egb-rotating-family.json'
FIGURE = 'artifacts/egb-rotating-family.png'
FAILED = 'results/egb-rotating-family-failed-continuation.json'
FAILED_SOURCE = 'reports/hito-4c-failed-continuation-source.txt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    data = json.loads((root/DATA).read_text())
    if checks(data) != data['checks']:
        raise ValueError('stored scientific gates differ from evidence')
    for name,expected in data['sources'].items():
        if digest(root/name) != expected:
            raise ValueError(f'source changed: {name}')
    validate_manifest(root/'artifacts/manifest.json')
    manifest = json.loads((root/'artifacts/manifest.json').read_text())
    for name in [DATA,FIGURE]:
        records = [r for r in manifest if r['path']==name]
        if len(records)!=1 or records[0]['sha256']!=digest(root/name):
            raise ValueError(f'artifact hash mismatch: {name}')
        expected_checks = [dict(name=k,passed=bool(v)) for k,v in data['checks'].items()]
        expected_status = 'verified' if all(data['checks'].values()) else 'candidate'
        if records[0]['checks']!=expected_checks or records[0]['status']!=expected_status:
            raise ValueError(f'manifest scientific status differs from evidence: {name}')
        for source,expected in records[0]['parameters']['source_sha256'].items():
            if digest(root/source)!=expected:
                raise ValueError(f'artifact source changed: {source}')
    failures = json.loads((root/FAILED).read_text())
    if digest(root/FAILED_SOURCE)!=failures['source_sha256']:
        raise ValueError('failed attempt source snapshot changed')
    return data


def generate():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads((ROOT/DATA).read_text())
    fig,axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for a in [.05,.1]:
        rows = [r for r in data['records'] if r['label']=='resolution' and r['alpha_gb']==a]
        axes[0,0].plot([r['q'] for r in rows],[r['observables']['E'] for r in rows],'o-',label=f'alpha_GB={a}')
    axes[0,0].set(xlabel='Omega_H (r_H=1)',ylabel='E (G5=1)',title='Validated family, N=32')
    axes[0,0].legend()
    law = data['first_law']
    h = np.array([r['step'] for r in law])
    err = np.array([abs(r['residual']) for r in law])
    axes[0,1].loglog(h,err,'o-',label='First-law residual')
    axes[0,1].loglog(h,err[0]*(h/h[0])**2,'--',label='Second order')
    axes[0,1].set(xlabel='Central difference step',ylabel='Absolute residual',title='alpha_GB=0.1, Omega_H=0.3')
    axes[0,1].legend()
    for p,style in zip(data['profiles'],['--','-']):
        for k,color,sign in [('b','C0',-1),('f','C1',1),('h_over_r2','C2',1),('w','C3',1)]:
            axes[1,0].plot(np.log10(p['r']),sign*np.array(p[k]),style,color=color,
                label=f'{"-b" if k=="b" else k}, alpha={4*p["alpha_gb"]:g}')
    profile_title = ('Figure 1b parameters' if data['checks']['figure_parameters']
                     else 'Available profiles; full Figure 1b target pair not reached')
    axes[1,0].set(xlabel='log10(r)',ylabel='Metric functions',title=profile_title)
    axes[1,0].legend(fontsize=7,ncol=2)
    e = data['external']
    axes[1,1].errorbar([r['alpha_paper'] for r in e],[r['published_radius'] for r in e],
        yerr=[r['rounding_half_unit'] for r in e],fmt='s',capsize=5,label='Paper text; adopted rounding tolerance')
    axes[1,1].plot([r['alpha_paper'] for r in e],[r['computed_radius'] for r in e],'o',label='Computed')
    axes[1,1].set(xlabel='alpha_paper',ylabel='Ergosurface radius',title='Omega_H=0.33: discrepancy retained')
    axes[1,1].legend(fontsize=8)
    fig.savefig(ROOT/FIGURE,dpi=160,metadata={'Software':'rotating_bh Hito 4C'})
    plt.close(fig)
    manifest = json.loads((ROOT/'artifacts/manifest.json').read_text())
    sources = dict(data['sources'])
    sources['experiments/egb_rotating_family_artifacts.py'] = digest(Path(__file__))
    sources['src/rotating_bh/provenance.py'] = digest(ROOT/'src/rotating_bh/provenance.py')
    for name in [FAILED,FAILED_SOURCE,'references/notes/1010.0860-hito-4c.md']:
        sources[name] = digest(ROOT/name)
    for name,kind in [(DATA,'result'),(FIGURE,'figure')]:
        record = dict(id='egb-rotating-family'+('-figure' if kind=='figure' else ''),kind=kind,path=name,
            command='python experiments/egb_rotating_family.py; python experiments/egb_rotating_family_artifacts.py',
            commit='working-tree',inputs=list(sources),parameters=dict(source_sha256=sources),
            environment='environment/requirements-lock.txt',agent='codex',
            prompt_refs=['prompts/prompt-log.md','plans/2026-09-10-hito-4c.md'],
            decisions=['Preserve failed external comparison; no automatic milestone closure',
                       'Paper error bars unavailable; displayed tolerance adopted from printed precision'],
            checks=[dict(name=k,passed=bool(v)) for k,v in data['checks'].items()],
            status='verified' if all(data['checks'].values()) else 'candidate',sha256=digest(ROOT/name))
        manifest = [r for r in manifest if r['id']!=record['id']]+[record]
    failure_record = dict(id='egb-rotating-family-failed-continuation',kind='result',path=FAILED,
        command='Archived experiment source in reports/hito-4c-failed-continuation-source.txt',
        commit='working-tree',inputs=[FAILED_SOURCE],parameters=dict(q=.33,alpha_gb=.6,
        previous_alpha_gb=.5,N=32,tolerance=1e-11,max_iterations=50),
        environment='environment/requirements-lock.txt',agent='codex',
        prompt_refs=['prompts/prompt-log.md'],decisions=['Preserve failed Newton continuation; no physical boundary claim'],
        checks=[dict(name='newton_converged',passed=False)],status='rejected',sha256=digest(ROOT/FAILED))
    manifest = [r for r in manifest if r['id']!=failure_record['id']]+[failure_record]
    (ROOT/'artifacts/manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    verify()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify()
        print('HITO_4C_ARTIFACT_INTEGRITY_OK')
        print('Scientific closure:',all(result['checks'].values()))
    else:
        generate()
