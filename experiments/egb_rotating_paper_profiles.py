"""Independent quantitative extraction of the red curves of paper Figure 1b.

Protocol fixed before comparing: five printed x ticks correspond to
log10(r)=0,.25,.5,.75,1; five y ticks to 1,.5,0,-.5,-1.
Detect their pixel centers on the frame and fit affine calibrations.
Compare columns 153,200,295; uncertainty box +/-1 x pixel, +/-2 y pixels
covers curve thickness/antialiasing and tick reading. Not an author error bar.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from matplotlib.image import imread


def extract(path):
    rgba = imread(path)
    def groups(indices):
        return np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    yt = [float(np.mean(g)) for g in groups(np.flatnonzero(np.max(rgba[:,51,:3],axis=1)<.1))][1:-1]
    xt = [float(np.mean(g)) for g in groups(np.flatnonzero(np.max(rgba[17,:,:3],axis=1)<.1))][1:-1]
    if len(xt)!=5 or len(yt)!=5:
        raise ValueError('Could not identify the five printed ticks on each axis')
    sx,ox = np.polyfit([0,.25,.5,.75,1],xt,1)
    sy,oy = np.polyfit([1,.5,0,-.5,-1],yt,1)
    rows = []
    for column in [153,200,295]:
        pixels = rgba[:,column,:3]
        red = np.flatnonzero((pixels[:,0]>.8)&(pixels[:,1]<.3)&(pixels[:,2]<.3))
        red_groups = groups(red)
        fields = ['h_over_r2','f','w','minus_b']
        if column==153 and len(red_groups)==3:
            # f and w cross in this column: discard the blended group,
            # retaining only unambiguous h/r² and -b readings.
            red_groups,fields = [red_groups[0],red_groups[-1]],['h_over_r2','minus_b']
        elif len(red_groups)!=4:
            raise ValueError(f'Expected four isolated red curves at x={column}; found {len(red_groups)}')
        for field,group in zip(fields,red_groups):
            y = float(np.mean(group))
            rows.append(dict(column=column,row=y,field=field,log10_r=float((column-ox)/sx),
                             value=float((y-oy)/sy),pixel_thickness=len(group),
                             x_pixels_per_unit=float(sx),y_pixels_per_unit=float(abs(sy)),
                             x_tick_centers=xt,y_tick_centers=yt))
    return rows


def run():
    source = ROOT/'references/data/1010.0860v1-figure-1b.png'
    family = ROOT/'results/egb-rotating-family.json'
    report = json.loads(family.read_text())
    profile = next(p for p in report['profiles'] if p['alpha_gb']==.75)
    x = np.log10(profile['r'])
    rows = extract(source)
    for row in rows:
        field = row['field']
        y = -np.array(profile['b']) if field=='minus_b' else np.array(profile[field])
        position = row['log10_r']
        value = float(np.interp(position,x,y))
        position_effect = max(abs(float(np.interp(position+offset/row['x_pixels_per_unit'],x,y))-value) for offset in [-1,1])
        row.update(computed=value,absolute_difference=abs(value-row['value']),
                   tolerance=2/row['y_pixels_per_unit']+position_effect)
        row['agrees'] = row['absolute_difference']<=row['tolerance']
    files = [source,family,Path(__file__)]
    data = dict(source_url='https://arxiv.org/html/1010.0860v1/profiles-alpha.png',
        alpha_paper=3.,alpha_gb=.75,q=.33,r_h=1.,
        method='Red pixel groups; manually calibrated printed ticks; +/-1 horizontal and +/-2 vertical pixels',
        status='additional comparison; does not replace failed ergosurface test',rows=rows,
        source_sha256={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output = ROOT/'results/egb-rotating-paper-profiles.json'
    output.write_text(json.dumps(data,indent=2)+'\n')
    manifest_path = ROOT/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    record = dict(id='egb-rotating-paper-profiles',kind='result',path=str(output.relative_to(ROOT)).replace('\\','/'),
        command='python experiments/egb_rotating_paper_profiles.py',commit='working-tree',
        inputs=list(data['source_sha256']),parameters=dict(source_sha256=data['source_sha256'],
        horizontal_pixels=1,vertical_pixels=2,columns=[153,200,295]),
        environment='environment/requirements-lock.txt',agent='codex',prompt_refs=['prompts/prompt-log.md'],
        decisions=['Affine calibration from five printed ticks per axis',
                   'Discard merged f/w crossing at column153; no enlargement of pixel uncertainty',
                   'Additional comparison does not erase the failed radius comparison'],
        checks=[dict(name='ten_separable_curve_values_agree',passed=bool(len(rows)==10 and all(r['agrees'] for r in rows)))],
        status='verified' if len(rows)==10 and all(r['agrees'] for r in rows) else 'rejected',
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    manifest = [r for r in manifest if r['id']!=record['id']]+[record]
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    run()
