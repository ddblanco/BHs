"""The extremal curves of fig. 1 (left panels) of arXiv:2303.12471, from the
vector paths in digitised.json.

Each *-MP2.eps file has three plot calibrations: the main panel (x = c0
alpha/M horizontal), the inset (j horizontal), and a repeat of the inset.
Paths drawn in plot 0 after its second filled polygon belong to the inset
(they precede the inset's own tick labels) and are discarded; the inset is
taken from plot 1. Black solid strokes (LT0) are the computed extremal set,
black dotted (LT5) the authors' hand extrapolation to (x, j) = (1, 0).

Output: kkr_curves.json with, per quantity (aH, s, tH): main = [[x, q], ...],
inset = [[j, q], ...], dotted_main, dotted_inset; plus the static (blue) and
Myers-Perry (red) curves for validation.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = {'aH': 'xjaH-MP2', 's': 'xjs-MP2', 'tH': 'xjtH-MP2'}
BLACK, BLUE, RED = [0., 0., 0.], [0., 0., 1.], [1., 0., 0.]


def split(paths):
    main, inset = [], []
    fills = 0
    for p in paths:
        if p['plot'] == 0:
            if p['kind'] == 'fill':
                fills += 1
            if fills <= 1:
                main.append(p)
        elif p['plot'] == 1:
            inset.append(p)
    return main, inset


def collect(paths, colour, linetype='LT0'):
    pts = [q for p in paths if p['kind'] == 'stroke' and p['colour'] == colour
           and p['linetype'] == linetype for q in p['points']]
    return sorted(set(map(tuple, pts)))


def main():
    d = json.loads((HERE/'digitised.json').read_text())
    out = {}
    for key, name in FILES.items():
        main_p, inset_p = split(d[name]['paths'])
        out[key] = dict(
            main=collect(main_p, BLACK), inset=collect(inset_p, BLACK),
            dotted_main=collect(main_p, BLACK, 'LT5'), dotted_inset=collect(inset_p, BLACK, 'LT5'),
            static_main=collect(main_p, BLUE), mp_inset=collect(inset_p, RED))
        m = np.array(out[key]['main'])
        i = np.array(out[key]['inset']) if out[key]['inset'] else np.zeros((0, 2))
        print(f"{key}: main extremal {len(m)} pts, x in [{m[:,0].min():.4f}, {m[:,0].max():.4f}]; "
              f"inset extremal {len(i)} pts" + (f", j in [{i[:,0].min():.4f}, {i[:,0].max():.4f}]" if len(i) else ''))
    # validation of the inset Myers-Perry curve: a_H = s = (1 + sqrt(1-j^2))/2
    for key in ('aH', 's'):
        mp = np.array(out[key]['mp_inset'])
        dev = np.max(np.abs(mp[:, 1]-(1+np.sqrt(np.clip(1-mp[:, 0]**2, 0, None)))/2))
        print(f'{key}: inset Myers-Perry curve vs closed form, max deviation {dev:.1e}')
    (HERE/'kkr_curves.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
