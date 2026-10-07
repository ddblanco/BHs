"""Extract the vector curves of the gnuplot EPS figures of arXiv:2303.12471.

The figures are gnuplot 4.4 EPS: every curve is a path of absolute (M, L, N)
and relative (V, R) moves in device units, and every axis tick is a short
LTb stroke followed by its label. Each plot of a multiplot (main panel and
inset) is calibrated separately from its own ticks, and every path drawn after
those ticks is mapped to data coordinates with that calibration.

Nothing is read off by eye: the output is the path vertices exactly as
gnuplot wrote them, which are the plotted data up to gnuplot's integer
rounding of device coordinates (1 unit = 1/20 pt).

Run: python eps_digitise.py  ->  digitised.json
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Only the equal-spin panels (left column of fig. 1, files *-MP2); *-MP and *-BR are
# the single-plane solutions, outside this reproduction.
SOURCES = sorted((HERE/'arxiv').glob('*-MP2.eps'))


def tokens(text):
    body = text.split('%%EndProlog', 1)[1]
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        yield line


def parse_ticks(path):
    """Second pass, simpler and explicit: pair every label with its position."""
    text = path.read_text(errors='replace').split('%%EndProlog', 1)[1]
    pattern = re.compile(
        r'(-?\d+) (-?\d+) M\s*\n\[ \[\([^)]*\) [\d.]+ [\d.-]+ \w+ \w+ \d+ '
        r'\(\s*([-+]?[0-9.]+(?:e[-+]?\d+)?)\s*\)\]\s*\n\] [-\d.]+ (MRshow|MCshow)')
    out = []
    for m in pattern.finditer(text):
        out.append((float(m.group(1)), float(m.group(2)), float(m.group(3)),
                    'y' if m.group(4) == 'MRshow' else 'x', m.start()))
    return out


def calibrations(path):
    """Split the label sequence into plots: each plot is a run of y labels
    followed by a run of x labels. Returns [(start, xfit, yfit)]."""
    ticks = parse_ticks(path)
    groups, current = [], None
    for px, py, value, axis, position in ticks:
        if current is None or (axis == 'y' and current['x']):
            current = dict(y=[], x=[], start=position)
            groups.append(current)
        current[axis].append((px if axis == 'x' else py, value))
    out = []
    for g in groups:
        fits = []
        for axis in ('x', 'y'):
            pairs = g[axis]
            (d0, v0), (d1, v1) = pairs[0], pairs[-1]
            scale = (v1-v0)/(d1-d0)
            # every intermediate tick must lie on the same line: this is the
            # check that the calibration is linear and correctly paired
            for d, v in pairs:
                # device coordinates are integers: allow one unit of rounding
                assert abs(v0+scale*(d-d0)-v) <= 1.5*abs(scale), (path.name, axis, d, v)
            fits.append((d0, v0, scale))
        out.append(dict(start=g['start'], x=fits[0], y=fits[1],
                        xticks=g['x'], yticks=g['y']))
    return out


def paths_by_plot(path):
    """All coloured paths, assigned to the plot whose labels precede them."""
    text = path.read_text(errors='replace').split('%%EndProlog', 1)[1]
    cals = calibrations(path)
    starts = [c['start'] for c in cals]
    # tokenise into (position, word)
    words = [(m.start(), m.group()) for m in re.finditer(r'\S+', text)]
    x = y = 0.
    colour, linetype = (0., 0., 0.), 'LTb'
    current, collected = [], []
    in_label = False

    def plot_index(position):
        index = 0
        for k, s in enumerate(starts):
            if position >= s:
                index = k
        return index

    def flush(position, kind='stroke'):
        nonlocal current
        if len(current) > 1 and linetype not in ('LTb',):
            collected.append(dict(plot=plot_index(position), colour=colour,
                                  linetype=linetype, kind=kind, points=current))
        current = []

    stack = []
    for position, w in words:
        if w.startswith('[') or in_label:
            in_label = True
            if w.endswith('show'):
                in_label = False
                stack = []
            continue
        if w in ('M', 'N'):
            flush(position)
            x, y = float(stack[-2]), float(stack[-1])
            current = [(x, y)]
        elif w == 'L':
            x, y = float(stack[-2]), float(stack[-1])
            current.append((x, y))
        elif w == 'V':
            x, y = x+float(stack[-2]), y+float(stack[-1])
            current.append((x, y))
        elif w == 'R':
            flush(position)
            x, y = x+float(stack[-2]), y+float(stack[-1])
            current = [(x, y)]
        elif w == 'C':
            colour = tuple(float(v) for v in stack[-3:])
        elif w.startswith('LT') and len(w) <= 4:
            flush(position)
            linetype = w
        elif w == 'stroke':
            flush(position)
        elif w == 'PolyFill':
            flush(position, 'fill')
        if re.fullmatch(r'[-+]?\d*\.?\d+(e[-+]?\d+)?', w):
            stack.append(w)
        else:
            stack = []
    flush(len(text))
    data = []
    for c in collected:
        cal = cals[c['plot']]
        (dx0, vx0, sx), (dy0, vy0, sy) = cal['x'], cal['y']
        pts = [(vx0+sx*(px-dx0), vy0+sy*(py-dy0)) for px, py in c['points']]
        data.append(dict(plot=c['plot'], colour=c['colour'], linetype=c['linetype'],
                         kind=c['kind'], points=pts))
    return cals, data


def main():
    result = {}
    for source in SOURCES:
        cals, data = paths_by_plot(source)
        result[source.stem] = dict(
            calibrations=[dict(x=c['x'], y=c['y'], xticks=c['xticks'],
                               yticks=c['yticks']) for c in cals],
            paths=data)
        print(f'{source.name}: {len(cals)} plots')
        for c in cals:
            print('   x ticks', [v for _, v in c['xticks']], ' y ticks', [v for _, v in c['yticks']])
        for d in data:
            xs = [p[0] for p in d['points']]
            ys = [p[1] for p in d['points']]
            print(f"   plot {d['plot']} {d['linetype']:4s} {d['kind']:6s} colour {d['colour']} "
                  f"n={len(xs):4d} x[{min(xs):.3f},{max(xs):.3f}] y[{min(ys):.3f},{max(ys):.3f}]")
    (HERE/'digitised.json').write_text(json.dumps(result))


if __name__ == '__main__':
    main()
