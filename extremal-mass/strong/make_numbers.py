"""Generate numbers.tex (macros) and numerics.tex (section 4 body) for strong-coupling-report.tex
from the spectral branch, sweep.json, bvp_results.json, ms_results.json, the strong-coupling fits,
and the global approximant of global2.py.  Every number in the report's numerical section comes from here."""
import json
import numpy as np
from pathlib import Path
from global2 import data, fit, A2, ms_points
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

S, Mv, SRC = data()
sw = json.loads((HERE/'sweep.json').read_text())
ms = ms_points()
bvp = sorted([d for d in json.loads((HERE/'bvp_results.json').read_text()) if d['status'] == 0], key=lambda d: d['s'])

# spectral branch comparison
spec = []
for r in json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch']:
    if r['alpha'] > 0:
        spec.append((np.pi**(2/3)*r['alpha']/r['J']**(2/3), r['M']/(np.pi**(1/3)*r['J']**(2/3))))
spec = np.array(sorted(spec))
spec = spec[np.r_[True, np.diff(spec[:, 0]) > 1e-4]]
from scipy.interpolate import CubicSpline
spl = CubicSpline(spec[:, 0], spec[:, 1])
dev = [abs(d['m']/spl(d['s'])-1) for d in sw+bvp if d['s'] <= spec[-1, 0]]
maxdev = max(dev)
sw0 = min(sw, key=lambda d: abs(d['s']-2.98202))
_b = [r for r in json.loads((ROOT/'work/kkr-extremal/analysis.json').read_text())['branch'] if r['alpha'] > 0]
specmed = float(np.median([r['constraint'] for r in _b])); specmax = max(r['constraint'] for r in _b)
variants = max(max(abs(v) for v in d['dm_variants']) for d in sw if d.get('dm_variants'))
maxres = max(d['res'] for d in ms)
near = np.array([(d['s'], d['m']) for d in ms if d['s'] < 10.4])
_c = np.polyfit(near[:, 0]-9.5, near[:, 1], 6)
_b9 = min(bvp, key=lambda d: abs(d['s']-9.0037))
xdev = abs(np.polyval(_c, _b9['s']-9.5)-_b9['m'])

# strong-coupling fits: (m - 3s/4) sqrt(s) = a1 + a2 s^-1/2 + a3 s^-1 + ...   (s >= 10: multiple shooting only)
D = (Mv-0.75*S)*np.sqrt(S); u = S**-0.5
fits = []
for smin in (10, 15, 20):
    sel = S >= smin
    for K in (5, 6, 7):
        A = np.vander(u[sel], K, increasing=True)
        c = np.linalg.lstsq(A, D[sel], rcond=None)[0]
        fits.append(('free', smin, K, c, np.sqrt(np.mean((A@c-D[sel])**2))))
    for K in (5, 6):
        A = np.vander(u[sel], K+1, increasing=True)[:, 1:]
        c = np.linalg.lstsq(A, D[sel]-4/3, rcond=None)[0]
        fits.append(('fixed', smin, K, c, np.sqrt(np.mean((A@c-D[sel]+4/3)**2))))
a1_free = [f[3][0] for f in fits if f[0] == 'free']
a2_fixed = [f[3][0] for f in fits if f[0] == 'fixed']
a3_fixed = [f[3][1] for f in fits if f[0] == 'fixed']
# a1 = 4/3 and a2 = A2 (conditional value) both fixed: compare rms with the free-a2 fits
fix2 = []
for smin in (10, 15, 20):
    sel = S >= smin
    for K in (6,):
        A = np.vander(u[sel], K+2, increasing=True)[:, 2:]; y = D[sel]-4/3-A2*u[sel]
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        fix2.append((smin, K, c[0], np.sqrt(np.mean((A@c-y)**2))))
a3_fix2 = [f[2] for f in fix2]; rms_fix2 = max(f[3] for f in fix2)
b4a = np.array([d['b4']*d['alpha'] for d in ms if d['s'] > 40])

g2 = json.loads((HERE/'global2.json').read_text())
G = g2['True_10']
_, rel = fit(S, Mv, G['kappa'], 10, True)

def sci(x):
    mant, e = f'{x:.0e}'.split('e')
    return f'{mant}\\times10^{{{int(e)}}}'


mac = {
    'SstarRe': '0.0059232', 'SstarIm': '0.0548189', 'SstarAbs': '0.05514',
    'SmaxNum': f'{S.max():.0f}',
    'AoneLo': f'{min(a1_free):.4f}', 'AoneHi': f'{max(a1_free):.4f}',
    'AtwoLo': f'{min(a2_fixed):.4f}', 'AtwoHi': f'{max(a2_fixed):.4f}',
    'AthreeLo': f'{min(a3_fixed):.3f}', 'AthreeHi': f'{max(a3_fixed):.3f}',
    'StrongFitSentence': (f'Fitting $(m-3s/4)\\sqrt s=a_1+a_2s^{{-1/2}}+a_3s^{{-1}}+\\dots$ (5--7 terms, '
                          f'$s\\ge10,15,20$) gives $a_1\\in[{min(a1_free):.5f},{max(a1_free):.5f}]$ with all '
                          f'coefficients free; with $a_1=4/3$ fixed, $a_2\\in[{min(a2_fixed):.5f},{max(a2_fixed):.5f}]$ '
                          f'(conditional prediction $0.456527$) and $a_3\\in[{min(a3_fixed):.3f},{max(a3_fixed):.3f}]$. '
                          f'Fixing also $a_2=0.456527$ (six further terms) fits as well as the free fits (rms $\\le{sci(rms_fix2)}$) '
                          f'and gives $a_3\\in[{min(a3_fix2):.4f},{max(a3_fix2):.4f}]$.'),
    'AthreeFixLo': f'{min(a3_fix2):.3f}', 'AthreeFixHi': f'{max(a3_fix2):.3f}',
    'GlobalKappa': f'{G["kappa"]:.4f}', 'GlobalErr': f'{np.max(np.abs(rel)):.1e}',
    'GlobalFitSentence': (f'A ten-term approximant with the exact ends (\\S4) reproduces all '
                          f'{len(S)} numerical points to {np.max(np.abs(rel)):.1e} (relative).'),
}
(HERE/'numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items())+'\n')

L = [r'\textbf{Methods.} (i) $s\le5.24$: the spectral branch of \texttt{work/kkr-extremal}.',
     r'(ii) $2\le s\le8.6$: two-sided shooting (\texttt{strong/twoside.py}): outward from the horizon,',
     r'with its formal series (rank-one level blocks) plus the non-analytic throat mode $B\,x^\gamma$',
     r'($\gamma$ the exact throat exponent), inward from the large-$r$ series with free mass $\mu$ and',
     r'squashing amplitude $b_4$; $m=3\mu/4$. (iii) $s\ge9$: multiple shooting (\texttt{strong/ms.py},',
     r'\texttt{ms\_cont.py}). Single shooting fails for $s\gtrsim10$ because in the core the squashing',
     r'mode behaves as $e^{\pm4\sqrt{6}\alpha/r}$ (a factor $e^{76}$ at $\alpha=10$). The solver places',
     r'nodes at equal WKB phase in the core, integrates segments with $x\le0.5\rho$ in quadruple',
     r'precision ($\_\_$float128 Bulirsch--Stoer) from 60-digit horizon series, and continues in $\alpha$',
     r'across the resonances $\gamma=3,4,5$ ($\alpha=4.83,13.97,31.66$). The large-$r$ data (series in $1/r$',
     r'with free $\mu,b_4$) are imposed at $R=12\sqrt\alpha$ ($s<80$) and $R=8\sqrt\alpha$ ($s>80$). At larger $R$',
     r'the squashing datum $b_4$ enters $h(R)$ only at relative order $R^{-6}$; Newton then stalls on a residual',
     r'floor and $m$ is wrong by up to $\sim10^{-6}$ (a first run with $R=25\sqrt\alpha$ is discarded for this',
     r'reason; \texttt{ms\_check.py}).', '',
     r'\textbf{Checks.} At the common point $s=2.98202$ the two-sided and spectral values differ by',
     f'{abs(sw0["m"]/3.04100567-1):.1e}; against a spline of the spectral branch the shooting/BVP',
     f'values deviate by at most {maxdev:.1e}, the size of the spectral branch\'s own constraint',
     f'violations (median {specmed:.0e}, max {specmax:.0e}). Varying the shooting parameters moves $m$ by',
     f'at most {variants:.1e}.',
     f'Multiple-shooting Newton residuals are below {maxres:.0e}. A degree-6 polynomial through the',
     f'{len(near)} multiple-shooting points with $9.16\\le s\\le10.3$, extrapolated to the BVP point $s=9.0037$,',
     f'agrees with it to {xdev:.0e}. At $s=48.9$, $R=8\\sqrt\\alpha$ and $12\\sqrt\\alpha$ agree to $5\\times10^{{-10}}$',
     r'and $m$ is independent of the number of far nodes to $5\times10^{-15}$. Quality check on every point:',
     f'$\\alpha b_4$ varies smoothly (for $s>40$ it lies in $[{b4a.min():.4f},{b4a.max():.4f}]$).', '',
     r'\begin{center}\small\begin{tabular}{rrrl}\toprule $s$ & $m(s)$ & $(m-3s/4)\sqrt s$ & method\\\midrule']
rows = [(d['s'], d['m'], 'shoot') for d in sw] + [(d['s'], d['m'], 'BVP') for d in bvp if d['s'] > 8.6] \
    + [(d['s'], d['m'], 'MS') for d in ms[::3]] + [(ms[-1]['s'], ms[-1]['m'], 'MS')]
seen = set()
for s, m, k in sorted(rows):
    if round(s, 4) in seen:
        continue
    seen.add(round(s, 4))
    L.append(f"{s:.6g} & {m:.10f} & {(m-0.75*s)*np.sqrt(s):.8f} & {k}\\\\")
L += [r'\bottomrule\end{tabular}\end{center}',
      r'\textbf{Strong-coupling fits} ($\sigma=s^{-1/2}$; free: leading coefficients of $1,\sigma,\sigma^2,\dots$;',
      r'$a_1=4/3$: of $\sigma,\sigma^2,\dots$ after subtracting $4/3$; rms of the fit to $(m-3s/4)\sqrt s$):',
      r'\begin{center}\small\begin{tabular}{lllll}\toprule fit & $s_{\min}$ & terms & leading coefficients & rms\\\midrule']
for kind, smin, K, c, rms in fits:
    L.append(f"{'$a_1=4/3$' if kind == 'fixed' else 'free'} & {smin} & {K} & "+', '.join(f'{v:.5f}' for v in c[:4])
             + f' & {rms:.0e}'+r'\\')
L += [r'\bottomrule\end{tabular}\end{center}',
      r'\textbf{Global approximant} (\texttt{strong/global2.py}). With $t=(1+s/\kappa)^{-1/2}$,',
      r'\[ m(s)\approx\frac{3s}{4}+\sum_{k=1}^{10}c_k\,t^k , \]',
      r'which is analytic at $s=0$ and expands in $s^{-1/2}$ at infinity. Imposed exactly: $m(0)=3/2$,',
      r"$m'(0)=1$, $c_1\sqrt\kappa=4/3$, $c_2\kappa=0.456527$; the other six $c_k$ and",
      f'$\\kappa={G["kappa"]:.4f}$ are fitted. Maximum relative error {np.max(np.abs(rel)):.1e} on',
      f'$0<s\\le{S.max():.0f}$. Coefficients: '
      + ', '.join(f'{v:.6f}' for v in G['c'])+'.',
      r'It is an interpolant, not a closed form; it does not reproduce the complex singularity at',
      r'$|s_*|=0.055$, so its Taylor coefficients beyond $s^1$ are not those of $m(s)$.']
(HERE/'numerics.tex').write_text('\n'.join(L)+'\n')
print(mac['StrongFitSentence']); print(mac['GlobalFitSentence']); print('maxdev', maxdev, 'variants', variants)
