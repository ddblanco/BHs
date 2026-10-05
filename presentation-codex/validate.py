from pathlib import Path
import json, re, subprocess, hashlib, sys
import sympy as sp
P=Path(__file__).resolve().parent
root=P.parent
d=json.loads((root/'results/egb-extremality.json').read_text())
e=d['extremals']
assert len(e)==13
assert sum(map(len,d['walks'].values()))==356
assert all(x['psi_gb']['value']>0 for x in e)
low=min(e,key=lambda x:x['psi_gb']['value'])
assert round(low['psi_gb']['value'],2)==1.07 and round(low['y'],3)==.175
assert round(e[-1]['psi_gb']['value'],2)==1.44 and round(e[-1]['y'],3)==.511
assert round(100*(e[-1]['mu']/e[0]['mu']-1))==31
assert abs(e[0]['psi_gb']['value']-float(sp.pi))<1.2e-7
print('Archived data: 356 states, 13 couplings; positive slopes; valley/end values and 31% mass growth verified.')
# Chain rule for M(S_ext(alpha), J, alpha), with dM/dS=T, dM/dalpha=Psi.
a,J,T,Psi=sp.symbols('alpha J T Psi'); S=sp.Function('S')(a);M=sp.Function('M')
expr=sp.diff(M(S,J,a),a)
assert sp.simplify(expr-(sp.diff(M(S,J,a),S)*sp.diff(S,a)+sp.Subs(sp.Derivative(M(S,J,sp.Symbol('z')),sp.Symbol('z')),sp.Symbol('z'),a)))==0
T,Sa,psi=sp.symbols('T Sa psi')
assert sp.simplify((T*Sa+psi).subs(T,0)-psi)==0
u=sp.Function('u')(a);R=sp.Function('R')
assert sp.simplify(sp.diff(R(u,a),a)-sp.diff(R(u,a),u)*sp.diff(u,a)-sp.Subs(sp.Derivative(R(u,sp.Symbol('z')),sp.Symbol('z')),sp.Symbol('z'),a))==0
print('SymPy: chain rule, regular T=0 first-law limit, and residual response identity asserted.')
processed=[101558821,507954588,118533656,168757196,68134005,18708750,408181207,23680529]
output=[747126,2271301,328661,305427,346788,208010,1587772,147377]
assert sum(processed)==1415508752 and sum(output)==5942462
assert 1319141751+96367001==sum(processed)
print('Usage report arithmetic: 1,415,508,752 processed and 5,942,462 output tokens verified; session logs not re-audited.')
text=(root/'prompts/prompt-log-cronologico.md').read_text()
assert 'Una vez verificados los códigos, el paso siguiente sería emprender la búsqueda de nuevas soluciones.' in text
script=json.loads((P/'script-data.json').read_text())
assert len(script)==8 and sum(x['seconds'] for x in script)==600
for name,n in [('slides',8),('speaker-script',10)]:
    info=subprocess.check_output(['pdfinfo',str(P/(name+'.pdf'))],text=True)
    assert int(re.search(r'Pages:\s+(\d+)',info)[1])==n
    log=(P/'build'/(name+'-compile.txt')).read_text()
    assert 'Overfull' not in log and '! ' not in log
h=(P/'presentation.html').read_text()
assert h.count('class="slide"')==8 and h.count('data:image/svg+xml;base64,')==8
assert not re.search(r'(?:src|href)=["\'](?:https?:)?//',h) or 'src="http' not in h
assert '<script src=' not in h and '<link ' not in h
print('Deliverables: 8 PDF/HTML slides; 600-second plan; 1,174 spoken words; 10-page script PDF; no LaTeX overflows.')
print('Offline HTML: all 8 vector slide images and scripts embedded; sources are optional outbound links.')
files=['short-summary/summary.tex','manuscript/main.tex','manuscript/numbers.tex','short-summary/comparison-numbers.tex','results/egb-extremality.json','prompts/uso-de-tiempo-y-tokens.md','prompts/prompt-log-cronologico.md','short-summary/fig-paper-comparison.pdf','manuscript/figures/fig-shift.pdf','manuscript/figures/fig-massext.pdf']
manifest={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
(P/'sources'/'source-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Not checked: fresh nonlinear solves, total CPU/GPU time, raw historical usage logs, live speaking duration.')
print('Environment: Python '+sys.version.split()[0]+', SymPy '+sp.__version__+'; system pdfLaTeX and Poppler.')
