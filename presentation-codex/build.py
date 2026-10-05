#!/usr/bin/env python3
"""Build matching PDF/HTML slides and an eight-part timed speaking script."""
from pathlib import Path
import base64, html, json, re, shutil, subprocess
P=Path(__file__).resolve().parent
B=P/'build'; B.mkdir(exist_ok=True)
data=json.loads((P/'script-data.json').read_text())
assert len(data)==8 and sum(s['seconds'] for s in data)==600

def tex(s):
    s=s.replace('–','--').replace('’',"'")
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#'}.get(c,c) for c in s)
def stamp(n): return f'{n//60}:{n%60:02d}'

affiliations=[
('IAFE Gravitation and Cosmology group', 'https://www.iafe.uba.ar/relatividad/gravity.html', 'Lists Cecilia Bejarano and Andrés Goya as permanent researchers, and David Blanco as an external collaborator at IFIBA, CONICET–UBA.'),
('UBA high-energy theory group', 'https://wp.df.uba.ar/hepth/people/', 'Lists David Blanco, Andrés Goya and Guillem Pérez-Nadal among faculty.'),
('UBA researcher directory: Guillem Pérez-Nadal', 'https://df.uba.ar/es/investigadores1/investigadores-y-becarios/miembro/75-Guillem_Perez_Nadal', 'Lists a UBA faculty appointment and CONICET researcher status.'),
('UBA department newsletter, May 2025', 'https://www.df.uba.ar/attachments/article/8731/Newsletter%20mayo%202025.pdf', 'Identifies Blanco, Goya and Pérez-Nadal among UBA–CONICET researchers organizing the school.')]
source_sections=[
('Scientific sources', 'Slides 3–6 summarize short-summary/summary.tex and manuscript/main.tex, using their existing figures and generated numbers. No new black-hole solutions were computed for this presentation. The response identity is rechecked algebraically in validate.py. The first law and analytic benchmarks are established inputs. The finite-coupling numerical response is the project result.'),
('Papers already stored in the project', 'Brihaye, Kleihaus, Kunz and Radu, arXiv:1010.0860, supplies the numerical family and Figure 1 comparison. Ma, Li and Lü, arXiv:2009.00015, supplies the first-order extremal mass correction. Kleihaus, Kunz and Radu, arXiv:2303.12471, already constructs the extremal branch directly. All three source PDFs are in papers/.'),
('Figures', 'The comparison and slope figures are unchanged project figures. Slide 6 shows only the left panel of fig-massext.pdf, with all axes of that panel preserved. The PDF slides retain vector figures. The HTML embeds a vector rendering of every slide and a searchable speaking script. No external fonts, scripts or images are required.'),
('Prompt provenance', 'Slide 2 quotes entry 001, dated 4 September 2026, in prompts/prompt-log-cronologico.md. Entries 002 and 003 supply the literature list and course-compliance request. The English emphasis line is a paraphrase, not another historical prompt.'),
('Usage provenance and limitations', 'Slide 7 transcribes prompts/uso-de-tiempo-y-tokens.md. Exact totals: 1,415,508,752 processed tokens and 5,942,462 output tokens. Claude Code cache reads alone account for 1,301,219,499 processed tokens. The report totals active time as 36.8 h, while its milestone table totals 36.6 h; this unresolved 0.2 h discrepancy is retained here rather than silently reconciled. Varying the inactivity cutoff from 10 to 30 minutes changes active time from 30.2 to 39.3 h. The report excludes human work, unattended numerical runs and missing sessions. These historical totals exclude preparation of this presentation. They were not recomputed from the original private session logs.'),
('AI acknowledgment', 'Suggested wording: “AI assistance: Claude (Anthropic) and ChatGPT / Codex (OpenAI), for code development, calculations, checks and drafting. The human authors take responsibility for the scientific claims.” The stored workflow uses Claude Code and Codex. The acknowledgment distinguishes assistance from authorship without implying that the tools verified their own output independently.'),
('Affiliations and event', 'Web check: 29 September 2026. Cecilia Bejarano and Andrés Goya: IAFE, CONICET–UBA. David Blanco and Guillem Pérez-Nadal: Departamento de Física, FCEN, UBA and IFIBA, CONICET–UBA. Institutional listings support the appointments; department and institute names are expanded on the title slide. Course title, leaders, venue and 1–2 October dates follow the presentation request. The year is 2026, consistent with the project history and current date.')]

preamble=r'''\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=24mm]{geometry}
\usepackage{xcolor,parskip}
\usepackage[colorlinks=true,urlcolor=teal]{hyperref}
\definecolor{ink}{HTML}{152D40}
\color{ink}
\setlength{\parindent}{0pt}
\pagestyle{plain}
\begin{document}
'''
script=[preamble]
sections=[]
elapsed=0
for i,s in enumerate(data,1):
    span=f"{stamp(elapsed)}–{stamp(elapsed+s['seconds'])}";elapsed+=s['seconds'];s['span']=span
    if i>1:script.append(r'\newpage')
    script.append(r'{\small Speaking script · 10-minute talk · 1--2 October 2026}\par')
    script.append(r'\section*{'+str(i)+'. '+tex(s['title'])+'}')
    script.append(r'\textbf{'+tex(span)+f" ({s['seconds']} seconds)"+r'}\par')
    script.append(r'{\itshape '+tex(s['cue'])+r'}\par\medskip')
    script.append(tex(s['text']))
    sections.append(f'<section><p class="eyebrow">Slide {i} · {span} · {s["seconds"]} seconds</p><h2>{html.escape(s["title"])}</h2><p class="cue">{html.escape(s["cue"])}</p>'+''.join('<p>'+html.escape(p)+'</p>' for p in s['text'].split('\n\n'))+'</section>')
script.append(r'\newpage\section*{Sources and preparation notes}')
script.append('Not part of the spoken ten minutes. Timing assumes a deliberate pace with pauses to point at the figures. Rehearse once to match your own delivery.')
for title,body in source_sections:
    if title=='AI acknowledgment': script.append(r'\newpage')
    script.append(r'\subsection*{'+tex(title)+'}\n'+tex(body))
script.append(r'\subsection*{Affiliation links}')
for title,url,desc in affiliations:script.append(r'\href{'+url+r'}{'+tex(title)+r'}. '+tex(desc)+'\par')
script.append(r'\end{document}')
(P/'speaker-script.tex').write_text('\n'.join(script))
for name in ['slides','speaker-script']:
    for _ in range(2):
        run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build',name+'.tex'],cwd=P,capture_output=True,text=True)
        (B/(name+'-compile.txt')).write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(run.stdout[-2000:])
    shutil.copy2(B/(name+'.pdf'),P/(name+'.pdf'))

css='''body{margin:0;font:20px/1.6 system-ui,sans-serif;color:#152d40;background:#f4f6f7}main{max-width:850px;margin:40px auto;padding:24px 40px;background:white}h1,h2{line-height:1.2}h2{font-size:30px}section{padding:24px 0;border-bottom:1px solid #ccd6db}a{color:#007e87}.eyebrow,.cue{color:#536575}.cue{font-style:italic}.eyebrow{font-size:16px}nav{display:flex;flex-wrap:wrap;gap:12px}details{margin:24px 0}summary{cursor:pointer}@media print{body{background:white;font-size:12pt}main{margin:0;padding:0}section{break-before:page;border:0}nav{display:none}}'''
sources_html=''.join(f'<h3>{html.escape(t)}</h3><p>{html.escape(b)}</p>' for t,b in source_sections)
sources_html+='<h3>Affiliation sources checked online</h3><ul>'+''.join(f'<li><a href="{html.escape(u)}">{html.escape(t)}</a>: {html.escape(d)}</li>' for t,u,d in affiliations)+'</ul>'
(P/'sources.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sources and scope</title><style>'+css+'</style><main><h1>Sources and scope</h1>'+sources_html+'</main></html>')
(P/'speaker-script.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Ten-minute speaking script</title><style>'+css+'</style><main><h1>Ten-minute speaking script</h1><p>Eight slides · 1–2 October 2026. Read at a deliberate pace and pause at the figures. Timing is a rehearsal target.</p>'+''.join(sections)+'<section><h2>Sources and preparation notes</h2><p>Not part of the spoken ten minutes.</p>'+sources_html+'</section></main></html>')

frames=[]
for i,s in enumerate(data,1):
    svg=B/f'slide-{i}.svg'
    subprocess.run(['pdftocairo','-svg','-f',str(i),'-l',str(i),str(P/'slides.pdf'),str(svg)],check=True)
    b64=base64.b64encode(svg.read_bytes()).decode()
    frames.append(f'<section class="slide" id="slide-{i}" aria-label="Slide {i}: {html.escape(s["title"])}"><img src="data:image/svg+xml;base64,{b64}" alt="{html.escape(s["title"])}. Full narration follows in the speaking script."></section>')
slidecss='''html,body{margin:0;background:#13212a;color:white;font:16px system-ui,sans-serif}#stage{height:calc(100dvh - 48px);display:grid;place-items:center}.slide{display:none;width:100%;height:100%}.slide.active{display:flex;align-items:center;justify-content:center}.slide img{max-width:100%;max-height:100%;object-fit:contain;width:100%;height:100%}nav{height:48px;display:flex;align-items:center;justify-content:center;gap:16px}button{font:inherit;border:1px solid #81939c;border-radius:4px;background:transparent;color:white;padding:4px 12px;cursor:pointer}button:focus-visible{outline:3px solid #54c8d0}#notes{display:none;background:white;color:#152d40;max-width:900px;margin:auto;padding:28px;font-size:20px;line-height:1.6}body.notes #stage{height:55dvh}body.notes #notes{display:block}#all-script{display:none}body.reading #all-script{display:block;max-width:850px;margin:auto;background:white;color:#152d40;padding:32px;line-height:1.6}body.reading #stage,body.reading #notes{display:none}#all-script section{margin-bottom:40px}a{color:#007e87}.cue{color:#536575;font-style:italic}@media print{nav,#notes,#all-script{display:none!important}#stage{display:block;height:auto}.slide{display:block!important;width:100vw;height:auto;break-after:page}.slide img{display:block;width:100%;height:auto}@page{size:320mm 180mm;margin:0}}'''
js='''const slides=[...document.querySelectorAll('.slide')],notes=document.getElementById('notes'),counter=document.getElementById('counter');let current=0;function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>s.classList.toggle('active',i===current));counter.textContent=`${current+1} / ${slides.length} · ${data[current].span}`;notes.replaceChildren();let title=document.createElement('h2');title.textContent=data[current].title;notes.append(title);let cue=document.createElement('p');cue.className='cue';cue.textContent=data[current].cue;notes.append(cue);for(const p of data[current].text.split('\\n\\n')){let el=document.createElement('p');el.textContent=p;notes.append(el)}history.replaceState(null,'',`#${current+1}`)}document.getElementById('prev').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('note-toggle').onclick=()=>document.body.classList.toggle('notes');document.getElementById('read-toggle').onclick=()=>document.body.classList.toggle('reading');document.getElementById('full').onclick=()=>{if(!document.fullscreenElement){document.documentElement.requestFullscreen?.()}else{document.exitFullscreen?.()}};addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1)}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key.toLowerCase()==='n')document.body.classList.toggle('notes')});show((Number(location.hash.slice(1))||1)-1);'''
(P/'presentation.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rotating black holes with AI agents</title><style>'+slidecss+'</style></head><body><main id="stage">'+''.join(frames)+'</main><nav aria-label="Presentation controls"><button id="prev" aria-label="Previous slide">Previous</button><span id="counter" aria-live="polite"></span><button id="next">Next</button><button id="note-toggle">Notes (N)</button><button id="read-toggle">Script &amp; sources</button><button id="full">Fullscreen</button></nav><aside id="notes"></aside><article id="all-script"><h1>Speaking script</h1>'+''.join(sections)+'<h2>Sources and scope</h2>'+sources_html+'</article><script>const data='+json.dumps(data,ensure_ascii=False)+';'+js+'</script></body></html>')
words=sum(len(s['text'].split()) for s in data)
print(f'Built 8 slides; planned duration 600 seconds; speaking script {words} words.')
print('Outputs: slides.pdf, slides.tex, presentation.html, speaker-script.pdf, speaker-script.tex, speaker-script.html, sources.html')
