"""Build the verification report in LaTeX and HTML from the ledger.

Both documents are generated from `results/verification-ledger.json`, so the
two cannot disagree with each other or with the suite that produced them. No
verdict and no number is typed into either template.

Run from the project root:
    PYTHONPATH=src python3 verification-report/build_report.py
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEDGER = json.loads((ROOT/'results/verification-ledger.json').read_text(encoding='utf-8'))
ROWS = LEDGER['ledger']
SUMMARY = LEDGER['summary']

KINDS = {
    'analytic': ('Rederived analytically',
                 'Redone here with sympy and the difference asserted to vanish '
                 'identically. No tolerance and no solver output enters.'),
    'numerical': ('Reproduced numerically',
                  "Computed by this project's solver and compared against a "
                  'closed form or a published equation. Each carries the '
                  'tolerance its assertion used.'),
    'transcription': ('Transcription checked against the source',
                      'A convention, normalisation or formula this work takes '
                      'from a paper, checked against the text of that paper as '
                      'downloaded in papers/.'),
    'called': ('Called, not reproduced',
               'Used on the authority of the source. Listed so that it cannot '
               'be mistaken for anything above.'),
}

FIGURES = [
    ('fig-myers-perry',
     'The vacuum limit against Myers--Perry.',
     'Upper: the mass of the solver\'s $\\alpha=0$ solutions against the closed '
     'form $M=3\\pi r_H^2/[8(1-q^2)]$, which follows from $3\\pi\\mu/8$ and the '
     'coordinate map $r^2=\\rho^2+a^2$ rederived in this report. Lower: the '
     'relative deviation, against the tolerance the assertion used. The '
     'comparison runs to $q=0.7$, past the vacuum extremal spin $1/\\sqrt2$ '
     'being approached.'),
    ('fig-boulware-deser',
     'The static limit against Boulware--Deser, at finite coupling.',
     'Upper: the potential $\\Psi$ measured at zero spin against the closed form '
     'derived in this report by differentiating the Boulware--Deser charges. '
     'This is the one exact check available at $\\alpha\\neq0$. The dotted line '
     'is the $\\alpha\\to0$ value $-9\\pi/4$, which the independent perturbative '
     'formula also returns. Lower: relative deviations in $M$, $T$ and $\\Psi$. '
     'The entropy is reproduced to machine precision and is omitted from the '
     'lower panel for that reason.'),
    ('fig-perturbative',
     'The zero-coupling response against the published perturbative potential.',
     'Upper: the implicit numerical response evaluated on the Myers--Perry '
     'background against $\\Psi_0(q)$, which this report re-derives by '
     'specialising the general-$D$ result of Wu and L\\"u to $D=5$ with equal '
     'spins rather than by copying it. The star marks the extremal endpoint '
     '$\\Psi_0=\\pi$, which is the published linear-order extremality shift and '
     'was not used in obtaining the curve. Lower: relative deviation.'),
    ('fig-near-horizon',
     'The extremal entropy against the published near-horizon branch.',
     'Upper: $\\sigma=S_\\mathrm{ext}/J$ from the near-horizon entropy function, '
     'rederived here and checked against eqs.~(4.11) and~(4.12) of '
     'arXiv:1010.0860 in the paper\'s own variables, against the zero-temperature '
     'extrapolation of the bulk solutions. The star is the extremal '
     'Myers--Perry value $2\\pi$. Lower: relative difference between the two '
     'routes.'),
]


def tex_escape(text):
    out = (text.replace('\\', r'\textbackslash{}').replace('&', r'\&')
               .replace('%', r'\%').replace('$', r'\$').replace('#', r'\#')
               .replace('_', r'\_').replace('{', r'\{').replace('}', r'\}')
               .replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}'))
    return out


def tex_number(value):
    if value is None:
        return '--'
    if value == 0:
        return '$0$'
    text = f'{value:.2e}'
    mantissa, exponent = text.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def html_number(value):
    if value is None:
        return '&mdash;'
    if value == 0:
        return '0'
    text = f'{value:.2e}'
    mantissa, exponent = text.split('e')
    return f'{mantissa}&times;10<sup>{int(exponent)}</sup>'


def build_tex():
    counts = SUMMARY['by_kind']
    lines = []
    for kind, (title, blurb) in KINDS.items():
        rows = [r for r in ROWS if r['kind'] == kind]
        if not rows:
            continue
        lines.append(rf'\subsection*{{{title}}}')
        lines.append(tex_escape(blurb).replace('papers/', r'\texttt{papers/}'))
        lines.append('')
        if kind == 'called':
            lines.append(r'\begin{longtable}{@{}L{.30\textwidth}L{.22\textwidth}L{.42\textwidth}@{}}')
            lines.append(r'\toprule')
            lines.append(r'what & source & status \\')
            lines.append(r'\midrule\endhead')
            for row in rows:
                lines.append(rf"{tex_escape(row['claim'])} & "
                             rf"{tex_escape(row['source'])} & "
                             rf"{tex_escape(row['detail'])} \\")
            lines.append(r'\bottomrule')
            lines.append(r'\end{longtable}')
        else:
            lines.append(r'\begin{longtable}{@{}L{.38\textwidth}L{.24\textwidth}rrc@{}}')
            lines.append(r'\toprule')
            lines.append(r'claim & source & result & tolerance & verdict \\')
            lines.append(r'\midrule\endhead')
            for row in rows:
                verdict = r'\PASS' if row['passed'] else r'\FAIL'
                lines.append(
                    rf"{tex_escape(row['claim'])} & {tex_escape(row['source'])} & "
                    rf"{tex_number(row['value'])} & {tex_number(row['tolerance'])} & "
                    rf"{verdict} \\")
            lines.append(r'\bottomrule')
            lines.append(r'\end{longtable}')
        lines.append('')

    figures = []
    for stem, caption, body in FIGURES:
        figures.append(rf"""\begin{{figure}}[p]
 \centering
 \includegraphics[width=.92\textwidth]{{figures/{stem}.pdf}}
 \caption{{\textbf{{{caption}}} {body}}}
 \label{{fig:{stem}}}
\end{{figure}}""")

    papers = '\n'.join(
        rf"\texttt{{{tex_escape(p['name'])}}} & {p['bytes']//1024}\,kB \\"
        for p in LEDGER['papers'])

    versions = LEDGER['versions']
    document = rf"""\documentclass[11pt,a4paper]{{article}}
\ifdefined\pdftexversion\pdfoutput=1\fi
\ifdefined\XeTeXrevision\PassOptionsToPackage{{xetex}}{{hyperref}}\fi
\usepackage[margin=24mm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage[table]{{xcolor}}
\usepackage{{caption}}
\usepackage[hidelinks]{{hyperref}}
\captionsetup{{font=small,labelfont=bf,labelsep=period}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{.55em}}
\definecolor{{pass}}{{HTML}}{{176B45}}
\definecolor{{fail}}{{HTML}}{{8B1E1E}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\newcommand{{\PASS}}{{\textcolor{{pass}}{{\textbf{{pass}}}}}}
\newcommand{{\FAIL}}{{\textcolor{{fail}}{{\textbf{{fail}}}}}}

\title{{What was rederived, what was reproduced, and what was only called\\[.35em]
\large A verification record for the finite-coupling extremality-shift calculation}}
\author{{}}
\date{{28 September 2026}}

\begin{{document}}
\maketitle
\thispagestyle{{empty}}

\begin{{abstract}}
\noindent
This is not a physics paper; it is an audit trail. It states, claim by claim,
which parts of the accompanying calculation were derived here, which were
computed here and checked against something external, which were copied from a
paper and verified against that paper, and which were simply used on the
author's authority. Of {SUMMARY['total']} entries,
{SUMMARY['checked']} carry a machine-checked verdict and
{counts['called']['total']} are listed as called and not reproduced.
{SUMMARY['failed']} checks fail.
The external anchors are the Myers--Perry solution in the vacuum limit, the
Boulware--Deser solution at finite coupling and zero spin, the perturbative
potential of Wu and L\"u at zero coupling, the extremal near-horizon branch of
Brihaye, Kleihaus, Kunz and Radu at finite coupling, and the extremality shift
of Ma, Li and L\"u. Everything is regenerated by
\texttt{{work/verification\_suite.py}}, which is what produced the tables below.
\end{{abstract}}

\section*{{How to read this}}

Four kinds of entry appear, and the distinction between them is the reason the
document exists. A calculation that agrees with itself has shown nothing; the
question is always what it was tested against, and with what.

\begin{{description}}
\item[Rederived analytically.] Redone here symbolically, with the difference
asserted to be exactly zero. These need no tolerance because nothing numerical
enters: either the expressions are identical or they are not.
\item[Reproduced numerically.] Produced by the spectral solver and compared with
a closed form or a published equation. Each carries the tolerance its assertion
used, and figures~\ref{{fig:fig-myers-perry}}--\ref{{fig:fig-near-horizon}} draw
the deviation against that tolerance so the margin is visible rather than
asserted.
\item[Transcription checked against the source.] Every convention this work
imports --- a coupling normalisation, a charge normalisation, a factor in a
near-horizon metric --- read back from the source PDF. These were previously the
weakest link: the manuscript's own referee record listed the literature
attributions as unverified because no source PDFs were present. They now are,
under \texttt{{papers/}}, and the checks below read them.
\item[Called, not reproduced.] Used and not re-derived. Listing these is the
only way the three categories above mean anything.
\end{{description}}

\section*{{The three conversions that everything else depends on}}

Three numbers relate this work's conventions to the literature, and a mistake in
any of them would move every finite-coupling comparison without breaking any
internal check. Each is now verified twice, by independent routes.

\paragraph{{The coupling.}} The source writes its action as
$I=(16\pi G)^{{-1}}\int\!\sqrt{{-g}}\,(R+\tfrac{{\alpha_{{\rm ref.}}}}{{4}}\mathcal L_{{\rm GB}})$
while this work writes $R+\alpha\mathcal L_{{\rm GB}}$, so
$\alpha_{{\rm ref.}}=4\alpha$. That is read directly from eq.~(2.1) of
arXiv:1010.0860. Independently, the ratio of the Gauss--Bonnet piece of the
published entropy~(3.10) to its Einstein piece equals the ratio our
Jacobson--Myers form gives \emph{{only}} if $\alpha_{{\rm ref.}}=4\alpha$; that
is checked symbolically. Two routes, one answer.

\paragraph{{The angular momentum.}} The source's $J$ is the momentum in
\emph{{each}} plane, not the total, which is the sentence introducing its
eq.~(3.6). Its normalisations $E=-3V_3U/16\pi G$ and $J=V_3W/8\pi G$ with
$V_3=2\pi^2$ reduce to the $-3\pi U/8$ and $\pi W/4$ used here.

\paragraph{{The near-horizon $k$.}} The source's near-horizon metric carries
$\sigma_3+2kr\,\mathrm{{d}}t$ where this work writes $\sigma_3+kr\,\mathrm{{d}}t$,
so $k=2k_{{\rm ref.}}$ and, since $J=\partial E/\partial k$,
$J_{{\rm ref.}}=2J$. This is what reconciles the source's
$S_{{\rm ext}}=\pi J_{{\rm ref.}}$ at zero coupling with the
$S_{{\rm ext}}=2\pi J$ obtained here, and both with extremal Myers--Perry,
$S=2\pi^2a^3$ and $J=\pi a^3$. Without the factor this would look like a
discrepancy of two; with it, the three statements are one statement.

\section*{{The ledger}}

{chr(10).join(lines)}

\section*{{Sources held locally}}

Every paper the calculation leans on that exists on arXiv is in
\texttt{{papers/}}, and the transcription checks above read them from there. The
three pre-arXiv sources --- Myers and Perry (1986), Boulware and Deser (1985)
and Lovelock (1971) --- are not available that way and are listed as called.

\begin{{center}}
\small
\begin{{tabular}}{{@{{}}lr@{{}}}}
\toprule
file & size \\
\midrule
{papers}
\bottomrule
\end{{tabular}}
\end{{center}}

\section*{{What this record does not establish}}

Agreement with an external result bounds the error at the point of comparison
and nowhere else. The vacuum and static checks are exact but sit at the edges of
the family --- zero coupling and zero spin --- and the near-horizon check
constrains the entropy, not the mass. No external result exists for the rotating
mass at finite coupling, which is precisely the quantity the accompanying
manuscript reports; that number rests on the solver, its convergence behaviour
and the zero-temperature extrapolation, and on nothing published. The tolerances
quoted here are the ones the assertions used, chosen to be loose enough to pass
reliably and not as estimates of the accuracy achieved: the achieved deviations
are typically several orders of magnitude smaller, which is what the lower
panels of the figures show.

{chr(10).join(figures)}

\vfill
\noindent\rule{{\textwidth}}{{.4pt}}\par
\footnotesize
Generated by \texttt{{verification-report/build\_report.py}} from
\texttt{{results/verification-ledger.json}}. Environment:
{tex_escape(versions['platform'])}, Python {versions['python']}, NumPy
{versions['numpy']}, SymPy {versions['sympy']}.

\end{{document}}
"""
    (HERE/'verification-report.tex').write_text(document, encoding='utf-8')
    print('wrote verification-report/verification-report.tex')


def build_html():
    counts = SUMMARY['by_kind']
    sections = []
    for kind, (title, blurb) in KINDS.items():
        rows = [r for r in ROWS if r['kind'] == kind]
        if not rows:
            continue
        if kind == 'called':
            head = '<tr><th>What</th><th>Source</th><th>Status</th></tr>'
            body = '\n'.join(
                f'<tr><td>{html.escape(r["claim"])}</td>'
                f'<td class="src">{html.escape(r["source"])}</td>'
                f'<td class="note">{html.escape(r["detail"])}</td></tr>'
                for r in rows)
        else:
            head = ('<tr><th>Claim</th><th>Source</th><th>Result</th>'
                    '<th>Tolerance</th><th>Verdict</th></tr>')
            body = '\n'.join(
                f'<tr><td>{html.escape(r["claim"])}'
                f'<div class="method">{html.escape(r["method"])}</div></td>'
                f'<td class="src">{html.escape(r["source"])}</td>'
                f'<td class="num">{html_number(r["value"])}</td>'
                f'<td class="num">{html_number(r["tolerance"])}</td>'
                f'<td><span class="v {"pass" if r["passed"] else "fail"}">'
                f'{"PASS" if r["passed"] else "FAIL"}</span></td></tr>'
                for r in rows)
        passed = counts[kind]['passed']
        total = counts[kind]['total']
        tally = (f'{passed} / {total}' if kind != 'called' else f'{total}')
        sections.append(f"""<h2>{html.escape(title)} <span class="tally">{tally}</span></h2>
<p class="blurb">{html.escape(blurb)}</p>
<table><thead>{head}</thead><tbody>
{body}
</tbody></table>""")

    figures = '\n'.join(f"""<figure>
  <img src="figures/{stem}.png" alt="{html.escape(re.sub(r'[$\\\\]', '', caption))}">
  <figcaption><b>{html.escape(caption)}</b> {html.escape(re.sub(r'[$\\\\]|mathrm|textbackslash', '', body))}</figcaption>
</figure>""" for stem, caption, body in FIGURES)

    papers = '\n'.join(
        f'<tr><td class="mono">{html.escape(p["name"])}</td>'
        f'<td class="num">{p["bytes"]//1024} kB</td></tr>'
        for p in LEDGER['papers'])

    versions = LEDGER['versions']
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verification Record</title>
<style>
  :root {{
    --bg:#fbfaf8; --panel:#fff; --ink:#1d1d1f; --soft:#5d6570; --line:#e2ded7;
    --accent:#8b3a1e; --good:#176b45; --code:#f4f2ee;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#17181a; --panel:#1f2124; --ink:#e8e6e3; --soft:#a4acb6;
      --line:#32353a; --accent:#e08a68; --good:#6dc496; --code:#26282c;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#17181a; --panel:#1f2124; --ink:#e8e6e3; --soft:#a4acb6;
    --line:#32353a; --accent:#e08a68; --good:#6dc496; --code:#26282c;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.62 Georgia,"Iowan Old Style","Times New Roman",serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:56px 16px 90px; }}
  header {{ border-bottom:2px solid var(--line); padding-bottom:22px; margin-bottom:30px; }}
  h1 {{ font-size:1.9rem; line-height:1.22; margin:0 0 .35rem; letter-spacing:-.012em; }}
  .sub {{ color:var(--soft); font-size:.95rem; font-style:italic; margin:0; }}
  h2 {{ font-size:1.25rem; margin:2.5rem 0 .5rem; padding-bottom:.3rem;
        border-bottom:1px solid var(--line); }}
  h3 {{ font-size:1rem; margin:1.6rem 0 .4rem; }}
  .tally {{ float:right; font-family:"SFMono-Regular",Menlo,Consolas,monospace;
            font-size:.82rem; color:var(--soft); font-weight:400; }}
  p,li {{ margin:0 0 .8rem; }}
  .blurb {{ color:var(--soft); font-size:.9rem; }}
  .lede {{ background:var(--panel); border:1px solid var(--line);
           border-left:4px solid var(--accent); border-radius:5px;
           padding:18px 22px; margin:0 0 26px; }}
  .lede p:last-child {{ margin-bottom:0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; margin:22px 0 28px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line);
           border-radius:5px; padding:13px 15px; }}
  .stat b {{ display:block; font-size:1.4rem; line-height:1.2;
             font-variant-numeric:tabular-nums; }}
  .stat span {{ color:var(--soft); font-size:.8rem; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; font-size:.86rem; }}
  th,td {{ text-align:left; padding:7px 9px; border-bottom:1px solid var(--line);
           vertical-align:top; }}
  th {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.05em;
        color:var(--soft); font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.src {{ color:var(--soft); font-size:.82rem; }}
  td.note {{ color:var(--soft); font-size:.82rem; }}
  .method {{ color:var(--soft); font-size:.76rem; margin-top:.2rem; }}
  .mono,code {{ font-family:"SFMono-Regular",Menlo,Consolas,monospace; }}
  code {{ background:var(--code); padding:.1em .35em; border-radius:3px; font-size:.85em; }}
  .v {{ font-family:"SFMono-Regular",Menlo,Consolas,monospace; font-size:.72rem;
        font-weight:700; letter-spacing:.05em; }}
  .v.pass {{ color:var(--good); }}
  .v.fail {{ color:var(--accent); }}
  dl dt {{ font-weight:700; margin-top:.7rem; }}
  dl dd {{ margin:0 0 .5rem; color:var(--ink); }}
  figure {{ margin:22px 0 10px; }}
  figure img {{ display:block; width:100%; height:auto; border:1px solid var(--line);
                border-radius:5px; background:#fff; }}
  figcaption {{ color:var(--soft); font-size:.83rem; margin-top:.5rem; }}
  footer {{ margin-top:44px; padding-top:18px; border-top:1px solid var(--line);
            color:var(--soft); font-size:.83rem; }}
  @media (max-width:620px) {{
    .wrap {{ padding:34px 16px 70px; }} h1 {{ font-size:1.5rem; }}
    table {{ font-size:.78rem; }} .tally {{ float:none; display:block; }}
  }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>What was rederived, what was reproduced,<br>and what was only called</h1>
  <p class="sub">A verification record for the finite-coupling extremality-shift
  calculation &mdash; 28 September 2026</p>
</header>

<div class="lede">
  <p>This is not a physics document; it is an audit trail. It states, claim by
  claim, which parts of the accompanying calculation were <b>derived here</b>,
  which were <b>computed here and checked against something external</b>, which
  were <b>copied from a paper and verified against that paper</b>, and which were
  <b>simply used</b> on the author's authority.</p>
  <p>A calculation that agrees with itself has shown nothing. The question is
  always what it was tested against, and with what &mdash; so the fourth category
  is as load-bearing as the first three.</p>
</div>

<div class="grid">
  <div class="stat"><b>{counts['analytic']['passed']} / {counts['analytic']['total']}</b><span>rederived analytically</span></div>
  <div class="stat"><b>{counts['numerical']['passed']} / {counts['numerical']['total']}</b><span>reproduced numerically</span></div>
  <div class="stat"><b>{counts['transcription']['passed']} / {counts['transcription']['total']}</b><span>transcriptions checked</span></div>
  <div class="stat"><b>{counts['called']['total']}</b><span>called, not reproduced</span></div>
</div>

<h2>The three conversions everything depends on</h2>
<p>Three numbers relate this work's conventions to the literature. A mistake in
any of them would move every finite-coupling comparison without breaking a single
internal check, which is exactly the kind of error an internally consistent
calculation cannot find. Each is now verified twice, by independent routes.</p>
<dl>
  <dt>The coupling: &alpha;<sub>ref.</sub> = 4&alpha;</dt>
  <dd>The source writes <i>R</i> + (&alpha;<sub>ref.</sub>/4)&thinsp;<i>L</i><sub>GB</sub>
  where this work writes <i>R</i> + &alpha;&thinsp;<i>L</i><sub>GB</sub>. Read
  directly from eq.&nbsp;(2.1) of arXiv:1010.0860. <b>Independently:</b> the ratio
  of the Gauss&ndash;Bonnet piece of the published entropy&nbsp;(3.10) to its
  Einstein piece matches the ratio our Jacobson&ndash;Myers form gives
  <em>only</em> if &alpha;<sub>ref.</sub>&nbsp;=&nbsp;4&alpha;. Two routes, one
  answer.</dd>
  <dt>The angular momentum: <i>J</i> is per plane</dt>
  <dd>The sentence introducing the source's eq.&nbsp;(3.6) says so, and its
  <i>E</i> = &minus;3<i>V</i><sub>3</sub><i>U</i>/16&pi;<i>G</i>,
  <i>J</i> = <i>V</i><sub>3</sub><i>W</i>/8&pi;<i>G</i> with
  <i>V</i><sub>3</sub> = 2&pi;<sup>2</sup> reduce to the
  &minus;3&pi;<i>U</i>/8 and &pi;<i>W</i>/4 used here.</dd>
  <dt>The near-horizon <i>k</i>: ours is twice theirs</dt>
  <dd>The source's near-horizon metric carries
  &sigma;<sub>3</sub>&nbsp;+&nbsp;2<i>kr</i>&thinsp;d<i>t</i> where this work
  writes &sigma;<sub>3</sub>&nbsp;+&nbsp;<i>kr</i>&thinsp;d<i>t</i>, so
  <i>J</i><sub>ref.</sub> = 2<i>J</i> because <i>J</i> = &part;<i>E</i>/&part;<i>k</i>.
  This is what reconciles the source's <i>S</i><sub>ext</sub> = &pi;<i>J</i><sub>ref.</sub>
  at zero coupling with the 2&pi;<i>J</i> obtained here, and both with extremal
  Myers&ndash;Perry (<i>S</i> = 2&pi;<sup>2</sup><i>a</i><sup>3</sup>,
  <i>J</i> = &pi;<i>a</i><sup>3</sup>). Without the factor this looks like a
  discrepancy of two; with it, the three statements are one statement.</dd>
</dl>

{chr(10).join(sections)}

<h2>The comparisons, with their margins</h2>
<p>Each figure pairs an external result with what this solver produced, and shows
underneath the deviation against the tolerance the assertion actually used. The
margin is the point: in every case the achieved deviation is orders of magnitude
below the bar it had to clear.</p>
{figures}

<h2>Sources held locally</h2>
<p>Every paper the calculation leans on that exists on arXiv is now in
<code>papers/</code>, and the transcription checks above read them from there.
This closes a gap the earlier referee record named explicitly: the literature
attributions had been unverifiable because no source PDFs were present. The three
pre-arXiv sources &mdash; Myers and Perry (1986), Boulware and Deser (1985),
Lovelock (1971) &mdash; are not available that way and are listed as called.</p>
<table><thead><tr><th>File</th><th>Size</th></tr></thead><tbody>
{papers}
</tbody></table>

<h2>What this record does not establish</h2>
<p>Agreement with an external result bounds the error at the point of comparison
and nowhere else. The vacuum and static checks are exact, but they sit at the
edges of the family &mdash; zero coupling and zero spin &mdash; and the
near-horizon check constrains the entropy, not the mass.</p>
<p><b>No external result exists for the rotating mass at finite coupling</b>,
which is precisely the quantity the accompanying manuscript reports. That number
rests on the solver, its convergence behaviour and the zero-temperature
extrapolation, and on nothing published. The tolerances above are the ones the
assertions used, chosen loose enough to pass reliably; they are not estimates of
the accuracy achieved, which is typically orders of magnitude better and is what
the lower panels of the figures show.</p>

<footer>
  Generated by <code>verification-report/build_report.py</code> from
  <code>results/verification-ledger.json</code>, which
  <code>work/verification_suite.py</code> produces. No verdict or number in this
  page is typed by hand. Environment: {html.escape(versions['platform'])},
  Python {versions['python']}, NumPy {versions['numpy']}, SymPy {versions['sympy']}.
</footer>

</div>
</body>
</html>
"""
    (HERE/'verification-report.html').write_text(page, encoding='utf-8')
    print('wrote verification-report/verification-report.html')


if __name__ == '__main__':
    build_tex()
    build_html()
