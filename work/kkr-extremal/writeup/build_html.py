"""Build kkr-extremal-writeup.html from kkr-extremal-writeup.tex (same source as the PDF).

Handles only the LaTeX subset the write-up uses: sections, paragraphs, lists,
equations (left to MathJax, numbered with tags='ams' as in LaTeX), figures
(PNG embedded as base64), tables, verbatim, citations and references. The
numbers come from ../report/numbers.tex, as in the PDF.

Run: python build_html.py   ->  kkr-extremal-writeup.html
"""
import base64
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE/'kkr-extremal-writeup.tex'
OUT = HERE/'kkr-extremal-writeup.html'
NUMBERS = HERE.parent/'report'/'numbers.tex'

MATH_ENVS = ('equation', 'multline', 'align', 'gather')


def braced(text, start):
    """Content of the {...} group opening at text[start] and the index after it."""
    assert text[start] == '{', text[start:start+40]
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == '\\':
            continue
        if c == '{' and text[i-1] != '\\':
            depth += 1
        elif c == '}' and text[i-1] != '\\':
            depth -= 1
            if depth == 0:
                return text[start+1:i], i+1
    raise ValueError('unbalanced braces')


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', line) for line in text.splitlines())


def load_macros():
    text = NUMBERS.read_text()
    out = {}
    for m in re.finditer(r'\\newcommand\{\\(R\w+)\}', text):
        body, _ = braced(text, m.end())
        out[m.group(1)] = strip_comments(body).strip()
    return out


def expand_macros(text, macros):
    for name in sorted(macros, key=len, reverse=True):
        text = re.sub(r'\\'+name+r'(?![A-Za-z])', lambda _m, v=macros[name]: v, text)
    return text


# ---------------------------------------------------------------- numbering
def collect_labels(body):
    """Section, figure and table numbers in source order, as LaTeX assigns them."""
    labels, sec, sub, fig, tab, appendix = {}, 0, 0, 0, 0, False
    current = None
    for m in re.finditer(r'\\(section|subsection|appendix|caption|label)\b|\\begin\{(figure|table)\}', body):
        kind = m.group(1) or m.group(2)
        if kind == 'appendix':
            appendix, sec = True, 0
        elif kind == 'section':
            sec, sub = sec+1, 0
            current = chr(64+sec) if appendix else str(sec)
        elif kind == 'subsection':
            sub += 1
            current = f'{chr(64+sec) if appendix else sec}.{sub}'
        elif kind == 'figure':
            fig += 1
            current = str(fig)
        elif kind == 'table':
            tab += 1
            current = str(tab)
        elif kind == 'label':
            name, _ = braced(body, m.end())
            if not name.startswith('eq:'):
                labels[name] = current
    return labels


# ---------------------------------------------------------------- inline text
ACCENTS = {r'\"u': 'ü', r'\"o': 'ö', r'\"a': 'ä', r"\'e": 'é'}


def inline(text, labels, cites):
    """Convert a run of LaTeX text that contains no display math."""
    # font commands first (their arguments may contain math): sentinels survive escaping
    for cmd, tag in (('emph', 'em'), ('textbf', 'strong'), ('texttt', 'code')):
        while f'\\{cmd}{{' in text:
            i = text.index(f'\\{cmd}{{')
            body, j = braced(text, i+len(cmd)+1)
            text = text[:i] + f'\x01{tag}\x02{body}\x01/{tag}\x02' + text[j:]
    text = re.sub(r'\\eqref\{([^}]*)\}', r'$\\eqref{\1}$', text)
    text = text.replace('<br>', '\x01br\x02')
    parts = re.split(r'(\$[^$]+\$)', text)
    out = []
    for k, part in enumerate(parts):
        if k % 2:
            out.append(html.escape(part, quote=False))
            continue
        out.append(inline_text(part, labels, cites))
    return ''.join(out).replace('\x01', '<').replace('\x02', '>')


def inline_text(t, labels, cites):
    for a, b in ACCENTS.items():
        t = t.replace(a, b)
    t = html.escape(t, quote=False)
    t = re.sub(r'\\ref\{([^}]*)\}', lambda m: f'<a href="#{m.group(1)}">{labels.get(m.group(1), "?")}</a>', t)
    t = re.sub(r'\\cite\{([^}]*)\}', lambda m: '[' + ', '.join(
        f'<a href="#ref-{c.strip()}">{cites.get(c.strip(), "?")}</a>' for c in m.group(1).split(',')) + ']', t)
    t = t.replace('``', '“').replace("''", '”').replace('---', '—').replace('--', '–')
    t = t.replace('~', '&nbsp;').replace(r'\,', '&thinsp;').replace('\\\n', ' ').replace(r'\ ', ' ')
    t = t.replace(r'\dots', '…').replace(r'\_', '_').replace(r'\%', '%').replace(r'\&', '&amp;')
    t = re.sub(r'\\(small|footnotesize|centering|noindent)\b', '', t)
    t = t.replace('{', '').replace('}', '')
    return t


# ---------------------------------------------------------------- blocks
def image_tag(path):
    png = (HERE/path).with_suffix('.png')
    data = base64.b64encode(png.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(png.stem)}">'


def convert_table(tab, labels, cites):
    body = re.sub(r'\\(toprule|midrule|bottomrule)', '', tab)
    rows = [r.strip() for r in re.split(r'\\\\', body) if r.strip()]
    out = ['<div class="table-wrap"><table>']
    for i, r in enumerate(rows):
        cells = [c.strip() for c in r.split('&')]
        tag = 'th' if i == 0 else 'td'
        out.append('<tr>' + ''.join(f'<{tag}>{inline(c, labels, cites)}</{tag}>' for c in cells) + '</tr>')
    out.append('</table></div>')
    return '\n'.join(out)


def tabular_body(env):
    i = env.index(r'\begin{tabular}')
    j = env.index('{', i+len(r'\begin{tabular}'))
    _, k = braced(env, j)
    return env[k:env.index(r'\end{tabular}')]


def convert(body, labels, cites, title):
    out = []
    pos = 0
    pattern = re.compile(
        r'\\begin\{(abstract|figure|table|itemize|enumerate|verbatim|center|thebibliography|'
        + '|'.join(MATH_ENVS) + r')\}(\[[^\]]*\])?(\{[^}]*\})?'
        r'|\\(section|subsection|paragraph)\*?\{|\\maketitle|\\tableofcontents|\\appendix'
        r'|\{\\footnotesize\s*|\\end\{verbatim\}\}')
    para = []

    def flush():
        text = ''.join(para).strip()
        para.clear()
        for chunk in re.split(r'\n\s*\n', text):
            chunk = chunk.strip()
            if chunk:
                out.append(f'<p>{inline(chunk, labels, cites)}</p>')

    state = dict(sec=0, sub=0, appendix=False)
    while True:
        m = pattern.search(body, pos)
        if not m:
            para.append(body[pos:])
            flush()
            break
        para.append(body[pos:m.start()])
        token = m.group(0)
        env = m.group(1)
        if token == r'\maketitle':
            flush()
            out.append(title)
            pos = m.end()
        elif token == r'\tableofcontents':
            flush()
            out.append('<!--TOC-->')
            pos = m.end()
        elif token == r'\appendix':
            flush()
            state.update(appendix=True, sec=0)
            pos = m.end()
        elif token.startswith(r'{\footnotesize') or token == r'\end{verbatim}}':
            pos = m.end() - (len(r'\end{verbatim}}') if token == r'\end{verbatim}}' else 0)
            if token == r'\end{verbatim}}':
                pos = m.end()
        elif m.group(4):
            flush()
            kind = m.group(4)
            heading, end = braced(body, m.end()-1)
            label = re.match(r'\s*\\label\{([^}]*)\}', body[end:])
            anchor = label.group(1) if label else None
            if label:
                end += label.end()
            if kind == 'paragraph':
                # run-in heading: attach to the following paragraph text
                para.append(f'\\textbf{{{heading}}} ')
            else:
                if kind == 'section':
                    state['sec'] += 1
                    state['sub'] = 0
                    num = chr(64+state['sec']) if state['appendix'] else str(state['sec'])
                    tag = 'h2'
                else:
                    state['sub'] += 1
                    sec = chr(64+state['sec']) if state['appendix'] else state['sec']
                    num = f'{sec}.{state["sub"]}'
                    tag = 'h3'
                ident = anchor or 'sec-' + num.replace('.', '-')
                prefix = 'Appendix ' + num if (state['appendix'] and tag == 'h2') else num
                out.append(f'<{tag} id="{ident}"><span class="num">{prefix}</span> {inline(heading, labels, cites)}</{tag}>')
            pos = end
        else:
            flush()
            end_tag = f'\\end{{{env}}}'
            stop = body.index(end_tag, m.end())
            inner = body[m.end():stop]
            pos = stop+len(end_tag)
            if env in MATH_ENVS:
                out.append('<div class="eq">' + html.escape(f'\\begin{{{env}}}{inner}{end_tag}', quote=False) + '</div>')
            elif env == 'abstract':
                out.append('<section class="abstract"><h2 class="abs">Abstract</h2>'
                           + convert(inner, labels, cites, title) + '</section>')
            elif env == 'figure':
                path = re.search(r'\\includegraphics(\[[^\]]*\])?\{([^}]*)\}', inner).group(2)
                ci = inner.index(r'\caption{')
                caption, _ = braced(inner, ci+len(r'\caption'))
                lab = re.search(r'\\label\{([^}]*)\}', inner).group(1)
                out.append(f'<figure id="{lab}">{image_tag(path)}<figcaption><strong>Figure {labels[lab]}.</strong> '
                           f'{inline(caption, labels, cites)}</figcaption></figure>')
            elif env == 'table':
                ci = inner.index(r'\caption{')
                caption, _ = braced(inner, ci+len(r'\caption'))
                lab = re.search(r'\\label\{([^}]*)\}', inner).group(1)
                out.append(f'<figure class="tab" id="{lab}"><figcaption><strong>Table {labels[lab]}.</strong> '
                           f'{inline(caption, labels, cites)}</figcaption>{convert_table(tabular_body(inner), labels, cites)}</figure>')
            elif env == 'center':
                if r'\begin{tabular}' in inner:
                    out.append(convert_table(tabular_body(inner), labels, cites))
                else:
                    out.append('<div class="center">' + convert(inner, labels, cites, title) + '</div>')
            elif env in ('itemize', 'enumerate'):
                tag = 'ul' if env == 'itemize' else 'ol'
                items = re.split(r'\\item\b', inner)[1:]
                out.append(f'<{tag}>' + ''.join(f'<li>{convert(it, labels, cites, title)}</li>' for it in items) + f'</{tag}>')
            elif env == 'verbatim':
                out.append('<pre>' + html.escape(inner.strip('\n')) + '</pre>')
                if body[pos:pos+1] == '}':
                    pos += 1
            elif env == 'thebibliography':
                items = re.split(r'\\bibitem\{([^}]*)\}', inner)[1:]
                refs = [f'<li id="ref-{items[i]}">{inline(items[i+1].strip(), labels, cites)}</li>'
                        for i in range(0, len(items), 2)]
                out.append('<h2 id="references">References</h2><ol class="refs">' + ''.join(refs) + '</ol>')
    html_out = '\n'.join(out)
    # paragraphs that only wrapped a run-in heading inside list items: unwrap single <p>
    return html_out


def toc(doc):
    items = re.findall(r'<(h2|h3) id="([^"]+)"><span class="num">([^<]*)</span> (.*?)</\1>', doc)
    lines = ['<nav class="toc"><h2 class="abs">Contents</h2><ul>']
    for tag, ident, num, text in items:
        cls = 'sub' if tag == 'h3' else ''
        lines.append(f'<li class="{cls}"><a href="#{ident}"><span class="num">{num}</span> {text}</a></li>')
    lines.append('<li><a href="#references">References</a></li></ul></nav>')
    return '\n'.join(lines)


CSS = """
:root{--bg:#fbfaf7;--fg:#1d2125;--muted:#5d6670;--rule:#dcdad4;--accent:#8a2f17;--card:#ffffff;
--code:#f1efe9;--link:#24507a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181b;--fg:#e4e6e8;--muted:#9aa3ab;
--rule:#30363c;--accent:#e08a6b;--card:#f7f7f5;--code:#22272c;--link:#8fb8e0;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15181b;--fg:#e4e6e8;--muted:#9aa3ab;--rule:#30363c;--accent:#e08a6b;
--card:#f7f7f5;--code:#22272c;--link:#8fb8e0;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:"Iowan Old Style","Palatino Linotype",Palatino,
"Book Antiqua",Georgia,serif;font-size:17px;line-height:1.6}
main{max-width:46rem;margin:0 auto;padding:2.5rem 16px 4rem}
header.title{text-align:center;margin-bottom:2rem}
header.title h1{font-size:1.75rem;line-height:1.25;font-weight:600;margin:0 0 .8rem;text-wrap:balance}
header.title .author{color:var(--muted);font-size:.95rem}
h2{font-size:1.3rem;margin:2.4rem 0 .6rem;font-weight:600}
h3{font-size:1.08rem;margin:1.8rem 0 .4rem;font-weight:600}
h2 .num,h3 .num{color:var(--accent);margin-right:.35rem}
h2.abs{font-size:1rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 .5rem}
.abstract{border-left:3px solid var(--accent);padding:.2rem 0 .2rem 1rem;margin:1.5rem 0;font-size:.97rem}
.toc{margin:1.5rem 0 2rem;font-size:.95rem}
.toc ul{list-style:none;padding:0;margin:0;columns:2;column-gap:2rem}
.toc li{break-inside:avoid;margin:.1rem 0}.toc li.sub{padding-left:1.4rem;font-size:.9rem}
.toc a{color:var(--fg);text-decoration:none}.toc a:hover{color:var(--link)}
.toc .num{color:var(--accent);display:inline-block;min-width:1.8rem}
a{color:var(--link)}
p{margin:.7rem 0}
.eq{overflow-x:auto;overflow-y:hidden;margin:.6rem 0}
figure{margin:1.8rem 0;padding:0}
figure img{display:block;width:100%;height:auto;background:var(--card);border-radius:4px;padding:6px}
figcaption{font-size:.9rem;color:var(--muted);margin-top:.5rem;line-height:1.45}
figure.tab figcaption{margin:0 0 .6rem}
.table-wrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;font-size:.82rem;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:.3rem .45rem;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
td:last-child{white-space:normal;min-width:12rem}
th{font-weight:600;border-bottom:1.5px solid var(--fg)}
code{font-family:"JetBrains Mono","SFMono-Regular",Menlo,Consolas,monospace;font-size:.86em;
background:var(--code);padding:.05rem .3rem;border-radius:3px}
pre{font-size:.78rem;background:var(--code);padding:.8rem 1rem;border-radius:4px;overflow-x:auto;line-height:1.45}
ol.refs{font-size:.92rem;padding-left:1.6rem}ol.refs li{margin:.35rem 0}
ul,ol{padding-left:1.5rem}li{margin:.25rem 0}li>p:first-child{margin-top:0}li>p:last-child{margin-bottom:0}
mjx-container[display="true"]{margin:.4em 0!important}
@media (max-width:600px){body{font-size:16px}.toc ul{columns:1}header.title h1{font-size:1.4rem}}
"""

MATHJAX = """<script>
window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['\\\\[','\\\\]']],tags:'ams',
macros:{ext:'\\\\mathrm{ext}'}},chtml:{scale:1.0},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""


def main():
    raw = strip_comments(SRC.read_text())
    macros = load_macros()
    raw = expand_macros(raw, macros)
    preamble, body = raw.split(r'\begin{document}', 1)
    body = body.split(r'\end{document}', 1)[0]
    title_tex, _ = braced(preamble, preamble.index(r'\title{')+len(r'\title'))
    author_tex, _ = braced(preamble, preamble.index(r'\author{')+len(r'\author'))
    date_tex, _ = braced(preamble, preamble.index(r'\date{')+len(r'\date'))
    labels = collect_labels(body)
    keys = re.findall(r'\\bibitem\{([^}]*)\}', body)
    cites = {k: str(i+1) for i, k in enumerate(keys)}
    title_text = inline(title_tex.replace('\\\\', ' '), labels, cites)
    author = inline(re.sub(r'\\\\(\[[^\]]*\])?', '<br>', author_tex), labels, cites)
    title = (f'<header class="title"><h1>{title_text}</h1>'
             f'<div class="author">{author}<br>{inline(date_tex, labels, cites)}</div></header>')
    doc = convert(body, labels, cites, title)
    doc = doc.replace('<!--TOC-->', toc(doc))
    plain_title = re.sub(r'<[^>]+>', '', title_text)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Extremal EGB Black Holes</title>
<meta name="description" content="{html.escape(plain_title)}">
<style>{CSS}</style>
{MATHJAX}
</head><body><main>
{doc}
</main></body></html>
"""
    OUT.write_text(page)
    print(f'wrote {OUT.name}: {len(page)/1e6:.2f} MB, {len(labels)} labels, {len(cites)} references')


if __name__ == '__main__':
    main()
