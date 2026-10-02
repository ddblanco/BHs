"""Build the two self-contained HTML files from the LaTeX sources.

  slides.html  <- src/slides.src.html, figures inlined, speaker notes from notes.tex
  notes.html   <- notes.tex, with slide thumbnails rendered from slides.pdf

notes.tex is the single source of the speaker script; this converts only the
small LaTeX subset that file uses and fails loudly on anything else.
"""

import base64
import html
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

MATH = {r"\sqrt{-g}": "√(−g)", r"\int": "∫", r"\nu": "ν", r"\rho": "ρ", r"\sigma": "σ", r"\alpha": "α", r"\Psi": "Ψ", r"\pi": "π", r"\Omega": "Ω", r"\omega": "ω", r"\mu": "μ", r"\chi": "χ",
        r"\kappa": "κ", r"\times": "×", r"\approx": "≈", r"\to": "→", r"\ge": "≥", r"\le": "≤",
        r"\sim": "~", r"\ne": "≠", r"\rm ": "", r"\,": " "}


def math(m):
    s = m.group(1)
    for k, v in MATH.items():
        s = s.replace(k, v)
    s = s.replace("-", "−")
    s = re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(\w)", r"<sup>\1</sup>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    return f'<span class="m">{s}</span>'


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\$(.+?)\$", math, t)
    t = re.sub(r"\\emph\{([^}]*)\}", r"<em>\1</em>", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", t)
    t = t.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    t = t.replace("~", "\u00a0").replace(r"\%", "%").replace("\\ ", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if "\\" in t or "{" in t:
        raise ValueError(f"unconverted LaTeX in: {t[:120]}")
    return t


def paras(block):
    return "".join(f"<p>{inline(p)}</p>" for p in re.split(r"\n\s*\n", block) if p.strip())


def parse_notes():
    tex = (HERE / "notes.tex").read_text()
    body = tex[tex.index(r"\slidenote{1}"):tex.index(r"\newpage")]
    parts = re.split(r"\\slidenote\{(\d+)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", body)[1:]
    notes = []
    backup_intro = ""
    for k in range(0, len(parts), 5):
        num, label, title, time, text = parts[k:k + 5]
        if r"\clearpage" in text:   # the backup heading sits between the last timed slide and A1
            text, rest = text.split(r"\clearpage")
            backup_intro = inline(re.search(r"\{\\color\{muted\}(.*?)\\par\}", rest, re.S).group(1))
        notes.append({"page": int(num), "label": label, "title": inline(title), "time": inline(time.replace(r"\quad", " ")),
                      "html": paras(text)})
    intro = tex[tex.index(r"{\color{muted}Five minutes"):tex.index(r"\slidenote{1}")]
    intro = inline(re.sub(r"^\{\\color\{muted\}|\\par\}\s*$", "", intro.strip()))
    notes[0]["backup_intro"] = backup_intro
    qa = tex[tex.index(r"\begin{description}"):tex.index(r"\end{description}")]
    items = re.findall(r"\\item\[(.*?)\]\s*(.*?)(?=\\item\[|$)", qa, re.S)
    qa_html = "".join(f"<dt>{inline(q)}</dt><dd>{inline(a)}</dd>" for q, a in items)
    credit = tex[tex.index(r"\section*{On crediting the AI}") + len(r"\section*{On crediting the AI}"):
                 tex.index(r"\end{document}")]
    return notes, intro, qa_html, paras(credit)


def data_uri(path, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def build_slides(notes):
    src = (HERE / "src" / "slides.src.html").read_text()
    src = re.sub(r'src="(figures/[^"]+\.png)"', lambda m: f'src="{data_uri(HERE / m.group(1))}"', src)
    src = re.sub(r'src="(figures/[^"]+\.jpg)"', lambda m: f'src="{data_uri(HERE / m.group(1), "image/jpeg")}"', src)
    assert "/*NOTES*/[]" in src
    src = src.replace("/*NOTES*/[]", json.dumps(notes, ensure_ascii=False))
    n_slides = src.count('<section class="slide')
    assert n_slides == len(notes), (n_slides, len(notes))
    (HERE / "slides.html").write_text(src)
    return n_slides


def thumbnails(n):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-png", "-r", "40", str(HERE / "slides.pdf"), f"{d}/t"], check=True)
        files = sorted(Path(d).glob("t-*.png"))
        return [data_uri(f) for f in files]


NOTES_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Speaker Script</title>
<style>
:root{--ink:#1f2937;--muted:#6b7280;--accent:#1f5fa8;--warm:#d9622b;--bg:#ffffff;--card:#f7f8fa;--line:#e5e7eb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#e5e7eb;--muted:#9ca3af;--accent:#7fb0ea;--warm:#f08a5a;--bg:#15171c;--card:#1d2027;--line:#2c313a}}
:root[data-theme="dark"]{--ink:#e5e7eb;--muted:#9ca3af;--accent:#7fb0ea;--warm:#f08a5a;--bg:#15171c;--card:#1d2027;--line:#2c313a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.6 Georgia,"Times New Roman",serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}
h1{font:700 32px/1.2 "Helvetica Neue",Arial,sans-serif;margin:0 0 4px}
.sub{font-size:19px;margin:0 0 6px}
.intro{color:var(--muted);font-size:16px;margin:0 0 28px}
section{border-top:1px solid var(--line);padding:22px 0 8px}
.head{display:flex;gap:18px;align-items:center;margin-bottom:10px}
.head img{width:210px;max-width:40%;border:1px solid var(--line);border-radius:4px;background:#fff}
.head h2{font:700 21px/1.25 "Helvetica Neue",Arial,sans-serif;color:var(--accent);margin:0}
.head .t{font:14px "Helvetica Neue",Arial,sans-serif;color:var(--muted);margin-top:4px}
p{margin:0 0 12px}
.m{font-style:italic}
h2.x{font:700 21px/1.25 "Helvetica Neue",Arial,sans-serif;margin:34px 0 10px}
dl{margin:0}dt{font-weight:700;margin-top:12px}dd{margin:2px 0 0 0}
@media (max-width:560px){.head{flex-direction:column;align-items:flex-start}.head img{max-width:100%;width:100%}}
@media print{body{font-size:12pt}section{break-inside:avoid}}
</style></head><body><main>
<h1>Speaker script</h1>
<p class="sub">Extremality shift of rotating black holes at finite Gauss–Bonnet coupling</p>
<p class="intro">__INTRO__</p>
__SECTIONS__
<h2 class="x">If there are questions</h2>
<dl>__QA__</dl>
<h2 class="x">On crediting the AI</h2>
__CREDIT__
</main></body></html>
"""


def build_notes(notes, intro, qa, credit, thumbs):
    secs = ""
    for k, n in enumerate(notes):
        if n["label"] == "A1":
            secs += f'<h2 class="x">Backup slides</h2><p class="intro">{notes[0]["backup_intro"]}</p>'
        secs += (f'<section><div class="head"><img src="{thumbs[n["page"] - 1]}" alt="Slide {n["label"]}">'
                 f'<div><h2>{n["label"]}. {n["title"]}</h2><div class="t">{n["time"]}</div></div></div>'
                 f'{n["html"]}</section>')
    page = (NOTES_PAGE.replace("__INTRO__", intro).replace("__SECTIONS__", secs)
            .replace("__QA__", qa).replace("__CREDIT__", credit))
    (HERE / "notes.html").write_text(page)


if __name__ == "__main__":
    notes, intro, qa, credit = parse_notes()
    n = build_slides(notes)
    build_notes(notes, intro, qa, credit, thumbnails(n))
    for f in ("slides.html", "notes.html"):
        print(f, f"{(HERE / f).stat().st_size / 1e6:.2f} MB")
