"""Snapshots for slide 3: the first roadmap file and a passage of arXiv:1010.0860.

  fig-roadmap.png  <- ruta-de-trabajo-inicial.txt, lines 52-62 and 76-84, as an editor view
  fig-smarr.png    <- 1010.0860, printed page 14 (PDF page 15): start of section 4.1.3
                      and footnote 14, rendered from the PDF, highlights placed from the
                      word boxes that pdftotext reports (nothing is retyped)

Needs Pillow, pdftoppm and pdftotext.  Run from this folder: python3 make_snapshots.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT = HERE / "figures"
PAPER = HERE.parent / "papers" / "1010.0860.pdf"
ACCENT = (31, 95, 168)
WARM = (217, 98, 43)
HIGHLIGHT = (253, 226, 199)
MUTED = (107, 114, 128)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def roadmap():
    lines = (HERE / "ruta-de-trabajo-inicial.txt").read_text().splitlines()
    shown = [*range(52, 63), None, *range(76, 85)]          # 1-based; None = elision
    marked = {54, 55, 56, 81}                                  # "minimum success", "only after Hito 4"
    assert lines[51].startswith("## Hito 4") and lines[75].startswith("## Hito 6")
    assert "Salida mínima exitosa" in lines[53] and "después del Hito 4" in lines[80]

    size, lh, pad, gutter = 30, 42, 28, 86
    font, bold = ImageFont.truetype(MONO, size), ImageFont.truetype(MONO_BOLD, size)
    cw = font.getlength("M")
    width = int(pad + gutter + cw * max(len(lines[n - 1]) for n in shown if n) + pad)
    bar = 58
    im = Image.new("RGB", (width, bar + 2 * pad // 2 + lh * len(shown) + pad // 2), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, bar], fill=(236, 239, 243))
    d.text((pad, bar / 2), "ruta-de-trabajo-inicial.txt", font=ImageFont.truetype(SANS, 28),
           fill=(55, 65, 81), anchor="lm")
    d.text((width - pad, bar / 2), "5 Sept 2026", font=ImageFont.truetype(SANS, 24),
           fill=MUTED, anchor="rm")
    y = bar + pad // 2
    for n in shown:
        if n is None:
            d.text((pad + gutter, y + lh / 2), "···  (Hito 5 — método híbrido asistido por IA)",
                   font=font, fill=MUTED, anchor="lm")
        else:
            text = lines[n - 1]
            if n in marked:
                d.rectangle([pad + gutter - 6, y + 2, pad + gutter + cw * len(text) + 6, y + lh - 2],
                            fill=HIGHLIGHT)
            d.text((pad + gutter - 22, y + lh / 2), str(n), font=font, fill=(156, 163, 175), anchor="rm")
            head = text.startswith("#")
            d.text((pad + gutter, y + lh / 2), text, font=bold if head else font,
                   fill=ACCENT if head else (31, 41, 55), anchor="lm")
        y += lh
    d.rectangle([0, 0, width - 1, im.size[1] - 1], outline=(209, 213, 219), width=3)
    im.save(OUT / "fig-roadmap.png")


def word_boxes(page):
    xml = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", str(PAPER), "-"],
                         capture_output=True, text=True, check=True).stdout
    return [(float(a), float(b), float(c), float(e), w) for a, b, c, e, w in re.findall(
        r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', xml)]


def span(words, first, last, after=0.0):
    """Boxes of the words from `first` to `last` (inclusive), first occurrence below y=after."""
    i = next(k for k, w in enumerate(words) if w[4] == first and w[1] > after)
    j = next(k for k in range(i, len(words)) if words[k][4] == last)
    return words[i:j + 1]


def smarr():
    page, dpi = 15, 400                                         # printed page "- 14 -"
    words = word_boxes(page)
    text = " ".join(w[4] for w in words)
    assert "4.1.3 Thermodynamical properties" in text
    assert "The Smarr relation in Einstein gravity" in text
    s = dpi / 72
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-hide-annotations", "-r", str(dpi), "-f", str(page), "-l", str(page),
                        "-singlefile", str(PAPER), f"{tmp}/p"], check=True)
        full = Image.open(f"{tmp}/p.png").convert("RGB")

    marks = span(words, "Also,", ".", after=550) + span(words, "It", "solutions.", after=700)
    over = Image.new("RGBA", full.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    rows = []                                                  # one bar per printed line
    for x0, y0, x1, y1, _ in marks:
        c = (y0 + y1) / 2                                      # word centres of one line agree to < 1 pt
        row = next((r for r in rows if abs(r[4] - c) < 5), None)
        if row is None:
            rows.append([x0, c - 5.5, x1, c + 5.5, c])
        else:
            row[0], row[2] = min(row[0], x0), max(row[2], x1)
    for x0, y0, x1, y1, _ in rows:
        d.rectangle([x0 * s - 4, y0 * s, x1 * s + 4, y1 * s], fill=(*WARM, 60))
    full = Image.alpha_composite(full.convert("RGBA"), over).convert("RGB")

    x0, x1 = 76, 519
    top = full.crop((round(x0 * s), round(435 * s), round(x1 * s), round(586 * s)))
    foot = full.crop((round(x0 * s), round(682 * s), round(x1 * s), round(727 * s)))
    gap = 70
    im = Image.new("RGB", (top.size[0], top.size[1] + gap + foot.size[1]), "white")
    im.paste(top, (0, 0))
    im.paste(foot, (0, top.size[1] + gap))
    d = ImageDraw.Draw(im)
    y = top.size[1] + gap // 2
    for x in range(0, im.size[0], 28):
        d.line([x, y, x + 14, y], fill=(190, 190, 190), width=3)
    d.rectangle([0, 0, im.size[0] - 1, im.size[1] - 1], outline=(209, 213, 219), width=4)
    im.save(OUT / "fig-smarr.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    roadmap()
    smarr()
    for f in ("fig-roadmap.png", "fig-smarr.png"):
        print(f, Image.open(OUT / f).size)
