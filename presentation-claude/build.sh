#!/usr/bin/env bash
# Rebuild everything in this folder: figures, slides.pdf, notes.pdf, slides.html, notes.html.
# Needs python3 with matplotlib and Pillow, pdflatex (beamer, metropolis), pdftoppm and pdftotext.
set -euo pipefail
cd "$(dirname "$0")"

# Figures copied from the project, so this folder stands on its own.
cp ../short-summary/fig-paper-comparison.pdf figures/
for f in profiles potential shift massext; do cp "../manuscript/figures/fig-$f.pdf" figures/; done
for f in paper-comparison profiles potential shift massext; do
  pdftoppm -png -r 220 -singlefile "figures/fig-$f.pdf" "figures/fig-$f"
done
# Left panel of fig-massext (the extremal mass curve); slides.tex trims the PDF the same way.
python3 -c "from PIL import Image; im=Image.open('figures/fig-massext.png'); im.crop((0,0,round(im.size[0]*662/1298),im.size[1])).save('figures/fig-massext-left.png')"
# Left panel of fig-paper-comparison (our curves against the published vertices), for backup slide A1;
# slides.tex trims the PDF at the same place (215 of 432 bp).
python3 -c "from PIL import Image; im=Image.open('figures/fig-paper-comparison.png'); im.crop((0,0,round(im.size[0]*215/432),im.size[1])).save('figures/fig-paper-comparison-left.png')"
python3 make_figures.py
python3 make_snapshots.py   # backup slide A5: roadmap file and 1010.0860 section 4.1.3 / footnote 14

for t in slides notes; do
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" | grep 'Output written'
done
python3 build_html.py
rm -f ./*.aux ./*.log ./*.nav ./*.out ./*.snm ./*.toc ./*.vrb
