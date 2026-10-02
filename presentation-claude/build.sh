#!/usr/bin/env bash
# Rebuild the showstopper version: slides.pdf, notes.pdf, slides.html, notes.html.
# Uses the figures in figures/ (make_figures.py and make_snapshots.py write there).
set -euo pipefail
cd "$(dirname "$0")"
# Thank-you background: JPEG copy of the PNG in this folder (smaller, no spaces in the name).
python3 -c "from PIL import Image; Image.open('Retro CRT Black Hole with AI Motifs.png').convert('RGB').save('figures/retro-crt.jpg', quality=88)"
for t in slides notes; do
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" | grep 'Output written'
done
python3 build_html.py
rm -f ./*.aux ./*.log ./*.nav ./*.out ./*.snm ./*.toc ./*.vrb
