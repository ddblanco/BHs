#!/usr/bin/env bash
# Rebuild the short version: slides.pdf, notes.pdf, slides.html, notes.html.
# Uses the figures in ../figures; run ../build.sh first if they changed.
set -euo pipefail
cd "$(dirname "$0")"
for t in slides notes; do
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" | grep 'Output written'
done
python3 build_html.py
rm -f ./*.aux ./*.log ./*.nav ./*.out ./*.snm ./*.toc ./*.vrb
