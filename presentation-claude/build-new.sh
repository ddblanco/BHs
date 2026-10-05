#!/usr/bin/env bash
# Build the new-motivation version: slides-new.pdf, notes-new.pdf, slides-new.html, notes-new.html.
# Same figures as build.sh; run ./build.sh once first if figures/retro-crt.jpg is missing.
set -euo pipefail
cd "$(dirname "$0")"
for t in slides-new notes-new; do
  if command -v pdflatex >/dev/null; then
    pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$t.tex" | grep 'Output written'
  else
    tectonic -X compile "$t.tex" 2>&1 | grep -E '^error|Writing'
  fi
done
python3 build_html.py -new
rm -f ./*.aux ./*.log ./*.nav ./*.out ./*.snm ./*.toc ./*.vrb
