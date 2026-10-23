#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p qa ../../output/pdf
for name in main referee-report; do
  for pass in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" > "qa/$name-build-$pass.txt"
  done
done
cp main.pdf ../../output/pdf/jhep-extremal-mass.pdf
cp referee-report.pdf ../../output/pdf/jhep-extremal-mass-referee.pdf
printf 'BUILD PASS: paper and referee PDFs compiled (3 passes each)\n'
