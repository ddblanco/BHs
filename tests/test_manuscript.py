"""Check the current LaTeX paper against independently read production data."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def regenerated(tmp_path_factory):
    root = tmp_path_factory.mktemp('paper')
    for name in ('manuscript/make_figures.py', 'manuscript/reanalysis.py',
                 'results/egb-extremality.json',
                 'results/egb-rotating-profiles.json',
                 'results/egb-rotating-resolution-study.json'):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    subprocess.run([sys.executable, str(root/'manuscript/make_figures.py')],
                   cwd=root, check=True, capture_output=True, text=True)
    return root/'manuscript'


def test_current_manuscript_macros_and_citations_resolve():
    subprocess.run([sys.executable, str(ROOT/'manuscript/check_manuscript.py')],
                   cwd=ROOT, check=True, capture_output=True, text=True)


def test_saved_numbers_equal_regenerated_numbers(regenerated):
    assert (ROOT/'manuscript/numbers.tex').read_text() == (regenerated/'numbers.tex').read_text()


def test_solution_table_contains_the_actual_observables(regenerated):
    def rows(path):
        with path.open(encoding='utf-8') as stream:
            return list(csv.DictReader(line for line in stream if not line.startswith('#')))
    saved = rows(ROOT/'manuscript/supplementary/solutions.csv')
    assert saved == rows(regenerated/'supplementary/solutions.csv')
    data = json.loads((ROOT/'results/egb-extremality.json').read_text())
    states = sorted((s for group in data['walks'].values() for s in group),
                    key=lambda s: (s['alpha_gb'], s['omega_h']))
    assert len(saved) == len(states)
    for row, state in zip(saved, states):
        for key, value in row.items():
            assert float(value) == state[key]


def test_sensitivity_audit_regenerates_and_encloses_alternatives(regenerated):
    path = 'supplementary/extrapolation-sensitivity.json'
    saved = json.loads((ROOT/'manuscript'/path).read_text())
    rebuilt = json.loads((regenerated/path).read_text())
    assert saved == rebuilt
    for name, expected in saved['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    for fit in saved['fits']:
        assert all(abs(value-fit['central']) <= fit['envelope']+1e-12
                   for value in fit['intercepts'].values())


def test_current_paper_preserves_external_anchors_and_limitation():
    text = (ROOT/'manuscript/main.tex').read_text(encoding='utf-8')
    for identifier in ('1010.0860', '2303.12471', '2009.00015'):
        assert identifier in text
    assert '1.102101' in text and '1.104' in text
    assert not re.search(r'\bnan\b|\binf\b|TODO|XXX|FIXME', text, re.I)
    assert not re.search(r'\\emailAdd\{[^}]+\}', text)


def test_current_pdf_and_generated_figures_are_present(regenerated):
    for name in ('main.pdf', 'figures/fig-profiles.pdf',
                 'figures/fig-potential.pdf', 'figures/fig-shift.pdf',
                 'figures/fig-entropy.pdf', 'figures/fig-extrapolation.pdf',
                 'figures/fig-resolution.pdf'):
        data = (ROOT/'manuscript'/name).read_bytes()
        assert data.startswith(b'%PDF-') and len(data) > 10_000
        if name.startswith('figures/'):
            assert (regenerated/name).read_bytes().startswith(b'%PDF-')
