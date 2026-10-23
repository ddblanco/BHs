"""Document checks only. Does not import or execute any physics code."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[2]
src = root / 'work/jhep-paper'
text = (src / 'main.tex').read_text()
keys = set(re.findall(r'\\bibitem\{([^}]+)\}', text))
cites = {key for group in re.findall(r'\\cite\{([^}]+)\}', text) for key in group.split(',')}
assert cites == keys, (cites - keys, keys - cites)
labels = re.findall(r'\\label\{([^}]+)\}', text)
assert len(labels) == len(set(labels)), 'duplicate labels'
refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text))
assert refs <= set(labels), refs - set(labels)
source_prefixes = ['1010.0860', '2009.00015', '2303.12471', 'hep-th-9305016',
                   'gr-qc-9307038', '1005.5053', '2404.16981', '1201.4994', '2303.07358']
assert all(list((root / 'papers').glob(p + '*.pdf')) for p in source_prefixes)
print('PASS: 9 cited papers have local PDFs; citations and cross-references resolve')
for name, target in [('main', 'jhep-extremal-mass'), ('referee-report', 'jhep-extremal-mass-referee')]:
    log = (src / f'{name}.log').read_text()
    assert not re.search(r'Overfull|Underfull|undefined|LaTeX Error|Fatal error', log, re.I), name
    info = subprocess.check_output(['pdfinfo', str(src / f'{name}.pdf')], text=True)
    pages = re.search(r'^Pages:\s+(\d+)', info, re.M).group(1)
    assert (src / f'{name}.pdf').read_bytes() == (root / f'output/pdf/{target}.pdf').read_bytes()
    print(f'PASS: {name}: {pages} pages; no box overflow or unresolved references; delivery copy matches')
print('NOTE: official JHEP style emits one harmless deprecated pagecolor warning')
print('NOT CHECKED: physics equations, numerical data, fits, horizon regularity, stability or uniqueness')
print('NO NEW PHYSICS CALCULATIONS PERFORMED')
