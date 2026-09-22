"""Independent negative control for the external image extraction."""
from pathlib import Path
import pytest
from experiments.egb_rotating_paper_profiles import extract
from rotating_bh.myers_perry import MyersPerry


IMAGE = Path(__file__).resolve().parents[1]/'references/data/1010.0860v1-figure-1b.png'
pytestmark = pytest.mark.skipif(not IMAGE.is_file(),
    reason='Optional third-party image not distributed; see references/data/README.md')

def test_separable_curves_and_printed_tick_calibration():
    root = Path(__file__).resolve().parents[1]
    rows = extract(root/'references/data/1010.0860v1-figure-1b.png')
    assert len(rows)==10
    assert {r['field'] for r in rows if r['column']==153}=={'h_over_r2','minus_b'}
    assert rows[0]['x_tick_centers']==[104.5,199.5,294.5,389.5,484.5]
    assert rows[0]['y_tick_centers']==[53.5,134.5,215.5,295.5,376.5]


def test_red_finite_coupling_curve_rejects_closed_vacuum_profile():
    root = Path(__file__).resolve().parents[1]
    rows = extract(root/'references/data/1010.0860v1-figure-1b.png')
    f_rows = [r for r in rows if r['field']=='f']
    for row in f_rows:
        vacuum = float(MyersPerry(1,.33).functions(10**row['log10_r'])['f'])
        # Generous 3-pixel control still detects omission of the GB term.
        assert abs(vacuum-row['value'])>3/row['y_pixels_per_unit']
