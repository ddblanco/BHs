"""End-to-end physical checks and reproducible artifact generation."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from rotating_bh.myers_perry import MyersPerry
from rotating_bh.myers_perry_benchmark import metric_evaluator, run_benchmark


def test_symbolic_metric_matches_independent_radial_reference():
    from rotating_bh.einstein import metric_from_ansatz
    evaluate = metric_evaluator()
    for q in [-0.68,0,0.33]:
        bh = MyersPerry(1.4,q/1.4)
        r,theta = 2.3,0.6
        v = bh.functions(r)
        expected = np.array(metric_from_ansatz(r,theta,v['b'],v['f'],v['h'],v['w']),float)
        mu,a = bh.r_h**2/(1-q*q),bh.r_h*q
        actual,_,_ = evaluate(0,r,theta,0,0,mu,a,0)
        np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=2e-14)
        # Independent expansion of MP g_tt and t-azimuth component.
        assert actual[0,0] == pytest.approx(-1+mu/r**2)
        assert actual[0,3] == pytest.approx(-mu*a*np.sin(theta)**2/r**2)


def test_full_benchmark_vacuum_and_negative_control():
    result = run_benchmark()
    assert result['runs']
    assert all(c['passed'] for c in result['checks'])
    assert max(row['einstein_residual'] for row in result['runs']) < 1e-8
    assert result['negative_control_residual'] > 1e-5
    assert {row['q'] for row in result['runs']} >= {-0.33,0,0.33,0.68}
    assert len({row['r_h'] for row in result['runs']}) >= 3


@pytest.mark.parametrize('q', [-0.68, 0, 0.33, 0.68])
def test_horizon_area_from_induced_metric_and_null_generator(q):
    from rotating_bh.einstein import metric_from_ansatz
    bh = MyersPerry(1.6,q/1.6,G5=2)
    # Only the t/angular block is needed; g_rr is singular at the horizon.
    # Give f an arbitrary finite value and never include its radial component.
    v = bh.functions(bh.r_h)
    nodes,weights = np.polynomial.legendre.leggauss(24)
    area_integral = 0.0
    chi = np.array([1,0,0,bh.omega_h,bh.omega_h])
    for theta,weight in zip((nodes+1)*np.pi/4,weights*np.pi/4):
        g = np.array(metric_from_ansatz(bh.r_h,theta,v['b'],1,v['h'],v['w']),float)
        area_integral += weight*np.sqrt(np.linalg.det(g[2:,2:]))
        assert abs(chi @ g @ chi) < 1e-14
    area_integral *= (2*np.pi)**2
    assert area_integral == pytest.approx(bh.thermodynamics()['A_H'],rel=2e-14)


def test_generation_repeats_data_and_hashes_in_local_copy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for folder in ('src','experiments','environment','docs','references'):
        shutil.copytree(root/folder,tmp_path/folder,ignore=shutil.ignore_patterns('__pycache__'))
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'artifacts/manifest.json').write_text('[]',encoding='utf-8')
    cache = tmp_path/'.cache'
    cache.mkdir()
    env = {**os.environ,'PYTHONPATH':str(tmp_path/'src'), 'TEMP':str(cache),
           'TMP':str(cache),'MPLCONFIGDIR':str(cache),'PYTHONNOUSERSITE':'1'}
    cmd = [sys.executable,'experiments/myers_perry_benchmark.py']
    first = subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert first.returncode == 0,first.stdout+first.stderr
    original = (tmp_path/'results/myers-perry.json').read_bytes()
    second = subprocess.run(cmd,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert second.returncode == 0,second.stdout+second.stderr
    assert (tmp_path/'results/myers-perry.json').read_bytes() == original
    records = json.loads((tmp_path/'artifacts/manifest.json').read_text(encoding='utf-8'))
    assert len(records) == 2
    assert {record['kind'] for record in records} == {'result','figure'}
    for record in records:
        assert record['status'] == 'verified'
        assert all(c['passed'] for c in record['checks'])
        assert hashlib.sha256((tmp_path/record['path']).read_bytes()).hexdigest() == record['sha256']
        for path,digest in record['parameters']['source_sha256'].items():
            assert hashlib.sha256((tmp_path/path).read_bytes()).hexdigest() == digest
