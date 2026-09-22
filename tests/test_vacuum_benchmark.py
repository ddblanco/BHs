def test_benchmark_accepts_fine_runs_and_preserves_coarse_rejections():
    from rotating_bh.vacuum_benchmark import run_benchmark
    result=run_benchmark()
    assert all(check['passed'] for check in result['checks'])
    assert len(result['runs'])==63
    assert any(not run['accepted'] for run in result['runs'])
    assert result['cutoff_study'][-1]['profile_error']<result['cutoff_study'][0]['profile_error']/20
    assert result['negative_control_residual']>1e-5
    assert all(not attempt['converged'] for attempt in result['failure_controls'])
