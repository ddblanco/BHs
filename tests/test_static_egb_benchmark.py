def test_benchmark_accepts_fine_runs_and_all_checks_pass():
    from rotating_bh.static_egb_benchmark import run_benchmark
    result = run_benchmark()
    assert all(check['passed'] for check in result['checks']), result['checks']
    assert len(result['runs']) == 2*7  # 2 methods x 7 alpha_hat targets
    assert result['negative_control_residual'] > 1e-5
    assert all(not attempt['converged'] for attempt in result['failure_controls'])
    assert result['resolution_sweep'][-1]['profile_error'] < result['resolution_sweep'][0]['profile_error']
    assert result['gr_limit'][0]['max_difference'] == 0.0
