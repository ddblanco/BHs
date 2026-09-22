import numpy as np
import pytest


def test_continuation_reaches_both_signs_and_records_attempts():
    from rotating_bh.vacuum_continuation import continue_family
    solutions,attempts=continue_family([0,.1,.2,.33,-.1,-.33],method='spectral',resolution=20,tol=1e-10)
    assert [s.q for s in solutions]==[0,.1,.2,.33,-.1,-.33]
    assert all(a['converged'] for a in attempts)
    x=np.linspace(.01,.99,30)
    assert np.max(abs(solutions[3].evaluate(x)[:3]-solutions[-1].evaluate(x)[:3]))<1e-7
    assert np.max(abs(solutions[3].evaluate(x)[3]+solutions[-1].evaluate(x)[3]))<1e-7


def test_real_iteration_failure_is_recorded_and_bounded():
    from rotating_bh.vacuum_continuation import continue_family, ContinuationFailure
    with pytest.raises(ContinuationFailure) as caught:
        continue_family([0,.3],method='spectral',resolution=12,tol=1e-12,max_iterations=0)
    assert caught.value.attempts
    assert not caught.value.attempts[-1]['converged']
    assert 'exhausted' in caught.value.attempts[-1]['reason']


def test_real_failed_step_is_halved_and_destination_recovered():
    from rotating_bh.vacuum_continuation import continue_family
    solutions,attempts=continue_family([0,.33],method='spectral',resolution=20,tol=1e-9,max_iterations=5)
    assert solutions[-1].q==.33
    failed=next(i for i,a in enumerate(attempts) if not a['converged'])
    bad=attempts[failed];retry=attempts[failed+1]
    assert retry['trial']==pytest.approx((bad['origin']+bad['trial'])/2)
    assert any(a['converged'] for a in attempts[failed+1:])
