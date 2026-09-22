"""Reject scientifically misleading resolution and tail evidence."""
from copy import deepcopy
import pytest
from rotating_bh.egb_rotating_family_validation import checks


def evidence():
    obs = dict.fromkeys(['mass_tail_spread','spin_tail_spread','mass_b_f_difference',
                         'spin_current_difference'],1e-10)
    rows = [dict(label='resolution',alpha_gb=a,q=q,iterations=3,
            max_tensor_residual=1e-10,boundary_residual=1e-11,observables=obs)
            for a in [.05,.1] for q in [0.,.1,.2,.3,.4]]
    return dict(records=rows, refinement=[dict(differences={'E':1e-10}) for _ in rows],
        route_difference=1e-10,first_law=[dict(residual=r,relative_residual=r)
                                       for r in [1e-4,2.5e-5,6.25e-6]],
        cutoff=[dict(max_tensor_residual=1e-8,boundary_residual=1e-9,
                     observables=obs,differences={'E':1e-8}) for _ in range(2)],
        profiles=[dict(alpha_gb=a) for a in [.0025,.75]],
        external_validation=[dict(tensor=1e-10,radius=1.104) for _ in range(2)],
        external=[dict(alpha_gb=a,computed_radius=r,published_radius=r,rounding_half_unit=.0005)
                  for a,r in [(.25,1.08),(.5,1.104)]])


@pytest.mark.parametrize('corruption', ['no_iteration','missing_spin','bad_tail','wrong_ergo','nan_tensor'])
def test_independent_gates_reject_bad_evidence(corruption):
    data = deepcopy(evidence())
    assert all(checks(data).values())
    if corruption=='no_iteration': data['records'][0]['iterations']=0
    if corruption=='missing_spin': data['records'].pop()
    if corruption=='bad_tail': data['records'][0]['observables']['mass_tail_spread']=.01
    if corruption=='wrong_ergo': data['external'][0]['computed_radius']=1.2
    if corruption=='nan_tensor': data['records'][0]['max_tensor_residual']=float('nan')
    assert not all(checks(data).values())
