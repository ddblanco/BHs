"""Recompute Hito 4C gates from numerical evidence, not saved booleans."""
import numpy as np


def checks(data):
    rows = data['records']
    expected = {(a,q) for a in [.05,.1] for q in [0.,.1,.2,.3,.4]}
    fine = [r for r in rows if r['label']=='resolution']
    law,cut,external = data['first_law'],data['cutoff'],data['external']
    obs = [r['observables'] for r in rows]+[r['observables'] for r in cut]
    return dict(
        complete_family={(r['alpha_gb'],r['q']) for r in fine} == expected,
        independent_tensor=all(np.isfinite(r['max_tensor_residual']) and
            r['max_tensor_residual']<1e-6 and r['boundary_residual']<1e-8 for r in rows),
        resolution_refined=len(fine)==10 and all(r['iterations']>0 for r in fine),
        resolution_observables=len(data['refinement'])==10 and
            max(max(r['differences'].values()) for r in data['refinement'])<1e-6,
        routes_agree=data['route_difference']<1e-8,
        tail_extraction=all(max(o[k] for k in ['mass_tail_spread','spin_tail_spread',
            'mass_b_f_difference','spin_current_difference'])<1e-6 for o in obs),
        first_law=len(law)==3 and law[-1]['relative_residual']<1e-4,
        first_law_refines=abs(law[-1]['residual'])<abs(law[0]['residual'])/8,
        cutoff_tensor=len(cut)==2 and all(r['max_tensor_residual']<1e-6 and
            r['boundary_residual']<1e-8 for r in cut),
        cutoff_observables=max(cut[-1]['differences'].values())<1e-5,
        external_agreement={r['alpha_gb'] for r in external}=={.25,.5} and all(
            abs(r['computed_radius']-r['published_radius'])<=r['rounding_half_unit'] for r in external),
        external_independent_method=len(data['external_validation'])==2 and all(
            r['tensor']<1e-6 and r.get('boundary_residual',0)<1e-8 and
            abs(r['radius']-external[-1]['computed_radius'])<1e-6 for r in data['external_validation']),
        figure_parameters={p['alpha_gb'] for p in data['profiles']}=={.0025,.75})
