# Extremality-shift paper review

Critical review of [[extremality-shift-rotating-black-holes]] and its finite-coupling extended-thermodynamics claim. Deliverables: `codex-review/referee-report.{html,tex,pdf}`. Recommendation: major revision.

Rewritten on 2026-09-27 for readability: the report now leads with the bottom line, separates established points from uncertainties, groups the critique into six prioritized revisions, and includes suggested replacement wording for the main claim. The scientific findings and scope did not change.

Main findings: the printed first-law/Smarr algebra checks; the saved central table has positive `Psi_ext`, increasing `mu`, and a robust broad decrease-and-rise under the stated fit sensitivities. The principal gaps are continuum control at all cold endpoints, the modeled rather than constructed extremal limit, correlated rather than independent validation, an over-strong global “minimum mass” formulation, and a non-frozen source/PDF/provenance state. No paper source PDFs were available, so literature attributions were not checked.

## Run receipt

- Date: 2026-09-27
- Command: `work/.venv-codex-review/bin/python work/review_checks.py`
- Environment: Linux; Python 3.12.3; NumPy 2.4.6; SciPy 1.17.1; SymPy 1.14.0; locked per-review virtual environment
- Output:

```text
PASS symbolic static-potential identity
PASS symbolic differentiated-Smarr constraint
PASS perturbative endpoints Psi_0(0)=-9*pi/4 and Psi_0(1)=pi
PASS perturbative sign zero q=0.634935845830
PASS saved table: 13 ordered y values, all Psi>0, all mu increasing
PASS saved-table non-monotonic trend survives stated sensitivities: fall gap=2.057093330, rise gap=0.369966342
CHECK trapezoid integral vs mass change: 0.694424504214 vs 0.691647023593; difference=0.002777480621
CHECK manuscript cubic-route running discrepancy: endpoint=0.000380737432, maximum=0.000437510799
CHECK source digest recorded=43aa372e0bbda345bf7d887fdd3578d4e0149d3b2ebd925d425515a2db8521ad
CHECK source digest current =3cf475762255b14bcfe922ff311028fa33bc6248109c24ba72e920c2a272c49c
CHECK digest match=False
NOT CHECKED continuum BVP solve or cited literature
```

Full-suite receipt: `12 failed, 474 passed, 2 skipped, 4 errors in 494.58s`; failures were classified in the report rather than reduced to a misleading pass count. The production BVP walks, direct extremal solutions, and cited literature remain unchecked.
