# Revisión del 27 de octubre de 2026

- `report/review-report.pdf`: reporte completo en castellano (verificaciones, errores, mejoras,
  factibilidad de una expresión analítica para M_ext); su última sección es el reporte en inglés.
- `report/analytic-expression-report.pdf`: el reporte en inglés, autocontenido.
- `checks/`: scripts de verificación independientes (`check_thermo.py`, `check_throat2.py`,
  `check_branch_consistency.py`, `uniformize_q.py`, `make_figures.py`) y `receipts/` con las salidas
  íntegras, incluida la suite de tests.

Correr desde la raíz del proyecto con `OPENBLAS_NUM_THREADS=1 .venv/bin/python -B checks/<script>.py`
(`.venv` = Python 3.11.18 con `environment/requirements-lock.txt`).
