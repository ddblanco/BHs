# Resumen de cinco páginas del proyecto

Entregables: `short-summary-codex/summary.tex` y `short-summary-codex/summary.pdf`.
Texto autónomo en español: problema, desarrollo del solver, respuesta termodinámica,
resultados, controles, revisiones y límites. Sigue la explicación española local y
el registro académico de las cuatro referencias solicitadas. La fuente TeX usa las tres figuras incluidas en `short-summary-codex/figures/`
y no necesita macros del manuscrito. Se añadió a `papers/` el PDF de arXiv:2303.07358.

Síntesis de resultados existentes, sin derivaciones físicas nuevas. Distingue antecedentes,
álgebra derivada en el proyecto y mediciones archivadas. No confunde el menor punto
muestreado con una localización precisa del mínimo ni las sensibilidades con intervalos
estadísticos. Enlaces: [[extremality-shift-paper-revision]],
[[extremality-shift-pedagogy-revision]], [[extremality-shift-verification-record]],
[[environment-drift-in-recorded-measurements]].

## Run receipt

- Fecha: 2026-09-28.
- Entorno: Linux; Python 3.12.3, NumPy 1.26.4, SymPy 1.12; Tectonic 0.17.0.
- Comandos desde la raíz:

```bash
python3 work/manuscript_audit.py > short-summary-codex/audit-output.txt
tectonic -X compile short-summary-codex/summary.tex --keep-logs
pdfinfo short-summary-codex/summary.pdf
```

Salida de auditoría (completa en `short-summary-codex/audit-output.txt`):

```text
43 passed, 0 failed
```

Comprobación adicional ejecutada mediante Python inline: comparación de las cuatro filas
con `Nextremaltable` en `manuscript/numbers.tex`, aserción de cinco páginas usando
`pdfinfo`, y búsqueda de errores de composición en el log. Salida guardada en
`short-summary-codex/validation-output.txt`:

```text
PASS summary table: 16 entries agree with manuscript table to displayed precision
PASS PDF: exactly 5 pages
PASS LaTeX: no overfull boxes, undefined references, or missing characters
```

Se inspeccionó visualmente la página final renderizada. No se ejecutaron la suite completa,
los recorridos de producción, nuevas extrapolaciones ni reproducciones de figuras.
Las cifras de controles históricos se describen como registros previos, no como nuevas
mediciones de esta sesión.

## Actualización: figuras solicitadas

Se incorporaron copias exactas de `fig-profiles.pdf`, `fig-shift.pdf` y
`fig-massext.pdf` del manuscrito. Texto abreviado para conservar cinco páginas;
pies en español explican los ejes reducidos, las convenciones de los perfiles y
la extrapolación extrema. La tabla abreviada anterior se sustituyó por las figuras.
También se corrigió un `quad` literal en la ecuación entrópica.

- Fecha y entorno: 2026-09-28; mismo Python y Tectonic del recibo anterior.
- Compilación: `tectonic -X compile short-summary-codex/summary.tex --keep-logs`.
- Comprobación Python inline: `pdfinfo` para contar páginas; comparación byte a byte
  de las tres figuras; `pdftotext` para los tres pies; inspección del log.
- Inspección visual de las páginas 2 y 4, renderizadas con `pdftoppm`.
- Salida actual en `short-summary-codex/validation-output.txt`:

```text
PASS PDF: exactly 5 pages
PASS figures: 3 embedded PDFs identical to manuscript originals
PASS captions: all 3 present in compiled PDF
PASS LaTeX: no warnings, overfull boxes, undefined references, or missing characters
```

Sin nuevas derivaciones, mediciones o regeneración de curvas. Se reutilizaron artefactos
existentes; no se ejecutaron nuevamente el solver ni la auditoría física de 43 controles.
Para compilar fuera del repositorio, conservar `summary.tex` junto con `figures/`.

## English translation

Translated `short-summary-codex/summary.tex` and recompiled `summary.pdf` in place,
including all captions, headings, and explanatory text. Retained the five-page
structure, three original figures, equations, numerical claims, and limitations.
No new derivations or recalled scientific claims were added.

Run receipt (2026-09-28; same Python and Tectonic environment as above):

```bash
tectonic -X compile short-summary-codex/summary.tex --keep-logs
pdfinfo short-summary-codex/summary.pdf
pdftoppm -f 4 -l 5 -scale-to 1200 -png short-summary-codex/summary.pdf short-summary-codex/english-page
```

Python inline assertions checked the page count, log, English captions, and figure
bytes against the manuscript originals. Actual output:

```text
PASS English PDF: exactly 5 pages
PASS figures: 3 embedded PDFs identical to manuscript originals
PASS English captions: all 3 present in compiled PDF
PASS LaTeX: no warnings, overfull boxes, undefined references, or missing characters
```

Inspected rendered pages 4 and 5. This was translation and typesetting only;
no physical calculations, production runs, or scientific audits were rerun.

## Comparison with arXiv:1010.0860

Added a short reproduction overview and Figure 2: the original Fig. 1(b) beside
colour plots of the archived profiles at matching parameters and axes. Preserved
all three existing figure PDFs byte for byte and the five-page English summary.
The paper's coupling is four times the project's; both plotted states have
r_H=1 and Omega_H=0.33. The plotted functions are f, -b, h/r², and w.

Recovered `references/data/1010.0860v1-figure-1b.png` from its recorded source URL;
the original SHA-256 matches. Repeated the archived extraction without overwriting
historical result files or manifests. Ten separated readings on the high-coupling
curves pass the original ±1 x-pixel / ±2 y-pixel protocol. Low-coupling agreement
is visual only. The authors' numerical data are unavailable. The distinct
alpha_paper=2 ergosurface discrepancy remains open.

New reproducible figure: `short-summary-codex/reference_comparison.py` (marimo),
exported as `reference_comparison.html`; PDF and PNG in the local `figures/`.
`comparison-provenance.json` hashes the image, saved profiles, extraction script,
and notebook. This re-plots calculated profiles and repeats image measurements;
no new physical identities were derived or assumed from memory.
Related: [[extremality-shift-verification-record]], [[extremality-shift-paper-review]].

### Run receipt

- Date: 2026-09-28.
- Environment: per-summary `.venv` (system packages exposed, new installations
  confined to the venv); Python 3.12.3, marimo 0.25.0, NumPy 1.26.4,
  matplotlib 3.6.3; Tectonic 0.17.0. Added dependencies recorded in
  `short-summary-codex/notebook-requirements.txt`.
- Commands from repository root:

```bash
curl -L --fail --max-time 40 https://arxiv.org/html/1010.0860v1/profiles-alpha.png -o references/data/1010.0860v1-figure-1b.png
short-summary-codex/.venv/bin/marimo export html short-summary-codex/reference_comparison.py -o short-summary-codex/reference_comparison.html
tectonic -X compile short-summary-codex/summary.tex --keep-logs
```

Actual notebook output (`comparison-checks.txt`):

```text
PASS reference-image SHA-256 matches archived input
PASS parameters: r_H=1, Omega_H=0.33, alpha_paper=4*alpha
PASS 10/10 separated digitised values within archived pixel tolerances
PASS re-extraction agrees with archived comparison to 1e-12
NOT CHECKED: new BVP solves; quantitative digitisation of alpha_paper=0.01
OPEN: separate ergosurface-radius discrepancy at alpha_paper=2
```

Python inline assertions used pdfinfo, pdftotext, byte comparison and SHA-256;
actual output (`validation-output.txt`):

```text
PASS PDF: exactly 5 pages and 4 figure captions
PASS original figures: all 3 unchanged
PASS comparison provenance: all input SHA-256 digests match
PASS LaTeX: no warnings, overfull boxes, undefined references, or missing characters
```

Visually inspected rendered pages 2 and 5. The full solver/test suite and
extremal extrapolations were not rerun. No claim of reproducing the full paper.

## Restore previous English summary

At the user's request, reverted the last summary edit: removed the new reference
comparison section and figure, restored the prior section 2, bibliography,
limitations wording, and three-figure layout. Recompiled `summary.tex` to
`summary.pdf`. The restored text represents the pre-comparison version; the
separate comparison notebook and receipt remain historical artifacts.

Run receipt: 2026-09-28; same Tectonic/Python environment as above.
Command: `tectonic -X compile short-summary-codex/summary.tex --keep-logs`.
Python assertions checked pdfinfo, extracted captions, log, and original figure bytes.
Actual output:

```text
PASS restored English summary: 5 pages, 3 original figures
PASS added reproduction section and comparison figure removed
PASS LaTeX: no warnings or layout errors
```

No new derivations or physical checks; only document restoration and build checks.
