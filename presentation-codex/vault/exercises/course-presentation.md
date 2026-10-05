# Course presentation, 1–2 October 2026

Eight slides and a 1,174-word script, planned for 600 seconds. HTML and LaTeX/PDF deliverables live in `presentation-codex/`. Start with `index.html`.

Summarizes [[extremality-shift-paper-revision]] and the comparison with [[1010-0860]]. Uses the archived figures, including the unresolved profile discrepancy, and labels extremal values as extrapolants. Initial prompts and usage totals come from the project history. No new solutions derived or computed; rechecked the first-law chain rule and implicit response identity with SymPy. Established first law and analytic benchmarks are recalled inputs.

Affiliations checked online. Claude and ChatGPT/Codex credited as assistance, human authors retain responsibility. Total CPU/GPU computation time is unavailable. Usage report has 36.8 h overall versus 36.6 h in its milestone table, noted in the source appendix.

## Run receipt

Date: 2026-09-29. Working directory: `presentation-codex/`.
Environment: Python 3.10.12, SymPy 1.14.0, system pdfLaTeX/Poppler, headless Google Chrome. No packages installed.
Commands: `../.venv/bin/python build.py`; `../.venv/bin/python validate.py`; `pdftoppm -scale-to 1000 -png speaker-script.pdf build/script`; headless Chrome with a temporary local QA page exercising navigation and notes.

Actual output:

```text
Archived data: 356 states, 13 couplings; positive slopes; valley/end values and 31% mass growth verified.
SymPy: chain rule, regular T=0 first-law limit, and residual response identity asserted.
Usage report arithmetic: 1,415,508,752 processed and 5,942,462 output tokens verified; session logs not re-audited.
Deliverables: 8 PDF/HTML slides; 600-second plan; 1,174 spoken words; 10-page script PDF; no LaTeX overflows.
Offline HTML: all 8 vector slide images and scripts embedded; sources are optional outbound links.
Not checked: fresh nonlinear solves, total CPU/GPU time, raw historical usage logs, live speaking duration.
Environment: Python 3.10.12, SymPy 1.14.0; system pdfLaTeX and Poppler.
PASS: 8 slides; next button; previous button; End key; Home key; right arrow; notes visible; script visible; 8 vector images decoded
```

Visual review: all eight slide layouts and the ten-page script inspected; author/affiliation and table paragraph breaks corrected. Browser opening and controls checked. Fullscreen activation and physical projector rendering were not tested. The PDF skill's optional operation-marker script could not run because Node is absent; PDF generation used the requested LaTeX toolchain.

The receipt is kept inside the presentation folder to honor the requested self-contained delivery location.
