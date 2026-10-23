# JHEP extremal-mass paper and referee report

2026-10-23. Editorial synthesis of `manuscript/main.tex`, `extremal-mass/extremal-mass-report.tex`, its strong-coupling supplement and `work/kkr-extremal/writeup/`. Source snapshot: `bad40de`. User requested a JHEP paper modelled on local 1010.0860 and 2009.00015 PDFs, numerical appendices, a separate critical referee report, no calculations, and commit/push.

- Paper: `output/pdf/jhep-extremal-mass.pdf` (12 pages); source `work/jhep-paper/main.tex`.
- Referee: `output/pdf/jhep-extremal-mass-referee.pdf` (4 pages); source `work/jhep-paper/referee-report.tex`. Recommendation: major revision.
- Standalone paper source archive: `output/pdf/jhep-extremal-mass-source.zip` (main, authors, official JHEP style; excludes referee).
- Main corrections: branch-specific mass; conditional rotating BTZ dictionary and zeta recognition; horizon branch points do not prove mass singularities; no theorem excluding closed forms; fit ranges are not full error bounds. Incorporates direct-extremal overlap evidence missing from the old review.
- No new derivation, fit, solver execution, numerical evaluation or symbolic check. Standard identities recalled; prior project derivations transcribed with their qualifications. No figures created or reproduced. Tables reuse recorded values.

Related: [[extremality-shift-paper-review]], [[kkr-extremal-reproduction]], [[non-smooth-extremal-horizon]], [[btz-universality-extremal-egb]], [[arxiv-2303.12471]].

## Author and format sources

Author order provided by user: Cecilia Bejarano, David Blanco, Andrés Goya, Guillem Pérez Nadal. Institutional affiliations and public emails checked online on 2026-10-23:

- [Cecilia, IAFE](https://www.iafe.uba.ar/recursos_humanos/bejarano-cecilia/): IAFE CONICET–UBA; cbejarano@iafe.uba.ar.
- [Andrés, IAFE](https://www.iafe.uba.ar/recursos_humanos/goya-andres/): IAFE CONICET–UBA; agoya@iafe.uba.ar. [UBA group roster](https://wp.df.uba.ar/hepth/people/) also lists him among faculty; included DF-FCEN-UBA affiliation.
- [David, UBA](https://www.df.uba.ar/es/component/researchers/miembro/409-David_Blanco), [group contact](https://wp.df.uba.ar/hepth/david-blanco/): DF-FCEN-UBA, CONICET; dblanco@df.uba.ar.
- [Guillem, UBA](https://www.df.uba.ar/es/difusion/102-difusion-eventos/8386-coloquio-la-paradoja-de-la-informacion-de-los-agujeros-negros), [group contact](https://wp.df.uba.ar/hepth/guillem-perez-nadal/): IFIBA and Exactas-UBA; guillem@df.uba.ar. IFIBA institutional designation also cross-checked for David in published front matter; no new scientific reference added.
- [JHEP author instructions](https://jhep.sissa.it/jhep/help/JHEP/JHEP_author.jsp): used official style, formula-free abstract, AI-assistance disclosure. [Official style download](https://jhep.sissa.it/jhep/help/JHEP/TeXclass/DOCS/jheppub.sty), retained unmodified with its licence. Corresponding-author role was not assigned or inferred. No journal upload performed.

## Run receipt

Date: 2026-10-23. Environment: Bash, Python 3.10.12, pdfTeX 3.141592653-2.6-1.40.22 (TeX Live 2022/dev/Debian), pdftoppm 22.02.0. No dependencies installed. Commands from repository root:

```sh
bash work/jhep-paper/build.sh
python3 work/jhep-paper/check_documents.py
pdftoppm -scale-to 1200 -png work/jhep-paper/main.pdf work/jhep-paper/qa/paper
pdftoppm -scale-to 1200 -png work/jhep-paper/referee-report.pdf work/jhep-paper/qa/referee
```

Actual build/check output:

```text
BUILD PASS: paper and referee PDFs compiled (3 passes each)
PASS: 9 cited papers have local PDFs; citations and cross-references resolve
PASS: main: 12 pages; no box overflow or unresolved references; delivery copy matches
PASS: referee-report: 4 pages; no box overflow or unresolved references; delivery copy matches
NOTE: official JHEP style emits one harmless deprecated pagecolor warning
NOT CHECKED: physics equations, numerical data, fits, horizon regularity, stability or uniqueness
NO NEW PHYSICS CALCULATIONS PERFORMED
```

Rendered and visually inspected all 16 pages via four contact sheets; no clipped equations, overlapping text or blank pages observed. Corrected a long coefficient equation before the final build. Rendering logs empty. QA images/build logs are local ignored intermediates. Existing untracked shooting data and PID files were left untouched and excluded from staging. The PDF quality marker ran successfully using `/usr/lib/chatgpt/resources/cua_node/bin/node` after `node` was absent from PATH.

Not checked: scientific validity, complete numerical error budgets, horizon regularity, exactness of integer-relation recognition, or unseen external literature. The referee report proposes future checks without performing them. Authors still need to assess the scientific recommendations and confirm their full assistance/funding disclosures before an actual submission.
