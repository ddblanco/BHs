# Working agreement

## The project

Physics work on gravitational waves, done as part of a course. The person you are working with is a physicist and does not need concepts explained unless they ask.

## Layout

```text
vault/          one small page per paper, per exercise, per concept
work/           scripts, notebooks, figures; the substance lives here
papers/         source PDFs; never cite a paper that is not in here
```

## How to work

- Prefer a check to a claim. If you derive something, verify it with `sympy` and assert the difference is zero. If you compute a number a paper printed, assert you reproduced it. A result with no assertion is a hypothesis; say so.
- Say which parts you derived and which you recalled, always and unprompted.
- Never claim to have run something you did not run. Paste the output.
- Reproducing a figure means reproducing the same axes and curve from the same kind of input. If the data were unavailable, say the figure was not reproduced.
- State what was not checked. Do not give a passing summary that hides unasserted work.

## Formats

- Write-ups: LaTeX compiled to PDF, not Markdown.
- Figures: a marimo notebook (`.py`) exported to HTML, not `.ipynb`.
- Anything interactive: one self-contained HTML file.
- Markdown is for the vault only.

## The vault

At the end of any work that produced a derivation, figure, or useful failed attempt, write or update a short page in `vault/` and link generously with `[[wiki-links]]`.

- `vault/papers/<slug>.md`: claims, evidence, and reproductions
- `vault/exercises/<slug>.md`: attempt, result, and run receipt
- `vault/concepts/<slug>.md`: definition and relations, no essay

A run receipt contains the date, command, environment, and actual output. Templates, when present, are in `vault/templates/`.

## Do not

- Do not write long prose summaries.
- Do not install into the base environment; create a per-exercise environment.
- Do not create files outside this directory.
- Do not commit or push unless asked.
