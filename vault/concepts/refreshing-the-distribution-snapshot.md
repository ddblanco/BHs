# Refreshing the distribution snapshot

`artifacts/distribution-sha256.json` is a release snapshot: the sha256 of every published
file, checked by `checks/verify_distribution.py`. That checker's docstring says it "never
updates it", and the repository ships no generator, so the refresh is a deliberate manual
step — the point of the snapshot is that it certifies a *reviewed* state, and a tool that
silently re-baselined it would certify nothing.

Consequence worth stating plainly: refreshing re-baselines the inventory onto whatever is
in the working tree at that moment. The audit going green afterwards is not evidence that
the contents are right; it is evidence that they have not changed *since the refresh*. The
scientific provenance lives elsewhere, in `artifacts/manifest.json` and the
`source_sha256` blocks of each result, checked by `checks/verify_recorded_digests.py`.

Do the walk with the checker's own rules rather than a second copy of them, so the two
cannot drift:

```python
import hashlib, importlib.util, json
from pathlib import Path
ROOT = Path('.').resolve()
spec = importlib.util.spec_from_file_location(
    'distribution', ROOT/'checks/verify_distribution.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
files = mod.published_files(ROOT)          # excludes .git/.venv/.cache/__pycache__,
                                           # .pyc/.aux/.log/.out/.toc, *.synctex.gz,
                                           # the index itself and the optional arXiv image
new = {name: hashlib.sha256(p.read_bytes()).hexdigest()
       for name, p in sorted(files.items())}
(ROOT/mod.INDEX).write_text(json.dumps(new, indent=2) + '\n', encoding='utf-8')
```

Check the diff before committing: it should account for every entry as added, removed or
changed, and every one of those should be a file you meant to touch. A stray temp file or
a cache directory showing up in the "added" list means the exclusion rules missed
something, not that the snapshot is ready.

Done once, after [[extremality-shift-pedagogy-revision]]: 227 -> 237 entries, 10 added, 17
changed, none removed, of which 4 belonged to an upstream commit rather than to that work.
Note the ordering trap: the inventory has to be generated *last*. Writing this page after
the first regeneration invalidated it, and it had to be redone.

Done again on 2026-10-08: 249 -> 438 entries, 190 added (`work/kkr-extremal`,
`work/critical-review`, `work/review`, `presentation-claude`, `output/pdf`, vault pages), 9
changed (7 regenerated manuscript figures, `main.tex`, `.gitignore`), 1 removed: the
gitignored third-party `references/data/1010.0860-src/profiles-alpha.eps`, which an earlier
refresh had picked up from a local copy. The walk does not read `.gitignore`, so check the
new entries against `git ls-files` before writing; here all 438 were tracked.

Related: [[environment-drift-in-recorded-measurements]] — six figure PDFs were among the
16 changed, re-rendered by a different matplotlib with identical data, which is exactly
the kind of byte change a snapshot records and a provenance check should not.
