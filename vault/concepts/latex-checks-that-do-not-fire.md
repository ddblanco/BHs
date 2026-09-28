# LaTeX checks that do not fire

`manuscript/check_manuscript.py` verifies three things: every `\N...` macro used is
generated and every generated one is used, no literal decimal survives in the body, and
every `\bibitem` is cited and every `\cite` resolves. It does **not** check `\label` /
`\ref`.

Nor, usefully, does the compiler here. Rewriting the introduction in
[[extremality-shift-pedagogy-revision]] deleted the display equation carrying
`\label{eq:firstlaw}`, which seven `\eqref`s still pointed at. Tectonic 0.17 exited 0 and
printed no undefined-reference warning; the PDF built. The dangling references were found
only by an explicit cross-check:

```python
labels = set(re.findall(r'\\label\{([^}]+)\}', text))
refs   = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text))
print(sorted(refs - labels))     # refs with no label
print(sorted(labels - refs))     # labels never referenced
```

Worth running after any structural rewrite, and worth folding into
`check_manuscript.py`. Note that `labels - refs` has one legitimate standing entry,
`sec:conventions`, which predates this work.
