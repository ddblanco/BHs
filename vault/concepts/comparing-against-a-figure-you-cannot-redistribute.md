# Comparing against a figure you cannot redistribute

arXiv's non-exclusive licence lets arXiv distribute a paper's figures. It does not
let a third party redistribute them. So a reproduction that wants to show "their
curve against ours" cannot ship their curve as an image. It can ship
**coordinates**, and there are two ways to get them, which are not equally good.

## The raster route, and its ceiling

`experiments/egb_rotating_paper_profiles.py` reads fig. 1b of arXiv:1010.0860 from
the published PNG: five printed ticks per axis fit an affine calibration, three
columns are sampled, red-pixel groups separate the curves, and each reading gets a
box of ±1 horizontal and ±2 vertical pixels. Ten points, each with a box of about
`1.5e-2`.

That protocol is honest — it is fixed before our own output is consulted, and the
one column where `f` and `w` cross has its blended group *discarded* rather than
assigned to whichever curve fits better. But its verdict was "all ten readings fall
inside the box", worst `1.49e-2` against a box of `1.52e-2`. A box that wide can
only ever say "not obviously different".

## The vector route

The arXiv **source** ships the same figure as `profiles-alpha.eps`, gnuplot 4.0
PostScript. Every curve is a polyline: `M`/`L` absolute, `V`/`R` relative, with the
linetype (`LT0` red, `LT6` dashed black) selecting which coupling. Every tick is a
`moveto` followed by its printed label, so the affine maps are *fitted from the
file itself* — `experiments/egb_rotating_paper_eps_profiles.py` refuses a file whose
ticks are not collinear to within the one device unit gnuplot rounds to. No
digitisation, 74–90 vertices per curve, device unit `5.17e-4`.

## What it found

At the horizon the two calculations agree to `2.07e-4`, a third of a device unit,
which pins the parameters and the boundary conditions. In the interior they differ
by up to `1.40e-2` — a smooth excursion peaking where each curve is steepest, about
twenty-seven device units wide. **The raster route could not have seen this**: its
box was wider than the effect.

Two hypotheses, both tested rather than argued:

- *A registration offset.* Fitting a common horizontal shift in `log10 r` moved the
  worst residual only from `1.40e-2` to `1.26e-2`. Rejected — the residual is not
  proportional to the derivative.
- *Our solver.* The same figure draws a near-Einstein coupling where our solution is
  Myers-Perry to thirteen digits. Its published curve sits `8.19e-3` from ours and
  `9.54e-3` from the **closed form**, which carries no numerical error at all.
  So the departure belongs to the published figure. Cause not established.

The lesson is the general one: *a comparison is only as sharp as the weaker side's
resolution, and it is worth finding out which side that is.* Upgrading the input
turned "agrees within the box" into a measured, structured difference plus a control
that localises it.

## Drawing it

`short-summary/make_comparison_figure.py`: left panel the overlay, right panel the
difference with the device-unit band shaded. The residual panel is the one that
carries information; the overlay alone would show two curves lying on top of each
other and say nothing. Palette validated with the dataviz checker, direct labels so
identity never rests on colour alone.

Related: [[extremality-shift-pedagogy-revision]],
[[quadrature-that-cannot-be-refined]] — both cases where the obvious diagnostic
measures something other than what it appears to.
