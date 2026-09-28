# Quadrature that cannot be refined

Usually more quadrature nodes means a better integral. Not when the integrand is a
cancellation evaluated in floating point and the change of variables amplifies what
survives.

Met while integrating the Gauss-Bonnet density over the exterior of a rotating black hole
in [[extremality-shift-pedagogy-revision]]. `L_GB = R^2 - 4 R_ab R^ab + R_abcd R^abcd` is a
difference of curvature squares. On an asymptotically flat solution each term is `O(r^-4)`
squared, i.e. `O(r^-8)`, but the *cancellation* between them is what leaves `O(r^-8)`: at
large `r` the computed value is a small difference of large float64 numbers and loses
relative accuracy. Compactifying by `chi = 1 - 1/r` puts nodes arbitrarily close to
`chi = 1`, and `dr/dchi = (1-chi)^-2` multiplies precisely there.

Measured behaviour (`alpha=0.1`, `q=0.4`, relative difference from the response value):

    80 nodes  1.6e-09
    200       2.6e-07
    400       1.7e-05
    800       6.3e-04
    1600      1.3e-01
    3200      3.8e+00

The error is *not* monotone in the node count and the divergence is catastrophic, so a
convergence study of the usual shape would report the opposite of the truth.

How to handle it:

- Pick the node count from the observed plateau, not from a refinement limit.
- Record the node count as a parameter of the measurement, and store the whole ladder, so
  the reader can see that refinement was tried and why it stops.
- Say in the text that the quadrature cannot be refined indefinitely and why. A silent
  "80-point Gauss-Legendre" invites a referee to ask for 800 and get a wrong answer.

Related: [[environment-drift-in-recorded-measurements]] — both are cases where the obvious
convergence diagnostic measures something other than what it appears to.
