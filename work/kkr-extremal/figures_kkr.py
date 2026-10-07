import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    ROOT = HERE.parents[1]
    FIG = HERE / "figures"
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.2, "figure.dpi": 150,
        "legend.frameon": False, "legend.fontsize": 7.5,
    })
    # identity colours (each also carries a distinct line style or marker)
    OURS, KKR, STATIC, MP, INK, MUTED = "#b03a1a", "#244a75", "#5b6b76", "#2f6f3e", "#1a1a1a", "#9aa7b0"
    return FIG, HERE, INK, KKR, MP, MUTED, OURS, ROOT, STATIC, json, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Extremal equal-spin EGB black holes: reproduction of arXiv:2303.12471 and the mass shift

        Data: `branch_rq.json` (extremal solutions built directly at $T=0$, this work),
        `kkr_curves.json` (vector curves of the paper's fig. 1, from the arXiv EPS),
        `results/egb-extremality.json` (the manuscript's non-extremal states and its
        $T\to0$ extrapolations), `nearhorizon.json` (perturbation exponents).
        Every figure below is written to `figures/` as PDF and PNG.
        """
    )
    return


@app.cell
def _(HERE, ROOT, json, np):
    branch = json.loads((HERE / "branch_rq.json").read_text())
    rows = [r for r in branch["rows"]]
    ours = {k: np.array([r["high"][k] for r in rows]) for k in
            ("alpha", "y", "x", "j", "mu", "aH", "s", "sigma", "omega", "M", "J")}
    ours["alpha"] = np.array([r["alpha"] for r in rows])
    slope = np.array([r["high"].get("mu_prime", np.pi) for r in rows])
    slope_dn = np.array([abs(r["high"].get("mu_prime", np.pi) - r["low"].get("mu_prime", np.pi)) for r in rows])
    mu_dn = np.array([abs(r["high"]["mu"] - r["low"]["mu"]) for r in rows])
    kkr = json.loads((HERE / "kkr_curves.json").read_text())
    ms = json.loads((ROOT / "results/egb-extremality.json").read_text())
    states = [s for g in ms["walks"].values() for s in g]
    ext_ms = sorted(ms["extremals"], key=lambda r: r["y"])
    nh = json.loads((HERE / "nearhorizon.json").read_text())
    return branch, ext_ms, kkr, ms, mu_dn, nh, ours, rows, slope, slope_dn, states


@app.cell
def _(FIG, INK, KKR, MP, MUTED, OURS, STATIC, kkr, np, ours, plt, states):
    # Figure 1: the left panels of fig. 1 of arXiv:2303.12471, main (vs x) and inset (vs j)
    fig1, axes = plt.subplots(3, 2, figsize=(6.6, 7.4))
    xs = np.linspace(0, 1, 400)
    js = np.linspace(0, 1, 400)
    sq = np.sqrt(1 - js**2)
    static = {"aH": (1 - xs)**1.5, "s": np.sqrt(1 - xs) * (1 + 5 * xs),
              "tH": np.sqrt(1 - xs) / (1 + xs)}
    mpj = {"aH": (1 + sq) / 2, "s": (1 + sq) / 2, "tH": 2 * sq / (1 + sq)}
    labels = {"aH": r"$a_H$", "s": r"$s$", "tH": r"$t_H$"}
    ms_key = {"aH": "a_scaled", "s": "s_scaled", "tH": "t_scaled"}
    for row, q in enumerate(("aH", "s", "tH")):
        left, right = axes[row]
        sx = np.array([s["x"] for s in states]); sj = np.array([s["j"] for s in states])
        sq_ = np.array([s[ms_key[q]] for s in states])
        for _ax, _coord in ((left, sx), (right, sj)):
            _ax.plot(_coord, sq_, ".", ms=1.6, color=MUTED, zorder=1,
                    label="non-extremal states (manuscript)")
        left.plot(xs, static[q], "-", color=STATIC, lw=1, label="static, closed form")
        left.axvline(0, color=MP, lw=1, ls=(0, (1, 1.5)))
        right.plot(js, mpj[q], "-.", color=MP, lw=1, label=r"Myers--Perry ($\alpha=0$)")
        k = np.array(kkr[q]["main"]); ki = np.array(kkr[q]["inset"])
        if q != "tH":
            left.plot(k[:, 0], k[:, 1], "--", color=KKR, lw=1.3, label="extremal, arXiv:2303.12471")
            right.plot(ki[:, 0], ki[:, 1], "--", color=KKR, lw=1.3)
            ours_q = ours[q]
        else:
            left.plot([0, .918], [0, 0], "--", color=KKR, lw=1.3, label="extremal, arXiv:2303.12471")
            ours_q = np.zeros_like(ours["x"])
        left.plot(ours["x"], ours_q, "-", color=OURS, lw=1.6, label="extremal, this work ($T=0$)")
        right.plot(ours["j"], ours_q, "-", color=OURS, lw=1.6)
        for _ax in (left, right):
            _ax.set_ylabel(labels[q]); _ax.set_xlim(-.02, 1.02)
            _ax.grid(alpha=.18, lw=.5)
        left.set_xlabel(r"$x=3\pi\alpha/(4M)$"); right.set_xlabel(r"$j$")
    axes[0, 0].legend(loc="upper right", handlelength=2.2)
    axes[0, 1].legend(loc="upper left", handlelength=2.2)
    fig1.tight_layout()
    fig1.savefig(FIG / "fig1-reproduction.pdf"); fig1.savefig(FIG / "fig1-reproduction.png", dpi=200)
    fig1
    return (fig1,)


@app.cell
def _(FIG, HERE, INK, OURS, ext_ms, json, mu_dn, np, ours, plt, slope, slope_dn):
    # Figure 2: the extremal mass and its slope, direct construction vs T -> 0 extrapolation
    pol = json.loads((HERE / "polished.json").read_text())
    good = (mu_dn < 1e-5) & (slope_dn < 1e-5)      # both resolutions agree
    fig2, (a1, a2, a3) = plt.subplots(3, 1, figsize=(5.6, 7.4), sharex=False,
                                      gridspec_kw=dict(height_ratios=[1.2, 1.2, 1.0]))
    y0 = np.linspace(0, ours["y"].max(), 200)
    mu0 = 1.5 * np.pi**(1 / 3)
    a1.plot(y0, mu0 + np.pi * y0, ":", color=INK, lw=1, label=r"first order, $\mu(0)+\pi y$")
    a1.plot(ours["y"][good], ours["mu"][good], "-", color=OURS, label="direct construction at $T=0$ (this work)")
    ym = np.array([r["y"] for r in ext_ms]); mum = np.array([r["mu"] for r in ext_ms])
    a1.plot(ym, mum, "o", ms=3.5, mfc="white", color=INK, label=r"manuscript, $T\to0$ extrapolation")
    a1.set_ylabel(r"$\mu=M_{\rm ext}/J^{2/3}$"); a1.legend(loc="upper left")
    a2.plot(ours["y"][good], slope[good], "-", color=OURS, label=r"$\mu'(y)$, implicit differentiation (this work)")
    a2.errorbar(ym, [r["psi_gb"]["value"] for r in ext_ms],
                yerr=[r["psi_uncertainty"] for r in ext_ms], fmt="o", ms=3.5, mfc="white",
                color=INK, capsize=2, lw=.8, label=r"manuscript $\Psi_{\rm ext}$ (20-fit envelope)")
    a2.plot([0], [np.pi], "*", ms=9, color="#2f6f3e", label=r"$\pi$ (first order)")
    a2.set_ylabel(r"$\partial M_{\rm ext}/\partial\alpha|_J$"); a2.set_xlabel(r"$y=\alpha/J^{2/3}$")
    a2.legend(loc="lower right")
    # bottom: absolute differences at the manuscript's couplings (same alpha, no
    # interpolation), against the manuscript's own spreads
    yp = np.array([p["value"]["y"] for p in pol])
    a3.semilogy(yp, [abs(p["value"]["mu"] - p["manuscript"]["mu"]) for p in pol], "o-", ms=4,
                color=INK, mfc="white", lw=.8, label=r"$|\mu_{\rm ms}-\mu|$")
    a3.semilogy(yp, [p["manuscript"]["mu_spread"] for p in pol], "o:", ms=3, color=INK, lw=.8,
                label=r"$\delta\mu_{\rm ms}$ (three-fit spread)")
    a3.semilogy(yp, [abs(p["value"]["mu_prime"] - p["manuscript"]["psi"]) for p in pol], "s-", ms=4,
                color=OURS, mfc="white", lw=.8, label=r"$|\Psi_{\rm ms}-\mu'|$")
    a3.semilogy(yp, [p["manuscript"]["psi_envelope"] for p in pol], "s:", ms=3, color=OURS, lw=.8,
                label=r"$\delta\Psi_{\rm ms}$ (20-fit envelope)")
    a3.set_xlabel(r"$y=\alpha/J^{2/3}$ (manuscript couplings)"); a3.set_ylabel("absolute difference")
    a3.legend(loc="center right", fontsize=7, ncol=2)
    for _ax in (a1, a2, a3):
        _ax.grid(alpha=.18, lw=.5)
    fig2.tight_layout()
    fig2.savefig(FIG / "fig2-mass-shift.pdf"); fig2.savefig(FIG / "fig2-mass-shift.png", dpi=200)
    fig2
    return (fig2,)


@app.cell
def _(FIG, INK, OURS, nh, np, plt):
    # Figure 3: the non-integer near-horizon exponent
    fig3, ax3 = plt.subplots(figsize=(5.0, 2.8))
    yy = np.array([r["y"] for r in nh]); gam = np.array([max(r["exponents"]) for r in nh])
    ax3.plot(yy, gam, "o-", ms=3, color=OURS, label=r"$\gamma$: mode $\rho^\gamma$ at the extremal horizon")
    # the partner -1-gamma lies outside the scanned window (-6, 6) at large y
    partner = np.array([min(r["exponents"]) if min(r["exponents"]) < -2.5 else np.nan for r in nh])
    ax3.plot(yy, -1 - partner, "x", ms=4,
             color=INK, label=r"$-1-\gamma'$ of its AdS$_2$ partner")
    ax3.axhline(2, color="#9aa7b0", lw=.7, ls=":")
    ax3.set_xscale("log"); ax3.set_xlim(8e-4, yy.max()*1.2)   # y = 0 (Myers-Perry: gamma = 2) off-axis
    ax3.set_xlabel(r"$y=\alpha/J^{2/3}$")
    ax3.set_ylabel(r"$\gamma$"); ax3.legend(loc="lower left"); ax3.grid(alpha=.18, lw=.5)
    fig3.tight_layout()
    fig3.savefig(FIG / "fig3-exponent.pdf"); fig3.savefig(FIG / "fig3-exponent.png", dpi=200)
    fig3
    return (fig3,)


if __name__ == "__main__":
    app.run()
