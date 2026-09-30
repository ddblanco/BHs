"""Figures made for the talk; everything else in figures/ is copied from the project.

Every number below is copied from a file in the repository, named beside it.
Run from this folder:  python3 make_figures.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})
INK = "#1f2937"
BLUE = "#1f5fa8"
ORANGE = "#d9622b"
GREY = "#9ca3af"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def question():
    """The known first-order line and the unknown finite-coupling curve.

    mu(0) = (3/2) pi^(1/3): Myers-Perry bound (short-summary/summary.tex).
    slope pi: Ma, Li, Lu, arXiv:2009.00015 (summary.tex, eq. published).
    No measured value is drawn here; the result is shown on a later slide.
    """
    mu0 = 1.5 * math.pi ** (1 / 3)
    y = [0, 0.55]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(y, [mu0 + math.pi * v for v in y], "--", color=GREY, lw=1.6,
            label=r"first order in $\alpha$: slope $\pi$")
    ax.fill_between([0, 0.55], [mu0 - 0.2, mu0 - 0.2], [mu0 + 1.9, mu0 + 1.9],
                    color=BLUE, alpha=0.06, lw=0)
    ax.text(0.40, mu0 + 0.52, "?", fontsize=46, color=BLUE, ha="center", va="center")
    ax.text(0.40, mu0 + 0.02, "finite coupling:\nnot known", color=BLUE,
            ha="center", va="bottom", fontsize=10.5)
    ax.plot([0], [mu0], marker="*", ms=15, color=ORANGE, zorder=5, clip_on=False,
            label=r"Einstein: $\frac{3}{2}\pi^{1/3}$ (Myers–Perry)")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(mu0 - 0.2, mu0 + 1.9)
    ax.set_xlabel(r"Gauss–Bonnet coupling  $y=\alpha/J^{2/3}$")
    ax.set_ylabel(r"extremal mass  $M_{\rm ext}/J^{2/3}$")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save(fig, "fig-question")


def checks():
    """Worst deviations of the independent checks.

    Values: manuscript/numbers.tex (as read by short-summary/summary.tex, table
    'How far it can be trusted').
    """
    rows = [
        ("Einstein limit of $M_{\\rm ext}$", 4.56e-11),
        ("static closed form (Boulware–Deser)", 2.60e-9),
        ("field equations, off-grid", 9.76e-9),
        ("first law, spin direction", 1.35e-8),
        ("published $\\Psi$ at first order", 2.86e-8),
        ("Einstein limit of the slope ($\\pi$)", 1.19e-7),
        ("on-shell Gauss–Bonnet action", 4.67e-7),
        ("Smarr relation", 1.45e-6),
        ("near-horizon entropy", 4.06e-6),
        ("coarser radial resolution", 6.89e-5),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ax.barh(labels, vals, color=BLUE, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1e-12, 1e-3)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        exp = math.floor(math.log10(v))
        ax.text(v * 1.6, i, rf"${v / 10**exp:.1f}\times10^{{{exp}}}$",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("worst deviation (smaller is better)")
    save(fig, "fig-checks")


def cost():
    """Agent wall-clock hours and processed tokens by project block.

    Values: prompts/uso-de-tiempo-y-tokens.md, section 2 ('Reparto por hito').
    """
    blocks = [
        ("M0–3  reproduce known solutions", 4.4, 101.558821, 0.747126),
        ("M4  rotating EGB solver + family", 11.2, 507.954588, 2.271301),
        ("M5  neural seed", 3.7, 118.533656, 0.328661),
        (r"M6  potential $\Psi$", 4.5, 168.757196, 0.305427),
        ("M7  reproducible delivery", 1.9, 68.134005, 0.346788),
        ("interim talks and audits", 0.4, 18.708750, 0.208010),
        (r"M8  $\Psi$ at $T\to0$, paper, referees", 8.7, 408.181207, 1.587772),
        ("GitHub + final revisions", 1.8, 23.680529, 0.147377),
    ]
    names = [b[0] for b in blocks][::-1]
    hours = [b[1] for b in blocks][::-1]
    toks = [b[2] for b in blocks][::-1]
    outs = [b[3] for b in blocks][::-1]
    hi = {"M4  rotating EGB solver + family", r"M8  $\Psi$ at $T\to0$, paper, referees"}
    cols = [ORANGE if n in hi else BLUE for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True,
                                 gridspec_kw={"wspace": 0.08})
    a1.barh(names, hours, color=cols, height=0.62)
    a1.set_xlabel("active hours with agents")
    a1.set_xlim(0, 13.5)
    for i, h in enumerate(hours):
        a1.text(h + 0.2, i, f"{h:.1f} h", va="center", fontsize=9)
    a2.barh(names, toks, color=cols, height=0.62)
    a2.set_xlabel("tokens processed (millions)")
    a2.set_xlim(0, 640)
    for i, (t, o) in enumerate(zip(toks, outs)):
        a2.text(t + 8, i, f"{t:.0f} M  ({o:.2f} M written)", va="center", fontsize=8.5)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    save(fig, "fig-cost")


def effort():
    """Compact version of fig-cost for the 'what we learned' slide.

    Values: prompts/uso-de-tiempo-y-tokens.md, section 2 ('Reparto por hito'),
    the same rows as cost() above.
    """
    rows = [
        ("reproduce all known\nsolutions (M0–3)", 4.4, 101.558821, BLUE),
        ("first new object: rotating\nGauss–Bonnet solver (M4)", 11.2, 507.954588, ORANGE),
        (r"$\Psi$ at $T\to0$, paper," + "\nreferee rounds (M8)", 8.7, 408.181207, ORANGE),
    ][::-1]
    ratio = rows[1][2] / rows[2][2]
    assert abs(ratio - 5.0) < 0.05, ratio          # "5x the tokens" on the slide
    fig, ax = plt.subplots(figsize=(4.9, 2.3))
    ax.barh([r[0] for r in rows], [r[2] for r in rows], color=[r[3] for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax.text(r[2] + 10, i, f"{r[2]:.0f} M · {r[1]:.1f} h", va="center", fontsize=10.5)
    ax.set_xlim(0, 800)
    ax.set_xlabel("tokens processed by the agents (millions)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig-effort")


def check_result_numbers():
    """Assert the derived numbers quoted on the result slide.

    mu(0), mu(y_last), y_last, Psi_min, Psi_last: manuscript/numbers.tex
    (\\Nmuzero, \\Nextremaltable last row, \\Nylast, \\Npsimin, \\Npsilast).
    """
    mu0, mu_last, y_last = 2.196887831, 2.888535, 0.5112
    shift = mu_last - mu0
    linear = math.pi * y_last
    assert round(100 * shift / mu0) == 31                    # "+31%"
    assert round(linear / shift, 1) == 2.3                   # "first order: 2.3x the shift"
    assert round((mu0 + linear) / mu_last, 2) == 1.32        # \Nlinearratio is this mass ratio
    assert round(3 * math.pi / 4, 2) == 2.36                 # slope the bound M > 3 pi alpha/4 forces
    print(f"shift {shift:.4f}, first-order shift {linear:.4f}, ratio {linear / shift:.3f}")


if __name__ == "__main__":
    check_result_numbers()
    question()
    checks()
    cost()
    effort()
    print("wrote", sorted(p.name for p in OUT.glob("fig-*")))
