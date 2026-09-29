import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import marimo as mo
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / "short-summary-codex/.cache/matplotlib"))
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    return Line2D, hashlib, importlib, json, mo, np, plt, root


@app.cell
def _(hashlib, importlib, json, np, root):
    source = root / "references/data/1010.0860v1-figure-1b.png"
    family_path = root / "results/egb-rotating-family.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "a5760ea5b2d2ae6c397560bef6431e55acd37e72fc31ce7b10a1ea0006673543"
    family = json.loads(family_path.read_text())
    profiles = family["profiles"]
    assert [p["alpha_gb"] for p in profiles] == [0.0025, 0.75]
    assert all(p["q"] == 0.33 for p in profiles)
    assert all(p["r"][0] == 1.0 for p in profiles)
    spec = importlib.util.spec_from_file_location("paper_extraction", root / "experiments/egb_rotating_paper_profiles.py")
    extraction = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extraction)
    readings = extraction.extract(source)
    high = profiles[1]
    rx = np.log10(high["r"])
    for _row in readings:
        values = -np.array(high["b"]) if _row["field"] == "minus_b" else np.array(high[_row["field"]])
        position = _row["log10_r"]
        computed = float(np.interp(position, rx, values))
        xeffect = max(abs(float(np.interp(position + sign/_row["x_pixels_per_unit"], rx, values))-computed) for sign in [-1, 1])
        tolerance = 2/_row["y_pixels_per_unit"] + xeffect
        _row.update(computed=computed, tolerance=tolerance, absolute_difference=abs(computed-_row["value"]))
        assert _row["absolute_difference"] <= tolerance
    assert len(readings) == 10
    saved = json.loads((root / "results/egb-rotating-paper-profiles.json").read_text())
    for actual, archived in zip(readings, saved["rows"]):
        assert actual["field"] == archived["field"]
        for key in ["value", "computed", "tolerance", "absolute_difference"]:
            assert abs(actual[key]-archived[key]) < 1e-12
    out = root / "short-summary-codex"
    receipt = ("PASS reference-image SHA-256 matches archived input\n"
               "PASS parameters: r_H=1, Omega_H=0.33, alpha_paper=4*alpha\n"
               "PASS 10/10 separated digitised values within archived pixel tolerances\n"
               "PASS re-extraction agrees with archived comparison to 1e-12\n"
               "NOT CHECKED: new BVP solves; quantitative digitisation of alpha_paper=0.01\n"
               "OPEN: separate ergosurface-radius discrepancy at alpha_paper=2\n")
    (out / "comparison-checks.txt").write_text(receipt)
    provenance = {"source_url": "https://arxiv.org/html/1010.0860v1/profiles-alpha.png", "figure": "Figure 1(b)",
                  "inputs": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [source, family_path, root / "experiments/egb_rotating_paper_profiles.py", Path(__file__)]},
                  "readings": readings, "scope": "Replot archived numerical profiles; repeat pixel comparison, not BVP solves."}
    (out / "comparison-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(receipt, end="")
    return out, profiles, readings, source


@app.cell
def _(Line2D, np, out, plt, profiles, readings, source):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    figure = plt.figure(figsize=(9.0, 3.6), layout="constrained")
    paper_ax, computed_ax = figure.subplots(1, 2)
    paper_ax.imshow(plt.imread(source))
    paper_ax.set_axis_off()
    paper_ax.set_title("Published: arXiv:1010.0860v1, Fig. 1(b)", fontsize=10)
    fields = [("h_over_r2", r"$h/r^2$", "#228833", 1), ("f", r"$f$", "#cc6600", 1),
              ("w", r"$w$", "#8844aa", 1), ("b", r"$-b$", "#0077bb", -1)]
    for profile in profiles:
        for field, label, colour, sign in fields:
            computed_ax.plot(np.log10(profile["r"]), sign*np.array(profile[field]), color=colour,
                             linestyle="--" if profile["alpha_gb"] < 0.1 else "-", linewidth=1.5)
    for _row in readings:
        colour = dict(h_over_r2="#228833", f="#cc6600", w="#8844aa", minus_b="#0077bb")[_row["field"]]
        computed_ax.plot(_row["log10_r"], _row["value"], 'o', color=colour, markerfacecolor='white', markersize=4, markeredgewidth=0.9)
    for xpos, ypos, label, colour in [(0.75,1.08,r"$h/r^2$","#228833"),(.80,.78,r"$f$","#cc6600"),(.75,.08,r"$w$","#8844aa"),(.73,-.78,r"$-b$","#0077bb")]:
        computed_ax.text(xpos,ypos,label,color=colour,fontsize=10)
    computed_ax.axvline(0, color="#777777", linestyle=":", linewidth=.8)
    computed_ax.set(xlim=(-.15,1.25), ylim=(-1.05,1.25), xlabel=r"$\log_{10}(r/r_H)$", yticks=[-1,-.5,0,.5,1], xticks=[0,.25,.5,.75,1,1.25])
    computed_ax.grid(alpha=.15)
    computed_ax.set_title("This project: archived numerical profiles", fontsize=10)
    computed_ax.legend(handles=[Line2D([],[],color="#333333",linestyle='-',label=r"$\alpha_{\rm paper}=3$"),
                               Line2D([],[],color="#333333",linestyle='--',label=r"$\alpha_{\rm paper}=0.01$")],
                       loc="lower left", bbox_to_anchor=(0,-.27), ncol=2, frameon=False, fontsize=8)
    figure.savefig(out / "figures/fig-reference-comparison.pdf", bbox_inches="tight")
    figure.savefig(out / "figures/fig-reference-comparison.png", dpi=180, bbox_inches="tight")
    figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reproduction of Fig. 1(b), arXiv:1010.0860v1.**
    The left panel is the original image (Brihaye, Kleihaus, Kunz and Radu;
    https://arxiv.org/html/1010.0860v1/profiles-alpha.png).
    The right panel uses the project's archived BVP profiles with the same axes,
    radial range, horizon radius and angular velocity. Its colours label the four
    functions; solid/dashed lines identify the two couplings. Open circles are
    ten separated readings from the paper's red, alpha_paper=3 curves.
    Their tolerance is ±1 horizontal and ±2 vertical pixels, not author error bars.
    The merged f/w crossing is excluded by the archived extraction protocol.
    The low-coupling comparison is visual only. The original numerical data of
    the authors are unavailable; this is a comparison to their published image.
    The separate ergosurface discrepancy at alpha_paper=2 remains unresolved.
    No field equations were re-solved for this figure.
    """)
    return


if __name__ == "__main__":
    app.run()
