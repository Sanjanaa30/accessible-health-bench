"""
src/generate_phase7_figures.py

Phase 7 visualizations — produces 7 figures total (2 per RQ + 1 overall
scorecard) from the CSVs already in results/. Each figure is saved as
PNG (display) and PDF (paper-quality vector).

Figures produced:
  results/figures/rq1_a_ingredient_vs_cost.png/.pdf
  results/figures/rq1_b_affordability_ranking.png/.pdf
  results/figures/rq2_a_western_centricity_predictor.png/.pdf
  results/figures/rq2_b_adapt_vs_quality.png/.pdf
  results/figures/rq3_a_score_distribution.png/.pdf
  results/figures/rq3_b_pairwise_by_rq.png/.pdf
  results/figures/overall_scorecard.png/.pdf

Run from repo root:
    python -m src.generate_phase7_figures
"""

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, MODEL_DISPLAY_NAMES

FIGURES_DIR = Path(RESULTS_DIR) / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colour palette for providers across all figures
PROVIDER_COLOR = {
    "deepseek":  "#2a9d8f",  # green
    "anthropic": "#e07a4d",  # orange
    "openai":    "#3a6ea5",  # blue
    "groq":      "#888888",  # grey
}
DISPLAY = {
    "deepseek":  "DeepSeek",
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "groq":      "Groq",
}


# =============================================================
# Helpers
# =============================================================
def _save(fig, name: str):
    """Save figure as both PNG and PDF in results/figures/."""
    png = FIGURES_DIR / f"{name}.png"
    pdf = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png} (+ .pdf)")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    n = total
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _aggregate_arena_winrates(rows: list[dict]) -> dict:
    """Aggregate arena_matrix.csv into per-(provider, dimension) win rates
    plus Wilson 95% CIs. Each pair contributes wins to both providers."""
    by_provider: dict[tuple, dict] = defaultdict(lambda: {"wins": 0, "decided": 0})
    for r in rows:
        x, y, dim = r["provider_x"], r["provider_y"], r["dimension"]
        n_x = int(r["n_x_wins"]); n_y = int(r["n_y_wins"])
        decided = n_x + n_y
        by_provider[(x, dim)]["wins"] += n_x
        by_provider[(x, dim)]["decided"] += decided
        by_provider[(y, dim)]["wins"] += n_y
        by_provider[(y, dim)]["decided"] += decided

    out = {}
    for (prov, dim), d in by_provider.items():
        if d["decided"] == 0:
            continue
        rate = d["wins"] / d["decided"]
        lo, hi = _wilson_ci(d["wins"], d["decided"])
        out[(prov, dim)] = {
            "winrate": rate,
            "ci_low": lo,
            "ci_high": hi,
            "wins": d["wins"],
            "decided": d["decided"],
        }
    return out


# =============================================================
# Figure 1A — Ingredients replaced vs cost quantified
# =============================================================
def fig_1a():
    """Single side-by-side bar pair showing the gap between ingredient
    replacement (~80%) and cost quantification (~11%) on financial
    constrained prompts."""
    # Compute Jaccard from similarity.csv (financial-constrained only)
    sim = _read_csv(Path(RESULTS_DIR) / "similarity.csv")
    fin_jaccards = [
        float(r["jaccard_ingredients"])
        for r in sim
        if r.get("category") == "financial"
    ]
    jaccard_pct = (sum(fin_jaccards) / len(fin_jaccards)) * 100 if fin_jaccards else 0

    # Compute % cost-quantified from coverage_report.csv (constrained-financial)
    cov = _read_csv(Path(RESULTS_DIR) / "coverage_report.csv")
    fin_constr = [r for r in cov
                  if r.get("category") == "financial"
                  and r.get("variant") == "constrained"]
    n_total = len(fin_constr)
    n_quantified = sum(
        1 for r in fin_constr
        if r.get("cost_period_status") in {"normalized", "per_meal_extrapolated"}
    )
    cost_pct = (n_quantified / n_total) * 100 if n_total else 0

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Ingredients replaced\n(Jaccard, financial-\nconstrained)",
              "Total cost quantified\n(extractable from\nresponse)"]
    values = [jaccard_pct, cost_pct]
    colors = ["#3a6ea5", "#e07a4d"]
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="black",
                  linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Percentage of constrained-financial responses (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 1A — Models change ingredients but rarely commit to a real cost",
                 fontsize=11)
    ax.text(0.5, -0.18,
            f"Across {n_total} constrained-financial responses (n quantified = "
            f"{n_quantified})",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "rq1_a_ingredient_vs_cost")


# =============================================================
# Figure 1B — Affordability quality ranking with CIs
# =============================================================
def fig_1b():
    """Horizontal bar chart of per-provider affordability arena win rates
    sorted descending, with Wilson 95% CI error bars."""
    arena = _read_csv(Path(RESULTS_DIR) / "arena_matrix.csv")
    agg = _aggregate_arena_winrates(arena)

    rows = []
    for prov in ("deepseek", "anthropic", "openai", "groq"):
        d = agg.get((prov, "affordability"))
        if d is None:
            continue
        rows.append((prov, d["winrate"], d["ci_low"], d["ci_high"]))

    rows.sort(key=lambda x: x[1])  # ascending so largest is on top in barh
    providers = [DISPLAY[r[0]] for r in rows]
    rates = [r[1] * 100 for r in rows]
    err_low = [(r[1] - r[2]) * 100 for r in rows]
    err_high = [(r[3] - r[1]) * 100 for r in rows]
    colors = [PROVIDER_COLOR[r[0]] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(providers, rates,
                   xerr=[err_low, err_high],
                   color=colors, edgecolor="black", linewidth=0.6,
                   error_kw={"ecolor": "black", "capsize": 4, "capthick": 1})
    for bar, val in zip(bars, rates):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")

    ax.axvline(50, linestyle="--", color="grey", alpha=0.7, linewidth=1)
    ax.text(50, len(providers) - 0.4, "  even (50%)", color="grey",
            fontsize=9, va="top")

    ax.set_xlabel("ArenaGEval pairwise affordability win rate (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Figure 1B — Affordability quality differs sharply across providers",
                 fontsize=11)
    ax.text(0.5, -0.16,
            "Error bars: Wilson 95% confidence intervals on decided games (ties excluded)",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "rq1_b_affordability_ranking")


# =============================================================
# Figure 2A — Western-centricity is the strongest predictor
# =============================================================
def fig_2a():
    """Horizontal bar chart of standardized logistic-regression coefficients
    for the cultural target, sorted by absolute magnitude. Negative bars
    red, positive blue."""
    rows = _read_csv(Path(RESULTS_DIR) / "ml_baseline_summary_cultural.csv")
    if not rows:
        print("  [skip] ml_baseline_summary_cultural.csv missing or empty")
        return
    summary = rows[0]
    coefs = {
        k.replace("coef_", ""): float(summary[k])
        for k in summary if k.startswith("coef_")
    }

    # Sort by absolute magnitude descending so the strongest is at the top
    items = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    # Reverse for barh so top item appears on top
    items = list(reversed(items))
    features = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]
    colors = ["#c44e52" if v < 0 else "#3a6ea5" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(features, values, color=colors, edgecolor="black",
                   linewidth=0.6)
    for bar, val in zip(bars, values):
        x_text = val + (0.04 if val >= 0 else -0.04)
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"β = {val:+.2f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10, fontweight="bold",
                color="black")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized logistic-regression coefficient (β)")
    ax.set_title("Figure 2A — Western-centricity is the strongest predictor of "
                 "cultural non-adherence",
                 fontsize=11)
    ax.text(0.5, -0.18,
            "Negative β (red) = feature increase predicts LESS adherence; "
            "trained on N=15 with leave-one-out CV.",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    # Slight x-padding so labels don't touch the axis edge
    xmin = min(values) - 0.25
    xmax = max(values) + 0.25
    ax.set_xlim(xmin, xmax)
    plt.tight_layout()
    _save(fig, "rq2_a_western_centricity_predictor")


# =============================================================
# Figure 2B — Adaptation magnitude vs pairwise quality
# =============================================================
def fig_2b():
    """Paired bar chart: each provider gets two bars — Sentence-BERT
    cosine_full mean on cultural-constrained (left), arena cultural win
    rate (right). Shows that adapting more does NOT mean adapting better."""
    sim = _read_csv(Path(RESULTS_DIR) / "similarity.csv")
    cosine_by_prov: dict[str, list[float]] = defaultdict(list)
    for r in sim:
        if r.get("category") == "cultural":
            cosine_by_prov[r["provider"]].append(float(r["cosine_full"]))

    arena = _read_csv(Path(RESULTS_DIR) / "arena_matrix.csv")
    agg = _aggregate_arena_winrates(arena)

    providers = ["deepseek", "anthropic", "openai", "groq"]
    cosines = [(sum(cosine_by_prov[p]) / len(cosine_by_prov[p])) * 100
               if cosine_by_prov[p] else 0
               for p in providers]
    arena_rates = [agg.get((p, "cultural"), {"winrate": 0})["winrate"] * 100
                   for p in providers]

    x = np.arange(len(providers))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars1 = ax.bar(x - width / 2, cosines, width,
                   label="Adaptation magnitude\n(cosine × 100, Sentence-BERT)",
                   color="#9b8cc4", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width / 2, arena_rates, width,
                   label="Pairwise quality\n(ArenaGEval win rate %)",
                   color="#2a9d8f", edgecolor="black", linewidth=0.6)

    for bars, vals in ((bars1, cosines), (bars2, arena_rates)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[p] for p in providers])
    ax.set_ylabel("Value (cosine × 100 OR win rate %)")
    ax.set_ylim(0, max(max(cosines), max(arena_rates)) * 1.25)
    ax.set_title("Figure 2B — Adapting more isn't the same as adapting better "
                 "(cultural prompts)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(0.5, -0.16,
            "Anthropic adapts MOST (left bars) but ranks 3rd in pairwise "
            "quality (right bars). DeepSeek wins despite less adaptation.",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "rq2_b_adapt_vs_quality")


# =============================================================
# Figure 3A — Score distribution by dimension (ceiling effect)
# =============================================================
def fig_3a():
    """Stacked horizontal bar chart showing the % of constrained responses
    at each integer score (1-5) for cultural, adherence, and feasibility.
    Cultural and adherence pile at 5; feasibility shows real spread."""
    rows = _read_csv(Path(RESULTS_DIR) / "scores.csv")
    constrained = [r for r in rows if r.get("variant") == "constrained"]

    dims = [
        ("score_cultural",      "Cultural"),
        ("score_adherence",     "Adherence"),
        ("score_feasibility",   "Feasibility"),
    ]
    counts = {}
    for col, label in dims:
        c = Counter()
        for r in constrained:
            v = r.get(col)
            if not v:
                continue
            try:
                s = int(round(float(v)))
            except (TypeError, ValueError):
                continue
            if 1 <= s <= 5:
                c[s] += 1
        counts[label] = c

    cmap = plt.get_cmap("Blues")
    score_colors = {
        1: cmap(0.18), 2: cmap(0.36), 3: cmap(0.54),
        4: cmap(0.72), 5: cmap(0.90),
    }

    labels = [label for _, label in dims]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bottoms = np.zeros(len(labels))
    for score in (1, 2, 3, 4, 5):
        pcts = []
        for label in labels:
            c = counts[label]
            total = sum(c.values()) or 1
            pcts.append(c.get(score, 0) / total * 100)
        bars = ax.barh(labels, pcts, left=bottoms,
                       color=score_colors[score],
                       edgecolor="white", linewidth=0.6,
                       label=f"Score {score}")
        for bar, p in zip(bars, pcts):
            if p > 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{p:.1f}%", ha="center", va="center", fontsize=9,
                        color="white" if score >= 4 else "black",
                        fontweight="bold")
        bottoms += np.array(pcts)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of constrained responses (%)")
    ax.set_title("Figure 3A — Feasibility scores spread across the full 1-5 "
                 "range, unlike cultural and adherence", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5,
              frameon=False, fontsize=9)
    ax.text(0.5, -0.32,
            "Cultural & adherence are ceiling-pressed (94-98% at 5/5); "
            "feasibility produces a real distribution.",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.invert_yaxis()  # so cultural shows on top
    plt.tight_layout()
    _save(fig, "rq3_a_score_distribution")


# =============================================================
# Figure 3B — Pairwise win rates across all 3 RQs
# =============================================================
def fig_3b():
    """Grouped bar chart: 3 groups (one per RQ), each with 4 bars (one per
    provider). Highlights that lifestyle has the widest provider spread."""
    arena = _read_csv(Path(RESULTS_DIR) / "arena_matrix.csv")
    agg = _aggregate_arena_winrates(arena)

    rqs = [("affordability", "Affordability"),
           ("cultural",      "Cultural"),
           ("feasibility",   "Feasibility")]
    providers = ["deepseek", "anthropic", "openai", "groq"]

    x = np.arange(len(rqs))
    width = 0.20

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, prov in enumerate(providers):
        rates = [
            agg.get((prov, dim), {"winrate": 0})["winrate"] * 100
            for dim, _ in rqs
        ]
        offset = (i - (len(providers) - 1) / 2) * width
        bars = ax.bar(x + offset, rates, width,
                      label=DISPLAY[prov],
                      color=PROVIDER_COLOR[prov],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in rqs])
    ax.set_ylabel("ArenaGEval pairwise win rate (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("Figure 3B — Lifestyle is where models differ MOST in "
                 "pairwise quality", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(0.5, -0.16,
            "Feasibility shows the widest spread (DeepSeek 87.2% vs "
            "Groq 19.4% = 67.8 percentage points).",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "rq3_b_pairwise_by_rq")


# =============================================================
# Figure 5 — Phase 6 within-1-point agreement
# =============================================================
def fig_5_human_validation():
    """Phase 6 human validation visualization. Two grouped bars per
    dimension: inter-human within-1-point agreement vs LLM-judge-vs-
    human-average within-1-point agreement, on the same 1-5 scale."""
    dimensions = ["Affordability", "Cultural", "Feasibility"]
    inter_human = [73, 40, 87]
    judge_vs_human = [80, 60, 93]

    x = np.arange(len(dimensions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars1 = ax.bar(x - width / 2, inter_human, width,
                   label="Inter-human (the two authors)",
                   color="#9bc4e2", edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width / 2, judge_vs_human, width,
                   label="LLM judge vs human average",
                   color="#2a9d8f", edgecolor="black", linewidth=0.6)

    for bars, vals in ((bars1, inter_human), (bars2, judge_vs_human)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.axhline(50, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylabel("Percentage of responses within 1 point of agreement (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Figure 5 — Human raters and LLM judges agree closely on most responses",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax.text(0.5, -0.16,
            "Phase 6 validation, N = 15. Higher bar = more agreement. "
            "Feasibility has the highest agreement (concrete anchors); "
            "cultural the lowest (most subjective).",
            ha="center", va="top", transform=ax.transAxes, fontsize=9,
            color="grey", style="italic")
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save(fig, "rq_phase6_human_validation")


# =============================================================
# Overall Figure — Bias & Accessibility Scorecard (heatmap)
# =============================================================
def fig_overall():
    """4x3 heatmap (providers x RQs) of pairwise arena win rates. The
    project's headline figure — answers the main RQ at a glance."""
    arena = _read_csv(Path(RESULTS_DIR) / "arena_matrix.csv")
    agg = _aggregate_arena_winrates(arena)

    providers = ["deepseek", "anthropic", "openai", "groq"]
    rqs = ["affordability", "cultural", "feasibility"]

    data = np.array([
        [agg.get((p, d), {"winrate": 0})["winrate"] * 100 for d in rqs]
        for p in providers
    ])

    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    # Compute display labels with row averages appended to provider names
    y_labels = [f"{DISPLAY[p]}\n(avg {row_means[i]:.1f}%)"
                for i, p in enumerate(providers)]
    x_labels = [f"{d.title()}\n(avg {col_means[j]:.1f}%)"
                for j, d in enumerate(rqs)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(rqs)))
    ax.set_xticklabels(x_labels, fontsize=10.5)
    ax.set_yticks(range(len(providers)))
    ax.set_yticklabels(y_labels, fontsize=10.5)

    # Annotate each cell with the percentage
    for i in range(len(providers)):
        for j in range(len(rqs)):
            v = data[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if v < 45 else "black",
                    fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("ArenaGEval pairwise win rate (%)", fontsize=10)

    ax.set_title("AccessibleHealthBench — Bias & Accessibility Scorecard\n"
                 "(higher = model produces more accessible / less biased advice)",
                 fontsize=12, pad=14)
    fig.text(
        0.5, 0.02,
        "Cell values: ArenaGEval pairwise win rates over 1080 comparisons "
        "(60 prompts x 6 model pairs x 3 dimensions).\n"
        "Row averages = provider performance across all 3 RQs; "
        "column averages = average across all providers per RQ.",
        ha="center", fontsize=9, color="grey", style="italic",
    )
    plt.subplots_adjust(bottom=0.22, left=0.18, right=0.92)
    _save(fig, "overall_scorecard")


# =============================================================
# Main
# =============================================================
def main():
    print(f"Generating Phase 7 figures into {FIGURES_DIR}/")
    print()

    figures = [
        ("RQ1 Figure A", fig_1a),
        ("RQ1 Figure B", fig_1b),
        ("RQ2 Figure A", fig_2a),
        ("RQ2 Figure B", fig_2b),
        ("RQ3 Figure A", fig_3a),
        ("RQ3 Figure B", fig_3b),
        ("Phase 6 human validation", fig_5_human_validation),
        ("Overall Scorecard", fig_overall),
    ]
    for name, fn in figures:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
        print()

    print(f"Done. {len(figures)} figures attempted.")
    print(f"Output: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
