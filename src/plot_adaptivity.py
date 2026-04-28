"""
src/plot_adaptivity.py

Phase 5, Step 2 — Visualize the similarity signals as Adaptivity Curves.

Reads results/similarity.csv (produced by src/similarity.py) and produces:

  results/figures/adaptivity_curves.png  + .pdf
      Box plots per (model × category) with scatter overlay, four sub-panels
      (one per distance signal). The headline figure for the paper.

  results/figures/distance_distributions.png  + .pdf
      Per-model histogram of the four distance signals — shows whether each
      model has bimodal adaptation (some pairs adapt fully, some don't).
      Step histograms used so overlapping signals stay readable.

  results/adaptivity_summary.csv
      Mean / std / median / IQR per (provider, category, signal). Goes in
      the paper as Table 2.

This is the de-risking step. If the box plots show flat distributions near
zero across all models, the headline finding is "uniform model rigidity."
If they show meaningful spread, the rest of Phase 5 (judges) is justified.

The end-of-run "diagnostic" line is a soft heuristic, NOT a final verdict.
Always read the histograms and the summary CSV before drawing conclusions.

Run from repo root:
    python -m src.plot_adaptivity
    python -m src.plot_adaptivity --in results/similarity.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib required: pip install matplotlib") from e

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_DISPLAY_NAMES, RESULTS_DIR

DEFAULT_INPUT = Path(RESULTS_DIR) / "similarity.csv"
FIGURES_DIR = Path(RESULTS_DIR) / "figures"

CATEGORY_ORDER = ["financial", "cultural", "lifestyle"]
SIGNAL_LABELS = {
    "cosine_full":         "Full response (Sentence-BERT)",
    "cosine_ingredients":  "Ingredient list (Sentence-BERT)",
    "cosine_structural":   "Structural digest (Sentence-BERT)",
    "jaccard_ingredients": "Ingredient set (Jaccard)",
}
SIGNAL_KEYS = list(SIGNAL_LABELS.keys())

# Diagnostic-line thresholds applied to the raw cosine_full distribution.
# Values were chosen as a priori sensible defaults; recalibrate against a
# small hand-labeled sample before treating them as load-bearing.
RIGID_MEAN_MAX = 0.12
RIGID_CELL_SPREAD_MAX = 0.04
ADAPTIVE_MEAN_MIN = 0.25
ADAPTIVE_CELL_SPREAD_MIN = 0.06


# =============================================================
# Loading
# =============================================================
def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src.similarity first.")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in SIGNAL_KEYS:
                if k in row and row[k] not in (None, ""):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = None
            rows.append(row)
    return rows


def category_sort_key(cat: str) -> tuple[int, str]:
    """Sort known categories first in CATEGORY_ORDER, unknowns alphabetically."""
    try:
        return (0, str(CATEGORY_ORDER.index(cat)))
    except ValueError:
        return (1, cat)


# =============================================================
# Aggregation
# =============================================================
def summarize(rows: list[dict]) -> dict:
    """Build per-(provider, category, signal) summary stats."""
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        for sig in SIGNAL_KEYS:
            v = r.get(sig)
            if v is None:
                continue
            bucket[(r["provider"], r["category"], sig)].append(v)

    summary = {}
    for (prov, cat, sig), vals in bucket.items():
        a = np.array(vals)
        summary[(prov, cat, sig)] = {
            "n":      len(a),
            "mean":   float(np.mean(a)),
            "std":    float(np.std(a)),
            "median": float(np.median(a)),
            "q1":     float(np.percentile(a, 25)),
            "q3":     float(np.percentile(a, 75)),
            "min":    float(np.min(a)),
            "max":    float(np.max(a)),
        }
    return summary


def write_summary_csv(summary: dict, path: Path):
    rows = []
    for (prov, cat, sig), stats in sorted(summary.items()):
        rows.append({
            "provider": prov,
            "category": cat,
            "signal":   sig,
            "n":        stats["n"],
            "mean":     round(stats["mean"], 4),
            "std":      round(stats["std"], 4),
            "median":   round(stats["median"], 4),
            "q1":       round(stats["q1"], 4),
            "q3":       round(stats["q3"], 4),
            "min":      round(stats["min"], 4),
            "max":      round(stats["max"], 4),
        })
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def _save_figure(fig, out_path: Path):
    """Save PNG (display) and PDF (paper) side-by-side."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {out_path} (+ .pdf)")


# =============================================================
# Plotting — adaptivity curves
# =============================================================
def plot_adaptivity(rows: list[dict], out_path: Path):
    providers_present = sorted({r["provider"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    box_width = 0.6
    box_gap = 0.7
    group_gap = 1.0

    category_colors = {
        "financial": "#4C72B0",
        "cultural":  "#DD8452",
        "lifestyle": "#55A467",
    }

    # Deterministic jitter — fixed seed so every run produces an identical plot.
    rng = np.random.default_rng(42)

    for ax_idx, signal in enumerate(SIGNAL_KEYS):
        ax = axes[ax_idx]
        positions = []
        data_groups = []
        colors = []

        x = 0.0
        provider_centers = []
        for provider in providers_present:
            group_start = x
            last_box_pos = group_start
            for cat in CATEGORY_ORDER:
                vals = [r[signal] for r in rows
                        if r["provider"] == provider
                        and r["category"] == cat
                        and r.get(signal) is not None]
                if vals:
                    positions.append(x)
                    data_groups.append(vals)
                    colors.append(category_colors[cat])
                    last_box_pos = x
                x += box_gap
            provider_centers.append((group_start + last_box_pos) / 2)
            x += group_gap

        if not data_groups:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(SIGNAL_LABELS[signal])
            continue

        bp = ax.boxplot(
            data_groups,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        # Scatter overlay (individual prompt pairs)
        for pos, vals, color in zip(positions, data_groups, colors):
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals, s=10, alpha=0.5, color=color,
                edgecolors="black", linewidths=0.3,
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xticks(provider_centers)
        ax.set_xticklabels(
            [MODEL_DISPLAY_NAMES.get(p, p) for p in providers_present],
            fontsize=9,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(SIGNAL_LABELS[signal], fontsize=11)
        # With sharey=True, only the left column needs a y-label.
        if ax_idx % 2 == 0:
            ax.set_ylabel("Distance (0=identical, 1=fully different)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6, edgecolor="black")
        for c in category_colors.values()
    ]
    fig.legend(handles, list(category_colors.keys()),
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02),
               frameon=False, fontsize=10)

    fig.suptitle(
        "Adaptivity Curves — distance from baseline to constrained response",
        y=1.06, fontsize=13,
    )
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


# =============================================================
# Plotting — distance distributions per model
# =============================================================
def plot_distributions(rows: list[dict], out_path: Path):
    providers_present = sorted({r["provider"] for r in rows})
    n_providers = len(providers_present)

    fig, axes = plt.subplots(n_providers, 1, figsize=(11, 2.6 * n_providers),
                             sharex=True)
    if n_providers == 1:
        axes = [axes]

    signal_colors = {
        "cosine_full":         "#4C72B0",
        "cosine_ingredients":  "#DD8452",
        "cosine_structural":   "#55A467",
        "jaccard_ingredients": "#C44E52",
    }

    for prov_i, provider in enumerate(providers_present):
        ax = axes[prov_i]
        for sig in SIGNAL_KEYS:
            vals = [r[sig] for r in rows
                    if r["provider"] == provider
                    and r.get(sig) is not None]
            if vals:
                # Step histograms keep overlapping signals legible.
                ax.hist(
                    vals, bins=20, alpha=0.85,
                    label=SIGNAL_LABELS[sig],
                    color=signal_colors[sig],
                    histtype="step", linewidth=1.6,
                )
        ax.set_title(MODEL_DISPLAY_NAMES.get(provider, provider), fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylabel("# prompt pairs")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        if prov_i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Distance from baseline")
    fig.suptitle(
        "Distance distribution per model — bimodality reveals split adaptation",
        fontsize=12,
    )
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


# =============================================================
# Diagnostic line (soft, not a verdict)
# =============================================================
def print_diagnostic(rows: list[dict], summary: dict):
    print("\n" + "=" * 70)
    print("ADAPTIVITY DIAGNOSTIC (soft heuristic — read the figures too)")
    print("=" * 70)

    # Raw distribution of cosine_full across every prompt pair × provider.
    all_full = [r["cosine_full"] for r in rows
                if r.get("cosine_full") is not None]
    if not all_full:
        print("No cosine_full values to diagnose.")
        return

    overall_mean = float(np.mean(all_full))
    overall_median = float(np.median(all_full))

    # Cell-level (provider × category) means used to measure between-cell spread.
    cell_means = [s["mean"] for (p, c, sig), s in summary.items()
                  if sig == "cosine_full"]
    cell_spread = float(np.std(cell_means)) if cell_means else 0.0

    print(f"  cosine_full pooled mean    : {overall_mean:.3f}")
    print(f"  cosine_full pooled median  : {overall_median:.3f}")
    print(f"  between-cell std (provider x category): {cell_spread:.3f}")
    print(f"  thresholds: rigid <{RIGID_MEAN_MAX}/{RIGID_CELL_SPREAD_MAX}, "
          f"adaptive >{ADAPTIVE_MEAN_MIN}/{ADAPTIVE_CELL_SPREAD_MIN}")

    if overall_mean < RIGID_MEAN_MAX and cell_spread < RIGID_CELL_SPREAD_MAX:
        print("\n>>> hint: uniform rigidity")
        print("  Pooled distance is small AND models barely differ across cells.")
        print("  Likely headline: 'no model adapts meaningfully'.")
        print("  Phase 5 judges still useful for grading the (small) "
              "differences that exist.")
    elif overall_mean > ADAPTIVE_MEAN_MIN and cell_spread > ADAPTIVE_CELL_SPREAD_MIN:
        print("\n>>> hint: meaningful adaptation with model variance")
        print("  Pooled distance is large AND models differ across cells.")
        print("  Adaptivity Curve is a strong headline figure.")
        print("  Judges (Phase 5 next steps) explain WHY models differ.")
    else:
        print("\n>>> hint: moderate / mixed signal")
        print("  Some adaptation, some rigidity, OR uniform spread.")
        print("  Check distance_distributions.png for bimodality before deciding.")
        print("  Judges (Phase 5 next steps) needed to disambiguate.")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot adaptivity curves")
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_INPUT),
                        help="Path to similarity.csv")
    parser.add_argument("--out-dir", default=str(FIGURES_DIR),
                        help="Output directory for figures")
    args = parser.parse_args()

    in_path = Path(args.input_path)
    out_dir = Path(args.out_dir)

    print(f"Loading {in_path}...")
    rows = load_rows(in_path)
    print(f"  {len(rows)} rows")

    if not rows:
        print("No rows. Run src.similarity first.")
        return

    summary = summarize(rows)

    print("\nWriting outputs...")
    write_summary_csv(summary, Path(RESULTS_DIR) / "adaptivity_summary.csv")
    plot_adaptivity(rows, out_dir / "adaptivity_curves.png")
    plot_distributions(rows, out_dir / "distance_distributions.png")

    print_diagnostic(rows, summary)


if __name__ == "__main__":
    main()
