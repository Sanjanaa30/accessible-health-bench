"""
src/ml_baseline.py

Phase 5, Track C — Logistic regression baseline for constraint adherence.

Trains a small interpretable classifier on extracted features (cosine
distances, Jaccard, grounding metrics) to predict whether a response
"adhered" to its stated constraint, using human labels from Phase 6 as
ground truth.

Why this matters for the paper:
  - Provides an interpretable counterpoint to the LLM judges. Reviewers
    can see exactly which features drive the prediction.
  - Reveals which non-LLM signal is the strongest predictor of adherence
    (cosine distance? Jaccard? western-centricity?).
  - Lets us compare LLM-judge accuracy vs. a simple linear model trained
    on hand-crafted features.

Methodology choices, justified:
  - Features standardized via StandardScaler so coefficient magnitudes are
    interpretable (each feature contributes per-standard-deviation, not
    per-raw-unit).
  - LeaveOneOut CV is used because N=30 is too small for k-fold to be
    stable (with k=5, a single fold has only 6 test samples).
  - Wilson 95% CI on accuracy reflects the small-N uncertainty honestly.
  - Only 4-5 features are used to keep ratio of features-to-samples sane.

Output:
  results/ml_baseline.csv          — per-fold predictions
  results/ml_baseline_summary.csv  — summary stats (accuracy, CI, coefficients)
  console summary

Run from repo root:
    python -m src.ml_baseline
    python -m src.ml_baseline --target b1_financial    # train per-RQ
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import JUDGED_DIR, ENRICHED_DIR, RESULTS_DIR, MODELS

# sklearn / numpy imports gated so the rest of the module loads even if
# sklearn isn't installed yet (it's already in requirements.txt).
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
except ImportError as e:
    raise SystemExit(
        "scikit-learn + numpy required. Install with:\n"
        "    pip install scikit-learn numpy"
    ) from e


VALIDATION_DIR = Path(RESULTS_DIR) / "validation"

# Features we extract per response. Kept small intentionally — at N=30
# we want at most ~5 features for stable logistic regression.
FEATURE_NAMES = [
    "cosine_full",
    "cosine_ingredients",
    "jaccard_ingredients",
    "western_centricity",
    "response_length_ratio",
]


# =============================================================
# Feature extraction
# =============================================================
def _load_similarity_index() -> dict[tuple, dict]:
    """Both members of a prompt pair carry the same similarity row, indexed
    by (provider, prompt_id) for both baseline and constrained ids."""
    path = Path(RESULTS_DIR) / "similarity.csv"
    if not path.exists():
        print(f"WARNING: {path} not found. Track B run was required.")
        return {}
    out: dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sig = {k: float(row.get(k) or 0) for k in (
                "cosine_full", "cosine_ingredients",
                "cosine_structural", "jaccard_ingredients",
            )}
            out[(row["provider"], row["baseline_id"])] = sig
            out[(row["provider"], row["constrained_id"])] = sig
    return out


def _load_features_for(provider: str, prompt_id: str,
                       similarity_idx: dict) -> Optional[dict]:
    """Build the feature vector for one (provider, prompt_id) row."""
    enriched_path = Path(ENRICHED_DIR) / provider / f"{prompt_id}.json"
    if not enriched_path.exists():
        return None
    try:
        rec = json.loads(enriched_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    grounding = rec.get("grounding") or {}
    if "_skipped" in grounding:
        return None

    sim = similarity_idx.get((provider, prompt_id), {})
    western = ((grounding.get("wikidata") or {}).get("western_centricity_ratio")) or 0.0
    response_text = rec.get("response_text") or ""
    # Length ratio relative to a reference (1500 chars); just a normalization.
    length_ratio = len(response_text) / 1500.0

    return {
        "cosine_full":          sim.get("cosine_full", 0.0),
        "cosine_ingredients":   sim.get("cosine_ingredients", 0.0),
        "jaccard_ingredients":  sim.get("jaccard_ingredients", 0.0),
        "western_centricity":   float(western),
        "response_length_ratio": length_ratio,
    }


# =============================================================
# Loading human labels — averaged across the two raters when both rated.
# Labels are binarized: yes -> 1, partial -> 1, no -> 0.
# =============================================================
# Branch targets (yes/partial/no DAG verdicts — sparse at N=15)
DAG_BRANCH_TARGETS = {
    "b1_financial": "human_adherence_b1_financial",
    "b2_cultural":  "human_adherence_b2_cultural",
    "b3_lifestyle": "human_adherence_b3_lifestyle",
}

# Dimension targets (1-5 scores — every row scored on every dimension)
DIMENSION_TARGETS = {
    "affordability":   "human_affordability",
    "cultural":        "human_cultural",
    "feasibility":     "human_feasibility",
    "health_accuracy": "human_health_accuracy",
}


def _read_csv_tolerant(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with p.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "ascii", b"", 0, 1,
        f"Could not decode {p} as utf-8 / utf-8-sig / cp1252 / latin-1.",
    )


def _to_int_or_none(s):
    if s is None: return None
    s = str(s).strip()
    if not s: return None
    try:
        v = int(round(float(s)))
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 5 else None


def load_human_labels(rater_paths: list[Path],
                      target: str) -> list[dict]:
    """Return one row per validation sample, with a binarized label.

    Two label modes:

    1. DAG-branch targets ("b1_financial", "b2_cultural", "b3_lifestyle"):
       Label is yes/partial/no per branch. We keep rows where BOTH raters
       gave a yes/partial/no verdict (drops "not_applicable") AND agree
       on the binarization (yes/partial → 1, no → 0). Strict — N tends
       to be small.

    2. Dimension targets ("affordability", "cultural", "feasibility",
       "health_accuracy"):
       Label is the AVERAGE of the two raters' 1-5 scores, binarized at
       >= 4.0 → 1 (good) vs < 4.0 → 0 (not good). Permissive — N=15
       since every row is scored on every dimension. Rows where either
       rater left a cell blank are dropped.
    """
    if target in DAG_BRANCH_TARGETS:
        target_col = DAG_BRANCH_TARGETS[target]
        return _load_dag_labels(rater_paths, target_col)

    if target in DIMENSION_TARGETS:
        target_col = DIMENSION_TARGETS[target]
        return _load_dimension_labels(rater_paths, target_col)

    raise ValueError(
        f"Unknown target: {target!r}. Choose from "
        f"{list(DAG_BRANCH_TARGETS) + list(DIMENSION_TARGETS)}"
    )


def _load_dag_labels(rater_paths, target_col):
    """Strict: both raters yes/partial/no AND agree on binary."""

    rows_by_id: dict[str, dict] = {}
    for path in rater_paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        for row in _read_csv_tolerant(path):
            row_id = row["row_id"]
            verdict = (row.get(target_col) or "").strip().lower()
            if verdict not in {"yes", "partial", "no"}:
                # not_applicable or blank → drop this rater's vote
                continue
            rows_by_id.setdefault(row_id, {
                "row_id":    row_id,
                "provider":  row["provider"],
                "prompt_id": row["prompt_id"],
                "votes":     [],
            })
            rows_by_id[row_id]["votes"].append(verdict)

    # Keep only rows where both raters voted AND agreed (binarized).
    out = []
    for rid, info in rows_by_id.items():
        if len(info["votes"]) != len(rater_paths):
            continue
        binary = {v: 1 if v in {"yes", "partial"} else 0 for v in info["votes"]}
        votes_binary = list(binary.values())
        if len(set(votes_binary)) != 1:
            # raters disagreed on the binary — drop.
            continue
        info["label"] = votes_binary[0]
        out.append(info)
    return out


def _load_dimension_labels(rater_paths, target_col):
    """Permissive: average the two raters' 1-5 scores; binarize at >= 4.0.
    All N=15 rows participate as long as both raters scored that cell."""
    rows_by_id: dict[str, dict] = {}
    for path in rater_paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        for row in _read_csv_tolerant(path):
            rid = row["row_id"]
            score = _to_int_or_none(row.get(target_col))
            if score is None:
                continue
            rows_by_id.setdefault(rid, {
                "row_id":    rid,
                "provider":  row["provider"],
                "prompt_id": row["prompt_id"],
                "scores":    [],
            })
            rows_by_id[rid]["scores"].append(score)

    out = []
    for rid, info in rows_by_id.items():
        if len(info["scores"]) != len(rater_paths):
            continue  # one rater left it blank — drop
        avg = sum(info["scores"]) / len(info["scores"])
        info["avg_score"] = round(avg, 2)
        out.append(info)

    # Adaptive threshold: start at 4.0 ("good"); if all samples are on one
    # side, drop to 3.5 then 3.0 then 2.5. Both classes must exist for
    # logistic regression to train.
    chosen = None
    for thr in (4.0, 3.5, 3.0, 2.5):
        pos = sum(1 for r in out if r["avg_score"] >= thr)
        neg = len(out) - pos
        if pos >= 2 and neg >= 2:
            chosen = thr
            break

    if chosen is None:
        for info in out:
            info["label"] = 1
        return out  # caller will abort on class imbalance

    print(f"  binarization threshold: avg-score >= {chosen}")
    for info in out:
        info["label"] = 1 if info["avg_score"] >= chosen else 0
    return out


# =============================================================
# Wilson 95% CI for a binomial proportion
# =============================================================
def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    n = n_total
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# =============================================================
# Main: train + evaluate
# =============================================================
def train_and_evaluate(target: str, rater_names: list[str]):
    rater_paths = [
        VALIDATION_DIR / f"human_scores_{name}.csv" for name in rater_names
    ]

    print(f"Loading human labels for target = {target}")
    rows = load_human_labels(rater_paths, target)
    print(f"  {len(rows)} rows where both raters agreed")

    if len(rows) < 8:
        print(f"  TOO FEW SAMPLES ({len(rows)}). Logistic regression "
              f"requires at least ~8 to be meaningful. Aborting.")
        return

    similarity_idx = _load_similarity_index()
    feature_vectors = []
    labels = []
    kept_rows = []
    for r in rows:
        feats = _load_features_for(r["provider"], r["prompt_id"], similarity_idx)
        if feats is None:
            continue
        feature_vectors.append([feats[k] for k in FEATURE_NAMES])
        labels.append(r["label"])
        kept_rows.append(r)

    X = np.array(feature_vectors, dtype=float)
    y = np.array(labels, dtype=int)
    print(f"  feature matrix: {X.shape}")
    print(f"  label distribution: 0 (no_adhere): {(y == 0).sum()}, "
          f"1 (adhere): {(y == 1).sum()}")

    if (y == 0).sum() < 2 or (y == 1).sum() < 2:
        print("  CANNOT TRAIN — need at least 2 samples per class.")
        return

    # Standardize features so coefficient magnitudes are comparable.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LeaveOneOut CV: train on N-1, predict the held-out 1, repeat N times.
    loo = LeaveOneOut()
    predictions = []
    truths = []
    fold_records = []

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
        # Refit scaler on the fold's training set (leakage-free)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])
        ytr = y[train_idx]

        if len(set(ytr)) < 2:
            # Test sample's class not represented in train — skip fold.
            continue

        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(Xtr, ytr)
        pred = int(clf.predict(Xte)[0])
        predictions.append(pred)
        truths.append(int(y[test_idx][0]))
        fold_records.append({
            "row_id":    kept_rows[int(test_idx[0])]["row_id"],
            "provider":  kept_rows[int(test_idx[0])]["provider"],
            "prompt_id": kept_rows[int(test_idx[0])]["prompt_id"],
            "true":      int(y[test_idx][0]),
            "pred":      pred,
            "correct":   int(int(y[test_idx][0]) == pred),
        })

    if not predictions:
        print("  No usable folds — class imbalance too severe.")
        return

    n_correct = sum(1 for p, t in zip(predictions, truths) if p == t)
    n_total = len(predictions)
    accuracy = n_correct / n_total
    ci_low, ci_high = wilson_ci(n_correct, n_total)

    # Refit on the full dataset to expose final coefficient magnitudes.
    final_clf = LogisticRegression(max_iter=1000, solver="liblinear")
    final_clf.fit(X_scaled, y)
    coefs = dict(zip(FEATURE_NAMES, final_clf.coef_[0].tolist()))
    intercept = float(final_clf.intercept_[0])

    # Per-class precision/recall (from confusion matrix)
    tp = sum(1 for p, t in zip(predictions, truths) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(predictions, truths) if p == 0 and t == 0)
    fp = sum(1 for p, t in zip(predictions, truths) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, truths) if p == 0 and t == 1)

    def _safe_div(a, b):
        return a / b if b > 0 else 0.0

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Write outputs
    out_dir = Path(RESULTS_DIR)
    fold_path = out_dir / "ml_baseline.csv"
    summary_path = out_dir / "ml_baseline_summary.csv"

    with fold_path.open("w", newline="", encoding="utf-8") as f:
        if fold_records:
            writer = csv.DictWriter(f, fieldnames=list(fold_records[0].keys()))
            writer.writeheader()
            writer.writerows(fold_records)
    print(f"  wrote {fold_path} ({len(fold_records)} fold rows)")

    summary = {
        "target":  target,
        "n_total":        n_total,
        "n_correct":      n_correct,
        "accuracy":       round(accuracy, 3),
        "accuracy_ci_low":  round(ci_low, 3),
        "accuracy_ci_high": round(ci_high, 3),
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "intercept":      round(intercept, 4),
        **{f"coef_{k}": round(v, 4) for k, v in coefs.items()},
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"  wrote {summary_path}")

    # Console
    print("\n" + "=" * 78)
    print(f"PHASE 5 TRACK C — LOGISTIC REGRESSION ({target})")
    print("=" * 78)
    print(f"  N samples:    {n_total}")
    print(f"  Accuracy:     {accuracy:.3f}  "
          f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  Precision:    {precision:.3f}")
    print(f"  Recall:       {recall:.3f}")
    print(f"  F1:           {f1:.3f}")
    print(f"  Confusion:    TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"\n  Standardized coefficients (effect per +1 standard-deviation in feature):")
    for k, v in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        sign = "+" if v >= 0 else "-"
        print(f"    {k:24s}  {sign}{abs(v):.3f}")
    print(f"  intercept:                {'+' if intercept >= 0 else '-'}"
          f"{abs(intercept):.3f}")

    print(f"\n  Interpretation:")
    print(f"    A positive coefficient means: as this feature increases,")
    print(f"    the model predicts adherence is MORE likely.")
    print(f"    The largest-magnitude feature is the strongest predictor.")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Track C — logistic baseline")
    parser.add_argument(
        "--target", default="affordability",
        choices=(list(DAG_BRANCH_TARGETS) + list(DIMENSION_TARGETS)),
        help=("DAG branch (small N, strict) OR dimension (uses all 15 samples, "
              "binarized at avg-score >= 4)"),
    )
    parser.add_argument("--rater-names", nargs=2, default=["sanjana", "snigdha"])
    args = parser.parse_args()

    train_and_evaluate(args.target, args.rater_names)


if __name__ == "__main__":
    main()
