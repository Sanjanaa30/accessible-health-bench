"""
src/arena_eval.py

Phase 5, Track D — Pairwise model-vs-model comparison (ArenaGEval style).

For a stratified 60-prompt subset (20 per category, all constrained variants),
present each pair of model responses to GPT-4o-mini side by side. Position is
randomized DETERMINISTICALLY per (prompt, pair, dimension) tuple — same
randomization on every rerun, so the unified LLM cache hits cleanly.

Output:
  results/arena_pairs.csv          — one row per pairwise comparison
  results/arena_matrix.csv         — long format: pair × dimension cell stats
  console table with Wilson 95% CIs

Run:
    python -m src.arena_eval                              # full run
    python -m src.arena_eval --pilot 5                    # 5 prompts/category
    python -m src.arena_eval --dimensions affordability cultural
"""

import argparse
import csv
import functools
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    MODELS, ENRICHED_DIR, RESULTS_DIR, JUDGE_MODEL,
    JUDGE_MODEL_PRICING, PROMPTS_PATH,
)
from src.clients.unified_llm import UnifiedLLM


ARENA_PROMPT_COUNT_PER_CATEGORY = 20
ARENA_RANDOM_SEED = 42

DIMENSIONS = {
    "affordability": (
        "Which response BETTER addresses any financial constraint stated in the prompt?\n"
        "Consider budget acknowledgment, cheap substitutions, avoidance of luxury items, "
        "and overall cost realism. If no financial constraint was stated, judge on which "
        "response is more reasonable for a typical household."
    ),
    "cultural": (
        "Which response BETTER respects any cultural, religious, or culinary tradition "
        "stated in the prompt?\n"
        "Consider authentic cuisine alignment, religious/dietary restriction compliance, "
        "and culturally-appropriate methods. If no cultural constraint was stated, judge "
        "on which response is less Western-defaulted in a way that would not generalize."
    ),
    "feasibility": (
        "Which response is MORE realistically executable given any time, equipment, mobility, "
        "or environmental constraint stated in the prompt?\n"
        "Consider real time totals, equipment matches, structural adaptation. If no lifestyle "
        "constraint was stated, judge on which response is more practical for a typical adult."
    ),
}


# =============================================================
# Stratified 60-prompt subset selection (deterministic via seed 42)
# =============================================================
def select_arena_prompts(pilot: Optional[int] = None) -> list[str]:
    """20 constrained prompts per category (60 total) by default.
    Pilot mode selects `pilot` per category."""
    rng = random.Random(ARENA_RANDOM_SEED)
    by_category: dict[str, list[str]] = defaultdict(list)

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("variant") != "constrained":
                continue
            by_category[rec["category"]].append(rec["id"])

    selected: list[str] = []
    n_per_cat = pilot if pilot else ARENA_PROMPT_COUNT_PER_CATEGORY
    for cat in ("financial", "cultural", "lifestyle"):
        ids = sorted(by_category.get(cat, []))
        rng.shuffle(ids)
        selected.extend(ids[:n_per_cat])

    return selected


# =============================================================
# Loading model responses + prompt text (cached)
# =============================================================
def load_response(provider: str, prompt_id: str) -> Optional[str]:
    """Get response_text from the enriched file."""
    path = Path(ENRICHED_DIR) / provider / f"{prompt_id}.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        return rec.get("response_text")
    except (OSError, json.JSONDecodeError):
        return None


@functools.lru_cache(maxsize=None)
def _all_prompt_texts() -> dict[str, str]:
    """Read prompts.jsonl once, return {prompt_id: prompt_text}."""
    out: dict[str, str] = {}
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec:
                out[rec["id"]] = rec.get("prompt_text", "")
    return out


def load_prompt_text(prompt_id: str) -> str:
    return _all_prompt_texts().get(prompt_id, "")


# =============================================================
# Pairwise comparison call
# =============================================================
def build_arena_prompt(
    user_prompt: str,
    response_a: str,
    response_b: str,
    dimension_key: str,
) -> str:
    dimension_instruction = DIMENSIONS[dimension_key]
    return (
        "You are evaluating two AI responses to the same user prompt. "
        "Pick which response is better on the dimension specified below.\n"
        "Position A vs B is randomized per comparison — do NOT let position "
        "influence your decision. Evaluate solely on the responses' content "
        "relative to the dimension.\n\n"
        f"=== USER PROMPT ===\n{user_prompt}\n\n"
        f"=== RESPONSE A ===\n{response_a}\n\n"
        f"=== RESPONSE B ===\n{response_b}\n\n"
        f"=== DIMENSION ===\n{dimension_instruction}\n\n"
        f"=== INSTRUCTIONS ===\n"
        "Provide one sentence of reasoning, then on a new line ONE of:\n"
        "VERDICT: A\n"
        "VERDICT: B\n"
        "VERDICT: TIE"
    )


_VERDICT_PRIMARY = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.IGNORECASE)
_VERDICT_FALLBACK = re.compile(r"\b(A|B|TIE)\b\s*$", re.IGNORECASE)


def parse_arena_verdict(text: str) -> tuple[Optional[str], str]:
    """Return (verdict, reasoning). Verdict is 'A' / 'B' / 'TIE' / None."""
    if not text:
        return None, ""
    text = text.strip()

    match = _VERDICT_PRIMARY.search(text)
    if match:
        verdict = match.group(1).upper()
        reasoning = text[:match.start()].strip()
        return verdict, reasoning

    # Fallback: bare A / B / TIE on the final line
    last_line = text.splitlines()[-1].strip() if text else ""
    fallback = _VERDICT_FALLBACK.search(last_line)
    if fallback:
        verdict = fallback.group(1).upper()
        return verdict, text

    return None, text


def _comparison_seed(prompt_id: str, prov_x: str, prov_y: str, dim: str) -> int:
    """Deterministic per-comparison seed so reruns hit the LLM cache."""
    h = hashlib.sha256(
        f"{prompt_id}|{prov_x}|{prov_y}|{dim}|{ARENA_RANDOM_SEED}".encode("utf-8")
    ).digest()
    return int.from_bytes(h[:8], "big")


def run_one_comparison(
    client: UnifiedLLM,
    user_prompt: str,
    response_x: str,
    response_y: str,
    dimension: str,
    prompt_id: str,
    prov_x: str,
    prov_y: str,
) -> dict:
    """One pairwise comparison with deterministic position randomization.
    Same (prompt, pair, dimension) always gets the same A/B assignment, so
    repeated runs hit the unified LLM cache."""
    seed = _comparison_seed(prompt_id, prov_x, prov_y, dimension)
    local_rng = random.Random(seed)
    x_is_a = local_rng.random() < 0.5
    if x_is_a:
        response_a, response_b = response_x, response_y
    else:
        response_a, response_b = response_y, response_x

    prompt = build_arena_prompt(user_prompt, response_a, response_b, dimension)
    response = client.generate(
        provider="openai",
        prompt=prompt,
        model=JUDGE_MODEL,
        params={"temperature": 0.0, "max_tokens": 250},
    )

    verdict_raw, reasoning = parse_arena_verdict(response["text"])

    if verdict_raw == "A":
        verdict_for_x = "win" if x_is_a else "loss"
    elif verdict_raw == "B":
        verdict_for_x = "loss" if x_is_a else "win"
    elif verdict_raw == "TIE":
        verdict_for_x = "tie"
    else:
        verdict_for_x = None

    verdict_for_y = (
        "win" if verdict_for_x == "loss"
        else "loss" if verdict_for_x == "win"
        else "tie" if verdict_for_x == "tie"
        else None
    )

    return {
        "verdict_for_x": verdict_for_x,
        "verdict_for_y": verdict_for_y,
        "raw_verdict": verdict_raw,
        "x_position": "A" if x_is_a else "B",
        "reasoning": reasoning,
        "input_tokens": response.get("input_tokens", 0),
        "output_tokens": response.get("output_tokens", 0),
        "from_cache": response.get("from_cache", False),
    }


# =============================================================
# Wilson score 95% CI for a binomial proportion. NOTE: ties are
# excluded from the denominator (decided games only).
# =============================================================
def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    n = total
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _estimate_cost(in_t: int, out_t: int) -> float:
    return (
        in_t * JUDGE_MODEL_PRICING["in"]
        + out_t * JUDGE_MODEL_PRICING["out"]
    ) / 1_000_000


# =============================================================
# Main runner
# =============================================================
def run(
    pilot: Optional[int] = None,
    dimensions: Optional[list[str]] = None,
):
    if dimensions is None:
        dimensions = list(DIMENSIONS.keys())

    print(f"ArenaGEval: dimensions={dimensions}")

    prompt_ids = select_arena_prompts(pilot=pilot)
    print(f"Selected {len(prompt_ids)} prompts (stratified by category)")

    providers = list(MODELS.keys())
    pairs = list(combinations(providers, 2))
    print(f"Model pairs: {pairs}")

    total_calls = len(prompt_ids) * len(pairs) * len(dimensions)
    print(f"Total comparisons: {total_calls}\n")

    client = UnifiedLLM()

    rows: list[dict] = []
    skipped_no_response = 0
    parse_errors = 0
    total_in_tokens = 0
    total_out_tokens = 0
    new_calls = 0
    cached_calls = 0

    with tqdm(total=total_calls, desc="Arena", unit="cmp") as pbar:
        for prompt_id in prompt_ids:
            user_prompt = load_prompt_text(prompt_id)
            if not user_prompt:
                pbar.update(len(pairs) * len(dimensions))
                continue

            responses_for_prompt = {
                p: load_response(p, prompt_id) for p in providers
            }

            for prov_a, prov_b in pairs:
                resp_a = responses_for_prompt[prov_a]
                resp_b = responses_for_prompt[prov_b]
                if not resp_a or not resp_b:
                    skipped_no_response += 1
                    pbar.update(len(dimensions))
                    continue

                for dimension in dimensions:
                    out = run_one_comparison(
                        client, user_prompt, resp_a, resp_b, dimension,
                        prompt_id, prov_a, prov_b,
                    )
                    total_in_tokens += out["input_tokens"]
                    total_out_tokens += out["output_tokens"]
                    if out["from_cache"]:
                        cached_calls += 1
                    else:
                        new_calls += 1
                    if out["verdict_for_x"] is None:
                        parse_errors += 1

                    rows.append({
                        "prompt_id": prompt_id,
                        "provider_x": prov_a,
                        "provider_y": prov_b,
                        "dimension": dimension,
                        "verdict_for_x": out["verdict_for_x"],
                        "x_assigned_position": out["x_position"],
                        "raw_verdict": out["raw_verdict"],
                        "reasoning": (out["reasoning"] or "")[:300],
                    })
                    pbar.set_postfix({
                        "new": new_calls,
                        "cached": cached_calls,
                        "parse_err": parse_errors,
                        "spend": f"${_estimate_cost(total_in_tokens, total_out_tokens):.3f}",
                    })
                    pbar.update(1)

    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = out_dir / "arena_pairs.csv"
    if rows:
        with pairs_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {pairs_csv} ({len(rows)} comparisons)")

    # Aggregate to (provider_x, provider_y, dimension) cells
    by_pair_dim: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["provider_x"], r["provider_y"], r["dimension"])
        by_pair_dim[key].append(r["verdict_for_x"])

    matrix_rows: list[dict] = []
    for (prov_x, prov_y, dim), verdicts in sorted(by_pair_dim.items()):
        n_total = len(verdicts)
        n_x_wins = sum(1 for v in verdicts if v == "win")
        n_y_wins = sum(1 for v in verdicts if v == "loss")
        n_ties = sum(1 for v in verdicts if v == "tie")
        n_unparsed = sum(1 for v in verdicts if v is None)
        decided = n_x_wins + n_y_wins

        if decided > 0:
            x_rate = n_x_wins / decided
            x_lo, x_hi = wilson_ci(n_x_wins, decided)
        else:
            x_rate, x_lo, x_hi = None, None, None

        matrix_rows.append({
            "provider_x": prov_x,
            "provider_y": prov_y,
            "dimension": dim,
            "n_total": n_total,
            "n_x_wins": n_x_wins,
            "n_y_wins": n_y_wins,
            "n_ties": n_ties,
            "n_unparsed": n_unparsed,
            "x_winrate_excl_ties": (round(x_rate, 3) if x_rate is not None else None),
            "x_winrate_ci_low":    (round(x_lo, 3) if x_lo is not None else None),
            "x_winrate_ci_high":   (round(x_hi, 3) if x_hi is not None else None),
        })

    matrix_csv = out_dir / "arena_matrix.csv"
    if matrix_rows:
        with matrix_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
            writer.writeheader()
            writer.writerows(matrix_rows)
        print(f"Wrote {matrix_csv} ({len(matrix_rows)} pair × dimension cells)")

    print("\n" + "=" * 78)
    print("ARENA SUMMARY")
    print("=" * 78)
    print(f"Total comparisons:    {len(rows)}")
    print(f"  new calls:          {new_calls}")
    print(f"  cache hits:         {cached_calls}")
    print(f"Parse errors:         {parse_errors}")
    print(f"Skipped (missing):    {skipped_no_response}")
    print(f"Tokens: in={total_in_tokens:,}, out={total_out_tokens:,}")
    print(f"Estimated NEW spend:  ${_estimate_cost(total_in_tokens, total_out_tokens):.4f}")
    print("(Cached calls cost $0 — only new API calls are billed.)")

    print(f"\nWin rates per (provider_x vs provider_y, dimension)")
    print("  rates exclude ties; CI is Wilson 95% on decided games only")
    print(f"  {'pair':30} {'dim':14} {'n_dec':>5} {'x_wins':>6} "
          f"{'rate':>8} {'95% CI':>14}")
    for r in matrix_rows:
        pair_name = f"{r['provider_x']:>9} vs {r['provider_y']:<10}"
        decided = r["n_x_wins"] + r["n_y_wins"]
        rate = r["x_winrate_excl_ties"]
        if rate is None:
            print(f"  {pair_name:30} {r['dimension']:14} {decided:5d} "
                  f"{r['n_x_wins']:6d} {'n/a':>8} {'n/a':>14}")
        else:
            ci = f"[{r['x_winrate_ci_low']:.2f}, {r['x_winrate_ci_high']:.2f}]"
            print(f"  {pair_name:30} {r['dimension']:14} {decided:5d} "
                  f"{r['n_x_wins']:6d} {rate:7.1%}  {ci:>14}")

    print("\nNext: python -m src.aggregate_scores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise model arena evaluation")
    parser.add_argument("--pilot", type=int, default=None,
                        help="N prompts per category (default 20)")
    parser.add_argument("--dimensions", nargs="+", default=None,
                        choices=list(DIMENSIONS.keys()))
    args = parser.parse_args()

    run(pilot=args.pilot, dimensions=args.dimensions)
