"""
src/run_judges.py

Phase 5, Step 5 — Run all 4 judges on every enriched response.

Reads:  data/enriched/{provider}/{prompt_id}.json
Writes: data/judged/{provider}/{prompt_id}.json — extends the enriched
        record with a "judges" block holding each judge's full output.

Resumable — skips judge calls that already produced a clean result for
the requested judge in the existing judged file. Re-runs judges where
parse_error was set on the prior attempt. Saves after each successful
judge so a mid-record crash never loses partial work.

Cost-aware: every judge call goes through UnifiedLLM's SQLite cache.

Run:
    python -m src.run_judges                         # full run, all 4 judges
    python -m src.run_judges --pilot 5               # 5 records per provider
    python -m src.run_judges --judges affordability cultural
    python -m src.run_judges --providers openai
    python -m src.run_judges --force --judges adherence   # redo specified
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS, ENRICHED_DIR, JUDGED_DIR, JUDGE_MODEL_PRICING
from src.clients.unified_llm import UnifiedLLM
from src.judges.affordability import AffordabilityJudge
from src.judges.cultural import CulturalJudge
from src.judges.feasibility import FeasibilityJudge
from src.judges.adherence import AdherenceJudge


ALL_JUDGES = ["affordability", "cultural", "feasibility", "adherence"]


def _judge_factory(name: str, client: UnifiedLLM):
    if name == "affordability":
        return AffordabilityJudge(client=client)
    if name == "cultural":
        return CulturalJudge(client=client)
    if name == "feasibility":
        return FeasibilityJudge(client=client)
    if name == "adherence":
        return AdherenceJudge(client=client)
    raise ValueError(f"Unknown judge: {name}")


def enriched_path(provider: str, prompt_id: str) -> Path:
    return Path(ENRICHED_DIR) / provider / f"{prompt_id}.json"


def judged_path(provider: str, prompt_id: str) -> Path:
    return Path(JUDGED_DIR) / provider / f"{prompt_id}.json"


def is_judge_done(judges_block: dict, judge_name: str) -> bool:
    """A judge counts as 'done' if a result exists with no parse_error.
    Score may legitimately be None (e.g. adherence baseline) — we don't
    require a numeric score, only a clean parse."""
    existing = judges_block.get(judge_name)
    if not isinstance(existing, dict):
        return False
    if existing.get("parse_error"):
        return False
    return True


def estimate_cost(in_tokens: int, out_tokens: int) -> float:
    return (
        in_tokens * JUDGE_MODEL_PRICING["in"]
        + out_tokens * JUDGE_MODEL_PRICING["out"]
    ) / 1_000_000


def _bucket_for_distribution(score) -> Optional[int]:
    """Round a score (int 1-5 or float adherence average) to nearest int 1-5.
    None / out-of-range -> None."""
    if score is None:
        return None
    try:
        s = int(round(float(score)))
    except (TypeError, ValueError):
        return None
    if 1 <= s <= 5:
        return s
    return None


def run(
    pilot: Optional[int] = None,
    providers: Optional[list[str]] = None,
    judges: Optional[list[str]] = None,
    force: bool = False,
):
    if providers is None:
        providers = list(MODELS.keys())
    if judges is None:
        judges = ALL_JUDGES

    print(f"Running judges: {judges}")
    print(f"On providers: {providers}")
    if pilot is not None:
        print(f"PILOT MODE: {pilot} records per provider")
    if force:
        print("FORCE MODE: re-running specified judges even if cached")

    enriched_root = Path(ENRICHED_DIR)
    if not enriched_root.exists():
        print(f"ERROR: {enriched_root} does not exist. Run Phase 4 first.")
        return

    client = UnifiedLLM()
    judge_instances = {name: _judge_factory(name, client) for name in judges}

    # Build task list
    tasks: list[tuple[str, Path]] = []
    for provider in providers:
        provider_dir = enriched_root / provider
        if not provider_dir.exists():
            print(f"  no enriched dir for {provider}, skipping")
            continue
        files = sorted(provider_dir.glob("*.json"))
        if pilot is not None:
            files = files[:pilot]
        for f in files:
            tasks.append((provider, f))

    print(f"\nTotal records to process: {len(tasks)}")
    print(f"Total judge calls expected (max): {len(tasks) * len(judges)}")

    # Trackers
    completed_records = 0
    new_judge_calls = 0
    skipped_judge_calls = 0
    failures: list[tuple[str, str, str]] = []
    parse_errors: Counter = Counter()
    total_in_tokens = 0
    total_out_tokens = 0
    score_distribution: dict[str, Counter] = {j: Counter() for j in judges}

    with tqdm(total=len(tasks), desc="Judging", unit="rec") as pbar:
        for provider, enriched_file in tasks:
            try:
                source = json.loads(enriched_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                failures.append((provider, enriched_file.stem, f"read_error: {e}"))
                pbar.update(1)
                continue

            prompt_id = source["prompt_id"]
            out_path = judged_path(provider, prompt_id)

            # Load existing judged file if present, so we only run missing judges
            if out_path.exists():
                try:
                    record = json.loads(out_path.read_text(encoding="utf-8"))
                    record.setdefault("judges", {})
                except (OSError, json.JSONDecodeError):
                    record = dict(source)
                    record["judges"] = {}
            else:
                record = dict(source)
                record["judges"] = {}

            # Skip records where Phase 4 grounding was skipped (extraction error)
            grounding = record.get("grounding") or {}
            if "_skipped" in grounding:
                record["judges"]["_skipped"] = grounding["_skipped"]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                pbar.update(1)
                continue

            # Run each requested judge that isn't already done
            for judge_name in judges:
                if not force and is_judge_done(record["judges"], judge_name):
                    skipped_judge_calls += 1
                    bucket = _bucket_for_distribution(
                        record["judges"][judge_name].get("score")
                    )
                    if bucket is not None:
                        score_distribution[judge_name][bucket] += 1
                    continue

                try:
                    result = judge_instances[judge_name].evaluate(record)
                except Exception as e:  # noqa: BLE001
                    failures.append((provider, prompt_id, f"{judge_name}: {e}"))
                    continue

                record["judges"][judge_name] = result.to_dict()
                new_judge_calls += 1
                total_in_tokens += result.input_tokens
                total_out_tokens += result.output_tokens
                bucket = _bucket_for_distribution(result.score)
                if bucket is not None:
                    score_distribution[judge_name][bucket] += 1
                if result.parse_error:
                    parse_errors[judge_name] += 1

                # Save after every successful judge call so a later crash
                # doesn't lose this judge's work.
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            completed_records += 1
            pbar.set_postfix({
                "new_calls": new_judge_calls,
                "cached": skipped_judge_calls,
                "fail": len(failures),
                "spend": f"${estimate_cost(total_in_tokens, total_out_tokens):.3f}",
            })
            pbar.update(1)

    # Summary
    print("\n" + "=" * 78)
    print("JUDGING COMPLETE")
    print("=" * 78)
    print(f"Records processed:        {completed_records}")
    print(f"New judge calls:          {new_judge_calls}")
    print(f"Cached / already-clean:   {skipped_judge_calls}")
    print(f"Hard failures:            {len(failures)}")
    print(f"Total input tokens:       {total_in_tokens:,}")
    print(f"Total output tokens:      {total_out_tokens:,}")
    print(f"New-call estimated spend: "
          f"${estimate_cost(total_in_tokens, total_out_tokens):.4f}")
    print("(Cost only reflects calls made in this run; cached calls cost $0.)")

    print("\nScore distributions per judge (1=worst, 5=best):")
    for j in judges:
        dist = score_distribution[j]
        total = sum(dist.values())
        if total == 0:
            print(f"  {j:14s} no scores yet")
            continue
        line_parts = [f"{j:14s} n={total:4d}"]
        for s in (1, 2, 3, 4, 5):
            n = dist.get(s, 0)
            pct = n / total * 100 if total else 0
            line_parts.append(f"{s}:{n:3d}({pct:4.1f}%)")
        print("  " + "  ".join(line_parts))
        if parse_errors[j]:
            print(f"  {' ' * 14}  parse_errors: {parse_errors[j]}")

    if failures:
        print("\nFailures (first 10):")
        for prov, pid, err in failures[:10]:
            print(f"  {prov}/{pid}: {err[:80]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    print("\nFiles per provider in data/judged/:")
    for prov in providers:
        d = Path(JUDGED_DIR) / prov
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        print(f"  {prov:12s} {n}")

    print("\nNext: python -m src.arena_eval (Track D)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4 judges on enriched responses")
    parser.add_argument("--pilot", type=int, default=None,
                        help="Process first N records per provider")
    parser.add_argument("--providers", nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Subset of providers")
    parser.add_argument("--judges", nargs="+", default=None,
                        choices=ALL_JUDGES,
                        help="Subset of judges to run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run specified judges even if already clean.")
    args = parser.parse_args()

    run(
        pilot=args.pilot,
        providers=args.providers,
        judges=args.judges,
        force=args.force,
    )
