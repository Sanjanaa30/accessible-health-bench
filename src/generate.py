"""
src/generate.py

Phase 2 — Full generation.

Loops over all 120 prompts × 4 providers = 480 responses.
Saves each response as JSON to data/responses/{provider}/{prompt_id}.json.
Uses the SQLite cache, so re-running this script is free and instant for
already-completed responses.

Run from repo root:
    python -m src.generate

For a smaller pilot run first:
    python -m src.generate --pilot 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clients.unified_llm import UnifiedLLM
from src.config import (
    MODELS,
    PROMPTS_PATH,
    RESPONSES_DIR,
)


# ============================================================
# Approximate prices per million tokens (April 2026)
# Used only for the running cost estimate during generation
# ============================================================
PRICING = {
    "openai":    {"in": 0.15,  "out": 0.60},  # gpt-4o-mini
    "anthropic": {"in": 1.00,  "out": 5.00},  # claude-haiku-4-5
    "gemini":    {"in": 0.0,   "out": 0.0},   # free tier
    "groq":      {"in": 0.0,   "out": 0.0},   # free tier
}


def estimate_cost(provider: str, in_tokens: int, out_tokens: int) -> float:
    """Estimate USD cost for a single call."""
    p = PRICING[provider]
    return (in_tokens * p["in"] + out_tokens * p["out"]) / 1_000_000


def load_prompts(path: str) -> list:
    """Load prompts from JSONL file."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def response_path(provider: str, prompt_id: str) -> Path:
    """Where this response gets saved."""
    return Path(RESPONSES_DIR) / provider / f"{prompt_id}.json"


def already_saved(provider: str, prompt_id: str) -> bool:
    """Check if response file already exists on disk."""
    return response_path(provider, prompt_id).exists()


def save_response(provider: str, prompt_id: str, prompt_record: dict, response: dict):
    """Save the response with full metadata to data/responses/{provider}/{id}.json."""
    out_path = response_path(provider, prompt_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "prompt_id": prompt_id,
        "category": prompt_record["category"],
        "variant": prompt_record["variant"],
        "category_type": prompt_record.get("category_type"),
        "stated_constraints": prompt_record.get("stated_constraints", {}),
        "prompt_text": prompt_record["prompt_text"],
        "provider": provider,
        "model": response["model"],
        "response_text": response["text"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "from_cache": response["from_cache"],
        "timestamp": response["timestamp"],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def generate_all(
    pilot: Optional[int] = None,
    providers: Optional[list] = None,
    sleep_between_calls: float = 0.0,
):
    """
    Run generation across all prompts and providers.

    Args:
        pilot: If set, only run on the first N prompts (for testing).
        providers: List of providers to use. Defaults to all 4.
        sleep_between_calls: Seconds to sleep between API calls (rate limiting).
    """
    if providers is None:
        providers = list(MODELS.keys())

    # Load prompts
    prompts = load_prompts(PROMPTS_PATH)
    if pilot is not None:
        prompts = prompts[:pilot]
        print(f"PILOT MODE: running on first {pilot} prompts only.\n")

    total_calls = len(prompts) * len(providers)
    print(f"Plan: {len(prompts)} prompts x {len(providers)} providers = {total_calls} calls")
    print(f"Providers: {providers}\n")

    # Initialize client
    client = UnifiedLLM()

    # Tracking
    completed = 0
    cached_hits = 0
    new_calls = 0
    failures = []
    total_cost = 0.0
    total_in_tokens = 0
    total_out_tokens = 0

    # Build list of all (provider, prompt) pairs to process
    tasks = []
    for prompt in prompts:
        for provider in providers:
            tasks.append((provider, prompt))

    # Main loop with progress bar
    with tqdm(total=len(tasks), desc="Generating", unit="resp") as pbar:
        for provider, prompt in tasks:
            prompt_id = prompt["id"]
            prompt_text = prompt["prompt_text"]

            # Skip if file already saved (cheap disk check)
            if already_saved(provider, prompt_id):
                pbar.set_postfix_str(f"skipped {provider}/{prompt_id}")
                completed += 1
                pbar.update(1)
                continue

            # Generate (cache hits are free; cache misses make the API call)
            try:
                response = client.generate(provider=provider, prompt=prompt_text)
                save_response(provider, prompt_id, prompt, response)

                if response["from_cache"]:
                    cached_hits += 1
                else:
                    new_calls += 1
                    total_in_tokens += response["input_tokens"]
                    total_out_tokens += response["output_tokens"]
                    total_cost += estimate_cost(
                        provider, response["input_tokens"], response["output_tokens"]
                    )
                    if sleep_between_calls > 0:
                        time.sleep(sleep_between_calls)

                completed += 1
                pbar.set_postfix({
                    "cost": f"${total_cost:.3f}",
                    "new": new_calls,
                    "cached": cached_hits,
                })

            except Exception as e:
                failures.append((provider, prompt_id, str(e)))
                pbar.set_postfix_str(f"FAILED {provider}/{prompt_id}")

            pbar.update(1)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total tasks:        {len(tasks)}")
    print(f"Completed:          {completed}")
    print(f"  New API calls:    {new_calls}")
    print(f"  Cache hits:       {cached_hits}")
    print(f"Failed:             {len(failures)}")
    print(f"\nTotal input tokens:  {total_in_tokens:,}")
    print(f"Total output tokens: {total_out_tokens:,}")
    print(f"Estimated cost:      ${total_cost:.4f}")

    if failures:
        print("\nFAILURES (rerun the script to retry these):")
        for provider, pid, err in failures[:10]:
            print(f"  {provider}/{pid}: {err[:80]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    # Verify counts per provider
    print("\nFiles saved per provider:")
    for provider in providers:
        provider_dir = Path(RESPONSES_DIR) / provider
        if provider_dir.exists():
            count = len(list(provider_dir.glob("*.json")))
            expected = len(prompts)
            status = "OK" if count == expected else "INCOMPLETE"
            print(f"  {provider:12s} {count}/{expected}  [{status}]")
        else:
            print(f"  {provider:12s} 0/{len(prompts)}  [MISSING DIR]")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses across 4 LLMs")
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Run on only the first N prompts (for testing). Omit for full run.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        choices=["openai", "anthropic", "gemini", "groq"],
        help="Subset of providers to run. Default: all 4.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (for rate limiting).",
    )
    args = parser.parse_args()

    generate_all(
        pilot=args.pilot,
        providers=args.providers,
        sleep_between_calls=args.sleep,
    )