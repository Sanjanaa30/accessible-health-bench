"""
src/extract.py

Phase 3 — Extraction.

Reads each of the 480 free-text responses from data/responses/ and converts
it into a rich structured JSON via GPT-4o-mini. Saves to data/extractions/.

The extraction prompt (prompts/extraction.txt) uses Python-style placeholders:
  {prompt_text}, {response_text}, {stated_constraints_json}
which this script fills via .format() before sending to the model.

Output format: data/extractions/{provider}/{prompt_id}.json

Run from repo root:
    python -m src.extract                    # full run
    python -m src.extract --pilot 10         # test on 10 responses first
    python -m src.extract --providers openai # only one provider
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clients.unified_llm import UnifiedLLM
from src.config import (
    MODELS,
    EXTRACTION_MODEL,
    RESPONSES_DIR,
    EXTRACTIONS_DIR,
)


# ============================================================
# Pricing for GPT-4o-mini (used for extraction)
# ============================================================
EXTRACTION_PRICING = {"in": 0.15, "out": 0.60}  # per million tokens


def estimate_extraction_cost(in_tokens: int, out_tokens: int) -> float:
    return (in_tokens * EXTRACTION_PRICING["in"]
            + out_tokens * EXTRACTION_PRICING["out"]) / 1_000_000


# ============================================================
# Load extraction prompt template
# ============================================================
def load_extraction_prompt() -> str:
    """Load the extraction prompt template from prompts/extraction.txt."""
    prompt_path = Path("prompts/extraction.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Extraction prompt not found at {prompt_path}. "
            "Place extraction.txt in the prompts/ directory."
        )
    return prompt_path.read_text(encoding="utf-8")


# ============================================================
# Build the extraction call payload via template substitution
# ============================================================
def build_extraction_input(
    extraction_prompt_template: str,
    user_prompt: str,
    ai_response: str,
    stated_constraints: dict,
) -> str:
    """
    Fill the extraction template's placeholders with the prompt, response,
    and stated_constraints. The template uses Python-style placeholders:
      {prompt_text}, {response_text}, {stated_constraints_json}

    We pre-escape any literal curly braces in the JSON-schema example portion
    of the template by NOT using them as format keys (i.e. they should be
    written as {{ }} in extraction.txt — your schema example already uses
    double braces correctly).
    """
    return extraction_prompt_template.format(
        prompt_text=user_prompt,
        response_text=ai_response,
        stated_constraints_json=json.dumps(stated_constraints, indent=2),
    )


# ============================================================
# Path helpers
# ============================================================
def response_path(provider: str, prompt_id: str) -> Path:
    return Path(RESPONSES_DIR) / provider / f"{prompt_id}.json"


def extraction_path(provider: str, prompt_id: str) -> Path:
    return Path(EXTRACTIONS_DIR) / provider / f"{prompt_id}.json"


def already_extracted(provider: str, prompt_id: str) -> bool:
    return extraction_path(provider, prompt_id).exists()


# ============================================================
# Call GPT-4o-mini for extraction
# ============================================================
def extract_one(
    client: UnifiedLLM,
    extraction_prompt_template: str,
    user_prompt: str,
    ai_response: str,
    stated_constraints: dict,
) -> dict:
    """
    Run extraction on a single response. Returns:
        {
            "extracted": {...parsed JSON...},
            "raw_text": "...",
            "input_tokens": int,
            "output_tokens": int,
            "from_cache": bool,
        }
    """
    full_prompt = build_extraction_input(
        extraction_prompt_template,
        user_prompt,
        ai_response,
        stated_constraints,
    )

    # Use the unified client — automatic caching applies here too.
    # max_tokens=8000 because the rich schema (meal + fitness + cultural + feasibility +
    # demographic + medical) can run long for multi-day plans. 2500 truncated mid-JSON
    # on cul_base_*; 4000 still truncated lif_base_* and fin_base_* (output reached
    # ~14000 chars). gpt-4o-mini's hard ceiling is 16384 output tokens, so 8000 leaves
    # headroom even for the densest 7-day plan + full fitness routine.
    response = client.generate(
        provider="openai",
        prompt=full_prompt,
        model=EXTRACTION_MODEL,
        params={"temperature": 0.0, "max_tokens": 8000},
    )

    raw_text = response["text"].strip()

    # Strip markdown fences if the model added them (it shouldn't, but just in case)
    if raw_text.startswith("```"):
        # Drop opening fence (and optional 'json' marker)
        raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].lstrip()
        # Drop closing fence
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0].rstrip()

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Return a marker dict so we can flag this for manual review
        extracted = {
            "_extraction_error": str(e),
            "_raw_text_snippet": raw_text[:500],
        }

    return {
        "extracted": extracted,
        "raw_text": raw_text,
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "from_cache": response["from_cache"],
    }


# ============================================================
# Save extraction with full metadata
# ============================================================
def save_extraction(provider: str, prompt_id: str, source_record: dict, extraction: dict):
    out_path = extraction_path(provider, prompt_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        # Carry forward all source metadata
        "prompt_id": source_record["prompt_id"],
        "category": source_record["category"],
        "variant": source_record["variant"],
        "category_type": source_record.get("category_type"),
        "stated_constraints": source_record.get("stated_constraints", {}),
        "prompt_text": source_record["prompt_text"],
        "provider": source_record["provider"],
        "model": source_record["model"],
        "response_text": source_record["response_text"],

        # New extraction fields
        "extracted": extraction["extracted"],
        "extraction_model": EXTRACTION_MODEL,
        "extraction_input_tokens": extraction["input_tokens"],
        "extraction_output_tokens": extraction["output_tokens"],
        "extraction_from_cache": extraction["from_cache"],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# ============================================================
# Main loop
# ============================================================
def extract_all(
    pilot: Optional[int] = None,
    providers: Optional[list] = None,
):
    """
    Run extraction across all responses and providers.

    Args:
        pilot: If set, only extract first N responses per provider.
        providers: Subset of providers to extract.
    """
    if providers is None:
        providers = list(MODELS.keys())

    extraction_prompt_template = load_extraction_prompt()
    client = UnifiedLLM()

    # Build task list
    tasks = []
    for provider in providers:
        provider_dir = Path(RESPONSES_DIR) / provider
        if not provider_dir.exists():
            print(f"WARNING: no responses for {provider} — skipping")
            continue
        response_files = sorted(provider_dir.glob("*.json"))
        if pilot is not None:
            response_files = response_files[:pilot]
        for resp_file in response_files:
            tasks.append((provider, resp_file))

    print(f"Plan: {len(tasks)} extractions across providers {providers}\n")

    # Tracking
    completed = 0
    cached_hits = 0
    new_calls = 0
    failures = []
    parse_errors = []
    total_cost = 0.0

    with tqdm(total=len(tasks), desc="Extracting", unit="resp") as pbar:
        for provider, resp_file in tasks:
            source_record = json.loads(resp_file.read_text(encoding="utf-8"))
            prompt_id = source_record["prompt_id"]

            # Skip if already saved
            if already_extracted(provider, prompt_id):
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"skipped {provider}/{prompt_id}")
                continue

            try:
                extraction = extract_one(
                    client=client,
                    extraction_prompt_template=extraction_prompt_template,
                    user_prompt=source_record["prompt_text"],
                    ai_response=source_record["response_text"],
                    stated_constraints=source_record.get("stated_constraints", {}),
                )

                save_extraction(provider, prompt_id, source_record, extraction)
                completed += 1

                if extraction["from_cache"]:
                    cached_hits += 1
                else:
                    new_calls += 1
                    total_cost += estimate_extraction_cost(
                        extraction["input_tokens"], extraction["output_tokens"]
                    )

                # Flag JSON parse errors for manual review
                if isinstance(extraction["extracted"], dict) and \
                   "_extraction_error" in extraction["extracted"]:
                    parse_errors.append((provider, prompt_id))

                pbar.set_postfix({
                    "cost": f"${total_cost:.3f}",
                    "new": new_calls,
                    "cached": cached_hits,
                    "parse_err": len(parse_errors),
                })

            except Exception as e:
                failures.append((provider, prompt_id, str(e)))
                pbar.set_postfix_str(f"FAILED {provider}/{prompt_id}")

            pbar.update(1)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total tasks:       {len(tasks)}")
    print(f"Completed:         {completed}")
    print(f"  New API calls:   {new_calls}")
    print(f"  Cache hits:      {cached_hits}")
    print(f"Parse errors:      {len(parse_errors)}")
    print(f"Hard failures:     {len(failures)}")
    print(f"Estimated cost:    ${total_cost:.4f}")

    if parse_errors:
        print("\nFiles with JSON parse errors (review manually or rerun):")
        for provider, pid in parse_errors[:10]:
            print(f"  {provider}/{pid}")
        if len(parse_errors) > 10:
            print(f"  ... and {len(parse_errors) - 10} more")

    if failures:
        print("\nHard failures (rerun the script to retry):")
        for provider, pid, err in failures[:5]:
            print(f"  {provider}/{pid}: {err[:80]}")

    # Per-provider count check
    print("\nFiles saved per provider:")
    for provider in providers:
        d = Path(EXTRACTIONS_DIR) / provider
        if d.exists():
            count = len(list(d.glob("*.json")))
            print(f"  {provider:12s} {count}")
        else:
            print(f"  {provider:12s} 0  [MISSING]")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured JSON from responses")
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Extract only first N responses per provider (for testing).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        choices=["openai", "anthropic", "deepseek", "groq"],
        help="Subset of providers. Default: all 4.",
    )
    args = parser.parse_args()

    extract_all(pilot=args.pilot, providers=args.providers)