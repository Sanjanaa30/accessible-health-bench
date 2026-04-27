"""
convert_csv_to_jsonl.py

Converts data/LLM_Prompts.csv into data/prompts.jsonl for the
AccessibleHealthBench pipeline.

Run from the repo root:
    python convert_csv_to_jsonl.py

Reads:  data/LLM_Prompts.csv
Writes: data/prompts.jsonl
"""

import csv
import json
from pathlib import Path

INPUT_CSV = Path("data/LLM_Prompts.csv")
OUTPUT_JSONL = Path("data/prompts.jsonl")

# Map the typo in row lif_con_15 ("work Constraineds" -> "work constraints")
TEXT_FIXES = {
    "work Constraineds": "work constraints",
}


def clean_text(text: str) -> str:
    """Apply known text fixes and strip whitespace."""
    text = text.strip()
    for bad, good in TEXT_FIXES.items():
        text = text.replace(bad, good)
    return text


def build_stated_constraints(variant: str, category: str, category_type: str) -> dict:
    """
    Build the structured constraints dict consumed by the DAGMetric judge.

    For baselines: empty dict (no constraints stated).
    For constrained: includes the category type and a flag indicating which
    research dimension this prompt tests.
    """
    if variant.lower() == "baseline":
        return {}

    return {
        "category": category.lower(),          # financial / cultural / lifestyle
        "constraint_type": category_type or "unspecified",
        "tests_rq1_financial": category.lower() == "financial",
        "tests_rq2_cultural": category.lower() == "cultural",
        "tests_rq3_lifestyle": category.lower() == "lifestyle",
    }


def convert():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_CSV}. Run from repo root.")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped = 0
    counts = {"financial": 0, "cultural": 0, "lifestyle": 0}
    variant_counts = {"baseline": 0, "constrained": 0}

    # utf-8-sig handles the BOM at the start of your CSV
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)

        for row in reader:
            # Skip empty rows (trailing blank lines in the CSV)
            if not row.get("ID") or not row["ID"].strip():
                skipped += 1
                continue

            prompt_id = row["ID"].strip()
            category = row["Category"].strip()
            variant = row["Varient"].strip()  # CSV header is misspelled "Varient"
            prompt_text = clean_text(row["Prompt Text"])
            category_type = (row.get("Category_Type") or "").strip()

            # Validation
            if variant not in ("Baseline", "Constrained"):
                print(f"  warn: unexpected variant '{variant}' on {prompt_id}")
            if category not in ("Financial", "Cultural", "Lifestyle"):
                print(f"  warn: unexpected category '{category}' on {prompt_id}")

            record = {
                "id": prompt_id,
                "category": category.lower(),
                "variant": variant.lower(),
                "prompt_text": prompt_text,
                "category_type": category_type if category_type else None,
                "stated_constraints": build_stated_constraints(
                    variant, category, category_type
                ),
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

            counts[category.lower()] = counts.get(category.lower(), 0) + 1
            variant_counts[variant.lower()] = variant_counts.get(variant.lower(), 0) + 1

    # Summary
    print(f"\nWrote {rows_written} prompts to {OUTPUT_JSONL}")
    print(f"Skipped {skipped} empty rows")
    print(f"\nBy category:")
    for cat, n in counts.items():
        print(f"  {cat:12s} {n}")
    print(f"\nBy variant:")
    for var, n in variant_counts.items():
        print(f"  {var:12s} {n}")

    if rows_written != 120:
        print(f"\n  WARNING: expected 120 prompts, got {rows_written}. Check CSV.")
    else:
        print(f"\n  All 120 prompts converted successfully.")


if __name__ == "__main__":
    convert()