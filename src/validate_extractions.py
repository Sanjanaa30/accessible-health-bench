"""
src/validate_extractions.py

Quick validator for Phase 3 extractions. Audits every file in
data/extractions/ for:
  - JSON parse errors (where extraction failed)
  - Missing top-level required fields
  - Wrong types on key fields (lists / dicts)
  - Suspicious empty extractions
  - Per-provider counts

Run from repo root:
    python -m src.validate_extractions

This is a pure local script — no API calls, no cost. Run after
every extraction batch to catch problems before Phase 4.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EXTRACTIONS_DIR, MODELS

# Top-level keys the rich schema must contain
REQUIRED_FIELDS = [
    "summary",
    "response_type",
    "primary_goal",
    "meal_components",
    "all_ingredients",
    "all_dishes_or_foods_named",
    "fitness_components",
    "routine_structure",
    "cost_information",
    "cultural_signals",
    "feasibility_signals",
    "household_and_demographic_context",
    "medical_or_health_signals",
    "constraint_adherence",
    "caveats_and_disclaimers",
    "extraction_notes",
]

# Fields that must be lists
ARRAY_FIELDS = [
    "meal_components",
    "all_ingredients",
    "all_dishes_or_foods_named",
    "fitness_components",
    "caveats_and_disclaimers",
]

# Fields that must be dicts
DICT_FIELDS = [
    "routine_structure",
    "cost_information",
    "cultural_signals",
    "feasibility_signals",
    "household_and_demographic_context",
    "medical_or_health_signals",
    "constraint_adherence",
]


def validate_extraction(record: dict) -> list:
    """Return a list of issues found. Empty list = file is valid."""
    issues = []

    extracted = record.get("extracted")
    if extracted is None:
        return ["missing 'extracted' key"]

    # Parse error marker (set by extract.py when JSON parsing failed)
    if isinstance(extracted, dict) and "_extraction_error" in extracted:
        issues.append(f"json_parse_error: {extracted['_extraction_error'][:60]}")
        return issues  # don't validate further if parse failed

    if not isinstance(extracted, dict):
        return [f"extracted should be a dict, got {type(extracted).__name__}"]

    # Check top-level required fields
    for field in REQUIRED_FIELDS:
        if field not in extracted:
            issues.append(f"missing field: {field}")

    # Check that array fields are lists
    for field in ARRAY_FIELDS:
        if field in extracted and not isinstance(extracted[field], list):
            issues.append(
                f"{field} should be a list, got {type(extracted[field]).__name__}"
            )

    # Check that dict fields are dicts
    for field in DICT_FIELDS:
        if field in extracted and not isinstance(extracted[field], dict):
            issues.append(
                f"{field} should be a dict, got {type(extracted[field]).__name__}"
            )

    # Sanity check: response should produce SOMETHING, unless it's pure advisory
    response_type = extracted.get("response_type", "")
    has_meal_content = bool(extracted.get("meal_components")) or bool(
        extracted.get("all_ingredients")
    )
    has_fitness_content = bool(extracted.get("fitness_components"))
    has_dishes = bool(extracted.get("all_dishes_or_foods_named"))

    if (
        not has_meal_content
        and not has_fitness_content
        and not has_dishes
        and response_type not in ("advisory", "other", "")
    ):
        issues.append(
            f"suspicious: empty content (no meals/fitness/dishes) but "
            f"response_type='{response_type}'"
        )

    return issues


def main():
    extractions_root = Path(EXTRACTIONS_DIR)
    if not extractions_root.exists():
        print(f"No extractions directory found at {extractions_root}")
        return

    total_files = 0
    files_with_issues = 0
    issue_counts = defaultdict(int)
    bad_files = []
    per_provider = defaultdict(int)
    per_provider_clean = defaultdict(int)

    for provider in MODELS.keys():
        provider_dir = extractions_root / provider
        if not provider_dir.exists():
            continue

        for ext_file in sorted(provider_dir.glob("*.json")):
            total_files += 1
            per_provider[provider] += 1

            try:
                record = json.loads(ext_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                files_with_issues += 1
                bad_files.append((provider, ext_file.stem, ["file itself is invalid JSON"]))
                issue_counts["file_corrupt"] += 1
                continue

            issues = validate_extraction(record)
            if issues:
                files_with_issues += 1
                bad_files.append((provider, ext_file.stem, issues))
                for issue in issues:
                    issue_type = issue.split(":", 1)[0].strip()
                    issue_counts[issue_type] += 1
            else:
                per_provider_clean[provider] += 1

    # Report
    print("=" * 60)
    print("EXTRACTION VALIDATION REPORT")
    print("=" * 60)
    print(f"Total extraction files:  {total_files}")
    print(f"Clean files:             {total_files - files_with_issues}")
    print(f"Files with issues:       {files_with_issues}")

    print("\nFiles per provider (clean / total):")
    for provider in MODELS.keys():
        total = per_provider.get(provider, 0)
        clean = per_provider_clean.get(provider, 0)
        marker = "OK" if total == 120 else f"expected 120"
        print(f"  {provider:12s} {clean:3d} clean / {total:3d} total  [{marker}]")

    if issue_counts:
        print("\nIssue type counts:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type:35s} {count}")

    if bad_files:
        print("\nFirst 15 problem files:")
        for provider, prompt_id, issues in bad_files[:15]:
            print(f"  {provider}/{prompt_id}: {issues[0]}")
        if len(bad_files) > 15:
            print(f"  ... and {len(bad_files) - 15} more")
    else:
        print("\nAll extractions look clean. Ready for Phase 4.")


if __name__ == "__main__":
    main()