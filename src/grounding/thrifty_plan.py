"""
src/grounding/thrifty_plan.py

Phase 4c — Budget calibration via USDA Cost of Food at Home.

Given a household composition extracted from a response, return USDA-anchored
weekly cost benchmarks (thrifty / low / moderate / liberal) and bucket the
response's own implied cost relative to those benchmarks.

Used by:
  - Affordability judge: did the response's plan land near the thrifty tier
    (which is what SNAP allotments approximate) or above the liberal tier?
  - Coverage report: how many responses had a household that USDA covers?

CLI:
    python -m src.grounding.thrifty_plan --household-type family_with_children
    python -m src.grounding.thrifty_plan --household-type single_parent --household-size 3 --cost 75
    python -m src.grounding.thrifty_plan --stats
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

THRIFTY_CSV_PATH = Path("data/external/usda_thrifty_plan.csv")
MANIFEST_PATH = Path("data/external/MANIFEST.json")

# USDA small-household adjustment scalars.
# Source: USDA "Cost of Food at Home" report footnote — costs assume the
# 4-person reference family; smaller and larger households get scaled
# because of economies of scale in food preparation and bulk pricing.
USDA_HOUSEHOLD_ADJUSTMENT = {
    1: 1.20,
    2: 1.10,
    3: 1.05,
    4: 1.00,
    5: 0.95,
    6: 0.95,
    # 7+ -> 0.90 (handled in code)
}

# Default household-type compositions. The `family_with_children` profile
# matches USDA's published "reference family of four" so the totals
# reproduce the official report.
HOUSEHOLD_PROFILES = {
    "single":               [("female_19_50_years", 1.0)],  # safer under-est
    "couple":               [("male_19_50_years", 1.0),
                             ("female_19_50_years", 1.0)],
    "family_with_children": [("male_19_50_years", 1.0),
                             ("female_19_50_years", 1.0),
                             ("child_6_8_years", 1.0),
                             ("child_9_11_years", 1.0)],
    "single_parent":        [("female_19_50_years", 1.0),
                             ("child_6_8_years", 1.0)],
    "roommates":            [("male_19_50_years", 1.0),
                             ("female_19_50_years", 1.0)],
    "shared_household":     [("male_19_50_years", 1.0),
                             ("female_19_50_years", 1.0)],
    "unspecified":          [("female_19_50_years", 1.0)],  # safer under-est
}

# Map age-reference strings (from extraction) to USDA age-sex group keys.
# We pick the closest USDA group; gender defaults to female (lower cost,
# under-estimate-safe). The judge can override gender via household_type.
AGE_REFERENCE_TO_USDA = {
    "toddler":            "child_1_3_years",
    "1_year_old":         "child_1_3_years",
    "2_year_old":         "child_1_3_years",
    "3_year_old":         "child_1_3_years",
    "4_year_old":         "child_4_5_years",
    "5_year_old":         "child_4_5_years",
    "child":              "child_6_8_years",
    "school_age":         "child_6_8_years",
    "preteen":            "child_9_11_years",
    "teenager":           "female_14_18_years",
    "college_student":    "female_19_50_years",
    "young_adult":        "female_19_50_years",
    "adult":              "female_19_50_years",
    "middle_aged":        "female_51_70_years",
    "senior":             "female_71_plus_years",
    "70_year_old":        "female_71_plus_years",
    "elderly":            "female_71_plus_years",
    "pregnant":           "pregnant_or_lactating",
    "lactating":          "pregnant_or_lactating",
}

# Symmetric band width used by the cost classifier (±15%).
CLASSIFY_BAND = 0.15


# =============================================================
# Grounder
# =============================================================
class ThriftyPlanGrounder:
    """USDA-anchored weekly food cost benchmarks for a household."""

    def __init__(self, csv_path: Path = THRIFTY_CSV_PATH,
                 manifest_path: Path = MANIFEST_PATH):
        self.csv_path = Path(csv_path)
        self.manifest_path = Path(manifest_path)
        self.table: dict[str, dict[str, float]] = {}
        self.snapshot_date: Optional[str] = None
        self._manifest_cached: Optional[dict] = None

        if not self.csv_path.exists():
            print(f"WARNING: {self.csv_path} not found. "
                  "Run `python -m src.download_external_data` first.")
            return

        skipped = 0
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    group = row["age_sex_group"].strip()
                    self.table[group] = {
                        "thrifty":  float(row["weekly_cost_thrifty_usd"]),
                        "low":      float(row["weekly_cost_low_usd"]),
                        "moderate": float(row["weekly_cost_moderate_usd"]),
                        "liberal":  float(row["weekly_cost_liberal_usd"]),
                    }
                    if self.snapshot_date is None:
                        self.snapshot_date = row.get("snapshot_date")
                except (KeyError, ValueError, TypeError):
                    skipped += 1
                    continue

        if not self.table:
            print(f"WARNING: {self.csv_path} loaded 0 valid rows "
                  f"(skipped {skipped}).")
        else:
            print(f"USDA Thrifty: loaded {len(self.table)} groups "
                  f"(snapshot {self.snapshot_date or 'unknown'}, "
                  f"skipped {skipped}).")

    # ---------------------------------------------------- helpers
    @staticmethod
    def _adjustment_factor(person_count: int) -> float:
        if person_count <= 0:
            return 1.0
        if person_count >= 7:
            return 0.90
        return USDA_HOUSEHOLD_ADJUSTMENT.get(person_count, 1.0)

    def _resolve_profile(
        self,
        household_size: Optional[int],
        household_type: Optional[str],
        ages_referenced: Optional[list[str]],
    ) -> tuple[list[tuple[str, float]], str]:
        """
        Determine the (group, weight) profile to use.
        Precedence:
          1. ages_referenced overrides everything if it resolves to USDA groups
          2. household_type fixes the composition
          3. household_size scales a mixed-gender adult pair
          4. Default to single adult female

        When BOTH household_type AND household_size are given and disagree,
        we use the type's composition but pad/trim with female_19_50 entries
        so the total person count matches household_size.
        """
        # Branch 1: ages_referenced
        if ages_referenced:
            mapped = [
                AGE_REFERENCE_TO_USDA[a] for a in ages_referenced
                if a in AGE_REFERENCE_TO_USDA
            ]
            if mapped:
                profile = [(g, 1.0) for g in mapped]
                return profile, f"ages_referenced({len(mapped)})"

        # Branch 2 + reconciliation with household_size
        if household_type and household_type in HOUSEHOLD_PROFILES:
            base = list(HOUSEHOLD_PROFILES[household_type])
            base_size = sum(w for _, w in base)
            if household_size and household_size != int(base_size):
                # Pad with adult females or trim trailing entries.
                target = int(household_size)
                if target > base_size:
                    extra = target - int(base_size)
                    base += [("female_19_50_years", 1.0)] * extra
                    profile_name = f"{household_type}_padded_to_{target}"
                else:
                    base = base[:target]
                    profile_name = f"{household_type}_trimmed_to_{target}"
                return base, profile_name
            return base, household_type

        # Branch 3: scale mixed-gender adults
        if household_size and household_size > 0:
            n = int(household_size)
            half = n // 2
            other = n - half
            profile = [
                ("male_19_50_years", float(half)),
                ("female_19_50_years", float(other)),
            ]
            # Drop zero-weight entries cleanly
            profile = [(g, w) for g, w in profile if w > 0]
            return profile, f"scaled_mixed_gender_size_{n}"

        # Branch 4: default
        return list(HOUSEHOLD_PROFILES["unspecified"]), "default_single_adult"

    # ---------------------------------------------------- public API
    def estimate_for_household(
        self,
        household_size: Optional[int] = None,
        household_type: Optional[str] = None,
        ages_referenced: Optional[list[str]] = None,
    ) -> dict:
        """Compute USDA-anchored weekly cost benchmarks for the household."""
        if not self.table:
            return self._empty_result(household_size, household_type)

        profile, profile_used = self._resolve_profile(
            household_size, household_type, ages_referenced
        )
        person_count = int(round(sum(w for _, w in profile))) or 1
        adjustment = self._adjustment_factor(person_count)

        totals = {"thrifty": 0.0, "low": 0.0, "moderate": 0.0, "liberal": 0.0}
        missing_groups: list[str] = []
        for group, weight in profile:
            if group not in self.table:
                missing_groups.append(group)
                continue
            for plan, cost in self.table[group].items():
                totals[plan] += cost * weight

        # Apply USDA small-household scalar
        for plan in totals:
            totals[plan] = round(totals[plan] * adjustment, 2)

        return {
            "profile_used": profile_used,
            "household_size": household_size,
            "household_type": household_type,
            "ages_referenced": ages_referenced,
            "person_count": person_count,
            "adjustment_factor": adjustment,
            "missing_groups": missing_groups,
            "weekly_baseline_usd": totals,
            "snapshot_date": self.snapshot_date,
            "source": "usda_thrifty_plan",
        }

    def classify_response_cost(
        self,
        response_estimated_cost_usd: Optional[float],
        household_baseline: dict,
    ) -> str:
        """Bucket a response's implied weekly cost against the household's USDA tiers."""
        if response_estimated_cost_usd is None:
            return "no_cost_provided"
        try:
            c = float(response_estimated_cost_usd)
        except (TypeError, ValueError):
            return "invalid_input"
        if c < 0:
            return "invalid_input"

        b = household_baseline.get("weekly_baseline_usd", {})
        if not b or any(b.get(k) in (None, 0.0) for k in
                        ("thrifty", "low", "moderate", "liberal")):
            return "no_baseline"

        thrifty, low, mod, lib = b["thrifty"], b["low"], b["moderate"], b["liberal"]

        # Symmetric ±CLASSIFY_BAND windows around each tier.
        if c < thrifty * (1 - CLASSIFY_BAND):
            return "well_below_thrifty"
        if c <= thrifty * (1 + CLASSIFY_BAND):
            return "near_thrifty"
        if c <= low * (1 + CLASSIFY_BAND):
            return "near_low"
        if c <= mod * (1 + CLASSIFY_BAND):
            return "near_moderate"
        if c <= lib * (1 + CLASSIFY_BAND):
            return "near_liberal"
        return "above_liberal"

    def classify_household_response(
        self,
        response_estimated_cost_usd: Optional[float],
        household_size: Optional[int] = None,
        household_type: Optional[str] = None,
        ages_referenced: Optional[list[str]] = None,
    ) -> dict:
        """Convenience: estimate the baseline AND classify the cost in one call."""
        baseline = self.estimate_for_household(
            household_size=household_size,
            household_type=household_type,
            ages_referenced=ages_referenced,
        )
        bucket = self.classify_response_cost(
            response_estimated_cost_usd, baseline
        )
        return {
            "bucket": bucket,
            "response_cost_usd": response_estimated_cost_usd,
            "baseline": baseline,
        }

    def lookup_batch(
        self,
        households: list[dict],
    ) -> list[dict]:
        """
        Estimate baselines for many households. Each input dict may include:
          household_size, household_type, ages_referenced.
        """
        return [
            self.estimate_for_household(
                household_size=h.get("household_size"),
                household_type=h.get("household_type"),
                ages_referenced=h.get("ages_referenced"),
            )
            for h in households
        ]

    # ---------------------------------------------------- introspection
    def manifest_info(self) -> dict:
        if self._manifest_cached is not None:
            return self._manifest_cached
        if not self.manifest_path.exists():
            self._manifest_cached = {"available": False}
            return self._manifest_cached
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifest_cached = data.get("files", {}).get(
                "usda_thrifty_plan.csv", {"available": False}
            )
        except (OSError, json.JSONDecodeError) as e:
            self._manifest_cached = {"available": False, "error": str(e)[:200]}
        return self._manifest_cached

    def stats(self) -> dict:
        return {
            "groups_loaded": len(self.table),
            "snapshot_date": self.snapshot_date,
            "csv_path": str(self.csv_path),
            "household_profiles_known": list(HOUSEHOLD_PROFILES.keys()),
            "adjustment_factors": {
                **USDA_HOUSEHOLD_ADJUSTMENT, "7+": 0.90,
            },
            "manifest": self.manifest_info(),
        }

    # ---------------------------------------------------- internals
    def _empty_result(self, household_size, household_type) -> dict:
        return {
            "profile_used": None,
            "household_size": household_size,
            "household_type": household_type,
            "ages_referenced": None,
            "person_count": 0,
            "adjustment_factor": None,
            "missing_groups": [],
            "weekly_baseline_usd": {
                "thrifty": None, "low": None, "moderate": None, "liberal": None,
            },
            "snapshot_date": None,
            "source": "unavailable",
        }


# =============================================================
# CLI
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="USDA Thrifty Plan grounder")
    parser.add_argument("--csv", default=str(THRIFTY_CSV_PATH))
    parser.add_argument("--household-type", default=None,
                        choices=list(HOUSEHOLD_PROFILES.keys()) + [None])
    parser.add_argument("--household-size", type=int, default=None)
    parser.add_argument("--ages", nargs="+", default=None,
                        help="Age reference tags (toddler, college_student, ...)")
    parser.add_argument("--cost", type=float, default=None,
                        help="Response's estimated weekly cost (USD); if given, "
                             "also runs classification.")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    g = ThriftyPlanGrounder(csv_path=Path(args.csv))

    if args.stats:
        print(json.dumps(g.stats(), indent=2, default=str))
        return

    if args.cost is not None:
        result = g.classify_household_response(
            response_estimated_cost_usd=args.cost,
            household_size=args.household_size,
            household_type=args.household_type,
            ages_referenced=args.ages,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    baseline = g.estimate_for_household(
        household_size=args.household_size,
        household_type=args.household_type,
        ages_referenced=args.ages,
    )
    print(json.dumps(baseline, indent=2, default=str))


if __name__ == "__main__":
    main()
