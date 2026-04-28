"""
src/grounding/compendium.py

Phase 4d — Fitness feasibility grounding via the 2024 Adult Compendium of
Physical Activities.

For each fitness activity in an extracted response, find its MET value in
data/external/compendium_activities.csv (built by src/download_external_data.py),
then compute calorie expenditure at the Compendium's reference body weight
(70 kg ~= 154 lb).

Used by:
  - Lifestyle Feasibility judge: real time + energy commitment, WHO compliance
  - Coverage report: % of fitness components groundable in the Compendium

Design notes:
  * The grounder reports `total_logged_minutes` — the sum of duration_minutes
    across the components it received. It does NOT extrapolate to a week
    unless given an explicit `time_horizon`. Mis-labeling a single day's
    minutes as "weekly" is the easiest way to mis-bucket WHO compliance,
    so we keep the labeling honest.
  * WHO 2020 guideline equivalence: 1 min vigorous (MET >= 6.0) = 2 min
    moderate (MET 3.0-5.9). Reported as `moderate_equivalent_minutes`.
  * Calories use kcal = MET * weight_kg * (minutes / 60), at 70 kg.
  * The CSV's `source` column distinguishes official Compendium 2024 rows
    (MET trusted) from `_custom_*` rows whose MET is a local estimate
    derived from a similar official entry. The grounder passes this
    distinction through so the judge can weight accordingly.

Citation:
  Herrmann SD, Willis EA, Ainsworth BE, et al. (2024). 2024 Adult Compendium
  of Physical Activities. Journal of Sport and Health Science 13(1):6-12.

Usage:
    from src.grounding.compendium import CompendiumGrounder
    g = CompendiumGrounder()

    r = g.lookup("push-ups", duration_minutes=10)

    matches = g.lookup_batch([
        {"activity_name": "push-ups", "duration_minutes": 10},
        {"activity_name": "running 6 mph", "duration_minutes": 30},
    ])
    report = g.coverage_report(components, time_horizon="weekly")

CLI:
    python -m src.grounding.compendium --test "push-ups" "running 6 mph" yoga
    python -m src.grounding.compendium --stats
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.grounding.wikidata import normalize_food_name as _normalize_text

COMPENDIUM_CSV_PATH = Path("data/external/compendium_activities.csv")
MANIFEST_PATH = Path("data/external/MANIFEST.json")

REFERENCE_BODY_WEIGHT_KG = 70.0
MATCH_SCORE_THRESHOLD = 70

# MET intensity bands (per ACSM / Compendium conventions)
MET_LIGHT_MAX = 3.0          # < 3.0 MET = light/sedentary
MET_MODERATE_MAX = 6.0       # 3.0 - 5.9 MET = moderate
# >= 6.0 MET = vigorous

# Fitness-name stopwords. Stopwords drop entirely from match comparison.
STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "with", "in", "for", "to",
    "general", "regular", "training", "exercise", "workout", "session",
    "min", "mins", "minute", "minutes", "hr", "hour", "hours",
    "mph", "kmh", "kph", "kg", "lb", "lbs",
    "pace", "speed", "level",
}

# Numeric tokens — keep separate so we can ignore them by default but
# resurrect them as descriptors when needed (e.g. "running 6 mph").
_NUMERIC_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")

# Descriptors: levels/styles that disambiguate within a category. They
# count for partial scoring but never as the sole basis for a match.
DESCRIPTORS = {
    "light", "moderate", "vigorous",
    "slow", "fast", "brisk", "easy", "intense",
    "low", "high", "medium",
    "leisure", "leisurely", "stationary",
    "ground", "level", "uphill", "downhill",
    "cross", "country",
    "sitting", "standing", "kneeling", "lying",
    "casual", "competitive",
}


# =============================================================
# Helpers
# =============================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    """Lowercase, drop punctuation, return token list (preserves order)."""
    return _TOKEN_RE.findall(s.lower())


def _content_tokens(tokens: list[str]) -> set[str]:
    """Tokens that count as content nouns: not stopwords, descriptors, or numbers."""
    return {
        t for t in tokens
        if t not in STOPWORDS and t not in DESCRIPTORS
        and not _NUMERIC_RE.match(t)
    }


def _meaningful_tokens(tokens: list[str]) -> set[str]:
    """Content tokens + descriptors (everything non-stopword, non-numeric)."""
    return {
        t for t in tokens
        if t not in STOPWORDS and not _NUMERIC_RE.match(t)
    }


def _calories_for_session(
    met: float,
    duration_minutes: float,
    body_weight_kg: float = REFERENCE_BODY_WEIGHT_KG,
) -> float:
    """Standard MET formula: kcal = MET * weight_kg * (minutes / 60)."""
    return met * body_weight_kg * (duration_minutes / 60.0)


def _intensity_band(met: float) -> str:
    if met < MET_LIGHT_MAX:
        return "light"
    if met < MET_MODERATE_MAX:
        return "moderate"
    return "vigorous"


def _moderate_equivalent_minutes(matched: list[dict]) -> dict:
    """
    Sum logged minutes by intensity band.
    WHO equivalence: 1 vigorous minute = 2 moderate-equivalent minutes.
    Light minutes don't count toward WHO targets.
    """
    light = moderate = vigorous = 0.0
    for m in matched:
        d = m.get("duration_minutes")
        if not d:
            continue
        band = _intensity_band(m["met_value"])
        if band == "light":
            light += d
        elif band == "moderate":
            moderate += d
        else:
            vigorous += d
    moderate_equiv = moderate + 2 * vigorous
    return {
        "light_minutes": round(light, 1),
        "moderate_minutes": round(moderate, 1),
        "vigorous_minutes": round(vigorous, 1),
        "moderate_equivalent_minutes": round(moderate_equiv, 1),
    }


def _who_bucket(moderate_equiv_minutes: float, time_horizon: Optional[str]) -> str:
    """
    Bucket moderate-equivalent minutes against WHO 2020 weekly targets.

    Only meaningful when the components actually represent a weekly cadence.
    For other horizons, returns 'indeterminate_horizon' rather than a
    numeric bucket — the judge can decide whether to extrapolate.
    """
    if time_horizon != "weekly":
        return "indeterminate_horizon"
    if moderate_equiv_minutes <= 0:
        return "no_fitness_content"
    if moderate_equiv_minutes < 75:
        return "minimal"
    if moderate_equiv_minutes < 150:
        return "below_who_guideline"
    if moderate_equiv_minutes <= 300:
        return "meets_who_guideline"
    if moderate_equiv_minutes <= 600:
        return "above_guideline"
    return "very_high_volume"


# =============================================================
# Grounder
# =============================================================
class CompendiumGrounder:
    """Fuzzy-match activity names against 2024 Compendium MET values."""

    def __init__(
        self,
        csv_path: Path = COMPENDIUM_CSV_PATH,
        manifest_path: Path = MANIFEST_PATH,
    ):
        self.csv_path = Path(csv_path)
        self.manifest_path = Path(manifest_path)
        self.entries: list[dict] = []
        self.snapshot_date: Optional[str] = None
        self._manifest_cached: Optional[dict] = None
        self._lookup_cache: dict[str, Optional[dict]] = {}

        if not self.csv_path.exists():
            print(f"WARNING: {self.csv_path} not found. "
                  "Run `python -m src.download_external_data` first.")
            return

        skipped = 0
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row["activity_name"].strip().lower()
                    name_normalized = _normalize_text(name)
                    name_tokens = _tokenize(name_normalized)
                    self.entries.append({
                        "code": row["code"].strip(),
                        "activity_name": name,
                        "activity_tokens": name_tokens,
                        "activity_content_tokens": _content_tokens(name_tokens),
                        "activity_meaningful_tokens": _meaningful_tokens(name_tokens),
                        "category": row.get("category", "").strip(),
                        "met_value": float(row["met_value"]),
                        "source": row.get("source", "compendium_2024").strip(),
                        "snapshot_date": row.get("snapshot_date"),
                    })
                    if self.snapshot_date is None:
                        self.snapshot_date = row.get("snapshot_date")
                except (KeyError, ValueError, TypeError):
                    skipped += 1
                    continue

        if not self.entries:
            print(f"WARNING: {self.csv_path} loaded 0 valid rows "
                  f"(skipped {skipped}).")
        else:
            n_official = sum(1 for e in self.entries
                             if e["source"] == "compendium_2024")
            n_custom = len(self.entries) - n_official
            print(f"Compendium: loaded {len(self.entries)} activities "
                  f"({n_official} official + {n_custom} custom; "
                  f"snapshot {self.snapshot_date or 'unknown'}, "
                  f"skipped {skipped}).")

    # ---------------------------------------------------- scoring
    def _score_match(
        self,
        query_tokens: list[str],
        query_content: set[str],
        entry: dict,
    ) -> int:
        """Same rubric as src/grounding/bls.py — see that file for tier doc."""
        cand_content = entry["activity_content_tokens"]
        cand_meaningful = entry["activity_meaningful_tokens"]

        if query_tokens == entry["activity_tokens"]:
            return 100

        if not query_content:
            return 0

        if query_content.issubset(cand_content):
            return 80

        if query_content.issubset(cand_meaningful):
            return 60

        overlap = query_content & cand_content
        if overlap:
            return 40 + 10 * len(overlap)

        return 0

    # ---------------------------------------------------- public lookup
    def lookup(
        self,
        activity_name: str,
        duration_minutes: Optional[float] = None,
    ) -> Optional[dict]:
        """Match one activity to a Compendium entry. Optionally compute kcal."""
        if not self.entries or not activity_name:
            return None

        normalized = _normalize_text(activity_name)
        if not normalized:
            return None

        # Cache the raw fuzzy match (without duration) so multiple
        # components mentioning the same activity don't re-score.
        cache_key = normalized
        cached = self._lookup_cache.get(cache_key, "MISS")
        if cached != "MISS":
            best = cached
        else:
            q_tokens = _tokenize(normalized)
            q_content = _content_tokens(q_tokens)

            best_score = 0
            best_entry: Optional[dict] = None
            for entry in self.entries:
                score = self._score_match(q_tokens, q_content, entry)
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    if score == 100:
                        break

            if best_entry is None or best_score < MATCH_SCORE_THRESHOLD:
                self._lookup_cache[cache_key] = None
                return None

            if best_score >= 90:
                confidence = "high"
            elif best_score >= 75:
                confidence = "medium"
            else:
                confidence = "low"

            best = {
                "matched_compendium_activity": best_entry["activity_name"],
                "compendium_code": best_entry["code"],
                "category": best_entry["category"],
                "met_value": best_entry["met_value"],
                "intensity_band": _intensity_band(best_entry["met_value"]),
                "compendium_source": best_entry["source"],
                "snapshot_date": best_entry["snapshot_date"],
                "match_score": best_score,
                "confidence": confidence,
                "source": "compendium_2024",
            }
            self._lookup_cache[cache_key] = best

        if best is None:
            return None

        # Build the per-call result (cached match + this call's duration/kcal)
        result = {"activity_name": activity_name, **best}
        if duration_minutes:
            try:
                d = float(duration_minutes)
            except (TypeError, ValueError):
                d = None
            if d and d > 0:
                result["duration_minutes"] = d
                result["estimated_kcal_70kg"] = round(
                    _calories_for_session(best["met_value"], d), 1
                )
        return result

    def lookup_batch(
        self,
        fitness_components: list[dict],
    ) -> list[Optional[dict]]:
        """
        Return one match (or None) per input component, preserving the
        component's original metadata alongside the Compendium match.
        """
        results: list[Optional[dict]] = []
        for comp in fitness_components:
            name = comp.get("activity_name") or comp.get("name")
            if not name:
                results.append(None)
                continue
            duration = comp.get("duration_minutes")
            match = self.lookup(name, duration_minutes=duration)
            if match is None:
                results.append(None)
                continue
            # Preserve original component context — judge needs it
            results.append({
                **match,
                "component_input": {
                    "day_or_slot": comp.get("day_or_slot"),
                    "activity_type": comp.get("activity_type"),
                    "intensity_self_reported": comp.get("intensity"),
                    "equipment_needed": comp.get("equipment_needed"),
                    "location": comp.get("location"),
                },
            })
        return results

    def coverage_report(
        self,
        fitness_components: list[dict],
        time_horizon: Optional[str] = None,
    ) -> dict:
        """
        Aggregate stats for a fitness response.

        Args:
          fitness_components: extraction's `fitness_components` list.
          time_horizon: extraction's `routine_structure.time_horizon`. Only
            "weekly" produces a numeric WHO bucket; others get
            "indeterminate_horizon".
        """
        if not fitness_components:
            return self._empty_coverage(time_horizon)

        per_component = self.lookup_batch(fitness_components)

        matched: list[dict] = []
        unmatched: list[str] = []
        total_logged_minutes = 0.0
        total_kcal = 0.0

        for comp, m in zip(fitness_components, per_component):
            name = comp.get("activity_name") or comp.get("name") or "<unnamed>"
            if m is None:
                unmatched.append(name)
                continue
            matched.append(m)
            d = m.get("duration_minutes")
            if d:
                total_logged_minutes += d
                total_kcal += m.get("estimated_kcal_70kg", 0.0)

        intensity_breakdown = _moderate_equivalent_minutes(matched)
        bucket = _who_bucket(
            intensity_breakdown["moderate_equivalent_minutes"], time_horizon
        )

        return {
            "matched": matched,
            "unmatched": unmatched,
            "matched_count": len(matched),
            "total_count": len(fitness_components),
            "coverage_ratio": (
                len(matched) / len(fitness_components)
                if fitness_components else 0.0
            ),
            "time_horizon": time_horizon,
            "total_logged_minutes": round(total_logged_minutes, 1),
            "total_logged_kcal_70kg": round(total_kcal, 1),
            **intensity_breakdown,
            "feasibility_assessment": bucket,
            "snapshot_date": self.snapshot_date,
            "source": "compendium_2024",
        }

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
                "compendium_activities.csv", {"available": False}
            )
        except (OSError, json.JSONDecodeError) as e:
            self._manifest_cached = {"available": False, "error": str(e)[:200]}
        return self._manifest_cached

    def stats(self) -> dict:
        n_official = sum(1 for e in self.entries
                         if e["source"] == "compendium_2024")
        n_custom = len(self.entries) - n_official
        by_band: dict[str, int] = {"light": 0, "moderate": 0, "vigorous": 0}
        for e in self.entries:
            by_band[_intensity_band(e["met_value"])] += 1
        return {
            "total_entries": len(self.entries),
            "official_entries": n_official,
            "custom_entries": n_custom,
            "by_intensity_band": by_band,
            "snapshot_date": self.snapshot_date,
            "csv_path": str(self.csv_path),
            "match_score_threshold": MATCH_SCORE_THRESHOLD,
            "manifest": self.manifest_info(),
        }

    # ---------------------------------------------------- internals
    def _empty_coverage(self, time_horizon: Optional[str]) -> dict:
        return {
            "matched": [],
            "unmatched": [],
            "matched_count": 0,
            "total_count": 0,
            "coverage_ratio": 0.0,
            "time_horizon": time_horizon,
            "total_logged_minutes": 0.0,
            "total_logged_kcal_70kg": 0.0,
            "light_minutes": 0.0,
            "moderate_minutes": 0.0,
            "vigorous_minutes": 0.0,
            "moderate_equivalent_minutes": 0.0,
            "feasibility_assessment": "no_fitness_content",
            "snapshot_date": self.snapshot_date,
            "source": "compendium_2024",
        }


# =============================================================
# CLI
# =============================================================
def _print_match(query: str, r: Optional[dict]):
    if r is None:
        print(f"  {query:35s}  -- no match")
        return
    line = (f"  {query:35s}  -> {r['matched_compendium_activity']}  "
            f"[{r['compendium_code']}, {r['intensity_band']}, "
            f"MET={r['met_value']}, score={r['match_score']}, "
            f"conf={r['confidence']}, source={r['compendium_source']}]")
    if "estimated_kcal_70kg" in r:
        line += f"  kcal@70kg={r['estimated_kcal_70kg']}"
    print(line)


def main():
    parser = argparse.ArgumentParser(description="2024 Compendium grounder")
    parser.add_argument("--csv", default=str(COMPENDIUM_CSV_PATH))
    parser.add_argument("--test", nargs="+",
                        default=["push-ups", "running 6 mph", "yoga",
                                 "irish step dance", "burpees", "salt"],
                        help="Activity names to test")
    parser.add_argument("--duration", type=float, default=None,
                        help="Minutes per activity for kcal computation.")
    parser.add_argument("--time-horizon", default=None,
                        choices=[None, "single_meal", "single_day",
                                 "multi_day", "weekly", "open_ended"],
                        help="Pass-through to coverage_report().")
    parser.add_argument("--coverage", action="store_true",
                        help="Run coverage_report and print summary.")
    parser.add_argument("--stats", action="store_true",
                        help="Print loader stats and exit.")
    args = parser.parse_args()

    g = CompendiumGrounder(csv_path=Path(args.csv))

    if args.stats:
        print(json.dumps(g.stats(), indent=2, default=str))
        return

    print(f"Looking up {len(args.test)} activity(ies) "
          f"against {g.csv_path}\n")
    for q in args.test:
        r = g.lookup(q, duration_minutes=args.duration)
        _print_match(q, r)

    if args.coverage:
        print("\nCoverage report:")
        components = [
            {"activity_name": q, "duration_minutes": args.duration or 30}
            for q in args.test
        ]
        report = g.coverage_report(components, time_horizon=args.time_horizon)
        compact = {k: v for k, v in report.items()
                   if k not in ("matched",)}
        print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()
