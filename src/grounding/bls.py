"""
src/grounding/bls.py

Phase 4b — Affordability grounding via BLS Average Retail Food Prices.

Fuzzy-matches each ingredient against data/external/bls_prices.csv (built by
src/download_external_data.py). Returns the matching item's real US average
retail price, the unit it's priced in, the year/period of the observation,
and the BLS series ID.

We deliberately do NOT compute a "total cost" estimate here. BLS prices are
in mixed units (per lb, per dozen, per gallon, ...); naive summation produces
a meaningless number. The affordability judge is responsible for converting
unit prices to a per-meal/per-week cost using portion sizes.

Used by:
  - Affordability judge: per-ingredient unit price + provenance
  - Coverage report: % of ingredients groundable in BLS

Usage:
    from src.grounding.bls import BLSGrounder
    g = BLSGrounder()

    r = g.lookup("rice")
    # -> {"matched_bls_item": "rice, white, long-grain, uncooked",
    #     "price_usd": 0.87, "unit": "per_lb", "year": "2024", "period": "M06",
    #     "series_id": "APU0000701311", "confidence": "high",
    #     "match_score": 100, "source": "bls"}

    batch = g.lookup_batch(["rice", "salt", "egusi"])      # per-ingredient list
    report = g.coverage_report(["rice", "salt", "egusi"])  # aggregate stats

CLI:
    python -m src.grounding.bls --test rice "chicken breast" eggs salt
    python -m src.grounding.bls --stats
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.grounding.wikidata import normalize_food_name

# Default path — keep in sync with src/download_external_data.py.
# Add BLS_PRICES_CSV to src/config.py if you want centralized config.
BLS_CSV_PATH = Path("data/external/bls_prices.csv")
MANIFEST_PATH = Path("data/external/MANIFEST.json")

# Tokens we never count as match evidence. These are descriptors, not the
# nouns that identify the food. Kept small on purpose — over-aggressive
# stopword lists silently break short queries.
STOPWORDS = {
    "a", "an", "and", "or", "of", "the", "with", "in", "for", "to",
    "fresh", "natural", "regular", "all", "100", "grade", "prepackaged",
    "bulk", "uncooked", "cooked", "raw",
}

# Tokens that describe color/condition rather than the food itself. These
# can match (lower weight than content nouns) but never alone — a query
# of pure descriptors won't pass the score threshold.
DESCRIPTORS = {
    "white", "red", "green", "black", "brown", "yellow",
    "low", "high", "lean", "extra", "light", "dark",
    "large", "medium", "small", "long", "short", "field",
    "pan", "ground", "boneless", "sliced", "whole", "fortified",
}

MATCH_SCORE_THRESHOLD = 70


# =============================================================
# Tokenization
# =============================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    """Lowercase, drop punctuation, return token list. Preserves order."""
    return _TOKEN_RE.findall(s.lower())


def _content_tokens(tokens: list[str]) -> set[str]:
    """Tokens that count as content nouns: not stopwords, not descriptors."""
    return {t for t in tokens if t not in STOPWORDS and t not in DESCRIPTORS}


def _all_meaningful_tokens(tokens: list[str]) -> set[str]:
    """Content tokens + descriptors (everything except stopwords)."""
    return {t for t in tokens if t not in STOPWORDS}


# =============================================================
# BLS grounder
# =============================================================
class BLSGrounder:
    """Fuzzy-match ingredient names against BLS staple food prices."""

    def __init__(self, csv_path: Path = BLS_CSV_PATH,
                 manifest_path: Path = MANIFEST_PATH):
        self.csv_path = Path(csv_path)
        self.manifest_path = Path(manifest_path)
        self.entries: list[dict] = []
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
                    item_text = row["item"].strip().lower()
                    item_tokens = _tokenize(item_text)
                    self.entries.append({
                        "item": item_text,
                        "item_tokens": item_tokens,
                        "item_content_tokens": _content_tokens(item_tokens),
                        "item_meaningful_tokens": _all_meaningful_tokens(item_tokens),
                        "price_usd": float(row["price_usd"]),
                        "unit": row.get("unit", ""),
                        "currency": row.get("currency", "USD"),
                        "area_code": row.get("area_code", "0000"),
                        "year": row.get("year"),
                        "period": row.get("period"),
                        "series_id": row.get("series_id"),
                    })
                except (KeyError, ValueError, TypeError) as e:
                    skipped += 1
                    continue

        if not self.entries:
            print(f"WARNING: {self.csv_path} loaded 0 valid rows "
                  f"(skipped {skipped}).")
        else:
            print(f"BLS: loaded {len(self.entries)} prices "
                  f"(skipped {skipped} malformed rows).")

    # ---------------------------------------------------- scoring
    def _score_match(self, query_tokens: list[str],
                     query_content: set[str],
                     entry: dict) -> int:
        """
        Return 0..100. Threshold for acceptance is MATCH_SCORE_THRESHOLD.

        Tiers:
          100  exact tokenized equality
           80  every query content token is in candidate's content tokens
           60  query content is a subset of candidate's meaningful tokens
                (i.e. content matches even if some descriptor overlaps)
           40 + 10*N   N shared content tokens (partial overlap)
            0  no content overlap at all
        """
        cand_content = entry["item_content_tokens"]
        cand_meaningful = entry["item_meaningful_tokens"]

        if query_tokens == entry["item_tokens"]:
            return 100

        # Disqualify queries with no content nouns at all (pure descriptors)
        if not query_content:
            return 0

        if query_content.issubset(cand_content) and query_content:
            return 80

        if query_content.issubset(cand_meaningful):
            return 60

        overlap = query_content & cand_content
        if overlap:
            return 40 + 10 * len(overlap)

        return 0

    # ---------------------------------------------------- public lookups
    def lookup(self, ingredient: str) -> Optional[dict]:
        """
        Find the best BLS match for one ingredient. Returns a result dict or
        None if no candidate cleared the score threshold.
        """
        if not self.entries or not ingredient:
            return None

        normalized = normalize_food_name(ingredient)
        if not normalized:
            return None

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
                    break  # can't improve

        if best_entry is None or best_score < MATCH_SCORE_THRESHOLD:
            return None

        if best_score >= 90:
            confidence = "high"
        elif best_score >= 75:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "ingredient": ingredient,
            "matched_bls_item": best_entry["item"],
            "price_usd": best_entry["price_usd"],
            "unit": best_entry["unit"],
            "currency": best_entry["currency"],
            "area_code": best_entry["area_code"],
            "year": best_entry["year"],
            "period": best_entry["period"],
            "series_id": best_entry["series_id"],
            "match_score": best_score,
            "confidence": confidence,
            "source": "bls",
        }

    def lookup_batch(self, ingredients: list[str]) -> list[Optional[dict]]:
        """Return one match (or None) per input ingredient, in input order.
        Duplicate ingredients reuse the first computed match."""
        cache: dict[str, Optional[dict]] = {}
        out: list[Optional[dict]] = []
        for ing in ingredients:
            key = normalize_food_name(ing or "")
            if key in cache:
                out.append(cache[key])
                continue
            r = self.lookup(ing)
            cache[key] = r
            out.append(r)
        return out

    def coverage_report(self, ingredients: list[str]) -> dict:
        """
        Aggregate stats — used for diagnostics, not for cost math.
        Deduplicates against the normalized form so a list with repeats
        doesn't inflate counts.
        """
        seen: set[str] = set()
        unique_inputs: list[str] = []
        for ing in ingredients:
            key = normalize_food_name(ing or "")
            if key and key not in seen:
                seen.add(key)
                unique_inputs.append(ing)

        matched: list[dict] = []
        unmatched: list[str] = []
        for ing in unique_inputs:
            r = self.lookup(ing)
            (matched if r else unmatched).append(r if r else ing)

        return {
            "total_unique_ingredients": len(unique_inputs),
            "matched_count": len(matched),
            "unmatched_count": len(unmatched),
            "coverage_ratio": (
                len(matched) / len(unique_inputs) if unique_inputs else 0.0
            ),
            "matched": matched,
            "unmatched": unmatched,
            "snapshot": self.manifest_info(),
        }

    # ---------------------------------------------------- introspection
    def manifest_info(self) -> dict:
        """Return the BLS file's entry from MANIFEST.json (snapshot date,
        SHA256, fetched_at_utc, etc.) so the judge can record provenance."""
        if self._manifest_cached is not None:
            return self._manifest_cached
        if not self.manifest_path.exists():
            self._manifest_cached = {"available": False}
            return self._manifest_cached
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifest_cached = data.get("files", {}).get(
                "bls_prices.csv", {"available": False}
            )
        except (OSError, json.JSONDecodeError) as e:
            self._manifest_cached = {"available": False, "error": str(e)[:200]}
        return self._manifest_cached

    def stats(self) -> dict:
        """Diagnostic counts."""
        units = {}
        for e in self.entries:
            units[e["unit"]] = units.get(e["unit"], 0) + 1
        return {
            "total_entries": len(self.entries),
            "csv_path": str(self.csv_path),
            "by_unit": units,
            "manifest": self.manifest_info(),
        }


# =============================================================
# CLI
# =============================================================
def _print_match(query: str, r: Optional[dict]):
    if r is None:
        print(f"  {query:30s}  -- no match")
        return
    print(f"  {query:30s}  -> {r['matched_bls_item']}")
    print(f"  {'':30s}     ${r['price_usd']:.2f} {r['unit']}  "
          f"({r['year']}-{r['period']}, score={r['match_score']}, "
          f"conf={r['confidence']})")


def main():
    parser = argparse.ArgumentParser(description="BLS price grounder sanity test")
    parser.add_argument("--csv", default=str(BLS_CSV_PATH),
                        help="Path to bls_prices.csv")
    parser.add_argument("--test", nargs="+",
                        default=["rice", "chicken breast", "quinoa", "eggs",
                                 "egusi", "ground beef", "milk", "salt"],
                        help="Ingredient names to test")
    parser.add_argument("--stats", action="store_true",
                        help="Print loader stats and exit.")
    parser.add_argument("--coverage", action="store_true",
                        help="Print a coverage_report() summary.")
    args = parser.parse_args()

    g = BLSGrounder(csv_path=Path(args.csv))

    if args.stats:
        print(json.dumps(g.stats(), indent=2, default=str))
        return

    print(f"Looking up {len(args.test)} ingredient(s) "
          f"against {g.csv_path}\n")
    for ing in args.test:
        r = g.lookup(ing)
        _print_match(ing, r)

    if args.coverage:
        print("\nCoverage report:")
        report = g.coverage_report(args.test)
        # Don't dump the matched array — it's large; show just the stats.
        compact = {k: v for k, v in report.items()
                   if k not in ("matched", "unmatched")}
        compact["unmatched"] = report["unmatched"]
        print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()
