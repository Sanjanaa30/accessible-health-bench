"""
src/grounding/wikidata.py

Phase 4a — Cuisine grounding via Wikidata SPARQL with LLM fallback.

For each ingredient or dish, query Wikidata for cuisine origin tags. Wikidata
hits are cached forever in data/wikidata_cache.sqlite. Misses are accumulated
across a session and resolved in a SINGLE batched LLM call at the end so the
fallback cost stays low.

Lookup strategy (for one food name):
  1. Cache check — return immediately on hit.
  2. EntitySearch via MWAPI to find candidate Wikidata entities by label OR
     alias (so "egusi" resolves even though its primary label differs).
  3. For each candidate, attempt three cuisine-resolution paths:
       (a) direct P2012 ("cuisine") — typically set on dishes only
       (b) walk P31/P279* subclass chain and look for P2012 on parent class
       (c) P495 ("country of origin") as a confidence-medium fallback
  4. If still no cuisine, queue the food for the LLM batch fallback.

Usage:
    from src.grounding.wikidata import WikidataGrounder
    g = WikidataGrounder()
    r = g.lookup("rice")
    # -> {"cuisines": [...], "countries": [...], "source": "wikidata", ...}

    # For a list of foods + LLM fallback in one batched call:
    results = g.lookup_batch(["rice", "egusi", "kichari"])
    fallback_results = g.resolve_misses()

CLI smoke test:
    python -m src.grounding.wikidata --test rice quinoa egusi salmon dal
    python -m src.grounding.wikidata --stats
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import WIKIDATA_CACHE_PATH, EXTRACTION_MODEL

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "AccessibleHealthBench/1.0 (research; contact via repo)"
SPARQL_TIMEOUT_S = 25
SPARQL_RETRY_ATTEMPTS = 3
SLEEP_BETWEEN_QUERIES_S = 0.5  # be a polite SPARQL client

# Wikidata IDs we reuse
WD_FOOD_CLASS = "Q2095"        # "food"
WD_CUISINE_CLASS = "Q1778821"  # "cuisine"

# Words ending in -s that are SINGULAR — don't strip the trailing s.
SINGULAR_S_EXCEPTIONS = {
    "hummus", "couscous", "asparagus", "hibiscus", "molasses",
    "lentils", "chickpeas", "oats", "grits", "greens",
    "swiss", "brussels", "berries",  # already plural-ish but treated as units
    "bass", "albacore", "salmon",
    "rice", "stew", "bread", "fish",
    "ramen", "miso", "kimchi",
    "boneless",
}

# Map Wikidata cuisine labels to our canonical snake_case vocab.
# Anything not in this map gets auto-normalized (lowercase, strip "_cuisine"
# suffix) but flagged as low-confidence.
CUISINE_NORMALIZER = {
    "American cuisine":               "american",
    "Andean cuisine":                 "andean",
    "Asian cuisine":                  "asian",
    "Bengali cuisine":                "bengali",
    "Cajun cuisine":                  "cajun",
    "Cantonese cuisine":              "cantonese",
    "Caribbean cuisine":              "caribbean",
    "Chinese cuisine":                "chinese",
    "East Asian cuisine":             "east_asian",
    "Ethiopian cuisine":              "ethiopian",
    "European cuisine":               "european",
    "French cuisine":                 "french",
    "Greek cuisine":                  "greek",
    "Gujarati cuisine":               "gujarati",
    "Indian cuisine":                 "indian",
    "Indigenous cuisine":             "indigenous",
    "Italian cuisine":                "italian",
    "Japanese cuisine":               "japanese",
    "Korean cuisine":                 "korean",
    "Latin American cuisine":         "latin_american",
    "Mediterranean cuisine":          "mediterranean",
    "Mexican cuisine":                "mexican",
    "Middle Eastern cuisine":         "middle_eastern",
    "Native American cuisine":        "indigenous_north_american",
    "Nigerian cuisine":               "west_african_nigerian",
    "North African cuisine":          "north_african",
    "North Indian cuisine":           "north_indian",
    "Pakistani cuisine":              "pakistani",
    "Persian cuisine":                "persian",
    "Punjabi cuisine":                "punjabi",
    "South Asian cuisine":            "south_asian",
    "South Indian cuisine":           "south_indian",
    "South American cuisine":         "south_american",
    "Southeast Asian cuisine":        "southeast_asian",
    "Spanish cuisine":                "spanish",
    "Sub-Saharan African cuisine":    "sub_saharan_african",
    "Thai cuisine":                   "thai",
    "Turkish cuisine":                "turkish",
    "Vietnamese cuisine":             "vietnamese",
    "West African cuisine":           "west_african",
    "Western cuisine":                "western",
}


# =============================================================
# Normalization
# =============================================================
def normalize_food_name(name: str) -> str:
    """Lowercase, strip surrounding punctuation, careful singularization."""
    n = name.strip().lower()
    n = n.strip(".,;:()[]{}\"'")
    if not n:
        return n

    # Don't singularize known-singular words ending in s
    if n in SINGULAR_S_EXCEPTIONS:
        return n

    if n.endswith("ies") and len(n) > 4:
        n = n[:-3] + "y"        # berries -> berry
    elif n.endswith("oes") and len(n) > 4:
        n = n[:-2]              # tomatoes -> tomato
    elif n.endswith("ses") and len(n) > 4:
        n = n[:-2]              # diseases -> disease
    elif n.endswith("ss"):
        pass                    # mass, pass, glass — already singular
    elif n.endswith("us") and len(n) > 3:
        pass                    # asparagus, hibiscus, focus — leave alone
    elif n.endswith("s") and len(n) > 3:
        n = n[:-1]              # general plural: oats -> oat, beans -> bean
    return n


def _canonicalize_cuisine_label(label: str) -> str:
    """
    Map a raw Wikidata cuisine label to our canonical snake_case form.
    Falls back to a defensive auto-snake-case, stripping trailing 'cuisine'.
    """
    if label in CUISINE_NORMALIZER:
        return CUISINE_NORMALIZER[label]
    # Defensive fallback for unknown cuisines.
    s = label.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_cuisine$", "", s)  # "Burmese cuisine" -> "burmese"
    return s


# =============================================================
# SQLite cache
# =============================================================
class WikidataCache:
    """
    Two namespaces inside one table:
      - 'food:<name>'  for direct Wikidata lookups
      - 'llm:<name>'   for LLM-fallback results
    Errors are NOT persisted — only successful resolutions, so a transient
    network failure doesn't poison the cache forever.
    """

    def __init__(self, path: str = WIKIDATA_CACHE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wikidata_cache (
                key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT result_json FROM wikidata_cache WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, result: dict):
        self.conn.execute(
            "INSERT OR REPLACE INTO wikidata_cache (key, result_json, created_at) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(result), datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def stats(self) -> dict:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM wikidata_cache"
        ).fetchone()[0]
        food = self.conn.execute(
            "SELECT COUNT(*) FROM wikidata_cache WHERE key LIKE 'food:%'"
        ).fetchone()[0]
        llm = self.conn.execute(
            "SELECT COUNT(*) FROM wikidata_cache WHERE key LIKE 'llm:%'"
        ).fetchone()[0]
        return {"total": total, "food_entries": food, "llm_fallback_entries": llm}

    def close(self):
        self.conn.close()


# =============================================================
# SPARQL query construction
# =============================================================
def _build_sparql_query(food_name: str) -> str:
    """
    Multi-path cuisine resolution for one food name.

    Path A: EntitySearch -> entity has direct P2012 (dish-style)
    Path B: EntitySearch -> entity's superclass (P31/P279*) has P2012
    Path C: EntitySearch -> entity has P495 (country of origin) as fallback

    All three paths are unioned so a single query returns whichever match
    Wikidata can offer. Filtering to the "food" superclass (Q2095) keeps
    homonyms like "rice" (the surname) out of the result.
    """
    safe = food_name.replace('"', '\\"').replace("\\", "\\\\")
    return f"""
    SELECT DISTINCT ?item ?itemLabel ?cuisineLabel ?countryLabel ?path WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:api "EntitySearch" .
        bd:serviceParam wikibase:endpoint "www.wikidata.org" .
        bd:serviceParam mwapi:search "{safe}" .
        bd:serviceParam mwapi:language "en" .
        bd:serviceParam mwapi:limit "10" .
        ?item wikibase:apiOutputItem mwapi:item .
      }}
      ?item wdt:P31?/wdt:P279* wd:{WD_FOOD_CLASS} .

      {{
        ?item wdt:P2012 ?cuisine .
        BIND("direct" AS ?path)
      }} UNION {{
        ?item wdt:P31?/wdt:P279+ ?parent .
        ?parent wdt:P2012 ?cuisine .
        BIND("subclass" AS ?path)
      }} UNION {{
        ?item wdt:P495 ?country .
        BIND("country" AS ?path)
      }}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 100
    """


# =============================================================
# SPARQL client with retry
# =============================================================
def _query_wikidata(food_name: str) -> dict:
    """Return a structured result dict; on transient failure raises so the
    caller can decide whether to cache."""
    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError as e:
        raise ImportError(
            "SPARQLWrapper required: pip install SPARQLWrapper"
        ) from e

    sparql = SPARQLWrapper(SPARQL_ENDPOINT, agent=USER_AGENT)
    sparql.setQuery(_build_sparql_query(food_name))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(SPARQL_TIMEOUT_S)

    last_exc = None
    for attempt in range(1, SPARQL_RETRY_ATTEMPTS + 1):
        try:
            results = sparql.query().convert()
            break
        except Exception as e:  # noqa: BLE001 — SPARQLWrapper raises many types
            last_exc = e
            if attempt < SPARQL_RETRY_ATTEMPTS:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Wikidata query failed: {e}") from e

    cuisines_raw: set[str] = set()
    countries_raw: set[str] = set()
    paths: set[str] = set()

    for row in results.get("results", {}).get("bindings", []):
        if "cuisineLabel" in row:
            cuisines_raw.add(row["cuisineLabel"]["value"])
        if "countryLabel" in row:
            countries_raw.add(row["countryLabel"]["value"])
        if "path" in row:
            paths.add(row["path"]["value"])

    cuisines_norm = sorted({_canonicalize_cuisine_label(c) for c in cuisines_raw})

    if cuisines_norm:
        confidence = "high" if "direct" in paths else "medium"
        source = "wikidata"
    elif countries_raw:
        confidence = "medium"
        source = "wikidata_country_only"
    else:
        confidence = "none"
        source = "wikidata_miss"

    return {
        "food_name": food_name,
        "cuisines": cuisines_norm,
        "countries": sorted(countries_raw),
        "source": source,
        "confidence": confidence,
        "resolved_via": sorted(paths),
    }


# =============================================================
# LLM fallback (one batched call per session)
# =============================================================
def _llm_fallback_lookup(food_names: list[str], llm_client) -> dict:
    """Single batched LLM call for ungrounded foods."""
    if not food_names:
        return {}

    allowed = sorted(set(CUISINE_NORMALIZER.values()) | {"global", "unknown"})
    food_list = "\n".join(f"- {f}" for f in sorted(food_names))

    prompt = (
        "For each food/dish below, list the cuisine(s) it is most commonly "
        "associated with.\n\n"
        "Use ONLY snake_case identifiers from this allowed vocabulary:\n"
        + ", ".join(allowed)
        + "\n\nReturn STRICT JSON only — no markdown, no commentary — mapping each "
          "food name to a list of cuisine identifiers. Example:\n"
          '{"egusi": ["west_african"], "kimchi": ["korean", "east_asian"], '
          '"oat": ["global"]}\n\n'
          'If a food is unclear, use ["unknown"]. If it is global '
          '(eaten everywhere with no specific origin), use ["global"].\n\n'
          "Foods:\n" + food_list
    )

    try:
        response = llm_client.generate(
            provider="openai",
            model=EXTRACTION_MODEL,
            prompt=prompt,
            params={"temperature": 0.0, "max_tokens": 2000},
        )
        text = response["text"].strip()
        # Strip markdown fences defensively
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0].rstrip()

        parsed = json.loads(text)
        return {
            name: {
                "food_name": name,
                "cuisines": [c for c in parsed.get(name, ["unknown"]) if c in allowed],
                "countries": [],
                "source": "llm_fallback",
                "confidence": "low",
            }
            for name in food_names
        }
    except Exception as e:  # noqa: BLE001
        return {
            name: {
                "food_name": name,
                "cuisines": [],
                "countries": [],
                "source": "ungrounded",
                "confidence": "none",
                "error": str(e)[:200],
            }
            for name in food_names
        }


# =============================================================
# Main grounder
# =============================================================
class WikidataGrounder:
    """Wikidata-first cuisine grounder with hybrid LLM fallback."""

    def __init__(
        self,
        cache_path: str = WIKIDATA_CACHE_PATH,
        sleep_between_queries: float = SLEEP_BETWEEN_QUERIES_S,
    ):
        self.cache = WikidataCache(cache_path)
        self.sleep = sleep_between_queries
        self._llm_client = None
        self._pending_misses: set[str] = set()

    # ------------------------------------------------------- helpers
    def _get_llm(self):
        if self._llm_client is None:
            from src.clients.unified_llm import UnifiedLLM
            self._llm_client = UnifiedLLM()
        return self._llm_client

    @staticmethod
    def _food_key(name: str) -> str:
        return f"food:{name}"

    @staticmethod
    def _llm_key(name: str) -> str:
        return f"llm:{name}"

    # ------------------------------------------------------- public API
    def lookup(self, food_name: str) -> dict:
        """Look up a single food. Returns cached or live result."""
        normalized = normalize_food_name(food_name)
        if not normalized:
            return {
                "food_name": food_name,
                "cuisines": [],
                "countries": [],
                "source": "empty_input",
                "confidence": "none",
            }

        cached = self.cache.get(self._food_key(normalized))
        if cached is not None:
            return cached

        try:
            result = _query_wikidata(normalized)
        except RuntimeError as e:
            # Don't cache transient errors — let the next session retry.
            return {
                "food_name": normalized,
                "cuisines": [],
                "countries": [],
                "source": "wikidata_error",
                "confidence": "none",
                "error": str(e)[:200],
            }

        # Only cache successful resolutions (including legitimate misses,
        # which ARE deterministic — Wikidata genuinely has no entry).
        self.cache.set(self._food_key(normalized), result)
        if self.sleep > 0:
            time.sleep(self.sleep)
        return result

    def lookup_batch(self, food_names: list[str]) -> list[dict]:
        """Look up many foods, accumulating misses for one LLM-batch call."""
        results: list[dict] = []
        for name in food_names:
            r = self.lookup(name)
            results.append(r)
            if not r.get("cuisines") and r.get("source") in {
                "wikidata_miss", "wikidata_country_only", "wikidata_error",
            }:
                self._pending_misses.add(normalize_food_name(name))
        return results

    def resolve_misses(self) -> dict:
        """Run the LLM fallback once for every accumulated miss."""
        if not self._pending_misses:
            return {}

        # Reuse cached LLM-fallback entries from prior sessions
        cached: dict[str, dict] = {}
        unresolved: list[str] = []
        for name in self._pending_misses:
            existing = self.cache.get(self._llm_key(name))
            if existing is not None:
                cached[name] = existing
            else:
                unresolved.append(name)

        new_results: dict[str, dict] = {}
        if unresolved:
            print(f"  LLM fallback for {len(unresolved)} ungrounded foods...")
            new_results = _llm_fallback_lookup(unresolved, self._get_llm())
            for name, result in new_results.items():
                # Only cache successful llm_fallback results, not ungrounded ones
                if result.get("source") == "llm_fallback":
                    self.cache.set(self._llm_key(name), result)

        self._pending_misses.clear()
        return {**cached, **new_results}

    def cache_stats(self) -> dict:
        return self.cache.stats()

    def close(self):
        self.cache.close()


# =============================================================
# CLI
# =============================================================
def _print_result(food: str, r: dict):
    print(f"  {food}:")
    print(f"    cuisines:   {r.get('cuisines', [])}")
    print(f"    countries:  {r.get('countries', [])}")
    print(f"    source:     {r.get('source', 'unknown')}")
    print(f"    confidence: {r.get('confidence', 'unknown')}")
    if r.get("resolved_via"):
        print(f"    paths:      {r['resolved_via']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Wikidata grounding sanity test")
    parser.add_argument(
        "--test", nargs="+",
        default=["rice", "quinoa", "egusi", "salmon", "dal", "kimchi", "kichari"],
        help="Food names to test",
    )
    parser.add_argument("--stats", action="store_true",
                        help="Print cache statistics and exit.")
    args = parser.parse_args()

    g = WikidataGrounder()

    if args.stats:
        print("Wikidata cache stats:", g.cache_stats())
        g.close()
        return

    print(f"Looking up {len(args.test)} foods...\n")
    for food in args.test:
        result = g.lookup(food)
        _print_result(food, result)
        if not result.get("cuisines"):
            g._pending_misses.add(normalize_food_name(food))

    misses = g.resolve_misses()
    if misses:
        print(f"\nLLM fallback resolved {len(misses)} food(s):")
        for name, r in misses.items():
            print(f"  {name}: {r.get('cuisines', [])}  ({r.get('source')})")

    print("\nFinal cache stats:", g.cache_stats())
    g.close()


if __name__ == "__main__":
    main()
