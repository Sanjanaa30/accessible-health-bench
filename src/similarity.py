"""
src/similarity.py

Phase 5, Track B — Semantic adaptation measurement.

For every prompt with both baseline and constrained variants, computes four
distance signals between the responses, per model:

  1. cosine_distance_full         — Sentence-BERT on full response text
                                    (chunk-and-mean for inputs > 512 tokens)
  2. cosine_distance_ingredients  — Sentence-BERT on ingredient list (joined)
  3. cosine_distance_structural   — Sentence-BERT on a structural digest
                                    (response_type + meal/fitness skeleton)
  4. jaccard_distance_ingredients — set-based, length-invariant; complements (2)

Why four signals?
  - Distance (1) measures overall response change.
  - Distance (2) measures ingredient-level wording change but is sensitive
    to length differences (a baseline of 25 ingredients vs a constrained of
    8 produces low similarity even when overlap is high).
  - Distance (4) corrects for that: pure set overlap, length-invariant,
    using the same normalize_food_name as Phase 4 grounding so "oats"
    and "oat" don't count as different ingredients.
  - Distance (3) measures plan-structure change independent of wording
    (did the meal structure or workout layout actually shift?).

Together they let the paper distinguish:
  - Surface rewording  (high cosine_full, low jaccard, low structural)
  - Genuine ingredient swap  (high cosine_full, high jaccard, low structural)
  - Plan restructure  (high cosine_full, low jaccard, high structural)
  - True adaptation  (all three high)

Output: results/similarity.csv — one row per (prompt_pair_id, provider).

Run from repo root:
    python -m src.similarity                              # full run
    python -m src.similarity --providers openai           # one provider
    python -m src.similarity --pilot 5                    # 5 pair_keys total
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS, ENRICHED_DIR, RESULTS_DIR
from src.grounding.wikidata import normalize_food_name

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_KEYS_PATH = Path("data/embeddings_cache.keys.json")
EMBEDDING_CACHE_VECS_PATH = Path("data/embeddings_cache.vectors.npy")

# all-MiniLM-L6-v2 truncates inputs to 256 word-pieces (~200 words).
# For full response cosine we chunk longer texts into MAX_CHARS_PER_CHUNK
# windows and mean-pool the embeddings — see encode_long_text.
MAX_CHARS_PER_CHUNK = 1000  # ~256 word-pieces of English

PROMPT_ID_RE = re.compile(r"^(.+?)_(base|con)_(\d+)$")


# =============================================================
# Sentence-BERT loader — lazy + cached
# =============================================================
class SentenceBERT:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from e
        print(f"Loading {model_name}...")
        self._model = SentenceTransformer(model_name)
        # Newer sentence-transformers renamed this; fall back for older versions.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim = int(self._model.get_embedding_dimension())
        else:
            self._dim = int(self._model.get_sentence_embedding_dimension())
        self._cache: dict[str, np.ndarray] = {}
        self.model_name = model_name
        self._load_cache()

    @property
    def dim(self) -> int:
        return self._dim

    def _load_cache(self):
        if EMBEDDING_CACHE_KEYS_PATH.exists() and EMBEDDING_CACHE_VECS_PATH.exists():
            try:
                keys = json.loads(EMBEDDING_CACHE_KEYS_PATH.read_text(encoding="utf-8"))
                vectors = np.load(EMBEDDING_CACHE_VECS_PATH)
                if len(keys) == vectors.shape[0]:
                    self._cache = {k: vectors[i] for i, k in enumerate(keys)}
                    print(f"  loaded {len(self._cache)} cached embeddings")
                else:
                    print(f"  cache mismatch (keys {len(keys)} vs vectors "
                          f"{vectors.shape[0]}); starting fresh")
            except Exception as e:  # noqa: BLE001
                print(f"  cache load failed ({e}); starting fresh")
                self._cache = {}

    def save_cache(self):
        if not self._cache:
            return
        EMBEDDING_CACHE_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache.keys())
        vectors = np.stack(list(self._cache.values()))
        EMBEDDING_CACHE_KEYS_PATH.write_text(
            json.dumps(keys), encoding="utf-8"
        )
        np.save(EMBEDDING_CACHE_VECS_PATH, vectors)

    def encode(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        if text in self._cache:
            return self._cache[text]
        vec = self._model.encode(text, convert_to_numpy=True,
                                 show_progress_bar=False)
        self._cache[text] = vec
        return vec

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        # Identify uncached non-empty texts (deduplicated)
        seen: set[str] = set()
        to_encode: list[str] = []
        for t in texts:
            if t and t.strip() and t not in self._cache and t not in seen:
                seen.add(t)
                to_encode.append(t)

        if to_encode:
            vecs = self._model.encode(to_encode, convert_to_numpy=True,
                                      show_progress_bar=False, batch_size=32)
            for t, v in zip(to_encode, vecs):
                self._cache[t] = v

        return [
            self._cache[t] if t and t.strip() else np.zeros(self._dim, dtype=np.float32)
            for t in texts
        ]

    def encode_long_text(self, text: str) -> np.ndarray:
        """
        Embed text that may exceed Sentence-BERT's 256-wordpiece truncation
        cap by splitting on character windows and mean-pooling.
        """
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)
        if len(text) <= MAX_CHARS_PER_CHUNK:
            return self.encode(text)
        # Cache key encodes the chunk-and-mean operation so a re-run hits cache.
        cache_key = f"__chunked__:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        chunks = [
            text[i:i + MAX_CHARS_PER_CHUNK]
            for i in range(0, len(text), MAX_CHARS_PER_CHUNK)
        ]
        chunk_vecs = self.encode_batch(chunks)
        pooled = np.mean(np.stack(chunk_vecs), axis=0)
        self._cache[cache_key] = pooled
        return pooled


# =============================================================
# Distance functions
# =============================================================
def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns 1 - cosine_similarity, clamped to [0, 2]. Empty vec → 1.0."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 1.0
    sim = float(np.dot(v1, v2) / (n1 * n2))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def jaccard_distance(set_a: set, set_b: set) -> float:
    """1 - |A ∩ B| / |A ∪ B|. Both empty → 0.0 (identical, no change)."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 1.0
    return 1.0 - len(set_a & set_b) / len(union)


# =============================================================
# Building the four signals from one enriched record
# =============================================================
def extract_full_text(record: dict) -> str:
    return record.get("response_text", "") or ""


def extract_ingredient_text(record: dict) -> str:
    """Joined ingredient string for Sentence-BERT."""
    extracted = record.get("extracted") or {}
    if not isinstance(extracted, dict):
        return ""
    ingredients = extracted.get("all_ingredients") or []
    if not isinstance(ingredients, list):
        return ""
    return ", ".join(str(i).lower().strip() for i in ingredients if i)


def extract_ingredient_set(record: dict) -> set:
    """
    Normalized ingredient set for Jaccard.
    Uses the same normalize_food_name as Phase 4 grounding so "oats" and
    "oat" map to the same canonical form.
    """
    extracted = record.get("extracted") or {}
    if not isinstance(extracted, dict):
        return set()
    ingredients = extracted.get("all_ingredients") or []
    if not isinstance(ingredients, list):
        return set()
    return {
        normalize_food_name(str(i)) for i in ingredients
        if i and normalize_food_name(str(i))
    }


def extract_structural_digest(record: dict) -> str:
    """
    Short, structure-focused string capturing plan layout independent of
    surface wording. The structural cosine compares these synthetic strings
    — high overlap of tokens like 'meals' / 'horizon' is expected, so the
    score behaves more like fuzzy structural matching than deep semantics.
    """
    extracted = record.get("extracted") or {}
    if not isinstance(extracted, dict):
        return ""

    parts: list[str] = []
    parts.append(f"type={extracted.get('response_type', 'unknown')}")

    routine = extracted.get("routine_structure") or {}
    if isinstance(routine, dict):
        parts.append(f"horizon={routine.get('time_horizon', 'unknown')}")
        parts.append(f"structured={routine.get('is_structured_schedule', False)}")

    meals = extracted.get("meal_components") or []
    if isinstance(meals, list):
        # str(... or "?") coerces None / "" / missing keys all to "?"
        meal_types = [str(m.get("meal_type") or "?") for m in meals
                      if isinstance(m, dict)]
        if meal_types:
            parts.append(f"meals=[{','.join(meal_types)}]")

    fitness = extracted.get("fitness_components") or []
    if isinstance(fitness, list):
        activity_types = [str(a.get("activity_type") or "?") for a in fitness
                          if isinstance(a, dict)]
        if activity_types:
            parts.append(f"fitness=[{','.join(activity_types)}]")

    feas = extracted.get("feasibility_signals") or {}
    if isinstance(feas, dict):
        kitchen = feas.get("kitchen_access_assumption", "unknown")
        parts.append(f"kitchen={kitchen}")
        equip = feas.get("fitness_equipment_required") or []
        if isinstance(equip, list) and equip:
            parts.append(f"equip=[{','.join(str(e) for e in equip[:3])}]")

    return " | ".join(parts)


# =============================================================
# Pair-finding
# =============================================================
def build_prompt_pairs(
    enriched_root: Path,
    providers: list[str],
) -> list[tuple]:
    """
    For each (provider, base_id) where both baseline and constrained exist,
    yield (provider, pair_key, baseline_record, constrained_record).

    pair_key is the multi-word category prefix + numeric index, e.g.
    "fin_04". Uses regex on the prompt-id stem so multi-word categories
    don't break the parser.

    Records flagged with grounding._skipped are excluded.
    """
    pairs = []
    for provider in providers:
        provider_dir = enriched_root / provider
        if not provider_dir.exists():
            continue

        baseline_files: dict[str, Path] = {}
        constrained_files: dict[str, Path] = {}

        for f in provider_dir.glob("*.json"):
            if f.name.startswith("_"):
                continue
            m = PROMPT_ID_RE.match(f.stem)
            if m is None:
                continue
            category, variant_marker, idx = m.groups()
            pair_key = f"{category}_{idx}"
            if variant_marker == "base":
                baseline_files[pair_key] = f
            else:  # "con"
                constrained_files[pair_key] = f

        for key, base_path in baseline_files.items():
            con_path = constrained_files.get(key)
            if con_path is None:
                continue
            try:
                base_rec = json.loads(base_path.read_text(encoding="utf-8"))
                con_rec = json.loads(con_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            # Skip pairs where either side was marked as a Phase 3 error
            base_g = base_rec.get("grounding") or {}
            con_g = con_rec.get("grounding") or {}
            if "_skipped" in base_g or "_skipped" in con_g:
                continue

            pairs.append((provider, key, base_rec, con_rec))

    return pairs


# =============================================================
# Per-pair computation
# =============================================================
def compute_pair_distances(
    base_rec: dict,
    con_rec: dict,
    embedder: SentenceBERT,
) -> dict:
    base_full = extract_full_text(base_rec)
    con_full = extract_full_text(con_rec)
    base_ing_text = extract_ingredient_text(base_rec)
    con_ing_text = extract_ingredient_text(con_rec)
    base_struct = extract_structural_digest(base_rec)
    con_struct = extract_structural_digest(con_rec)
    base_ing_set = extract_ingredient_set(base_rec)
    con_ing_set = extract_ingredient_set(con_rec)

    # Long-form: chunk-and-mean to avoid silent truncation of full responses.
    full_base_emb = embedder.encode_long_text(base_full)
    full_con_emb = embedder.encode_long_text(con_full)

    # Short-form: ingredient text + structural digest fit in one window.
    short_embs = embedder.encode_batch([
        base_ing_text, con_ing_text, base_struct, con_struct,
    ])

    return {
        "cosine_distance_full":         cosine_distance(full_base_emb, full_con_emb),
        "cosine_distance_ingredients":  cosine_distance(short_embs[0], short_embs[1]),
        "cosine_distance_structural":   cosine_distance(short_embs[2], short_embs[3]),
        "jaccard_distance_ingredients": jaccard_distance(base_ing_set, con_ing_set),
        "baseline_ingredient_count":    len(base_ing_set),
        "constrained_ingredient_count": len(con_ing_set),
        "baseline_response_chars":      len(base_full),
        "constrained_response_chars":   len(con_full),
        "baseline_category":            base_rec.get("category"),
        "constrained_category_type":    con_rec.get("category_type") or "",
    }


# =============================================================
# Main
# =============================================================
def run(
    pilot: Optional[int] = None,
    providers: Optional[list[str]] = None,
    out_path: Optional[Path] = None,
):
    if providers is None:
        providers = list(MODELS.keys())
    if out_path is None:
        out_path = Path(RESULTS_DIR) / "similarity.csv"

    enriched_root = Path(ENRICHED_DIR)
    if not enriched_root.exists():
        print(f"No enriched dir at {enriched_root}. Run Phase 4 first.")
        return

    print("Building prompt pairs...")
    pairs = build_prompt_pairs(enriched_root, providers)
    print(f"  found {len(pairs)} (provider, prompt_pair) entries "
          f"across {len(providers)} provider(s)")

    if pilot is not None:
        # Restrict to the first N pair_keys (alphabetical), keeping all
        # providers per key — so a 5-pair pilot covers 5*N providers responses.
        all_keys = sorted({p[1] for p in pairs})
        keep_keys = set(all_keys[:pilot])
        pairs = [p for p in pairs if p[1] in keep_keys]
        print(f"  PILOT MODE: {len(keep_keys)} pair_keys × providers "
              f"= {len(pairs)} entries")

    if not pairs:
        print("No pairs to process. Exiting.")
        return

    embedder = SentenceBERT()

    rows = []
    for provider, pair_key, base_rec, con_rec in tqdm(pairs, desc="Computing"):
        d = compute_pair_distances(base_rec, con_rec, embedder)
        rows.append({
            "provider":            provider,
            "pair_key":            pair_key,
            "category":            d["baseline_category"],
            "category_type":       d["constrained_category_type"],
            "baseline_id":         base_rec.get("prompt_id"),
            "constrained_id":      con_rec.get("prompt_id"),
            "cosine_full":         round(d["cosine_distance_full"], 4),
            "cosine_ingredients":  round(d["cosine_distance_ingredients"], 4),
            "cosine_structural":   round(d["cosine_distance_structural"], 4),
            "jaccard_ingredients": round(d["jaccard_distance_ingredients"], 4),
            "baseline_ing_count":     d["baseline_ingredient_count"],
            "constrained_ing_count":  d["constrained_ingredient_count"],
            "baseline_chars":         d["baseline_response_chars"],
            "constrained_chars":      d["constrained_response_chars"],
            "embedding_model":        embedder.model_name,
        })

    embedder.save_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Console summary
    print("\n" + "=" * 70)
    print("SIMILARITY COMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Total pairs analyzed: {len(rows)}")
    print(f"Output: {out_path}")
    print(f"Embedding model: {embedder.model_name} (dim={embedder.dim})")

    if not rows:
        return

    grouped: dict[tuple, dict[str, list]] = defaultdict(
        lambda: {"cf": [], "ci": [], "cs": [], "j": []}
    )
    for r in rows:
        key = (r["provider"], r["category"])
        grouped[key]["cf"].append(r["cosine_full"])
        grouped[key]["ci"].append(r["cosine_ingredients"])
        grouped[key]["cs"].append(r["cosine_structural"])
        grouped[key]["j"].append(r["jaccard_ingredients"])

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print(f"\nMean distances by provider × category (higher = more change):")
    print(f"  {'provider':10} {'category':10} {'n':>4} "
          f"{'full':>8} {'ingred':>8} {'struct':>8} {'jacc':>8}")
    for (provider, category), bucket in sorted(grouped.items()):
        print(f"  {provider:10} {category:10} {len(bucket['cf']):4d} "
              f"{_mean(bucket['cf']):8.3f} {_mean(bucket['ci']):8.3f} "
              f"{_mean(bucket['cs']):8.3f} {_mean(bucket['j']):8.3f}")

    print("\nQuick interpretation guide:")
    print("  full < 0.10  → response is nearly identical to baseline (rigid)")
    print("  full > 0.30  → response substantially changed")
    print("  jacc > full  → ingredient set turned over more than wording suggests")
    print("  struct high  → plan structure (meal/fitness layout) shifted")


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Phase 5 Track B similarity signals",
    )
    parser.add_argument(
        "--pilot", type=int, default=None,
        help="Only process first N pair_keys total (across all providers).",
    )
    parser.add_argument(
        "--providers", nargs="+", default=None,
        choices=["openai", "anthropic", "deepseek", "groq"],
    )
    parser.add_argument(
        "--out", default=None,
        help="Output CSV path (default: results/similarity.csv).",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None
    run(pilot=args.pilot, providers=args.providers, out_path=out_path)
