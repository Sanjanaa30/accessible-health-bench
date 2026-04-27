"""
src/config.py

Single source of truth for model IDs and generation parameters.
Every script in the pipeline reads from here.
"""

# ============================================================
# Model IDs — current as of April 2026
# ============================================================

MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}

# Display names for figures, papers, dashboards
MODEL_DISPLAY_NAMES = {
    "openai": "GPT-4o-mini",
    "anthropic": "Claude Haiku 4.5",
    "gemini": "Gemini 2.5 Flash",
    "groq": "Llama 3.3 70B",
}

# ============================================================
# Generation parameters — same across all 4 providers
# ============================================================
GENERATION_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1500,
}

# ============================================================
# Cheap model for extraction + judging (always GPT-4o-mini)
# ============================================================
EXTRACTION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# ============================================================
# Paths
# ============================================================
LLM_CACHE_PATH = "data/llm_cache.sqlite"
WIKIDATA_CACHE_PATH = "data/wikidata_cache.sqlite"
PROMPTS_PATH = "data/prompts.jsonl"
RESPONSES_DIR = "data/responses"
EXTRACTIONS_DIR = "data/extractions"
ENRICHED_DIR = "data/enriched"
RESULTS_DIR = "results"