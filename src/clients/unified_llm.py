"""
src/clients/unified_llm.py

Unified interface for OpenAI, Anthropic, DeepSeek, and Groq with SQLite-backed
caching. Every call is hashed and cached — re-running a script never re-pays.

"""

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()

# Import from project config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS, GENERATION_PARAMS, LLM_CACHE_PATH


# ============================================================
# SQLite cache — keyed on hash(provider, model, prompt, params)
# ============================================================
class LLMCache:
    """SQLite cache for LLM responses. Keyed on the hash of all call inputs."""

    def __init__(self, path: str = LLM_CACHE_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                provider TEXT,
                model TEXT,
                prompt TEXT,
                response_json TEXT,
                created_at TEXT
            )
            """
        )
        self.conn.commit()

    def _make_key(self, provider: str, model: str, prompt: str, params: dict) -> str:
        # Sort params for deterministic hashing
        params_str = json.dumps(params, sort_keys=True)
        raw = f"{provider}||{model}||{prompt}||{params_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, provider: str, model: str, prompt: str, params: dict) -> Optional[dict]:
        key = self._make_key(provider, model, prompt, params)
        cursor = self.conn.execute(
            "SELECT response_json FROM llm_cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, provider: str, model: str, prompt: str, params: dict, response: dict):
        key = self._make_key(provider, model, prompt, params)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (key, provider, model, prompt, response_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                provider,
                model,
                prompt,
                json.dumps(response),
                datetime.now().isoformat(),
            ),
        )
        self.conn.commit()

    def stats(self) -> dict:
        """Return cache statistics — useful for monitoring."""
        cursor = self.conn.execute(
            "SELECT provider, COUNT(*) FROM llm_cache GROUP BY provider"
        )
        return {row[0]: row[1] for row in cursor.fetchall()}


# ============================================================
# Unified LLM client
# ============================================================
class UnifiedLLM:
    """
    Single entry point for all 4 providers. Handles caching automatically.

    Lazy-imports each provider SDK so missing packages don't break unrelated calls.
    """

    def __init__(self, cache_path: str = LLM_CACHE_PATH):
        self.cache = LLMCache(cache_path)
        self._openai_client = None
        self._anthropic_client = None
        self._deepseek_client = None
        self._groq_client = None

    # -------------------------------------------------------
    # Lazy client initialization (so missing keys don't crash unrelated calls)
    # -------------------------------------------------------
    def _get_openai(self):
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in .env")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _get_anthropic(self):
        if self._anthropic_client is None:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_deepseek(self):
        """
        DeepSeek uses OpenAI-compatible API — same SDK, different base_url.
        Endpoint: https://api.deepseek.com
        """
        if self._deepseek_client is None:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set in .env")
            self._deepseek_client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        return self._deepseek_client

    def _get_groq(self):
        if self._groq_client is None:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set in .env")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    # -------------------------------------------------------
    # Provider-specific call methods
    # -------------------------------------------------------
    def _call_openai(self, model: str, prompt: str, params: dict) -> dict:
        client = self._get_openai()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def _call_anthropic(self, model: str, prompt: str, params: dict) -> dict:
        client = self._get_anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def _call_deepseek(self, model: str, prompt: str, params: dict) -> dict:
        """
        DeepSeek V4 Flash uses OpenAI-compatible chat completions.
        Thinking mode is disabled to keep parity with other providers.
        """
        client = self._get_deepseek()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            # Disable thinking mode for fair comparison with other models
            extra_body={"thinking": {"type": "disabled"}},
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    def _call_groq(self, model: str, prompt: str, params: dict) -> dict:
        client = self._get_groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    # -------------------------------------------------------
    # Main public method
    # -------------------------------------------------------
    def generate(
        self,
        provider: str,
        prompt: str,
        model: Optional[str] = None,
        params: Optional[dict] = None,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> dict:
        """
        Generate a response from the given provider.

        Args:
            provider: one of "openai", "anthropic", "deepseek", "groq"
            prompt: the user prompt
            model: override the default model from config (optional)
            params: override default generation params (optional)
            retries: number of retries on failure
            retry_delay: seconds to wait between retries

        Returns:
            dict with text, provider, model, from_cache, input_tokens,
            output_tokens, timestamp
        """
        if provider not in MODELS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Must be one of: {list(MODELS.keys())}"
            )

        model = model or MODELS[provider]
        params = params or GENERATION_PARAMS

        # Check cache first
        cached = self.cache.get(provider, model, prompt, params)
        if cached is not None:
            cached["from_cache"] = True
            return cached

        # Cache miss — call the provider with retries
        call_methods = {
            "openai":    self._call_openai,
            "anthropic": self._call_anthropic,
            "deepseek":  self._call_deepseek,
            "groq":      self._call_groq,
        }
        call_fn = call_methods[provider]

        last_error = None
        for attempt in range(retries):
            try:
                result = call_fn(model, prompt, params)
                response = {
                    "text": result["text"],
                    "provider": provider,
                    "model": model,
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "timestamp": datetime.now().isoformat(),
                    "from_cache": False,
                }
                self.cache.set(provider, model, prompt, params, response)
                return response
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    print(f"  retry {attempt + 1}/{retries} for {provider}: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # backoff
                continue

        raise RuntimeError(
            f"Failed to generate from {provider} after {retries} retries. "
            f"Last error: {last_error}"
        )


# ============================================================
# Quick smoke test — run this file directly to verify all 4 providers
# ============================================================
if __name__ == "__main__":
    print("Running unified LLM smoke test...\n")

    client = UnifiedLLM()
    test_prompt = "In one sentence, what is a healthy breakfast?"

    for provider in ["openai", "anthropic", "deepseek", "groq"]:
        print(f"--- {provider} ({MODELS[provider]}) ---")
        try:
            response = client.generate(provider=provider, prompt=test_prompt)
            print(f"  Response: {response['text'][:120]}...")
            print(f"  Tokens: in={response['input_tokens']}, "
                  f"out={response['output_tokens']}")
            print(f"  From cache: {response['from_cache']}")
        except Exception as e:
            print(f"  FAILED: {e}")
        print()

    print(f"Cache stats: {client.cache.stats()}")