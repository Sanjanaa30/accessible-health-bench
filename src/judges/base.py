"""
src/judges/base.py

Shared infrastructure for the three G-Eval-style judges (Affordability,
Cultural, Feasibility). Loads the rubric from prompts/judge_*.txt, fills
in the four placeholders the rubric declares, calls GPT-4o-mini via the
cached unified client, parses the strict JSON the rubric demands, and
returns a normalized JudgeResult.

The DAGMetric Constraint Adherence judge has a structurally different
output (per-branch verdicts), but reuses build_judge_input from this
module — see adherence.py.

Single source of truth: the rubric file. The rubric's INPUT block defines
which placeholders get substituted, the rubric's OUTPUT block defines the
expected JSON schema. This module never appends its own scaffolding.
"""

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.clients.unified_llm import UnifiedLLM
from src.config import JUDGE_MODEL


@dataclass
class JudgeResult:
    """Normalized output schema for any G-Eval-style judge."""
    judge_name: str
    score: Optional[int]               # 1-5 or None if parse failed
    reasoning: str
    details: dict = field(default_factory=dict)   # full parsed JSON object
    raw_response: str = ""
    parse_error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    from_cache: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================
# Prompt building — fills the rubric's own placeholders
# =============================================================
def build_judge_input(rubric: str, record: dict) -> str:
    """
    Format the rubric template by filling its four placeholders:
      {prompt_text}, {stated_constraints_json}, {response_text},
      {grounding_block_json}

    The rubric file is the complete prompt — we do NOT append our own
    INPUT/OUTPUT sections. Whatever the rubric specifies is what GPT
    sees verbatim.
    """
    prompt_text = record.get("prompt_text", "") or ""
    response_text = record.get("response_text", "") or ""
    stated = record.get("stated_constraints") or {}
    grounding = record.get("grounding") or {}

    # We pass the full grounding block as JSON so the rubric can refer to
    # any field. Indented for readability in the prompt.
    grounding_json = json.dumps(grounding, indent=2, ensure_ascii=False, default=str)
    stated_json = json.dumps(stated, indent=2, ensure_ascii=False, default=str)

    try:
        return rubric.format(
            prompt_text=prompt_text,
            stated_constraints_json=stated_json,
            response_text=response_text,
            grounding_block_json=grounding_json,
        )
    except KeyError as e:
        raise ValueError(
            f"Rubric template references unknown placeholder {e}. "
            "Allowed placeholders: prompt_text, stated_constraints_json, "
            "response_text, grounding_block_json. "
            "Did you forget to double a literal '{' / '}' in the rubric?"
        ) from e


# =============================================================
# Result parsing
# =============================================================
def _strip_markdown_fences(text: str) -> str:
    """Drop leading ```...``` fences if the model added them."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (and optional 'json' marker)
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].rstrip()
    return text


def _extract_json_object(text: str) -> Optional[str]:
    """
    Find the outermost balanced JSON object in `text`. Returns the substring
    or None if no balanced object is found. Stack-based: avoids the greedy
    `\\{.*\\}` regex pitfall.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_judge_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Parse a judge's raw output as JSON. Returns (parsed_dict, parse_error).
    Strips markdown fences, extracts outermost balanced object.
    """
    if not text:
        return None, "empty response"

    cleaned = _strip_markdown_fences(text)

    # Try direct parse first
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    # Fall back: locate outermost balanced object in case the model
    # surrounded the JSON with prose.
    candidate = _extract_json_object(cleaned)
    if candidate is None:
        return None, "no balanced JSON object found in response"

    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"


def _coerce_score(raw_score) -> Optional[int]:
    """Coerce a parsed score field to a clamped integer 1-5, or None."""
    if raw_score is None:
        return None
    try:
        s = int(round(float(raw_score)))
    except (TypeError, ValueError):
        return None
    if 1 <= s <= 5:
        return s
    return None


# =============================================================
# Base judge runner
# =============================================================
class BaseGEvalJudge:
    """
    Base class for G-Eval-style 1-5 judges. Subclasses override:
      - judge_name (str)
      - rubric_path (str): path to prompts/judge_*.txt

    The rubric file is the single source of truth for both input
    formatting and expected output schema.
    """

    judge_name: str = "base"
    rubric_path: str = ""
    max_tokens: int = 1000

    def __init__(self, client: Optional[UnifiedLLM] = None):
        self.client = client or UnifiedLLM()
        self._rubric_cache: Optional[str] = None

    def _load_rubric(self) -> str:
        if self._rubric_cache is not None:
            return self._rubric_cache
        path = Path(self.rubric_path)
        if not path.exists():
            raise FileNotFoundError(f"Rubric not found: {path}")
        self._rubric_cache = path.read_text(encoding="utf-8")
        return self._rubric_cache

    def evaluate(self, record: dict) -> JudgeResult:
        """Run this judge on one enriched record, return JudgeResult."""
        rubric = self._load_rubric()
        full_prompt = build_judge_input(rubric, record)

        response = self.client.generate(
            provider="openai",
            prompt=full_prompt,
            model=JUDGE_MODEL,
            params={"temperature": 0.0, "max_tokens": self.max_tokens},
        )

        raw = response["text"]
        parsed, parse_err = parse_judge_json(raw)

        if parsed is None:
            return JudgeResult(
                judge_name=self.judge_name,
                score=None,
                reasoning="",
                details={},
                raw_response=raw,
                parse_error=parse_err,
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                from_cache=response.get("from_cache", False),
            )

        score = _coerce_score(parsed.get("score"))
        reasoning = str(parsed.get("reasoning", "") or "")

        # Soft validation — if the rubric demands a score and we got nothing,
        # surface that in parse_error without dropping the rest of the data.
        if score is None and "score" in parsed:
            parse_err = f"score field present but invalid: {parsed.get('score')!r}"

        return JudgeResult(
            judge_name=self.judge_name,
            score=score,
            reasoning=reasoning,
            details=parsed,
            raw_response=raw,
            parse_error=parse_err,
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
            from_cache=response.get("from_cache", False),
        )
