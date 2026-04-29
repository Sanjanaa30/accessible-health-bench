"""
src/judges/adherence.py — Constraint Adherence DAG judge.

Cross-cutting judge that produces deterministic, auditable per-branch
verdicts (one branch per RQ). Output is a structured dict, not a single
score, so it doesn't subclass BaseGEvalJudge.

For aggregation purposes, we also derive a 1-5 numeric score from the
three branches: average of branch verdicts where each verdict maps to
{yes:5, partial:3, no:1}; "not_applicable" branches are excluded.

Usage:
    from src.judges.adherence import AdherenceJudge
    judge = AdherenceJudge()
    result = judge.evaluate(enriched_record)
    print(result.score)              # numeric for aggregation, may be None
    print(result.branch_verdicts)    # full structured DAG output
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.clients.unified_llm import UnifiedLLM
from src.config import JUDGE_MODEL
from src.judges.base import build_judge_input, parse_judge_json


# Verdict-to-numeric mapping used when averaging across branches.
# "no" maps to 1 (worst); "yes" maps to 5 (best). "not_applicable"
# branches are skipped from the average entirely.
VERDICT_TO_NUMERIC = {
    "yes": 5,
    "partial": 3,
    "no": 1,
}

BRANCH_KEYS = ("branch_1_financial", "branch_2_cultural", "branch_3_lifestyle")


@dataclass
class AdherenceResult:
    judge_name: str = "adherence"
    score: Optional[float] = None        # averaged across applicable branches
    branch_verdicts: dict = field(default_factory=dict)
    applicable_branches: int = 0
    contradictions: list[str] = field(default_factory=list)
    raw_response: str = ""
    parse_error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    from_cache: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class AdherenceJudge:
    judge_name = "adherence"
    rubric_path = "prompts/judge_adherence_dag.txt"
    max_tokens: int = 1500

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

    def evaluate(self, record: dict) -> AdherenceResult:
        rubric = self._load_rubric()
        # The rubric is fully self-contained — INPUT block + JSON OUTPUT
        # schema both live in the file. build_judge_input just fills the
        # four placeholders; we do not append our own scaffolding.
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
            return AdherenceResult(
                raw_response=raw,
                parse_error=parse_err,
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                from_cache=response.get("from_cache", False),
            )

        score, applicable_count, contradictions = self._derive_score(parsed)

        return AdherenceResult(
            score=score,
            branch_verdicts=parsed,
            applicable_branches=applicable_count,
            contradictions=contradictions,
            raw_response=raw,
            parse_error=parse_err,
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
            from_cache=response.get("from_cache", False),
        )

    @staticmethod
    def _derive_score(verdicts: dict) -> tuple[Optional[float], int, list[str]]:
        """
        Compute the numeric adherence score from per-branch verdicts.

        Counts "applicable" branches via the explicit `applicable: true` flag
        (more authoritative than checking verdict != "not_applicable" — the
        rubric's discipline section warns the model to set both).

        Returns (score_or_None, applicable_count, contradictions_list).
        """
        applicable_scores: list[int] = []
        applicable_count = 0
        contradictions: list[str] = []

        for branch_key in BRANCH_KEYS:
            branch = verdicts.get(branch_key)
            if not isinstance(branch, dict):
                continue

            applicable = branch.get("applicable") is True
            verdict = str(branch.get("verdict", "")).strip().lower()

            # Detect rubric-violation cases where applicable + verdict disagree
            if applicable and verdict == "not_applicable":
                contradictions.append(
                    f"{branch_key}: applicable=true but verdict=not_applicable"
                )
                continue
            if not applicable and verdict not in ("not_applicable", ""):
                contradictions.append(
                    f"{branch_key}: applicable=false but verdict={verdict!r}"
                )
                # Trust the verdict — model said it judged this branch.
                applicable = True

            if not applicable:
                continue

            applicable_count += 1
            numeric = VERDICT_TO_NUMERIC.get(verdict)
            if numeric is not None:
                applicable_scores.append(numeric)
            else:
                contradictions.append(
                    f"{branch_key}: applicable but verdict not in "
                    f"{{yes,partial,no}}: {verdict!r}"
                )

        if not applicable_scores:
            return None, applicable_count, contradictions

        avg = sum(applicable_scores) / len(applicable_scores)
        return round(avg, 2), applicable_count, contradictions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.judges.adherence <enriched_json_path>")
        sys.exit(1)

    record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    judge = AdherenceJudge()
    result = judge.evaluate(record)
    print(json.dumps(result.to_dict(), indent=2))
