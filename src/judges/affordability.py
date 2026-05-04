"""
src/judges/affordability.py — RQ1 judge.

CLI:
    python -m src.judges.affordability data/enriched/openai/fin_con_01.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judges.base import BaseGEvalJudge


class AffordabilityJudge(BaseGEvalJudge):
    judge_name = "affordability"
    rubric_path = "prompts/judge_affordability.txt"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.judges.affordability <enriched_json_path>")
        sys.exit(1)

    record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    judge = AffordabilityJudge()
    result = judge.evaluate(record)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
