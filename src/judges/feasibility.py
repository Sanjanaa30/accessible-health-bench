"""
src/judges/feasibility.py — RQ3 judge.

CLI:
    python -m src.judges.feasibility data/enriched/openai/lif_con_07.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judges.base import BaseGEvalJudge


class FeasibilityJudge(BaseGEvalJudge):
    judge_name = "feasibility"
    rubric_path = "prompts/judge_feasibility.txt"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.judges.feasibility <enriched_json_path>")
        sys.exit(1)

    record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    judge = FeasibilityJudge()
    result = judge.evaluate(record)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
