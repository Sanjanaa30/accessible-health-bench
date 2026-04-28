"""
scripts/mini_pilot.py

One-off targeted mini-pilot.

Grounds exactly 6 prompts (1 per sub-category) across all 4 providers = 24
enriched files. Verifies that all three research-question pipelines fire:
  RQ1 affordability  — fin_con_01 should produce a normalized cost + Thrifty bucket
  RQ2 cultural       — cul_con_01 should produce non-Western cuisine tags
  RQ3 feasibility    — lif_con_01 should produce a Compendium WHO bucket

Approach: monkeypatch Path.glob with a filter that only returns files matching
the target prompt IDs. The patch is scoped via try/finally so other code in
the same Python session is unaffected if anything raises.

Run from repo root:
    python scripts/mini_pilot.py
"""

import shutil
import sys
from pathlib import Path

# Ensure repo root on sys.path so `from src import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TARGET_PROMPT_IDS = {
    "fin_base_01",
    "fin_con_01",
    "cul_base_01",
    "cul_con_01",
    "lif_base_02",   # "daily fitness routine" — exercises Compendium grounding
    "lif_con_07",    # "build muscle without gym equipment" — explicit fitness
}

ENRICHED_DIR = Path("data/enriched")


def main():
    # Step 1: clear stale enriched files so we ground fresh.
    # The Wikidata SQLite cache is left intact for speed.
    if ENRICHED_DIR.exists():
        # Only remove provider sub-dirs and any sidecar JSON; preserve the dir.
        for entry in ENRICHED_DIR.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            elif entry.is_file() and entry.suffix == ".json":
                entry.unlink()
        print(f"[setup] cleared {ENRICHED_DIR}/")

    # Step 2: install a glob filter so ground_all only sees our 6 prompt IDs.
    original_glob = Path.glob

    def filtered_glob(self, pattern):
        # Only filter when scanning extraction or enriched provider dirs;
        # apply to all *.json globs since both targets use that pattern.
        results = list(original_glob(self, pattern))
        if pattern == "*.json":
            return [p for p in results if p.stem in TARGET_PROMPT_IDS]
        return results

    Path.glob = filtered_glob
    try:
        # Step 3: run the orchestrator. It will only see the 24 target files.
        from src import ground_all as ga
        print("\n[mini-pilot] running ground_all on 24 target files...")
        ga.ground_all(pilot=None, providers=None)

        # Step 4: run coverage report. The glob filter still applies, so the
        # report aggregates only over our 24 files.
        print("\n[mini-pilot] running coverage_report...")
        # Reset argv so coverage_report's argparse uses defaults.
        sys.argv = ["coverage_report"]
        from src import coverage_report
        coverage_report.main()
    finally:
        Path.glob = original_glob
        print("\n[mini-pilot] glob filter removed; Path.glob restored.")


if __name__ == "__main__":
    main()
