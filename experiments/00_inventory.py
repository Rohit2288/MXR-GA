"""Task 0.2 — instance inventory.

Loads and validates every instance JSON in both frozen corpora, tabulates
counts per suite / difficulty category / distribution / n, and verifies them
against (a) each corpus manifest and (b) the counts claimed in the paper /
quoted by reviewers. Any discrepancy is flagged as DECISION(Rohit).

Usage: .venv/bin/python experiments/00_inventory.py
Output: results/inventory/inventory.json + console report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.instance import (FSSP1_DIR, FSSP2_DIR, load_corpus,
                               load_manifest)

# Counts asserted in the submitted manuscript (as quoted in reviewer_comments.md):
# R3.2 quotes "384 Suite U instances"; R4 quotes "866 instances" for FSSP-2.
# FSSP-1 total is taken from its own manifest (paper says "over 3,000 paired
# instances" — R4).
PAPER_CLAIMS = {
    "fssp2_total": 866,
    "fssp2_suite_U": 384,
}

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "inventory"


def tabulate(instances):
    return {
        "total": len(instances),
        "by_suite": dict(Counter(i.suite or "-" for i in instances)),
        "by_distribution": dict(Counter(i.distribution for i in instances)),
        "by_n": {str(k): v for k, v in sorted(
            Counter(i.n for i in instances).items())},
        "by_difficulty_category": dict(Counter(
            i.difficulty.get("category", "?") for i in instances)),
    }


def main() -> int:
    report = {}
    problems = []

    for label, root in (("FSSP-1", FSSP1_DIR), ("FSSP-2", FSSP2_DIR)):
        print(f"Loading + validating {label} from {root} ...")
        instances = load_corpus(root)
        manifest = load_manifest(root)
        tab = tabulate(instances)
        tab["manifest_total"] = manifest["total_instances"]
        tab["manifest_entries"] = len(manifest["instances"])

        ids = [i.instance_id for i in instances]
        if len(set(ids)) != len(ids):
            problems.append(f"{label}: duplicate instance_ids on disk")
        if tab["total"] != manifest["total_instances"]:
            problems.append(
                f"{label}: {tab['total']} files on disk but manifest claims "
                f"{manifest['total_instances']}")
        manifest_ids = {e["id"] for e in manifest["instances"]}
        disk_ids = set(ids)
        if manifest_ids != disk_ids:
            problems.append(
                f"{label}: manifest/disk id mismatch "
                f"(only-manifest={len(manifest_ids - disk_ids)}, "
                f"only-disk={len(disk_ids - manifest_ids)})")
        report[label] = tab

    f2 = report["FSSP-2"]
    if f2["total"] != PAPER_CLAIMS["fssp2_total"]:
        problems.append(
            f"FSSP-2 total {f2['total']} != paper claim "
            f"{PAPER_CLAIMS['fssp2_total']}")
    if f2["by_suite"].get("U", 0) != PAPER_CLAIMS["fssp2_suite_U"]:
        problems.append(
            f"Suite U count {f2['by_suite'].get('U', 0)} != paper claim "
            f"{PAPER_CLAIMS['fssp2_suite_U']}")

    report["paper_claims"] = PAPER_CLAIMS
    report["discrepancies"] = problems

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "inventory.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    if problems:
        print("\n*** DECISION(Rohit) — discrepancies found: ***")
        for p in problems:
            print(" -", p)
        return 1
    print(f"\nAll counts verified. Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
