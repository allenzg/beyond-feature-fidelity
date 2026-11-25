#!/usr/bin/env python3
"""Build formal VAL medium/heavy corruption manifest for ICLR-DUG-V2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.iclr_v2.build_corruptions import stable_seed
except ModuleNotFoundError:
    from build_corruptions import stable_seed

FAMILIES = ["overexposure", "underexposure", "gaussian_blur", "gaussian_noise", "jpeg"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-manifest", default=Path("data/manifests/iclr_v2/paired_clean_corrupt.jsonl"), type=Path)
    parser.add_argument("--corruption-protocol", default=Path("configs/iclr_v2/corruption_protocol.json"), type=Path)
    parser.add_argument("--output", default=Path("data/manifests/iclr_v2/val_corrupt_formal.jsonl"), type=Path)
    parser.add_argument("--summary", default=Path("reports/iclr_v2/val_corrupt_formal_summary.json"), type=Path)
    args = parser.parse_args()

    protocol = load_protocol(args.corruption_protocol)
    protocol_id = str(protocol["protocol_id"])
    allowed: set[tuple[str, int, str]] = set()
    for family in FAMILIES:
        for severity in protocol["validation_severities"][family]:
            allowed.add((family, int(severity), "medium"))
        for severity in protocol["heldout_heavy_severities"][family]:
            allowed.add((family, int(severity), "heavy"))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in read_jsonl(args.paired_manifest):
        if row.get("split") != "val":
            continue
        family = str(row["corruption_type"])
        severity = int(row["severity"])
        group = "heavy" if (family, severity, "heavy") in allowed else "medium"
        if (family, severity, group) not in allowed:
            continue
        key = (str(row["sample_id"]), family, severity)
        if key in seen:
            continue
        seen.add(key)
        sample_id = str(row["sample_id"])
        rows.append(
            {
                "sample_id": sample_id,
                "split": "val",
                "clean_image": row["clean_image"],
                "clean_feature": row["clean_feature"],
                "corrupt_feature": f"cache/iclr_v2/dino/val/{group}/{family}/{sample_id}_severity{severity}.pt",
                "corruption_type": family,
                "severity": severity,
                "severity_group": group,
                "corruption_seed": stable_seed(protocol_id, sample_id, family, severity),
                "corruption_params": protocol["parameters"][family][str(severity)],
                "corruption_protocol_id": protocol_id,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    combo_counts = Counter(f"{row['severity_group']}:{row['corruption_type']}:{row['severity']}" for row in rows)
    summary = {
        "status": "ok",
        "row_count": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "combo_counts": dict(sorted(combo_counts.items())),
        "output": str(args.output),
        "protocol_id": protocol_id,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
