#!/usr/bin/env python3
"""Generate immutable deterministic ICLR-DUG-V2 splits and paired manifests.

Input source manifest is JSONL. Each row must provide a stable sample id and a clean
image path. Accepted aliases:
  sample_id: sample_id, id, image_id, question_id
  clean_image: clean_image, image_path, image, path

The script intentionally refuses to overwrite frozen V2 artifacts unless
--force is supplied. Use --force only before protocol freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLIT_FILES = ("train_ids.json", "val_ids.json", "test_ids.json")
MANIFEST_FILES = (
    "source_images.jsonl",
    "paired_clean_corrupt.jsonl",
    "projector_clean_corrupt.jsonl",
    "dataset_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}; pass --force only before freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}; pass --force only before freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(item)
    return rows


def normalize_source_manifest(path: Path, strict_paths: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, item in enumerate(read_jsonl(path), 1):
        sample_id = str(item.get("sample_id") or item.get("id") or item.get("image_id") or item.get("question_id") or "")
        clean_image = str(item.get("clean_image") or item.get("image_path") or item.get("image") or item.get("path") or "")
        if not sample_id or not clean_image:
            raise ValueError(
                f"{path}:{line_no}: must contain sample_id/id/image_id/question_id and clean_image/image_path/image/path"
            )
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {path}")
        seen.add(sample_id)
        image_path = Path(clean_image)
        if strict_paths and not image_path.exists():
            raise FileNotFoundError(f"image path for {sample_id!r} does not exist: {image_path}")
        rows.append(
            {
                "sample_id": sample_id,
                "clean_image": clean_image,
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "answer": item.get("answer", item.get("answers", item.get("gt"))),
                "source_row": item,
            }
        )
    if not rows:
        raise ValueError(f"empty source manifest: {path}")
    return rows


def validate_fractions(train_frac: float, val_frac: float) -> None:
    if train_frac <= 0 or val_frac <= 0:
        raise ValueError("train-frac and val-frac must be positive")
    if train_frac + val_frac >= 1:
        raise ValueError("train-frac + val-frac must be < 1")


def split_ids(sample_ids: list[str], seed: int, train_frac: float, val_frac: float) -> dict[str, list[str]]:
    validate_fractions(train_frac, val_frac)
    ids = sorted(sample_ids)
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    if n_total < 3:
        raise ValueError(f"need at least 3 unique samples to create train/val/test, got {n_total}")
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_train = min(max(n_train, 1), n_total - 2)
    n_val = min(max(n_val, 1), n_total - n_train - 1)
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def validate_splits(splits: dict[str, list[str]], expected_ids: set[str]) -> None:
    split_sets = {name: set(values) for name, values in splits.items()}
    if any(len(split_sets[name]) != len(splits[name]) for name in split_sets):
        raise ValueError("duplicate sample_id inside a split")
    for left in split_sets:
        for right in split_sets:
            if left >= right:
                continue
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"split overlap {left}/{right}: {sorted(overlap)[:5]}")
    union = set().union(*split_sets.values())
    if union != expected_ids:
        missing = sorted(expected_ids - union)[:5]
        extra = sorted(union - expected_ids)[:5]
        raise ValueError(f"split coverage mismatch; missing={missing}, extra={extra}")


def corruption_entries(protocol: dict[str, Any], severity_field: str, severity_group: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for corruption_type, severities in protocol[severity_field].items():
        for severity in severities:
            severity_key = str(severity)
            entries.append(
                {
                    "corruption_type": corruption_type,
                    "severity": int(severity),
                    "severity_group": severity_group,
                    "params": protocol["parameters"][corruption_type][severity_key],
                }
            )
    return entries


def entries_for_split(protocol: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split == "train":
        return corruption_entries(protocol, "train_severities", "medium")
    if split == "val":
        return corruption_entries(protocol, "validation_severities", "medium")
    if split == "test":
        return corruption_entries(protocol, "validation_severities", "medium") + corruption_entries(
            protocol, "heldout_heavy_severities", "heavy"
        )
    raise ValueError(f"unknown split: {split}")


def deterministic_corruption_id(protocol_id: str, sample_id: str, corruption_type: str, severity: int) -> str:
    payload = f"{protocol_id}\t{sample_id}\t{corruption_type}\t{severity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_rows(source_rows: list[dict[str, Any]], splits: dict[str, list[str]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_of = {sample_id: split for split, values in splits.items() for sample_id in values}
    source_by_id = {row["sample_id"]: row for row in source_rows}
    source_out: list[dict[str, Any]] = []
    paired_out: list[dict[str, Any]] = []
    projector_out: list[dict[str, Any]] = []
    for sample_id in sorted(source_by_id):
        row = source_by_id[sample_id]
        split = split_of[sample_id]
        clean_image = row["clean_image"]
        base = {
            "sample_id": sample_id,
            "split": split,
            "clean_image": clean_image,
            "question_id": row.get("question_id"),
            "question": row.get("question"),
            "answer": row.get("answer"),
        }
        source_out.append({**base, "source_row": row["source_row"]})
        for corr in entries_for_split(protocol, split):
            corruption_type = corr["corruption_type"]
            severity = corr["severity"]
            corruption_id = deterministic_corruption_id(protocol["protocol_id"], sample_id, corruption_type, severity)
            rel_stem = f"{sample_id}/{corruption_type}/severity{severity}"
            paired_row = {
                **base,
                "corrupt_image": f"cache/iclr_v2/corrupt_images/{rel_stem}.png",
                "corruption_type": corruption_type,
                "severity": severity,
                "severity_group": corr["severity_group"],
                "corruption_id": corruption_id,
                "corruption_params": corr["params"],
                "clean_feature": f"cache/iclr_v2/dino_clean/{sample_id}.pt",
                "corrupt_feature": f"cache/iclr_v2/dino_corrupt/{rel_stem}.pt",
            }
            paired_out.append(paired_row)
            if split in {"train", "val"}:
                projector_out.append({**paired_row, "target": f"cache/iclr_v2/llava15_native_tokens/{sample_id}.pt"})
    return source_out, paired_out, projector_out


def write_split_spec(report_path: Path, *, source_manifest: Path, output_dir: Path, protocol_path: Path, seed: int, train_frac: float, val_frac: float, splits: dict[str, list[str]], files: list[Path], source_hash: str, protocol_hash: str, force: bool) -> None:
    if report_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen artifact: {report_path}; pass --force only before freeze")
    n_total = sum(len(v) for v in splits.values())
    lines = [
        "# ICLR-DUG-V2 Split Spec",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inputs",
        "",
        f"- Source manifest: `{source_manifest}`",
        f"- Source manifest SHA256: `{source_hash}`",
        f"- Corruption protocol: `{protocol_path}`",
        f"- Corruption protocol SHA256: `{protocol_hash}`",
        "",
        "## Deterministic Split",
        "",
        f"- Seed: `{seed}`",
        f"- Fractions: train `{train_frac}`, validation `{val_frac}`, test `{1 - train_frac - val_frac:.12g}`",
        f"- Total unique samples: `{n_total}`",
        f"- Train samples: `{len(splits['train'])}`",
        f"- Validation samples: `{len(splits['val'])}`",
        f"- Test samples: `{len(splits['test'])}`",
        "- Invariant: sample IDs are unique and split sets are non-overlapping.",
        "- Freeze rule: generated split/manifest files are immutable after protocol freeze; reruns refuse overwrite without `--force`.",
        "",
        "## Outputs And SHA256",
        "",
    ]
    for path in files:
        lines.append(f"- `{path}` `{sha256_file(path)}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/manifests/iclr_v2"), type=Path)
    parser.add_argument("--corruption-protocol", default=Path("configs/iclr_v2/corruption_protocol.json"), type=Path)
    parser.add_argument("--split-spec", default=Path("reports/iclr_v2/split_spec.md"), type=Path)
    parser.add_argument("--seed", default=20260920, type=int)
    parser.add_argument("--train-frac", default=0.8, type=float)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--strict-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow regeneration before the V2 artifacts are frozen")
    args = parser.parse_args()

    source_rows = normalize_source_manifest(args.source_manifest, args.strict_paths)
    sample_ids = [row["sample_id"] for row in source_rows]
    protocol = load_json(args.corruption_protocol)
    splits = split_ids(sample_ids, args.seed, args.train_frac, args.val_frac)
    validate_splits(splits, set(sample_ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        write_json(args.output_dir / f"{split_name}_ids.json", ids, args.force)

    source_out, paired_out, projector_out = build_rows(source_rows, splits, protocol)
    write_jsonl(args.output_dir / "source_images.jsonl", source_out, args.force)
    write_jsonl(args.output_dir / "paired_clean_corrupt.jsonl", paired_out, args.force)
    write_jsonl(args.output_dir / "projector_clean_corrupt.jsonl", projector_out, args.force)

    dataset_manifest = {
        "protocol": "ICLR-DUG-V2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "corruption_protocol": str(args.corruption_protocol),
        "corruption_protocol_sha256": sha256_file(args.corruption_protocol),
        "split_seed": args.seed,
        "train_fraction": args.train_frac,
        "validation_fraction": args.val_frac,
        "test_fraction": 1 - args.train_frac - args.val_frac,
        "counts": {
            "samples_total": len(sample_ids),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "paired_clean_corrupt_rows": len(paired_out),
            "projector_rows": len(projector_out),
        },
        "invariants": {
            "sample_ids_unique": True,
            "splits_non_overlapping": True,
            "shared_corruption_instances_across_models": True,
            "heldout_heavy_only_on_test_split": True,
        },
    }
    write_json(args.output_dir / "dataset_manifest.json", dataset_manifest, args.force)

    output_files = [args.output_dir / name for name in SPLIT_FILES + MANIFEST_FILES]
    write_split_spec(
        args.split_spec,
        source_manifest=args.source_manifest,
        output_dir=args.output_dir,
        protocol_path=args.corruption_protocol,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        splits=splits,
        files=output_files,
        source_hash=sha256_file(args.source_manifest),
        protocol_hash=sha256_file(args.corruption_protocol),
        force=args.force,
    )
    print(json.dumps({"status": "ok", "counts": dataset_manifest["counts"], "split_spec": str(args.split_spec)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
