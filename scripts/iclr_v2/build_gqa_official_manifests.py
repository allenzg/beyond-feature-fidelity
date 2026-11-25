#!/usr/bin/env python3
"""Build ICLR-DUG-V2 manifests from official GQA train/val balanced splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

CORRUPTION_GROUPS = {
    "train": ("train_severities",),
    "val": ("validation_severities", "heldout_heavy_severities"),
}
SEVERITY_GROUP = {
    "train_severities": "medium",
    "validation_severities": "medium",
    "heldout_heavy_severities": "heavy",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected official GQA question dict: {path}")
    return data


def image_path(images_dir: Path, image_id: str) -> Path:
    direct = images_dir / f"{image_id}.jpg"
    if direct.exists():
        return direct.resolve()
    nested = list(images_dir.rglob(f"{image_id}.jpg"))
    if len(nested) == 1:
        return nested[0].resolve()
    raise FileNotFoundError(f"cannot resolve imageId {image_id} under {images_dir}")


def question_rows(questions: dict[str, dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows = []
    image_ids = set()
    for question_id in sorted(questions):
        item = questions[question_id]
        image_id = str(item["imageId"])
        image_ids.add(image_id)
        types = item.get("types") or {}
        rows.append(
            {
                "question_id": str(question_id),
                "image_id": image_id,
                "sample_id": f"gqa_{image_id}",
                "split": split,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "full_answer": item.get("fullAnswer"),
                "is_balanced": bool(item.get("isBalanced", True)),
                "structural_type": types.get("structural"),
                "semantic_type": types.get("semantic"),
                "detailed_type": types.get("detailed"),
            }
        )
    return rows, image_ids


def validate_images(image_ids: set[str], images_dir: Path, split: str, sample_limit: int | None) -> list[dict[str, Any]]:
    rows = []
    ids = sorted(image_ids)
    if sample_limit is not None:
        ids = ids[:sample_limit]
    for image_id in ids:
        path = image_path(images_dir, image_id)
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
        except Exception as exc:
            raise ValueError(f"broken image for {image_id}: {path}") from exc
        rows.append(
            {
                "sample_id": f"gqa_{image_id}",
                "original_id": image_id,
                "dataset": "GQA",
                "split": split,
                "image_path": str(path),
                "clean_image": str(path),
                "width": width,
                "height": height,
            }
        )
    return rows


def corruption_id(protocol_id: str, sample_id: str, corruption_type: str, severity: int) -> str:
    return hashlib.sha256(f"{protocol_id}\t{sample_id}\t{corruption_type}\t{severity}".encode()).hexdigest()[:16]


def paired_rows(universe: list[dict[str, Any]], protocol: dict[str, Any], include_for_training: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired = []
    projector = []
    for row in universe:
        split = row["split"]
        sample_id = row["sample_id"]
        original_id = row["original_id"]
        clean_image = row["image_path"]
        for severity_field in CORRUPTION_GROUPS[split]:
            for corruption_type, severities in protocol[severity_field].items():
                for severity in severities:
                    severity = int(severity)
                    severity_group = SEVERITY_GROUP[severity_field]
                    rel = f"{split}/{original_id}/{corruption_type}/severity{severity}"
                    clean_feature = f"cache/iclr_v2/dino/{split}/clean/{sample_id}.pt"
                    corrupt_root = "corrupt" if split == "train" else severity_group
                    item = {
                        "sample_id": sample_id,
                        "original_id": original_id,
                        "dataset": "GQA",
                        "split": split,
                        "clean_image": clean_image,
                        "corrupt_image": f"cache/iclr_v2/corrupt_images/{rel}.png",
                        "corruption_type": corruption_type,
                        "severity": severity,
                        "severity_group": severity_group,
                        "corruption_id": corruption_id(protocol["protocol_id"], sample_id, corruption_type, severity),
                        "corruption_params": protocol["parameters"][corruption_type][str(severity)],
                        "clean_feature": clean_feature,
                        "corrupt_feature": f"cache/iclr_v2/dino/{split}/{corrupt_root}/{rel}.pt",
                    }
                    paired.append(item)
                    if include_for_training and split == "train":
                        projector.append({**item, "target": f"cache/iclr_v2/native/llava15/train/{sample_id}.pt"})
    return paired, projector


def append_hashes(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n# GQA official manifests\n")
        for file in files:
            handle.write(f"{sha256_file(file)}  {file}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gqa-root", default=Path("datasets/GQA"), type=Path)
    parser.add_argument("--output-dir", default=Path("data/manifests/iclr_v2"), type=Path)
    parser.add_argument("--corruption-protocol", default=Path("configs/iclr_v2/corruption_protocol.json"), type=Path)
    parser.add_argument("--hash-file", default=Path("reports/iclr_v2/manifest_hashes.txt"), type=Path)
    parser.add_argument("--sample-limit", type=int, help="debug only; do not use for formal run")
    args = parser.parse_args()

    questions_dir = args.gqa_root / "questions"
    images_dir = args.gqa_root / "images"
    train_questions = load_questions(questions_dir / "train_balanced_questions.json")
    val_questions = load_questions(questions_dir / "val_balanced_questions.json")
    train_q_rows, train_image_ids = question_rows(train_questions, "train")
    val_q_rows, val_image_ids = question_rows(val_questions, "val")
    overlap = train_image_ids & val_image_ids
    if overlap:
        raise ValueError(f"GQA train/val imageId overlap: {len(overlap)} examples, first={sorted(overlap)[:10]}")

    train_universe = validate_images(train_image_ids, images_dir, "train", args.sample_limit)
    val_universe = validate_images(val_image_ids, images_dir, "val", args.sample_limit)
    if args.sample_limit is not None:
        allowed_train = {row["original_id"] for row in train_universe}
        allowed_val = {row["original_id"] for row in val_universe}
        train_q_rows = [row for row in train_q_rows if row["image_id"] in allowed_train]
        val_q_rows = [row for row in val_q_rows if row["image_id"] in allowed_val]

    train_paths = [row["image_path"] for row in train_universe]
    val_paths = [row["image_path"] for row in val_universe]
    if len(set(train_paths)) != len(train_paths) or len(set(val_paths)) != len(val_paths):
        raise ValueError("duplicate image path within official GQA split")
    if set(train_paths) & set(val_paths):
        raise ValueError("train/evaluation image path overlap")

    protocol = json.loads(args.corruption_protocol.read_text(encoding="utf-8"))
    paired_train, projector_train = paired_rows(train_universe, protocol, include_for_training=True)
    paired_val, _ = paired_rows(val_universe, protocol, include_for_training=False)
    paired = paired_train + paired_val

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "source_universe": args.output_dir / "source_universe.jsonl",
        "evaluation_universe": args.output_dir / "evaluation_universe.jsonl",
        "train_ids": args.output_dir / "train_ids.json",
        "val_ids": args.output_dir / "val_ids.json",
        "gqa_train_questions": args.output_dir / "gqa_train_questions.jsonl",
        "gqa_val_questions": args.output_dir / "gqa_val_questions.jsonl",
        "paired": args.output_dir / "paired_clean_corrupt.jsonl",
        "projector": args.output_dir / "projector_clean_corrupt.jsonl",
        "dataset_manifest": args.output_dir / "dataset_manifest.json",
    }
    write_jsonl(outputs["source_universe"], train_universe)
    write_jsonl(outputs["evaluation_universe"], val_universe)
    write_json(outputs["train_ids"], [row["sample_id"] for row in train_universe])
    write_json(outputs["val_ids"], [row["sample_id"] for row in val_universe])
    write_jsonl(outputs["gqa_train_questions"], train_q_rows)
    write_jsonl(outputs["gqa_val_questions"], val_q_rows)
    write_jsonl(outputs["paired"], paired)
    write_jsonl(outputs["projector"], projector_train)
    manifest = {
        "protocol": "ICLR-DUG-V2",
        "dataset": "GQA official balanced split",
        "gqa_root": str(args.gqa_root),
        "train_questions": str(questions_dir / "train_balanced_questions.json"),
        "val_questions": str(questions_dir / "val_balanced_questions.json"),
        "images_dir": str(images_dir),
        "counts": {
            "train_questions": len(train_q_rows),
            "val_questions": len(val_q_rows),
            "train_images": len(train_universe),
            "val_images": len(val_universe),
            "train_val_image_overlap": 0,
            "paired_rows": len(paired),
            "projector_rows": len(projector_train),
        },
        "leakage_audit": {
            "train_image_ids_intersect_val_image_ids": 0,
            "train_source_images_intersect_evaluation_images": 0,
            "projector_training_uses_only_train_images": True,
            "formal_gqa_questions_reference_only_val_images": True,
        },
        "archives": {
            "questions1.2_zip_sha256": sha256_file(args.gqa_root / "archives" / "questions1.2.zip") if (args.gqa_root / "archives" / "questions1.2.zip").exists() else None,
            "images_zip_sha256": sha256_file(args.gqa_root / "archives" / "images.zip") if (args.gqa_root / "archives" / "images.zip").exists() else None,
        },
    }
    write_json(outputs["dataset_manifest"], manifest)
    append_hashes(args.hash_file, [*outputs.values(), args.corruption_protocol])
    print(json.dumps({"status": "ok", "counts": manifest["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
