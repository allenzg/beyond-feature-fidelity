#!/usr/bin/env python3
"""Materialize deterministic shared corrupt-image instances for ICLR-DUG-V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(protocol_id: str, sample_id: str, corruption_type: str, severity: int) -> int:
    payload = f"{protocol_id}\t{sample_id}\t{corruption_type}\t{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_corruption(image: Image.Image, corruption_type: str, params: dict[str, Any], seed: int) -> Image.Image:
    image = image.convert("RGB")
    if corruption_type == "overexposure":
        return ImageEnhance.Brightness(image).enhance(float(params["brightness_factor"]))
    if corruption_type == "underexposure":
        return ImageEnhance.Brightness(image).enhance(float(params["brightness_factor"]))
    if corruption_type == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(params["radius"])))
    if corruption_type == "gaussian_noise":
        sigma = float(params["sigma"])
        rng = np.random.default_rng(seed)
        array = np.asarray(image).astype(np.float32)
        noise = rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    if corruption_type == "jpeg":
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(params["quality"]))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    raise ValueError(f"unsupported corruption_type: {corruption_type}")


def valid_rgb_png_shape(path: Path) -> list[int] | None:
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode == "RGB" and image.width > 0 and image.height > 0:
                return list(image.size[::-1]) + [3]
            return None
    except Exception:
        return None


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image.save(tmp_path, format="PNG", compress_level=1)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def iter_paired_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_instances: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"sample_id", "clean_image", "corrupt_image", "corruption_type", "severity", "split"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {missing}")
            key = (str(row["sample_id"]), str(row["corruption_type"]), int(row["severity"]))
            if key in seen_instances:
                raise ValueError(f"{path}:{line_no}: duplicate corruption instance {key}")
            seen_instances.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty paired manifest: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-manifest", default=Path("data/manifests/iclr_v2/paired_clean_corrupt.jsonl"), type=Path)
    parser.add_argument("--corruption-protocol", default=Path("configs/iclr_v2/corruption_protocol.json"), type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require 0 <= shard-index < num-shards")
    protocol = load_protocol(args.corruption_protocol)
    all_rows = iter_paired_rows(args.paired_manifest)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    selected_groups = [grouped[key] for idx, key in enumerate(sorted(grouped)) if idx % args.num_shards == args.shard_index]
    rows = [row for group in selected_groups for row in group]
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.output_manifest is None:
        args.output_manifest = Path(f"cache/iclr_v2/corrupt_images/manifest_shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    image_cache: tuple[Path | None, Image.Image | None] = (None, None)
    with args.output_manifest.open("w", encoding="utf-8") as manifest_out:
        for row in rows:
            clean_image = Path(row["clean_image"])
            if not clean_image.exists():
                raise FileNotFoundError(clean_image)
            out_path = Path(row["corrupt_image"])
            shape = valid_rgb_png_shape(out_path) if out_path.exists() and not args.force else None
            if shape is not None:
                image_hash = sha256_file(out_path)
            else:
                if image_cache[0] != clean_image:
                    if image_cache[1] is not None:
                        image_cache[1].close()
                    image_cache = (clean_image, Image.open(clean_image).convert("RGB"))
                assert image_cache[1] is not None
                seed = stable_seed(protocol["protocol_id"], str(row["sample_id"]), str(row["corruption_type"]), int(row["severity"]))
                params = row.get("corruption_params") or protocol["parameters"][row["corruption_type"]][str(row["severity"])]
                corrupt = apply_corruption(image_cache[1], str(row["corruption_type"]), params, seed)
                save_png_atomic(corrupt, out_path)
                shape = list(corrupt.size[::-1]) + [3]
                image_hash = sha256_file(out_path)
            stat = out_path.stat()
            manifest_out.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "clean_image": str(clean_image),
                        "corrupt_image": str(out_path),
                        "corruption_type": row["corruption_type"],
                        "severity": int(row["severity"]),
                        "severity_group": row.get("severity_group"),
                        "source": str(args.paired_manifest),
                        "shape": shape,
                        "dtype": "uint8",
                        "checksum": image_hash,
                        "size": stat.st_size,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
    if image_cache[1] is not None:
        image_cache[1].close()
    print(json.dumps({"status": "ok", "written": written, "output_manifest": str(args.output_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
