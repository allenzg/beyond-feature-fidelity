#!/usr/bin/env python3
"""Build sharded frozen-DINO feature caches for ICLR-DUG-V2.

Typical 4-GPU use:
  CUDA_VISIBLE_DEVICES=0 python scripts/iclr_v2/build_dino_cache.py --model-path /path/to/dinov3 --mode clean --shard-index 0 --num-shards 4
  CUDA_VISIBLE_DEVICES=1 python scripts/iclr_v2/build_dino_cache.py --model-path /path/to/dinov3 --mode clean --shard-index 1 --num-shards 4
  CUDA_VISIBLE_DEVICES=2 python scripts/iclr_v2/build_dino_cache.py --model-path /path/to/dinov3 --mode corrupt --shard-index 2 --num-shards 4
  CUDA_VISIBLE_DEVICES=3 python scripts/iclr_v2/build_dino_cache.py --model-path /path/to/dinov3 --mode corrupt --shard-index 3 --num-shards 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

try:
    from scripts.iclr_v2.build_corruptions import apply_corruption, load_protocol, stable_seed
except ModuleNotFoundError:
    from build_corruptions import apply_corruption, load_protocol, stable_seed


ALLOWED_CACHE_DTYPES = {torch.bfloat16, torch.float16}
MAX_SINGLE_FEATURE_CACHE_BYTES = 4 * 1024 * 1024
DEFAULT_DINO_CACHE_ROOT = Path("cache/iclr_v2/dino")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "sample_id" not in row:
                raise ValueError(f"{path}:{line_no}: missing sample_id")
            rows.append(row)
    return rows


def clean_jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in by_sample:
            by_sample[sample_id] = {
                "cache_kind": "clean",
                "cache_key": sample_id,
                "sample_id": sample_id,
                "split": row["split"],
                "image": row["clean_image"],
                "out": row["clean_feature"],
                "corruption": "clean",
                "corruption_type": "clean",
                "severity": 0,
            }
    return [by_sample[key] for key in sorted(by_sample)]


def corrupt_jobs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs = []
    seen: set[str] = set()
    for row in rows:
        key = f"{row['sample_id']}\t{row['corruption_type']}\t{row['severity']}"
        if key in seen:
            continue
        seen.add(key)
        jobs.append(
            {
                "cache_kind": "corrupt",
                "cache_key": key,
                "sample_id": str(row["sample_id"]),
                "split": row["split"],
                "clean_image": row["clean_image"],
                "corruption_params": row.get("corruption_params"),
                "image": row["corrupt_image"],
                "out": row["corrupt_feature"],
                "corruption": row["corruption_type"],
                "corruption_type": row["corruption_type"],
                "severity": int(row["severity"]),
            }
        )
    return jobs


def corrupt_groups(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return [grouped[key] for idx, key in enumerate(sorted(grouped)) if idx % num_shards == shard_index]


def expected_patch_tokens(image_size: int, patch_size: int) -> int:
    return (image_size // patch_size) ** 2


def tensor_raw_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def assert_formal_feature(tensor: torch.Tensor, expected_tokens: int, expected_dim: int) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"DINO feature must be rank-2, got shape={list(tensor.shape)}")
    if tuple(tensor.shape) != (expected_tokens, expected_dim):
        raise ValueError(f"DINO feature shape must be {(expected_tokens, expected_dim)}, got {list(tensor.shape)}")
    if tensor.dtype not in ALLOWED_CACHE_DTYPES:
        raise ValueError(f"DINO cache dtype must be bf16/fp16, got {tensor.dtype}")
    expected_bytes = expected_tokens * expected_dim * 2
    raw_bytes = tensor_raw_bytes(tensor)
    if raw_bytes != expected_bytes:
        raise ValueError(f"DINO feature raw bytes must be {expected_bytes}, got {raw_bytes}")
    storage_bytes = tensor.untyped_storage().nbytes()
    if storage_bytes != raw_bytes:
        raise ValueError(
            f"DINO feature storage bytes must equal raw bytes to avoid serializing batch storage: "
            f"storage={storage_bytes}, raw={raw_bytes}"
        )


def feature_for_cache(tensor: torch.Tensor, expected_tokens: int, expected_dim: int, cache_dtype: torch.dtype) -> torch.Tensor:
    feature = tensor.detach().to(device="cpu", dtype=cache_dtype).contiguous().clone()
    assert_formal_feature(feature, expected_tokens, expected_dim)
    return feature


def remap_cache_path(path: Path, cache_root: Path) -> Path:
    if path.is_absolute():
        return path
    try:
        relative = path.relative_to(DEFAULT_DINO_CACHE_ROOT)
    except ValueError:
        return path
    return cache_root / relative


def load_feature_if_valid(path: Path, expected_tokens: int, expected_dim: int) -> torch.Tensor | None:
    try:
        if path.stat().st_size > MAX_SINGLE_FEATURE_CACHE_BYTES:
            return None
        payload = torch.load(path, map_location="cpu")
        tensor = payload["features"] if isinstance(payload, dict) and "features" in payload else payload
        if not isinstance(tensor, torch.Tensor):
            return None
        assert_formal_feature(tensor, expected_tokens, expected_dim)
        return tensor.contiguous().clone()
    except Exception:
        return None


def save_feature_atomic(payload: Any, path: Path, max_file_bytes: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        if max_file_bytes is not None and tmp_path.stat().st_size > max_file_bytes:
            raise ValueError(
                f"refusing oversized DINO cache file {tmp_path}: "
                f"{tmp_path.stat().st_size} bytes > {max_file_bytes} bytes"
            )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def strip_and_validate_features(
    model,
    tensor: torch.Tensor,
    expected_tokens: int,
    expected_dim: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"unexpected DINO output rank: {list(tensor.shape)}")
    special_tokens = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    if tensor.shape[1] == expected_tokens + special_tokens:
        tensor = tensor[:, special_tokens:, :]
    elif tensor.shape[1] == expected_tokens + 1:
        tensor = tensor[:, 1:, :]
    elif tensor.shape[1] != expected_tokens:
        raise ValueError(
            f"unexpected DINO token count: got {tensor.shape[1]}, "
            f"expected {expected_tokens} patches after special-token removal"
        )
    tensor = tensor.detach().to(device="cpu", dtype=cache_dtype).contiguous()
    if tensor.shape[1:] != (expected_tokens, expected_dim):
        raise ValueError(f"unexpected DINO feature shape: {list(tensor.shape)}")
    return tensor


def extract_image_batch(
    model,
    processor,
    images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
    image_size: int,
    expected_tokens: int,
    expected_dim: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt", size={"height": image_size, "width": image_size})
    inputs = {key: value.to(device) for key, value in inputs.items()}
    autocast_enabled = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        outputs = model(**inputs)
    tensor = getattr(outputs, "last_hidden_state", None)
    if tensor is None:
        tensor = outputs[0]
    return strip_and_validate_features(model, tensor, expected_tokens, expected_dim, cache_dtype)


def extract_feature_batch(
    model,
    processor,
    image_paths: list[Path],
    device: torch.device,
    dtype: torch.dtype,
    image_size: int,
    expected_tokens: int,
    expected_dim: int,
    cache_dtype: torch.dtype,
) -> torch.Tensor:
    return extract_image_batch(
        model,
        processor,
        [load_rgb(path) for path in image_paths],
        device,
        dtype,
        image_size,
        expected_tokens,
        expected_dim,
        cache_dtype,
    )


def feature_uri(path: Path, key: str) -> str:
    return f"{path}::{key}"


def split_feature_uri(value: str) -> tuple[Path, str | None]:
    if "::" not in value:
        return Path(value), None
    path, key = value.rsplit("::", 1)
    return Path(path), key


def load_packed_feature_if_valid(path: Path, key: str, expected_tokens: int, expected_dim: int) -> torch.Tensor | None:
    try:
        payload = torch.load(path, map_location="cpu")
        features = payload.get("features") if isinstance(payload, dict) else None
        tensor = features.get(key) if isinstance(features, dict) else None
        if not isinstance(tensor, torch.Tensor):
            return None
        assert_formal_feature(tensor, expected_tokens, expected_dim)
        return tensor.contiguous().clone()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-manifest", default=Path("data/manifests/iclr_v2/paired_clean_corrupt.jsonl"), type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--mode", choices=["clean", "corrupt", "both"], default="both")
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--expected-tokens", default=576, type=int)
    parser.add_argument("--expected-dim", default=1024, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--cache-root", default=DEFAULT_DINO_CACHE_ROOT, type=Path)
    parser.add_argument("--corruption-protocol", default=Path("configs/iclr_v2/corruption_protocol.json"), type=Path)
    parser.add_argument("--no-pack-corrupt-by-sample", dest="pack_corrupt_by_sample", action="store_false")
    parser.set_defaults(pack_corrupt_by_sample=True)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require 0 <= shard-index < num-shards")
    if not args.paired_manifest.exists():
        raise FileNotFoundError(args.paired_manifest)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    cache_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    patch_size = int(getattr(model.config, "patch_size", 16) or 16)
    derived_tokens = expected_patch_tokens(args.image_size, patch_size)
    if derived_tokens != args.expected_tokens:
        raise ValueError(
            f"image-size/patch-size imply {derived_tokens} tokens, not expected {args.expected_tokens}: "
            f"image_size={args.image_size}, patch_size={patch_size}"
        )
    hidden_size = int(getattr(model.config, "hidden_size", args.expected_dim) or args.expected_dim)
    if hidden_size != args.expected_dim:
        raise ValueError(f"DINO hidden dim mismatch: got {hidden_size}, expected {args.expected_dim}")

    paired_rows = read_jsonl(args.paired_manifest)
    if args.mode == "corrupt" and args.pack_corrupt_by_sample:
        protocol = load_protocol(args.corruption_protocol)
        groups = corrupt_groups(paired_rows, args.shard_index, args.num_shards)
        if args.limit is not None:
            groups = groups[: args.limit]
        if args.output_manifest is None:
            args.output_manifest = Path(f"cache/iclr_v2/dino_manifest_{args.mode}_shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl")
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.output_manifest.open("w", encoding="utf-8", buffering=1) as manifest_out:
            for group in groups:
                first = group[0]
                sample_id = str(first["sample_id"])
                split = str(first["split"])
                clean_image = Path(first["clean_image"])
                if not clean_image.exists():
                    raise FileNotFoundError(clean_image)
                pack_path = args.cache_root / split / "corrupt_by_sample" / f"{sample_id}.pt"
                keys = [f"{row['corruption_type']}:{int(row['severity'])}" for row in group]
                tensors: list[torch.Tensor] | None = None
                if pack_path.exists() and not args.force:
                    loaded = [load_packed_feature_if_valid(pack_path, key, args.expected_tokens, args.expected_dim) for key in keys]
                    if all(tensor is not None for tensor in loaded):
                        tensors = [tensor for tensor in loaded if tensor is not None]
                if tensors is None:
                    base_image = load_rgb(clean_image)
                    images = []
                    for row, key in zip(group, keys, strict=True):
                        seed = stable_seed(protocol["protocol_id"], sample_id, str(row["corruption_type"]), int(row["severity"]))
                        params = row.get("corruption_params") or protocol["parameters"][row["corruption_type"]][str(row["severity"])]
                        images.append(apply_corruption(base_image, str(row["corruption_type"]), params, seed))
                    features = extract_image_batch(
                        model,
                        processor,
                        images,
                        device,
                        dtype,
                        args.image_size,
                        args.expected_tokens,
                        args.expected_dim,
                        cache_dtype,
                    )
                    tensors = [feature_for_cache(tensor, args.expected_tokens, args.expected_dim, cache_dtype) for tensor in features]
                    save_feature_atomic(
                        {
                            "features": {key: tensor for key, tensor in zip(keys, tensors, strict=True)},
                            "sample_id": sample_id,
                            "source": str(clean_image),
                            "corruption_keys": keys,
                            "dino_model_path": str(args.model_path),
                            "image_size": args.image_size,
                            "patch_size": patch_size,
                            "num_register_tokens": int(getattr(model.config, "num_register_tokens", 0) or 0),
                            "storage": "corrupt_by_sample_pack",
                        },
                        pack_path,
                        max_file_bytes=MAX_SINGLE_FEATURE_CACHE_BYTES * len(tensors),
                    )
                checksum = sha256_file(pack_path)
                stat = pack_path.stat()
                for row, key, tensor in zip(group, keys, tensors, strict=True):
                    manifest_out.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "split": row["split"],
                                "path": feature_uri(pack_path, key),
                                "pack_path": str(pack_path),
                                "pack_key": key,
                                "shape": list(tensor.shape),
                                "dtype": str(tensor.dtype),
                                "source": str(clean_image),
                                "corrupt_image": row.get("corrupt_image"),
                                "corruption": row["corruption_type"],
                                "severity": int(row["severity"]),
                                "checksum": checksum,
                                "size": stat.st_size,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    written += 1
        print(json.dumps({"status": "ok", "written": written, "output_manifest": str(args.output_manifest)}, sort_keys=True))
        return 0

    jobs: list[dict[str, Any]] = []
    if args.mode in {"clean", "both"}:
        jobs.extend(clean_jobs(paired_rows))
    if args.mode in {"corrupt", "both"}:
        jobs.extend(corrupt_jobs(paired_rows))
    jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        jobs = jobs[: args.limit]

    if args.output_manifest is None:
        args.output_manifest = Path(f"cache/iclr_v2/dino_manifest_{args.mode}_shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    batch: list[dict[str, Any]] = []

    def write_record(manifest_out, job: dict[str, Any], out_path: Path, tensor: torch.Tensor) -> None:
        stat = out_path.stat()
        manifest_out.write(
            json.dumps(
                {
                    "sample_id": job["sample_id"],
                    "split": job["split"],
                    "path": str(out_path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "source": str(job["image"]),
                    "corruption": job["corruption_type"],
                    "severity": job["severity"],
                    "checksum": sha256_file(out_path),
                    "size": stat.st_size,
                },
                sort_keys=True,
            )
            + "\n"
        )

    def flush_batch(manifest_out) -> int:
        if not batch:
            return 0
        image_paths = [Path(job["image"]) for job in batch]
        tensors = extract_feature_batch(
            model,
            processor,
            image_paths,
            device,
            dtype,
            args.image_size,
            args.expected_tokens,
            args.expected_dim,
            cache_dtype,
        )
        count = 0
        for job, tensor in zip(batch, tensors, strict=True):
            out_path = remap_cache_path(Path(job["out"]), args.cache_root)
            feature = feature_for_cache(tensor, args.expected_tokens, args.expected_dim, cache_dtype)
            save_feature_atomic(feature, out_path, max_file_bytes=MAX_SINGLE_FEATURE_CACHE_BYTES)
            write_record(manifest_out, job, out_path, feature)
            count += 1
        batch.clear()
        return count

    with args.output_manifest.open("w", encoding="utf-8") as manifest_out:
        for job in jobs:
            image_path = Path(job["image"])
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            out_path = remap_cache_path(Path(job["out"]), args.cache_root)
            if out_path.exists() and not args.force:
                tensor = load_feature_if_valid(out_path, args.expected_tokens, args.expected_dim)
            else:
                tensor = None
            if tensor is None:
                batch.append(job)
                if len(batch) >= args.batch_size:
                    written += flush_batch(manifest_out)
            else:
                write_record(manifest_out, job, out_path, tensor)
                written += 1
        written += flush_batch(manifest_out)
    print(json.dumps({"status": "ok", "written": written, "output_manifest": str(args.output_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
