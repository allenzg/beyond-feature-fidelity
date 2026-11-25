#!/usr/bin/env python3
"""Summarize ICLR-DUG-V2 tensor cache integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape_dtype(path: Path) -> tuple[list[int] | None, str | None]:
    value = torch.load(path, map_location="cpu")
    tensor = value if isinstance(value, torch.Tensor) else None
    if isinstance(value, dict):
        for key in ("features", "feature", "dino", "tokens", "target", "visual_tokens"):
            if key in value and isinstance(value[key], torch.Tensor):
                tensor = value[key]
                break
    if tensor is None:
        return None, None
    return list(tensor.shape), str(tensor.dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    files = sorted(p for p in args.cache_root.rglob("*.pt") if p.is_file())
    samples = []
    total_size = 0
    for path in files:
        total_size += path.stat().st_size
    for path in files[: args.limit]:
        shape, dtype = tensor_shape_dtype(path)
        samples.append({"path": str(path), "shape": shape, "dtype": dtype, "size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"cache_root": str(args.cache_root), "n_files": len(files), "total_size": total_size, "first": samples[:5], "last_paths": [str(p) for p in files[-5:]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
