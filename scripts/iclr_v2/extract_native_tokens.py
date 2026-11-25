#!/usr/bin/env python3
"""Extract clean-image native visual-token targets for ICLR-DUG-V2.

Supported backends:
- llava15: LLaVA-1.5 style CLIP vision tower hidden states.
- qwen25/qwen35: Qwen VL visual module tokens.

The script uses only clean images from `source_universe.jsonl` and writes one tensor per
sample plus a sharded JSONL manifest. It never consumes QA labels as supervision.
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
from torch import nn
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageTextToText, AutoProcessor, CLIPVisionModel

MODEL_CACHE_ROOTS = {
    "llava15": Path("cache/iclr_v2/native/llava15/train"),
    "qwen35": Path("cache/iclr_v2/native/qwen35/train"),
    "onevision2": Path("cache/iclr_v2/native/onevision2/train"),
    "qwen25": Path("cache/iclr_v2/native/qwen25/train"),
}
DEFAULT_PROMPT = "Describe the image briefly."
MAX_NATIVE_TOKEN_CACHE_BYTES = 64 * 1024 * 1024
DEFAULT_NATIVE_STAGE_BUDGET_BYTES = 512 * 1024**3


class LegacyLlava15Native(nn.Module):
    def __init__(self, vision_tower: nn.Module, mm_projector: nn.Module, select_layer: int, select_feature: str) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.select_layer = select_layer
        self.select_feature = select_feature


def load_legacy_llava15(model_path: Path, device: torch.device, dtype: torch.dtype):
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if config.get("mm_projector_type") != "mlp2x_gelu":
        raise ValueError(f"unsupported LLaVA mm_projector_type: {config.get('mm_projector_type')}")
    vision_tower_id = str(config["mm_vision_tower"])
    vision_tower_path = Path(vision_tower_id)
    if not vision_tower_path.exists() and vision_tower_id == "openai/clip-vit-large-patch14-336":
        vision_tower_path = Path("models/clip-vit-large-patch14-336")
    if not vision_tower_path.exists():
        raise FileNotFoundError(
            f"Missing local LLaVA-1.5 vision tower: {vision_tower_id}. "
            "Provide local openai/clip-vit-large-patch14-336 files; native cache must not use random vision weights."
        )
    processor = AutoImageProcessor.from_pretrained(vision_tower_path, local_files_only=True)
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    mm_projector = nn.Sequential(
        nn.Linear(int(config["mm_hidden_size"]), int(config["hidden_size"])),
        nn.GELU(),
        nn.Linear(int(config["hidden_size"]), int(config["hidden_size"])),
    )
    index = json.loads((model_path / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    shard_names = sorted({index["weight_map"][key] for key in index["weight_map"] if key.startswith("model.mm_projector.")})
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = torch.load(model_path / shard_name, map_location="cpu")
        for key, value in shard.items():
            if key.startswith("model.mm_projector."):
                state[key.removeprefix("model.mm_projector.")] = value
    missing, unexpected = mm_projector.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"mm_projector load mismatch: missing={missing}, unexpected={unexpected}")
    mm_projector = mm_projector.to(device=device, dtype=dtype).eval()
    model = LegacyLlava15Native(
        vision_tower=vision_tower,
        mm_projector=mm_projector,
        select_layer=int(config.get("mm_vision_select_layer", -2)),
        select_feature=str(config.get("mm_vision_select_feature", "patch")),
    ).to(device).eval()
    return processor, model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"{path}:{line_no}: missing sample_id")
            if sample_id in seen:
                continue
            seen.add(sample_id)
            rows.append(row)
    return sorted(rows, key=lambda item: str(item["sample_id"]))


def load_model(model_family: str, model_path: Path, device: torch.device, dtype: torch.dtype):
    if model_family == "llava15":
        return load_legacy_llava15(model_path, device, dtype)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model_cls = AutoModelForImageTextToText if model_family == "onevision2" else AutoModel
    model = model_cls.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    return processor, model


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().nbytes()


def prepare_native_tokens_for_cache(tensor: torch.Tensor) -> torch.Tensor:
    tokens = tensor.detach().cpu().contiguous().clone()
    if tokens.ndim != 2:
        raise ValueError(f"native visual tokens must be rank-2, got shape={list(tokens.shape)}")
    if tokens.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError(f"unexpected native visual token dtype: {tokens.dtype}")
    raw_bytes = tokens.numel() * tokens.element_size()
    if tensor_storage_bytes(tokens) != raw_bytes:
        raise ValueError(f"native token storage mismatch: storage={tensor_storage_bytes(tokens)} raw={raw_bytes}")
    return tokens


def save_tokens_atomic(payload: dict[str, Any], tensor: torch.Tensor, path: Path, max_file_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp_path)
        size = tmp_path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"refusing oversized native token cache {tmp_path}: {size} bytes > {max_file_bytes}")
        raw_bytes = tensor.numel() * tensor.element_size()
        if size > max(raw_bytes * 2, raw_bytes + 1024 * 1024):
            raise ValueError(f"native token cache overhead too large: file={size} raw={raw_bytes}")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_llava15(processor, model, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", processor)
    inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    vision_tower = getattr(model, "vision_tower", None) or getattr(model, "vision_model", None)
    if vision_tower is None and hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise AttributeError("could not locate LLaVA vision tower on loaded model")
    select_layer = getattr(model, "select_layer", getattr(getattr(model, "config", object()), "mm_vision_select_layer", -2))
    select_feature = getattr(model, "select_feature", getattr(getattr(model, "config", object()), "mm_vision_select_feature", "patch"))
    projector = getattr(model, "multi_modal_projector", None) or getattr(model, "mm_projector", None)
    if projector is None:
        inner = getattr(model, "model", None)
        projector = getattr(inner, "mm_projector", None) if inner is not None else None
    if projector is None:
        raise AttributeError("could not locate LLaVA multimodal projector for LM-consumed visual tokens")
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        outputs = vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[select_layer]
        if select_feature == "patch":
            hidden = hidden[:, 1:]
        projected = projector(hidden)
    return prepare_native_tokens_for_cache(projected.squeeze(0))


def extract_llava15_batch(processor, model, images: list[Image.Image], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", processor)
    inputs = image_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    vision_tower = getattr(model, "vision_tower", None) or getattr(model, "vision_model", None)
    if vision_tower is None and hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise AttributeError("could not locate LLaVA vision tower on loaded model")
    select_layer = getattr(model, "select_layer", getattr(getattr(model, "config", object()), "mm_vision_select_layer", -2))
    select_feature = getattr(model, "select_feature", getattr(getattr(model, "config", object()), "mm_vision_select_feature", "patch"))
    projector = getattr(model, "multi_modal_projector", None) or getattr(model, "mm_projector", None)
    if projector is None:
        inner = getattr(model, "model", None)
        projector = getattr(inner, "mm_projector", None) if inner is not None else None
    if projector is None:
        raise AttributeError("could not locate LLaVA multimodal projector for LM-consumed visual tokens")
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        outputs = vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[select_layer]
        if select_feature == "patch":
            hidden = hidden[:, 1:]
        projected = projector(hidden)
    return projected.detach().to(device="cpu", dtype=dtype).contiguous()


def qwen_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def extract_qwen_vl(processor, model, image_path: Path, image: Image.Image, prompt: str, dtype: torch.dtype) -> torch.Tensor:
    if not hasattr(model, "visual"):
        raise AttributeError("Qwen VL model does not expose model.visual")
    text = processor.apply_chat_template(qwen_messages(image_path, prompt), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device).type(model.visual.dtype)
    image_grid_thw = inputs["image_grid_thw"].to(model.device)
    patch_positions = inputs.get("patch_positions")
    if patch_positions is not None:
        patch_positions = patch_positions.to(model.device)
    with torch.no_grad():
        visual_kwargs = {"grid_thw": image_grid_thw}
        if patch_positions is not None:
            visual_kwargs["patch_positions"] = patch_positions
        visual_tokens = model.visual(pixel_values, **visual_kwargs)
    if hasattr(visual_tokens, "pooler_output"):
        pooler_output = visual_tokens.pooler_output
        if isinstance(pooler_output, (tuple, list)):
            if len(pooler_output) != 1:
                raise ValueError(f"expected one Qwen visual tensor, got {len(pooler_output)} tensors")
            visual_tokens = pooler_output[0]
        else:
            visual_tokens = pooler_output
    return prepare_native_tokens_for_cache(visual_tokens.to(dtype=torch.float32))


def unpack_qwen_visual_output(visual_tokens: Any) -> torch.Tensor:
    if hasattr(visual_tokens, "last_hidden_state") and isinstance(visual_tokens.last_hidden_state, torch.Tensor):
        return visual_tokens.last_hidden_state
    if hasattr(visual_tokens, "pooler_output"):
        pooler_output = visual_tokens.pooler_output
        if isinstance(pooler_output, (tuple, list)):
            if len(pooler_output) != 1:
                raise ValueError(f"expected one Qwen visual tensor, got {len(pooler_output)} tensors")
            return pooler_output[0]
        return pooler_output
    return visual_tokens


def extract_qwen_vl_batch(
    processor,
    model,
    image_paths: list[Path],
    images: list[Image.Image],
    prompt: str,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    if not hasattr(model, "visual"):
        raise AttributeError("Qwen VL model does not expose model.visual")
    texts = [processor.apply_chat_template(qwen_messages(image_path, prompt), tokenize=False, add_generation_prompt=True) for image_path in image_paths]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(model.device, non_blocking=True).type(model.visual.dtype)
    image_grid_thw = inputs["image_grid_thw"].to(model.device, non_blocking=True)
    patch_positions = inputs.get("patch_positions")
    if patch_positions is not None:
        patch_positions = patch_positions.to(model.device, non_blocking=True)
    with torch.inference_mode():
        visual_kwargs = {"grid_thw": image_grid_thw}
        if patch_positions is not None:
            visual_kwargs["patch_positions"] = patch_positions
        visual_tokens = unpack_qwen_visual_output(model.visual(pixel_values, **visual_kwargs)).to(dtype=torch.float32)
    token_counts = (inputs["input_ids"] == int(model.config.image_token_id)).sum(dim=1).tolist()
    if int(sum(token_counts)) != int(visual_tokens.shape[0]):
        raise ValueError(f"Qwen visual token split mismatch: prompt_counts={token_counts}, visual_shape={list(visual_tokens.shape)}")
    return [prepare_native_tokens_for_cache(tensor) for tensor in visual_tokens.split([int(count) for count in token_counts], dim=0)]


def extract_tokens(model_family: str, processor, model, image_path: Path, image: Image.Image, device: torch.device, dtype: torch.dtype, prompt: str) -> torch.Tensor:
    if model_family == "llava15":
        return extract_llava15(processor, model, image, device, dtype)
    if model_family in {"qwen25", "qwen35", "onevision2"}:
        return extract_qwen_vl(processor, model, image_path, image, prompt, dtype)
    raise ValueError(f"unsupported model_family: {model_family}")


def extract_token_batch(
    model_family: str,
    processor,
    model,
    image_paths: list[Path],
    images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
    prompt: str,
) -> list[torch.Tensor]:
    if model_family == "llava15":
        return [prepare_native_tokens_for_cache(tensor) for tensor in extract_llava15_batch(processor, model, images, device, dtype)]
    if model_family in {"qwen25", "qwen35", "onevision2"}:
        return extract_qwen_vl_batch(processor, model, image_paths, images, prompt, dtype)
    raise ValueError(f"unsupported model_family: {model_family}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", required=True, choices=sorted(MODEL_CACHE_ROOTS))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-manifest", default=Path("data/manifests/iclr_v2/source_universe.jsonl"), type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--num-shards", default=1, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing-fast", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-file-bytes", default=MAX_NATIVE_TOKEN_CACHE_BYTES, type=int)
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require 0 <= shard-index < num-shards")
    if not args.source_manifest.exists():
        raise FileNotFoundError(args.source_manifest)
    output_root = args.output_root or MODEL_CACHE_ROOTS[args.model_family]
    output_manifest = args.output_manifest or output_root / f"manifest_shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    processor, model = load_model(args.model_family, args.model_path, device, dtype)
    rows = [row for idx, row in enumerate(read_jsonl(args.source_manifest)) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_manifest.open("w", encoding="utf-8") as manifest_out:
        for start in range(0, len(rows), max(1, args.batch_size)):
            batch_rows = rows[start : start + max(1, args.batch_size)]
            pending: list[tuple[dict[str, Any], Path, Path]] = []
            for row in batch_rows:
                sample_id = str(row["sample_id"])
                clean_image = Path(row.get("clean_image") or row["image_path"])
                if not clean_image.exists():
                    raise FileNotFoundError(clean_image)
                out_path = output_root / f"{sample_id}.pt"
                if out_path.exists() and not args.force:
                    if args.skip_existing_fast:
                        written += 1
                        continue
                    tensor = torch.load(out_path, map_location="cpu")
                    if isinstance(tensor, dict):
                        tensor = tensor.get("tokens", tensor.get("visual_tokens", tensor.get("features")))
                    stat = out_path.stat()
                    manifest_out.write(json.dumps({"sample_id": sample_id, "split": row.get("split"), "path": str(out_path), "shape": list(tensor.shape), "dtype": str(tensor.dtype), "source": str(clean_image), "model_family": args.model_family, "prompt": args.prompt, "supervision": "clean_native_visual_pathway_no_qa", "checksum": None if args.skip_checksum else sha256_file(out_path), "size": stat.st_size}, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                else:
                    pending.append((row, clean_image, out_path))
            if not pending:
                continue
            output_root.mkdir(parents=True, exist_ok=True)
            images = [Image.open(clean_image).convert("RGB") for _, clean_image, _ in pending]
            try:
                tensors = extract_token_batch(args.model_family, processor, model, [clean_image for _, clean_image, _ in pending], images, device, dtype, args.prompt)
            finally:
                for image in images:
                    image.close()
            for (row, clean_image, out_path), tensor in zip(pending, tensors, strict=True):
                sample_id = str(row["sample_id"])
                if args.no_save:
                    written += 1
                    continue
                save_tokens_atomic({"tokens": tensor, "sample_id": sample_id, "source": str(clean_image), "model_family": args.model_family, "supervision": "clean_native_visual_pathway_no_qa"}, tensor, out_path, args.max_file_bytes)
                stat = out_path.stat()
                manifest_out.write(json.dumps({"sample_id": sample_id, "split": row.get("split"), "path": str(out_path), "shape": list(tensor.shape), "dtype": str(tensor.dtype), "source": str(clean_image), "model_family": args.model_family, "prompt": args.prompt, "supervision": "clean_native_visual_pathway_no_qa", "checksum": None if args.skip_checksum else sha256_file(out_path), "size": stat.st_size}, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
    print(json.dumps({"status": "ok", "written": written, "output_manifest": str(output_manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
