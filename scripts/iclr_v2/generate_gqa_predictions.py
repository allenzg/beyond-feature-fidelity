#!/usr/bin/env python3
"""Generate ICLR-DUG-V2 GQA prediction JSONL from trained interface checkpoints.

This is intentionally a thin inference launcher: it loads an existing frozen
projector checkpoint, optionally applies the frozen purifier exactly as the
training config declares, injects the resulting visual tokens into the frozen
backbone, and writes rows consumable by scripts/iclr_v2/eval_gqa.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, LlamaConfig, LlamaForCausalLM, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.iclr_controls.train_projector_control import load_projector, load_purifier, resample_sequence_tokens
    from scripts.iclr_v2.build_corruptions import apply_corruption
except ModuleNotFoundError:
    from train_projector_control import load_projector, load_purifier, resample_sequence_tokens
    from build_corruptions import apply_corruption


DEFAULT_DINO = Path("models/dinov3-vitl16-pretrain-lvd1689m")
DEFAULT_LLAVA = Path("models/llava-v1.5-7b")
DEFAULT_QWEN25 = Path("models/Qwen2.5-VL-7B-Instruct")
DEFAULT_QWEN35 = Path("models/Qwen3.5-9B")
DEFAULT_ONEVISION2 = Path("models/LLaVA-OneVision-2-8B-Instruct")
QUESTION_PROMPT = "Answer the question using a single word or short phrase.\nQuestion: {question}\nAnswer:"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("features", "feature", "dino", "tokens", "target", "visual_tokens"):
            val = obj.get(key)
            if isinstance(val, torch.Tensor):
                return val.float()
    raise ValueError(f"unsupported tensor payload: {path}")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def strip_dino_tokens(model: Any, tensor: torch.Tensor, expected_tokens: int = 576, expected_dim: int = 1024) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"unexpected DINO output rank: {list(tensor.shape)}")
    special = 1 + int(getattr(model.config, "num_register_tokens", 0) or 0)
    if tensor.shape[1] == expected_tokens + special:
        tensor = tensor[:, special:, :]
    elif tensor.shape[1] == expected_tokens + 1:
        tensor = tensor[:, 1:, :]
    elif tensor.shape[1] != expected_tokens:
        raise ValueError(f"unexpected DINO token count {tensor.shape[1]}; expected {expected_tokens}")
    tensor = tensor.squeeze(0).detach().float().cpu().contiguous()
    if tuple(tensor.shape) != (expected_tokens, expected_dim):
        raise ValueError(f"unexpected DINO feature shape {list(tensor.shape)}")
    return tensor


def extract_dino_feature(dino_model: Any, dino_processor: Any, image: Image.Image, device: torch.device) -> torch.Tensor:
    inputs = dino_processor(images=[image], return_tensors="pt", size={"height": 384, "width": 384})
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = dino_model(**inputs)
    tensor = getattr(outputs, "last_hidden_state", None)
    if tensor is None:
        tensor = outputs[0]
    return strip_dino_tokens(dino_model, tensor)


def load_dino(device: torch.device, model_path: Path = DEFAULT_DINO):
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    return processor, model


def llama_config(model_path: Path) -> LlamaConfig:
    config = load_json(model_path / "config.json")
    return LlamaConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
        max_position_embeddings=config["max_position_embeddings"],
        rms_norm_eps=config["rms_norm_eps"],
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
        pad_token_id=config.get("pad_token_id", 0),
        tie_word_embeddings=config.get("tie_word_embeddings", False),
    )


def llava_prompt(question: str) -> tuple[str, str]:
    return "USER: ", "\n" + QUESTION_PROMPT.format(question=question) + "\nASSISTANT:"


def greedy_from_embeds(llm: Any, tokenizer: Any, embeds: torch.Tensor, max_new_tokens: int) -> tuple[list[int], str]:
    generated: list[int] = []
    current = embeds
    eos_ids = {x for x in [getattr(tokenizer, "eos_token_id", None), getattr(llm.config, "eos_token_id", None)] if isinstance(x, int)}
    for _ in range(max_new_tokens):
        outputs = llm(inputs_embeds=current, use_cache=False)
        next_id = int(outputs.logits[:, -1, :].float().argmax(dim=-1).item())
        if next_id in eos_ids:
            break
        generated.append(next_id)
        next_embed = llm.get_input_embeddings()(torch.tensor([[next_id]], device=current.device))
        current = torch.cat([current, next_embed], dim=1)
    return generated, tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def load_llava_model(model_path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    llm = LlamaForCausalLM.from_pretrained(model_path, config=llama_config(model_path), local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    return tokenizer, llm


def generate_llava(llm: Any, tokenizer: Any, visual_tokens: torch.Tensor, question: str, device: torch.device, max_new_tokens: int) -> str:
    prefix_text, suffix_text = llava_prompt(question)
    emb = llm.get_input_embeddings()
    prefix = emb(tokenizer(prefix_text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device))
    suffix = emb(tokenizer(suffix_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device))
    visual = visual_tokens.to(device=device, dtype=prefix.dtype).unsqueeze(0)
    _, text = greedy_from_embeds(llm, tokenizer, torch.cat([prefix, visual, suffix], dim=1), max_new_tokens)
    return text


def qwen_messages(image_path: Path, question: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": QUESTION_PROMPT.format(question=question)}]}]


def apply_vlm_chat_template(processor: Any, image_path: Path, question: str, *, enable_thinking: bool) -> str:
    messages = qwen_messages(image_path, question)
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


@contextmanager
def patched_qwen_image_features(model: Any, projected_tokens: torch.Tensor, calls: dict[str, int]):
    original = model.model.get_image_features

    def replacement(self, pixel_values, image_grid_thw=None, **kwargs):
        calls["get_image_features_calls"] = calls.get("get_image_features_calls", 0) + 1
        output = BaseModelOutputWithPooling()
        output.pooler_output = (projected_tokens.to(device=pixel_values.device, dtype=self.visual.dtype),)
        return output

    model.model.get_image_features = MethodType(replacement, model.model)
    try:
        yield
    finally:
        model.model.get_image_features = original


@contextmanager
def patched_onevision_image_features(model: Any, projected_by_row: list[torch.Tensor], calls: dict[str, int]):
    original = model.model.get_image_features

    def replacement(self, pixel_values, image_grid_thw=None, patch_positions=None, **kwargs):
        calls["get_image_features_calls"] = calls.get("get_image_features_calls", 0) + 1
        dtype = getattr(self.visual, "dtype", projected_by_row[0].dtype)
        return [tokens.to(device=pixel_values.device, dtype=dtype) for tokens in projected_by_row]

    model.model.get_image_features = MethodType(replacement, model.model)
    try:
        yield
    finally:
        model.model.get_image_features = original


def load_qwen_model(model_path: Path, device: str):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    loader = Qwen3_5ForConditionalGeneration if model_path == DEFAULT_QWEN35 else AutoModelForImageTextToText
    model = loader.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16, device_map=device, attn_implementation="eager").eval()
    return processor, model


def load_onevision_model(model_path: Path, device: str):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    return processor, model


def prepare_qwen_inputs(model: Any, processor: Any, image_path: Path, question: str) -> tuple[dict[str, Any], int, int]:
    text = apply_vlm_chat_template(processor, image_path, question, enable_thinking=False)
    image = load_rgb(image_path)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = to_device(inputs, model.device)
    prompt_len = inputs["input_ids"].shape[1]
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Qwen config lacks image_token_id")
    image_token_count = int((inputs["input_ids"] == int(image_token_id)).sum().item())
    if image_token_count <= 0:
        raise ValueError("Qwen prompt produced no image tokens")
    return inputs, prompt_len, image_token_count


def generate_qwen(model: Any, processor: Any, inputs: dict[str, Any], prompt_len: int, visual_tokens: torch.Tensor, max_new_tokens: int) -> str:
    expected = int((inputs["input_ids"] == int(model.config.image_token_id)).sum().item())
    if visual_tokens.shape[0] != expected:
        raise ValueError(f"Qwen visual token mismatch before generation: expected {expected}, got {visual_tokens.shape[0]}")
    dummy = dict(inputs)
    dummy["pixel_values"] = torch.empty_like(inputs["pixel_values"])
    calls: dict[str, int] = {}
    with torch.no_grad(), patched_qwen_image_features(model, visual_tokens, calls):
        output_ids = model.generate(**dummy, do_sample=False, max_new_tokens=max_new_tokens)
    if calls.get("get_image_features_calls", 0) < 1:
        raise RuntimeError("Qwen image-feature patch was not exercised")
    cont = output_ids[0, prompt_len:].detach().cpu()
    return processor.batch_decode(cont.unsqueeze(0), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def generate_qwen_batch(
    model: Any,
    processor: Any,
    projected_by_row: list[torch.Tensor],
    rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    texts = [apply_vlm_chat_template(processor, Path(row["clean_image"]), row["question"], enable_thinking=False) for row in rows]
    images = [load_rgb(Path(row["clean_image"])) for row in rows]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = to_device(inputs, model.device)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Qwen config lacks image_token_id")
    token_counts = (inputs["input_ids"] == int(image_token_id)).sum(dim=1).detach().cpu().tolist()
    if len(token_counts) != len(projected_by_row):
        raise ValueError("Qwen batch token-count mismatch")
    for idx, (tokens, expected) in enumerate(zip(projected_by_row, token_counts)):
        if tokens.shape[0] != int(expected):
            raise ValueError(f"Qwen visual token mismatch before generation at batch item {idx}: expected {expected}, got {tokens.shape[0]}")
    prompt_len = inputs["input_ids"].shape[1]
    dummy = dict(inputs)
    dummy["pixel_values"] = torch.empty_like(inputs["pixel_values"])
    calls: dict[str, int] = {}
    projected = torch.cat([tokens.detach() for tokens in projected_by_row], dim=0)
    with torch.no_grad(), patched_qwen_image_features(model, projected, calls):
        output_ids = model.generate(**dummy, do_sample=False, max_new_tokens=max_new_tokens)
    if calls.get("get_image_features_calls", 0) < 1:
        raise RuntimeError("Qwen image-feature patch was not exercised")
    continuations = output_ids[:, prompt_len:].detach().cpu()
    return [text.strip() for text in processor.batch_decode(continuations, skip_special_tokens=True, clean_up_tokenization_spaces=False)]


def generate_onevision_batch(
    model: Any,
    processor: Any,
    projected_by_row: list[torch.Tensor],
    rows: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    texts = [apply_vlm_chat_template(processor, Path(row["clean_image"]), row["question"], enable_thinking=False) for row in rows]
    images = [load_rgb(Path(row["clean_image"])) for row in rows]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = to_device(inputs, model.device)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("OneVision config lacks image_token_id")
    token_counts = (inputs["input_ids"] == int(image_token_id)).sum(dim=1).detach().cpu().tolist()
    if len(token_counts) != len(projected_by_row):
        raise ValueError("OneVision batch token-count mismatch")
    for idx, (tokens, expected) in enumerate(zip(projected_by_row, token_counts)):
        if tokens.shape[0] != int(expected):
            raise ValueError(f"OneVision visual token mismatch before generation at batch item {idx}: expected {expected}, got {tokens.shape[0]}")
    prompt_len = inputs["input_ids"].shape[1]
    dummy = dict(inputs)
    dummy["pixel_values"] = torch.empty_like(inputs["pixel_values"])
    calls: dict[str, int] = {}
    with torch.no_grad(), patched_onevision_image_features(model, projected_by_row, calls):
        output_ids = model.generate(**dummy, do_sample=False, max_new_tokens=max_new_tokens)
    if calls.get("get_image_features_calls", 0) < 1:
        raise RuntimeError("OneVision image-feature patch was not exercised")
    continuations = output_ids[:, prompt_len:].detach().cpu()
    return [text.strip() for text in processor.batch_decode(continuations, skip_special_tokens=True, clean_up_tokenization_spaces=False)]


def image_id_to_sample_id(image_id: str) -> str:
    return image_id if image_id.startswith("gqa_") else f"gqa_{image_id}"


def question_sample_id(row: dict[str, Any]) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return image_id_to_sample_id(str(row["image_id"]))


def stable_int(*parts: object) -> int:
    return int(hashlib.sha256("\t".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def build_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = read_jsonl(ROOT / args.questions)
    universe = {row["sample_id"]: row for row in read_jsonl(ROOT / args.eval_universe)}
    corrupt_by_sample_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in read_jsonl(ROOT / args.corrupt_manifest):
        corrupt_by_sample_group.setdefault((row["sample_id"], row["severity_group"].lower()), []).append(row)
    for value in corrupt_by_sample_group.values():
        value.sort(key=lambda row: (row["corruption_type"], int(row["severity"])))
    selected: list[dict[str, Any]] = []
    per_condition = {"clean": 0, "medium": 0, "heavy": 0}
    max_per = args.max_questions_per_condition
    condition_order = [item.strip().lower() for item in args.conditions.split(",") if item.strip()]
    for q in sorted(questions, key=lambda item: str(item["question_id"])):
        sample_id = question_sample_id(q)
        if sample_id not in universe:
            continue
        if "clean" in condition_order and (max_per is None or per_condition["clean"] < max_per):
            base = universe[sample_id]
            clean_feature = base.get("clean_feature") or base.get("feature_path") or f"cache/iclr_v2/dino/val/clean/{sample_id}.pt"
            selected.append({**q, "sample_id": sample_id, "condition": "Clean", "severity_group": "clean", "corruption_type": "clean", "severity": 0, "clean_image": base["image_path"], "clean_feature": clean_feature, "feature_kind": "clean"})
            per_condition["clean"] += 1
        for group in ("medium", "heavy"):
            if group not in condition_order or (max_per is not None and per_condition[group] >= max_per):
                continue
            choices = corrupt_by_sample_group.get((sample_id, group), [])
            if not choices:
                continue
            row = choices[stable_int(q["question_id"], group) % len(choices)]
            selected.append({**q, **row, "condition": group.capitalize(), "feature_kind": "corrupt"})
            per_condition[group] += 1
        if args.limit_total and len(selected) >= args.limit_total:
            selected = selected[: args.limit_total]
            break
        if max_per is not None and all(per_condition[c] >= max_per for c in condition_order):
            break
    return [row for idx, row in enumerate(selected) if idx % args.num_shards == args.shard_index]


def projector_checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.checkpoint:
        return ROOT / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    out_root = resolve(config.get("output_root"))
    if out_root is None:
        raise ValueError("config lacks output_root")
    out_dir = out_root / f"seed{args.seed}"
    for name in ("best.pt", "last.pt"):
        candidate = out_dir / name
        if candidate.exists():
            return candidate
    frozen = config.get("checkpoints", {}).get("frozen_projector")
    if frozen and config.get("training", {}).get("enabled") is False:
        return ROOT / frozen
    raise FileNotFoundError(f"no best.pt/last.pt under {out_dir}")


def project_feature(
    feature: torch.Tensor,
    projector: Any,
    purifier: Any | None,
    config: dict[str, Any],
    device: torch.device,
    family: str,
    target_tokens: int | None = None,
) -> torch.Tensor:
    if target_tokens is None:
        target_tokens = 160 if family == "qwen35" else 576
    if bool(config.get("training", {}).get("align_features_to_target_tokens", False)) and feature.shape[0] != target_tokens:
        feature = resample_sequence_tokens(feature, target_tokens)
    features = feature.unsqueeze(0).to(device)
    if purifier is not None:
        features = purifier(features)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        visual = projector(features)
    return visual.squeeze(0).detach()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=["llava15", "qwen25", "qwen35", "onevision2"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint", help="Optional explicit projector checkpoint; defaults to output_root/seed{seed}/best.pt or last.pt")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dino-model-path", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--questions", default="data/manifests/iclr_v2/gqa_val_questions.jsonl")
    parser.add_argument("--eval-universe", default="data/manifests/iclr_v2/evaluation_universe.jsonl")
    parser.add_argument("--corrupt-manifest", default="data/manifests/iclr_v2/val_corrupt_formal.jsonl")
    parser.add_argument("--conditions", default="clean,medium,heavy")
    parser.add_argument("--max-questions-per-condition", type=int, default=10000)
    parser.add_argument("--limit-total", type=int)
    parser.add_argument("--only-eval-id", help="Generate only the row with this eval_id after normal V2 row construction")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_json(config_path)
    checkpoint = projector_checkpoint(args, config)
    out_path = ROOT / args.output_jsonl if not Path(args.output_jsonl).is_absolute() else Path(args.output_jsonl)
    summary_path = ROOT / args.summary_json if not Path(args.summary_json).is_absolute() else Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing.add(str(row.get("eval_id")))
    elif out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {out_path}; pass --resume to append missing rows")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dino_processor, dino_model = load_dino(device, args.dino_model_path)
    projector = load_projector(checkpoint, device, config.get("model", {})).eval().requires_grad_(False)
    purifier = None
    use_purifier = str(config.get("training", {}).get("input_mode", "")).startswith("purified")
    purifier_path = resolve(config.get("checkpoints", {}).get("purifier"))
    if use_purifier:
        if purifier_path is None or not purifier_path.exists():
            raise FileNotFoundError(f"purified method requires purifier checkpoint: {purifier_path}")
        purifier = load_purifier(purifier_path, device).eval().requires_grad_(False)

    if args.model_family == "llava15":
        tokenizer, vlm = load_llava_model(args.model_path or DEFAULT_LLAVA, device)
        generate = lambda visual, row: generate_llava(vlm, tokenizer, visual, row["question"], device, args.max_new_tokens)
    elif args.model_family in {"qwen25", "qwen35"}:
        default_qwen_path = DEFAULT_QWEN25 if args.model_family == "qwen25" else DEFAULT_QWEN35
        processor, vlm = load_qwen_model(args.model_path or default_qwen_path, str(device))
        generate = None
    else:
        processor, vlm = load_onevision_model(args.model_path or DEFAULT_ONEVISION2, str(device))
        generate = None

    rows = build_eval_rows(args)
    if args.only_eval_id:
        rows = [
            row for row in rows
            if f"{row['question_id']}|{row['condition']}|{row.get('corruption_type','clean')}|{row.get('severity',0)}" == args.only_eval_id
        ]
        if len(rows) != 1:
            raise ValueError(f"--only-eval-id matched {len(rows)} rows: {args.only_eval_id}")
    start = time.time()
    written = 0
    def load_eval_feature(row: dict[str, Any]) -> torch.Tensor:
        if row["feature_kind"] == "clean":
            clean_path = resolve(row.get("clean_feature"))
            if clean_path and clean_path.exists():
                return load_tensor(clean_path)
            return extract_dino_feature(dino_model, dino_processor, load_rgb(Path(row["clean_image"])), device)
        corrupt_path = resolve(row.get("corrupt_feature"))
        if corrupt_path and corrupt_path.exists():
            return load_tensor(corrupt_path)
        image = apply_corruption(load_rgb(Path(row["clean_image"])), row["corruption_type"], row.get("corruption_params", {}), int(row.get("corruption_seed", 0)))
        return extract_dino_feature(dino_model, dino_processor, image, device)

    def make_output(row: dict[str, Any], prediction: str, checkpoint_path: Path) -> dict[str, Any]:
        output = {
            "eval_id": f"{row['question_id']}|{row['condition']}|{row.get('corruption_type','clean')}|{row.get('severity',0)}",
            "sample_id": row["sample_id"],
            "question_id": str(row["question_id"]),
            "image_id": str(row["image_id"]),
            "dataset": row.get("dataset", "GQA"),
            "question": row["question"],
            "prediction": prediction,
            "condition": row["condition"],
            "severity_group": row.get("severity_group"),
            "corruption_type": row.get("corruption_type", "clean"),
            "severity": row.get("severity", 0),
            "method": config.get("method", "unknown"),
            "seed": args.seed,
            "model_family": args.model_family,
            "checkpoint": str(checkpoint_path),
            "prompt": QUESTION_PROMPT,
        }
        if "answers" in row:
            output["answers"] = row["answers"]
        elif "answer" in row:
            output["answer"] = row["answer"]
        elif "gt" in row:
            output["gt"] = row["gt"]
        for optional_key in ("issimple", "tallyqa_split"):
            if optional_key in row:
                output[optional_key] = row[optional_key]
        return output

    with out_path.open("a", encoding="utf-8") as handle:
        pending: list[tuple[int, dict[str, Any]]] = []
        for idx, row in enumerate(rows, 1):
            eval_id = f"{row['question_id']}|{row['condition']}|{row.get('corruption_type','clean')}|{row.get('severity',0)}"
            if eval_id in existing:
                continue
            if args.model_family in {"qwen25", "qwen35", "onevision2"}:
                pending.append((idx, row))
                if len(pending) < args.inference_batch_size:
                    continue
                batch_rows = [item[1] for item in pending]
                texts = [apply_vlm_chat_template(processor, Path(batch_row["clean_image"]), batch_row["question"], enable_thinking=False) for batch_row in batch_rows]
                images = [load_rgb(Path(batch_row["clean_image"])) for batch_row in batch_rows]
                batch_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
                batch_inputs = to_device(batch_inputs, vlm.device)
                token_counts = (batch_inputs["input_ids"] == int(vlm.config.image_token_id)).sum(dim=1).detach().cpu().tolist()
                visuals = [project_feature(load_eval_feature(batch_row), projector, purifier, config, device, args.model_family, target_tokens=int(target_count)) for batch_row, target_count in zip(batch_rows, token_counts)]
                predictions = generate_qwen_batch(vlm, processor, visuals, batch_rows, args.max_new_tokens) if args.model_family in {"qwen25", "qwen35"} else generate_onevision_batch(vlm, processor, visuals, batch_rows, args.max_new_tokens)
                for (_, output_row), prediction in zip(pending, predictions):
                    handle.write(json.dumps(make_output(output_row, prediction, checkpoint), ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                pending.clear()
            else:
                feature = load_eval_feature(row)
                visual_tokens = project_feature(feature, projector, purifier, config, device, args.model_family)
                prediction = generate(visual_tokens, row)
                handle.write(json.dumps(make_output(row, prediction, checkpoint), ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
            if idx % 100 == 0:
                handle.flush()
                print(json.dumps({"event": "progress", "rows_seen": idx, "written": written, "elapsed_s": time.time() - start}, sort_keys=True), flush=True)
        if args.model_family in {"qwen25", "qwen35", "onevision2"} and pending:
            batch_rows = [item[1] for item in pending]
            texts = [apply_vlm_chat_template(processor, Path(batch_row["clean_image"]), batch_row["question"], enable_thinking=False) for batch_row in batch_rows]
            images = [load_rgb(Path(batch_row["clean_image"])) for batch_row in batch_rows]
            batch_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            batch_inputs = to_device(batch_inputs, vlm.device)
            token_counts = (batch_inputs["input_ids"] == int(vlm.config.image_token_id)).sum(dim=1).detach().cpu().tolist()
            visuals = [project_feature(load_eval_feature(batch_row), projector, purifier, config, device, args.model_family, target_tokens=int(target_count)) for batch_row, target_count in zip(batch_rows, token_counts)]
            predictions = generate_qwen_batch(vlm, processor, visuals, batch_rows, args.max_new_tokens) if args.model_family in {"qwen25", "qwen35"} else generate_onevision_batch(vlm, processor, visuals, batch_rows, args.max_new_tokens)
            for (_, output_row), prediction in zip(pending, predictions):
                handle.write(json.dumps(make_output(output_row, prediction, checkpoint), ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
    summary = {
        "status": "complete",
        "model_family": args.model_family,
        "method": config.get("method"),
        "seed": args.seed,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "output_jsonl": str(out_path),
        "rows_target_shard": len(rows),
        "rows_written_this_run": written,
        "rows_existing_skipped": len(existing),
        "conditions": args.conditions,
        "max_questions_per_condition": args.max_questions_per_condition,
        "limit_total": args.limit_total,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_new_tokens": args.max_new_tokens,
        "decoding": {"do_sample": False, "temperature": 0.0},
        "runtime_s": time.time() - start,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
