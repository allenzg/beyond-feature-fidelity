#!/usr/bin/env python3
"""Train ICLR DUG projector controls without touching canonical artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def optional_sha256(root: Path, value: str | None) -> str | None:
    path = resolve_path(root, value)
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


def run_fingerprint(root: Path, config_path: Path, config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "experiment_id": config["experiment_id"],
        "method": config.get("method"),
        "config_sha256": sha256_file(config_path),
        "init_projector_sha256": optional_sha256(root, config["checkpoints"].get("init_projector")),
        "purifier_sha256": optional_sha256(root, config["checkpoints"].get("purifier")),
        "manifest_sha256": optional_sha256(root, config["data"].get("manifest")),
        "train_ids_sha256": optional_sha256(root, config["data"].get("train_ids_json")),
        "val_ids_sha256": optional_sha256(root, config["data"].get("val_ids_json")),
    }


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def reserve_output_dir(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096, output_dim: int = 4096) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(inputs)))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.trunk = nn.ModuleDict(
            {
                "fc1": nn.Linear(input_dim, hidden_dim),
                "ln1": nn.LayerNorm(hidden_dim),
                "fc2": nn.Linear(hidden_dim, hidden_dim),
                "ln2": nn.LayerNorm(hidden_dim),
            }
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.trunk["ln1"](self.trunk["fc1"](inputs)))
        hidden = F.gelu(self.trunk["ln2"](self.trunk["fc2"](hidden)))
        return self.mu_head(hidden), self.logvar_head(hidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.trunk = nn.ModuleDict(
            {
                "fc1": nn.Linear(latent_dim, hidden_dim),
                "ln1": nn.LayerNorm(hidden_dim),
                "fc2": nn.Linear(hidden_dim, hidden_dim),
                "ln2": nn.LayerNorm(hidden_dim),
            }
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.trunk["ln1"](self.trunk["fc1"](latents)))
        hidden = F.gelu(self.trunk["ln2"](self.trunk["fc2"](hidden)))
        return self.head(hidden)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.mu_head = nn.Linear(input_dim, latent_dim)
        self.logvar_head = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu_head(inputs), self.logvar_head(inputs)


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(latent_dim, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.head(latents)


class ShallowEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.ln(self.fc(inputs)))
        return self.mu_head(hidden), self.logvar_head(hidden)


class ShallowDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.head(F.gelu(self.ln(self.fc(latents))))


class DualStreamVAE(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        variant = str(config.get("capacity_variant", config.get("variant", "full"))).lower()
        input_dim = int(config.get("input_dim", 1024))
        output_dim = int(config.get("output_dim", input_dim))
        content_hidden = int(config.get("content_hidden", 1024))
        content_latent = int(config.get("content_latent", 128))
        nuisance_hidden = int(config.get("nuisance_hidden", 512))
        nuisance_latent = int(config.get("nuisance_latent", 64))
        decoder_hidden = int(config.get("decoder_hidden", 1024))
        self.dsvae = nn.Module()
        if variant == "linear":
            self.dsvae.content_encoder = LinearEncoder(input_dim, content_latent)
            self.dsvae.nuisance_encoder = LinearEncoder(input_dim, nuisance_latent)
            self.dsvae.decoder = LinearDecoder(content_latent + nuisance_latent, output_dim)
        elif variant == "shallow":
            self.dsvae.content_encoder = ShallowEncoder(input_dim, content_hidden, content_latent)
            self.dsvae.nuisance_encoder = ShallowEncoder(input_dim, nuisance_hidden, nuisance_latent)
            self.dsvae.decoder = ShallowDecoder(content_latent + nuisance_latent, decoder_hidden, output_dim)
        else:
            self.dsvae.content_encoder = Encoder(input_dim, content_hidden, content_latent)
            self.dsvae.nuisance_encoder = Encoder(input_dim, nuisance_hidden, nuisance_latent)
            self.dsvae.decoder = Decoder(content_latent + nuisance_latent, decoder_hidden, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        content_mu, _ = self.dsvae.content_encoder(inputs)
        nuisance_mu, _ = self.dsvae.nuisance_encoder(inputs)
        delta = self.dsvae.decoder(torch.cat([content_mu, nuisance_mu], dim=-1))
        return inputs + delta


def load_projector(path: Path | None, device: torch.device, fallback_config: dict[str, Any]) -> MLPProjector:
    if path is None:
        return MLPProjector(
            input_dim=int(fallback_config.get("input_dim", 1024)),
            hidden_dim=int(fallback_config.get("hidden_dim", 4096)),
            output_dim=int(fallback_config.get("output_dim", 4096)),
        ).to(device)
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", fallback_config)
    model = MLPProjector(
        input_dim=int(config.get("input_dim", 1024)),
        hidden_dim=int(config.get("hidden_dim", 4096)),
        output_dim=int(config.get("output_dim", 4096)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device)


def load_purifier(path: Path, device: torch.device) -> DualStreamVAE:
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config_snapshot", {}).get("model", {}) if isinstance(checkpoint, dict) else {}
    model = DualStreamVAE(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)


def load_tensor(path: Path) -> torch.Tensor:
    path_text = str(path)
    pack_key = None
    if "::" in path_text:
        path_text, pack_key = path_text.rsplit("::", 1)
        path = Path(path_text)
    value = torch.load(path, map_location="cpu")
    if pack_key is not None:
        if not isinstance(value, dict) or "features" not in value or pack_key not in value["features"]:
            raise KeyError(f"Packed tensor {path} lacks key {pack_key}")
        value = value["features"][pack_key]
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, dict):
        for key in ("features", "feature", "dino", "tokens", "target", "visual_tokens"):
            if key in value and isinstance(value[key], torch.Tensor):
                return value[key].float()
    raise ValueError(f"Unsupported tensor cache format: {path}")


@dataclass
class ManifestRow:
    sample_id: str
    clean_feature: Path
    corrupt_feature: Path | None
    target: Path


class TensorManifestDataset(Dataset):
    def __init__(self, manifest: Path, root: Path, split: str, input_mode: str, align_features_to_target_tokens: bool = False) -> None:
        self.root = root
        self.input_mode = input_mode
        self.align_features_to_target_tokens = align_features_to_target_tokens
        self.rows: list[ManifestRow] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if split and item.get("split", split) != split:
                    continue
                self.rows.append(
                    ManifestRow(
                        sample_id=str(item["sample_id"]),
                        clean_feature=self._path(item, "clean_feature", "clean_path", "clean_dino"),
                        corrupt_feature=self._optional_path(item, "corrupt_feature", "corrupt_path", "corrupt_dino"),
                        target=self._path(item, "target", "target_tokens", "llava_tokens", "teacher_tokens"),
                    )
                )
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {manifest}")

    def _path(self, item: dict[str, Any], *keys: str) -> Path:
        path = self._optional_path(item, *keys)
        if path is None:
            raise KeyError(f"Manifest row lacks any of {keys}")
        return path

    def _optional_path(self, item: dict[str, Any], *keys: str) -> Path | None:
        for key in keys:
            if key in item and item[key]:
                value = Path(item[key])
                return value if value.is_absolute() else self.root / value
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.rows[index]
        use_corrupt = "clean_corrupt" in self.input_mode and row.corrupt_feature is not None and random.random() < 0.5
        feature_path = row.corrupt_feature if use_corrupt else row.clean_feature
        if feature_path is None:
            feature_path = row.clean_feature
        features = load_tensor(feature_path)
        target = load_tensor(row.target)
        if self.align_features_to_target_tokens and features.ndim == 2 and target.ndim == 2 and features.shape[0] != target.shape[0]:
            features = resample_sequence_tokens(features, target.shape[0])
        return features, target, "corrupt" if use_corrupt else "clean", row.sample_id


class SyntheticDataset(Dataset):
    def __init__(self, n: int = 24, patches: int = 8) -> None:
        gen = torch.Generator().manual_seed(7)
        self.features = torch.randn(n, patches, 1024, generator=gen)
        weight = torch.randn(1024, 4096, generator=gen) / 32
        self.targets = self.features @ weight

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        return self.features[index], self.targets[index], "synthetic", f"synthetic_{index}"


def resample_sequence_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[0] == target_tokens:
        return tokens
    if tokens.ndim != 2:
        raise ValueError(f"expected rank-2 token tensor, got shape={list(tokens.shape)}")
    values = tokens.transpose(0, 1).unsqueeze(0)
    resized = F.interpolate(values, size=target_tokens, mode="linear", align_corners=False)
    return resized.squeeze(0).transpose(0, 1).contiguous()


def pad_token_tensors(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = max(tensor.shape[0] for tensor in tensors)
    dim = tensors[0].shape[1]
    if any(tensor.ndim != 2 or tensor.shape[1] != dim for tensor in tensors):
        raise ValueError(f"incompatible token tensor shapes: {[list(tensor.shape) for tensor in tensors]}")
    batch = tensors[0].new_zeros((len(tensors), max_tokens, dim))
    mask = torch.zeros((len(tensors), max_tokens), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        token_count = tensor.shape[0]
        batch[index, :token_count] = tensor
        mask[index, :token_count] = True
    return batch, mask


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, str, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, list[str]]:
    xs, ys, modes, sample_ids = zip(*batch)
    features, feature_mask = pad_token_tensors(xs)
    targets, target_mask = pad_token_tensors(ys)
    if features.shape[:2] != targets.shape[:2]:
        raise ValueError(f"feature/target token counts are not aligned: features={list(features.shape)}, targets={list(targets.shape)}")
    return features, targets, list(modes), feature_mask & target_mask, list(sample_ids)


def masked_tokens(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    active = mask.to(device=pred.device, dtype=torch.bool)
    if not bool(active.any().item()):
        raise ValueError("empty token mask in batch")
    return pred[active], target[active]


def metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    active_pred, active_target = masked_tokens(pred.float(), target.float(), mask)
    mse = F.mse_loss(active_pred, active_target).item()
    norm_mse = (active_pred - active_target).pow(2).sum(dim=-1).mean() / active_target.pow(2).sum(dim=-1).mean().clamp_min(1e-12)
    cosine = F.cosine_similarity(active_pred, active_target, dim=-1).mean().item()
    if mask is None:
        pred_global = pred.float().mean(dim=1)
        target_global = target.float().mean(dim=1)
    else:
        active = mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        counts = active.sum(dim=1).clamp_min(1.0)
        pred_global = (pred.float() * active).sum(dim=1) / counts
        target_global = (target.float() * active).sum(dim=1) / counts
    global_cos = F.cosine_similarity(pred_global, target_global, dim=-1).mean().item()
    return {"raw_mse": mse, "normalized_mse": float(norm_mse), "cosine": cosine, "cosine_loss": 1.0 - cosine, "global_alignment": 1.0 - global_cos}


def training_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    active_pred, active_target = masked_tokens(pred.float(), target.float(), mask)
    return F.mse_loss(active_pred, active_target) + (1.0 - F.cosine_similarity(active_pred, active_target, dim=-1).mean())


@torch.no_grad()
def evaluate(projector: nn.Module, purifier: nn.Module | None, loader: DataLoader, device: torch.device, use_purifier: bool) -> dict[str, float]:
    projector.eval()
    total: dict[str, float] = {}
    n = 0
    for features, targets, _, mask, _ in loader:
        features = features.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        if use_purifier:
            features = purifier(features) if purifier is not None else features
        pred = projector(features)
        batch_metrics = metrics(pred, targets, mask)
        batch_size = features.shape[0]
        for key, value in batch_metrics.items():
            total[key] = total.get(key, 0.0) + value * batch_size
        n += batch_size
    return {key: value / max(n, 1) for key, value in total.items()} | {"n_val": float(n)}


def write_metadata(out_dir: Path, config_path: Path, config: dict[str, Any], seed: int, command: str, fingerprint: dict[str, Any]) -> None:
    shutil.copy2(config_path, out_dir / "config_snapshot.json")
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (out_dir / "seed.txt").write_text(str(seed) + "\n", encoding="utf-8")
    (out_dir / "git_commit.txt").write_text(git_text(["rev-parse", "HEAD"]), encoding="utf-8")
    (out_dir / "git_diff.patch").write_text(git_text(["diff", "--", "."]), encoding="utf-8")
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": config,
        "fingerprint": fingerprint,
    }
    dump_json(out_dir / "environment.json", env)
    dump_json(out_dir / "run_fingerprint.json", fingerprint)


def validate_resume(out_dir: Path, fingerprint: dict[str, Any]) -> None:
    saved_path = out_dir / "run_fingerprint.json"
    if not saved_path.exists():
        raise FileNotFoundError(f"Cannot resume without run_fingerprint.json: {saved_path}")
    saved = load_json(saved_path)
    mismatches = {key: {"saved": saved.get(key), "current": value} for key, value in fingerprint.items() if saved.get(key) != value}
    if mismatches:
        raise ValueError(f"Refusing unsafe resume; fingerprint mismatch: {json.dumps(mismatches, sort_keys=True)}")


def latest_resume_checkpoint(out_dir: Path) -> Path | None:
    checkpoints = sorted(out_dir.glob("step*.pt"))
    return checkpoints[-1] if checkpoints else None


def guard_trainable_parameters(projector: nn.Module, purifier: nn.Module | None) -> None:
    trainable = [(name, param.numel()) for name, param in projector.named_parameters() if param.requires_grad]
    purifier_trainable = [(name, param.numel()) for name, param in purifier.named_parameters() if param.requires_grad] if purifier is not None else []
    allowed_prefixes = ("fc1.", "fc2.")
    unexpected = [name for name, _ in trainable if not name.startswith(allowed_prefixes)]
    payload = {
        "event": "trainable_parameters",
        "allowed_scope": "DINO-to-native interface/projector only",
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameter_count": int(sum(count for _, count in trainable)),
        "trainable_tensor_count": len(trainable),
        "purifier_trainable_parameter_names": [name for name, _ in purifier_trainable],
        "purifier_trainable_parameter_count": int(sum(count for _, count in purifier_trainable)),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if not trainable:
        raise ValueError("No trainable projector parameters found")
    if unexpected or purifier_trainable:
        raise ValueError(f"Unexpected trainable parameters: projector={unexpected}, purifier={[name for name, _ in purifier_trainable]}")


def grad_norm(parameters) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for param in parameters:
        if param.grad is not None:
            total += param.grad.detach().double().pow(2).sum().cpu()
    return float(total.sqrt().item())


def max_parameter_delta(before: dict[str, torch.Tensor], module: nn.Module) -> float:
    delta = 0.0
    for name, param in module.named_parameters():
        current = param.detach().cpu()
        delta = max(delta, float((current - before[name]).abs().max().item()))
    return delta


def assert_finite_metric(name: str, value: float) -> None:
    if not torch.isfinite(torch.tensor(float(value))):
        raise ValueError(f"Non-finite {name}: {value}")


def run(args: argparse.Namespace) -> int:
    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_json(config_path)
    seed = int(args.seed)
    if args.preflight:
        checks: list[dict[str, str | bool]] = []
        for label, value in (
            ("init_projector", config["checkpoints"].get("init_projector")),
            ("purifier", config["checkpoints"].get("purifier")),
            ("manifest", config["data"].get("manifest")),
            ("train_ids_json", config["data"].get("train_ids_json")),
            ("val_ids_json", config["data"].get("val_ids_json")),
            ("clean_cache_root", config["data"].get("clean_cache_root")),
            ("corrupt_cache_root", config["data"].get("corrupt_cache_root")),
            ("target_cache_root", config["data"].get("target_cache_root")),
        ):
            path = resolve_path(root, value)
            checks.append({"name": label, "path": "" if path is None else str(path), "exists": path is None or path.exists()})
        missing = [row for row in checks if not row["exists"]]
        out_dir = root / "reports" / "iclr_controls" / "preflight"
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_json(out_dir / f"{config['experiment_id']}_seed{seed}.json", {"config": str(config_path), "seed": seed, "checks": checks, "ready": not missing})
        if missing:
            print(json.dumps({"ready": False, "missing": missing}, indent=2))
            return 2
        print(json.dumps({"ready": True, "checks": checks}, indent=2))
        return 0
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = resolve_path(root, args.output_root or config["output_root"])
    assert out_root is not None
    if args.smoke_test:
        out_dir = out_root if args.output_root else out_root.with_name(f"{out_root.name}_smoke")
    else:
        out_dir = out_root / f"seed{seed}"
    fingerprint = run_fingerprint(root, config_path, config, seed)
    if args.resume and (out_dir / "final_status.json").exists():
        final_status = load_json(out_dir / "final_status.json")
        if final_status.get("status") == "complete":
            print(f"Run already complete: {out_dir}")
            return 0
    reserve_output_dir(out_dir, args.allow_overwrite or args.resume)
    if args.resume and any(out_dir.iterdir()):
        validate_resume(out_dir, fingerprint)
        with (out_dir / "resume_commands.txt").open("a", encoding="utf-8") as handle:
            handle.write(" ".join(args.argv) + "\n")
    else:
        write_metadata(out_dir, config_path, config, seed, " ".join(args.argv), fingerprint)

    training_enabled = bool(config["training"].get("enabled", True))
    init_projector = resolve_path(root, config["checkpoints"].get("init_projector"))
    if not training_enabled and init_projector is None:
        init_projector = resolve_path(root, config["checkpoints"].get("frozen_projector"))
    projector = load_projector(init_projector, device, config["model"])
    purifier = None
    use_purifier = config["training"]["input_mode"].startswith("purified")
    purifier_path = resolve_path(root, config["checkpoints"].get("purifier"))
    if use_purifier:
        if purifier_path is None:
            raise ValueError("Purified mode requires a purifier checkpoint")
        purifier = load_purifier(purifier_path, device)
    guard_trainable_parameters(projector, purifier)

    manifest = resolve_path(root, config["data"]["manifest"])
    assert manifest is not None
    if not manifest.exists():
        dump_json(out_dir / "blocked.json", {"reason": "missing_manifest", "path": str(manifest)})
        raise FileNotFoundError(f"Required manifest is missing: {manifest}")
    align_features = bool(config["training"].get("align_features_to_target_tokens", False))
    train_set = TensorManifestDataset(manifest, root, "train", config["training"]["input_mode"], align_features)
    try:
        val_set = TensorManifestDataset(manifest, root, "val", config["training"]["input_mode"], align_features)
    except ValueError:
        if not config["training"].get("val_from_train_if_missing", False):
            raise
        val_set = TensorManifestDataset(manifest, root, "train", config["training"]["input_mode"], align_features)
        val_count = int(config["training"].get("val_from_train_count", 1024))
        val_set.rows = val_set.rows[:val_count]
    if args.limit_samples is not None or args.smoke_test:
        limit_samples = max(1, int(args.limit_samples or 64))
        train_set.rows = train_set.rows[:limit_samples]
        val_set.rows = val_set.rows[: min(limit_samples, len(val_set.rows))]
    max_steps = min(int(args.max_steps or 5), 5) if args.smoke_test else int(args.max_steps or config["training"].get("max_steps", 0))

    batch_size = int(args.batch_size or config["training"]["per_gpu_batch_size"])
    grad_accum = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    num_workers = max(0, int(config["training"].get("num_workers", 0)))
    pin_memory = bool(config["training"].get("pin_memory", False)) and device.type == "cuda"
    persistent_workers = bool(config["training"].get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(config["training"].get("prefetch_factor", 2)) if num_workers > 0 else None
    loader_kwargs = {"num_workers": num_workers, "pin_memory": pin_memory, "persistent_workers": persistent_workers}
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=not args.smoke_test, collate_fn=collate, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate, **loader_kwargs)
    optim_cfg = config["optim"]
    optimizer = torch.optim.AdamW(projector.parameters(), lr=float(optim_cfg["lr"]), weight_decay=float(optim_cfg["weight_decay"]), betas=tuple(optim_cfg["betas"]))
    scaler_enabled = config["training"].get("precision") == "bf16" and device.type == "cuda"
    metrics_path = out_dir / "train_metrics.csv"
    best_loss = float("inf")
    best_step = 0
    final_train_loss: float | None = None
    start_step = 0
    initial_projector_state = {name: param.detach().cpu().clone() for name, param in projector.named_parameters()}
    smoke_checks: dict[str, Any] = {
        "enabled": bool(args.smoke_test),
        "real_v2_samples_loaded": False,
        "input_shape_pass": False,
        "target_shape_pass": False,
        "sample_ids_aligned": False,
        "only_projector_trainable": True,
        "loss_finite": False,
        "backward_pass": False,
        "projector_gradient_finite_nonzero": False,
        "frozen_modules_no_gradient": True,
        "optimizer_step_changed_projector": False,
        "checkpoint_save_pass": False,
        "checkpoint_reload_pass": False,
        "formal_seed_outputs_overwritten": False,
        "samples": len(train_set),
        "batch_size": batch_size,
        "max_steps": max_steps,
        "losses": [],
    }
    smoke_reload_features: torch.Tensor | None = None
    if not training_enabled:
        start = time.time()
        val = evaluate(projector, purifier, val_loader, device, use_purifier)
        best_loss = val["raw_mse"] + val["cosine_loss"]
        best_step = 0
        torch.save({"state_dict": projector.state_dict(), "config": config["model"], "step": 0, "seed": seed, "mode": config["training"]["input_mode"], "val_metrics": val, "run_id": config["experiment_id"], "parent_p0": str(init_projector) if init_projector else None, "parent_purifier": str(purifier_path) if purifier_path else None, "args": vars(args)}, out_dir / "best.pt")
        dump_json(out_dir / "final_status.json", {"status": "complete", "seed": seed, "wall_s": time.time() - start, "final_metrics": val, "best_loss": best_loss, "eval_only": True})
        return 0
    if args.resume and any(out_dir.glob("step*.pt")):
        checkpoint_path = latest_resume_checkpoint(out_dir)
        assert checkpoint_path is not None
        checkpoint = torch.load(checkpoint_path, map_location=device)
        required = {"state_dict", "optimizer_state", "step", "seed", "run_id", "fingerprint"}
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Refusing unsafe resume from checkpoint missing {missing}: {checkpoint_path}")
        if checkpoint["seed"] != seed or checkpoint["run_id"] != config["experiment_id"] or checkpoint["fingerprint"] != fingerprint:
            raise ValueError(f"Refusing unsafe resume from incompatible checkpoint: {checkpoint_path}")
        projector.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_step = int(checkpoint.get("best_step", start_step))
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)
    start = time.time()
    append_metrics = args.resume and metrics_path.exists() and start_step > 0
    with metrics_path.open("a" if append_metrics else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "split", "loss", "raw_mse", "normalized_mse", "cosine", "cosine_loss", "global_alignment", "n_val", "wall_s"])
        if not append_metrics:
            writer.writeheader()
        step = start_step
        while step < max_steps:
            for features, targets, modes, mask, sample_ids in train_loader:
                step += 1
                projector.train()
                if args.smoke_test and step == 1:
                    first_batch = {
                        "event": "first_batch",
                        "sample_ids": sample_ids,
                        "input_shape": list(features.shape),
                        "input_dtype": str(features.dtype),
                        "target_shape": list(targets.shape),
                        "target_dtype": str(targets.dtype),
                        "modes": modes,
                    }
                    print(json.dumps(first_batch, sort_keys=True), flush=True)
                    if features.ndim != 3 or features.shape[-1] != int(config["model"].get("input_dim", 1024)):
                        raise ValueError(f"Unexpected smoke input shape: {list(features.shape)}")
                    if targets.ndim != 3 or targets.shape[-1] != int(config["model"].get("output_dim", 4096)):
                        raise ValueError(f"Unexpected smoke target shape: {list(targets.shape)}")
                    if features.shape[:2] != targets.shape[:2]:
                        raise ValueError(f"Smoke feature/target token mismatch: features={list(features.shape)}, targets={list(targets.shape)}")
                    if config.get("model_family") == "LLaVA-1.5-7B" and (list(features.shape)[1:] != [576, 1024] or list(targets.shape)[1:] != [576, 4096]):
                        raise ValueError(f"Unexpected LLaVA smoke shapes: features={list(features.shape)}, targets={list(targets.shape)}")
                    if any(mode != "clean" for mode in modes):
                        raise ValueError(f"Smoke expected clean real rows only, got modes={modes}")
                    if len(sample_ids) != len(set(sample_ids)):
                        raise ValueError(f"Duplicate sample IDs in first smoke batch: {sample_ids}")
                    smoke_checks["real_v2_samples_loaded"] = True
                    smoke_checks["input_shape_pass"] = True
                    smoke_checks["target_shape_pass"] = True
                    smoke_checks["sample_ids_aligned"] = True
                    smoke_reload_features = features.detach().clone()
                features = features.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                if use_purifier:
                    with torch.no_grad():
                        features = purifier(features) if purifier is not None else features
                optimizer.zero_grad(set_to_none=True)
                lr = float(optimizer.param_groups[0]["lr"])
                assert_finite_metric("learning_rate", lr)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=scaler_enabled):
                    pred = projector(features)
                    loss = training_loss(pred, targets, mask) / grad_accum
                batch_metrics = metrics(pred.detach(), targets.detach(), mask)
                total_loss = float((loss.detach().float() * grad_accum).item())
                assert_finite_metric("total_loss", total_loss)
                for metric_name, metric_value in batch_metrics.items():
                    assert_finite_metric(metric_name, metric_value)
                loss.backward()
                current_grad_norm = grad_norm(projector.parameters())
                assert_finite_metric("projector_grad_norm", current_grad_norm)
                if current_grad_norm <= 0.0:
                    raise ValueError("Projector gradient norm is zero")
                if purifier is not None:
                    purifier_grads = [name for name, param in purifier.named_parameters() if param.grad is not None]
                    if purifier_grads:
                        raise ValueError(f"Frozen purifier unexpectedly has gradients: {purifier_grads[:20]}")
                nn.utils.clip_grad_norm_(projector.parameters(), float(optim_cfg["grad_clip"]))
                optimizer.step()
                final_train_loss = total_loss
                current_delta = max_parameter_delta(initial_projector_state, projector)
                smoke_checks["loss_finite"] = True
                smoke_checks["backward_pass"] = True
                smoke_checks["projector_gradient_finite_nonzero"] = True
                smoke_checks["optimizer_step_changed_projector"] = current_delta > 0.0
                smoke_checks["losses"].append(total_loss)
                if args.smoke_test:
                    print(
                        json.dumps(
                            {
                                "event": "smoke_step",
                                "step": step,
                                "total_loss": total_loss,
                                "projector_grad_norm": current_grad_norm,
                                "projector_max_delta_from_initial": current_delta,
                                "lr": lr,
                                **batch_metrics,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                if step == 1 or step % int(config["training"]["eval_every"]) == 0 or step >= max_steps:
                    val = evaluate(projector, purifier, val_loader, device, use_purifier)
                    row = {"step": step, "split": "val", "loss": val["raw_mse"] + val["cosine_loss"], **val, "wall_s": time.time() - start}
                    writer.writerow(row)
                    handle.flush()
                    if not args.smoke_test:
                        print(json.dumps({"event": "train_progress", "seed": seed, "step": step, "train_loss": final_train_loss, "val_loss": row["loss"], "best_val_loss": best_loss, "lr": lr}, sort_keys=True), flush=True)
                    if row["loss"] < best_loss:
                        best_loss = row["loss"]
                        best_step = step
                        torch.save({"state_dict": projector.state_dict(), "config": config["model"], "step": step, "seed": seed, "mode": config["training"]["input_mode"], "val_metrics": val, "run_id": config["experiment_id"], "parent_p0": str(init_projector) if init_projector else None, "parent_purifier": str(purifier_path) if purifier_path else None, "args": vars(args)}, out_dir / "best.pt")
                    if step % int(config["training"]["checkpoint_every"]) == 0 and not args.smoke_test:
                        torch.save({"state_dict": projector.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config["model"], "step": step, "seed": seed, "mode": config["training"]["input_mode"], "val_metrics": val, "best_loss": best_loss, "best_step": best_step, "run_id": config["experiment_id"], "fingerprint": fingerprint, "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}, out_dir / f"step{step:06d}.pt")
                if step >= max_steps:
                    break
    final_metrics = evaluate(projector, purifier, val_loader, device, use_purifier)
    checkpoint_payload = {"state_dict": projector.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config["model"], "step": max_steps, "seed": seed, "mode": config["training"]["input_mode"], "val_metrics": final_metrics, "best_loss": best_loss, "best_step": best_step, "run_id": config["experiment_id"], "parent_p0": str(init_projector) if init_projector else None, "parent_purifier": str(purifier_path) if purifier_path else None, "fingerprint": fingerprint, "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
    torch.save(checkpoint_payload, out_dir / "last.pt")
    smoke_checks["checkpoint_save_pass"] = (out_dir / "last.pt").is_file()
    if args.smoke_test:
        if not smoke_checks["optimizer_step_changed_projector"]:
            raise ValueError("Optimizer step did not change projector parameters")
        if smoke_reload_features is None:
            raise ValueError("Missing first smoke batch for reload check")
        reloaded = torch.load(out_dir / "last.pt", map_location=device)
        reload_config = reloaded["config"]
        fresh_projector = MLPProjector(
            input_dim=int(reload_config.get("input_dim", 1024)),
            hidden_dim=int(reload_config.get("hidden_dim", 4096)),
            output_dim=int(reload_config.get("output_dim", 4096)),
        ).to(device)
        fresh_projector.load_state_dict(reloaded["state_dict"], strict=True)
        fresh_projector.eval()
        with torch.no_grad():
            reload_pred = fresh_projector(smoke_reload_features.to(device))
        expected_output_dim = int(config["model"].get("output_dim", 4096))
        if reload_pred.ndim != 3 or reload_pred.shape[-1] != expected_output_dim or reload_pred.shape[:2] != smoke_reload_features.shape[:2]:
            raise ValueError(f"Unexpected reload forward shape: {list(reload_pred.shape)}")
        if not torch.isfinite(reload_pred).all().item():
            raise ValueError("Reload forward produced non-finite values")
        smoke_checks["checkpoint_reload_pass"] = True
        print(json.dumps({"event": "checkpoint_reload", "status": "PASS", "forward_shape": list(reload_pred.shape), "finite": True}, sort_keys=True), flush=True)
    final_status = {"status": "complete", "seed": seed, "wall_s": time.time() - start, "final_metrics": final_metrics, "best_loss": best_loss, "best_step": best_step, "final_train_loss": final_train_loss, "final_step": max_steps}
    if args.smoke_test:
        final_status["smoke_checks"] = smoke_checks
        final_status["initial_loss"] = smoke_checks["losses"][0] if smoke_checks["losses"] else None
        final_status["final_loss"] = smoke_checks["losses"][-1] if smoke_checks["losses"] else None
    dump_json(out_dir / "final_status.json", final_status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.argv = ["train_projector_control.py", *os.sys.argv[1:]]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
