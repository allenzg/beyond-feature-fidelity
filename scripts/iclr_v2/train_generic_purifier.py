#!/usr/bin/env python3
"""Train a task-agnostic DINO feature purifier on packed clean/corrupt features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.iclr_controls.train_projector_control import DualStreamVAE, load_tensor, pad_token_tensors
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1] / "iclr_controls"))
    from train_projector_control import DualStreamVAE, load_tensor, pad_token_tensors


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass
class Row:
    sample_id: str
    split: str
    clean_feature: Path
    corrupt_feature: Path


class FeatureCache:
    def __init__(self, max_items: int = 2) -> None:
        self.max_items = max_items
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def load(self, uri: Path) -> torch.Tensor:
        text = str(uri)
        if "::" not in text:
            return load_tensor(uri)
        path_text, key = text.rsplit("::", 1)
        value = self.cache.get(path_text)
        if value is None:
            value = torch.load(path_text, map_location="cpu")
            self.cache[path_text] = value
            self.cache.move_to_end(path_text)
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        features = value.get("features") if isinstance(value, dict) else None
        if not isinstance(features, dict) or key not in features:
            raise KeyError(f"{path_text} lacks feature key {key}")
        tensor = features[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"feature {key} in {path_text} is not a tensor")
        return tensor.float()


class PurifierDataset(Dataset):
    def __init__(self, manifest: Path, root: Path, split: str, limit: int | None = None) -> None:
        self.root = root
        self.rows: list[Row] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("split") != split:
                    continue
                self.rows.append(Row(
                    sample_id=str(item["sample_id"]),
                    split=split,
                    clean_feature=self._resolve(item["clean_feature"]),
                    corrupt_feature=self._resolve(item["corrupt_feature"]),
                ))
                if limit is not None and len(self.rows) >= limit:
                    break
        if not self.rows:
            raise ValueError(f"no rows for split={split} in {manifest}")
        self.cache = FeatureCache(max_items=4)

    def _resolve(self, value: str) -> Path:
        if "::" in value:
            path, key = value.rsplit("::", 1)
            p = Path(path)
            p = p if p.is_absolute() else self.root / p
            return Path(str(p) + "::" + key)
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        corrupt = self.cache.load(row.corrupt_feature)
        clean = self.cache.load(row.clean_feature)
        return corrupt, clean, row.sample_id


def collate(batch):
    xs, ys, ids = zip(*batch)
    x, xmask = pad_token_tensors(xs)
    y, ymask = pad_token_tensors(ys)
    return x, y, xmask & ymask, list(ids)


def masked(pred, target, mask):
    active = mask.to(device=pred.device, dtype=torch.bool)
    return pred.float()[active], target.float()[active]


def global_mean(x, mask):
    active = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    return (x.float() * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)


def metrics(pred, target, mask):
    p, t = masked(pred, target, mask)
    mse = F.mse_loss(p, t).item()
    norm_mse = ((p - t).pow(2).sum(dim=-1).mean() / t.pow(2).sum(dim=-1).mean().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(p, t, dim=-1).mean().item()
    gcos = F.cosine_similarity(global_mean(pred, mask), global_mean(target, mask), dim=-1).mean().item()
    return {"mse": mse, "normalized_mse": norm_mse, "cosine_similarity": cos, "global_alignment": gcos}


def model_stats(model: DualStreamVAE, x: torch.Tensor):
    c_mu, c_logvar = model.dsvae.content_encoder(x)
    n_mu, n_logvar = model.dsvae.nuisance_encoder(x)
    z = torch.cat([c_mu, n_mu], dim=-1)
    delta = model.dsvae.decoder(z)
    return x + delta, c_mu, c_logvar, n_mu, n_logvar, delta


def relation_loss(pred, target, mask, n_patches: int):
    active = mask.to(device=pred.device, dtype=torch.bool)
    losses = []
    for p, t, m in zip(pred.float(), target.float(), active, strict=True):
        p = p[m]
        t = t[m]
        if p.shape[0] > n_patches:
            idx = torch.linspace(0, p.shape[0] - 1, n_patches, device=p.device).long()
            p = p[idx]
            t = t[idx]
        p = F.normalize(p, dim=-1)
        t = F.normalize(t, dim=-1)
        losses.append(F.mse_loss(p @ p.T, t @ t.T))
    return torch.stack(losses).mean() if losses else pred.new_tensor(0.0)


def kl_loss(mu, logvar):
    return (-0.5 * (1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())).mean()


def corr_sq(a, b):
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    c = a.T @ b / max(a.shape[0] - 1, 1)
    return c.pow(2).mean()


def loss_fn(model, corrupt, clean, mask, weights, relation_patches, step, cfg):
    purified, c_mu, c_logvar, n_mu, n_logvar, delta = model_stats(model, corrupt)
    p, t = masked(purified, clean, mask)
    loss = corrupt.new_tensor(0.0, dtype=torch.float32)
    parts = {}
    parts['restoration_mse'] = F.mse_loss(p, t)
    parts['restoration_cosine'] = 1.0 - F.cosine_similarity(p, t, dim=-1).mean()
    parts['restoration_smooth_l1'] = F.smooth_l1_loss(p, t)
    parts['global_cos'] = 1.0 - F.cosine_similarity(global_mean(purified, mask), global_mean(clean, mask), dim=-1).mean()
    purified_clean, *_ = model_stats(model, clean)
    pc, tc = masked(purified_clean, clean, mask)
    parts['min_clean_identity'] = F.mse_loss(pc, tc)
    parts['relation'] = relation_loss(purified, clean, mask, relation_patches)
    cycled = model(purified)
    cycle_pred, cycle_target = masked(cycled, purified.detach(), mask)
    parts['cycle'] = F.mse_loss(cycle_pred, cycle_target)
    parts['kl'] = kl_loss(c_mu, c_logvar) + float(cfg.get('kl_schedule', {}).get('nuisance_scale', 1.0)) * kl_loss(n_mu, n_logvar)
    parts['tc'] = corr_sq(c_mu, n_mu)
    parts['min_residual_l2'] = -delta.float().pow(2).mean().sqrt().clamp_max(1.0)
    for key, value in parts.items():
        loss = loss + float(weights.get(key, 0.0)) * value
    return loss, {k: float(v.detach().cpu()) for k, v in parts.items()}, purified


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None):
    model.eval()
    total = {}; n = 0
    for idx, (corrupt, clean, mask, _) in enumerate(loader):
        corrupt = corrupt.to(device); clean = clean.to(device); mask = mask.to(device)
        pred = model(corrupt)
        m = metrics(pred, clean, mask)
        bs = corrupt.shape[0]
        for k,v in m.items(): total[k] = total.get(k, 0.0) + v * bs
        n += bs
        if max_batches is not None and idx + 1 >= max_batches: break
    return {k: v / max(n, 1) for k, v in total.items()} | {"n_eval": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--limit-samples', type=int)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    root = repo_root()
    cfg_path = resolve(root, args.config)
    cfg = read_json(cfg_path)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    manifest = resolve(root, cfg['data']['manifest'])
    out_root = resolve(root, cfg['output_root']) / f'seed{args.seed}'
    out_root.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps or cfg['curriculum']['total_max_steps'])
    batch_size = int(args.batch_size or cfg['training']['per_gpu_batch_size'])
    train_set = PurifierDataset(manifest, root, 'train', args.limit_samples)
    val_set = PurifierDataset(manifest, root, 'val', min(args.limit_samples, 1024) if args.limit_samples else None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(cfg['training'].get('num_workers', 4)), collate_fn=collate, pin_memory=True, persistent_workers=bool(cfg['training'].get('persistent_workers', True)))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(cfg['training'].get('num_workers', 4)), collate_fn=collate, pin_memory=True, persistent_workers=bool(cfg['training'].get('persistent_workers', True)))
    model = DualStreamVAE(cfg.get('model', {})).to(device)
    opt_cfg = cfg['optim']
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg['base_lr']), weight_decay=float(opt_cfg['weight_decay']), betas=tuple(opt_cfg['betas']))
    start_step = 0; best_score = float('inf')
    if args.resume:
        ckpts = sorted(out_root.glob('step*.pt'))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device)
            model.load_state_dict(ckpt['model_state'], strict=True)
            optimizer.load_state_dict(ckpt['optim_state'])
            start_step = int(ckpt['global_step'])
            best_score = float(ckpt.get('best_score', best_score))
    meta = {"config": str(cfg_path), "seed": args.seed, "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "canonical_reference": cfg.get('canonical_reference'), "argv": os.sys.argv}
    write_json(out_root / 'run_metadata.json', meta)
    metrics_path = out_root / 'train_metrics.csv'
    append = args.resume and metrics_path.exists() and start_step > 0
    start = time.time(); step = start_step
    fieldnames = ['step','split','loss','mse','normalized_mse','cosine_similarity','global_alignment','wall_s']
    with metrics_path.open('a' if append else 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append: writer.writeheader()
        while step < max_steps:
            for corrupt, clean, mask, sample_ids in train_loader:
                model.train(); step += 1
                corrupt = corrupt.to(device); clean = clean.to(device); mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, parts, pred = loss_fn(model, corrupt, clean, mask, cfg['losses'], int(cfg['relation']['n_patches']), step, cfg)
                if not torch.isfinite(loss):
                    raise ValueError(f'non-finite loss at step {step}: {float(loss)}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg['grad_clip']))
                optimizer.step()
                if step == 1 or step % int(cfg['training']['eval_every']) == 0 or step >= max_steps:
                    train_m = metrics(pred.detach(), clean, mask)
                    val_m = evaluate(model, val_loader, device, max_batches=int(cfg['training'].get('val_max_batches', 16)))
                    row = {'step':step,'split':'val','loss':float(loss.detach().cpu()), **val_m, 'wall_s':time.time()-start}
                    writer.writerow({k: row.get(k, '') for k in fieldnames}); handle.flush()
                    print(json.dumps({'event':'train_progress','step':step,'loss':float(loss.detach().cpu()),'val':val_m,'parts':parts}, sort_keys=True), flush=True)
                    score = val_m['normalized_mse'] + (1.0 - val_m['cosine_similarity'])
                    if score < best_score:
                        best_score = score
                        torch.save({'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),'global_step':step,'seed':args.seed,'config_snapshot':cfg,'manifest_sha256':sha256_file(manifest),'val_metrics':val_m,'best_score':best_score,'saved_at':datetime.utcnow().isoformat()+'Z'}, out_root/'best.pt')
                    if step % int(cfg['training']['checkpoint_every']) == 0:
                        torch.save({'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),'global_step':step,'seed':args.seed,'config_snapshot':cfg,'manifest_sha256':sha256_file(manifest),'val_metrics':val_m,'best_score':best_score,'saved_at':datetime.utcnow().isoformat()+'Z'}, out_root/f'step{step:06d}.pt')
                if step >= max_steps:
                    break
    final = evaluate(model, val_loader, device, max_batches=None)
    payload={'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),'global_step':step,'phase':'openimages_generic','seed':args.seed,'config_snapshot':cfg,'manifest_sha256':sha256_file(manifest),'val_metrics':final,'best_score':best_score,'saved_at':datetime.utcnow().isoformat()+'Z'}
    torch.save(payload, out_root/'last.pt')
    write_json(out_root/'final_status.json', {'status':'complete','seed':args.seed,'global_step':step,'best_score':best_score,'final_metrics':final,'wall_s':time.time()-start})
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
