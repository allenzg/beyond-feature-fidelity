"""Run the published recovery and readout weights on a small DINO token tensor."""
import argparse
import json
from pathlib import Path

import torch
from scripts.iclr_controls.train_projector_control import load_projector, load_purifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--family", choices=["llava15", "qwen25", "qwen35", "onevision2"], default="llava15")
    parser.add_argument("--condition", choices=["m1", "m4", "m6", "m7"], default="m6")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    family = args.family
    cfg = json.loads((root / "configs/iclr_v2" / family / f"{family}_{args.condition}_v2.json").read_text())
    ckpt = cfg["checkpoints"]
    if args.condition in {"m1", "m4"} and family != "onevision2":
        checkpoint = ckpt.get("frozen_projector") or ckpt["init_projector"]
    else:
        checkpoint = str(Path(cfg["output_root"]) / "seed123/best.pt")
    device = torch.device("cpu")
    projector = load_projector(args.weights / checkpoint, device, cfg["model"]).eval()
    torch.manual_seed(123)
    tokens = torch.randn(1, 8, 1024)
    with torch.inference_mode():
        if ckpt.get("purifier"):
            tokens = load_purifier(args.weights / ckpt["purifier"], device)(tokens)
        output = projector(tokens)
    assert output.shape == (1, 8, cfg["model"]["output_dim"]) and torch.isfinite(output).all()
    print(f"{family} {args.condition}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
