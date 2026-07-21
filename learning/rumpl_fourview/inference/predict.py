from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.rumpl_fourview.inference.checkpoint_loader import load_checkpoint
from learning.rumpl_fourview.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a precomputed ray-token sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-npz", required=True, help="NPZ with ray_tokens and view_mask arrays.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    model, _, _ = load_checkpoint(args.checkpoint, device=args.device)

    sample = np.load(args.input_npz)
    ray_tokens = torch.from_numpy(sample["ray_tokens"]).float().unsqueeze(0).to(args.device)
    view_mask = torch.from_numpy(sample["view_mask"].astype(np.bool_)).unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(ray_tokens, view_mask=view_mask)
    pred_3d = outputs["pred_3d"].squeeze(0).cpu().numpy()
    np.savez(output_dir / "prediction.npz", pred_3d=pred_3d)
    with open(output_dir / "prediction.json", "w", encoding="utf-8") as handle:
        json.dump({"pred_3d": pred_3d.tolist()}, handle, indent=2)
    print(f"Saved inference outputs to {output_dir}")


if __name__ == "__main__":
    main()
