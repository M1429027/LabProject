from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.rumpl_fourview.datasets.amass_projector import create_synthetic_sample, load_joints_from_npz
from learning.rumpl_fourview.datasets.camera_sampler import CameraDistributionConfig, make_camera, sample_active_view_indices, sample_camera_rig
from learning.rumpl_fourview.utils import ensure_dir, load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate precomputed synthetic ray-token samples from prepared 3D joint sequences.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-rig-json", required=True)
    parser.add_argument("--source-glob", required=True, help="Glob for prepared .npz files containing joints_3d arrays.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on saved frames; 0 keeps all.")
    return parser.parse_args()


def load_base_rig(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = []
    for item in payload.get("cameras", []):
        cameras.append(
            make_camera(
                name=item["name"],
                K=np.asarray(item["K"], dtype=np.float64),
                R=np.asarray(item["R"], dtype=np.float64),
                t=np.asarray(item["t"], dtype=np.float64),
                image_size=np.asarray(item.get("image_size", [1920, 1080]), dtype=np.int32),
            )
        )
    if len(cameras) < 2:
        raise ValueError("Base rig must contain at least two cameras.")
    return cameras


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dist_cfg = CameraDistributionConfig.from_mapping(config.get("camera_distribution"))
    noise_cfg = config.get("synthetic_noise", {})
    output_dir = ensure_dir(args.output_dir)
    train_dir = ensure_dir(output_dir / "train")
    val_dir = ensure_dir(output_dir / "val")
    base_cameras = load_base_rig(args.base_rig_json)

    rng = np.random.default_rng(int(config.get("experiment", {}).get("seed", 42)))
    entries = []
    saved_count = 0
    source_paths = [Path(path) for path in sorted(glob.glob(args.source_glob))]
    for source_path in source_paths:
        joints_sequence = load_joints_from_npz(source_path)
        for frame_index, joints_3d in enumerate(joints_sequence):
            if args.max_samples and saved_count >= args.max_samples:
                break
            sampled_cameras = sample_camera_rig(base_cameras, rng=rng, distribution=dist_cfg)
            active_indices = sample_active_view_indices(len(sampled_cameras), rng=rng, min_views=2)
            sample = create_synthetic_sample(
                joints_3d=joints_3d,
                cameras=sampled_cameras,
                active_view_indices=active_indices,
                rng=rng,
                noise_cfg=noise_cfg,
            )
            split = "val" if saved_count % 10 == 0 else "train"
            split_dir = val_dir if split == "val" else train_dir
            sample_id = f"{source_path.stem}_frame_{frame_index:05d}"
            sample_path = split_dir / f"{sample_id}.npz"
            np.savez_compressed(sample_path, **sample)
            entries.append(
                {
                    "id": sample_id,
                    "split": split,
                    "path": str(sample_path),
                    "num_views": int(sample["view_mask"].sum()),
                    "num_joints": 17,
                    "feature_dim": 7,
                    "source": str(source_path),
                    "frame_index": frame_index,
                }
            )
            saved_count += 1
        if args.max_samples and saved_count >= args.max_samples:
            break

    manifest = {
        "schema_version": "rumpl_fourview_manifest_v1",
        "entries": entries,
    }
    save_json(output_dir / "manifest.json", manifest)
    print(f"Generated {saved_count} synthetic samples in {output_dir}")


if __name__ == "__main__":
    main()


