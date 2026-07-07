"""Smoke-test AMASS .npz files for SMPLH mesh rendering compatibility."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import trimesh
from body_visualizer.tools.vis_tools import colors
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from pyrender import DirectionalLight, Mesh, OffscreenRenderer, PerspectiveCamera, Scene

from learning.karate_selfcal.stage_a.build_amass_yolo_dataset import (
    build_camera_rig,
    camera_pose_opengl,
    normalize_gender,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether AMASS npz files can be loaded and rendered.")
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--support-dir", default="support_data")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--render", action="store_true", help="Also render one low-resolution view per sequence.")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def body_model_paths(support_dir: Path, gender: str) -> tuple[Path, Path, str]:
    gender = gender if gender in {"male", "female", "neutral"} else "neutral"
    bm_path = support_dir / "body_models" / "smplh" / gender / "model.npz"
    dmpl_path = support_dir / "body_models" / "dmpls" / gender / "model.npz"
    if not bm_path.exists():
        gender = "neutral"
        bm_path = support_dir / "body_models" / "smplh" / gender / "model.npz"
    if not dmpl_path.exists():
        dmpl_path = support_dir / "body_models" / "dmpls" / "neutral" / "model.npz"
    return bm_path, dmpl_path, gender


def load_one_frame_payload(npz_path: Path, frame_index: int) -> tuple[dict[str, torch.Tensor], str, int, int]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["poses", "trans", "betas", "dmpls", "gender"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    num_frames = int(len(data["trans"]))
    if num_frames <= 0:
        raise ValueError("empty sequence")
    idx = min(max(int(frame_index), 0), num_frames - 1)
    poses = np.asarray(data["poses"], dtype=np.float32)[idx : idx + 1]
    trans = np.asarray(data["trans"], dtype=np.float32)[idx : idx + 1]
    dmpls = np.asarray(data["dmpls"], dtype=np.float32)[idx : idx + 1]
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(1, -1)[:, :16]
    gender = normalize_gender(data["gender"])
    if poses.shape[-1] < 156:
        raise ValueError(f"poses has too few dims: {poses.shape}")
    if dmpls.shape[-1] < 8:
        raise ValueError(f"dmpls has too few dims: {dmpls.shape}")
    payload = {
        "root_orient": torch.as_tensor(poses[:, :3], dtype=torch.float32),
        "pose_body": torch.as_tensor(poses[:, 3:66], dtype=torch.float32),
        "pose_hand": torch.as_tensor(poses[:, 66:156], dtype=torch.float32),
        "trans": torch.as_tensor(trans, dtype=torch.float32),
        "betas": torch.as_tensor(betas, dtype=torch.float32),
        "dmpls": torch.as_tensor(dmpls[:, :8], dtype=torch.float32),
    }
    return payload, gender, num_frames, idx


def main() -> None:
    args = parse_args()
    source_paths = [Path(path) for path in sorted(glob.glob(args.source_glob, recursive=True))]
    if args.max_files > 0:
        source_paths = source_paths[: args.max_files]
    if not source_paths:
        raise FileNotFoundError(args.source_glob)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    support_dir = Path(args.support_dir)
    device = resolve_device(args.device)
    bm_cache: dict[str, BodyModel] = {}
    faces_cache: dict[str, np.ndarray] = {}

    renderer = OffscreenRenderer(args.resolution, args.resolution) if args.render else None
    scene = Scene(ambient_light=[0.3, 0.3, 0.3]) if args.render else None
    start = time.time()
    results: list[dict[str, Any]] = []

    try:
        for index, npz_path in enumerate(source_paths, start=1):
            item: dict[str, Any] = {"path": str(npz_path), "ok": False}
            try:
                payload, gender, num_frames, used_frame = load_one_frame_payload(npz_path, args.frame_index)
                bm_path, dmpl_path, resolved_gender = body_model_paths(support_dir, gender)
                if resolved_gender not in bm_cache:
                    bm_cache[resolved_gender] = BodyModel(
                        bm_path=str(bm_path),
                        num_betas=16,
                        model_type="smplh",
                        batch_size=1,
                        path_dmpl=str(dmpl_path),
                    ).to(device)
                    faces_cache[resolved_gender] = c2c(bm_cache[resolved_gender].f)
                bm = bm_cache[resolved_gender]
                payload = {key: value.to(device) for key, value in payload.items()}
                with torch.no_grad():
                    body = bm(**payload)
                verts = np.asarray(c2c(body.v[0]), dtype=np.float32)
                joints = np.asarray(c2c(body.Jtr[0]), dtype=np.float32)
                if not np.isfinite(verts).all() or not np.isfinite(joints).all():
                    raise ValueError("non-finite body output")
                height = float(verts[:, 2].max() - verts[:, 2].min())
                center = verts.mean(axis=0).astype(np.float64)
                if height <= 0.2:
                    raise ValueError(f"unusual body height: {height:.4f}")
                if renderer is not None and scene is not None:
                    cameras = build_camera_rig(
                        center=center,
                        height=height,
                        width=args.resolution,
                        image_height=args.resolution,
                        fov_deg=60.0,
                        distance_scale=3.5,
                        rng=np.random.default_rng(1234),
                    )
                    cam = cameras[0]
                    scene.clear()
                    mesh = trimesh.Trimesh(
                        vertices=verts,
                        faces=faces_cache[resolved_gender],
                        vertex_colors=np.tile(colors["grey"], (verts.shape[0], 1)),
                    )
                    scene.add(Mesh.from_trimesh(mesh, smooth=False))
                    pose = camera_pose_opengl(cam["position"], cam["target"])
                    cam_node = scene.add(PerspectiveCamera(yfov=np.deg2rad(float(cam.get("fov_degree", 60.0)))), pose=pose)
                    light_node = scene.add(DirectionalLight([1.0, 1.0, 1.0], 3.0), pose=pose)
                    color, _ = renderer.render(scene)
                    scene.remove_node(cam_node)
                    scene.remove_node(light_node)
                    if color.size == 0 or not np.isfinite(color).all():
                        raise ValueError("render output invalid")
                item.update(
                    {
                        "ok": True,
                        "gender": str(gender),
                        "resolved_gender": str(resolved_gender),
                        "num_frames": int(num_frames),
                        "used_frame": int(used_frame),
                        "height_m": height,
                        "center_m": center.astype(float).tolist(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report all per-file failures.
                item["error"] = repr(exc)
            results.append(item)
            if args.log_every > 0 and (index == 1 or index % args.log_every == 0 or index == len(source_paths)):
                ok_count = sum(1 for row in results if row.get("ok"))
                print(json.dumps({"checked": index, "total": len(source_paths), "ok": ok_count, "failed": index - ok_count}, ensure_ascii=False))
    finally:
        if renderer is not None:
            renderer.delete()

    ok_count = sum(1 for row in results if row.get("ok"))
    payload = {
        "stage": "amass_renderability_smoke_test",
        "source_glob": args.source_glob,
        "num_files": len(results),
        "ok": ok_count,
        "failed": len(results) - ok_count,
        "render_checked": bool(args.render),
        "elapsed_sec": time.time() - start,
        "results": results,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "num_files": len(results), "ok": ok_count, "failed": len(results) - ok_count}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
