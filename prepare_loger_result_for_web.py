#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prepare_web_data import fit_floor_plane, rotation_from_vectors, rotation_x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a saved LoGeR .pt result into web viewer binary data.")
    parser.add_argument("--source", required=True, type=Path, help="Path to LoGeR prediction .pt file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output data directory.")
    parser.add_argument("--label", default="Model 2", help="Viewer label.")
    parser.add_argument("--source-name", default="", help="Human-readable source name.")
    parser.add_argument("--spatial-stride", default=3, type=int, help="Spatial subsampling stride.")
    return parser.parse_args()


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    pose = as_numpy(payload["camera_poses"]).astype(np.float32)
    points = as_numpy(payload["points"]).astype(np.float32)
    conf = as_numpy(payload["conf"]).astype(np.float32)
    rgb = as_numpy(payload["images"]).astype(np.float32)

    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if rgb.max() <= 1.0:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    spatial_stride = args.spatial_stride
    frames = []
    all_valid = []

    fit_points = points[:, ::spatial_stride, ::spatial_stride, :].reshape(-1, 3).astype(np.float64)
    fit_conf = conf[:, ::spatial_stride, ::spatial_stride].reshape(-1).astype(np.float64)
    floor_normal, floor_d, floor_info = fit_floor_plane(fit_points, fit_conf)
    roll_degrees = -90.0
    floor_rotation = rotation_from_vectors(floor_normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    world_roll = rotation_x(roll_degrees)
    display_rotation = world_roll @ floor_rotation
    display_translation = world_roll @ np.array([0.0, 0.0, floor_d], dtype=np.float64)
    display_transform = np.eye(4, dtype=np.float64)
    display_transform[:3, :3] = display_rotation
    display_transform[:3, 3] = display_translation
    pose_display = (display_transform[None, :, :] @ pose.astype(np.float64)).astype(np.float32)

    for frame_id in range(points.shape[0]):
        frame_points = points[frame_id, ::spatial_stride, ::spatial_stride, :].reshape(-1, 3).astype(np.float32)
        frame_conf = conf[frame_id, ::spatial_stride, ::spatial_stride].reshape(-1).astype(np.float32)
        frame_rgb = rgb[frame_id, ::spatial_stride, ::spatial_stride, :].reshape(-1, 3).astype(np.uint8)

        finite = np.isfinite(frame_points).all(axis=1) & np.isfinite(frame_conf)
        frame_points = frame_points[finite]
        frame_conf = frame_conf[finite]
        frame_rgb = frame_rgb[finite]
        frame_points = (frame_points.astype(np.float64) @ display_rotation.T + display_translation).astype(np.float32)

        points_name = f"points_{frame_id:02d}.f32"
        conf_name = f"conf_{frame_id:02d}.f32"
        rgb_name = f"rgb_{frame_id:02d}.u8"
        frame_points.tofile(args.out_dir / points_name)
        frame_conf.tofile(args.out_dir / conf_name)
        frame_rgb.tofile(args.out_dir / rgb_name)

        reliable = frame_conf > 0.1
        if reliable.any():
            all_valid.append(frame_points[reliable])

        frames.append(
            {
                "frame": frame_id,
                "sourceFrame": frame_id,
                "count": int(frame_points.shape[0]),
                "points": points_name,
                "conf": conf_name,
                "rgb": rgb_name,
                "pose": pose_display[frame_id].reshape(-1).astype(float).tolist(),
            }
        )

    if all_valid:
        stacked = np.concatenate(all_valid, axis=0)
        bounds_min = stacked.min(axis=0).astype(float).tolist()
        bounds_max = stacked.max(axis=0).astype(float).tolist()
        center = ((stacked.min(axis=0) + stacked.max(axis=0)) * 0.5).astype(float).tolist()
    else:
        bounds_min = [0, 0, 0]
        bounds_max = [0, 0, 0]
        center = [0, 0, 0]

    meta = {
        "label": args.label,
        "source": str(args.source),
        "sourceVideo": args.source_name,
        "frames": len(frames),
        "height": int(points.shape[1]),
        "width": int(points.shape[2]),
        "spatialStride": spatial_stride,
        "floorPlane": floor_info,
        "displayRollDegrees": roll_degrees,
        "displayRollAxis": "world_x",
        "displayTransform": display_transform.reshape(-1).astype(float).tolist(),
        "coordinateSystem": "Accumulated reliable pointmap is fit to a support/floor plane, rotated so the fitted floor normal is +Z, translated so that fitted floor is z=0, then map and camera poses are rolled 90 degrees around the world X axis.",
        "boundsMin": bounds_min,
        "boundsMax": bounds_max,
        "center": center,
        "frameData": frames,
    }
    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(frames)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
