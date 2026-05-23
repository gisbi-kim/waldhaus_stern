#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


SOURCE = Path("/Users/gsk/Downloads/loger_20f_pose_points_conf_rgb/loger_20f_pose_points_conf_rgb.npz")
SOURCE_META = Path("/Users/gsk/Downloads/loger_20f_pose_points_conf_rgb/metadata.json")
OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "data"


def fit_floor_plane(points: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, float, dict]:
    reliable = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0.25)
    reliable_points = points[reliable]
    if reliable_points.shape[0] < 1000:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0, {"method": "fallback", "reason": "not enough reliable points"}

    z_cut = np.quantile(reliable_points[:, 2], 0.45)
    candidates = reliable_points[reliable_points[:, 2] <= z_cut]
    if candidates.shape[0] < 1000:
        candidates = reliable_points

    rng = np.random.default_rng(7)
    max_candidates = 120_000
    if candidates.shape[0] > max_candidates:
        candidates = candidates[rng.choice(candidates.shape[0], max_candidates, replace=False)]

    score_points = candidates
    threshold = 0.025
    best_normal = None
    best_d = 0.0
    best_score = -1.0
    best_inliers = None

    for _ in range(1800):
        tri = score_points[rng.choice(score_points.shape[0], 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        # The source reconstruction already has an approximate vertical axis. Keep
        # the floor search biased toward support planes, not walls.
        if normal[2] < 0.45:
            continue
        d = -float(normal @ tri[0])
        distances = np.abs(score_points @ normal + d)
        inliers = distances < threshold
        inlier_count = int(inliers.sum())
        if inlier_count < 500:
            continue
        signed_height = float(np.median(score_points[inliers] @ normal + d))
        score = inlier_count - 80.0 * max(0.0, signed_height)
        if score > best_score:
            best_score = score
            best_normal = normal
            best_d = d
            best_inliers = inliers

    if best_normal is None or best_inliers is None:
        floor_z = float(reliable_points[:, 2].min())
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), -floor_z, {
            "method": "fallback_min_z",
            "reason": "ransac found no stable support plane",
            "floorZSource": floor_z,
        }

    inlier_points = score_points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    d = -float(normal @ centroid)
    distances = np.abs(score_points @ normal + d)
    inlier_count = int((distances < threshold).sum())

    return normal.astype(np.float64), d, {
        "method": "ransac_low_support_plane",
        "threshold": threshold,
        "normal": normal.astype(float).tolist(),
        "d": float(d),
        "candidateCount": int(score_points.shape[0]),
        "inlierCount": inlier_count,
        "inlierRatio": float(inlier_count / score_points.shape[0]),
    }


def rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + 2 * (vx @ vx)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rotation_x(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(SOURCE)
    pose = npz["pose"].astype(np.float32)
    points = npz["points"]
    conf = npz["conf"][..., 0]
    rgb = npz["rgb"]

    with SOURCE_META.open("r", encoding="utf-8") as f:
        source_meta = json.load(f)

    spatial_stride = 3
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

        frame_points.tofile(OUT_DIR / points_name)
        frame_conf.tofile(OUT_DIR / conf_name)
        frame_rgb.tofile(OUT_DIR / rgb_name)

        reliable = frame_conf > 0.1
        if reliable.any():
            all_valid.append(frame_points[reliable])

        frames.append(
            {
                "frame": frame_id,
                "sourceFrame": source_meta.get("source_frame_indices", [frame_id])[frame_id],
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
        "source": str(SOURCE),
        "sourceVideo": source_meta.get("source_video"),
        "frames": len(frames),
        "height": int(source_meta["height"]),
        "width": int(source_meta["width"]),
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
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(frames)} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
