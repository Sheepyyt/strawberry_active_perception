#!/usr/bin/env python3
"""Save the latest learned Observation as a visual, without any control client."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from strawberry_learned_mask.provider import decode_depth_32fc1, decode_rgb8
from strawberry_perception_interfaces.msg import Observation


def _decode_mask(image: object) -> np.ndarray:
    if str(image.encoding).casefold() != "mono8":
        raise ValueError("target mask must be mono8")
    height, width, step = int(image.height), int(image.width), int(image.step)
    raw = np.frombuffer(bytes(image.data), dtype=np.uint8)
    if height <= 0 or width <= 0 or step < width or raw.size != height * step:
        raise ValueError("target mask layout is invalid")
    return raw.reshape(height, step)[:, :width].copy()


class SnapshotSubscriber(Node):
    """One transient-local subscriber and no publisher/client/action surface."""

    def __init__(self) -> None:
        super().__init__("learned_mask_snapshot")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.observation: Observation | None = None
        self.create_subscription(
            Observation,
            "/strawberry/perception/learned_observation",
            self._callback,
            qos,
        )

    def _callback(self, message: Observation) -> None:
        self.observation = message


def capture(output_directory: Path, timeout_sec: float) -> dict[str, object]:
    rclpy.init()
    node = SnapshotSubscriber()
    deadline = time.monotonic() + timeout_sec
    try:
        while node.observation is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.observation is None:
            raise TimeoutError("no learned Observation received")
        message = node.observation
    finally:
        node.destroy_node()
        rclpy.shutdown()

    rgb = decode_rgb8(message.color)
    depth = decode_depth_32fc1(message.depth)
    mask = _decode_mask(message.target_mask)
    if not (rgb.shape[:2] == depth.shape == mask.shape):
        raise ValueError("learned Observation image grids differ")
    selected = mask > 0
    valid = selected & np.isfinite(depth) & (depth > 0.0)
    overlay = rgb.copy()
    overlay[selected] = (
        0.38 * overlay[selected] + 0.62 * np.array([0, 255, 80])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
    title = f"YOLO11 strawberry mask | pixels={np.count_nonzero(selected)}"
    cv2.rectangle(overlay_bgr, (0, 0), (rgb.shape[1], 34), (25, 29, 38), -1)
    cv2.putText(
        overlay_bgr,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    overlay_path = output_directory / "yolo11_live_overlay.png"
    mask_path = output_directory / "yolo11_live_mask.png"
    if not cv2.imwrite(str(overlay_path), overlay_bgr):
        raise RuntimeError("failed to save live overlay")
    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError("failed to save live mask")
    report = {
        "schema": "strawberry_yolo11_live_snapshot/v1",
        "control_capability": "subscriber_only_no_robot_interface",
        "robot_motion_command_count_during_prior_probe": 0,
        "scene_id": message.scene_id,
        "observation_id": message.observation_id,
        "source_name": message.source_name,
        "stamp_ns": int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec),
        "image_size": [int(message.color.width), int(message.color.height)],
        "mask_pixels": int(np.count_nonzero(selected)),
        "valid_depth_pixels_in_mask": int(np.count_nonzero(valid)),
        "valid_depth_ratio_in_mask": float(np.count_nonzero(valid) / np.count_nonzero(selected)),
        "median_target_depth_m": float(np.median(depth[valid])),
        "overlay": str(overlay_path.relative_to(Path.cwd())),
        "mask": str(mask_path.relative_to(Path.cwd())),
    }
    (output_directory / "yolo11_live_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    try:
        report = capture(args.output_directory.resolve(), args.timeout)
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
