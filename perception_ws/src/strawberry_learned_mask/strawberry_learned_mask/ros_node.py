"""ROS 2 adapter that replaces only Observation.target_mask."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from strawberry_perception_interfaces.msg import Observation

from .provider import choose_instance_mask, decode_depth_32fc1, decode_rgb8
from .ultralytics_backend import UltralyticsBackend


DEFAULT_CHECKPOINT_SHA256 = (
    "7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357"
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _stamp_ns(message: Observation) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


class LearnedMaskNode(Node):
    """Inference-only node with one Observation input and one output."""

    def __init__(self, backend: Any | None = None) -> None:
        super().__init__("strawberry_learned_mask")
        self.declare_parameter(
            "checkpoint_path",
            "artifacts/models/yolo11m_strawberry_best.pt",
        )
        self.declare_parameter("checkpoint_sha256", DEFAULT_CHECKPOINT_SHA256)
        self.declare_parameter("accept_pickle_checkpoint", False)
        self.declare_parameter(
            "input_observation_topic", "/strawberry/perception/observation"
        )
        self.declare_parameter(
            "output_observation_topic",
            "/strawberry/perception/learned_observation",
        )
        self.declare_parameter("allowed_labels", ["strawberry"])
        self.declare_parameter("confidence_threshold", 0.70)
        self.declare_parameter("mask_threshold", 0.50)
        self.declare_parameter("instance_policy", "highest_confidence")
        self.declare_parameter("minimum_mask_pixels", 200)
        self.declare_parameter("minimum_valid_depth_pixels", 100)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_size", 640)
        self.declare_parameter(
            "audit_directory", "artifacts/learned_mask/runtime"
        )

        self.confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.instance_policy = str(
            self.get_parameter("instance_policy").value
        ).strip()
        self.allowed_labels = tuple(
            str(value) for value in self.get_parameter("allowed_labels").value
        )
        self.minimum_mask_pixels = int(
            self.get_parameter("minimum_mask_pixels").value
        )
        self.minimum_valid_depth_pixels = int(
            self.get_parameter("minimum_valid_depth_pixels").value
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        if self.minimum_mask_pixels < 1 or self.minimum_valid_depth_pixels < 1:
            raise ValueError("minimum pixel gates must be positive")
        self.audit_directory = Path(
            str(self.get_parameter("audit_directory").value)
        ).resolve()

        if backend is None:
            if self.get_parameter("accept_pickle_checkpoint").value is not True:
                raise RuntimeError(
                    "Ultralytics .pt is pickle-bearing; set "
                    "accept_pickle_checkpoint:=true only for the audited SHA"
                )
            backend = UltralyticsBackend(
                str(self.get_parameter("checkpoint_path").value),
                str(self.get_parameter("checkpoint_sha256").value),
                device=str(self.get_parameter("device").value),
                image_size=int(self.get_parameter("image_size").value),
            )
        self.backend = backend
        self.checkpoint_sha256 = str(
            getattr(
                backend,
                "checkpoint_sha256",
                self.get_parameter("checkpoint_sha256").value,
            )
        )

        input_topic = str(
            self.get_parameter("input_observation_topic").value
        ).strip()
        output_topic = str(
            self.get_parameter("output_observation_topic").value
        ).strip()
        if not input_topic or not output_topic or input_topic == output_topic:
            raise ValueError("input/output Observation topics must be non-empty and different")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(Observation, output_topic, qos)
        self.subscription = self.create_subscription(
            Observation, input_topic, self._on_observation, qos
        )
        self.get_logger().info(
            "YOLO11 learned mask ready: "
            f"sha={self.checkpoint_sha256[:12]}, confidence="
            f"{self.confidence_threshold:.2f}, labels={self.allowed_labels}, "
            f"input={input_topic}, output={output_topic}"
        )

    def _on_observation(self, message: Observation) -> None:
        started = time.perf_counter()
        try:
            rgb = decode_rgb8(message.color)
            depth = decode_depth_32fc1(message.depth)
            if depth.shape != rgb.shape[:2]:
                raise ValueError("canonical RGB and depth grids differ")
            predictions, backend_audit = self.backend.predict(
                rgb, self.confidence_threshold
            )
            decision = choose_instance_mask(
                predictions,
                rgb.shape[:2],
                allowed_labels=self.allowed_labels,
                confidence_threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
                policy=self.instance_policy,
            )
            mask_pixels = int(np.count_nonzero(decision.mask))
            valid_depth_pixels = int(
                np.count_nonzero(
                    (decision.mask == 255) & np.isfinite(depth) & (depth > 0.0)
                )
            )
            accepted_for_nbv = bool(
                mask_pixels >= self.minimum_mask_pixels
                and valid_depth_pixels >= self.minimum_valid_depth_pixels
            )
            output_mask = (
                decision.mask
                if accepted_for_nbv
                else np.zeros_like(decision.mask)
            )
            output = copy.deepcopy(message)
            output.target_mask.header = copy.deepcopy(output.color.header)
            output.target_mask.encoding = "mono8"
            output.target_mask.is_bigendian = 0
            output.target_mask.height, output.target_mask.width = output_mask.shape
            output.target_mask.step = int(output.target_mask.width)
            output.target_mask.data = np.ascontiguousarray(output_mask).tobytes()
            decision_label = "accepted" if accepted_for_nbv else "rejected"
            output.source_name = (
                f"{message.source_name}|mask=yolo11m:{self.checkpoint_sha256[:12]}"
                f":conf{self.confidence_threshold:.2f}:{decision_label}"
            )
            report = {
                "schema": "strawberry_yolo11_mask_observation/v1",
                "scene_id": str(message.scene_id),
                "observation_id": str(message.observation_id),
                "stamp_ns": _stamp_ns(message),
                "checkpoint_sha256": self.checkpoint_sha256,
                "confidence_threshold": self.confidence_threshold,
                "mask_threshold": self.mask_threshold,
                "allowed_labels": list(self.allowed_labels),
                "instance_policy": self.instance_policy,
                "accepted_instances": list(decision.accepted),
                "rejected_instances": list(decision.rejected),
                "mask_pixels": mask_pixels,
                "valid_depth_pixels": valid_depth_pixels,
                "minimum_mask_pixels": self.minimum_mask_pixels,
                "minimum_valid_depth_pixels": self.minimum_valid_depth_pixels,
                "accepted_for_nbv": accepted_for_nbv,
                "backend": backend_audit,
                "total_callback_ms": (time.perf_counter() - started) * 1000.0,
                "output_topic_published": True,
                "robot_control_capability": "none",
            }
            audit_name = (
                f"{str(message.scene_id).replace('/', '_')}_"
                f"{str(message.observation_id).replace('/', '_')}.json"
            )
            _write_json_atomic(self.audit_directory / audit_name, report)
            self.publisher.publish(output)
            log = self.get_logger().info if accepted_for_nbv else self.get_logger().warning
            log(
                f"learned mask {decision_label}: scene={message.scene_id}, "
                f"id={message.observation_id}, mask={mask_pixels}, "
                f"valid_depth={valid_depth_pixels}, "
                f"instances={len(decision.accepted)}"
            )
        except Exception as error:
            self.get_logger().error(
                f"learned mask rejected Observation without publishing: {error}"
            )


def main() -> int:
    rclpy.init()
    node: LearnedMaskNode | None = None
    try:
        node = LearnedMaskNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"ERROR: {error}")
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
