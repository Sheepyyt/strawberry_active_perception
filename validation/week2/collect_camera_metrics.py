#!/usr/bin/env python3
"""Collect reproducible RGB-D transport and geometry metrics without cv_bridge."""

from __future__ import annotations

import argparse
import array
import bisect
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import socket
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Sequence


SCHEMA_VERSION = 1
SUPPORTED_DEPTH_ENCODINGS = {"16UC1", "32FC1"}


def utc_now() -> str:
    """Return a JSON-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp_to_ns(stamp: Any) -> int:
    """Convert a ROS builtin_interfaces/Time-like object to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated percentile, or None for an empty input."""
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def finite_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Summarize finite scalar measurements without emitting JSON NaN values."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"min": None, "median": None, "p99": None, "max": None, "mean": None}
    return {
        "min": min(finite),
        "median": percentile(finite, 0.5),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
        "mean": sum(finite) / len(finite),
    }


def estimate_dropped_frames(stamps_ns: Sequence[int], expected_fps: float) -> int:
    """Estimate missing frames from positive stamp gaps and an explicit target rate."""
    if expected_fps <= 0.0:
        raise ValueError("expected_fps must be positive")
    period_ns = 1_000_000_000.0 / expected_fps
    dropped = 0
    for previous, current in zip(stamps_ns, stamps_ns[1:]):
        gap = current - previous
        if gap <= 0:
            continue
        represented_periods = int(math.floor(gap / period_ns + 0.5))
        dropped += max(0, represented_periods - 1)
    return dropped


class ImageStreamAccumulator:
    """Bounded-size metadata accumulator for one image stream."""

    def __init__(self) -> None:
        self.stamps_ns: list[int] = []
        self.dimensions: set[tuple[int, int]] = set()
        self.encodings: set[str] = set()
        self.frame_ids: set[str] = set()
        self.non_monotonic_count = 0

    def add(
        self,
        stamp_ns: int,
        width: int,
        height: int,
        encoding: str,
        frame_id: str,
    ) -> None:
        """Add one message's stable contract fields."""
        if self.stamps_ns and stamp_ns <= self.stamps_ns[-1]:
            self.non_monotonic_count += 1
        self.stamps_ns.append(int(stamp_ns))
        self.dimensions.add((int(width), int(height)))
        self.encodings.add(str(encoding))
        self.frame_ids.add(str(frame_id))

    def report(self, expected_fps: float, collection_elapsed_sec: float) -> dict[str, Any]:
        """Return timing, estimated-drop, and property-stability metrics."""
        positive_gaps_ms = [
            (current - previous) / 1_000_000.0
            for previous, current in zip(self.stamps_ns, self.stamps_ns[1:])
            if current > previous
        ]
        stamp_span_sec = (
            (self.stamps_ns[-1] - self.stamps_ns[0]) / 1_000_000_000.0
            if len(self.stamps_ns) >= 2
            else 0.0
        )
        sensor_rate_hz = (
            (len(self.stamps_ns) - 1) / stamp_span_sec
            if len(self.stamps_ns) >= 2 and stamp_span_sec > 0.0
            else None
        )
        dropped = estimate_dropped_frames(self.stamps_ns, expected_fps)
        expected_total = len(self.stamps_ns) + dropped
        return {
            "message_count": len(self.stamps_ns),
            "first_stamp_ns": self.stamps_ns[0] if self.stamps_ns else None,
            "last_stamp_ns": self.stamps_ns[-1] if self.stamps_ns else None,
            "stamp_span_sec": stamp_span_sec,
            "sensor_stamp_rate_hz": sensor_rate_hz,
            "wall_receive_rate_hz": (
                len(self.stamps_ns) / collection_elapsed_sec
                if collection_elapsed_sec > 0.0
                else None
            ),
            "stamp_monotonic": self.non_monotonic_count == 0,
            "non_monotonic_count": self.non_monotonic_count,
            "gap_ms": finite_summary(positive_gaps_ms),
            "expected_fps_for_drop_estimate": expected_fps,
            "estimated_dropped_frames": dropped,
            "estimated_drop_fraction": (
                dropped / expected_total if expected_total > 0 else None
            ),
            "dimensions": [
                {"width": width, "height": height}
                for width, height in sorted(self.dimensions)
            ],
            "dimensions_stable": len(self.dimensions) <= 1,
            "encodings": sorted(self.encodings),
            "encoding_stable": len(self.encodings) <= 1,
            "frame_ids": sorted(self.frame_ids),
            "frame_id_stable": len(self.frame_ids) <= 1,
        }


class CameraInfoAccumulator:
    """Track depth CameraInfo dimensions, optical frame, stamp, and K stability."""

    def __init__(self) -> None:
        self.stream = ImageStreamAccumulator()
        self.k_values: set[tuple[float, ...]] = set()
        self.invalid_k_count = 0

    def add(
        self,
        stamp_ns: int,
        width: int,
        height: int,
        frame_id: str,
        k: Iterable[float],
    ) -> None:
        """Add one CameraInfo message."""
        self.stream.add(stamp_ns, width, height, "CameraInfo", frame_id)
        values = tuple(round(float(value), 12) for value in k)
        if len(values) != 9 or not all(math.isfinite(value) for value in values):
            self.invalid_k_count += 1
        else:
            self.k_values.add(values)

    def report(self, expected_fps: float, collection_elapsed_sec: float) -> dict[str, Any]:
        """Return CameraInfo stability metrics."""
        result = self.stream.report(expected_fps, collection_elapsed_sec)
        result.pop("encodings")
        result.pop("encoding_stable")
        result["k_stable"] = len(self.k_values) <= 1 and self.invalid_k_count == 0
        result["unique_k_count"] = len(self.k_values)
        result["invalid_k_count"] = self.invalid_k_count
        result["k"] = list(next(iter(self.k_values))) if len(self.k_values) == 1 else None
        return result


@dataclass(frozen=True)
class DepthFrameAnalysis:
    """Pure analysis result for one raw depth image."""

    total_pixel_count: int
    valid_pixel_count: int
    valid_fraction: float
    plane_tested_sample_count: int
    plane_sample_count: int
    plane_sample_sum_m: float
    plane_error_sum_m: float
    plane_squared_error_sum_m2: float
    plane_frame_median_m: float | None


def _central_roi(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("central ROI fraction must be in (0, 1]")
    roi_width = max(1, min(width, int(round(width * fraction))))
    roi_height = max(1, min(height, int(round(height * fraction))))
    x0 = (width - roi_width) // 2
    y0 = (height - roi_height) // 2
    return x0, x0 + roi_width, y0, y0 + roi_height


def analyze_depth_buffer(
    *,
    data: bytes | bytearray | memoryview,
    width: int,
    height: int,
    step: int,
    encoding: str,
    is_bigendian: bool,
    depth_scale_m_per_unit: float,
    depth_min_m: float,
    depth_max_m: float,
    known_plane_distance_m: float | None,
    central_roi_fraction: float,
    max_plane_samples_per_frame: int,
) -> DepthFrameAnalysis:
    """Decode 16UC1/32FC1 bytes and compute validity and optional plane metrics."""
    normalized_encoding = str(encoding).upper()
    if normalized_encoding not in SUPPORTED_DEPTH_ENCODINGS:
        raise ValueError(f"unsupported depth encoding: {encoding!r}")
    if width <= 0 or height <= 0:
        raise ValueError("depth dimensions must be positive")
    if depth_scale_m_per_unit <= 0.0:
        raise ValueError("depth scale must be positive")
    if not 0.0 <= depth_min_m < depth_max_m:
        raise ValueError("depth range must satisfy 0 <= min < max")
    if max_plane_samples_per_frame <= 0:
        raise ValueError("max plane samples per frame must be positive")

    typecode, item_size = ("H", 2) if normalized_encoding == "16UC1" else ("f", 4)
    packed_width = width * item_size
    if step < packed_width:
        raise ValueError("depth row step is smaller than packed row width")
    required_bytes = step * height
    view = memoryview(data)
    if view.nbytes < required_bytes:
        raise ValueError(
            f"depth buffer has {view.nbytes} bytes, expected at least {required_bytes}"
        )

    use_plane = known_plane_distance_m is not None
    if use_plane and (
        not math.isfinite(float(known_plane_distance_m))
        or float(known_plane_distance_m) <= 0.0
    ):
        raise ValueError("known plane distance must be finite and positive")
    x0, x1, y0, y1 = _central_roi(width, height, central_roi_fraction)
    roi_pixels = (x1 - x0) * (y1 - y0)
    sample_stride = max(1, math.ceil(roi_pixels / max_plane_samples_per_frame))

    total = width * height
    valid = 0
    roi_ordinal = 0
    plane_values: list[float] = []
    plane_tested_samples = 0
    source_bigendian = bool(is_bigendian)
    host_bigendian = sys.byteorder == "big"

    for y in range(height):
        offset = y * step
        row = array.array(typecode)
        row.frombytes(view[offset : offset + packed_width].tobytes())
        if source_bigendian != host_bigendian:
            row.byteswap()
        for x, raw_value in enumerate(row):
            depth_m = (
                float(raw_value) * depth_scale_m_per_unit
                if normalized_encoding == "16UC1"
                else float(raw_value)
            )
            is_valid = (
                math.isfinite(depth_m)
                and depth_m > 0.0
                and depth_min_m <= depth_m <= depth_max_m
            )
            if is_valid:
                valid += 1
            if use_plane and x0 <= x < x1 and y0 <= y < y1:
                if roi_ordinal % sample_stride == 0:
                    plane_tested_samples += 1
                    if is_valid:
                        plane_values.append(depth_m)
                roi_ordinal += 1

    plane_distance = float(known_plane_distance_m) if use_plane else 0.0
    plane_error_sum = sum(value - plane_distance for value in plane_values)
    plane_squared_error_sum = sum(
        (value - plane_distance) ** 2 for value in plane_values
    )
    return DepthFrameAnalysis(
        total_pixel_count=total,
        valid_pixel_count=valid,
        valid_fraction=valid / total,
        plane_tested_sample_count=plane_tested_samples,
        plane_sample_count=len(plane_values),
        plane_sample_sum_m=sum(plane_values),
        plane_error_sum_m=plane_error_sum,
        plane_squared_error_sum_m2=plane_squared_error_sum,
        plane_frame_median_m=percentile(plane_values, 0.5),
    )


class DepthAccumulator:
    """Aggregate depth validity and optional controlled-plane measurements."""

    def __init__(self, known_plane_distance_m: float | None) -> None:
        self.known_plane_distance_m = known_plane_distance_m
        self.frame_count = 0
        self.total_pixels = 0
        self.valid_pixels = 0
        self.valid_fractions: list[float] = []
        self.decode_error_count = 0
        self.decode_errors: list[str] = []
        self.plane_tested_sample_count = 0
        self.plane_sample_count = 0
        self.plane_sample_sum_m = 0.0
        self.plane_error_sum_m = 0.0
        self.plane_squared_error_sum_m2 = 0.0
        self.plane_frame_medians_m: list[float] = []

    def add(self, analysis: DepthFrameAnalysis) -> None:
        """Add one successfully decoded frame."""
        self.frame_count += 1
        self.total_pixels += analysis.total_pixel_count
        self.valid_pixels += analysis.valid_pixel_count
        self.valid_fractions.append(analysis.valid_fraction)
        self.plane_tested_sample_count += analysis.plane_tested_sample_count
        self.plane_sample_count += analysis.plane_sample_count
        self.plane_sample_sum_m += analysis.plane_sample_sum_m
        self.plane_error_sum_m += analysis.plane_error_sum_m
        self.plane_squared_error_sum_m2 += analysis.plane_squared_error_sum_m2
        if analysis.plane_frame_median_m is not None:
            self.plane_frame_medians_m.append(analysis.plane_frame_median_m)

    def add_error(self, error: Exception) -> None:
        """Record an actionable decode error without terminating collection."""
        self.decode_error_count += 1
        if len(self.decode_errors) < 10:
            self.decode_errors.append(str(error))

    def report(self) -> dict[str, Any]:
        """Return weighted validity and optional plane-distance errors."""
        result: dict[str, Any] = {
            "decoded_frame_count": self.frame_count,
            "decode_error_count": self.decode_error_count,
            "decode_errors_first_10": self.decode_errors,
            "total_pixel_count": self.total_pixels,
            "valid_pixel_count": self.valid_pixels,
            "valid_depth_fraction_weighted": (
                self.valid_pixels / self.total_pixels if self.total_pixels else None
            ),
            "valid_depth_fraction_per_frame": finite_summary(self.valid_fractions),
            "known_plane": None,
        }
        if self.known_plane_distance_m is not None:
            median_depth = percentile(self.plane_frame_medians_m, 0.5)
            result["known_plane"] = {
                "reference_distance_m": self.known_plane_distance_m,
                "frames_with_valid_roi_samples": len(self.plane_frame_medians_m),
                "sampled_roi_pixel_count": self.plane_tested_sample_count,
                "sampled_valid_pixel_count": self.plane_sample_count,
                "sampled_valid_fraction": (
                    self.plane_sample_count / self.plane_tested_sample_count
                    if self.plane_tested_sample_count
                    else None
                ),
                "median_of_frame_medians_m": median_depth,
                "median_signed_error_m": (
                    median_depth - self.known_plane_distance_m
                    if median_depth is not None
                    else None
                ),
                "median_absolute_error_m": (
                    abs(median_depth - self.known_plane_distance_m)
                    if median_depth is not None
                    else None
                ),
                "sampled_mean_depth_m": (
                    self.plane_sample_sum_m / self.plane_sample_count
                    if self.plane_sample_count
                    else None
                ),
                "sampled_mean_signed_error_m": (
                    self.plane_error_sum_m / self.plane_sample_count
                    if self.plane_sample_count
                    else None
                ),
                "sampled_rmse_m": (
                    math.sqrt(
                        self.plane_squared_error_sum_m2 / self.plane_sample_count
                    )
                    if self.plane_sample_count
                    else None
                ),
            }
        return result


_DEPTH_WORKER_STOP = object()


class DepthAnalysisWorker:
    """Sample depth frames and decode them away from ROS transport callbacks."""

    def __init__(
        self,
        *,
        sample_every_n_frames: int,
        queue_capacity: int,
        analyze: Callable[[Any], DepthFrameAnalysis],
        accumulator: DepthAccumulator,
    ) -> None:
        if sample_every_n_frames <= 0:
            raise ValueError("sample_every_n_frames must be positive")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.sample_every_n_frames = int(sample_every_n_frames)
        self.queue_capacity = int(queue_capacity)
        self._analyze = analyze
        self._accumulator = accumulator
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_capacity)
        self._thread = threading.Thread(
            target=self._run,
            name="week2_depth_analysis",
            daemon=True,
        )
        self.transport_frame_count = 0
        self.unsampled_frame_count = 0
        self.scheduled_frame_count = 0
        self.queue_drop_count = 0
        self.processed_frame_count = 0
        self.worker_finished = False
        self._closed = False
        self._thread.start()

    def submit(self, message: Any) -> None:
        """Perform only constant-time sampling and non-blocking enqueue work."""
        if self._closed:
            raise RuntimeError("depth analysis worker is closed")
        self.transport_frame_count += 1
        if (self.transport_frame_count - 1) % self.sample_every_n_frames != 0:
            self.unsampled_frame_count += 1
            return
        self.scheduled_frame_count += 1
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            self.queue_drop_count += 1

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            try:
                if message is _DEPTH_WORKER_STOP:
                    return
                try:
                    self._accumulator.add(self._analyze(message))
                except (ValueError, OverflowError) as error:
                    self._accumulator.add_error(error)
                self.processed_frame_count += 1
            finally:
                self._queue.task_done()

    def close(self, timeout_sec: float = 30.0) -> None:
        """Drain already-scheduled samples and stop the worker with a timeout."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(_DEPTH_WORKER_STOP, timeout=timeout_sec)
        except queue.Full:
            self.worker_finished = False
            return
        self._thread.join(timeout=timeout_sec)
        self.worker_finished = not self._thread.is_alive()

    def report(self) -> dict[str, Any]:
        """Describe sampling coverage and whether analysis kept up."""
        return {
            "architecture": "bounded_background_worker",
            "transport_callback_decodes_pixels": False,
            "sample_every_n_frames": self.sample_every_n_frames,
            "queue_capacity": self.queue_capacity,
            "transport_frame_count": self.transport_frame_count,
            "unsampled_frame_count": self.unsampled_frame_count,
            "scheduled_frame_count": self.scheduled_frame_count,
            "processed_frame_count": self.processed_frame_count,
            "queue_drop_count": self.queue_drop_count,
            "worker_finished": self.worker_finished,
        }


@dataclass(frozen=True)
class TimestampPair:
    """One one-to-one color/depth association."""

    color_stamp_ns: int
    depth_stamp_ns: int

    @property
    def signed_skew_ns(self) -> int:
        return self.color_stamp_ns - self.depth_stamp_ns


def pair_nearest_timestamps(
    color_stamps_ns: Sequence[int],
    depth_stamps_ns: Sequence[int],
    max_skew_ns: int,
) -> list[TimestampPair]:
    """Greedily select globally closest one-to-one pairs within max_skew_ns."""
    if max_skew_ns < 0:
        raise ValueError("max_skew_ns must be non-negative")
    colors = sorted(int(value) for value in color_stamps_ns)
    depths = sorted(int(value) for value in depth_stamps_ns)
    candidates: list[tuple[int, int, int]] = []
    for depth_index, depth_stamp in enumerate(depths):
        first = bisect.bisect_left(colors, depth_stamp - max_skew_ns)
        last = bisect.bisect_right(colors, depth_stamp + max_skew_ns)
        for color_index in range(first, last):
            candidates.append(
                (abs(colors[color_index] - depth_stamp), color_index, depth_index)
            )
    candidates.sort()
    used_colors: set[int] = set()
    used_depths: set[int] = set()
    pairs: list[TimestampPair] = []
    for _, color_index, depth_index in candidates:
        if color_index in used_colors or depth_index in used_depths:
            continue
        used_colors.add(color_index)
        used_depths.add(depth_index)
        pairs.append(TimestampPair(colors[color_index], depths[depth_index]))
    pairs.sort(key=lambda pair: (pair.depth_stamp_ns, pair.color_stamp_ns))
    return pairs


def pairing_report(
    color_stamps_ns: Sequence[int],
    depth_stamps_ns: Sequence[int],
    max_skew_ms: float,
) -> dict[str, Any]:
    """Report nearest one-to-one RGB-D pairing and skew percentiles."""
    pairs = pair_nearest_timestamps(
        color_stamps_ns,
        depth_stamps_ns,
        int(round(max_skew_ms * 1_000_000.0)),
    )
    signed_skews_ms = [pair.signed_skew_ns / 1_000_000.0 for pair in pairs]
    absolute_skews_ms = [abs(value) for value in signed_skews_ms]
    return {
        "algorithm": "global_closest_greedy_one_to_one",
        "max_allowed_skew_ms": max_skew_ms,
        "matched_pair_count": len(pairs),
        "unmatched_color_count": len(color_stamps_ns) - len(pairs),
        "unmatched_depth_count": len(depth_stamps_ns) - len(pairs),
        "absolute_skew_ms": finite_summary(absolute_skews_ms),
        "signed_skew_ms": finite_summary(signed_skews_ms),
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Atomically write strict JSON, creating only the requested artifact parent."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.duration_sec <= 0.0:
        raise ValueError("--duration-sec must be positive")
    if arguments.expected_fps <= 0.0:
        raise ValueError("--expected-fps must be positive")
    if arguments.max_pair_skew_ms < 0.0:
        raise ValueError("--max-pair-skew-ms must be non-negative")
    if not 0.0 <= arguments.depth_min_m < arguments.depth_max_m:
        raise ValueError("depth range must satisfy 0 <= min < max")
    if arguments.depth_scale_m_per_unit <= 0.0:
        raise ValueError("--depth-scale-m-per-unit must be positive")
    if not 0.0 < arguments.central_roi_fraction <= 1.0:
        raise ValueError("--central-roi-fraction must be in (0, 1]")
    if arguments.max_plane_samples_per_frame <= 0:
        raise ValueError("--max-plane-samples-per-frame must be positive")
    if arguments.depth_analysis_every_n_frames <= 0:
        raise ValueError("--depth-analysis-every-n-frames must be positive")
    if arguments.depth_analysis_queue_capacity <= 0:
        raise ValueError("--depth-analysis-queue-capacity must be positive")
    if (
        arguments.known_plane_distance_m is not None
        and arguments.known_plane_distance_m <= 0.0
    ):
        raise ValueError("--known-plane-distance-m must be positive")


def run_ros_collection(arguments: argparse.Namespace) -> dict[str, Any]:
    """Subscribe with system rclpy and return a complete metrics document."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python imports failed; source /opt/ros/jazzy/setup.bash and use "
            "the system Python"
        ) from error

    color_stream = ImageStreamAccumulator()
    depth_stream = ImageStreamAccumulator()
    camera_info_stream = CameraInfoAccumulator()
    depth_metrics = DepthAccumulator(arguments.known_plane_distance_m)

    def analyze_depth_message(message: Any) -> DepthFrameAnalysis:
        return analyze_depth_buffer(
            data=message.data,
            width=message.width,
            height=message.height,
            step=message.step,
            encoding=message.encoding,
            is_bigendian=message.is_bigendian,
            depth_scale_m_per_unit=arguments.depth_scale_m_per_unit,
            depth_min_m=arguments.depth_min_m,
            depth_max_m=arguments.depth_max_m,
            known_plane_distance_m=arguments.known_plane_distance_m,
            central_roi_fraction=arguments.central_roi_fraction,
            max_plane_samples_per_frame=arguments.max_plane_samples_per_frame,
        )

    depth_worker = DepthAnalysisWorker(
        sample_every_n_frames=arguments.depth_analysis_every_n_frames,
        queue_capacity=arguments.depth_analysis_queue_capacity,
        analyze=analyze_depth_message,
        accumulator=depth_metrics,
    )

    class CameraMetricsNode(Node):
        def __init__(self) -> None:
            super().__init__(arguments.node_name)
            self.create_subscription(
                Image,
                arguments.color_topic,
                self._on_color,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                arguments.depth_topic,
                self._on_depth,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                arguments.camera_info_topic,
                self._on_camera_info,
                qos_profile_sensor_data,
            )

        @staticmethod
        def _on_color(message: Any) -> None:
            color_stream.add(
                stamp_to_ns(message.header.stamp),
                message.width,
                message.height,
                message.encoding,
                message.header.frame_id,
            )

        @staticmethod
        def _on_depth(message: Any) -> None:
            depth_stream.add(
                stamp_to_ns(message.header.stamp),
                message.width,
                message.height,
                message.encoding,
                message.header.frame_id,
            )
            depth_worker.submit(message)

        @staticmethod
        def _on_camera_info(message: Any) -> None:
            camera_info_stream.add(
                stamp_to_ns(message.header.stamp),
                message.width,
                message.height,
                message.header.frame_id,
                message.k,
            )

    started_utc = utc_now()
    started = time.monotonic()
    interrupted = False
    rclpy.init(args=None)
    node = CameraMetricsNode()
    try:
        while rclpy.ok():
            remaining = arguments.duration_sec - (time.monotonic() - started)
            if remaining <= 0.0:
                break
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        elapsed = time.monotonic() - started
        depth_worker.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    pairing = pairing_report(
        color_stream.stamps_ns,
        depth_stream.stamps_ns,
        arguments.max_pair_skew_ms,
    )
    enough_data = (
        len(color_stream.stamps_ns) >= 2
        and len(depth_stream.stamps_ns) >= 2
        and len(camera_info_stream.stream.stamps_ns) >= 1
        and pairing["matched_pair_count"] >= 1
    )
    contract_consistency = {
        "color_depth_dimensions_match": bool(
            color_stream.dimensions
            and depth_stream.dimensions
            and color_stream.dimensions == depth_stream.dimensions
        ),
        "depth_camera_info_dimensions_match": bool(
            depth_stream.dimensions
            and camera_info_stream.stream.dimensions
            and depth_stream.dimensions == camera_info_stream.stream.dimensions
        ),
        "registered_color_depth_frame_id_match": bool(
            color_stream.frame_ids
            and depth_stream.frame_ids
            and color_stream.frame_ids == depth_stream.frame_ids
        ),
        "depth_camera_info_frame_id_match": bool(
            depth_stream.frame_ids
            and camera_info_stream.stream.frame_ids
            and depth_stream.frame_ids == camera_info_stream.stream.frame_ids
        ),
        "color_encoding_is_rgb8": {
            value.lower() for value in color_stream.encodings
        }
        == {"rgb8"},
        "depth_encoding_supported": bool(depth_stream.encodings)
        and {
            value.upper() for value in depth_stream.encodings
        }.issubset(SUPPORTED_DEPTH_ENCODINGS),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "collect_camera_metrics.py",
        "generated_utc": utc_now(),
        "collection_started_utc": started_utc,
        "status": (
            "interrupted"
            if interrupted
            else "complete" if enough_data else "insufficient_data"
        ),
        "gate_evaluation": "not_evaluated",
        "host": {
            "hostname": socket.gethostname(),
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "python": sys.version.split()[0],
        },
        "configuration": {
            "topics": {
                "color": arguments.color_topic,
                "depth": arguments.depth_topic,
                "depth_camera_info": arguments.camera_info_topic,
            },
            "requested_duration_sec": arguments.duration_sec,
            "actual_duration_sec": elapsed,
            "expected_fps": arguments.expected_fps,
            "max_pair_skew_ms": arguments.max_pair_skew_ms,
            "depth_encoding_contract": sorted(SUPPORTED_DEPTH_ENCODINGS),
            "depth_scale_m_per_unit_for_16uc1": (
                arguments.depth_scale_m_per_unit
            ),
            "accepted_depth_range_m": [
                arguments.depth_min_m,
                arguments.depth_max_m,
            ],
            "known_plane_distance_m": arguments.known_plane_distance_m,
            "central_roi_fraction": arguments.central_roi_fraction,
            "max_plane_samples_per_frame": (
                arguments.max_plane_samples_per_frame
            ),
            "depth_analysis_every_n_frames": (
                arguments.depth_analysis_every_n_frames
            ),
            "depth_analysis_queue_capacity": (
                arguments.depth_analysis_queue_capacity
            ),
        },
        "color": color_stream.report(arguments.expected_fps, elapsed),
        "depth": depth_stream.report(arguments.expected_fps, elapsed),
        "depth_camera_info": camera_info_stream.report(
            arguments.expected_fps, elapsed
        ),
        "rgb_depth_pairing": pairing,
        "registered_contract_consistency": contract_consistency,
        "depth_quality": depth_metrics.report(),
        "depth_analysis_sampling": depth_worker.report(),
        "notes": [
            "Dropped frames are estimates derived from the explicit expected FPS; "
            "they are not device-side counters.",
            "Transport callbacks collect only message metadata. Depth payload quality "
            "is sampled and decoded by a bounded background worker to avoid callback "
            "backpressure changing the measured stream rates.",
            "No gate is auto-passed by this artifact. Apply the G0-G4 criteria in "
            "validation/week2/README.md.",
        ],
    }


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument(
        "--camera-info-topic", default="/camera/depth/camera_info"
    )
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--expected-fps", type=float, default=10.0)
    parser.add_argument("--max-pair-skew-ms", type=float, default=5.0)
    parser.add_argument("--depth-min-m", type=float, default=0.20)
    parser.add_argument("--depth-max-m", type=float, default=2.50)
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=0.001)
    parser.add_argument("--known-plane-distance-m", type=float)
    parser.add_argument("--central-roi-fraction", type=float, default=0.5)
    parser.add_argument("--max-plane-samples-per-frame", type=int, default=4096)
    parser.add_argument("--depth-analysis-every-n-frames", type=int, default=30)
    parser.add_argument("--depth-analysis-queue-capacity", type=int, default=2)
    parser.add_argument("--node-name", default="week2_camera_metrics")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    try:
        _validate_arguments(arguments)
        document = run_ros_collection(arguments)
        exit_code = 0 if document["status"] == "complete" else 2
    except Exception as error:  # Preserve a machine-readable artifact on setup failure.
        document = {
            "schema_version": SCHEMA_VERSION,
            "tool": "collect_camera_metrics.py",
            "generated_utc": utc_now(),
            "status": "error",
            "gate_evaluation": "not_evaluated",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 2
    write_json_atomic(arguments.output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
