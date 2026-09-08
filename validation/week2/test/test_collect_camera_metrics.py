"""Pure unit tests for the Week 2 camera metrics calculations."""

from __future__ import annotations

import array
import math
from pathlib import Path
import sys
import threading


WEEK2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEEK2_DIR))

import collect_camera_metrics as metrics  # noqa: E402


def _native_bytes(typecode, values):
    payload = array.array(typecode, values)
    return payload.tobytes()


def test_percentile_and_nearest_one_to_one_pairing():
    assert metrics.percentile([], 0.99) is None
    assert metrics.percentile([0.0, 10.0], 0.5) == 5.0

    pairs = metrics.pair_nearest_timestamps(
        color_stamps_ns=[100, 205, 310],
        depth_stamps_ns=[90, 200, 500],
        max_skew_ns=20,
    )
    assert [(pair.color_stamp_ns, pair.depth_stamp_ns) for pair in pairs] == [
        (100, 90),
        (205, 200),
    ]
    report = metrics.pairing_report([100, 205, 310], [90, 200, 500], 0.00002)
    assert report["matched_pair_count"] == 2
    assert report["unmatched_color_count"] == 1
    assert report["unmatched_depth_count"] == 1


def test_stream_metrics_detect_drop_and_non_monotonic_stamp():
    stream = metrics.ImageStreamAccumulator()
    for stamp in (0, 100_000_000, 300_000_000, 250_000_000):
        stream.add(stamp, 640, 400, "rgb8", "camera_color_optical_frame")
    report = stream.report(expected_fps=10.0, collection_elapsed_sec=0.4)
    assert report["estimated_dropped_frames"] == 1
    assert report["stamp_monotonic"] is False
    assert report["non_monotonic_count"] == 1
    assert report["dimensions_stable"] is True
    assert report["encoding_stable"] is True
    assert report["frame_id_stable"] is True


def test_analyze_16uc1_with_padding_scale_validity_and_plane_error():
    rows = [
        [0, 500, 1000, 4000],
        [1000, 1000, 1000, 1000],
    ]
    data = b"".join(_native_bytes("H", row) + b"\xaa\xbb" for row in rows)
    result = metrics.analyze_depth_buffer(
        data=data,
        width=4,
        height=2,
        step=10,
        encoding="16UC1",
        is_bigendian=sys.byteorder == "big",
        depth_scale_m_per_unit=0.001,
        depth_min_m=0.1,
        depth_max_m=3.0,
        known_plane_distance_m=1.0,
        central_roi_fraction=1.0,
        max_plane_samples_per_frame=100,
    )
    assert result.total_pixel_count == 8
    assert result.valid_pixel_count == 6
    assert result.valid_fraction == 0.75
    assert result.plane_tested_sample_count == 8
    assert result.plane_sample_count == 6
    assert result.plane_frame_median_m == 1.0
    assert math.isclose(result.plane_error_sum_m, -0.5)


def test_analyze_32fc1_rejects_nan_zero_and_out_of_range():
    data = _native_bytes("f", [math.nan, 0.0, 0.5, 1.0, 2.0])
    result = metrics.analyze_depth_buffer(
        data=data,
        width=5,
        height=1,
        step=20,
        encoding="32FC1",
        is_bigendian=sys.byteorder == "big",
        depth_scale_m_per_unit=0.001,
        depth_min_m=0.1,
        depth_max_m=1.5,
        known_plane_distance_m=None,
        central_roi_fraction=0.5,
        max_plane_samples_per_frame=10,
    )
    assert result.valid_pixel_count == 2
    assert result.valid_fraction == 0.4
    assert result.plane_tested_sample_count == 0
    assert result.plane_sample_count == 0


def test_camera_info_k_stability_is_explicit():
    accumulator = metrics.CameraInfoAccumulator()
    k = [500.0, 0.0, 320.0, 0.0, 500.0, 200.0, 0.0, 0.0, 1.0]
    accumulator.add(1, 640, 400, "camera_depth_optical_frame", k)
    accumulator.add(2, 640, 400, "camera_depth_optical_frame", k)
    stable = accumulator.report(expected_fps=10.0, collection_elapsed_sec=1.0)
    assert stable["k_stable"] is True
    assert stable["unique_k_count"] == 1

    changed = list(k)
    changed[0] = 501.0
    accumulator.add(3, 640, 400, "camera_depth_optical_frame", changed)
    assert accumulator.report(10.0, 1.0)["k_stable"] is False


def test_depth_analysis_is_sampled_on_background_worker_not_transport_thread():
    caller_thread = threading.get_ident()
    analysis_threads = []
    accumulator = metrics.DepthAccumulator(known_plane_distance_m=None)

    def analyze(message):
        analysis_threads.append(threading.get_ident())
        return metrics.DepthFrameAnalysis(
            total_pixel_count=1,
            valid_pixel_count=1,
            valid_fraction=float(message),
            plane_tested_sample_count=0,
            plane_sample_count=0,
            plane_sample_sum_m=0.0,
            plane_error_sum_m=0.0,
            plane_squared_error_sum_m2=0.0,
            plane_frame_median_m=None,
        )

    worker = metrics.DepthAnalysisWorker(
        sample_every_n_frames=3,
        queue_capacity=4,
        analyze=analyze,
        accumulator=accumulator,
    )
    for value in range(1, 8):
        worker.submit(value / 10.0)
    worker.close()

    report = worker.report()
    assert report["transport_callback_decodes_pixels"] is False
    assert report["transport_frame_count"] == 7
    assert report["unsampled_frame_count"] == 4
    assert report["scheduled_frame_count"] == 3
    assert report["processed_frame_count"] == 3
    assert report["queue_drop_count"] == 0
    assert report["worker_finished"] is True
    assert len(analysis_threads) == 3
    assert all(thread_id != caller_thread for thread_id in analysis_threads)
    assert accumulator.frame_count == 3
