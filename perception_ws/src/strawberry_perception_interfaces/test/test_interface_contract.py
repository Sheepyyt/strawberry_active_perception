"""Lock the public field names and status constants of the generated interfaces."""

from strawberry_perception_interfaces.action import ComputeNextView
from strawberry_perception_interfaces.msg import NextView, Observation
from strawberry_perception_interfaces.srv import (
    CaptureObservation,
    ConfigureNBV,
    ResetNBVMap,
)


def _field_names(message_type):
    return tuple(message_type.get_fields_and_field_types())


def test_observation_contract():
    assert _field_names(Observation) == (
        "header",
        "scene_id",
        "observation_id",
        "source_type",
        "source_name",
        "color",
        "depth",
        "target_mask",
        "camera_info",
        "camera_pose",
        "pose_valid",
        "valid_depth_fraction",
        "color_depth_skew_sec",
    )
    assert (
        Observation.SOURCE_UNKNOWN,
        Observation.SOURCE_REAL,
        Observation.SOURCE_OFFLINE,
        Observation.SOURCE_SYNTHETIC,
        Observation.SOURCE_REPLAY,
    ) == (0, 1, 2, 3, 4)


def test_next_view_contract():
    assert _field_names(NextView) == (
        "scene_id",
        "observation_id",
        "success",
        "code",
        "reason",
        "pose",
        "gain",
        "coverage",
        "total_voxel_count",
        "observed_voxel_count",
        "occupied_voxel_count",
        "unknown_voxel_count",
        "optimization_iterations",
        "compute_time_ms",
    )
    assert NextView.SUCCESS == 0
    assert NextView.OBSERVATION_NOT_FOUND == 11
    assert NextView.INTERNAL_ERROR == 255


def test_capture_contract():
    assert _field_names(CaptureObservation.Request) == (
        "scene_id",
        "not_before",
        "timeout",
        "discard_frames",
        "require_color",
        "require_mask",
        "require_pose",
    )
    assert _field_names(CaptureObservation.Response) == (
        "success",
        "code",
        "reason",
        "observation_id",
        "stamp",
    )
    assert CaptureObservation.Response.TIMEOUT == 32
    assert CaptureObservation.Response.CAMERA_MODEL_MISMATCH == 21


def test_configure_and_reset_contracts():
    assert _field_names(ConfigureNBV.Request) == (
        "scene_id",
        "world_frame",
        "target_center",
        "map_size",
        "target_roi_size",
        "observation_min",
        "observation_max",
        "voxel_size",
        "depth_min",
        "depth_max",
        "samples_per_ray",
        "optimization_steps",
        "max_step",
        "random_seed",
    )
    assert _field_names(ConfigureNBV.Response) == ("success", "code", "reason")
    assert _field_names(ResetNBVMap.Request) == ("scene_id",)
    assert _field_names(ResetNBVMap.Response) == ("success", "code", "reason")
    assert ResetNBVMap.Response.ALREADY_EMPTY == 1


def test_compute_next_view_action_contract():
    assert _field_names(ComputeNextView.Goal) == ("scene_id", "observation_id")
    assert _field_names(ComputeNextView.Result) == ("next_view",)
    assert _field_names(ComputeNextView.Feedback) == (
        "phase",
        "coverage",
        "elapsed",
        "iteration",
    )
    assert ComputeNextView.Feedback.PHASE_UPDATING_MAP == 2
    assert ComputeNextView.Feedback.PHASE_OPTIMIZING == 3
