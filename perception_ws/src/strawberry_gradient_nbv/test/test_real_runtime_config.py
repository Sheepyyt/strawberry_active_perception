from pathlib import Path


def test_real_runtime_consumes_only_pose_corrected_observation_topic():
    config = (
        Path(__file__).parents[1] / "config" / "real_nbv.yaml"
    ).read_text(encoding="utf-8")
    assert (
        "observation_topic: /strawberry/perception/real_nbv_observation"
        in config
    )
    assert "observation_topic: /strawberry/perception/observation\n" not in config
