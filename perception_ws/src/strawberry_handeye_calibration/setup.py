"""Package metadata for the offline eye-in-hand calibration tools."""

from setuptools import find_packages, setup


PACKAGE_NAME = "strawberry_handeye_calibration"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        ("share/" + PACKAGE_NAME, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "scipy"],
    zip_safe=True,
    maintainer="yyt",
    maintainer_email="22313022@zju.edu.cn",
    description="Offline link7-to-camera eye-in-hand calibration core.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "handeye_calibrate = strawberry_handeye_calibration.cli:main",
            "handeye_make_fixture = strawberry_handeye_calibration.fixture:main",
            "handeye_import_npz = strawberry_handeye_calibration.npz_cli:main",
            "handeye_reprocess_camera_model = "
            "strawberry_handeye_calibration.camera_model_reprocess_cli:main",
            "handeye_validate_prospective = "
            "strawberry_handeye_calibration.prospective_cli:main",
            "handeye_validate_stability = strawberry_handeye_calibration.stability_cli:main",
        ],
    },
)
