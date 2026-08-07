"""MeshCat kinematic viewer for measured state, target and planned trajectory."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

import meshcat
import numpy as np
import placo
import placo_utils.visualization as visualization
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

from .models import NERO_JOINT_NAMES
from .ros_utils import ordered_joint_arrays, pose_to_matrix


class NeroMeshcatViewer(Node):
    """Display the kinematic model without pretending to be a physics simulator."""

    def __init__(self) -> None:
        super().__init__("nero_meshcat_viewer")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("joint_feedback_topic", "feedback/joint_states")
        self.declare_parameter("target_pose_topic", "target_pose")
        self.declare_parameter("planned_trajectory_topic", "planned_joint_trajectory")
        self.declare_parameter("model_tip_frame", "link7")
        self.declare_parameter("meshcat_port", 0)
        self.declare_parameter("viewer_use_collision_meshes", True)

        urdf_path = self._resolve_urdf(
            str(self.get_parameter("urdf_path").value)
        )
        port = int(self.get_parameter("meshcat_port").value)
        visualization.viewer = self._start_viewer(port)
        viewer_url = visualization.viewer.url()

        use_collision_meshes = bool(
            self.get_parameter("viewer_use_collision_meshes").value
        )
        robot_flags = int(placo.Flags.ignore_collisions)
        if use_collision_meshes:
            # The official NERO link4 DAE is not rendered reliably by the
            # browser-side Collada loader. The corresponding STL collision
            # mesh is complete and has the same URDF link transform.
            robot_flags |= int(placo.Flags.collision_as_visual)
        self._robot = placo.RobotWrapper(urdf_path, robot_flags)
        self._visualizer = visualization.robot_viz(self._robot, "nero")
        self._tip_frame = str(self.get_parameter("model_tip_frame").value)
        self._lock = threading.RLock()
        self._last_positions = np.zeros(7)
        self._show_positions(self._last_positions)

        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_feedback_topic").value),
            self._joint_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_pose_topic").value),
            self._target_callback,
            10,
        )
        self.create_subscription(
            JointTrajectory,
            str(self.get_parameter("planned_trajectory_topic").value),
            self._trajectory_callback,
            10,
        )
        self.get_logger().info(
            "MeshCat viewer started. It is a kinematic visualizer, not a physics "
            f"or environment-collision simulator. Meshes="
            f"{'solid STL' if use_collision_meshes else 'colored DAE'}. "
            f"Open {viewer_url}"
        )

    def _start_viewer(self, requested_port: int):
        """Start MeshCat, automatically recovering from a busy fixed port."""
        if requested_port < 0 or requested_port > 65535:
            raise ValueError("meshcat_port must be between 0 and 65535")
        if requested_port == 0:
            return meshcat.Visualizer()

        if not self._tcp_port_available(requested_port):
            self.get_logger().warning(
                f"MeshCat ZMQ port {requested_port} is already in use; "
                "selecting an available port automatically"
            )
            return meshcat.Visualizer()

        try:
            return meshcat.Visualizer(
                server_args=[
                    f"--zmq-url=tcp://127.0.0.1:{requested_port}"
                ]
            )
        except RuntimeError as error:
            # Another process can claim the port between the availability
            # check and server startup. Recover instead of killing the viewer.
            self.get_logger().warning(
                f"MeshCat could not use ZMQ port {requested_port} "
                f"({error}); selecting an available port automatically"
            )
            return meshcat.Visualizer()

    @staticmethod
    def _tcp_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    @staticmethod
    def _resolve_urdf(configured_path: str) -> str:
        if configured_path:
            path = Path(os.path.expanduser(configured_path)).resolve()
        else:
            share = Path(get_package_share_directory("agx_arm_description"))
            path = (
                share
                / "agx_arm_urdf"
                / "nero"
                / "urdf"
                / "nero_description.urdf"
            )
        if not path.is_file():
            raise FileNotFoundError(f"NERO URDF not found: {path}")
        return str(path)

    def _set_model_positions(self, positions) -> None:
        for name, value in zip(NERO_JOINT_NAMES, positions):
            self._robot.set_joint(name, float(value))
        self._robot.update_kinematics()

    def _show_positions(self, positions) -> None:
        with self._lock:
            self._set_model_positions(positions)
            self._visualizer.display(self._robot.state.q)
            visualization.robot_frame_viz(
                self._robot, self._tip_frame, opacity=0.8, scale=0.6
            )

    def _joint_callback(self, message: JointState) -> None:
        try:
            positions, _ = ordered_joint_arrays(message, NERO_JOINT_NAMES)
        except ValueError as error:
            self.get_logger().warning(f"Ignoring invalid joint state: {error}")
            return
        self._last_positions = positions.copy()
        self._show_positions(positions)

    def _target_callback(self, message: PoseStamped) -> None:
        try:
            target = pose_to_matrix(message.pose)
        except ValueError as error:
            self.get_logger().warning(f"Ignoring invalid target pose: {error}")
            return
        visualization.frame_viz("target", target, opacity=0.9, scale=0.8)

    def _trajectory_callback(self, message: JointTrajectory) -> None:
        if tuple(message.joint_names) != NERO_JOINT_NAMES:
            self.get_logger().warning(
                "Ignoring trajectory whose joint order is not joint1 through joint7"
            )
            return
        points = []
        with self._lock:
            for trajectory_point in message.points:
                positions = np.asarray(trajectory_point.positions, dtype=float)
                if positions.shape != (7,) or not np.all(np.isfinite(positions)):
                    self.get_logger().warning(
                        "Ignoring trajectory with incomplete or non-finite point"
                    )
                    self._set_model_positions(self._last_positions)
                    return
                self._set_model_positions(positions)
                points.append(
                    np.asarray(
                        self._robot.get_T_world_frame(self._tip_frame)[:3, 3]
                    )
                )
            self._set_model_positions(self._last_positions)
            self._visualizer.display(self._robot.state.q)
        if len(points) >= 2:
            visualization.path_viz("link7_planned", np.asarray(points), 0x00A0FF)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NeroMeshcatViewer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
