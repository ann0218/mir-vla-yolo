#!/usr/bin/env python3
"""Publish /amcl_pose from Isaac's ground-truth odometry (drop-in for AMCL).

WHY THIS EXISTS
---------------
B病房 is a copy of A病房 translated 5.0 m, so hospital_2wards.pgm is genuinely
ambiguous. A global likelihood-field search (AMCL's own sensor model,
sigma_hit=0.2) over every free cell x 72 headings scores the WRONG ward higher
than the true pose:

    L=0.9937  (-13.40, 1.00) 185deg   <- 5.01 m away, same heading
    L=0.9867  ( -8.35, 1.00) 185deg   <- the truth

Every hypothesis comes in pairs exactly 5.00 m apart. No amount of tuning fixes
that -- the wrong answer fits better. And with recovery_alpha_slow/fast both at
0.0, AMCL cannot re-inject particles to escape once it commits, so it degrades
across episodes until map->odom is ~170 deg out. That corrupts the dataset as
well as navigation, because record_episode.py samples observation.state from
/amcl_pose.

WHAT THIS DOES
--------------
mir_isaac_sim.py --publish-odom emits *ground-truth* odometry, and the odom
frame origin is the robot's spawn pose, which is the map origin -- so odom
coordinates ARE map coordinates (the same assumption capture_scan.py documents,
and which the scan-vs-map check confirms: 100% of beams land on a wall and the
likelihood at the odom pose is 0.9949).

So this node restamps /odom as /amcl_pose in the map frame. A static identity
map->odom (see gt_localization.launch.py) completes the TF chain. It keeps the
/amcl_pose name deliberately: ward_nav.py, record_episode.py, collect_episodes.py
and Nav2 then all work with no changes.

Static map->odom has a second benefit: it is valid for all time, so the
"Lookup would require extrapolation into the future" failures that motivated
mir_nav_params_slowsim.yaml cannot happen on this transform at all.
"""
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


class GroundTruthPose(Node):
    def __init__(self):
        super().__init__("gt_pose_pub")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("frame_id", "map")
        odom_topic = self.get_parameter("odom_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        # transient_local so a late subscriber (ward_nav, the recorder) still
        # gets a pose immediately instead of waiting for the next /odom -- this
        # matches the QoS real AMCL uses for /amcl_pose.
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                       history=QoSHistoryPolicy.KEEP_LAST, depth=1))
        self.create_subscription(Odometry, odom_topic, self._on_odom, SENSOR_QOS)

        # BasicNavigator.waitUntilNav2Active(localizer="amcl") -- which
        # ward_nav.py calls -- blocks on the /amcl/get_state lifecycle service
        # and hangs forever if nothing answers. Real AMCL provides it by being a
        # lifecycle node; we are not one, so serve that single call directly.
        # Absolute name, so this node keeps its own name and only borrows the
        # /amcl service namespace.
        self._state_srv = self.create_service(
            GetState, "/amcl/get_state", self._on_get_state)

        self.n = 0
        self.get_logger().info(
            f"ground-truth pose: {odom_topic} -> "
            f"{self.get_parameter('pose_topic').value} (frame {self.frame_id})")

    def _on_get_state(self, request, response):
        response.current_state = State(id=State.PRIMARY_STATE_ACTIVE,
                                       label="active")
        return response

    def _on_odom(self, msg):
        out = PoseWithCovarianceStamped()
        # keep the source stamp: the recorder pairs this with camera/scan on
        # sim time, so re-stamping with "now" would skew the alignment
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.pose = msg.pose.pose
        # ground truth, but not literally zero -- a singular covariance makes
        # some consumers (rviz, costmap plugins) unhappy
        for i in (0, 7, 14):
            out.pose.covariance[i] = 1e-6
        for i in (21, 28, 35):
            out.pose.covariance[i] = 1e-6
        self.pub.publish(out)
        self.n += 1
        if self.n % 200 == 0:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.get_logger().info(
                f"pose ({p.x:.2f}, {p.y:.2f}, {math.degrees(yaw):.1f}deg)")


def main():
    rclpy.init()
    node = GroundTruthPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
