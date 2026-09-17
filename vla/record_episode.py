#!/usr/bin/env python3
"""Record one VLA episode while something else drives the robot.

Runs inside my_isaac_sim, which only has rclpy / cv_bridge / cv2 / numpy --
no pandas, pyarrow, av or ffmpeg. So this writes a raw dump (PNG frames +
one JSON) and `to_lerobot.py` on the host turns it into a LeRobot v2.1
dataset, where those packages do exist.

    python3 record_episode.py --out /root/vla/raw/ep_000000 \
        --instruction "移動到B病房" --waypoint B病房

Records until SIGINT/SIGTERM (the collector sends it when Nav2 reports the
goal reached), then flushes. Timesteps are sampled on a fixed-rate timer with
latest-message-wins, so every stream lands on the same clock even though they
arrive at different rates (camera ~30 Hz, scan 12 Hz, amcl ~2 Hz).
"""
import argparse
import json
import math
import os
import signal
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import Buffer, TransformListener

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
# AMCL publishes its pose transient-local; a plain subscription can sit empty
# for a long time because AMCL only republishes when the filter updates.
AMCL_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EpisodeRecorder(Node):
    def __init__(self, args):
        # sim time on purpose: Isaac runs slower than wall clock, so a 10 Hz
        # wall-clock timer would space the timesteps unevenly in the world the
        # policy actually sees.
        super().__init__(
            "vla_episode_recorder",
            parameter_overrides=[rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True)],
        )
        self.args = args
        self.bridge = CvBridge()

        # One directory per camera. The single-camera layout ("frames") is kept
        # for a lone camera named "front" so older datasets and to_lerobot.py
        # keep working unchanged.
        self.cams = []
        for spec in (args.cameras or [f"front:{args.image_topic}"]):
            if ":" not in spec:
                raise SystemExit(f"--cameras wants NAME:TOPIC, got {spec!r}")
            nm, topic = spec.split(":", 1)
            d = os.path.join(args.out,
                             "frames" if nm == "front" and len(args.cameras or []) <= 1
                             else f"frames_{nm}")
            os.makedirs(d, exist_ok=True)
            self.cams.append({"name": nm, "topic": topic, "dir": d, "msg": None})

        self.image = None
        self.scan = None
        self.pose = None            # (x, y, yaw) from AMCL
        self.twist = (0.0, 0.0)     # (v, w) measured, from odom
        self.tilt_deg = 0.0
        # The PEAK, not the latest. A robot that lurched to 40 deg
        # and settled back would otherwise report 0 and look clean.
        self.max_tilt = 0.0
        self.action = (0.0, 0.0)    # (v, w) commanded, from Nav2
        self.steps = []
        self.scans = []
        self.done = False
        self.tipped = False
        # Distance to the person, recorded per step so bad frames can be
        # filtered at conversion. Deliberately NOT part of observation.state:
        # the policy has to find the person in the image, and a number that
        # says where they are would let it skip that entirely. This is a label
        # for the dataset builder, not an input.
        self._tf = Buffer()
        self._tfl = TransformListener(self._tf, self)
        self.person_d = []

        for c in self.cams:
            # default-argument binding, not closure capture: a lambda that reads
            # `c` from the enclosing scope would see the LAST camera for every
            # callback, and all three streams would land in one directory.
            self.create_subscription(
                Image, c["topic"],
                lambda m, cam=c: cam.__setitem__("msg", m), SENSOR_QOS)
        self.create_subscription(LaserScan, args.scan_topic, self._on_scan, SENSOR_QOS)
        self.create_subscription(PoseWithCovarianceStamped, args.pose_topic,
                                 self._on_pose, AMCL_QOS)
        self.create_subscription(Odometry, args.odom_topic, self._on_odom, SENSOR_QOS)
        self.create_subscription(Twist, args.cmd_topic, self._on_cmd, 10)
        # The target the expert is following RIGHT NOW, from the expert itself.
        # Not a schedule replayed here: two independent copies of a schedule
        # drift by however long the two processes took to start, and the frames
        # that end up mislabelled are the ones just after a switch -- exactly
        # the frames the whole switching idea exists to create.
        self.cur_target = args.target
        self.create_subscription(String, args.target_topic,
                                 self._on_target, 10)

        self.timer = self.create_timer(1.0 / args.fps, self._tick)
        self.get_logger().info(
            f"recording {args.out} @ {args.fps} Hz -- instruction: {args.instruction}")

    # ----------------------------------------------------------- subscribers
    def _on_image(self, msg):
        self.image = msg

    def _on_scan(self, msg):
        self.scan = msg

    def _on_pose(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_odom(self, msg):
        self.twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)
        # Roll/pitch, to notice the robot going over. A tipped MiR keeps
        # publishing odom and keeps accepting commands, so the recorder happily
        # writes a full-length episode of a machine lying on its side while the
        # expert issues turns that do nothing. Measured: three consecutive
        # 300-step episodes with 0.00 m of movement and 0 deg of turning, every
        # one of them marked ok, because the only success test was "does
        # episode.json exist".
        q = msg.pose.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                          1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.tilt_deg = math.degrees(max(abs(roll), abs(pitch)))
        self.max_tilt = max(self.max_tilt, self.tilt_deg)

    def _on_cmd(self, msg):
        self.action = (msg.linear.x, msg.angular.z)

    def _on_target(self, msg):
        self.cur_target = msg.data

    # ------------------------------------------------------------- recording
    def _tick(self):
        # a timestep without vision is useless to a VLA policy, so wait for the
        # first frame rather than padding the episode with blanks. scan is in the
        # same guard on purpose: it is the slowest topic (~2.5 Hz vs the 10 Hz
        # tick), so without it the opening ticks would append a step + frame but
        # no scan row, and scans[i] would stay one step ahead of steps[i] for the
        # rest of the episode -- silently, since only the row count differs.
        if (self.pose is None or self.scan is None
                or any(c["msg"] is None for c in self.cams)):
            return

        # Before anything is written. Following has no terminal state so the
        # episode is cut by the clock, but cutting after the frame and scan are
        # on disk and before the step is appended breaks the invariant this
        # file exists to protect: num_steps == scan rows == frame count.
        # Measured when this check sat lower down: 300 steps, 301 of each.
        #
        # A flag, not raise: an exception out of a timer callback skips flush()
        # in main() and loses the whole episode.
        if self.args.seconds and len(self.steps) >= int(self.args.seconds * self.args.fps):
            self.done = True
            return

        if self.args.tip_deg > 0.0 and self.tilt_deg > self.args.tip_deg:
            if not self.tipped:
                self.get_logger().error(
                    f"ROBOT TIPPED OVER (tilt {self.tilt_deg:.0f} deg) at step "
                    f"{len(self.steps)} -- stopping; this episode is not usable")
                self.tipped = True
            self.done = True
            return

        # While perturb.py is shoving the robot off the demonstrated path, the
        # commands on cmd_vel are ITS commands, not Nav2's. Recording them would
        # teach the policy to drive off course -- the exact opposite of the
        # point. Skip the whole timestep (frame, scan and step together, so the
        # num_steps == scan rows == frame count invariant still holds); the
        # frames worth having are the ones after the shove, where Nav2
        # demonstrates the way back.
        if self.args.pause_flag and os.path.exists(self.args.pause_flag):
            return

        i = len(self.steps)
        for c in self.cams:
            img = self.bridge.imgmsg_to_cv2(c["msg"], desired_encoding="bgr8")
            cv2.imwrite(os.path.join(c["dir"], f"{i:06d}.png"), img)

        if self.scan is not None:
            r = np.asarray(self.scan.ranges, dtype=np.float32)
            # inf means "nothing out there" -- clamp so the array stays finite
            r[~np.isfinite(r)] = self.scan.range_max
            self.scans.append(r)

        x, y, yaw = self.pose
        stamp = self.get_clock().now().nanoseconds
        self.steps.append({
            "t": stamp * 1e-9,
            # cos/sin instead of raw yaw: no +-pi discontinuity for the policy
            "state": [x, y, math.cos(yaw), math.sin(yaw), self.twist[0], self.twist[1]],
            "action": [self.action[0], self.action[1]],
            "person_dist": self._person_dist(),
            "people": self._people_pose(),
            "target": self.cur_target,
        })

    def _person_dist(self):
        """Metres to the person, or -1 when TF has nothing (non-follow tasks)."""
        try:
            t = self._tf.lookup_transform("base_link", "person", rclpy.time.Time())
        except Exception:
            return -1.0
        return float(math.hypot(t.transform.translation.x,
                                t.transform.translation.y))

    def _people_pose(self):
        """{name: [distance, bearing_rad]} for every person frame being tracked.

        Scoring only, and it must stay that way. The whole question this task
        asks is whether the policy reads its instruction and finds the named
        person IN THE IMAGE; a state vector carrying "purple is 24 degrees to
        your left" answers that question for it, and the experiment would then
        measure nothing at all. to_lerobot.py writes these into the dataset as
        labels and never into observation.state.
        """
        out = {}
        for nm in self.args.people:
            try:
                t = self._tf.lookup_transform("base_link", nm, rclpy.time.Time())
            except Exception:  # noqa: BLE001
                out[nm] = [-1.0, 0.0]
                continue
            x = t.transform.translation.x
            y = t.transform.translation.y
            out[nm] = [float(math.hypot(x, y)), float(math.atan2(y, x))]
        return out

    def flush(self):
        if not self.steps:
            self.get_logger().error("no timesteps recorded -- were the topics publishing?")
            return False

        scans = None
        if self.scans:
            width = max(len(s) for s in self.scans)
            scans = np.stack([np.pad(s, (0, width - len(s)), constant_values=s[-1] if len(s) else 0.0)
                              for s in self.scans])
            np.save(os.path.join(self.args.out, "scan.npy"), scans.astype(np.float32))

        meta = {
            "instruction": self.args.instruction,
            "waypoint": self.args.waypoint,
            "fps": self.args.fps,
            "num_steps": len(self.steps),
            "scan_dim": int(scans.shape[1]) if scans is not None else 0,
            "state_names": ["x", "y", "cos_yaw", "sin_yaw", "v", "w"],
            "action_names": ["v_cmd", "w_cmd"],
            "image_topic": self.args.image_topic,
            "cameras": [{"name": c["name"], "topic": c["topic"],
                         "dir": os.path.basename(c["dir"])} for c in self.cams],
            "people": list(self.args.people),
            "target": self.args.target,
            "targets_seen": sorted({s["target"] for s in self.steps
                                    if s.get("target")}),
            "switches": sum(1 for i in range(1, len(self.steps))
                            if self.steps[i].get("target")
                            != self.steps[i - 1].get("target")),
            "tipped": bool(self.tipped),
            "max_tilt_deg": round(float(self.max_tilt), 1),
            "steps": self.steps,
        }
        with open(os.path.join(self.args.out, "episode.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        self.get_logger().info(f"wrote {len(self.steps)} steps to {self.args.out}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--waypoint", default="")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--cameras", nargs="*", default=[], metavar="NAME:TOPIC",
                    help="record several cameras. Each gets its own\n"
                         "frames_<NAME>/ directory, all written on the\n"
                         "same tick so frame i of every camera is the\n"
                         "same instant. Falls back to --image-topic as a\n"
                         "single camera named front.")
    ap.add_argument("--people", nargs="*", default=[],
                    help="TF frames whose distance and bearing to record\n"
                         "per step, as SCORING LABELS. Never inputs.")
    ap.add_argument("--target", default="",
                    help="which of --people this episode demonstrates\n"
                         "following. Written to the metadata so the\n"
                         "converter can attach the matching instruction.")
    ap.add_argument("--image-topic", default="/realsense/color/image_raw")
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--pose-topic", default="/amcl_pose")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--cmd-topic", default="/diff_cont/cmd_vel_unstamped")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this many seconds of RECORDED steps "
                         "(0 = run until killed). Used by collect_follow.py: "
                         "following has no terminal state.")
    ap.add_argument("--target-topic", default="/follow_target",
                    help="the expert publishes which person it is\n"
                         "following here; each step is labelled with what\n"
                         "was in force when it was sampled.")
    ap.add_argument("--tip-deg", type=float, default=30.0,
                    help="stop and mark the episode unusable if the base\n"
                         "rolls or pitches past this. 0 disables.")
    ap.add_argument("--pause-flag", default="",
                    help="path of a file whose existence pauses recording; "
                         "perturb.py uses it to hide its own commands")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rclpy.init()
    node = EpisodeRecorder(args)

    stopping = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))

    while rclpy.ok() and not stopping["now"] and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)

    ok = node.flush()
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
