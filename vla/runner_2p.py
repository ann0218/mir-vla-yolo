#!/usr/bin/env python3
"""Closed loop for the two-colour task: turn towards the person you were told.

Runs INSIDE my_isaac_sim. The robot is parked and only turns; the two people
orbit it. What is being tested is whether changing ONE WORD of the instruction
changes which of them the robot tracks.

The offline swap test answers "does the policy read its instruction" frame by
frame. This answers the question that was actually asked -- can the robot find
the named person and stay with them -- which the offline number does not
guarantee: on the colour-parking task a policy with excellent offline metrics
sat completely still in closed loop, because the two are not the same question.

Scoring uses both people's TF, which the policy never receives:

    on_target   fraction of ticks with the named person closer to the
                centreline than the other one. 50 % is chance.
    |bearing|   how well centred the named person is kept.
    lock        longest unbroken run of on-target ticks, in seconds. A policy
                that flickers between the two averages well and has followed
                neither.

    python3 runner_2p.py --instruction "follow the person in purple" \\
        --target purple --seconds 60 --trace /root/vla/2p_purple.csv
"""
import argparse
import base64
import json
import math
import os
import urllib.request

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import Buffer, TransformListener

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)

# Dataset camera name -> the topic it came from -> the key the policy wants.
# Must match the --rename_map used at training; the server rejects a mismatch
# rather than silently running on whichever cameras happened to line up.
CAMERAS = [
    ("front", "/platform_camera/color/image_raw", "observation.images.camera1"),
    ("corner_a", "/corner_camera_a/color/image_raw", "observation.images.camera2"),
    ("corner_b", "/corner_camera_b/color/image_raw", "observation.images.camera3"),
]


class Runner(Node):
    def __init__(self, a):
        super().__init__("runner_2p", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time",
                                      rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.bridge = CvBridge()
        self.imgs = {}
        self.scan = None
        self.n = 0
        self.on = 0
        self.bearings = []
        self.run = 0
        self.best_run = 0
        self.tilt_deg = 0.0

        for name, topic, key in CAMERAS:
            self.create_subscription(
                Image, topic,
                lambda m, k=key: self.imgs.__setitem__(k, m), SENSOR_QOS)
        self.create_subscription(LaserScan, a.scan_topic, self._scan, SENSOR_QOS)
        self.create_subscription(Odometry, a.odom_topic, self._odom, SENSOR_QOS)
        self.pub = self.create_publisher(Twist, a.cmd_topic, 10)
        self._tf = Buffer()
        self._tfl = TransformListener(self._tf, self)

        self.trace = open(a.trace, "w") if a.trace else None
        if self.trace:
            self.trace.write("step,bear_target,bear_other,w_cmd,on_target,"
                             "v_cmd,dist_target,dist_other,estop,"
                             "v_policy,cam_y2,cam_stop,gt_close,accel_clip,"
                             "pred_x1,pred_y1,pred_x2,pred_y2,pred_vis\n")

        with urllib.request.urlopen(f"http://{a.server}/info", timeout=10) as r:
            info = json.loads(r.read())
        self.get_logger().info(
            f"server -> {info.get('run')}/{info.get('step')} "
            f"(state {info.get('state_dim')}d, cameras {info.get('image_keys')})")
        want = a.scan_sectors
        if info.get("state_dim") != want:
            self.get_logger().error(
                f"STATE DIM MISMATCH: server wants {info.get('state_dim')}, "
                f"this runner builds {want}")

        self.w_prev = 0.0
        self.d_target, self.d_min, self.estops = [], float("inf"), 0
        self.v_prev, self.cam_stops, self.gt_closes, self.accel_clips = 0.0, 0, 0, 0
        self.frame_n = 0
        if a.save_frames:
            os.makedirs(a.save_frames, exist_ok=True)
        self.warmed = a.warmup <= 0.0
        self.warm_n = 0
        self.locked = 0

        self._post("/reset", {})
        self.create_timer(1.0 / a.fps, self._tick)
        self.get_logger().info(f'instruction: "{a.instruction}" (target {a.target})')

    def _scan(self, m):
        self.scan = m

    def _odom(self, m):
        q = m.pose.pose.orientation
        roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                          1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
        self.tilt_deg = math.degrees(max(abs(roll), abs(pitch)))

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"http://{self.a.server}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
            return json.loads(r.read())

    def _state(self):
        """24 min-pooled scan sectors. No pose, no velocity, no person.

        The robot's own [v, w] is deliberately absent: the expert's turn is
        smooth, so w(t) predicts w(t+1) with r=0.895, and a policy handed it
        just continues turning the way it already is -- reproducing the whole
        demonstration without reading the instruction. Leaving it in cost a
        full training run.
        """
        r = np.asarray(self.scan.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = self.a.scan_clip
        r = np.clip(r, 0.0, self.a.scan_clip)
        k = self.a.scan_sectors
        if len(r) % k:
            r = r[:len(r) // k * k]
        return r.reshape(k, -1).min(axis=1).tolist()

    def _bearing(self, frame):
        try:
            t = self._tf.lookup_transform("base_link", frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        return math.atan2(t.transform.translation.y, t.transform.translation.x)

    def _dist(self, frame):
        try:
            t = self._tf.lookup_transform("base_link", frame, rclpy.time.Time())
        except Exception:  # noqa: BLE001
            return None
        return math.hypot(t.transform.translation.x, t.transform.translation.y)

    def _front_clear(self):
        """Nearest lidar return in the forward arc.

        Default (--true-front): selected by beam ANGLE. /scan is
        virtual_laser_link, yaw 0 against base_link, angle_min -180 deg, so
        beam 0 points straight BEHIND. follow_expert.front_clear() takes beams
        around index 0 as "ahead" -- it has always watched the rear arc, and so
        did the first --drive batch. Without --true-front that computation is
        reproduced, to match the training data.
        """
        r = np.asarray(self.scan.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = self.scan.range_max
        n = len(r)
        if self.a.true_front:
            ang = self.scan.angle_min + np.arange(n) * self.scan.angle_increment
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            fwd = r[np.abs(ang) <= math.radians(self.a.front_arc_deg / 2.0)]
        else:
            half = int(n * self.a.front_arc_deg / 720.0)
            fwd = np.concatenate([r[:half], r[n - half:]])
        return float(fwd.min()) if len(fwd) else float("inf")

    def _save_frame(self, pred_box, boxes, bt, bo, dt, do, v, w):
        """The platform view with what the policy said drawn on it.

        Green: the box the policy predicted, only when it says it can see the
        named person. Grey: every detection the crop pipeline made. Circles on
        the bottom edge: where each person REALLY is, from TF, which the policy
        never receives -- so green landing on the wrong circle is a mistake the
        viewer can see for themselves.
        """
        msg = self.imgs.get(CAMERAS[0][2])
        if msg is None:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8").copy()
        for b in boxes:
            x1, y1, x2, y2 = [int(c) for c in b[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
        for bear, colour, name in ((bt, (0, 200, 255), "named"),
                                   (bo, (200, 200, 200), "other")):
            if bear is None or abs(bear) > math.radians(37.6):
                continue
            u = int(320.0 - 415.75 * math.tan(bear))
            cv2.circle(img, (u, 470), 7, colour, -1)
            cv2.putText(img, name, (u - 18, 458), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, colour, 1)
        if pred_box[4] > 0.5:
            x1, y1, x2, y2 = [int(c) for c in pred_box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.putText(img, f"policy: {self.a.target} here", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 2)
        else:
            cv2.putText(img, "policy: cannot see them", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 2)
        cv2.putText(img, f'"{self.a.instruction}"', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 240), 2)
        cv2.putText(img, f"v={v:+.2f} w={w:+.2f}  named {dt:.1f}m  other {do:.1f}m",
                    (8, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 240), 1)
        cv2.imwrite(os.path.join(self.a.save_frames, f"{self.frame_n:05d}.png"), img)
        self.frame_n += 1

    def _tick(self):
        if self.scan is None or len(self.imgs) != len(CAMERAS):
            return
        if self.tilt_deg > self.a.tip_deg:
            self.get_logger().error(
                f"ROBOT TIPPED OVER (tilt {self.tilt_deg:.0f} deg) -- aborting")
            self.pub.publish(Twist())
            raise SystemExit(3)

        payload_imgs = {}
        # In CAMERAS order, never self.imgs order: that dict is filled as
        # messages arrive, which put a corner camera first in 5 of 5 runs.
        for _name, _topic, key in CAMERAS:
            msg = self.imgs[key]
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                return
            payload_imgs[key] = base64.b64encode(jpg.tobytes()).decode()

        try:
            out = self._post("/act", {"images": payload_imgs,
                                      "state": self._state(),
                                      "task": self.a.instruction})
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"policy server: {e}")
            return
        v, w = out["action"]
        seen = out.get("people", -1)
        boxes = out.get("boxes", [])
        # A policy trained with --box-action also predicts where it thinks the
        # named person is. Logged only -- it never touches control, and it is
        # what separates "followed the wrong person" from "found the right
        # person and steered badly". Zeros for a policy without those dims.
        _full = out.get("action_full") or []
        pred_box = [float(x) for x in _full[2:7]] if len(_full) >= 7 else [0.0] * 5

        # Warm-up. An episode used to open with the robot pointing wherever it
        # happened to stop, which is out of distribution: the expert that
        # produced the data always had the people in frame (88-100 % of frames
        # tracked), so the policy has never seen two blank crops and has no
        # behaviour for recovering from them. Measured without this: the crop
        # pipeline found nobody in 44 % of frames and both people in 6 %.
        # Ground truth is used to aim, never to drive -- once handed over the
        # policy is on its own.
        if not self.warmed:
            self.warm_n += 1
            bt, bo = self._bearing(self.a.target), self._bearing(self.a.other)
            if bt is None or bo is None:
                return
            # Bisector of the two, so both sit as far from the frame edge as
            # the pair allows. Averaged as unit vectors, not as angles, which
            # would jump at the +-180 deg wrap.
            mid = math.atan2(math.sin(bt) + math.sin(bo),
                             math.cos(bt) + math.cos(bo))
            self.locked = self.locked + 1 if (abs(mid) < math.radians(8)
                                              and seen >= 2) else 0
            if self.locked >= 3:
                self.warmed = True
                self.get_logger().info(
                    f"warm-up done after {self.warm_n / self.a.fps:.1f}s: "
                    f"target {math.degrees(bt):+.0f}deg "
                    f"other {math.degrees(bo):+.0f}deg, both in frame")
            elif self.warm_n >= int(self.a.warmup * self.a.fps):
                self.warmed = True
                self.get_logger().warn(
                    f"warm-up gave up after {self.a.warmup:.0f}s "
                    f"(saw {seen} people, bisector {math.degrees(mid):+.0f}deg)"
                    " -- scoring anyway, the run starts out of distribution")
            else:
                tw = Twist()
                tw.angular.z = float(max(-self.a.max_w,
                                         min(self.a.max_w, 1.2 * mid)))
                self.pub.publish(tw)
                return

        w = float(max(-self.a.max_w, min(self.a.max_w, w)))
        # Angular acceleration limit. This task is rotation only, so v is always
        # zero and the |v*w| lateral limit that fixed tipping while driving can
        # never trigger -- yet the robot still went over at 85 deg with w pinned
        # at -0.60 through a long in-place spin. Capping how fast w may change
        # is an actuator constraint, applied to whatever the policy asks for.
        if self.a.max_dw > 0.0:
            lim = self.a.max_dw / self.a.fps
            w = float(max(self.w_prev - lim, min(self.w_prev + lim, w)))
        self.w_prev = w
        estop = 0
        v_policy = float(v)
        cam_y2, cam_stop, gt_close, accel_clip = 0.0, 0, 0, 0
        if self.a.drive:
            # The same limits, in the same order, that follow_expert.py applied
            # to every recorded action: clip, forward wall limit on the front
            # lidar arc, then |v*w|. The policy was trained on commands that had
            # already been through this. The expert's ground-truth back-off and
            # stuck recovery are NOT copied -- the policy has to supply those.
            v = float(max(-self.a.max_v, min(self.a.max_v, v)))
            # Forward acceleration limit. Only speeding up while going forward
            # is capped; braking and reversing pass straight through. A run
            # tipped at 62 deg after v jumped 0.35 -> 0.87 m/s in one tick, with
            # |v*w| inside its limit -- and the expert itself makes such jumps
            # (10.7 % of its speed-ups exceed 0.3 m/s per tick).
            if self.a.max_accel > 0.0 and v > 0.0 and v > self.v_prev:
                cap = max(self.v_prev, 0.0) + self.a.max_accel / self.a.fps
                if v > cap:
                    v, accel_clip = cap, 1
                    self.accel_clips += 1
            clear = self._front_clear()
            if clear < self.a.wall_slow and v > 0.0:
                span = max(self.a.wall_slow - self.a.wall_stop, 1e-3)
                v *= max(0.0, (clear - self.a.wall_stop) / span)
            if self.a.max_lat > 0.0 and abs(v * w) > self.a.max_lat:
                v = math.copysign(self.a.max_lat / max(abs(w), 1e-3), v)
            # Camera stop: a person whose box reaches the bottom of the frame is
            # closer than ~1.55 m (camera 1.0 m up, 3 deg down, 60 deg VFOV,
            # 1.73 m person), and inside the central band they are in the path.
            # Sensor-only -- the policy sees the same image.
            if self.a.cam_stop_y2 > 0.0:
                band = self.a.fx * math.tan(math.radians(self.a.cam_stop_halfdeg))
                ahead = [b for b in boxes if abs((b[0] + b[2]) / 2.0 - self.a.cx) <= band]
                cam_y2 = max((b[3] for b in ahead), default=0.0)
                if cam_y2 >= self.a.cam_stop_y2 and v > 0.0:
                    v, cam_stop = 0.0, 1
                    self.cam_stops += 1
            # Ground-truth monitor: would the old emergency stop have fired?
            # Logged only, never acted on unless --estop-dist is also set.
            if self.a.gt_monitor_dist > 0.0:
                for fr in (self.a.target, self.a.other):
                    d, b = self._dist(fr), self._bearing(fr)
                    if (d is not None and b is not None
                            and d < self.a.gt_monitor_dist and abs(b) < math.pi / 2):
                        gt_close = 1
                self.gt_closes += gt_close
            # Emergency stop, not part of the expert. The people have no
            # collision, so nothing physical stops the base driving through
            # one. It reads ground truth, so every firing is counted and
            # reported: a run that leans on it is not following on its own.
            if self.a.estop_dist > 0.0 and v > 0.0:
                for fr in (self.a.target, self.a.other):
                    d, b = self._dist(fr), self._bearing(fr)
                    if (d is not None and b is not None
                            and d < self.a.estop_dist and abs(b) < math.pi / 2):
                        v, estop = 0.0, 1
                self.estops += estop
        else:
            # The original protocol: rotation only, v thrown away. Note the
            # checkpoint it was used with (smolvla_mswall) was trained on
            # driving data and asked for v ~ +0.3 m/s most of the time.
            v = 0.0
        self.v_prev = v
        t = Twist()
        t.linear.x, t.angular.z = v, w
        self.pub.publish(t)

        bt = self._bearing(self.a.target)
        bo = self._bearing(self.a.other)
        dt, do = self._dist(self.a.target), self._dist(self.a.other)
        self.n += 1
        if bt is not None and bo is not None:
            hit = abs(bt) < abs(bo)
            self.on += 1 if hit else 0
            self.bearings.append(abs(math.degrees(bt)))
            self.run = self.run + 1 if hit else 0
            self.best_run = max(self.best_run, self.run)
            if dt is not None:
                self.d_target.append(dt)
            for d in (dt, do):
                if d is not None:
                    self.d_min = min(self.d_min, d)
            if self.trace:
                nan = float("nan")
                self.trace.write(f"{self.n},{math.degrees(bt):.1f},"
                                 f"{math.degrees(bo):.1f},{w:.3f},{int(hit)},"
                                 f"{v:.3f},{dt if dt is not None else nan:.3f},"
                                 f"{do if do is not None else nan:.3f},{estop},"
                                 f"{v_policy:.3f},{cam_y2:.1f},{cam_stop},{gt_close},{accel_clip},"
                                 f"{pred_box[0]:.1f},{pred_box[1]:.1f},{pred_box[2]:.1f},"
                                 f"{pred_box[3]:.1f},{pred_box[4]:.3f}\n")
            if self.a.save_frames:
                self._save_frame(pred_box, boxes, bt, bo,
                                 dt if dt is not None else -1.0,
                                 do if do is not None else -1.0, v, w)

        if self.n % (self.a.fps * 5) == 0:
            self.get_logger().info(
                f"[{self.n:5d}] target {math.degrees(bt or 0):+.0f}deg "
                f"other {math.degrees(bo or 0):+.0f}deg v={v:+.2f} w={w:+.2f} "
                f"on_target={100 * self.on / max(self.n, 1):.0f}%")

        if self.a.seconds and self.n >= int(self.a.seconds * self.a.fps):
            self.pub.publish(Twist())
            b = self.bearings
            # Reported in two halves. An episode opens with the named person
            # wherever they happen to be -- often behind the robot -- so the
            # first stretch is a search, and averaging it together with the
            # tracking that follows understates both. Measured on one run:
            # 45 % on-target and a 128 deg median overall, while the last 40 %
            # of the same run held the target at 11-38 deg. The second half is
            # what "can it hold the person it was told to" means.
            half = len(b) // 2
            late = b[half:] if b else []
            self.get_logger().info(
                f"RESULT target={self.a.target} steps={self.n} "
                f"on_target={100 * self.on / max(self.n, 1):.0f}% "
                f"median|bearing| all={np.median(b) if b else -1:.0f}deg "
                f"late={np.median(late) if len(late) else -1:.0f}deg "
                f"longest_lock={self.best_run / self.a.fps:.1f}s"
                + (f" drive: target_dist_mean={np.mean(self.d_target):.2f} "
                   f"band_{self.a.band[0]:g}_{self.a.band[1]:g}m="
                   f"{100 * np.mean([self.a.band[0] <= d <= self.a.band[1] for d in self.d_target]):.0f}% "
                   f"closest_person={self.d_min:.2f} estops={self.estops} "
                   f"cam_stops={self.cam_stops} gt_close={self.gt_closes} "
                   f"accel_clips={self.accel_clips}"
                   if self.a.drive and self.d_target else ""))
            if self.trace:
                self.trace.close()
            raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:8770")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--target", required=True,
                    help="TF frame of the person the instruction names. Scoring "
                         "only -- the policy is never given it.")
    ap.add_argument("--other", default="",
                    help="the distractor's frame; defaults to the other of "
                         "purple/yellow")
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--cmd-topic", default="/diff_cont/cmd_vel_unstamped")
    ap.add_argument("--scan-sectors", type=int, default=24)
    ap.add_argument("--scan-clip", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-w", type=float, default=0.6,
                    help="matches the cap the demonstrations were collected "
                         "under; the base tips over above it.")
    ap.add_argument("--max-dw", type=float, default=1.2, metavar="RAD/S2",
                    help="angular acceleration limit. 0 disables. The robot tips "
                         "over in a sustained in-place spin, and with v pinned at "
                         "zero the |v*w| limit used elsewhere cannot fire.")
    ap.add_argument("--drive", action="store_true",
                    help="publish the policy's forward speed too. Without it v is "
                         "forced to 0, the original rotation-only protocol.")
    ap.add_argument("--max-v", type=float, default=1.0)
    ap.add_argument("--max-lat", type=float, default=0.45,
                    help="|v*w| limit, as in follow_expert.py")
    ap.add_argument("--front-arc-deg", type=float, default=90.0)
    ap.add_argument("--wall-slow", type=float, default=1.6)
    ap.add_argument("--wall-stop", type=float, default=0.7)
    ap.add_argument("--estop-dist", type=float, default=1.0,
                    help="with --drive: no forward speed while a person is closer "
                         "than this ahead (ground truth). Counted and reported; "
                         "0 disables.")
    ap.add_argument("--max-accel", type=float, default=0.0, metavar="M/S2",
                    help="with --drive: cap forward speed-up at this rate; braking "
                         "and reversing are not limited. 0 disables.")
    ap.add_argument("--true-front", action="store_true",
                    help="wall limit on the real forward arc (by beam angle). "
                         "Without it the expert's computation is reproduced, "
                         "which watches the REAR arc.")
    ap.add_argument("--cam-stop-y2", type=float, default=0.0, metavar="PX",
                    help="with --drive: no forward speed while a person box inside "
                         "the central band reaches this image row (480 = bottom). "
                         "0 disables.")
    ap.add_argument("--cam-stop-halfdeg", type=float, default=25.0)
    ap.add_argument("--fx", type=float, default=415.75, help="platform camera focal px")
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--gt-monitor-dist", type=float, default=1.0,
                    help="log steps with a person closer than this ahead (ground "
                         "truth), without acting on it")
    ap.add_argument("--band", type=float, nargs=2, default=[1.5, 3.0],
                    help="scoring: distance range counted as following")
    ap.add_argument("--warmup", type=float, default=25.0, metavar="SECONDS",
                    help="before scoring starts, turn the robot to the bisector "
                         "of the two people using ground truth and wait until "
                         "the crop pipeline reports both in frame, so the "
                         "episode begins in the distribution the expert "
                         "demonstrated. 0 disables. Ground truth is used to aim "
                         "only; the policy drives every scored step.")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--tip-deg", type=float, default=30.0)
    ap.add_argument("--save-frames", default="",
                    help="write every tick's platform view here, with the policy's\n"
                         "predicted box, the detections and the real positions drawn\n"
                         "on it. For making a clip; costs one PNG per tick.")
    ap.add_argument("--trace", default="")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()
    if not a.other:
        a.other = "yellow" if a.target == "purple" else "purple"

    rclpy.init()
    n = Runner(a)
    try:
        rclpy.spin(n)
    except SystemExit:
        pass
    finally:
        n.pub.publish(Twist())
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
