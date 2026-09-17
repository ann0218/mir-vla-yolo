#!/usr/bin/env python3
"""Ground-truth person-following controller: the expert the policy clones.

Runs INSIDE my_isaac_sim. It reads the person straight out of TF as
base_link->person -- Isaac publishes odom->person (--walk-person) and
odom->base_link (--publish-odom), and tf2 composes them -- then drives
/diff_cont/cmd_vel_unstamped to hold a standoff distance while facing them.

Needing only TF means this task needs no occupancy map and no localisation
stack: mir_isaac.launch.py for the controller is the whole ROS side.

This is to person-following what Nav2 was to the colour task: a controller with
privileged information, used only to generate demonstrations. **The policy never
sees the person's pose** -- it gets the platform camera and its own velocity,
and has to infer the rest. If the two ever get confused the whole experiment is
worthless, which is why the pose only enters here and never reaches
to_lerobot.py's state vector.

    docker exec my_isaac_sim bash -c '
      source /opt/ros/humble/setup.bash; source /root/ros2_ws/install/setup.bash
      export FASTRTPS_DEFAULT_PROFILES_FILE=/root/fastdds_udp_only.xml
      python3 -u /root/vla/follow_expert.py --seconds 120'
"""
import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import Buffer, TransformListener

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)

def parse_schedule(items):
    """["0:purple", "10:yellow"] -> [(0.0, "purple"), (10.0, "yellow")], sorted.

    Sorted rather than trusted in the order given: an out-of-order entry would
    otherwise be applied and then immediately overridden, producing a schedule
    that silently differs from the one written on the command line.
    """
    out = []
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--target-schedule wants SECONDS:NAME, got {it!r}")
        t, nm = it.split(":", 1)
        out.append((float(t), nm))
    out.sort()
    if not out or out[0][0] > 0.0:
        raise SystemExit("--target-schedule must start with an entry at 0")
    return out


class FollowExpert(Node):
    def __init__(self, a):
        super().__init__("follow_expert", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.twist = (0.0, 0.0)
        self._tf = Buffer()
        self._tfl = TransformListener(self._tf, self)
        self.create_subscription(Odometry, a.odom_topic, self._odm, SENSOR_QOS)
        self.create_subscription(LaserScan, a.scan_topic, self._scn, SENSOR_QOS)
        self.scan = None
        self.n_stuck = 0
        self.n_stall = 0
        self.reversing = 0
        self.pub = self.create_publisher(Twist, a.cmd_topic, 10)
        # Which person is being followed right now, published so the recorder
        # labels each frame with the instruction that was actually in force.
        # A single source of truth on purpose: if the expert and the recorder
        # each ran their own copy of the schedule, a few hundred ms of startup
        # skew would offset every label around a switch, silently teaching the
        # policy the opposite of the truth at exactly the frames that matter.
        self.tgt_pub = self.create_publisher(String, a.target_topic, 10)
        self.schedule = parse_schedule(a.target_schedule) if a.target_schedule else []
        self.cur_target = self.schedule[0][1] if self.schedule else a.target
        self.n = 0
        self.n_lost = 0
        self.dists = []
        self.create_timer(1.0 / a.fps, self._tick)
        self.get_logger().info(
            f"following at {a.standoff:.2f} m, max {a.max_v:.2f} m/s")

    def _odm(self, m):
        self.twist = (m.twist.twist.linear.x, m.twist.twist.angular.z)

    def _scn(self, m):
        self.scan = m

    def front_clear(self):
        """Nearest return in the forward arc, metres. inf if no scan yet."""
        if self.scan is None:
            return float("inf")
        r = np.asarray(self.scan.ranges, dtype=np.float32)
        r[~np.isfinite(r)] = self.scan.range_max
        n = len(r)
        # /scan is the merged 360 deg sweep, index 0 straight ahead.
        half = int(n * self.a.front_arc_deg / 720.0)
        fwd = np.concatenate([r[:half], r[n - half:]])
        return float(fwd.min()) if len(fwd) else float("inf")

    def person_rel(self):
        """(distance, bearing) of the person in the ROBOT's frame.

        Straight from TF base_link->person, which is what the controller
        actually wants -- no map, no localisation stack, no /amcl_pose. Isaac
        publishes odom->base_link (--publish-odom) and odom->person
        (--walk-person), and tf2 composes them.

        None while TF has not caught up, or if the walk was never enabled: this
        controller must never invent a target and drive off after it.
        """
        try:
            t = self._tf.lookup_transform(self.a.base_frame, self.cur_target,
                                          rclpy.time.Time())
        except Exception:
            return None
        x, y = t.transform.translation.x, t.transform.translation.y
        return math.hypot(x, y), math.atan2(y, x)

    def _tick(self):
        p = self.person_rel()
        if p is None:
            self.n_lost += 1
            if self.n_lost in (20, 100) or self.n_lost % 200 == 0:
                self.get_logger().warning(
                    f"no person TF for {self.n_lost} ticks -- is Isaac running "
                    f"with --walk-person?")
            self.pub.publish(Twist())
            return
        self.n_lost = 0
        dist, bearing = p

        # Arc, never pivot. This robot cannot turn on the spot: the MiR100's
        # castors have to scrub round before the base will yaw, and measured in
        # this sim a stationary robot achieves only 9-26% of the commanded
        # angular rate (0.5 and 1.0 rad/s), against 52-80% once it is rolling at
        # 0.3-0.5 m/s.
        #
        # An expert that stops to turn therefore never turns: it sits there with
        # w saturated at the limit while the bearing error stays open. Measured
        # on the first three episodes collected that way, 84% of all frames were
        # stopped-and-spinning with w pinned at -1.2 -- a dataset that teaches a
        # policy to pirouette.
        w = self.a.k_w * bearing
        v = max(0.0, self.a.k_v * (dist - self.a.standoff))   # never reverse

        # Scale the throttle by how well aimed we are. Without this the robot
        # drives away from a target behind it: v is set by DISTANCE, so a person
        # 10 m back puts v at the limit, and full speed on a +140 deg bearing is
        # a sprint in the wrong direction. Measured that way, distance grew from
        # 8 m to 12 m with both v and w pinned at their limits.
        v *= max(0.0, math.cos(bearing))

        if dist < self.a.min_dist:
            # Too close: back off, whatever the bearing. Two reasons, and the
            # first is the one that bites.
            #
            # The camera cannot frame a 1.73 m person nearer than 1.54 m at this
            # 60 deg VFOV, so anything closer is a cropped target -- and cropped
            # targets are most of what the policy would then be trained on.
            # Measured before this guard existed: 20% of the collected samples
            # sat under 1.54 m and the closest was 0.99 m, because the
            # turn-assist below forces v >= 0.3 regardless of distance, so a
            # person slightly off to one side got driven at, not followed.
            #
            # Second, the character has no collision (on purpose -- the robot
            # must not be able to shove it), so there is nothing to stop the
            # base driving straight through them.
            #
            # Reversing also happens to give the castors the roll the turn
            # needs, so this costs nothing in manoeuvrability.
            # Retreat faster the closer it gets. A flat 0.3 m/s loses: the
            # person closes at up to 0.9, so the gap still shrinks at 0.6 m/s
            # and 1.7 m of margin is gone in under three seconds. Measured that
            # way, the person walked to 0.15 m dead ahead of the robot.
            urgency = (self.a.min_dist - dist) / self.a.min_dist
            v = -min(self.a.max_back_v,
                     self.a.v_turn_min + urgency * self.a.max_back_v)
        elif abs(bearing) > math.radians(self.a.turn_assist_deg):
            # Enough roll for the castors to scrub round, and no more: at
            # v_turn_min with w at the limit the turn radius is ~0.25 m, so
            # coming about costs almost no ground.
            v = max(v, self.a.v_turn_min)

        # Wall escape. This base cannot pivot -- the castors need to be rolling
        # before the yaw command bites -- so a robot nose-in to a wall is
        # completely stuck: forward is blocked, and with no forward motion there
        # is no turn either. Measured: after driving into the south wall the
        # expert sat at v=0.3 w=1.2 for 50 s with the bearing pinned behind it.
        #
        # Reversing restores the same castor authority in the other direction
        # AND clears the wall, so it is the one case where backing up is right.
        # Two ways to notice being stuck, because one is not enough.
        #
        # The scan only sees what is in the forward arc, and the robot jams on
        # things beside it just as happily -- measured, it wedged on a pillar at
        # its shoulder with the front arc clear, and sat there commanding
        # v=1.0 while its odometry read 0.00 m/s.
        #
        # So the second test is the general one: commanded to move, not moving.
        # That catches every geometry without having to guess which.
        clear = self.front_clear()
        # Feed-forward wall limit, before the stuck logic below gets a chance to
        # be needed. The recovery there is reactive: it fires once the robot is
        # already commanded forward and not moving, by which point it is against
        # the wall. Measured in the two-person scene, the expert drove itself
        # into a corner at (-0.25, -5.68) with 0.57 m of clearance and the
        # target 7.8 m away at 82 deg -- from there the platform camera sees
        # nobody at all, and every frame recorded is useless.
        #
        # The single-person task never hit this because --person-keep-away kept
        # the person clear of the robot, and so away from the walls it was
        # heading for. That flag is off here: in a two-person scene it amounts
        # to ordering the distractor never to appear on camera.
        if clear < self.a.wall_slow and v > 0.0:
            span = max(self.a.wall_slow - self.a.wall_stop, 1e-3)
            v *= max(0.0, (clear - self.a.wall_stop) / span)
        # Not in rotate-only: v there is the speed the follower WOULD have used,
        # and it is thrown away before publishing. Feeding it to the stall test
        # reports "commanded to move, not moving" on every tick of a robot that
        # was never asked to move -- a warning that is pure noise, and noise
        # that looks exactly like the real fault it is meant to catch.
        rot = self.a.rotate_only
        blocked = (not rot) and clear < self.a.stuck_clear and v > 0.0
        not_moving = (not rot) and v > 0.15 and abs(self.twist[0]) < 0.03
        self.n_stall = self.n_stall + 1 if not_moving else 0

        # Latch the reverse. Without this it lasts exactly one tick: the moment
        # the robot starts backing up, |v_meas| clears the not-moving threshold,
        # the counter resets and forward drive resumes into whatever it was
        # wedged on. Measured that way, the run cycled 8 ticks of stalled
        # forward against 1 tick of reverse and never got free.
        if self.reversing > 0:
            self.reversing -= 1
        elif blocked or self.n_stall >= self.a.stall_ticks:
            self.reversing = self.a.reverse_ticks
            self.n_stall = 0

        if self.reversing > 0:
            # Reverse clears the obstacle and restores the castor authority the
            # turn needs, which pivoting in place cannot.
            v = -self.a.v_turn_min
            self.n_stuck += 1
            if self.n_stuck % 20 == 1:
                why = f"front {clear:.2f} m" if blocked else f"stalled {self.n_stall} ticks"
                self.get_logger().warning(
                    f"stuck ({why}) clear={clear:.2f} v_cmd={v:+.2f} "
                    f"v_meas={self.twist[0]:+.3f} w_cmd={w:+.2f} "
                    f"w_meas={self.twist[1]:+.3f} -- backing out")
        else:
            self.n_stuck = 0

        if self.a.rotate_only:
            # Turn towards the target and nothing else. For the two-colour task
            # the robot is parked and the people orbit it, so the only thing an
            # action can express is WHICH of them the robot is looking at, and
            # the whole demonstration lives in w.
            #
            # Zeroed here rather than by returning early: everything above only
            # ever writes v (the stuck recovery included), and returning would
            # also skip the tick counter and the --seconds stop, which the
            # recorder relies on to end an episode.
            v = 0.0
        v = float(max(-self.a.max_v, min(self.a.max_v, v)))
        w = float(max(-self.a.max_w, min(self.a.max_w, w)))

        # Lateral-acceleration limit: v * w is the centripetal acceleration of a
        # differential base, and it is what rolls a MiR100 carrying a UR5 onto
        # its side. Capping v and w separately does not prevent it -- both were
        # inside their own limits when a run went over at v=1.00 with w=0.83,
        # a 1.2 m radius taken at full speed. Slow the FORWARD speed rather than
        # the turn: a follower that turns too slowly loses the person entirely,
        # measured at a 124 deg bearing while the distance looked fine, whereas
        # one that arrives more slowly still arrives.
        if self.a.max_lat > 0.0 and abs(v * w) > self.a.max_lat:
            v = math.copysign(self.a.max_lat / max(abs(w), 1e-3), v)
            v = float(max(-self.a.max_v, min(self.a.max_v, v)))
        t = Twist()
        t.linear.x, t.angular.z = v, w
        self.pub.publish(t)

        # Switch on the tick counter, not on wall time: the recorder samples on
        # the same simulated clock, and Isaac runs several times slower than
        # real, so a wall-clock schedule would put the switches at different
        # points of the episode every run.
        if self.schedule:
            t_ep = self.n / float(self.a.fps)
            want = self.schedule[0][1]
            for t0, nm in self.schedule:
                if t_ep >= t0:
                    want = nm
            if want != self.cur_target:
                self.get_logger().info(
                    f"target switch at {t_ep:.1f}s: {self.cur_target} -> {want}")
                self.cur_target = want
        self.tgt_pub.publish(String(data=self.cur_target))

        self.n += 1
        self.dists.append(dist)
        if self.n % (self.a.fps * 5) == 0:
            recent = self.dists[-self.a.fps * 5:]
            self.get_logger().info(
                f"[{self.n:5d}] dist {dist:.2f} m (mean {sum(recent)/len(recent):.2f}) "
                f"bearing {math.degrees(bearing):+.0f}deg  v={v:+.2f} w={w:+.2f}")

        if self.a.seconds and self.n >= self.a.seconds * self.a.fps:
            self.pub.publish(Twist())
            d = self.dists
            self.get_logger().info(
                f"done: {self.n} ticks, distance mean {sum(d)/len(d):.2f} m, "
                f"min {min(d):.2f}, max {max(d):.2f}")
            raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standoff", type=float, default=2.0,
                    help="distance to hold (m). The floor is set by the camera, "
                         "not by comfort: at the platform camera's 60 deg VFOV a "
                         "whole 1.73 m person fits from 1.54 m, so 2.0 leaves "
                         "margin for the overshoot when the person stops "
                         "suddenly. Going closer crops the head off, and a "
                         "dataset of cropped targets is a harder problem for no "
                         "reason.")
    ap.add_argument("--k-v", type=float, default=0.8)
    ap.add_argument("--k-w", type=float, default=1.6)
    ap.add_argument("--turn-assist-deg", type=float, default=12.0,
                    help="beyond this bearing error, hold at least --v-turn-min "
                         "forward so the castors scrub round and the turn "
                         "actually happens. Pivoting in place does not work on "
                         "this base -- see the comment in _tick.")
    ap.add_argument("--min-dist", type=float, default=1.70,
                    help="never get closer than this (m); back off if it "
                         "happens. Set above the 1.54 m at which the platform "
                         "camera stops fitting a whole person in frame, with "
                         "margin for the overshoot when the person stops dead.")
    ap.add_argument("--max-back-v", type=float, default=0.80,
                    help="fastest retreat when the person is closing in. Has to "
                         "beat their walking speed or the standoff is only a "
                         "suggestion.")
    ap.add_argument("--v-turn-min", type=float, default=0.30,
                    help="the forward speed that buys angular authority. "
                         "Measured: 0.3 m/s takes the achieved turn rate from "
                         "26%% of commanded to 52%%, 0.5 m/s to 80%%.")
    ap.add_argument("--max-v", type=float, default=1.0)
    ap.add_argument("--max-w", type=float, default=1.2,
                    help="Turn-rate cap. Do NOT lower this for a task\n"
                         "where the robot drives: tracking a person at a\n"
                         "2 m standoff who moves at 0.9 m/s needs\n"
                         "0.9/2 = 0.45 rad/s of yaw, so a 0.4 cap leaves\n"
                         "the robot unable to keep them in front at all --\n"
                         "measured, distance held at 2.08 m while the\n"
                         "bearing sat at 124 deg, i.e. following backwards.\n"
                         "The low cap belongs to the rotate-only task,\n"
                         "where spinning on the spot tips the base.")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=0,
                    help="stop after this long; 0 = run until killed")
    ap.add_argument("--base-frame", default="base_link",
                    help="the robot frame the person is looked up in. Using TF "
                         "directly means this needs no map and no localisation "
                         "stack -- only mir_isaac.launch.py for the controller.")
    ap.add_argument("--rotate-only", action="store_true",
                    help="hold position and only turn towards the target. "
                         "Pairs with the simulator's --person-orbit.")
    ap.add_argument("--target-schedule", nargs="*", default=[],
                    metavar="SECONDS:NAME",
                    help="switch target part-way through, e.g.\n"
                         "0:purple 10:yellow 20:purple. Without this an\n"
                         "episode has one target throughout, and 'keep\n"
                         "tracking whoever you are already tracking'\n"
                         "reproduces the whole demonstration -- so the\n"
                         "instruction is never needed and the policy\n"
                         "learns to ignore it. A switch makes the same\n"
                         "scene demand a different action purely because\n"
                         "the words changed.")
    ap.add_argument("--target-topic", default="/follow_target",
                    help="where the current target name is published, for\n"
                         "the recorder to label frames by.")
    ap.add_argument("--target", default="person",
                    help="TF frame of the person to follow. The two-colour\n"
                         "scene publishes one frame per person (purple,\n"
                         "yellow); this is how an episode says which of\n"
                         "them the demonstration is for.")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--front-arc-deg", type=float, default=90.0,
                    help="width of the forward arc checked for obstacles")
    ap.add_argument("--reverse-ticks", type=int, default=15,
                    help="how long a recovery reverse is held (1.5 s at 10 Hz). "
                         "Long enough to actually come off whatever it caught "
                         "on -- a one-tick nudge just resets the detector.")
    ap.add_argument("--stall-ticks", type=int, default=8,
                    help="reverse after this many consecutive ticks of being "
                         "told to drive but not moving (0.8 s at 10 Hz). Covers "
                         "the jams the forward scan arc cannot see.")
    ap.add_argument("--max-lat", type=float, default=0.45,
                    help="cap on |v*w|, the lateral acceleration, in\n"
                         "m/s^2. 0 disables. 0.45 allows full speed up to\n"
                         "0.45 rad/s of turn, and 0.55 m/s at the 0.8\n"
                         "rad/s that rolled the base over.")
    ap.add_argument("--wall-slow", type=float, default=1.6,
                    help="start easing off the throttle when the front\n"
                         "scan is this close to something.")
    ap.add_argument("--wall-stop", type=float, default=0.7,
                    help="forward speed reaches zero here. Turning is\n"
                         "untouched, so the robot can still swing to face\n"
                         "the person while pinned against a wall.")
    ap.add_argument("--stuck-clear", type=float, default=0.75,
                    help="reverse instead of driving forward when the forward "
                         "arc is closer than this (m). The chassis half-length "
                         "is ~0.45 m, so this fires while there is still room "
                         "to turn out rather than after contact.")
    ap.add_argument("--cmd-topic", default="/diff_cont/cmd_vel_unstamped")
    a = ap.parse_args()

    rclpy.init()
    node = FollowExpert(a)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
