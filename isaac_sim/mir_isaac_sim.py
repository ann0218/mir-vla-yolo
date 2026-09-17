#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Isaac Sim back-end for the MiR100 + UR5 + Robotiq85 + D435i robot.
# Replaces the Gazebo simulator. Bridges the robot articulation to ROS2 /
# ros2_control / MoveIt via the topic_based_ros2_control interface:
#
#   Isaac  --(sensor_msgs/JointState)-->  /isaac_joint_states   --> ros2_control
#   ros2_control --(sensor_msgs/JointState)--> /isaac_joint_commands --> Isaac
#   Isaac  --(rosgraph_msgs/Clock)-->     /clock                (use_sim_time)
#
# On the ROS side run, in another terminal:
#   ros2 launch mir_gazebo mir_isaac.launch.py
#
# Run this script with the Isaac Sim python environment, e.g.:
#   cd /mnt/data/IsaacSim/_build/linux-x86_64/release
#   ./python.sh /mnt/data/mir_isaac/mir_ur5_humble/isaac_sim/mir_isaac_sim.py \
#       --usd /mnt/data/mir_isaac/mir_ur5_humble/isaac_sim/usd/mir_isaac.usd

import argparse
import math
import os
import sys

# ----------------------------------------------------------------- CLI args
# (parsed before SimulationApp so we can honour --headless)
parser = argparse.ArgumentParser(description="Isaac Sim ROS2 bridge for MiR+UR5")
_default_usd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "mir_isaac.usd")
parser.add_argument("--usd", default=_default_usd, help="Path to the robot USD.")
parser.add_argument("--headless", action="store_true", help="Run without a GUI.")
parser.add_argument("--test-walls", action="store_true",
                    help="Diagnostic: build a closed box of walls around the "
                         "robot so every direction has a wall — used to check the "
                         "lidar covers a full 360 (gaps = real sensor blind spot).")
parser.add_argument("--top-down", action="store_true",
                    help="Start the GUI viewport looking straight down (top-down "
                         "/ bird's-eye), like the RViz map view, instead of the "
                         "default 3/4 perspective.")
parser.add_argument("--top-down-height", type=float, default=18.0,
                    help="Camera height (m) for --top-down. Raise to see more of "
                         "the map, lower to zoom in.")
parser.add_argument("--commands-topic", default="isaac_joint_commands",
                    help="(legacy single-topic; kept for the home-pose ramp)")
parser.add_argument("--command-topics", nargs="+",
                    default=["isaac_arm_commands", "isaac_base_commands",
                             "isaac_gripper_commands"],
                    help="One command topic per ros2_control hardware component. "
                         "Separate topics avoid the 3-publisher interleaving that "
                         "starved the arm on a single /isaac_joint_commands.")
parser.add_argument("--states-topic", default="isaac_joint_states")
parser.add_argument("--robot-prim", default="",
                    help="Articulation root prim path. Auto-detected if empty.")
parser.add_argument("--world", default="",
                    help="Path to an environment USD to load under /World/Env "
                         "(e.g. isaac_sim/usd/maze.usd). If omitted, only a "
                         "ground plane is added.")
parser.add_argument("--arm-stiffness", type=float, default=10000.0,
                    help="UR joint position-drive stiffness (tracks MoveIt "
                         "trajectories tightly).")
parser.add_argument("--arm-damping", type=float, default=1000.0,
                    help="UR joint position-drive damping.")
parser.add_argument("--arm-max-force", type=float, default=330.0,
                    help="Per-joint torque cap (N*m) ~ UR5 limits. Enough to "
                         "follow trajectories, gentle enough that the arm cannot "
                         "flip the 67 kg MiR base (verified upright to 500 N*m).")
parser.add_argument("--arm-armature", type=float, default=0.0,
                    help="Rotor inertia (armature) added to the 6 UR joints. The "
                         "wrist links have small inertia, so the stiff drive rings "
                         "(arm jitter) the same way the gripper did; armature "
                         "raises the effective inertia into the integrable range. "
                         "0.0 disables it.")
parser.add_argument("--fold-arm", action="store_true",
                    help="home the UR5 folded over the deck instead of reaching "
                         "forward. The default pose sticks ~0.2 m past the "
                         "chassis at 1.46 m, above the scan plane, and wedges "
                         "the robot on furniture the laser never sees.")
parser.add_argument("--no-arm-home", action="store_true",
                    help="Skip the startup ramp that eases the arm to its ROS "
                         "home pose (the ramp prevents the base from flipping "
                         "when ros2_control first commands the arm).")
parser.add_argument("--base-linear-damping", type=float, default=0.0,
                    help="Linear damping on the MiR chassis (base_link). Default "
                         "0.0 = original behaviour. WARNING: high values (e.g. 2.0) "
                         "drag the base so hard it barely drives under Nav2. "
                         "Optional idle-jitter knob only.")
parser.add_argument("--base-angular-damping", type=float, default=0.0,
                    help="Angular (yaw) damping on the MiR chassis (base_link). "
                         "Default 0.0 = original behaviour. Optional knob for idle "
                         "yaw wobble; high values also resist commanded turns.")
parser.add_argument("--wheel-drive-damping", type=float, default=1.0e5,
                    help="Velocity-drive damping (gain) on the 2 drive wheels. "
                         "Higher = the wheels track the commanded rad/s more "
                         "stiffly under load (helps in-place rotation, where the "
                         "wheels otherwise lag/slip and the base barely turns).")
parser.add_argument("--wheel-friction", type=float, default=2.0,
                    help="Static/dynamic friction for the drive wheels + ground. "
                         "The URDF's high wheel friction is a <gazebo> tag Isaac "
                         "ignores, so without this the wheels slip on the default "
                         "low-friction ground: the robot drives SLOWER than the "
                         "commanded cmd_vel and turns erratically (sometimes "
                         "over-turns, sometimes not at all). 0.0 disables it.")
parser.add_argument("--caster-swivel-damping", type=float, default=0.0,
                    help="Rotational damping on the 4 passive caster *swivel* "
                         "joints. The rolling caster wheels stay fully free; only "
                         "the swivel gets light damping so the trailing casters "
                         "cannot pump yaw into the chassis (base slowly spinning "
                         "in place when idle). 0.0 = fully free (old behaviour).")
parser.add_argument("--gripper-stiffness", type=float, default=1.0e4,
                    help="Stiffness of the Robotiq master position drive. The "
                         "finger links have tiny inertia, so a very stiff drive "
                         "rings; lower this (e.g. 1e3) together with --gripper-"
                         "armature if the gripper buzzes.")
parser.add_argument("--gripper-damping", type=float, default=1.0e3,
                    help="Damping on the Robotiq master position drive. Raise to "
                         "settle gripper/finger jitter from the PhysX mimic "
                         "coupling.")
parser.add_argument("--gripper-armature", type=float, default=0.05,
                    help="Rotor inertia (armature) added to the 6 Robotiq joints. "
                         "The fingers' inertia is ~1e-5 kg*m^2, so the stiff drive "
                         "+ hard PhysX mimic coupling oscillate faster than the "
                         "sim step can integrate -> buzzing. Armature raises the "
                         "joints' effective inertia into the integrable range and "
                         "is the standard fix. 0.0 disables it.")
parser.add_argument("--solver-position-iterations", type=int, default=32,
                    help="PhysX articulation solver position-iteration count. The "
                         "Robotiq finger linkage is over-constrained by the hard "
                         "mimic joints; more iterations converge those constraints "
                         "each step and cut gripper jitter. (default importer ~4)")
parser.add_argument("--solver-velocity-iterations", type=int, default=4,
                    help="PhysX articulation solver velocity-iteration count.")
parser.add_argument("--person-keep-away", action="store_true",
                    help="pick walking goals that do not lead towards the "
                         "robot, instead of only swerving once it is close. "
                         "Changes what the scene does, so runs recorded with "
                         "it are not comparable with measurements taken "
                         "without it.")
parser.add_argument("--person-corridor", type=float, default=2.5,
                    help="half-width of the clear corridor --person-keep-away "
                         "requires along the person's path. A goal whose "
                         "straight line passes within this of the robot, with "
                         "the robot part-way along it, is rejected.")
parser.add_argument("--max-depenetration-velocity", type=float, default=0.0,
                    help="Cap (m/s) on how fast PhysX may push the robot's bodies "
                         "back out of geometry they have penetrated. 0.0 (default) "
                         "leaves it unlimited. "
                         "EXPERIMENTAL, and off by default for a reason: it was "
                         "added to stop Nav2 wedging the base into a wall corner "
                         "and PhysX then ejecting it out of the map in one step "
                         "(seen at the maze's 0.93 m gap: z 0.001 -> 2.1 m, "
                         "y 3.8 -> 20.7 m in 0.5 s, which takes AMCL with it). "
                         "At 1.0 it does prevent that, but it causes a WORSE "
                         "failure: with depenetration throttled the constraint "
                         "error accumulates until the solver diverges, and the "
                         "robot explodes from a standstill (measured: rest at "
                         "(-0.08,-0.30) -> 53 m away in one 0.43 s step, then "
                         "runaway to 2088 m and NaN). If you want to experiment, "
                         "try values >= 5.0; the real fix for the wedging is "
                         "ObstaclesCritic.consider_footprint: true on the Nav2 "
                         "side, which stops the base entering the corner at all.")
parser.add_argument("--no-arm-collision", action="store_true",
                    help="Strip collision from the UR5 + Robotiq links, keeping "
                         "their visuals. The arm reaches 0.642 m ahead of the "
                         "0.89 m chassis at 1.46 m height -- above the SICK scan "
                         "plane -- so the robot is physically longer than "
                         "anything it can sense, and wedges on furniture and "
                         "door frames while the laser still reports a metre of "
                         "clearance. For navigation-only work the arm is "
                         "decoration; dropping its collision removes a failure "
                         "mode that is easy to mistake for a bad policy. "
                         "Visuals stay so recorded camera frames are unchanged.")
parser.add_argument("--keep-self-collisions", action="store_true",
                    help="Keep intra-robot self-collisions enabled. By default "
                         "they are DISABLED, because the Robotiq fingers grazing "
                         "each other under the PhysX mimic coupling make the "
                         "gripper jitter; external-object collision (grasping) is "
                         "unaffected either way.")
parser.add_argument("--wheel-test", action="store_true",
                    help="Diagnostic: spin the drive wheels directly via the "
                         "articulation API at startup and report rotation, to "
                         "isolate drive vs OmniGraph controller faults.")
parser.add_argument("--publish-odom", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Publish odom->base_footprint TF + ground-truth "
                         "odometry from Isaac. ON by default, and it has to be: "
                         "the ROS side runs with enable_odom_tf=false "
                         "(diffdrive_controller_isaac.yaml), so with this off "
                         "NOTHING broadcasts odom->base_footprint and Nav2/AMCL "
                         "have no TF chain to the robot. Use --no-publish-odom "
                         "only together with enable_odom_tf:=true.")
parser.add_argument("--base-frame", default="base_footprint")
parser.add_argument("--odom-frame", default="odom")
parser.add_argument("--no-imu", action="store_true",
                    help="Skip the PhysX IMU sensor (/imu_data).")
# Lidars / camera are OPT-IN: standalone-script RTX rendering through ROS2
# in Isaac Sim 5.0 has an SDG annotator-registration race that defeats the
# writer attach in headless mode (NVIDIA's own rtx_lidar.py standalone example
# crashes here too). Sensor prims and OG nodes are still built so the topics
# exist; for actual data flow open the produced USD in the Isaac Sim GUI and
# build the sensor OmniGraphs via the menu shortcuts.
parser.add_argument("--lasers", action="store_true",
                    help="Enable the 2 SICK S300 PhysX lidars (/f_scan, /b_scan). "
                         "Pure OmniGraph ray-cast; works headless. 240° FOV, "
                         "0.05–29 m. Ray count = --laser-rays.")
parser.add_argument("--laser-rays", type=int, default=541,
                    help="Rays per SICK lidar over the 240° FOV. Hardware-exact "
                         "default is 541 (matches the Gazebo sensor, which achieves "
                         "full 360° merged coverage at 541). Raise it only to "
                         "experiment with denser reprojected /scan bins.")
parser.add_argument("--rtx-lasers", action="store_true",
                    help="Enable the 2 SICK S300 RTX lidars (/f_scan, /b_scan) "
                         "using the SICK_S300.json profile (IsaacSensorCreateRtxLidar "
                         "+ ROS2RtxLidarHelper). Requires a viewport (not headless). "
                         "RTX gives physically-based reflectance/noise vs. PhysX "
                         "ideal ray-cast. Use --lasers instead for headless runs.")
parser.add_argument("--body-camera", action="store_true",
                    help="Add a forward-looking camera on the CHASSIS, where a "
                         "real MiR100 carries its 3D obstacle cameras, and "
                         "publish it as /body_camera/color/image_raw. The stock "
                         "RealSense in this URDF hangs off ur_flange -- it is an "
                         "eye-in-hand camera for the arm, sitting 1.12 m up, "
                         "0.36 m left of centre, looking level. Measured: it "
                         "mostly frames blank wall, and it MOVES WITH THE ARM, "
                         "so any arm motion changes the observation. A chassis "
                         "camera is what a navigation policy actually wants.")
parser.add_argument("--body-camera-xyz", type=float, nargs=3,
                    default=[0.50, 0.0, 0.30], metavar=("X", "Y", "Z"),
                    help="Mount point in base_link. The shell mesh runs to "
                         "x=0.487 (NOT the 0.45 the URDF ultrasonics suggest -- "
                         "mounting there puts the camera inside the bodywork and "
                         "it renders solid black), so this sits just proud of the "
                         "front face, above the ultrasonics at z=0.16 and below "
                         "the 0.352 m deck: where the real MiR100 carries its "
                         "D435 pair.")
parser.add_argument("--body-camera-tilt", type=float, default=25.0,
                    help="Downward pitch in degrees. NOTE: this is an "
                         "engineering choice, not a MiR spec -- mir_description's "
                         "own mir_100_v1.urdf.xacro carries no chassis camera at "
                         "all (its only camera is a D435i someone bolted to "
                         "ur_flange for the arm), so there is nothing here to "
                         "copy. 25 deg was picked from coverage: at 0.30 m with "
                         "the D435's 42.5 deg VFOV it sees ground from 0.30 m to "
                         "4.58 m ahead of the bumper, putting the horizon right "
                         "at the top edge. Shallower wastes the upper half on "
                         "wall and ceiling (the wrist camera's failing); steeper "
                         "loses the 2 m band where doorways and obstacles appear. "
                         "If you have the MiR Robot Reference Guide, back the "
                         "angle out of its 3D-camera detection volume instead.")
parser.add_argument("--body-camera-rot", type=float, nargs=3,
                    default=[90.0, 0.0, -90.0], metavar=("RX", "RY", "RZ"),
                    help="Optical-frame rotation, XYZ order. USD applies it "
                         "as Rz*Ry*Rx and a USD camera looks down its own -Z "
                         "with +Y as image-up, so rx=+90 gives Z_cam=-X_parent "
                         "(camera faces forward) and Y_cam=+Z_parent (image up = "
                         "world up). rx=-90 -- the value the wrist camera above "
                         "inherits from the URDF -- points the camera BACKWARD "
                         "and inverts the picture. Verified with the depth image "
                         "rather than by eye: for a level camera the near floor "
                         "must be the SHALLOWEST rows, and on the wrist camera "
                         "they come out at the TOP (2.51 m at the top vs 4.12 m "
                         "in the middle, nothing at the bottom), which is what an "
                         "inverted frame looks like. displayColor markers are no "
                         "use for this check -- RTX ignores them without a bound "
                         "material.")
parser.add_argument("--body-camera-res", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"),
                    help="Match the wrist camera so the same preprocessing works")
parser.add_argument("--reset-robot-xy", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("X", "Y"),
                    help="where --reset-file puts the robot back, always at yaw "
                         "0. Its heading decides how much of an episode is spent "
                         "searching before the first person comes into view, so "
                         "leaving it wherever the previous run stopped makes two "
                         "runs incomparable.")
parser.add_argument("--person-home", action="append", default=[],
                    metavar="NAME:X,Y[,YAW]",
                    help="override where --reset-file sends this person back to. "
                         "Worth setting because the platform camera sees only "
                         "75 deg: two people further apart than that cannot both "
                         "be in frame however the robot turns, and a policy that "
                         "is handed a blank crop for one of them has no way to "
                         "choose between them. Placing both within about 45 deg "
                         "of each other guarantees every episode opens with both "
                         "visible. Repeat per person.")
parser.add_argument("--reset-file", default="",
                    help="poll this path; whenever its mtime changes, teleport "
                         "every walking person back to the pose the scene spawned "
                         "them at and restart their random walk. Write an integer "
                         "into the file to seed that walk. Lets consecutive "
                         "episodes open from an identical configuration, which "
                         "is what makes two policy runs comparable.")
parser.add_argument("--walk-person", action="store_true",
                    help="Drive /World/Person along a random-waypoint walk and "
                         "publish its ground-truth pose on /person_pose "
                         "(geometry_msgs/PoseStamped). The person has to be "
                         "moved from inside this process: it is a USD prim, and "
                         "the ROS side lives in a container that cannot touch "
                         "the stage. The pose topic is what the demonstration "
                         "expert follows -- the policy itself only ever gets "
                         "the camera.")
parser.add_argument("--corner-cameras", action="store_true",
                    help="add two cameras at opposite corners of the room, "
                         "looking in across it. Oblique rather than overhead: "
                         "measured on the overhead view a person is about 18 px "
                         "across and only the head and shoulders carry colour, "
                         "while from a corner the whole torso faces the lens. "
                         "Two of them, diagonally opposite, so one sees round "
                         "whatever the other has hidden behind a body.")
parser.add_argument("--corner-camera-room", type=float, nargs=4,
                    default=[-8.0, 8.0, -6.0, 6.0],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="room bounds the corners are taken from; must match "
                         "the scene, or the cameras end up inside the walls.")
parser.add_argument("--corner-camera-height", type=float, default=5.0,
                    help="5 m. Chosen for the ground projection, which is what "
                         "turns a detection into a bearing the robot can act "
                         "on. At 2 m the ray is nearly parallel to the floor "
                         "and one pixel of detection error moves the estimate "
                         "0.31 m; at 5 m it is 0.067 m. Height costs almost "
                         "nothing in apparent size here -- a 1.73 m person is "
                         "77 px tall from 2 m and 71 px from 5 m, because the "
                         "slant range barely changes over a 9.3 m room.")
parser.add_argument("--corner-camera-inset", type=float, default=0.5,
                    help="how far in from the corner, so the camera is not "
                         "embedded in the wall geometry.")
parser.add_argument("--corner-camera-aim-z", type=float, default=0.9,
                    help="height aimed at, in metres. Torso height rather than "
                         "the floor: the torso is what carries the colour that "
                         "identifies each person.")
parser.add_argument("--corner-camera-res", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"))
parser.add_argument("--corner-camera-aperture", type=float, nargs=2,
                    default=[2.971, 2.229], metavar=("HORIZ", "VERT"),
                    help="75x60 deg, matching the other cameras.")
parser.add_argument("--person-orbit", action="store_true",
                    help="walk the people on circles around a fixed centre "
                         "instead of between random waypoints. For the "
                         "two-colour task with a stationary robot: random "
                         "wandering left the distractor out of frame 100 %% of "
                         "the time (measured over 900 ticks), so the "
                         "instruction had nothing to choose between. Orbiting "
                         "brings both past the camera again and again.")
parser.add_argument("--person-orbit-centre", type=float, nargs=2,
                    default=[0.0, 0.0], metavar=("X", "Y"),
                    help="normally the robot's own position, so the people "
                         "circle it.")
parser.add_argument("--person-orbit-radii", type=float, nargs="*",
                    default=[3.5, 3.5], metavar="R",
                    help="orbit radius per person, in the order given to "
                         "--people. EQUAL is deliberate. Unequal radii put the "
                         "two people at reliably different distances, and "
                         "distance then identifies them: a policy can answer "
                         "'which one is purple' with 'the nearer one' and score "
                         "well while never once looking at the colour or "
                         "reading the instruction -- and the lidar sectors in "
                         "observation.state hand it that shortcut directly. At "
                         "equal radii there is no distance cue to exploit. The "
                         "reason unequal radii were needed before -- the faster "
                         "walker passing through the slower one -- is handled "
                         "by --person-orbit-bounce instead.")
parser.add_argument("--person-orbit-bounce", type=float, default=25.0,
                    metavar="DEG",
                    help="when two people on the same circle close to within "
                         "this angle, both reverse and draw a new speed. Keeps "
                         "them from walking through each other without needing "
                         "different radii, and the new speeds stop the pair "
                         "settling into one repeating pattern. 0 disables.")
parser.add_argument("--person-orbit-flip-deg", type=float, default=15.0,
                    help="how closely the robot must be facing someone for "
                         "--person-orbit-flip-secs to count as tracking them.")
parser.add_argument("--person-orbit-flip-secs", type=float, default=2.0,
                    help="after being tracked this long, a person reverses and "
                         "changes speed. Applies to WHOEVER is being looked at, "
                         "not to the episode's target: the simulator is not "
                         "told which person the instruction names, so a "
                         "reversal can never leak the answer -- it only ever "
                         "happens once the robot is already correct. It also "
                         "removes the absorbing state, where the robot faces "
                         "the target, commands nothing, and every remaining "
                         "frame teaches 'stay still'. 0 disables.")
parser.add_argument("--person-orbit-speed-range", type=float, nargs=2,
                    default=[0.15, 0.32], metavar=("MIN", "MAX"),
                    help="speeds drawn from here on every reversal.")
parser.add_argument("--person-orbit-radius", type=float, default=3.0,
                    help="far enough that the whole of a 1.73 m person is in "
                         "frame (measured: 1.54 m at this camera's 60 deg "
                         "VFOV), close enough that the colour is unambiguous.")
parser.add_argument("--person-orbit-speeds", type=float, nargs="*",
                    default=[0.20, 0.26], metavar="RAD_S",
                    help="angular speed per person, in the order given to "
                         "--people. They MUST differ: at equal speeds the pair "
                         "holds a fixed relative angle forever, so the scene "
                         "shows one configuration and the robot can pass by "
                         "memorising it. Different speeds sweep the separation "
                         "through every value, including both-in-frame and "
                         "one-only.")
parser.add_argument("--person-orbit-phases", type=float, nargs="*", default=[],
                    metavar="DEG",
                    help="starting angles. Default spreads them evenly.")
parser.add_argument("--global-camera", action="store_true",
                    help="add a fixed overhead camera looking straight down at "
                         "the room, published as /global_camera/color/image_raw. "
                         "For the two-person task: measured with the onboard "
                         "camera alone, the two were in frame together 0 %% of "
                         "the time over 800 ticks, so the instruction had "
                         "nothing to choose between and the policy could ignore "
                         "it. This view always contains both. It is world-fixed, "
                         "so it does not rotate with the robot -- pair it with "
                         "the platform camera rather than replacing it, or the "
                         "policy has no egocentric heading cue.")
parser.add_argument("--global-camera-height", type=float, default=12.0,
                    help="metres above the floor. Must be high enough that the "
                         "whole walkable area fits, or the view reintroduces "
                         "the very problem it exists to remove -- but no higher, "
                         "because the people shrink with it: at 12 m the 16x12 m "
                         "room fits inside an 18.4 x 13.9 m view at ~35 px/m, "
                         "and a person is about 18 px across in 640x480.")
parser.add_argument("--global-camera-yaw", type=float, default=0.0,
                    help="which world axis runs across the image. 0 puts world "
                         "+X along the long side of the frame, which is what "
                         "makes a room wider than it is deep fit without crops.")
parser.add_argument("--global-camera-res", type=int, nargs=2, default=[640, 480],
                    metavar=("W", "H"))
parser.add_argument("--global-camera-aperture", type=float, nargs=2,
                    default=[2.971, 2.229], metavar=("HORIZ", "VERT"),
                    help="75x60 deg against the 1.93 mm focal length, matching "
                         "the platform camera so the two views are directly "
                         "comparable.")
parser.add_argument("--people", nargs="*", default=[], metavar="NAME:PRIM",
                    help="drive several people instead of one, e.g. "
                         "purple:/World/PersonPurple yellow:/World/PersonYellow. "
                         "Each gets its own TF frame named person_<NAME>, so a "
                         "consumer picks a target by frame name. Overrides "
                         "--person-prim when given; --walk-person is still what "
                         "switches the walking on.")
parser.add_argument("--person-prim", default="/World/Env/Person",
                    help="--world is referenced in at /World/Env, so a scene "
                         "that authors /World/Person ends up here. If the path "
                         "is missing, the stage is searched for a prim named "
                         "Person before giving up.")
parser.add_argument("--person-speed", type=float, nargs=2, default=[0.5, 0.9],
                    metavar=("MIN", "MAX"),
                    help="walking speed range (m/s). A MiR100 tops out around "
                         "1.1 m/s, so a person much above 0.9 cannot be caught.")
parser.add_argument("--person-turn-rate", type=float, default=60.0,
                    help="how fast the person can change heading (deg/s). Real "
                         "walking does not pivot instantly, and a target that "
                         "does teleport its heading produces demonstrations the "
                         "robot cannot imitate.")
parser.add_argument("--person-area", type=float, nargs=4,
                    default=[-6.5, 6.5, -4.5, 4.5],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="waypoints are sampled here -- inside the room walls "
                         "with a margin so the person never clips a wall.")
parser.add_argument("--person-pause", type=float, nargs=2, default=[0.0, 1.5],
                    metavar=("MIN", "MAX"),
                    help="seconds to stand still on reaching a waypoint. Some "
                         "pausing is realistic, but every stationary frame is a "
                         "frame teaching 'stopped -> stay stopped', which is the "
                         "absorbing state that made the colour ACT policies "
                         "unable to move at all. Keep it short.")
parser.add_argument("--person-seed", type=int, default=0)
parser.add_argument("--person-avoid-hard", type=float, default=1.2,
                    help="inside this the person steers away whatever direction "
                         "the robot is in, and at full strength. Covers the "
                         "robot creeping up from behind, which the forward cone "
                         "below deliberately ignores.")
parser.add_argument("--person-avoid", type=float, default=2.6,
                    help="the person steers round the robot inside this radius "
                         "(m), and only when it is roughly in front of them. "
                         "Being forward-only is what makes a radius above the "
                         "follower's 2.0 m standoff safe: a robot trailing "
                         "behind is outside the cone and never triggers it. An "
                         "all-directions version at this radius left the "
                         "avoidance permanently on and the person turned in "
                         "circles instead of walking -- 0.29 m travelled and "
                         "778 deg of turning over 40 s.")
parser.add_argument("--person-gait", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="drive a procedural walk cycle on the character's rig. "
                         "The asset has a skeleton but no animation clip, so "
                         "without this it slides along in its T-pose. Off falls "
                         "back to that, which still works for tracking.")
parser.add_argument("--person-stride", type=float, default=0.75,
                    help="metres per step; the cycle is phased on distance "
                         "travelled, not time, so the feet do not skate when "
                         "the person changes speed or stops.")
parser.add_argument("--platform-camera", action="store_true",
                    help="Add a camera on the MiR's PLATFORM (the deck the arm "
                         "sits on) and publish it as "
                         "/platform_camera/color/image_raw. Separate from "
                         "--body-camera on purpose: that one is aimed at the "
                         "ground (20 deg down for the colour task) and its top "
                         "edge sits 1.35 deg above horizontal, so it can only "
                         "ever see up to ~0.9 m however far away the subject is "
                         "-- half a standing person. Giving this one its own "
                         "topic keeps the two datasets from silently mixing: "
                         "same topic name with a different view is the failure "
                         "mode that is hardest to notice, because the images "
                         "still arrive and only the geometry is wrong.")
parser.add_argument("--platform-camera-xyz", type=float, nargs=3,
                    default=[0.35, 0.0, 1.00], metavar=("X", "Y", "Z"),
                    help="Mount point in base_link. Default sits above the deck "
                         "(0.352 m) on a short mast, where a real MiR carries an "
                         "add-on pan-tilt head. Measured: between z=0.8 and "
                         "z=1.4 the fraction of a 1.73 m person in frame barely "
                         "moves (tilt dominates at these ranges), so height is "
                         "a mounting choice rather than an optical one.")
parser.add_argument("--platform-camera-tilt", type=float, default=3.0,
                    help="Downward pitch in degrees. This is the variable that "
                         "matters: at 20 deg (the colour task's value) a person "
                         "is only ever ~50%% in frame at any range; at 0-5 deg "
                         "it is 68%% at 1.5 m, 90%% at 2 m and all of them "
                         "beyond 3 m.")
parser.add_argument("--platform-camera-rot", type=float, nargs=3,
                    default=[90.0, 0.0, -90.0], metavar=("RX", "RY", "RZ"),
                    help="Optical-frame rotation. See --body-camera-rot: rx=+90 "
                         "is what keeps the image the right way up.")
parser.add_argument("--platform-camera-res", type=int, nargs=2,
                    default=[640, 480], metavar=("W", "H"))
parser.add_argument("--platform-camera-aperture", type=float, nargs=2,
                    default=[2.971, 2.229], metavar=("HORIZ", "VERT"),
                    help="Sensor aperture in mm against the 1.93 mm focal "
                         "length. Default is 75x60 deg, wider than the D435's "
                         "69x42.7: measured on this scene, 42.7 deg VFOV cannot "
                         "fit a 1.73 m person until 2.21 m away, which is "
                         "further than a follower wants to sit. At 60 deg the "
                         "whole person is in frame from 1.54 m, and the wider "
                         "horizontal view also keeps them from sliding out of "
                         "shot when they turn.")
parser.add_argument("--camera", action="store_true",
                    help="Enable the D435i RGB-D camera "
                         "(/realsense/{color,depth,camera_info}). Requires viewport.")
args, _ = parser.parse_known_args()
# Convert opt-in to the inverse flags the rest of the script uses.
# --rtx-lasers takes precedence: if both are given, PhysX is skipped.
args.no_lasers = not (args.lasers or args.rtx_lasers)
args.no_camera = not args.camera

# Who walks, as (tf_frame, prim_path). One entry reproduces the single-person
# scene exactly; more than one is the two-people-two-colours task, where the
# frame name is how a consumer says WHICH person it means.
PEOPLE = []
for _spec in args.people:
    if ":" not in _spec:
        print(f"[mir_isaac_sim] ERROR: --people wants NAME:PRIM, got {_spec!r}",
              file=sys.stderr)
        sys.exit(1)
    _nm, _pp = _spec.split(":", 1)
    PEOPLE.append((_nm, _pp))
if not PEOPLE:
    PEOPLE = [("person", args.person_prim)]

if not os.path.isfile(args.usd):
    print(f"[mir_isaac_sim] ERROR: USD not found: {args.usd}\n"
          f"  Generate it first with isaac_sim/convert_mir_to_usd.sh", file=sys.stderr)
    sys.exit(1)

from isaacsim import SimulationApp  # noqa: E402

# enable_cameras=True is required for RTX-lidar / camera ROS publishers to work
# even in headless mode — without it the SDG pipeline does not register the
# `PostProcessDispatch*`/`LdrColor*`/`DistanceToImagePlane*` annotators that the
# laser_scan / image writers depend on. Every official ROS2 sensor sample uses
# this flag (see standalone_examples/api/isaacsim.ros2.bridge/*.py).
simulation_app = SimulationApp({
    "renderer": "RaytracedLighting",
    "headless": args.headless,
    "enable_cameras": True,
})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import usdrt.Sdf  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane  # noqa: E402
from isaacsim.core.utils import extensions, stage, viewports  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

# enable ROS2 bridge extension. omni.syntheticdata is what registers the
# `PostProcessDispatchIsaacSimulationGate` annotator that the RTX lidar
# ROS2 LaserScan writer depends on; enable it explicitly so the writer can
# attach (in some standalone runs it does not come up until first render).
extensions.enable_extension("omni.syntheticdata")
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
simulation_app.update()

ROBOT_PRIM_PATH = "/World/Robot"

simulation_context = SimulationContext(stage_units_in_meters=1.0)

# ------------------------------------------------------------------- scene
if args.top_down:
    # straight-down view; tiny Y offset gives the camera a defined "up" so the
    # look-down isn't degenerate (eye exactly above target).
    viewports.set_camera_view(eye=np.array([0.0, -0.01, args.top_down_height]),
                              target=np.array([0.0, 0.0, 0.0]))
else:
    viewports.set_camera_view(eye=np.array([3.0, 3.0, 2.0]),
                              target=np.array([0.0, 0.0, 0.5]))

# ground + light so the robot has something to stand on and be visible
GroundPlane(prim_path="/World/GroundPlane", size=50.0, z_position=0.0)
stage_handle = simulation_context.stage
dome = UsdLux.DomeLight.Define(stage_handle, "/World/DomeLight")
dome.CreateIntensityAttr(1000.0)

# diagnostic: closed box of walls around the robot (every direction has a wall)
if args.test_walls:
    _D = 3.0  # wall distance from origin
    _walls = [(_D, 0.0, 0.1, _D), (-_D, 0.0, 0.1, _D),
              (0.0, _D, _D, 0.1), (0.0, -_D, _D, 0.1)]
    for _i, (_x, _y, _hx, _hy) in enumerate(_walls):
        _p = f"/World/TestWall{_i}"
        _cube = UsdGeom.Cube.Define(stage_handle, _p)
        _cube.CreateSizeAttr(2.0)
        _xf = UsdGeom.XformCommonAPI(_cube.GetPrim())
        _xf.SetTranslate(Gf.Vec3d(_x, _y, 0.5))
        _xf.SetScale(Gf.Vec3f(_hx, _hy, 0.5))  # half-extents (cube size 2)
        UsdPhysics.CollisionAPI.Apply(_cube.GetPrim())
    carb.log_warn(f"[mir_isaac_sim] TEST: {len(_walls)} walls at +-{_D} m around robot")
    simulation_app.update()

# optional environment (e.g. the converted Gazebo maze)
if args.world:
    if not os.path.isfile(args.world):
        carb.log_error(f"[mir_isaac_sim] --world not found: {args.world}")
    else:
        stage.add_reference_to_stage(usd_path=args.world, prim_path="/World/Env")
        carb.log_warn(f"[mir_isaac_sim] environment: {args.world}")
        simulation_app.update()

# load the robot USD as a reference under /World/Robot
stage.add_reference_to_stage(usd_path=args.usd, prim_path=ROBOT_PRIM_PATH)
simulation_app.update()


# --------------------------------------------------- find articulation root
def find_articulation_root(root_path):
    """Return the prim path that carries the Articulation Root API."""
    for prim in stage_handle.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_path):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return p
    # fall back to the reference root (SimulationContext will treat it as the
    # articulation if a root API lives anywhere beneath it)
    return root_path


robot_prim_path = args.robot_prim or find_articulation_root(ROBOT_PRIM_PATH)
carb.log_warn(f"[mir_isaac_sim] articulation root: {robot_prim_path}")


# ------------------------------ articulation tuning -------------------------
# Two articulation-level knobs that both target the gripper jitter:
#   * self-collisions: the Robotiq 2F-85 finger links (knuckle / inner_knuckle /
#     finger_tip on both sides) sit very close and, driven together through the
#     PhysX mimic coupling, keep grazing each other. Those internal contacts
#     fight the position drive and make the gripper buzz. Disabling
#     self-collisions turns off collision *between links of the same
#     articulation* only — external objects are separate bodies, so grasping
#     still works — and MoveIt still does its own SRDF self-collision checks.
#   * solver iteration counts: the Robotiq finger linkage is over-constrained by
#     the hard PhysX mimic joints (knuckle/inner_knuckle/finger_tip all geared to
#     the master); the importer's default ~4 position iterations can't converge
#     them each step, so they jitter. More iterations settle the coupling.
_root_prim = stage_handle.GetPrimAtPath(robot_prim_path)
if _root_prim and _root_prim.IsValid():
    _art_api = PhysxSchema.PhysxArticulationAPI.Apply(_root_prim)
    if not args.keep_self_collisions:
        _art_api.CreateEnabledSelfCollisionsAttr().Set(False)
        carb.log_warn("[mir_isaac_sim] self-collisions DISABLED on articulation "
                      f"{robot_prim_path}")
    _art_api.CreateSolverPositionIterationCountAttr().Set(
        int(args.solver_position_iterations))
    _art_api.CreateSolverVelocityIterationCountAttr().Set(
        int(args.solver_velocity_iterations))
    carb.log_warn(f"[mir_isaac_sim] solver iterations: pos="
                  f"{args.solver_position_iterations} "
                  f"vel={args.solver_velocity_iterations}")
else:
    carb.log_warn("[mir_isaac_sim] could not resolve articulation root prim "
                  "for self-collision / solver tuning")


# ------------------------------ depenetration clamp -------------------------
# Nav2 will occasionally wedge the base against a wall corner (the maze's 0.93 m
# gap is the reliable spot) and keep commanding into it while the progress
# checker counts down. PhysX lets the penetration build up and then resolves it
# in a single step at an unbounded velocity — the robot is fired out of the map
# and every downstream consumer (AMCL, the costmaps) goes with it. Clamping the
# depenetration velocity turns that catastrophic ejection into a slow push-out
# the controller can recover from. Applies to every rigid body in the robot,
# since any link can be the one in contact.
# Two-phase on purpose:
#   * BEFORE play() we only apply the schema and author a permissive value.
#     Applying an API schema while the sim is playing re-parses the physics
#     scene mid-step and segfaults Isaac, so the schema has to exist up front.
#   * The real clamp is set AFTER the startup arm-home teleport has settled.
#     set_joint_positions() is instantaneous and leaves transient penetrations
#     that PhysX needs full speed to resolve; clamped from the start, those
#     penetrations persist and the contact forces flip the base within ~50 s of
#     sim time (measured: 212/308 samples tipped, vs 0/308 unclamped).
#     Writing a new *value* to an existing attribute mid-run is safe.
_DEPEN_ATTRS = []
UNCLAMPED_DEPENETRATION = 1.0e6   # PhysX default is effectively unlimited


def init_depenetration_attrs():
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        try:
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = api.CreateMaxDepenetrationVelocityAttr()
        except Exception:  # noqa: BLE001 — fall back to the raw attribute
            attr = prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                                        Sdf.ValueTypeNames.Float)
        attr.Set(float(UNCLAMPED_DEPENETRATION))
        _DEPEN_ATTRS.append(attr)


def set_max_depenetration_velocity(v):
    if v <= 0.0:
        print("[mir_isaac_sim] depenetration clamp disabled", flush=True)
        return
    for attr in _DEPEN_ATTRS:
        attr.Set(float(v))
    # print, not carb.log_warn: post-play carb messages do not reach stdout.
    # Read one back so the line is proof the value took, not just that we asked.
    back = _DEPEN_ATTRS[0].Get() if _DEPEN_ATTRS else None
    print(f"[mir_isaac_sim] max depenetration velocity {v} m/s -> "
          f"{len(_DEPEN_ATTRS)} rigid bodies (readback {back})", flush=True)


init_depenetration_attrs()


# ------------------------------ gripper armature ----------------------------
# The Robotiq finger inertias are ~1e-5 kg*m^2. A stiff position drive + the hard
# mimic coupling oscillate far faster than the sim step can integrate -> buzzing.
# Adding armature (rotor inertia) on the 6 gripper joints raises their effective
# inertia into the integrable range — the standard PhysX fix for low-inertia
# joint jitter. Set on the USD joint prims before play().
def set_joint_armature(match, armature):
    """Apply armature to joints. `match` is a substring (str) or an explicit set
    of joint names (any iterable)."""
    if armature <= 0.0:
        return
    names = None if isinstance(match, str) else set(match)
    n = 0
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        nm = prim.GetName()
        if (nm not in names) if names is not None else (match not in nm):
            continue
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue
        try:
            api = PhysxSchema.PhysxJointAPI.Apply(prim)
            api.CreateArmatureAttr().Set(float(armature))
        except Exception:  # noqa: BLE001 — fall back to the raw attribute
            prim.CreateAttribute("physxJoint:armature",
                                 Sdf.ValueTypeNames.Float).Set(float(armature))
        n += 1
    carb.log_warn(f"[mir_isaac_sim] armature {armature} -> {n} joints ({match})")


set_joint_armature("robotiq_85_", args.gripper_armature)


# ------------------------------ make the drive wheels velocity-controlled --
# The arm/gripper joints keep their position drives (set by the URDF
# importer); the two MiR drive wheels must follow velocity commands, so we
# switch their angular drive to velocity mode (stiffness 0, damping high).
def set_velocity_drive(joint_substr, damping=1.0e4, max_force=1.0e6):
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        name = prim.GetName()
        if joint_substr not in name:
            continue
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateDampingAttr().Set(damping)
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(max_force)
        carb.log_warn(f"[mir_isaac_sim] velocity drive -> {prim.GetPath()}")


for wheel in ("left_wheel_joint", "right_wheel_joint"):
    set_velocity_drive(wheel, damping=args.wheel_drive_damping)


# ------------------------------ wheel-ground friction -----------------------
# The URDF gives the wheels mu=300 only inside <gazebo> tags, which Isaac drops
# on import -> the drive wheels sit on the default low-friction ground and SLIP.
# A velocity-driven wheel then spins at the commanded rad/s but the 67 kg base
# moves SLOWER than commanded (cmd_vel speed mismatch) and differential turning
# becomes erratic (one wheel grips, the other slips -> over-turn or no turn).
# Fix: bind a high-friction PhysicsMaterial to the drive-wheel links AND the
# ground, with combine mode "max" so the contact uses the high value.
def _make_friction_material(path, friction):
    mat = UsdShade.Material.Define(stage_handle, path)
    prim = mat.GetPrim()
    pm = UsdPhysics.MaterialAPI.Apply(prim)
    pm.CreateStaticFrictionAttr().Set(float(friction))
    pm.CreateDynamicFrictionAttr().Set(float(friction))
    pm.CreateRestitutionAttr().Set(0.0)
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(prim)
    pxm.CreateFrictionCombineModeAttr().Set("max")
    return mat


def _bind_physics_material(prim, mat):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


if args.wheel_friction > 0.0:
    # Bind high friction ONLY to the 2 drive-wheel links, with combine mode
    # "max" so the contact uses the high value even against the default ground.
    # Do NOT bind it to the ground: a high-friction ground would also make the 4
    # passive casters grip hard, and grippy casters cannot slide/scrub during an
    # in-place pivot -> they block rotation. Leaving the ground at default keeps
    # the casters low-friction (free to scrub) while the drive wheels still grip.
    _fric_mat = _make_friction_material("/World/PhysicsMaterials/drive_wheel",
                                        args.wheel_friction)
    _bound = []
    for _wl in ("left_wheel_link", "right_wheel_link"):
        for prim in stage_handle.Traverse():
            if (prim.GetName() == _wl
                    and prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH)):
                _bind_physics_material(prim, _fric_mat)
                _bound.append(_wl)
                break
    carb.log_warn(f"[mir_isaac_sim] drive-wheel friction {args.wheel_friction} "
                  f"(ground left default so casters can pivot) -> {_bound}")


# ------------------------------ settle the chassis --------------------------
# Idle, the base is not perfectly still: it micro-vibrates / wobbles in yaw on
# its free casters (odom twist shows angular ~0.01-0.04 rad/s with all axes
# non-zero). That wobble is small but the ~0.8 m arm amplifies it into visible
# end-effector motion — which reads as "the arm is moving" even though the UR
# joints are steady. Adding linear+angular damping to the base_link rigid body
# bleeds that energy off. It does NOT block commanded driving: diff_cont drives
# the wheels, and a modest chassis damping is just realistic rolling/turn drag.
def set_body_damping(link_name, linear, angular):
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        if prim.GetName() != link_name:
            continue
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr().Set(float(linear))
        rb.CreateAngularDampingAttr().Set(float(angular))
        carb.log_warn(f"[mir_isaac_sim] chassis damping lin={linear} "
                      f"ang={angular} -> {prim.GetPath()}")
        return
    carb.log_warn(f"[mir_isaac_sim] chassis link '{link_name}' not found "
                  "for damping")


set_body_damping("base_link", args.base_linear_damping, args.base_angular_damping)


# ------------------------------ free the passive caster joints --------------
# The 4 MiR casters (swivel "rotation" joint + rolling "wheel" joint each, 8
# joints total) carry NO <dynamics> in the URDF, so the importer gives every one
# of them a default position drive (stiffness ~100). That LOCKS each caster at
# angle 0 — 8 joints clamping the chassis to the ground like brakes — so even a
# correctly-commanded drive wheel cannot move the robot. Casters must be passive
# free-wheels: zero stiffness so the importer's default position drive no longer
# clamps the chassis to the ground. The rolling wheel joints get zero
# damping/force (truly free). The swivel "rotation" joints instead get a small
# damping: a real MiR caster trails behind its swivel axis
# (caster_wheel_dx = -0.0382 m), so an UNdamped swivel turns solver noise into a
# yaw torque on the chassis, and because the drive wheels are in velocity mode
# (they hold velocity ~0 but provide no heading restoring force) that torque
# integrates into a slow continuous in-place rotation when idle. Light swivel
# damping bleeds that off without re-locking the casters (that earlier warning
# was about the drive wheels' 1e4 damping). NOTE: maxForce must be > 0 or the
# damping has no effect at all.
def set_passive_drive(joint_substr, damping=0.0, max_force=0.0):
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        if joint_substr not in prim.GetName():
            continue
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(float(damping))
        drive.CreateMaxForceAttr().Set(float(max_force))
        carb.log_warn(f"[mir_isaac_sim] passive drive (damping={damping}) "
                      f"-> {prim.GetPath()}")


# swivel joints: light damping (tunable) to stop the idle in-place rotation
set_passive_drive("caster_rotation_joint",
                  damping=args.caster_swivel_damping,
                  max_force=(1.0e3 if args.caster_swivel_damping > 0.0 else 0.0))
# rolling wheel joints: fully free
set_passive_drive("caster_wheel_joint")


# ------------------------------ stiffen the arm/gripper position drives -----
# The URDF importer gives every joint a soft position drive (stiffness ~100),
# which is far too weak for the UR5 to hold against gravity -> the arm droops
# and never tracks the MoveIt target. Raise stiffness/damping on the 6 UR
# joints (and the gripper master) so they hold and follow position commands.
def set_position_drive(joint_names, stiffness, damping, max_force=1.0e7):
    """Stiffen the angular position drive on the named joints. (Targets are set
    later through the Articulation API, which—unlike the raw USD DriveAPI—handles
    the per-joint axis sign correctly; setting targetPosition here drives some UR
    joints to the mirror pose.)"""
    targets = set(joint_names)
    for prim in stage_handle.Traverse():
        if not prim.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            continue
        if prim.GetName() not in targets:
            continue
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.Joint)):
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(float(stiffness))
        drive.CreateDampingAttr().Set(float(damping))
        drive.CreateMaxForceAttr().Set(max_force)
        carb.log_warn(f"[mir_isaac_sim] position drive ({stiffness}/{damping}) "
                      f"-> {prim.GetName()}")


# UR initial pose = ur_description initial_positions.yaml (the ros2_control
# state initial_value), so Isaac and ros2_control agree on the start state.
UR_HOME = {
    "ur_shoulder_pan_joint": 0.0,
    "ur_shoulder_lift_joint": -1.57,
    "ur_elbow_joint": 0.0,
    "ur_wrist_1_joint": -1.57,
    "ur_wrist_2_joint": 0.0,
    "ur_wrist_3_joint": 0.0,
}

# The pose above reaches FORWARD, not up: measured, base_footprint ->
# ur_wrist_3_link is [0.642, 0.251, 1.463], i.e. ~0.2 m past the front of the
# 0.89 m chassis, at 1.46 m -- far above the SICK scan plane. The robot is
# therefore longer than anything it can see, and drives its arm into beds and
# door frames while the laser still reports 0.4 m of clearance; it then cannot
# move at all. Folding the elbow up brings the whole arm back over the deck.
#
# This is opt-in rather than the new default because the recorded VLA episodes
# were collected with the arm forward (it is visible in the camera frames), so
# flipping the default would put the trained policy out of distribution.
# Measured with tf2_echo base_footprint -> ur_wrist_3_link, chassis half-length
# 0.445 m, circumscribed radius 0.531 m:
#   default  [0, -1.57,  0.0, -1.57]  -> 0.682 m ahead, 1.41 m up  (+24 cm proud)
#   elbow-up [0, -1.57,  1.57,  0.0]  -> 0.53-0.70 m, 1.04-1.36 m  (still proud,
#                                        and now at bed height, which is worse)
#   this     [0, -2.60,  2.60, -1.57]  -> 0.460 m ahead, 1.09 m up (+1.5 cm)
# 0.460 m is inside the chassis's own circumscribed radius, so the arm no longer
# sticks out of the silhouette either driving forward or turning on the spot.
UR_FOLDED = dict(UR_HOME, ur_shoulder_lift_joint=-2.60,
                 ur_elbow_joint=2.60, ur_wrist_1_joint=-1.57)
set_position_drive(UR_HOME.keys(), stiffness=args.arm_stiffness,
                   damping=args.arm_damping, max_force=args.arm_max_force)
# armature on the 6 UR joints — same low-inertia jitter fix as the gripper, most
# needed on the small-inertia wrist joints.
set_joint_armature(set(UR_HOME.keys()), args.arm_armature)
# the Robotiq master joint (the 5 mimic joints follow it via PhysX coupling)
set_position_drive(["robotiq_85_left_knuckle_joint"],
                   stiffness=args.gripper_stiffness, damping=args.gripper_damping,
                   max_force=1.0e3)
simulation_app.update()

# ------------------------------------------------ build the ROS2 action graph
graph_keys = og.Controller.Keys
nodes = [
    ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
]
connect = [
    ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
    ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
    ("Context.outputs:context", "PublishJointState.inputs:context"),
    ("Context.outputs:context", "PublishClock.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
]
set_values = [
    ("PublishJointState.inputs:topicName", args.states_topic),
    ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
]

# One Subscribe + ArticulationController chain PER command topic. The 3
# ros2_control hardware components (UR arm, MiR base, Robotiq gripper) each
# publish to their OWN topic, so a single shared /isaac_joint_commands no longer
# has 3 publishers interleaving — which starved the arm of commands. Each
# ArticulationController applies only the joints in its message to the same
# articulation.
# TopicBasedSystem publishes a JointState whose `name` field lists ALL joints of
# the hardware component but whose value arrays (position/velocity/effort) only
# cover the joints that actually have a *command* interface. The MiR base lists
# 10 joints (2 drive wheels + 8 passive casters) yet only the 2 wheels carry a
# velocity command interface -> name has 10 entries, velocity has 2. The Isaac
# ArticulationController requires len(command) == len(jointNames), so it drops
# the whole message and the wheels never move. For such topics we pin an explicit
# jointNames token array (the actually-commanded joints, in command-interface
# declaration order) instead of forwarding the 10-name list from the message.
EXPLICIT_JOINTS = {
    "isaac_base_commands": ["left_wheel_joint", "right_wheel_joint"],
}

for i, topic in enumerate(args.command_topics):
    sub = f"SubscribeJointState{i}"
    art = f"ArticulationController{i}"
    nodes += [
        (sub, "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        (art, "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    connect += [
        ("OnImpulseEvent.outputs:execOut", f"{sub}.inputs:execIn"),
        ("OnImpulseEvent.outputs:execOut", f"{art}.inputs:execIn"),
        ("Context.outputs:context", f"{sub}.inputs:context"),
        (f"{sub}.outputs:positionCommand", f"{art}.inputs:positionCommand"),
        (f"{sub}.outputs:velocityCommand", f"{art}.inputs:velocityCommand"),
        (f"{sub}.outputs:effortCommand", f"{art}.inputs:effortCommand"),
    ]
    set_values += [
        (f"{sub}.inputs:topicName", topic),
        (f"{art}.inputs:robotPath", robot_prim_path),
    ]
    explicit = next((v for k, v in EXPLICIT_JOINTS.items() if k in topic), None)
    if explicit is not None:
        # pin the commanded joints so they match the short value arrays
        set_values.append((f"{art}.inputs:jointNames", explicit))
        carb.log_warn(f"[mir_isaac_sim] command chain {i}: {topic} -> "
                      f"pinned jointNames {explicit}")
    else:
        # forward the message's own jointNames (lengths already match)
        connect.append((f"{sub}.outputs:jointNames", f"{art}.inputs:jointNames"))
        carb.log_warn(f"[mir_isaac_sim] command chain {i}: subscribes {topic}")

# optional: ground-truth odometry (odom -> base_footprint) straight from Isaac
if args.publish_odom:
    nodes += [
        ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("PublishRawTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]
    connect += [
        ("OnImpulseEvent.outputs:execOut", "ComputeOdometry.inputs:execIn"),
        ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
        ("ComputeOdometry.outputs:execOut", "PublishRawTF.inputs:execIn"),
        ("Context.outputs:context", "PublishOdometry.inputs:context"),
        ("Context.outputs:context", "PublishRawTF.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishRawTF.inputs:timeStamp"),
        ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
        ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
        ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
        ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
        ("ComputeOdometry.outputs:position", "PublishRawTF.inputs:translation"),
        ("ComputeOdometry.outputs:orientation", "PublishRawTF.inputs:rotation"),
    ]
    set_values += [
        ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(robot_prim_path)]),
        ("PublishOdometry.inputs:odomFrameId", args.odom_frame),
        ("PublishOdometry.inputs:chassisFrameId", args.base_frame),
        ("PublishRawTF.inputs:parentFrameId", args.odom_frame),
        ("PublishRawTF.inputs:childFrameId", args.base_frame),
    ]

if args.walk_person:
    # The person's pose has to reach the container somehow, and a raw TF frame is
    # the one channel already crossing that boundary. RawTransformTree takes the
    # translation/rotation as plain inputs, so the walk below just writes them
    # every frame -- no extra compute node, and no dependency on the person prim
    # having physics (it must not have any: a rigid body would let the robot
    # shove the target around, which is not the task).
    for _nm, _ in PEOPLE:
        _n = f"PublishPersonTF_{_nm}"
        nodes += [(_n, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree")]
        connect += [
            ("OnImpulseEvent.outputs:execOut", f"{_n}.inputs:execIn"),
            ("Context.outputs:context", f"{_n}.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", f"{_n}.inputs:timeStamp"),
        ]
        set_values += [
            (f"{_n}.inputs:parentFrameId", args.odom_frame),
            (f"{_n}.inputs:childFrameId", _nm),
        ]

try:
    og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            graph_keys.CREATE_NODES: nodes,
            graph_keys.CONNECT: connect,
            graph_keys.SET_VALUES: set_values,
        },
    )
except Exception as e:  # noqa: BLE001
    carb.log_error(f"[mir_isaac_sim] failed to build action graph: {e}")
    simulation_app.close()
    sys.exit(1)

simulation_app.update()

# DIAGNOSTIC: read back what each command chain actually wired, so we can tell
# whether the pinned jointNames took effect (carb.log_warn isn't captured in the
# redirected log, so use print).
for i, topic in enumerate(args.command_topics):
    try:
        jn = og.Controller.get(
            og.Controller.attribute(f"/ActionGraph/ArticulationController{i}.inputs:jointNames"))
        print(f"[graph_check] chain {i} ({topic}) ArticulationController jointNames = {list(jn)}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[graph_check] chain {i} ({topic}) readback failed: {e}", flush=True)


# =================================================================== SENSORS
# Lasers, IMU and the D435i RGB-D camera. Each block builds on the same /ROS_Sensors
# action graph; the OG nodes tick on render frames (push evaluator) so they keep
# running without us having to drive impulses for them.

def find_prim(name):
    """First prim under /World/Robot whose leaf name matches."""
    for p in stage_handle.Traverse():
        if p.GetName() == name and p.GetPath().pathString.startswith(ROBOT_PRIM_PATH):
            return p
    return None


def unblock_lidar_self_collision(laser_links):
    """Stop the PhysX lidars from ray-casting the robot's OWN body.

    Root cause of the "/scan is missing a section that rotates with the robot"
    bug (diagnosed 2026-06-27): the two SICK S300 origins land at the chassis
    corners (~0.43,0.24 and -0.36,-0.24, z=0.19) which are INSIDE the base
    collision box. Isaac's PhysX RangeSensor ray-casts against EVERY collider
    (Gazebo's ray sensor, by contrast, ignores its own model), so ~260° of each
    240° FOV hits the body at <1 m and never reaches the walls — the merged
    /scan ends up with two fixed empty sectors (measured: each sensor covered
    only ~100° instead of 240°, 287/541 beams returned <1 m).

    Gazebo's behaviour is reproduced here by disabling collision ONLY on the
    colliders that actually cross the horizontal laser plane (z ≈ 0.19) — i.e.
    the chassis body box. Everything else KEEPS its collision: the wheels/casters
    keep the robot on the floor, and the arm + gripper (which sit ABOVE the laser
    plane and never block a horizontal beam) keep collision so manipulation /
    grasping still work. The robot is therefore NOT collisionless — only the
    chassis stops physically bumping walls, which is fine under Nav2 (costmap
    avoidance + wheels keep floor contact). Reprojection is exact: the lidar
    origins stay on the URDF laser_links, so /f_scan + /b_scan reproject with
    ZERO offset error (unlike nudging the sensors outward).
    """
    # The robot's links are INSTANCEABLE; stage.Traverse() does not descend into
    # instance prototypes, so the collision meshes are hidden and uneditable.
    # De-instance the robot subtree first so the geometry colliders become
    # reachable and we can author collisionEnabled overrides on them.
    deinst = 0
    for p in stage_handle.Traverse():
        if p.GetPath().pathString.startswith(ROBOT_PRIM_PATH) and p.IsInstance():
            p.SetInstanceable(False)
            deinst += 1
    print(f"[mir_isaac_sim] self-collision unblock: de-instanced {deinst} robot prim(s)")

    # Laser plane height (both SICK at z≈0.19); a collider only blocks the
    # horizontal beams if its world bbox spans this z.
    xc = UsdGeom.XformCache()
    zs = [xc.GetLocalToWorldTransform(p).ExtractTranslation()[2] for p in laser_links]
    z_lo, z_hi = (min(zs), max(zs)) if zs else (0.19, 0.19)
    # ignoreVisibility=True is ESSENTIAL: Isaac's collision meshes are marked
    # invisible, and a default BBoxCache returns an EMPTY bound for invisible
    # prims -> every collider would be skipped.
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                             UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
                            useExtentsHint=True, ignoreVisibility=True)

    def is_collider(p):
        # Real collision GEOMETRY only — exclude PhysicsJoints (which also carry a
        # physics:collisionEnabled attr but have no geometry / EMPTY bbox).
        if "/joints/" in p.GetPath().pathString:
            return False
        return (p.HasAPI(UsdPhysics.CollisionAPI)
                or p.HasAPI(PhysxSchema.PhysxCollisionAPI))

    def set_collision(p, enabled):
        attr = p.GetAttribute("physics:collisionEnabled")
        if not attr:
            attr = UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr()
        attr.Set(enabled)

    if args.no_arm_collision:
        # The per-link "collisions" scope is a real prim even though the meshes
        # under it are instance proxies (which cannot be edited directly), so
        # deactivating the scope takes the whole collision subtree out of physics
        # while leaving "visuals" untouched.
        _arm = ("ur_", "robotiq")
        _off = 0
        for _p in stage_handle.Traverse():
            _path = _p.GetPath().pathString
            if _p.GetName() != "collisions" or "/joints/" in _path:
                continue
            _link = _p.GetParent().GetName()
            if _link.startswith(_arm):
                _p.SetActive(False)
                _off += 1
        carb.log_warn(f"[mir_isaac_sim] arm/gripper collision DISABLED "
                      f"({_off} link collision scopes); visuals kept")

    colliders = [p for p in stage_handle.Traverse()
                 if p.GetPath().pathString.startswith(ROBOT_PRIM_PATH) and is_collider(p)]
    print(f"[mir_isaac_sim] self-collision unblock: {len(colliders)} robot geometry "
          f"colliders found; laser plane z=[{z_lo:.3f},{z_hi:.3f}]")

    # Disable collision ONLY on colliders crossing the laser plane (the chassis).
    # Wheels/casters are kept explicitly too (belt-and-suspenders); the arm,
    # gripper and any other geometry above/below the plane keep their collision.
    KEEP = ("wheel", "caster")
    pad = 0.03
    disabled, kept = [], []
    for p in colliders:
        path = p.GetPath().pathString
        rng = bbc.ComputeWorldBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            kept.append(path)
            continue
        lo, hi = rng.GetMin()[2], rng.GetMax()[2]
        crosses = (lo - pad <= z_hi) and (hi + pad >= z_lo)
        if crosses and not any(k in path.lower() for k in KEEP):
            set_collision(p, False)
            disabled.append(path)
        else:
            kept.append(path)
    print(f"[mir_isaac_sim] self-collision unblock: disabled {len(disabled)} "
          f"laser-plane (chassis) collider(s), kept {len(kept)} "
          f"(wheels/casters/arm/gripper)")
    print(f"[mir_isaac_sim]   disabled: {[p.split('/')[-1] for p in disabled]}")
    if not disabled:
        print("[mir_isaac_sim] WARNING: 0 disabled — no collider crosses the laser "
              "plane; paste this log so the blocking prim can be targeted directly.")
    return disabled


def add_lidar_safe_chassis_collider():
    """Restore the MiR footprint collision below the horizontal lidar plane.

    The imported chassis mesh surrounds both SICK origins, so PhysX ray casts
    see the robot itself.  ``unblock_lidar_self_collision`` therefore disables
    that mesh collider.  Leaving it at that makes the casters and drive wheels
    the first parts to hit a wall; if Nav2 keeps pushing at a narrow entrance,
    those small curved contacts can wedge the articulation and PhysX can eject
    it.  A low box supplies the same 0.89 x 0.58 m XY footprint as Nav2/Gazebo,
    while its 0.14 m top remains safely below the lidar plane at z=0.1914 m.
    """
    base = find_prim("base_link")
    if base is None:
        carb.log_warn("[mir_isaac_sim] chassis proxy skipped: base_link not found")
        return None
    path = base.GetPath().AppendChild("lidar_safe_chassis_collision")
    cube = UsdGeom.Cube.Define(stage_handle, path)
    cube.CreateSizeAttr(2.0)
    xf = UsdGeom.XformCommonAPI(cube.GetPrim())
    # Nav2 footprint: x=-0.39..0.50, y=-0.29..0.29.  Keep the bottom 2 cm
    # above the floor so this bumper cannot add ground drag.
    xf.SetTranslate(Gf.Vec3d(0.055, 0.0, 0.08))
    xf.SetScale(Gf.Vec3f(0.445, 0.29, 0.06))
    cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    carb.log_warn("[mir_isaac_sim] chassis collision proxy: "
                  "x=-0.39..0.50 y=-0.29..0.29 z=0.02..0.14 "
                  "(below lidar z=0.1914)")
    return cube.GetPrim()


sensor_nodes, sensor_connect, sensor_set = [], [], []

# ---------------------------------------------------------- 2x SICK S300 lidars
# RTX lidars need a render product each (they're cameras under the hood). The
# Replicator writers attach the ROS2 publishers in the SDG pipeline.
# Mounting orientation matches the URDF macro:
#   the sensor is parented to the laser_link so its Z axis points up after the
#   default Isaac lidar Z-up convention. No extra rotation needed because the
#   link frame already matches the URDF.
LIDARS = []
if not args.no_lasers and not args.rtx_lasers:
    # PhysX (physics ray-cast) lidar. Pure-OmniGraph, no render product or SDG
    # pipeline — works headless. SICK S300: 0.05–29 m, ±120° (240° FOV), 541 rays.
    #
    # OnPlaybackTick runs at the 60 Hz simulation rate.  Publishing every tick
    # used to make /f_scan, /b_scan and the merged /scan run at ~50-60 Hz,
    # unlike the Gazebo SICK plugin's 12.5 Hz.  Besides being an inaccurate
    # sensor model, that needlessly loads AMCL, both costmaps and MPPI at the
    # maze's narrow entrance.  A five-tick simulation gate produces 12 Hz,
    # which is the closest integer divisor of 60 Hz to Gazebo's 12.5 Hz.
    #
    # max_range = 30.0 (NOT 29.0) on purpose: the PhysX RangeSensor returns its
    # max_range for a NO-HIT beam (GenericSensor.h: `linearDepth=maxDepth`), so a
    # no-hit reads as a finite point at max_range — Gazebo instead returns +inf.
    # The scan merger reprojects each beam into virtual_laser_link (the sensors
    # sit ~0.49 m off-centre) and then KEEPS any point with range <= its
    # range_max (29.0) while leaving empty bins at +inf. If the lidar max_range
    # were also 29.0, no-hit beams would reproject to ~28.5–29.5 m: the ones just
    # under 29 get kept as a phantom far ring, the rest become inf — an
    # inconsistent partial ring that does NOT match Gazebo. Setting the lidar
    # max_range to 30.0 (> merger range_max 29.0 + the 0.49 m mount offset)
    # guarantees EVERY no-hit reprojects past 29.0 and is dropped -> the bin
    # stays +inf, exactly like Gazebo. Real hits <=29 m are unaffected; the
    # published /scan still reports range_max 29.0 from the merger.
    #
    # Before creating the lidars, stop them from ray-casting the robot's own
    # chassis (the sensor origins sit inside the base collision box). Without
    # this, ~260° of each FOV hits the body and /scan loses two big sectors.
    _laser_links = [lk for lk in (find_prim("front_laser_link"),
                                  find_prim("back_laser_link")) if lk is not None]
    if _laser_links:
        _disabled_chassis = unblock_lidar_self_collision(_laser_links)
        if _disabled_chassis:
            add_lidar_safe_chassis_collider()
    sensor_nodes += [
        ("PhysXLidarGate", "isaacsim.core.nodes.IsaacSimulationGate"),
    ]
    sensor_connect += [
        ("OnPlaybackTick.outputs:tick", "PhysXLidarGate.inputs:execIn"),
    ]
    sensor_set += [
        ("PhysXLidarGate.inputs:step", 5),
    ]
    for idx, (link, topic) in enumerate((("front_laser_link", "f_scan"),
                                         ("back_laser_link", "b_scan"))):
        parent = find_prim(link)
        if parent is None:
            carb.log_warn(f"[mir_isaac_sim] {link} not in USD, skipping {topic}")
            continue
        _, lidar = omni.kit.commands.execute(
            "RangeSensorCreateLidar",
            path="/sick",
            parent=parent.GetPath().pathString,
            min_range=0.05, max_range=30.0,
            draw_points=False, draw_lines=False,
            horizontal_fov=240.0,
            vertical_fov=1.0,            # must be > 0; 1 row -> 2D scan
            horizontal_resolution=240.0 / max(int(args.laser_rays), 2),
            vertical_resolution=1.0,
            rotation_rate=0.0, high_lod=False, yaw_offset=0.0,
            enable_semantics=False,
        )
        lidar_path = lidar.GetPath().pathString
        rd, pub = f"ReadLidar{idx}", f"LaserPub{idx}"
        sensor_nodes += [
            (rd, "isaacsim.sensors.physx.IsaacReadLidarBeams"),
            (pub, "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
        ]
        sensor_connect += [
            ("PhysXLidarGate.outputs:execOut", f"{rd}.inputs:execIn"),
            (f"{rd}.outputs:execOut", f"{pub}.inputs:execIn"),
            ("ContextSensors.outputs:context", f"{pub}.inputs:context"),
            ("ReadSimTimeSensors.outputs:simulationTime", f"{pub}.inputs:timeStamp"),
            (f"{rd}.outputs:azimuthRange", f"{pub}.inputs:azimuthRange"),
            (f"{rd}.outputs:depthRange", f"{pub}.inputs:depthRange"),
            (f"{rd}.outputs:horizontalFov", f"{pub}.inputs:horizontalFov"),
            (f"{rd}.outputs:horizontalResolution", f"{pub}.inputs:horizontalResolution"),
            (f"{rd}.outputs:intensitiesData", f"{pub}.inputs:intensitiesData"),
            (f"{rd}.outputs:linearDepthData", f"{pub}.inputs:linearDepthData"),
            (f"{rd}.outputs:numCols", f"{pub}.inputs:numCols"),
            (f"{rd}.outputs:numRows", f"{pub}.inputs:numRows"),
            (f"{rd}.outputs:rotationRate", f"{pub}.inputs:rotationRate"),
        ]
        sensor_set += [
            (f"{rd}.inputs:lidarPrim", [usdrt.Sdf.Path(lidar_path)]),
            (f"{pub}.inputs:topicName", topic),
            (f"{pub}.inputs:frameId", link),
        ]
        LIDARS.append((topic, link, lidar_path))
        carb.log_warn(f"[mir_isaac_sim] PhysX lidar -> /{topic} (frame {link})")
    carb.log_warn("[mir_isaac_sim] PhysX lidar publish gate: 60 Hz / 5 = 12 Hz "
                  "(Gazebo SICK update_rate: 12.5 Hz)")

# ----------------------------------------- 2x SICK S300 RTX lidars (--rtx-lasers)
# Uses IsaacSensorCreateRtxLidar with the SICK_S300 JSON profile, then publishes
# via ROS2RtxLidarHelper (type=laser_scan). CRITICAL: the render product must be
# created with the RTX-lidar render vars ["GenericModelOutput","RtxSensorMetadata"]
# — the camera-oriented IsaacCreateRenderProduct OG node does NOT set these, so a
# render product from that node carries no lidar data and the helper publishes
# nothing. We therefore build the render product directly with
# rep.create.render_product(...) and feed its .path to the helper, exactly the
# pattern that passes in isaacsim.ros2.bridge test_rtx_sensor.py on this build.
# Requires a viewport; --lasers (PhysX) is the headless-safe alternative.
_RTX_LIDAR_RPS = []  # keep references alive so the render products aren't GC'd
if not args.no_lasers and args.rtx_lasers:
    # Reference the SICK_S300.usda OmniLidar directly instead of going through
    # IsaacSensorCreateRtxLidar(config=...). That command resolves `config` by
    # NAME against SUPPORTED_LIDAR_CONFIGS (USD assets pulled from the asset
    # server); a file path does NOT match, so it silently falls back to a default
    # 3D rotary lidar (elevationDeg = -15°) and IsaacComputeRTXLidarFlatScan then
    # refuses to run ("Lidar prim is not a 2D Lidar"). SICK_S300.usda is a 2D
    # OmniLidar (elevationDeg all 0) carrying OmniSensorGenericLidarCoreAPI, so a
    # direct reference gives the helper a prim it accepts and FlatScan can read.
    _sick_usda = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "usd", "SICK_S300.usda")
    if not os.path.isfile(_sick_usda):
        carb.log_error(f"[mir_isaac_sim] SICK_S300.usda not found: {_sick_usda}")
    else:
        for idx, (link, topic) in enumerate((("front_laser_link", "f_scan"),
                                             ("back_laser_link", "b_scan"))):
            parent = find_prim(link)
            if parent is None:
                carb.log_warn(f"[mir_isaac_sim] {link} not in USD, skipping RTX {topic}")
                continue
            lidar_path = parent.GetPath().pathString + "/sick_rtx"
            # prim_type MUST be "OmniLidar": add_reference_to_stage defaults to
            # "Xform", and a local "Xform" type opinion overrides the referenced
            # OmniLidar type, so the helper's `GetTypeName() == "OmniLidar"` check
            # fails ("Render product not attached to RTX Lidar").
            stage.add_reference_to_stage(usd_path=_sick_usda, prim_path=lidar_path,
                                         prim_type="OmniLidar")
            lidar_prim = stage_handle.GetPrimAtPath(lidar_path)
            if not lidar_prim or not lidar_prim.IsValid():
                carb.log_error(f"[mir_isaac_sim] failed to reference SICK_S300.usda "
                               f"at {lidar_path}, skipping RTX {topic}")
                continue
            # Belt-and-suspenders: ensure the lidar-core API is applied so the
            # helper's HasAPI("OmniSensorGenericLidarCoreAPI") check also passes
            # even if the referenced apiSchemas don't compose as expected.
            if not lidar_prim.HasAPI("OmniSensorGenericLidarCoreAPI"):
                lidar_prim.AddAppliedSchema("OmniSensorGenericLidarCoreAPI")
            # RTX lidar render product WITH the sensor render vars (see note above)
            rp = rep.create.render_product(
                lidar_path, resolution=(128, 128),
                render_vars=["GenericModelOutput", "RtxSensorMetadata"],
                force_new=True,
            )
            _RTX_LIDAR_RPS.append(rp)
            helper_node = f"LidarHelper{idx}"
            sensor_nodes += [
                (helper_node, "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ]
            sensor_connect += [
                ("OnPlaybackTick.outputs:tick", f"{helper_node}.inputs:execIn"),
                ("ContextSensors.outputs:context", f"{helper_node}.inputs:context"),
            ]
            sensor_set += [
                (f"{helper_node}.inputs:renderProductPath", rp.path),
                (f"{helper_node}.inputs:topicName", topic),
                (f"{helper_node}.inputs:frameId", link),
                (f"{helper_node}.inputs:type", "laser_scan"),
            ]
            LIDARS.append((topic, link, lidar_path))
            carb.log_warn(f"[mir_isaac_sim] RTX lidar (SICK_S300) -> /{topic} "
                          f"(frame {link}, rp {rp.path})")

# NOTE: A single 360° lidar at virtual_laser_link publishing /scan directly is
# DELIBERATELY NOT provided. The real MiR100 has TWO SICK S300 scanners
# (front-left + back-right corners); the simulation must keep both and feed the
# ira_laser_tools merger. Replacing them with one centre lidar is forbidden — see
# README_isaac.md ("Two SICK S300 lidars — do NOT replace with one").

# --------------------------------------------------------------- PhysX IMU
if not args.no_imu:
    imu_parent = find_prim("imu_link") or find_prim("imu_frame")
    if imu_parent is None:
        carb.log_warn("[mir_isaac_sim] imu_link / imu_frame not in USD, skipping IMU")
    else:
        _, imu_prim = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/imu",
            parent=imu_parent.GetPath().pathString,
            sensor_period=1.0 / 50.0,
            translation=Gf.Vec3d(0, 0, 0),
            orientation=Gf.Quatd(1, 0, 0, 0),
        )
        imu_path = imu_prim.GetPath().pathString
        sensor_nodes += [
            ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
            ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
        ]
        sensor_connect += [
            ("OnPlaybackTick.outputs:tick", "ReadIMU.inputs:execIn"),
            ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
            ("ContextSensors.outputs:context", "PublishIMU.inputs:context"),
            ("ReadSimTimeSensors.outputs:simulationTime", "PublishIMU.inputs:timeStamp"),
            ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
            ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
            ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
        ]
        sensor_set += [
            ("ReadIMU.inputs:imuPrim", [usdrt.Sdf.Path(imu_path)]),
            ("PublishIMU.inputs:topicName", "imu_data"),
            ("PublishIMU.inputs:frameId", "imu_frame"),
        ]
        carb.log_warn(f"[mir_isaac_sim] PhysX IMU -> /imu_data (frame imu_frame)")

# --------------------------------------------------------------- D435i RGB-D
_CAMERA_RPS = []  # keep references alive so the render product isn't GC'd
# Two image streams + one camera_info, all sharing the same render product on a
# camera prim placed at /World/Robot/realsense_link. Frames match the URDF.
if not args.no_camera:
    rs_parent = find_prim("realsense_link")
    if rs_parent is None:
        carb.log_warn("[mir_isaac_sim] realsense_link not in USD, skipping camera")
    else:
        cam_path = rs_parent.GetPath().pathString + "/d435i_camera"
        cam_prim = UsdGeom.Camera(stage_handle.DefinePrim(cam_path, "Camera"))
        # D435i optical convention: +Z forward, +X right, +Y down. URDF parents
        # the color/depth optical frames with a -90deg rotation about each axis;
        # here we sit on realsense_link and apply the same color-optical-frame
        # transform so the published image matches the TF tree.
        xform = UsdGeom.XformCommonAPI(cam_prim)
        xform.SetRotate((-90, 0, -90), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        # Intel D435 intrinsics (approx.): 69deg HFOV @ 1920x1080 ~> focal ~1.39mm
        # on a 36mm sensor model. Replicator/Isaac uses a 24mm camera by default;
        # pick aperture so horizontalAperture / focalLength matches tan(HFOV/2)*2.
        cam_prim.GetFocalLengthAttr().Set(1.93)
        cam_prim.GetHorizontalApertureAttr().Set(2.682)  # 69deg HFOV
        cam_prim.GetVerticalApertureAttr().Set(1.509)   # 42.5deg VFOV
        cam_prim.GetClippingRangeAttr().Set((0.1, 100.0))

        # Same shortcut pattern as the lidars: fire IsaacCreateRenderProduct
        # off RunOnce.outputs:step, then the camera helpers consume the produced
        # renderProductPath. CameraInfoHelper attaches to the same render product.
        # Same lesson as the RTX lidar above: build the render product in Python
        # rather than with the IsaacCreateRenderProduct OG node. Fired off
        # RunOnce, that node produces a render product the SDG pipeline never
        # renders, so the camera helpers publish nothing at all — and log no
        # error, so it looks like the camera simply does not exist.
        _cam_rp = rep.create.render_product(cam_path, resolution=(640, 480),
                                            force_new=True)
        _CAMERA_RPS.append(_cam_rp)   # keep alive so it isn't GC'd
        sensor_nodes += [
            ("CamHelperRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CamHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CamHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ]
        sensor_connect += [
            ("OnPlaybackTick.outputs:tick", "CamHelperRGB.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "CamHelperInfo.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "CamHelperDepth.inputs:execIn"),
            ("ContextSensors.outputs:context", "CamHelperRGB.inputs:context"),
            ("ContextSensors.outputs:context", "CamHelperInfo.inputs:context"),
            ("ContextSensors.outputs:context", "CamHelperDepth.inputs:context"),
        ]
        sensor_set += [
            ("CamHelperRGB.inputs:renderProductPath", _cam_rp.path),
            ("CamHelperInfo.inputs:renderProductPath", _cam_rp.path),
            ("CamHelperDepth.inputs:renderProductPath", _cam_rp.path),
            ("CamHelperRGB.inputs:topicName", "realsense/color/image_raw"),
            ("CamHelperRGB.inputs:frameId", "realsense_color_optical_frame"),
            ("CamHelperRGB.inputs:type", "rgb"),
            ("CamHelperInfo.inputs:topicName", "realsense/color/camera_info"),
            ("CamHelperInfo.inputs:frameId", "realsense_color_optical_frame"),
            ("CamHelperDepth.inputs:topicName", "realsense/depth/image_rect_raw"),
            ("CamHelperDepth.inputs:frameId", "realsense_depth_optical_frame"),
            ("CamHelperDepth.inputs:type", "depth"),
        ]
        carb.log_warn("[mir_isaac_sim] D435i camera -> /realsense/{color,depth,camera_info}")

# ------------------------------------------------------- chassis nav camera
# Mounted on base_link, not on the arm. Two prims: an Xform carrying the mount
# pose and the downward pitch (base_link is ROS-style X forward / Y left / Z up,
# so a positive rotation about Y pitches the view down), and the Camera itself
# carrying the optical-frame convention (+Z forward, +X right, +Y down), the
# same (-90, 0, -90) used for the wrist camera above.
if args.body_camera:
    base = find_prim("base_link")
    if base is None:
        carb.log_warn("[mir_isaac_sim] base_link not in USD, skipping body camera")
    else:
        mount_path = base.GetPath().pathString + "/body_camera_mount"
        mount = UsdGeom.Xform(stage_handle.DefinePrim(mount_path, "Xform"))
        mx = UsdGeom.XformCommonAPI(mount)
        mx.SetTranslate(Gf.Vec3d(*args.body_camera_xyz))
        mx.SetRotate((0.0, float(args.body_camera_tilt), 0.0),
                     UsdGeom.XformCommonAPI.RotationOrderXYZ)

        bcam_path = mount_path + "/body_camera"
        bcam = UsdGeom.Camera(stage_handle.DefinePrim(bcam_path, "Camera"))
        # (90, 0, -90), NOT the (-90, 0, -90) the wrist camera above uses.
        # A USD camera looks down its own -Z with +Y as image-up, and USD applies
        # RotationOrderXYZ as Rz*Ry*Rx, so solving for
        #     X_cam = -Y_parent (image right = robot right)
        #     Y_cam = +Z_parent (image up    = world up)
        #     Z_cam = -X_parent (camera looks forward)
        # gives rx=+90. With rx=-90 the Y axis lands on world-DOWN and every
        # frame comes out upside down -- measured: the dark strip of shadowed
        # ground right under the bumper rendered at the TOP of the image, and
        # rotating the robot did not change the picture at all.
        # That matters beyond tidiness: SmolVLA's vision tower is SmolVLM2
        # pretrained on upright natural images, so feeding it inverted frames
        # throws away most of what that pretraining is worth.
        UsdGeom.XformCommonAPI(bcam).SetRotate(
            tuple(args.body_camera_rot), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        bcam.GetFocalLengthAttr().Set(1.93)
        bcam.GetHorizontalApertureAttr().Set(2.682)   # 69 deg HFOV, as the D435
        bcam.GetVerticalApertureAttr().Set(1.509)     # 42.5 deg VFOV
        bcam.GetClippingRangeAttr().Set((0.05, 100.0))

        _body_rp = rep.create.render_product(
            bcam_path, resolution=tuple(args.body_camera_res), force_new=True)
        _CAMERA_RPS.append(_body_rp)
        sensor_nodes += [
            ("BodyCamRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("BodyCamInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        sensor_connect += [
            ("OnPlaybackTick.outputs:tick", "BodyCamRGB.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "BodyCamInfo.inputs:execIn"),
            ("ContextSensors.outputs:context", "BodyCamRGB.inputs:context"),
            ("ContextSensors.outputs:context", "BodyCamInfo.inputs:context"),
        ]
        sensor_set += [
            ("BodyCamRGB.inputs:renderProductPath", _body_rp.path),
            ("BodyCamInfo.inputs:renderProductPath", _body_rp.path),
            ("BodyCamRGB.inputs:topicName", "body_camera/color/image_raw"),
            ("BodyCamRGB.inputs:frameId", "body_camera_optical_frame"),
            ("BodyCamRGB.inputs:type", "rgb"),
            ("BodyCamInfo.inputs:topicName", "body_camera/color/camera_info"),
            ("BodyCamInfo.inputs:frameId", "body_camera_optical_frame"),
        ]
        carb.log_warn(
            "[mir_isaac_sim] body camera at base_link %s, %.0f deg down "
            "-> /body_camera/color/image_raw"
            % (tuple(args.body_camera_xyz), args.body_camera_tilt))

if args.platform_camera:
    base = find_prim("base_link")
    if base is None:
        carb.log_warn("[mir_isaac_sim] base_link not in USD, skipping platform camera")
    else:
        mount_path = base.GetPath().pathString + "/platform_camera_mount"
        mount = UsdGeom.Xform(stage_handle.DefinePrim(mount_path, "Xform"))
        UsdGeom.XformCommonAPI(mount).SetTranslate(Gf.Vec3d(*args.platform_camera_xyz))
        UsdGeom.XformCommonAPI(mount).SetRotate(
            (0.0, float(args.platform_camera_tilt), 0.0),
            UsdGeom.XformCommonAPI.RotationOrderXYZ)

        pcam_path = mount_path + "/platform_camera"
        pcam = UsdGeom.Camera(stage_handle.DefinePrim(pcam_path, "Camera"))
        # Same (90, 0, -90) as the body camera: a USD camera looks down its own
        # -Z with +Y as image-up, and rx=+90 is what puts image-up on world-up.
        # rx=-90 renders every frame upside down and nothing else complains.
        UsdGeom.XformCommonAPI(pcam).SetRotate(
            tuple(args.platform_camera_rot), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        pcam.GetFocalLengthAttr().Set(1.93)
        pcam.GetHorizontalApertureAttr().Set(args.platform_camera_aperture[0])
        pcam.GetVerticalApertureAttr().Set(args.platform_camera_aperture[1])
        pcam.GetClippingRangeAttr().Set((0.05, 100.0))

        _plat_rp = rep.create.render_product(
            pcam_path, resolution=tuple(args.platform_camera_res), force_new=True)
        _CAMERA_RPS.append(_plat_rp)
        sensor_nodes += [
            ("PlatCamRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("PlatCamInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        sensor_connect += [
            ("OnPlaybackTick.outputs:tick", "PlatCamRGB.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "PlatCamInfo.inputs:execIn"),
            ("ContextSensors.outputs:context", "PlatCamRGB.inputs:context"),
            ("ContextSensors.outputs:context", "PlatCamInfo.inputs:context"),
        ]
        sensor_set += [
            ("PlatCamRGB.inputs:renderProductPath", _plat_rp.path),
            ("PlatCamInfo.inputs:renderProductPath", _plat_rp.path),
            ("PlatCamRGB.inputs:topicName", "platform_camera/color/image_raw"),
            ("PlatCamRGB.inputs:frameId", "platform_camera_optical_frame"),
            ("PlatCamRGB.inputs:type", "rgb"),
            ("PlatCamInfo.inputs:topicName", "platform_camera/color/camera_info"),
            ("PlatCamInfo.inputs:frameId", "platform_camera_optical_frame"),
        ]
        hf = 2 * math.degrees(math.atan(args.platform_camera_aperture[0] / 2 / 1.93))
        vf = 2 * math.degrees(math.atan(args.platform_camera_aperture[1] / 2 / 1.93))
        carb.log_warn(
            "[mir_isaac_sim] platform camera at base_link %s, %.0f deg down, "
            "%.0fx%.0f deg FOV -> /platform_camera/color/image_raw"
            % (tuple(args.platform_camera_xyz), args.platform_camera_tilt, hf, vf))

if args.global_camera:
    # World-fixed, not parented to the robot: the point is a view that always
    # contains everyone, which a robot-mounted camera cannot guarantee.
    gcam_path = "/World/global_camera"
    gcam = UsdGeom.Camera(stage_handle.DefinePrim(gcam_path, "Camera"))
    # Looking straight down. A USD camera looks along its own -Z, so an
    # unrotated camera already points down in a Z-up stage; the yaw only decides
    # which world axis runs across the image.
    #
    # It must be 0, not -90. The coverage is 18.5 x 13.9 m at 12 m; turning the
    # camera 90 deg puts the room's 16 m axis against the 13.9 m side and crops
    # about a metre off each end -- which shows up as a view with walls on two
    # sides only, and quietly hides part of the area the people walk in.
    UsdGeom.XformCommonAPI(gcam).SetTranslate(
        Gf.Vec3d(0.0, 0.0, float(args.global_camera_height)))
    UsdGeom.XformCommonAPI(gcam).SetRotate(
        (0.0, 0.0, float(args.global_camera_yaw)),
        UsdGeom.XformCommonAPI.RotationOrderXYZ)
    gcam.GetFocalLengthAttr().Set(1.93)
    gcam.GetHorizontalApertureAttr().Set(args.global_camera_aperture[0])
    gcam.GetVerticalApertureAttr().Set(args.global_camera_aperture[1])
    gcam.GetClippingRangeAttr().Set((0.05, 200.0))

    _glob_rp = rep.create.render_product(
        gcam_path, resolution=tuple(args.global_camera_res), force_new=True)
    _CAMERA_RPS.append(_glob_rp)
    sensor_nodes += [
        ("GlobCamRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("GlobCamInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
    ]
    sensor_connect += [
        ("OnPlaybackTick.outputs:tick", "GlobCamRGB.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "GlobCamInfo.inputs:execIn"),
        ("ContextSensors.outputs:context", "GlobCamRGB.inputs:context"),
        ("ContextSensors.outputs:context", "GlobCamInfo.inputs:context"),
    ]
    sensor_set += [
        ("GlobCamRGB.inputs:renderProductPath", _glob_rp.path),
        ("GlobCamInfo.inputs:renderProductPath", _glob_rp.path),
        ("GlobCamRGB.inputs:topicName", "global_camera/color/image_raw"),
        ("GlobCamRGB.inputs:frameId", "global_camera_optical_frame"),
        ("GlobCamRGB.inputs:type", "rgb"),
        ("GlobCamInfo.inputs:topicName", "global_camera/color/camera_info"),
        ("GlobCamInfo.inputs:frameId", "global_camera_optical_frame"),
    ]
    _ghf = 2 * math.degrees(math.atan(args.global_camera_aperture[0] / 2 / 1.93))
    _gvf = 2 * math.degrees(math.atan(args.global_camera_aperture[1] / 2 / 1.93))
    _gw = 2 * args.global_camera_height * math.tan(math.radians(_ghf / 2))
    _gd = 2 * args.global_camera_height * math.tan(math.radians(_gvf / 2))
    print("[mir_isaac_sim] global camera at z=%.1f m, %.0fx%.0f deg FOV -> "
          "covers %.1f x %.1f m -> /global_camera/color/image_raw"
          % (args.global_camera_height, _ghf, _gvf, _gw, _gd), flush=True)

if args.corner_cameras:
    _xmin, _xmax, _ymin, _ymax = args.corner_camera_room
    _ins, _h = args.corner_camera_inset, args.corner_camera_height
    _corners = [("a", _xmin + _ins, _ymin + _ins),
                ("b", _xmax - _ins, _ymax - _ins)]
    for _tag, _cx, _cy in _corners:
        # Aim at the middle of the room at torso height. Mount carries the
        # aiming, camera carries the optical convention -- same split as the
        # platform camera, because a USD camera looks down its own -Z and the
        # (90, 0, -90) is what turns that into "looks along the mount's +X with
        # the image the right way up".
        _dx, _dy = 0.0 - _cx, 0.0 - _cy
        _flat = math.hypot(_dx, _dy)
        _yaw = math.degrees(math.atan2(_dy, _dx))
        _pitch = math.degrees(math.atan2(_h - args.corner_camera_aim_z, _flat))
        _mount_path = f"/World/corner_camera_{_tag}_mount"
        _mount = UsdGeom.Xform(stage_handle.DefinePrim(_mount_path, "Xform"))
        UsdGeom.XformCommonAPI(_mount).SetTranslate(Gf.Vec3d(_cx, _cy, _h))
        # XYZ order: pitch about Y first, then yaw about Z, so the nose is
        # tipped down and then swung to face the centre.
        UsdGeom.XformCommonAPI(_mount).SetRotate(
            (0.0, float(_pitch), float(_yaw)),
            UsdGeom.XformCommonAPI.RotationOrderXYZ)

        _cpath = _mount_path + f"/corner_camera_{_tag}"
        _ccam = UsdGeom.Camera(stage_handle.DefinePrim(_cpath, "Camera"))
        UsdGeom.XformCommonAPI(_ccam).SetRotate(
            (90.0, 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        _ccam.GetFocalLengthAttr().Set(1.93)
        _ccam.GetHorizontalApertureAttr().Set(args.corner_camera_aperture[0])
        _ccam.GetVerticalApertureAttr().Set(args.corner_camera_aperture[1])
        _ccam.GetClippingRangeAttr().Set((0.05, 200.0))

        _rp = rep.create.render_product(
            _cpath, resolution=tuple(args.corner_camera_res), force_new=True)
        _CAMERA_RPS.append(_rp)
        _N = f"CornCam{_tag.upper()}"
        sensor_nodes += [
            (f"{_N}RGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (f"{_N}Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        sensor_connect += [
            ("OnPlaybackTick.outputs:tick", f"{_N}RGB.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", f"{_N}Info.inputs:execIn"),
            ("ContextSensors.outputs:context", f"{_N}RGB.inputs:context"),
            ("ContextSensors.outputs:context", f"{_N}Info.inputs:context"),
        ]
        sensor_set += [
            (f"{_N}RGB.inputs:renderProductPath", _rp.path),
            (f"{_N}Info.inputs:renderProductPath", _rp.path),
            (f"{_N}RGB.inputs:topicName", f"corner_camera_{_tag}/color/image_raw"),
            (f"{_N}RGB.inputs:frameId", f"corner_camera_{_tag}_optical_frame"),
            (f"{_N}RGB.inputs:type", "rgb"),
            (f"{_N}Info.inputs:topicName", f"corner_camera_{_tag}/color/camera_info"),
            (f"{_N}Info.inputs:frameId", f"corner_camera_{_tag}_optical_frame"),
        ]
        print("[mir_isaac_sim] corner camera %s at (%.1f, %.1f, %.1f), "
              "yaw %.0f deg, %.0f deg down -> /corner_camera_%s/color/image_raw"
              % (_tag, _cx, _cy, _h, _yaw, _pitch, _tag), flush=True)

# Build the sensor graph if anything was registered. The IMU & camera ride on a
# push-tick action graph so they fire every frame; the RTX lidars are driven by
# the SDG pipeline through replicator and need no node here.
if sensor_nodes:
    sensor_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("ContextSensors", "isaacsim.ros2.bridge.ROS2Context"),
        ("ReadSimTimeSensors", "isaacsim.core.nodes.IsaacReadSimulationTime"),
    ] + sensor_nodes
    sensor_connect = [
        ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
    ] + sensor_connect
    sensor_graph = None
    # Build render-product nodes with enabled=False so they don't try to attach
    # their replicator writers during evaluate_sync (the SDG annotators are not
    # registered yet at that point — they only come up after play() renders a
    # frame). After play() + warmup frames we'll enable them. Pattern from
    # carter_stereo.py standalone example.
    sensor_set = sensor_set + [
        (name + ".inputs:enabled", False)
        for name, ntype in sensor_nodes
        if ntype == "isaacsim.core.nodes.IsaacCreateRenderProduct"
    ]
    try:
        (sensor_graph, _, _, _) = og.Controller.edit(
            {"graph_path": "/ROS_Sensors", "evaluator_name": "execution"},
            {
                graph_keys.CREATE_NODES: sensor_nodes,
                graph_keys.CONNECT: sensor_connect,
                graph_keys.SET_VALUES: sensor_set,
            },
        )
        carb.log_warn(f"[mir_isaac_sim] sensor graph built ({len(sensor_nodes)} nodes)")
    except Exception as e:  # noqa: BLE001
        carb.log_error(f"[mir_isaac_sim] failed to build sensor graph: {e}")

simulation_app.update()

# ----------------------------------------------------------------- run loop
simulation_context.initialize_physics()
simulation_context.play()


# Start the arm AT its home pose instead of letting it droop. The importer's
# joint state has an offset vs the URDF, so a drooped arm reads ~3 rad away from
# ros2_control's initial_value; when the controllers activate they then yank the
# arm across that gap fast enough to flip the whole MiR base. We TELEPORT the
# articulation to home (set_joint_positions = instantaneous, no momentum, so the
# base can't be kicked) and set the drive targets to hold it. ros2_control then
# reads a steady home pose and holds it -> no lurch. (rclpy isn't usable here:
# Isaac runs Python 3.11, Humble's rclpy C-ext is 3.10.)
if not args.no_arm_home:
    try:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        _art = SingleArticulation(prim_path=robot_prim_path)
        _art.initialize()
        _dofs = list(_art.dof_names)
        _pose = UR_FOLDED if args.fold_arm else UR_HOME
        _home = np.array([_pose.get(n, 0.0) for n in _dofs])
        _pos0, _quat0 = _art.get_world_pose()   # straight spawn orientation
        _art.set_joint_positions(_home)
        _art.apply_action(ArticulationAction(joint_positions=_home))
        for _ in range(30):
            simulation_context.step(render=not args.headless)
        # During this settling the free-caster base can pick up a small yaw and
        # the strong wheel velocity-drive (damping) can snap it -> the robot
        # "jumps crooked". Snap the base orientation back to the straight spawn
        # orientation (keep the settled position), zero velocities, re-settle.
        try:
            _posN, _ = _art.get_world_pose()
            _art.set_world_pose(_posN, _quat0)
            _art.set_joint_velocities(np.zeros(len(_dofs)))
            _art.apply_action(ArticulationAction(joint_positions=_home))
            for _ in range(15):
                simulation_context.step(render=not args.headless)
            carb.log_warn("[mir_isaac_sim] base orientation reset to straight")
        except Exception as _e2:  # noqa: BLE001
            carb.log_warn(f"[mir_isaac_sim] base orient reset skipped: {_e2}")
        carb.log_warn("[mir_isaac_sim] arm teleported to home pose")
    except Exception as e:  # noqa: BLE001
        carb.log_warn(f"[mir_isaac_sim] arm home init skipped: {e}")

# Now that the teleport's transient penetrations have been resolved at full
# speed, clamp depenetration for the rest of the run. See the function's
# definition for why this protects against Nav2 wedging the base into a corner.
set_max_depenetration_velocity(args.max_depenetration_velocity)

# DIAGNOSTIC: drive the wheels directly through the articulation API (the same
# path the arm-home teleport uses, which is known to work) to isolate whether a
# stationary base is a drive problem or an OmniGraph ArticulationController
# problem. Spins both drive wheels at a fixed velocity for ~90 steps and reports
# how far each rotated. If they move here but not via /isaac_base_commands, the
# OmniGraph velocity path is at fault; if they don't move here either, the wheel
# DriveAPI/velocity-drive setup is.
if args.wheel_test:
    try:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        _a = SingleArticulation(prim_path=robot_prim_path)
        _a.initialize()
        _dn = list(_a.dof_names)
        _li = _dn.index("left_wheel_joint")
        _ri = _dn.index("right_wheel_joint")
        _p0 = _a.get_joint_positions()
        print(f"[wheel_test] dof idx left={_li} right={_ri}; "
              f"start pos L={_p0[_li]:.3f} R={_p0[_ri]:.3f}", flush=True)
        _act = ArticulationAction(joint_velocities=np.array([5.0, 5.0]),
                                  joint_indices=np.array([_li, _ri]))
        for _ in range(90):
            _a.apply_action(_act)
            simulation_context.step(render=not args.headless)
        _p1 = _a.get_joint_positions()
        print(f"[wheel_test] after 90 steps @5rad/s: "
              f"L {_p0[_li]:.3f}->{_p1[_li]:.3f} (d={_p1[_li]-_p0[_li]:.3f})  "
              f"R {_p0[_ri]:.3f}->{_p1[_ri]:.3f} (d={_p1[_ri]-_p0[_ri]:.3f})", flush=True)
        print("[wheel_test] if d~0 -> velocity DRIVE is broken; "
              "if d large -> OmniGraph ArticulationController is the issue", flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[wheel_test] failed: {e}\n{traceback.format_exc()}", flush=True)

# Warm up the SDG pipeline for several rendered frames FIRST. Once a real render
# frame has gone through, omni.syntheticdata registers the gate annotators
# (PostProcessDispatchIsaacSimulationGate, LdrColorSDIsaacConvertRGBAToRGB,
# DistanceToImagePlaneSDIsaacPassthroughImagePtr, IsaacComputeRTXLidarFlatScan)
# that the ROS2 writers depend on. Attaching before this fails permanently.
# (Pattern: carter_stereo.py / rtx_lidar.py.) Needs a viewport, i.e. NOT headless.
if sensor_nodes or LIDARS:
    for _ in range(20):
        simulation_context.step(render=True)

    # camera: enable the IsaacCreateRenderProduct OG nodes
    for name, ntype in sensor_nodes:
        if ntype == "isaacsim.core.nodes.IsaacCreateRenderProduct":
            try:
                og.Controller.set(
                    og.Controller.attribute(f"/ROS_Sensors/{name}.inputs:enabled"), True)
                carb.log_warn(f"[mir_isaac_sim] enabled {name}")
            except Exception as e:  # noqa: BLE001
                carb.log_warn(f"[mir_isaac_sim] enable {name}: {e}")

    # (PhysX lidars need no writer attach — they publish via OG nodes.)
    for _ in range(5):
        simulation_context.step(render=not args.headless)

_extra = []
if LIDARS:
    _extra += [f"/{t}" for t, *_ in LIDARS]
if not args.no_imu:
    _extra.append("/imu_data")
if not args.no_camera:
    _extra.append("/realsense/{color,depth}")
print("[mir_isaac_sim] running. Publishing /{0}, /clock{2}; subscribing {1}".format(
    args.states_topic, ", ".join("/" + t for t in args.command_topics),
    (" + " + " ".join(_extra)) if _extra else ""), flush=True)
carb.log_warn("[mir_isaac_sim] entering run loop")

_base_idx = next((i for i, t in enumerate(args.command_topics) if "base" in t), None)
_dbg = 0

class _PersonWalk:
    """Move /World/Person between random waypoints and report its pose.

    Deliberately kinematic: the prim is teleported each frame rather than
    simulated. The character has a skeleton but no walk cycle (omni.anim.people
    is not installed and the animation assets 404), so it slides instead of
    stepping. What the task needs from it is a person-shaped thing that moves
    plausibly -- speed capped near what a MiR100 can chase, heading rate capped
    so it does not pivot instantly into a turn no wheeled base could imitate.
    """

    def __init__(self, stage, prim_path, a, index=0, name=''):
        import random
        self.index = index
        # Only for log lines. A reversal that says which person it
        # was is the difference between a readable trace and a
        # stream of anonymous events.
        self.name = name or prim_path.rsplit('/', 1)[-1]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            # Fall back to a search rather than failing on a mount-point
            # mismatch: where a --world lands is a property of this script, not
            # of the scene that authored the prim.
            # Search by the NAME asked for, not the literal string "Person":
            # with two people the names are PersonPurple and PersonYellow, and a
            # hard-coded "Person" matches neither. --world remounts the scene
            # under /World/Env, so this mismatch happens on every run.
            want = prim_path.rstrip("/").rsplit("/", 1)[-1]
            found = [p for p in stage.Traverse() if p.GetName() == want]
            if not found:
                raise RuntimeError(
                    f"--walk-person: no prim at {prim_path}, and nothing named "
                    f"{want!r} anywhere on the stage")
            if len(found) > 1:
                raise RuntimeError(
                    f"--walk-person: {len(found)} prims named {want!r} "
                    f"({[q.GetPath().pathString for q in found]}); "
                    f"give an exact path")
            prim = found[0]
            # Keep prim_path in step: the gait is built from it further down,
            # and a stale path would silently give the walk no skeleton.
            prim_path = prim.GetPath().pathString
            print(f"[mir_isaac_sim] person prim not at the given path; using "
                  f"{prim_path}", flush=True)
        self._x = UsdGeom.XformCommonAPI(prim)
        self.a = a
        self.rng = random.Random(a.person_seed)
        # Start from wherever the scene put the person, rather than a second
        # copy of that number on the command line -- two sources for one
        # position is a disagreement waiting to happen.
        t, r, _s, _p, _rot = self._x.GetXformVectors(Usd.TimeCode.Default())
        self.pos = [float(t[0]), float(t[1])]
        self.yaw = math.radians(float(r[2]))

        # The robot, so the walk can steer round it.
        #
        # This MUST come from the physics view, not from the USD stage. The base
        # is driven by PhysX, whose transforms land in Fabric; the USD prim keeps
        # the pose it was spawned with. Reading the stage here returned a
        # constant (0.00, 0.00) for the whole run, so the person was dodging a
        # phantom robot parked at the origin and walked straight through the real
        # one -- which is what every "person hit the robot" report came down to.
        # get_world_pose() reads the articulation's live root pose.
        self._art = None
        self._art_tries = 0
        # Other walkers, set after they all exist. Two people who ignore each
        # other end up standing in the same place, and once they overlap the
        # colour that identifies each of them is no longer separable in the
        # image -- which is the whole signal this scene exists to provide.
        self.others = []
        self._robot = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        self._xc = UsdGeom.XformCache()

        # Optional walk cycle. The character is rigged but ships no animation
        # clip, so the legs are driven joint by joint -- see person_gait.py.
        self.gait = None
        if a.person_gait:
            try:
                sys.path.insert(0, "/home/itri/mir_isaac_test/isaac_sim")
                from person_gait import PersonGait
                self.gait = PersonGait(stage, prim_path, stride=a.person_stride)
                print("[mir_isaac_sim] person gait on (stride %.2f m)"
                      % a.person_stride, flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[mir_isaac_sim] person gait DISABLED: {e}", flush=True)
                traceback.print_exc()
        # Where the scene put this person. Kept so an episode can be replayed
        # from the same opening configuration -- see reset().
        self.home = (self.pos[0], self.pos[1], self.yaw)
        for _spec in getattr(a, "person_home", []) or []:
            _who, _, _xyz = _spec.partition(":")
            if _who != self.name:
                continue
            _f = [float(v) for v in _xyz.split(",")]
            if len(_f) < 2:
                raise SystemExit(f"--person-home {_spec!r}: need at least X,Y")
            self.home = (_f[0], _f[1],
                         math.radians(_f[2]) if len(_f) > 2 else self.yaw)
            # Start there too, not only after the first reset -- otherwise the
            # opening episode is the one run that differs from all the others.
            self.pos = [_f[0], _f[1]]
            self.yaw = self.home[2]
            print(f"[mir_isaac_sim] {self.name} home set to "
                  f"({_f[0]:+.2f}, {_f[1]:+.2f})", flush=True)
        self.goal = self._pick()
        self.pause_left = 0.0
        self.repick_cd = 0
        self.speed = self.rng.uniform(*a.person_speed)

    def reset(self, seed, home=None):
        """Teleport back to the spawn pose and start a fresh random walk.

        Episodes otherwise inherit wherever the previous one left the people,
        so every run opens in a different configuration and two runs of the
        same policy are not comparable. With this, the opening geometry is
        identical across episodes and only the walk that follows differs --
        and it differs reproducibly, because the seed is an input.
        """
        import random
        h = home if home is not None else self.home
        self.pos = [h[0], h[1]]
        self.yaw = h[2]
        # Offset by index so two people given the same episode seed do not walk
        # the identical path.
        self.rng = random.Random(seed + 1009 * self.index)
        self.goal = self._pick()
        self.pause_left = 0.0
        self.repick_cd = 0
        self.speed = self.rng.uniform(*self.a.person_speed)
        if self.gait is not None and hasattr(self.gait, "reset"):
            self.gait.reset()

    def _robot_yaw(self):
        """Robot heading in world, or None. Physics view, never the USD stage."""
        art = self._articulation()
        if art is None:
            return None
        try:
            _p, q = art.get_world_pose()
        except Exception:  # noqa: BLE001
            return None
        # isaacsim returns (w, x, y, z)
        w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _orbit_init(self):
        i = self.index
        ph = self.a.person_orbit_phases
        if i < len(ph):
            self._theta = math.radians(ph[i])
        else:
            # Evenly spread, so two people start on opposite sides of the robot
            # rather than on top of each other.
            self._theta = 2 * math.pi * i / max(len(self.a.people), 1)
        sp = self.a.person_orbit_speeds
        self._speed = abs(sp[i]) if i < len(sp) else (abs(sp[-1]) if sp else 0.22)
        # Opposite directions to begin with: on one circle that guarantees they
        # meet, which is what the bounce exists to handle, instead of trailing
        # each other at a fixed gap forever.
        self._dir = 1.0 if i % 2 == 0 else -1.0
        self._cool = 0.0        # seconds before another reversal may fire
        self._stare = 0.0       # seconds the robot has been facing this person

    def _reverse(self, why):
        lo, hi = self.a.person_orbit_speed_range
        self._dir = -self._dir
        self._speed = self.rng.uniform(lo, hi)
        self._cool = 1.5
        self._stare = 0.0
        print("[mir_isaac_sim] %s reversed (%s), new speed %.2f rad/s"
              % (self.name, why, self._speed), flush=True)

    def _orbit(self, dt):
        """Walk a circle around the orbit centre, reacting to two things.

        No goal picking and no obstacle steering: the path is a circle and the
        robot sits at the centre of it, so there is nothing to route around.
        Heading is tangential, so the person faces the way they are walking.
        """
        if not hasattr(self, "_theta"):
            self._orbit_init()
        if self._cool > 0.0:
            self._cool -= dt

        # 1. Bounce off the other person. Both are on the same circle, so
        #    without this the faster one walks through the slower one, and in
        #    those frames neither colour is readable.
        if self._cool <= 0.0 and self.a.person_orbit_bounce > 0.0:
            for o in self.others:
                ot = getattr(o, "_theta", None)
                if ot is None:
                    continue
                gap = abs((self._theta - ot + math.pi) % (2 * math.pi) - math.pi)
                if gap < math.radians(self.a.person_orbit_bounce):
                    self._reverse("met %s" % o.name)
                    break

        # 2. Reverse once the robot has held this person in view. Symmetric
        #    across people on purpose -- see --person-orbit-flip-secs.
        if self._cool <= 0.0 and self.a.person_orbit_flip_secs > 0.0:
            ry = self._robot_yaw()
            rp = self._robot_xy()
            if ry is not None and rp is not None:
                bearing = math.atan2(self.pos[1] - rp[1], self.pos[0] - rp[0]) - ry
                bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
                if abs(bearing) < math.radians(self.a.person_orbit_flip_deg):
                    self._stare += dt
                    if self._stare >= self.a.person_orbit_flip_secs:
                        self._reverse("tracked for %.1f s"
                                      % self.a.person_orbit_flip_secs)
                else:
                    self._stare = 0.0

        self._theta += self._dir * self._speed * dt
        cx, cy = self.a.person_orbit_centre
        rr = self.a.person_orbit_radii
        i = self.index
        r = rr[i] if i < len(rr) else (rr[-1] if rr else self.a.person_orbit_radius)
        self.pos[0] = cx + r * math.cos(self._theta)
        self.pos[1] = cy + r * math.sin(self._theta)
        self.yaw = self._theta + (math.pi / 2 if self._dir >= 0 else -math.pi / 2)

    def xy(self):
        """Where this person is, for the other walkers to steer around."""
        return (self.pos[0], self.pos[1])

    def _articulation(self):
        """Physics-view handle, made on first use and retried a few times.

        Built lazily because the view is only valid once physics is stepping,
        and this object is constructed while the scene is still being assembled.
        """
        if self._art is not None or self._art_tries > 20:
            return self._art
        self._art_tries += 1
        try:
            from isaacsim.core.prims import SingleArticulation
            art = SingleArticulation(prim_path=robot_prim_path)
            art.initialize()
            art.get_world_pose()          # fails now if the view is not ready
            self._art = art
            print("[mir_isaac_sim] person walk: robot pose via physics view",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            if self._art_tries == 20:
                print("[mir_isaac_sim] person walk: physics-view pose "
                      f"unavailable, falling back to USD: {e}", flush=True)
        return self._art

    def _blocked_by_robot(self, g, rp):
        """Does the straight line to goal `g` run at the robot?

        Corridor test rather than an angle: an angle alone rejects a goal
        directly behind the robot 20 m away as readily as one that ends on its
        nose. Reject only when the robot is BOTH roughly on the way (its
        projection falls between here and the goal) and within the corridor
        half-width of that line.
        """
        if rp is None:
            return False
        dx, dy = g[0] - self.pos[0], g[1] - self.pos[1]
        leg = math.hypot(dx, dy)
        if leg < 1e-6:
            return False
        ux, uy = dx / leg, dy / leg
        rx, ry = rp[0] - self.pos[0], rp[1] - self.pos[1]
        along = rx * ux + ry * uy
        if along < 0.0 or along > leg + self.a.person_corridor:
            return False            # robot is behind, or past the far end
        perp = abs(rx * uy - ry * ux)
        return perp < self.a.person_corridor

    def _pick(self):
        xmin, xmax, ymin, ymax = self.a.person_area
        blockers = []
        if self.a.person_keep_away:
            blockers = [q for q in ([self._robot_xy()] + [o.xy() for o in self.others])
                        if q is not None]
        # Two passes. The first also demands a clear corridor; if the robot has
        # the person cornered no such goal exists, and insisting would leave the
        # person standing still -- so the second pass drops that requirement and
        # lets the close-range swerve handle it, as before.
        for require_clear in (True, False):
            for _ in range(60):
                g = (self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax))
                if math.hypot(g[0] - self.pos[0], g[1] - self.pos[1]) <= 2.0:
                    continue
                if require_clear and any(self._blocked_by_robot(g, q)
                                         for q in blockers):
                    continue
                return g
        return (self.rng.uniform(xmin, xmax), self.rng.uniform(ymin, ymax))

    def _robot_xy(self):
        art = self._articulation()
        if art is not None:
            try:
                pos, _quat = art.get_world_pose()
                return float(pos[0]), float(pos[1])
            except Exception:  # noqa: BLE001
                self._art = None          # view went stale; rebuild next tick
        # Fallback only. Known to be stale under Fabric -- see __init__.
        if not self._robot or not self._robot.IsValid():
            return None
        try:
            self._xc.Clear()
            t = self._xc.GetLocalToWorldTransform(self._robot).ExtractTranslation()
            return float(t[0]), float(t[1])
        except Exception:  # noqa: BLE001
            return None

    def step(self, dt):
        _px0, _py0 = self.pos[0], self.pos[1]
        if self.a.person_orbit:
            self._orbit(dt)
            if self.gait is not None:
                self.gait.step(math.hypot(self.pos[0] - _px0, self.pos[1] - _py0))
            self._x.SetTranslate(Gf.Vec3d(self.pos[0], self.pos[1], 0.0))
            self._x.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(self.yaw)),
                              UsdGeom.XformCommonAPI.RotationOrderXYZ)
            return self.pos, self.yaw
        if self.pause_left > 0.0:
            self.pause_left -= dt
        else:
            dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
            dist = math.hypot(dx, dy)
            if dist < 0.25:
                self.goal = self._pick()
                self.pause_left = self.rng.uniform(*self.a.person_pause)
                self.speed = self.rng.uniform(*self.a.person_speed)
            else:
                want = math.atan2(dy, dx)

                # Steer round the robot. Without this the person walks straight
                # through it -- the character has no collision on purpose, and
                # the walk is a scripted path with no awareness of anything.
                #
                # It is not a cosmetic issue. The robot backs off at 0.3 m/s and
                # the person closes at up to 0.9, so the person wins: measured
                # over 40 episodes the gap reached 0.10 m, and at that range the
                # person is completely outside the camera's view. Those frames
                # carry an expert action with nothing in the image to explain
                # it, which is exactly the kind of label a policy cannot learn
                # and will imitate anyway.
                rp = self._robot_xy()
                obstacles = [q for q in ([rp] + [o.xy() for o in self.others])
                             if q is not None]
                # The robot moves, so a goal that was clear when it was chosen
                # can come to point straight at it. Re-pick, rather than leave
                # it to the close-range swerve: swerving still walks the person
                # towards the robot and only bends the last few metres of it.
                # Cooldown because without one a cornered person re-picks every
                # tick and never gets moving.
                if self.repick_cd > 0:
                    self.repick_cd -= 1
                elif (self.a.person_keep_away
                        and any(self._blocked_by_robot(self.goal, q)
                                for q in obstacles)):
                    self.goal = self._pick()
                    self.repick_cd = 30
                    dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
                    dist = math.hypot(dx, dy)
                    want = math.atan2(dy, dx)
                # Deflect around ONE obstacle, the nearest that actually
                # triggers -- not around each in turn. Deflections are capped at
                # 85 deg apiece, so applying two in the same tick can swing the
                # heading 170 deg and send the person back the way they came,
                # which looks like the pirouetting this code was written to stop.
                _trig = []
                for _ob in obstacles:
                    _dx, _dy = _ob[0] - self.pos[0], _ob[1] - self.pos[1]
                    _d = math.hypot(_dx, _dy)
                    _rel = (math.atan2(_dy, _dx) - want + math.pi) % (2 * math.pi) - math.pi
                    if ((_d < self.a.person_avoid and abs(_rel) < math.radians(70.0))
                            or _d < self.a.person_avoid_hard):
                        _trig.append((_d, _ob))
                for _d, _ob in sorted(_trig)[:1]:
                    rdx, rdy = _ob[0] - self.pos[0], _ob[1] - self.pos[1]
                    rdist = math.hypot(rdx, rdy)
                    # Only dodge what is IN THE WAY. A robot following behind is
                    # ignored entirely -- people do not spin away from things
                    # trailing them, and making this person do so had exactly
                    # that effect: with the radius above the follower's standoff
                    # the avoidance never switched off, so the person turned
                    # away, the robot came round behind, and the "away"
                    # direction rotated with it. Measured over 40 s: 0.29 m
                    # travelled and 778 deg of turning -- pirouetting on the
                    # spot instead of walking.
                    rel = (math.atan2(rdy, rdx) - want + math.pi) % (2 * math.pi) - math.pi
                    ahead = abs(rel)
                    # Two tiers. The wide one only looks forward, so a follower
                    # sitting behind never triggers it and the person walks on
                    # normally -- that separation is what stopped the pirouetting
                    # this used to cause. The tight one ignores direction: at
                    # arm's length it does not matter where the robot came from.
                    near = rdist < self.a.person_avoid and ahead < math.radians(70.0)
                    very_near = rdist < self.a.person_avoid_hard
                    if near or very_near:
                        # Full deflection well before contact, not a ramp that is
                        # still near zero at the trigger radius. Closing speed
                        # here is up to 1.9 m/s (person 0.9 into robot 1.0), so a
                        # response that only reaches full strength at touching
                        # distance arrives after the fact: measured that way, the
                        # gap still closed to 0.15 m with the person walking
                        # straight at the robot's nose.
                        span = max(self.a.person_avoid - self.a.person_avoid_hard, 1e-3)
                        blend = min(1.0, max(0.0, (self.a.person_avoid - rdist) / span))
                        if very_near:
                            blend = 1.0
                        side = 1.0 if rel < 0 else -1.0
                        want += side * math.radians(85.0) * blend

                err = (want - self.yaw + math.pi) % (2 * math.pi) - math.pi
                lim = math.radians(self.a.person_turn_rate) * dt
                self.yaw += max(-lim, min(lim, err))
                # Only walk forwards once roughly facing the goal, the way a
                # person turns first and then sets off.
                if abs(err) < math.radians(60.0):
                    step = min(self.speed * dt, dist)
                    self.pos[0] += step * math.cos(self.yaw)
                    self.pos[1] += step * math.sin(self.yaw)
        if self.gait is not None:
            self.gait.step(math.hypot(self.pos[0] - _px0, self.pos[1] - _py0))
        self._x.SetTranslate(Gf.Vec3d(self.pos[0], self.pos[1], 0.0))
        # Vec3f: that is what SetRotate takes. The prim this drives is the
        # clean parent Xform the scene generator makes for exactly this reason
        # -- the character reference sits on a child, so no foreign op
        # declarations are in the way.
        self._x.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(self.yaw)),
                          UsdGeom.XformCommonAPI.RotationOrderXYZ)
        return self.pos, self.yaw


_walkers = []          # [(tf_frame, _PersonWalk)]
if args.walk_person:
    for _nm, _pp in PEOPLE:
        try:
            _walkers.append((_nm, _PersonWalk(stage_handle, _pp, args,
                                              index=len(_walkers),
                                              name=_nm)))
            # print, not carb.log_warn: warnings raised this late do not reach
            # the log file, and a walk that silently never starts looks exactly
            # like a person standing still.
            if args.person_orbit:
                _i = len(_walkers) - 1
                _r = (args.person_orbit_radii[_i]
                      if _i < len(args.person_orbit_radii)
                      else args.person_orbit_radius)
                _w = (args.person_orbit_speeds[_i]
                      if _i < len(args.person_orbit_speeds) else 0.2)
                print("[mir_isaac_sim] person orbit: %s (%s), r=%.1f m, "
                      "%.2f rad/s about %s -> TF %s->%s"
                      % (_pp, _nm, _r, _w, args.person_orbit_centre,
                         args.odom_frame, _nm), flush=True)
            else:
                print("[mir_isaac_sim] person walk on: %s (%s), speed %.1f-%.1f "
                      "m/s in %s -> TF %s->%s"
                      % (_pp, _nm, args.person_speed[0], args.person_speed[1],
                         args.person_area, args.odom_frame, _nm), flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[mir_isaac_sim] walker {_nm} ({_pp}) DISABLED: {e}",
                  flush=True)
            traceback.print_exc()
    # Introduce them to each other only now that they all exist, so each one
    # steers around the others as well as around the robot.
    for _nm, _w in _walkers:
        _w.others = [o for m, o in _walkers if m != _nm]
    if len(_walkers) != len(PEOPLE):
        # Loud: with one of two people missing the scene silently reverts to the
        # single-person task, and a dataset collected then teaches the policy to
        # ignore the instruction -- which is exactly what this scene exists to
        # prevent.
        print(f"[mir_isaac_sim] WARNING: {len(_walkers)} of {len(PEOPLE)} "
              f"walkers started", flush=True)

_person_dt = 1.0 / 60.0
_tf_err_shown = False

_reset_seen = None
_homes = None          # spawn poses, captured on the first reset

while simulation_app.is_running():
    simulation_context.step(render=True)

    # Episode reset, triggered by touching a file. It cannot be a ROS service:
    # Isaac runs Python 3.11 and Humble's rclpy C extension is built for 3.10,
    # so this process has no ROS client at all (same reason the TF above goes
    # out through an OmniGraph node). The file's contents, if any, are used as
    # the episode seed so a run can be repeated exactly.
    if args.reset_file and _walkers:
        try:
            _st = os.stat(args.reset_file).st_mtime_ns
        except OSError:
            _st = None
        if _st is not None and _st != _reset_seen:
            _reset_seen = _st
            try:
                with open(args.reset_file) as _f:
                    _seed = int((_f.read().strip() or "0"))
            except (OSError, ValueError):
                _seed = 0
            # Swap who starts on which side on odd seeds. Without this the
            # named colour and the side it appears on are perfectly correlated,
            # and a policy that never looks at the image -- one that just maps
            # the word "purple" to a right turn -- scores above the 100 % sum
            # that is supposed to be the evidence it read the instruction.
            # Alternating the sides makes that rule score exactly 100 % again,
            # so anything above it has to come from the picture.
            if _homes is None:
                _homes = [_w.home for _, _w in _walkers]
            _order = list(range(len(_walkers)))
            if _seed % 2 == 1 and len(_order) == 2:
                _order = [1, 0]
            for _i, (_nm, _w) in enumerate(_walkers):
                _w.reset(_seed, home=_homes[_order[_i]])
            # The robot too, otherwise it opens each episode pointing wherever
            # the last one left it. That alone decides how long the run spends
            # searching before it sees anyone, so two runs of different
            # instructions are not comparable without it -- measured 15.8 s of
            # warm-up on one and 19.8 s on the next from the same people seed.
            # Must go through the physics view: writing the USD prim moves a
            # copy PhysX does not read.
            _art = _walkers[0][1]._articulation()
            if _art is not None:
                try:
                    import numpy as _np
                    _art.set_world_pose(position=_np.array(args.reset_robot_xy
                                                           + [0.0]),
                                        orientation=_np.array([1.0, 0.0,
                                                               0.0, 0.0]))
                    _art.set_linear_velocity(_np.zeros(3))
                    _art.set_angular_velocity(_np.zeros(3))
                except Exception as _e:  # noqa: BLE001
                    print(f"[mir_isaac_sim] robot reset FAILED: {_e}", flush=True)
                else:
                    print("[mir_isaac_sim] robot reset to "
                          f"({args.reset_robot_xy[0]:+.2f}, "
                          f"{args.reset_robot_xy[1]:+.2f}) yaw 0", flush=True)
            print(f"[mir_isaac_sim] people reset (seed {_seed}, sides "
                  f"{'SWAPPED' if _order != list(range(len(_walkers))) else 'normal'}"
                  f"): " + ", ".join(
                      f"{_nm}@({_w.pos[0]:+.1f},{_w.pos[1]:+.1f})"
                      for _nm, _w in _walkers), flush=True)

    for _nm, _w in _walkers:
        (_px, _py), _pyaw = _w.step(_person_dt)
        try:
            _node = f"/ActionGraph/PublishPersonTF_{_nm}"
            og.Controller.set(
                og.Controller.attribute(f"{_node}.inputs:translation"),
                [float(_px), float(_py), 0.0])
            # The node declares inputs:rotation as quatd[4] in IJKR order --
            # x, y, z, w -- which is the opposite of Gf.Quatd(w, x, y, z). Pass
            # the array in the node's order; a Gf.Quatd is rejected outright.
            og.Controller.set(
                og.Controller.attribute(f"{_node}.inputs:rotation"),
                [0.0, 0.0, math.sin(_pyaw / 2), math.cos(_pyaw / 2)])
        except Exception as _e:  # noqa: BLE001
            # Report once rather than swallowing it: a silently failing pose
            # publish leaves the expert following a target frozen at the origin,
            # which produces a full dataset of confidently wrong behaviour.
            if not _tf_err_shown:
                print(f"[mir_isaac_sim] person TF publish failed: {_e}", flush=True)
                _tf_err_shown = True
    # tick the ROS2 publish/subscribe nodes once per frame
    og.Controller.set(
        og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True
    )
    # DIAGNOSTIC: dump what EVERY command chain receives, so the working arm can
    # be compared against the stationary base.
    _dbg += 1
    if args.wheel_test and _dbg % 60 == 0:
        for _i, _t in enumerate(args.command_topics):
            try:
                _s = f"/ActionGraph/SubscribeJointState{_i}"
                pc = list(og.Controller.get(og.Controller.attribute(f"{_s}.outputs:positionCommand")))
                vc = list(og.Controller.get(og.Controller.attribute(f"{_s}.outputs:velocityCommand")))
                jn = list(og.Controller.get(og.Controller.attribute(f"{_s}.outputs:jointNames")))
                print(f"[rt] chain{_i} {_t}: jn={len(jn)} pos={len(pc)} vel={len(vc)} "
                      f"velvals={vc}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[rt] chain{_i} readback failed: {e}", flush=True)

simulation_context.stop()
simulation_app.close()
