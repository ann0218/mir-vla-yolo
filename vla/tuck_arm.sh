#!/bin/bash
# Fold the UR5 back over the deck, via ros2_control.
#
# Doing this at Isaac startup does NOT work: joint_trajectory_controller comes
# up afterwards and holds the arm at its own initial state, overriding whatever
# pose the sim script teleported it to (measured 0.616 m instead of 0.460 m).
# The controller is the only thing that actually decides where the arm sits, so
# the command has to go through it, after the stack is up.
#
# Measured from base_footprint (chassis half-length 0.445 m, circumscribed
# radius 0.531 m):
#   default  wrist 0.682 m ahead -> 24 cm proud of the chassis, and 1.41 m up
#            where the laser never sees it: this is what wedged the robot on
#            door frames and furniture over and over.
#   tucked   wrist 0.460 m, forearm -0.027 m -> 1.5 cm proud, and inside the
#            chassis's own turning circle, so it cannot catch while rotating.
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/root/fastdds_udp_only.xml
J="[ur_shoulder_pan_joint,ur_shoulder_lift_joint,ur_elbow_joint,ur_wrist_1_joint,ur_wrist_2_joint,ur_wrist_3_joint]"
timeout 25 ros2 topic pub --times 5 /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: $J, points: [{positions: [0.0,-2.60,2.60,-1.57,0.0,0.0], time_from_start: {sec: 3}}]}" \
  >/dev/null 2>&1
# Poll the JOINTS, not the wrist pose. During the fold the wrist x/y pass
# through their final values long before the arm has settled (measured
# [0.438, 0.328, 1.625] on the way to [0.460, 0.331, 1.10]), so a check on
# x/y alone reports success ~15 s early. The elbow angle is unambiguous.
for i in $(seq 1 25); do
  e=$(timeout 6 ros2 topic echo /joint_states --once 2>/dev/null \
      | python3 -c "
import sys,yaml
d=yaml.safe_load(sys.stdin.read().split('---')[0])
print('%.3f'%d['position'][d['name'].index('ur_elbow_joint')])
" 2>/dev/null)
  [ -n "$e" ] && awk -v v="$e" 'BEGIN{exit !(v>2.55 && v<2.65)}' && break
  sleep 2
done
r=$(timeout 10 ros2 run tf2_ros tf2_echo base_footprint ur_wrist_3_link 2>/dev/null | grep -m1 "Translation")
echo "[tuck] elbow=$e  $r"
echo "$r" | grep -qE "1\.(09|1[01])" && echo "[tuck] OK (folded, wrist at chassis height)" || echo "[tuck] WARNING: arm not fully tucked"
