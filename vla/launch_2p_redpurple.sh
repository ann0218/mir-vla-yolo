#!/usr/bin/env bash
# Bring up the two-colour scene for the MOVING follow task.
#
# Differs from the orbit build in one flag, and the reason is measured: with the
# people circling at a fixed radius the robot cannot close on anyone. It starts
# at the centre, the target moves sideways relative to it, and the gap sat at
# 5.15 m against a 2.0 m standoff for a whole minute of expert driving. Orbiting
# is the right scene for "turn to face the named person" and the wrong one for
# "go to them", so this build puts the people back on random waypoints.
#
# --person-keep-away is deliberately OFF. It tells each person to avoid the
# robot's path, which in a two-person scene amounts to ordering the distractor
# never to appear in front of the camera: measured with it on, the distractor
# was in frame 0 % of 900 ticks and the instruction had nothing to choose
# between.
#
#   bash launch_2p_move.sh
# No `set -u` on LD_LIBRARY_PATH: it is normally unset in a fresh
# shell, and an unbound-variable abort here looks like Isaac failing
# to start rather than the launcher refusing to try.
set -uo pipefail
cd /home/itri/mir_isaac_test/mir_robot/isaac_sim
S=/home/itri/mir_isaac_test/vla/results_2p

DISPLAY=:1 OMNI_KIT_ACCEPT_EULA=Y ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
FASTRTPS_DEFAULT_PROFILES_FILE=/home/itri/mir_isaac_test/fastdds_udp_only.xml \
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/itri/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib" \
setsid nohup /home/itri/anaconda3/envs/env_isaaclab/bin/python mir_isaac_sim.py \
  --lasers --platform-camera --global-camera --corner-cameras \
  --walk-person --person-gait \
  --people purple:/World/PersonPurple yellow:/World/PersonYellow \
  --person-area -6.0 6.0 -4.0 4.0 \
  --person-home purple:4.5,-1.3 --person-home yellow:4.5,1.3 \
  --reset-file /tmp/mir_reset_people \
  --person-avoid 3.5 --person-avoid-hard 2.0 \
  --fold-arm --max-depenetration-velocity 3.0 \
  --world /home/itri/mir_isaac_test/scenes/follow_room_2p_v2_redpurple.usd \
  --top-down --top-down-height 22 \
  > "$S/isaac_2p_redpurple.log" 2>&1 < /dev/null &

echo "Isaac starting; give it ~100 s, then check:"
echo "  grep -a 'person walk on\\|corner camera\\|global camera' $S/isaac_2p_redpurple.log"
