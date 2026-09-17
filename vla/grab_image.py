#!/usr/bin/env python3
"""Save one frame from a ROS image topic, with correct colours.

Runs inside my_isaac_sim:  python3 grab_image.py /platform_camera/color/image_raw out.png

Goes through cv_bridge to bgr8, the same conversion record_episode.py and
runner_2p.py use. An earlier ad-hoc version wrote msg.data straight to
cv2.imwrite; the camera publishes rgb8, so red and blue came out swapped -- the
yellow person looked teal and the purple one magenta, and that screenshot was
wrongly taken as what the policy sees.
"""
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

topic, out = sys.argv[1], sys.argv[2]
rclpy.init()
node = rclpy.create_node("grab_image")
got = []
node.create_subscription(Image, topic, got.append, 10)
while len(got) < 3:
    rclpy.spin_once(node, timeout_sec=1.0)
msg = got[-1]
cv2.imwrite(out, CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8"))
print(f"saved {out} (topic encoding {msg.encoding})")
