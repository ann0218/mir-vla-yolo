#!/usr/bin/env python3
"""Turn a directory of runner_2p.py --save-frames PNGs into an mp4.

Neither the host nor the container has ffmpeg, so this uses OpenCV's mp4v
writer, which is present in env_lerobot.

    python3 make_clip.py results_2p/cl_batch_box/vid_box out.mp4 --fps 10
"""
import argparse
import glob
import os
import sys

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="the runner ticks at 10 Hz, so 10 plays at sim speed")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.frames, "*.png")))
    if not files:
        sys.exit(f"no PNGs in {a.frames}")
    first = cv2.imread(files[0])
    if first is None:
        sys.exit(f"cannot read {files[0]}")
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    if not vw.isOpened():
        sys.exit("OpenCV could not open an mp4v writer")
    n = 0
    for f in files:
        img = cv2.imread(f)
        if img is None or img.shape[:2] != (h, w):
            continue
        vw.write(img)
        n += 1
    vw.release()
    print(f"wrote {a.out}: {n} frames, {w}x{h}, {a.fps:g} fps, {n/a.fps:.1f} s")


if __name__ == "__main__":
    main()
