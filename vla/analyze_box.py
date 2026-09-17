#!/usr/bin/env python3
"""Score the box a --box-action policy predicts, from closed-loop traces.

The policy trained with --box-action outputs [v, w, x1, y1, x2, y2, visible].
Only v and w drive the robot; runner_2p.py logs the rest as pred_x1..pred_vis.
This reads those columns and asks two things the action alone cannot answer:

  is it looking at the right person   predicted box centre against the image
                                      column where the NAMED person actually is
  does it know when it has lost them  predicted visibility against whether the
                                      named person is really inside the 75.2 deg
                                      view

Ground truth comes from the bearings the runner logged for scoring, mapped to an
image column with the same formula the labels used:

    u = cx - fx * (y / x),  x = d cos(bearing),  y = d sin(bearing)

The minus is not cosmetic: ROS bearings grow anticlockwise, image columns grow to
the right. With it the wrong way round, labels land on the other person (measured
over 73 frames: median error 267 px against 20 px).

    python3 analyze_box.py results_2p/cl_batch_box
"""
import csv
import glob
import math
import os
import statistics as st
import sys

FX, CX = 415.75, 320.0
HALF_FOV = math.radians(37.6)
NEAR_PX = 60.0          # a predicted centre this close counts as "on the person"


def column(dist, bearing):
    x, y = dist * math.cos(bearing), dist * math.sin(bearing)
    return None if x <= 0.05 else CX - FX * (y / x)


def main(d):
    files = sorted(glob.glob(os.path.join(d, "s*_*.csv")))
    if not files:
        sys.exit(f"no traces in {d}")
    n = vis_frames = 0
    say_vis = agree = 0
    errs_named, errs_other = [], []
    on_named = on_other = on_neither = 0
    per_run = []
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if not rows or "pred_vis" not in rows[0]:
            continue
        run_named = run_n = 0
        for r in rows:
            n += 1
            bt, bo = math.radians(float(r["bear_target"])), math.radians(float(r["bear_other"]))
            dt, do = float(r["dist_target"]), float(r["dist_other"])
            really = abs(bt) < HALF_FOV
            vis_frames += really
            said = float(r["pred_vis"]) > 0.5
            say_vis += said
            agree += (said == really)
            if not said:
                continue
            cx_pred = (float(r["pred_x1"]) + float(r["pred_x2"])) / 2.0
            ut, uo = column(dt, bt), column(do, bo)
            if ut is not None and really:
                e = abs(cx_pred - ut)
                errs_named.append(e)
                run_n += 1
                run_named += e < NEAR_PX
            near_t = ut is not None and abs(cx_pred - ut) < NEAR_PX
            near_o = uo is not None and abs(cx_pred - uo) < NEAR_PX
            if uo is not None and abs(bo) < HALF_FOV:
                errs_other.append(abs(cx_pred - uo))
            if near_t and not near_o:
                on_named += 1
            elif near_o and not near_t:
                on_other += 1
            elif not near_t and not near_o:
                on_neither += 1
        if run_n:
            per_run.append((os.path.basename(f), 100 * run_named / run_n, run_n))
    if not n:
        sys.exit("traces have no pred_* columns -- was the policy trained with --box-action?")
    print(f"{len(files)} runs, {n} steps")
    print(f"visibility: policy says visible {100*say_vis/n:.1f} % of steps, "
          f"named person really in view {100*vis_frames/n:.1f} %, agreement {100*agree/n:.1f} %")
    if errs_named:
        e = errs_named
        print(f"box centre vs the named person ({len(e)} steps where it says visible and they are): "
              f"median {st.median(e):.0f} px, p90 {sorted(e)[int(.9*len(e))-1]:.0f} px, "
              f"within {NEAR_PX:.0f} px {100*sum(x < NEAR_PX for x in e)/len(e):.0f} %")
    print(f"where the predicted box points: named person {on_named}, other person {on_other}, "
          f"neither {on_neither}")
    print("\nper run (steps scored, % of them with the box on the named person):")
    for name, pct, cnt in per_run:
        print(f"  {name:<18} {cnt:4d}  {pct:5.1f} %")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_2p/cl_batch_box")
