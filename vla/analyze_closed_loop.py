#!/usr/bin/env python3
"""Paired analysis of eval_closed_loop.sh output.

Two scores per run:

  on_target  the runner's own: named person nearer the centreline than the
             other. Weak -- it still counts when both are behind the robot.
  in_view    fraction of steps with the named person inside the platform
             camera's 75.2 deg horizontal view (|bearing| < 37.6 deg). Needs
             the per-step trace, so only seeds run with --trace have it.

For each seed the purple and yellow runs start from the same configuration, so
the test is on the per-seed SUM. A policy that ignores the instruction and
always tracks one physical person scores about 100 % on on_target; odd seeds
swap which side purple starts on, so a colour-word-to-turn-direction rule also
lands at 100 %. in_view has no such anchor -- a policy that stares at whoever
is in front can keep both in view -- so it is compared as purple-run vs
yellow-run difference in "named minus other in view", which is zero for any
rule that ignores the instruction.

    python3 analyze_closed_loop.py results_2p/cl_batch_balanced
"""
import csv
import glob
import math
import os
import re
import statistics as st
import sys
from math import comb

HALF_FOV = 37.6


def t_two_sided(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan")
    sd = st.stdev(xs)
    if sd == 0:
        return float("inf"), 0.0
    t = st.mean(xs) / (sd / math.sqrt(n))
    v = n - 1
    c = math.gamma((v + 1) / 2) / (math.sqrt(v * math.pi) * math.gamma(v / 2))
    N, span = 40000, 80.0
    tail = sum(c * (1 + ((abs(t) + (k + .5) * span / N) ** 2) / v) ** (-(v + 1) / 2)
               * span / N for k in range(N))
    return t, 2 * tail


def sign_p(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "s*_*.log")):
        s, c = re.match(r"s(\d+)_(\w+)\.log", os.path.basename(f)).groups()
        m = re.search(r"on_target=(\d+)% median\|bearing\| all=(\d+)deg "
                      r"late=(\d+)deg longest_lock=([\d.]+)s",
                      open(f, errors="ignore").read())
        if not m:
            print(f"no RESULT in {f} (still running, tipped, or crashed) "
                  f"-- pair dropped")
            continue
        r = {"on": float(m.group(1)), "med": float(m.group(2)),
             "late": float(m.group(3))}
        tr = f[:-4] + ".csv"
        if os.path.exists(tr):
            rows = list(csv.DictReader(open(tr)))
            bt = [abs(float(x["bear_target"])) for x in rows]
            bo = [abs(float(x["bear_other"])) for x in rows]
            r["tv"] = 100 * sum(b < HALF_FOV for b in bt) / max(len(bt), 1)
            r["ov"] = 100 * sum(b < HALF_FOV for b in bo) / max(len(bo), 1)
            # Driving runs (runner_2p.py --drive) also log distances.
            if rows and "dist_target" in rows[0]:
                dts = [float(x["dist_target"]) for x in rows if x["dist_target"] not in ("", "nan")]
                dos = [float(x["dist_other"]) for x in rows if x["dist_other"] not in ("", "nan")]
                vs = [float(x["v_cmd"]) for x in rows]
                if dts:
                    r["dmean"] = st.mean(dts)
                    r["band"] = 100 * sum(1.5 <= d <= 3.0 for d in dts) / len(dts)
                    r["closest"] = min(dts + dos)
                    r["estops"] = sum(int(x["estop"]) for x in rows)
                    r["vmean"] = st.mean(vs)
        runs[(int(s), c)] = r

    words = sorted({c for _, c in runs})
    if len(words) != 2:
        raise SystemExit(f"expected two instruction words, found {words}")
    W0, W1 = words
    seeds = sorted({s for s, _ in runs if (s, W0) in runs and (s, W1) in runs})
    print(f"seed side   {W0:<6} on/view(named,other)   {W1:<6} on/view(named,other)   on-sum"
          "   (side = where the purple TF frame starts)")
    sums, diffs = [], []
    for s in seeds:
        p, y = runs[(s, W0)], runs[(s, W1)]
        sums.append(p["on"] + y["on"])
        vw = ""
        if "tv" in p and "tv" in y:
            # named-minus-other in view, summed over both runs: a rule that
            # ignores the instruction gives the same (target,other) split with
            # roles swapped, so the two terms cancel to 0.
            diffs.append((p["tv"] - p["ov"]) + (y["tv"] - y["ov"]))
            vw = f"  view-diff {diffs[-1]:+5.0f}"
        f = lambda r: (f"{r['on']:3.0f}% " + (f"({r['tv']:3.0f}%,{r['ov']:3.0f}%)"
                                               if "tv" in r else "(   -,    -)"))
        print(f"  {s:2d}  {'L' if s % 2 else 'R'}     {f(p)}          {f(y)}         "
              f"{sums[-1]:4.0f}%{vw}")

    d100 = [x - 100 for x in sums]
    t, p = t_two_sided(d100)
    k = sum(x > 0 for x in d100)
    print(f"\non_target sum - 100 over {len(d100)} pairs: mean {st.mean(d100):+.1f}, "
          f"t = {t:.2f}, two-sided p = {p:.3f}; {k}/{len(d100)} above, "
          f"sign p = {sign_p(k, len(d100)):.3f}")
    if diffs:
        t, p = t_two_sided(diffs)
        k = sum(x > 0 for x in diffs)
        print(f"in-view named-minus-other over {len(diffs)} traced pairs: "
              f"mean {st.mean(diffs):+.1f} pts, t = {t:.2f}, two-sided p = {p:.3f}; "
              f"{k}/{len(diffs)} above 0, sign p = {sign_p(k, len(diffs)):.3f}")
        tv = [r["tv"] for r in runs.values() if "tv" in r]
        print(f"named person in view, mean over traced runs: {st.mean(tv):.1f} %")
    drv = [r for r in runs.values() if "dmean" in r]
    if drv:
        print(f"driving ({len(drv)} runs): named-person distance mean {st.mean(r['dmean'] for r in drv):.2f} m, "
              f"within 1.5-3.0 m {st.mean(r['band'] for r in drv):.1f} %, "
              f"commanded v mean {st.mean(r['vmean'] for r in drv):+.2f} m/s")
        print(f"  closest approach to anyone: min {min(r['closest'] for r in drv):.2f} m, "
              f"median {st.median(r['closest'] for r in drv):.2f} m; "
              f"emergency stops {sum(r['estops'] for r in drv)} steps in "
              f"{sum(r['estops'] > 0 for r in drv)}/{len(drv)} runs")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_2p/cl_batch_balanced")
