#!/usr/bin/env python3
"""Collect two-colour demonstrations where the instruction SWITCHES mid-episode.

Runs INSIDE my_isaac_sim. Each episode picks one of the two people, runs the
expert against that person only, and records the instruction naming them. The
robot is parked and turns on the spot; the people orbit it.

What makes this dataset different from the single-person one, and the whole
reason it exists:

    the same image has two different correct actions, and only the
    instruction says which

In the single-person dataset the instruction was the same string on all 19,373
frames, so it carried no information and the trained VLA ignored it completely
-- measured afterwards, replacing "follow the person" with "stop immediately and
do not move" changed the commanded velocity by 0.06 against a 0.34 spread across
frames, and the nonsense string "banana telescope kettle" moved it just as much.
Here the target alternates, so a policy that ignores the text can be right at
best half the time.

One target per episode was not enough. Measured on that dataset, across five
checkpoints and three training configurations, the instruction-swap rate never
left 0-2.8 %: with the target fixed for a whole episode, "keep tracking whoever
you are already tracking" reproduces the demonstration, so the instruction is
never needed and the model learns to ignore it.

Switching part-way removes that. The same scene, the same robot state, a
different correct action -- and the only thing that changed is the words. A
policy that ignores the text cannot be right on both sides of a switch.

Switch times are jittered per episode so the model cannot learn "turn around at
15 seconds" instead of reading the instruction.

    docker exec my_isaac_sim bash -c '
      source /opt/ros/humble/setup.bash; source /root/ros2_ws/install/setup.bash
      export FASTRTPS_DEFAULT_PROFILES_FILE=/root/fastdds_udp_only.xml
      python3 -u /root/vla/collect_2p.py --out /root/vla/raw_2p -n 40'
"""
import argparse
import json
import os
import random
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# One instruction per person. Deliberately differing in ONE word: if the two
# strings differed in length, punctuation or phrasing the model could separate
# them on those instead of on the colour, and would look like it had learned the
# task while having learned the shape of the sentence.
INSTRUCTIONS = {
    "purple": "follow the person in purple",
    "yellow": "follow the person in yellow",
}

# Only the platform camera is recorded. The tracker's crops are generated
# offline at conversion time, on the host, because the container has no torch --
# that is why the policy server exists in the first place. Recording one stream
# instead of three also cuts collection time and disk by two thirds.
CAMERAS = ["front:/platform_camera/color/image_raw"]


def run_bg(cmd, log):
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", "--episodes", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="episode length. Turning to face someone has no "
                         "terminal state, so the clock ends it. 45 s at 0.06 "
                         "rad/s of relative drift is about a quarter of a full "
                         "sweep of the two people's separation, so successive "
                         "episodes start from visibly different geometry.")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--max-w", type=float, default=1.2,
                    help="cap on the turn rate. 1.2, NOT the 0.4 the\n"
                         "rotate-only task used: following a person at a 2 m\n"
                         "standoff who walks at 0.9 m/s needs 0.45 rad/s of\n"
                         "yaw, so a 0.4 cap cannot keep them in front at all --\n"
                         "measured, the distance held at 2.08 m while the\n"
                         "bearing sat at 124 deg, following backwards.")
    ap.add_argument("--people", nargs="+", default=["purple", "yellow"])
    ap.add_argument("--switch-window", type=float, nargs=2, default=[8.0, 14.0],
                    metavar=("MIN", "MAX"),
                    help="seconds between switches, drawn per episode. Jittered "
                         "so the switch cannot be predicted from the clock.")
    ap.add_argument("--fps", type=float, default=10.0)
    # The recorder waits for a pose before it writes anything, so gt_pose_pub.py
    # must already be publishing /amcl_pose. It is ground truth standing in for
    # AMCL; nothing here localises. Checked up front rather than discovered as
    # forty empty episodes.
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for i in range(a.episodes):
        # Alternate which person the episode OPENS with, then switch.
        first = a.people[i % len(a.people)]
        second = a.people[(i + 1) % len(a.people)]
        rng = random.Random(1000 + i)
        t1 = rng.uniform(*a.switch_window)
        t2 = t1 + rng.uniform(*a.switch_window)
        sched = [f"0:{first}", f"{t1:.1f}:{second}"]
        if t2 < a.seconds - 4.0:
            sched.append(f"{t2:.1f}:{first}")
        target = first
        ep = os.path.join(a.out, f"ep_{i:06d}")
        os.makedirs(ep, exist_ok=True)
        instr = INSTRUCTIONS[target]
        print(f"[collect_2p] episode {i}/{a.episodes} schedule={' '.join(sched)}",
              flush=True)

        exp = run_bg(
            ["python3", "-u", os.path.join(HERE, "follow_expert.py"),
             "--target-schedule", *sched,
             # Capped well below the 1.2 default. The MiR100 is a castor base
             # carrying a UR5, and sustained maximum-rate spinning on the spot
             # rolls it onto its side: measured, the episode that went over had
             # 779 deg of turning against 328-499 for the ones that survived,
             # because the reversal-on-being-tracked rule keeps the robot
             # re-acquiring instead of settling. A real MiR would not spin at
             # 69 deg/s with an arm up either.
             "--max-w", str(a.max_w),
             "--seconds", str(a.seconds + a.settle + 5)],
            os.path.join(ep, "expert.log"))
        # Let the expert take up the turn before recording: the opening frames
        # of a run that starts mid-swing show the robot already pointed at the
        # target with a near-zero command, which is the same absorbing "stay
        # still" signal that stalled the colour-parking policy.
        time.sleep(a.settle)

        rec = run_bg(
            ["python3", "-u", os.path.join(HERE, "record_episode.py"),
             "--out", ep, "--instruction", instr,
             "--cameras", *CAMERAS,
             "--people", *a.people, "--target", target,
             "--fps", str(a.fps), "--seconds", str(a.seconds),
             ],
            os.path.join(ep, "record.log"))
        rec.wait()
        exp.terminate()
        try:
            exp.wait(timeout=10)
        except subprocess.TimeoutExpired:
            exp.kill()

        # "episode.json exists" is not success. A tipped robot writes a
        # complete, full-length, entirely useless episode: measured, three in a
        # row with 0.00 m of movement and 0 deg of turning, all marked ok. So
        # check what the recorder actually saw.
        meta_path = os.path.join(ep, "episode.json")
        ok, why, steps = False, "no episode.json", 0
        if os.path.exists(meta_path):
            em = json.load(open(meta_path))
            steps = em.get("num_steps", 0)
            if em.get("tipped"):
                why = f"robot tipped ({em.get('max_tilt_deg')} deg)"
            elif steps < 0.5 * a.seconds * a.fps:
                why = f"only {steps} steps"
            else:
                ok, why = True, f"{steps} steps"
        manifest.append({"episode": f"ep_{i:06d}", "target": target,
                         "schedule": sched, "instruction": instr,
                         "ok": ok, "steps": steps, "why": why})
        print(f"[collect_2p]   -> {'ok' if ok else 'FAILED'}: {why}", flush=True)

        # Stop rather than grind through the rest. Once the robot is over it
        # stays over -- nothing here rights it -- so every later episode is the
        # same wasted recording, and the run would end with a directory that
        # looks full and is mostly unusable.
        recent = [m["ok"] for m in manifest[-3:]]
        if len(recent) == 3 and not any(recent):
            print("[collect_2p] STOPPING: 3 consecutive failures. The sim needs "
                  "attention (a tipped robot does not recover on its own).",
                  flush=True)
            with open(os.path.join(a.out, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            break
        with open(os.path.join(a.out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    good = sum(1 for m in manifest if m["ok"])
    # Count FRAMES per target, not episodes. With switching, every episode
    # contains both targets, so an episode-level count says nothing about
    # whether the two instructions are balanced -- which is the thing that
    # stops a policy scoring well by always picking the same person.
    frames = {p: 0 for p in a.people}
    switches = 0
    for m in manifest:
        if not m["ok"]:
            continue
        ep_meta = os.path.join(a.out, m["episode"], "episode.json")
        if not os.path.exists(ep_meta):
            continue
        em = json.load(open(ep_meta))
        switches += em.get("switches", 0)
        for st in em["steps"]:
            t = st.get("target")
            if t in frames:
                frames[t] += 1
    tot = sum(frames.values()) or 1
    print(f"[collect_2p] {good}/{len(manifest)} episodes, {switches} switches; "
          f"frames per target " +
          ", ".join(f"{p} {frames[p]} ({100*frames[p]/tot:.0f}%)"
                    for p in a.people), flush=True)
    if good and min(frames.values()) < 0.3 * tot / len(a.people):
        print("[collect_2p] WARNING: the two instructions are badly unbalanced; "
              "a policy can score by always picking the commoner one",
              flush=True)


if __name__ == "__main__":
    main()
