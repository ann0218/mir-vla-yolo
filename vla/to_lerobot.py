#!/usr/bin/env python3
"""Turn the raw episode dumps into a LeRobot v3.0 dataset.

Runs on the HOST, not in my_isaac_sim: the container has no pandas / pyarrow /
av / ffmpeg, and adding them there would mean upgrading numpy out from under
the ROS packages. Use the conda env built for this:

    /home/itri/anaconda3/envs/env_lerobot/bin/python to_lerobot.py \
        --raw /home/itri/mir_isaac_test/vla/raw \
        --repo-id itri/mir_ward_nav \
        --root /home/itri/mir_isaac_test/vla/lerobot

Verified against lerobot 0.4.4 (CODEBASE_VERSION v3.0), where the task string
rides inside the frame dict and save_episode() takes no arguments.

Input is whatever record_episode.py wrote (frames/*.png + episode.json +
scan.npy per episode); output is a dataset LeRobot can load directly and
`lerobot-train` can consume.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

# lerobot moved this module between releases; support both rather than pinning
# the user to one version.
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # lerobot < 0.3
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, "episode.json")) as f:
        meta = json.load(f)
    scan_path = os.path.join(ep_dir, "scan.npy")
    scans = np.load(scan_path) if os.path.exists(scan_path) else None
    return meta, scans


def minpool_scan(row, k, clip):
    """360 beams -> k sectors, keeping the NEAREST return in each sector.

    Min (not mean) because what matters for obstacle avoidance is the closest
    thing in a direction; averaging lets one far reading hide a near one.
    Ranges are clipped because the sensor reports its 29 m max-range value for
    "no return", and letting that through would dominate normalisation while
    saying nothing useful about a corridor 2 m wide.
    """
    return np.minimum(row.reshape(k, -1).min(axis=1), clip)


def goal_in_robot_frame(state, goal_xy):
    """Where the goal is from the robot's point of view: (forward, left) metres.

    Absolute pose plus a *language* goal is what the first model got, and it
    could not use it: measured on this data, knowing the destination barely
    changes the action Nav2 took, because every destination sits down the same
    corridor and the paths only diverge in the last few metres. A goal vector
    in the robot frame states the direction outright, which is the one thing
    the observation was missing -- and it is exactly what disambiguates the
    divergence at the end.
    """
    x, y, cos_yaw, sin_yaw = state[0], state[1], state[2], state[3]
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    return np.array([cos_yaw * dx + sin_yaw * dy,       # ahead of the robot
                     -sin_yaw * dx + cos_yaw * dy],     # to its left
                    dtype=np.float32)


def spin_mask(steps, min_run):
    """Frames inside a long stretch of turning on the spot.

    A second or two of rotating in place is a legitimate correction, and Nav2
    does it whenever the heading is badly wrong. Runs of tens of seconds are
    something else: the MPPI livelock this project has hit before, where the
    robot oscillates in an open space making no progress. Measured here,
    perturbing the robot doubles in-place rotation (4.5% -> 10.8% of frames)
    and produced a single 687-frame (68 s) spin -- half of that episode.

    Training on those frames teaches the policy to stand and spin, which is the
    one failure mode that guarantees it never reaches a goal, so drop them.
    """
    if not min_run:
        return np.zeros(len(steps), dtype=bool)
    a = np.asarray([s["action"] for s in steps], dtype=np.float32)
    spinning = (np.abs(a[:, 0]) < 1e-6) & (np.abs(a[:, 1]) > 0.5)
    mask = np.zeros(len(steps), dtype=bool)
    run = 0
    for i, v in enumerate(spinning):
        run = run + 1 if v else 0
        if not v and run == 0:
            continue
        if run >= min_run:                      # mark the whole run, not its tail
            mask[i - run + 1:i + 1] = True
    return mask


def lead_still_count(steps):
    """How many frames at the START of an episode the expert spent not moving.

    Every episode opens with ~2 s of the robot sitting still while Nav2 plans
    (measured: median 19-21 frames, 3.3-3.8% of all frames). Those frames teach
    "stopped -> stay stopped", and a memoryless policy that learns it faithfully
    can never start: act_red held v=-0.003 for 500 closed-loop steps and the
    robot did not move at all, while reproducing the expert exactly everywhere
    else. ACT imitates this MORE precisely than SmolVLA did, which is why it
    stalls where SmolVLA merely drove badly.

    Dropping them makes the first frame a policy ever sees the one where the
    expert commits, and that frame carries the whole decision: over the 10
    frames after motion starts, mean w is -0.64 for the right-hand bay, -0.00
    for the middle and +0.65 for the left (consistent across all three colours).
    Same argument as spin_mask() -- drop the frames that teach a behaviour which
    guarantees failure.
    """
    a = np.asarray([s["action"] for s in steps], dtype=np.float32)
    moving = np.abs(a).sum(axis=1) > 1e-6
    return int(np.argmax(moving)) if moving.any() else len(steps)


# Target name -> the instruction that names them. Kept here rather than read
# from the dump so a dataset cannot end up with two spellings of the same
# instruction, which would look like two tasks to the trainer.
PER_STEP_TASK = {
    "purple": "follow the person in purple",
    "yellow": "follow the person in yellow",
}


def build_state(step, scans, i, k, clip, goal_xy=None, drop_xy=False,
                drop_yaw=False, drop_vel=False):
    """Pose, then the goal vector, then the down-sampled scan.

    SmolVLA reads only `observation.state` (see README): separate features such
    as `observation.scan` are silently ignored by the policy, so anything the
    model must see has to be concatenated here. Order is part of the contract --
    policy_runner.py rebuilds this same layout at inference time.
    """
    state = np.asarray(step["state"], dtype=np.float32)
    goal = goal_in_robot_frame(state, goal_xy) if goal_xy is not None else None
    # Layout in: [x, y, cos_yaw, sin_yaw, v, w]. Both drops are computed against
    # that fixed layout, and the goal vector above is taken BEFORE either, since
    # it needs the pose it is relative to.
    drop = set()
    if drop_xy:
        # Absolute x,y is the shortcut that lets the policy ignore the goal:
        # measured on this data, pose alone predicts the action with R^2 0.77-0.85,
        # so memorising "at this spot, do this" scores well without ever reading
        # the instruction -- and that is exactly the policy that drives confidently
        # to the wrong place. Everything that survives here is relative to the
        # robot, so the goal vector becomes the only thing saying where to go.
        drop |= {0, 1}
    if drop_yaw:
        # Global heading is the second shortcut, and for the colour task it is
        # the decisive one. The three cubes never move, so a policy that reads
        # cos/sin yaw can learn "red means finish facing 5 deg right" without
        # ever using a pixel -- and the correlation is real and clean: over nine
        # parking runs Nav2 left bay 0 at -5.3 deg and bay 2 at +5.3 deg every
        # single time. Dropping it leaves [v, w] plus the scan, so the ONLY
        # thing that can say which cube is red is the camera.
        drop |= {2, 3}
    if drop_vel:
        # The robot's own [v, w] is the last shortcut, and on the colour task it
        # is the one that mattered. The expert commits to a bay in the first
        # second and turns towards it, so from then on "keep doing what I am
        # doing" reproduces the right action without the colour ever being read.
        # Measured on smolvla_color: corr(commanded w, the w already in state)
        # = +0.56, sign accuracy 90% while already turning against 50% when
        # nearly stationary, and counterfactual colours changed nothing.
        drop |= {4, 5}
    if drop:
        state = np.array([v for j, v in enumerate(state) if j not in drop],
                         dtype=np.float32)
    if goal is not None:
        state = np.concatenate([state, goal])
    if k and scans is not None and i < len(scans):
        state = np.concatenate([state, minpool_scan(scans[i], k, clip)])
    return state.astype(np.float32)


# --box-action: where the instruction's person is in the frame, as extra action
# dimensions. The detections already exist (the crop pipeline runs YOLO on every
# frame and throws the boxes away), and the recorded ground truth says which
# person the instruction names, so the label is free.
#
# Picking the right detection: a person at distance d and bearing b sits at
# image column u = cx - fx * (y/x), y = d sin b, x = d cos b. The MINUS matters
# -- ROS bearings count anticlockwise (left positive) while image columns grow
# to the right. With the sign the wrong way round the match lands on the other
# side of the frame: measured over 73 frames, median error 267 px and 18 % clean
# matches, against 20 px and 87 % with it right.
BOX_NAMES = ["box_x1", "box_y1", "box_x2", "box_y2", "box_visible"]
BOX_FX, BOX_CX = 415.75, 320.0
BOX_HALF_FOV = math.radians(37.6)
BOX_MATCH_PX = 60.0      # a detection further than this is not the named person
BOX_MARGIN_PX = 20.0     # and if the OTHER person is about as close, drop it


def _box_u(dist, bearing):
    x, y = dist * math.cos(bearing), dist * math.sin(bearing)
    return None if x <= 0.05 else BOX_CX - BOX_FX * (y / x)


def box_label(step, boxes):
    """[x1, y1, x2, y2, visible] for the person the instruction names.

    All zeros when the named person is out of frame, undetected, or cannot be
    told apart from the other one -- about 13 % of frames in the training dump.
    The visibility flag is what the policy can use to say "I have lost them",
    which its two action dimensions could never express.
    """
    pp = step.get("people") or {}
    tgt = step.get("target")
    others = [k for k in pp if k != tgt]
    zero = np.zeros(len(BOX_NAMES), dtype=np.float32)
    if tgt not in pp or not others or not boxes:
        return zero
    dist, bear = pp[tgt][0], pp[tgt][1]
    if abs(bear) > BOX_HALF_FOV:
        return zero
    ut = _box_u(dist, bear)
    if ut is None:
        return zero
    uo = _box_u(pp[others[0]][0], pp[others[0]][1])
    cxs = [b["cx"] for b in boxes]
    j = min(range(len(cxs)), key=lambda k: abs(cxs[k] - ut))
    err = abs(cxs[j] - ut)
    if err > BOX_MATCH_PX:
        return zero
    if uo is not None and abs(cxs[j] - uo) < err + BOX_MARGIN_PX:
        return zero
    x1, y1, x2, y2 = boxes[j]["xyxy"]
    return np.asarray([x1, y1, x2, y2, 1.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory of ep_* dumps")
    ap.add_argument("--repo-id", default="itri/mir_ward_nav")
    ap.add_argument("--root", required=True, help="where to write the dataset")
    ap.add_argument("--robot-type", default="mir100")
    ap.add_argument("--skip-failed", action="store_true",
                    help="drop episodes the manifest marked unsuccessful")
    ap.add_argument("--scan-in-state", type=int, default=0, metavar="K",
                    help="min-pool the scan into K sectors and append them to "
                         "observation.state, so policies that only read the state "
                         "vector (SmolVLA) can actually see the lidar. K must "
                         "divide the scan width, and 6+K must stay within the "
                         "policy's max_state_dim (32 for SmolVLA). 0 = off.")
    ap.add_argument("--scan-clip", type=float, default=10.0, metavar="M",
                    help="clip pooled ranges to M metres (default 10)")
    ap.add_argument("--drop-spin", type=int, default=0, metavar="N",
                    help="drop frames inside an in-place rotation lasting N or "
                         "more steps (50 = 5 s at 10 Hz). Targets the MPPI "
                         "livelock; short corrective spins are kept. 0 = off.")
    ap.add_argument("--make-crops", action="store_true",
                    help="run a person tracker over the recorded camera\n"
                         "and emit three streams instead of one: the frame\n"
                         "with every person boxed, then the left and right\n"
                         "person each cropped and blown up to full size.\n"
                         "Done here rather than during collection because\n"
                         "the container has no torch.")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--box-action", action="store_true",
                    help="append [x1, y1, x2, y2, visible] of the person the\n"
                         "instruction names to every action, so the policy is\n"
                         "trained to say WHERE it thinks that person is as well\n"
                         "as how to move. Only the first two dims drive the\n"
                         "robot; the rest are read back to see whether a wrong\n"
                         "turn came from looking at the wrong person or from\n"
                         "steering badly. Requires --make-crops.")
    ap.add_argument("--min-separation", type=float, default=0.0,
                    help="drop frames where the two recorded people are\n"
                         "within N degrees of each other as seen from the\n"
                         "robot. At a small separation both instructions\n"
                         "call for the same action, so the frame cannot\n"
                         "teach -- or test -- whether the instruction was\n"
                         "read. 0 disables.")
    ap.add_argument("--drop-near", type=float, default=0.0, metavar="M",
                    help="drop frames where the recorded person_dist is under M "
                         "metres. For the follow task: the platform camera "
                         "cannot frame a person nearer than ~1.5 m and by 1.0 m "
                         "they are out of shot entirely, so those frames carry "
                         "an expert action with nothing in the image to explain "
                         "it -- the same unlearnable pairing that made the "
                         "colour policies imitate a standstill. Four rounds of "
                         "tuning the simulated person's avoidance still left "
                         "~9%% of frames under 1.2 m, so this filters what the "
                         "simulation would not reliably prevent. Requires "
                         "record_episode.py to have logged person_dist.")
    ap.add_argument("--holdout", type=int, default=0, metavar="N",
                    help="keep N episodes OUT of the dataset and write them to "
                         "<root>_val instead, as a same-distribution validation "
                         "split. lerobot-train has no validation split of its "
                         "own, so without this the only check on generalisation "
                         "is a differently-distributed proxy, where the gap's "
                         "absolute size confounds overfitting with distribution "
                         "shift. Episodes are taken evenly across stations so "
                         "the split is not all one bay.")
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--drop-lead-still", action="store_true",
                    help="drop the stationary frames at the START of each "
                         "episode, before the expert first moves. Required for "
                         "any memoryless policy: those frames teach 'stopped -> "
                         "stay stopped', which is an absorbing state the policy "
                         "can never leave (measured on ACT: 0 m travelled in 500 "
                         "steps). See lead_still_count().")
    ap.add_argument("--no-raw-scan", action="store_true",
                    help="do not write the separate 360-dim observation.scan "
                         "feature. SmolVLA ignores it anyway, but ACT keys every "
                         "observation.* of STATE type into its state token and "
                         "then needs it present at inference too. With "
                         "--scan-in-state the pooled sectors already ride inside "
                         "observation.state, so the raw scan is pure overhead "
                         "for ACT. Leave off for SmolVLA datasets.")
    ap.add_argument("--drop-vel", action="store_true",
                    help="also drop the robot's own v,w. Removes the "
                         "motion-continuity shortcut, which is what let the "
                         "colour policy score well without reading colour")
    ap.add_argument("--drop-yaw", action="store_true",
                    help="also drop cos_yaw/sin_yaw. Required for the colour "
                         "task, where global heading alone identifies the bay; "
                         "policy_runner must be given the matching flag")
    ap.add_argument("--drop-abs-xy", action="store_true",
                    help="omit absolute x,y from observation.state, leaving only "
                         "robot-relative quantities. Removes the pose-memorisation "
                         "shortcut so the goal vector has to carry the direction.")
    ap.add_argument("--goal-in-state", action="store_true",
                    help="append the goal, in the robot's own frame, to "
                         "observation.state. The goal is taken from each "
                         "episode's LAST recorded pose -- exact for every "
                         "waypoint including the region-sampled 大廳, whose "
                         "per-episode goal is not recorded anywhere else.")
    args = ap.parse_args()
    if args.box_action and not args.make_crops:
        sys.exit("--box-action needs --make-crops: the boxes come from the "
                 "crop pipeline's detections")

    import cv2

    ep_dirs = sorted(d for d in os.listdir(args.raw) if d.startswith("ep_"))
    # Drop what the collector marked failed, and anything with no episode.json.
    # A tipped robot leaves a directory behind -- sometimes with frames and a
    # truncated recording, sometimes empty -- and scanning the filesystem picks
    # both up: the run that produced this dump crashed on an empty ep_000025
    # and had already folded 39 steps of a robot going over into the dataset.
    bad = [d for d in ep_dirs
           if not os.path.exists(os.path.join(args.raw, d, "episode.json"))]
    if bad:
        print(f"skipping {len(bad)} episode(s) with no episode.json: "
              + ", ".join(bad))
        ep_dirs = [d for d in ep_dirs if d not in bad]
    if not ep_dirs:
        sys.exit(f"no ep_* directories in {args.raw}")

    manifest_path = os.path.join(args.raw, "manifest.json")
    ok, station, why = {}, {}, {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            man = json.load(f)
        # Two manifest shapes exist. The colour and single-person collectors
        # wrote {"dir": ..., "success": ...}; collect_2p.py writes
        # {"episode": "ep_000000", "ok": ...}. Accept both rather than making
        # the converter care which collector produced the dump.
        def _name(m):
            if "dir" in m:
                return os.path.basename(m["dir"])
            e = m.get("episode")
            return e if isinstance(e, str) else f"ep_{int(e):06d}"

        def _ok(m):
            return m["success"] if "success" in m else m.get("ok", True)

        ok = {_name(m): _ok(m) for m in man}
        why = {_name(m): m.get("why", "") for m in man}
        station = {_name(m): m.get("station") for m in man}

    # Spread the validation split across stations: all-one-bay would measure
    # nothing about the choice the policy actually has to make.
    holdout = set()
    if args.holdout:
        import random as _random
        rng = _random.Random(args.holdout_seed)
        usable = [d for d in ep_dirs if ok.get(d) is not False]
        by_station = {}
        for d in usable:
            by_station.setdefault(station.get(d), []).append(d)
        stations = sorted(by_station, key=lambda s: (s is None, s))
        i = 0
        while len(holdout) < args.holdout and stations:
            pool = by_station[stations[i % len(stations)]]
            pick = [d for d in pool if d not in holdout]
            if pick:
                holdout.add(rng.choice(pick))
            elif all(not [d for d in by_station[s] if d not in holdout]
                     for s in stations):
                break
            i += 1
        print(f"holding out {len(holdout)} episodes for validation: "
              + ", ".join(f"{d}(st{station.get(d)})" for d in sorted(holdout)))

    first_meta, first_scans = load_episode(os.path.join(args.raw, ep_dirs[0]))
    fps = int(round(first_meta["fps"]))
    # Cameras, from the dump's own metadata rather than a flag: which cameras
    # were recorded is a property of the recording, and a mismatch between a
    # flag and the files on disk would surface as a missing-frame truncation
    # halfway through an episode.
    cams = first_meta.get("cameras") or [{"name": "front", "dir": "frames"}]
    sizes = {}
    for c in cams:
        f0 = os.path.join(args.raw, ep_dirs[0], c["dir"], "000000.png")
        smp = cv2.imread(f0)
        if smp is None:
            sys.exit(f"could not read {f0} -- is the dump complete?")
        sizes[c["name"]] = smp.shape[:2]
    h, w = sizes[cams[0]["name"]]
    print("cameras: " + ", ".join(
        f"{c['name']} ({sizes[c['name']][1]}x{sizes[c['name']][0]})" for c in cams))

    state_names = list(first_meta["state_names"])
    if args.drop_abs_xy:
        state_names = [n for n in state_names if n not in ("x", "y")]
        print("dropping absolute x,y -> observation.state loses 2 dims")
    if args.drop_yaw:
        state_names = [n for n in state_names
                       if n not in ("cos_yaw", "sin_yaw")]
        print("dropping global heading -> observation.state loses 2 more dims")
    if args.drop_vel:
        state_names = [n for n in state_names if n not in ("v", "w")]
        print("dropping own velocity -> observation.state loses 2 more dims")
    if args.goal_in_state:
        state_names += ["goal_dx", "goal_dy"]
        print("appending the goal in robot frame -> observation.state gains 2 dims")
    if args.scan_in_state:
        k = args.scan_in_state
        if first_scans is None:
            sys.exit("--scan-in-state given but the dump has no scan.npy")
        width = int(first_scans.shape[1])
        if width % k:
            sys.exit(f"--scan-in-state {k} does not divide the scan width {width}")
        state_names += [f"scan{j:02d}" for j in range(k)]
        print(f"appending {k} min-pooled scan sectors "
              f"({width // k} beams each, clipped at {args.scan_clip} m) "
              f"-> observation.state is {len(state_names)}-dim")

    features = {}
    if args.make_crops:
        if len(cams) != 1:
            sys.exit("--make-crops expects exactly one recorded camera; "
                     f"this dump has {[c['name'] for c in cams]}")
        ch, cw = sizes[cams[0]["name"]]
        features["observation.images.front"] = {
            "dtype": "video", "shape": (ch, cw, 3),
            "names": ["height", "width", "channel"]}
        for nm in ("left", "right"):
            features[f"observation.images.{nm}"] = {
                "dtype": "video",
                "shape": (args.crop_size, args.crop_size, 3),
                "names": ["height", "width", "channel"]}
        print(f"crops on: front ({cw}x{ch}) + left/right "
              f"({args.crop_size}x{args.crop_size})")
    else:
        for c in cams:
            ch, cw = sizes[c["name"]]
            features[f"observation.images.{c['name']}"] = {
                "dtype": "video", "shape": (ch, cw, 3),
                "names": ["height", "width", "channel"],
            }
    action_names = list(first_meta["action_names"]) + (
        BOX_NAMES if args.box_action else [])
    features.update({
        "observation.state": {
            "dtype": "float32", "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32", "shape": (len(action_names),),
            "names": action_names,
        },
    })
    if first_scans is not None and not args.no_raw_scan:
        features["observation.scan"] = {
            "dtype": "float32", "shape": (int(first_scans.shape[1]),), "names": None,
        }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, root=args.root,
        robot_type=args.robot_type, features=features, use_videos=True,
    )
    ds_val = None
    if holdout:
        ds_val = LeRobotDataset.create(
            repo_id=args.repo_id + "_val", fps=fps, root=args.root + "_val",
            robot_type=args.robot_type, features=features, use_videos=True,
        )

    written = written_val = 0
    for name in ep_dirs:
        if ok.get(name) is False:
            # Not gated on --skip-failed: an episode the collector marked failed
            # is a recording of a robot on its side, and it looks like data.
            print(f"skip {name} (marked failed: {why.get(name, '?')})")
            continue
        ep_dir = os.path.join(args.raw, name)
        meta, scans = load_episode(ep_dir)
        steps = meta["steps"]
        instruction = meta["instruction"]

        # the pose Nav2 actually settled at == the goal it was given. Read
        # before any filtering, so dropped frames can never move the goal.
        goal_xy = None
        if args.goal_in_state:
            goal_xy = (steps[-1]["state"][0], steps[-1]["state"][1])

        is_val = name in holdout
        target = ds_val if is_val else ds

        if args.make_crops:
            # A fresh tracker per episode: ids only mean anything inside one
            # continuous sequence, and carrying an instance over would drag the
            # previous episode's tracks into the opening frames of this one.
            from person_crops import PersonCrops
            cropper = PersonCrops(size=args.crop_size)

        drop = spin_mask(steps, args.drop_spin)
        lead = lead_still_count(steps) if args.drop_lead_still else 0
        if lead:
            drop[:lead] = True
        # Frames where the two people are nearly in the same direction. The
        # instruction names one of them, but at a few degrees apart both names
        # demand the same turn, so the frame teaches nothing about reading the
        # instruction while still counting as a training example. Measured on
        # one episode: target at 29 deg and distractor at 38 deg for the whole
        # second half -- a 9 deg difference, which no policy could be graded on.
        # Dropping them concentrates the dataset on the frames where the two
        # instructions actually disagree.
        close = 0
        if args.min_separation > 0.0:
            thr = math.radians(args.min_separation)
            for _i, _s in enumerate(steps):
                _pp = _s.get("people") or {}
                if len(_pp) != 2:
                    continue
                (_b1, _b2) = [v[1] for v in _pp.values()]
                _sep = abs((_b1 - _b2 + math.pi) % (2 * math.pi) - math.pi)
                if _sep < thr:
                    drop[_i] = True
                    close += 1

        near = 0
        if args.drop_near > 0.0:
            for _i, _s in enumerate(steps):
                _d = _s.get("person_dist", -1.0)
                if 0.0 <= _d < args.drop_near:
                    drop[_i] = True
                    near += 1

        for i, step in enumerate(steps):
            if drop[i]:
                continue
            imgs = {}
            missing = False
            if args.make_crops:
                im = cv2.imread(os.path.join(ep_dir, cams[0]["dir"],
                                             f"{i:06d}.png"))
                if im is None:
                    print(f"  {name}: frame {i} missing, truncating here")
                    break
                marked, crops, _boxes = cropper(im)
                act_vec = np.asarray(step["action"], dtype=np.float32)
                if args.box_action:
                    act_vec = np.concatenate(
                        [act_vec, box_label(step, _boxes)])
                imgs["observation.images.front"] = marked[:, :, ::-1].copy()
                imgs["observation.images.left"] = crops[0][:, :, ::-1].copy()
                imgs["observation.images.right"] = crops[1][:, :, ::-1].copy()
                frame = {
                    **imgs,
                    "observation.state": build_state(
                        step, scans, i, args.scan_in_state, args.scan_clip,
                        goal_xy, args.drop_abs_xy, args.drop_yaw, args.drop_vel),
                    "action": act_vec,
                    "task": PER_STEP_TASK.get(step.get("target"), instruction),
                }
                if scans is not None and i < len(scans) and not args.no_raw_scan:
                    frame["observation.scan"] = scans[i].astype(np.float32)
                target.add_frame(frame)
                continue
            for c in cams:
                im = cv2.imread(os.path.join(ep_dir, c["dir"], f"{i:06d}.png"))
                if im is None:
                    print(f"  {name}: {c['name']} frame {i} missing, "
                          f"truncating episode here")
                    missing = True
                    break
                # cv2 reads BGR; LeRobot expects RGB
                imgs[f"observation.images.{c['name']}"] = im[:, :, ::-1].copy()
            if missing:
                break
            frame = {
                **imgs,
                "observation.state": build_state(
                    step, scans, i, args.scan_in_state, args.scan_clip, goal_xy,
                    args.drop_abs_xy, args.drop_yaw, args.drop_vel),
                "action": np.asarray(step["action"], dtype=np.float32),
                # Per-frame, when the dump recorded one. An episode whose
                # instruction switches part-way has two correct answers for the
                # same scene, and labelling every frame with the episode's
                # opening instruction would teach the opposite of the truth
                # after each switch. Falls back to the episode string for dumps
                # recorded before switching existed.
                "task": PER_STEP_TASK.get(step.get("target"), instruction),
            }
            if scans is not None and i < len(scans) and not args.no_raw_scan:
                frame["observation.scan"] = scans[i].astype(np.float32)
            target.add_frame(frame)

        target.save_episode()
        if is_val:
            written_val += 1
        else:
            written += 1
        dropped = int(drop.sum())
        bits = []
        if lead:
            bits.append(f"-{lead} lead-still")
        if near:
            bits.append(f"-{near} too-near")
        if close:
            bits.append(f"-{close} same-bearing")
        if dropped - lead - near - close > 0:
            bits.append(f"-{dropped - lead - near - close} spin")
        if is_val:
            bits.append("VAL")
        note = f" ({', '.join(bits)})" if bits else ""
        sw = sum(1 for i in range(1, len(steps))
                 if steps[i].get("target") != steps[i - 1].get("target"))
        seen = sorted({s.get("target") for s in steps if s.get("target")})
        lbl = f"{'/'.join(seen)} ({sw} switches)" if sw else f'"{instruction}"'
        rate = f", tracked {100 * cropper.rate():.0f}%" if args.make_crops else ""
        print(f"{name}: {len(steps) - dropped} steps{note}{rate} -- {lbl}")

    print(f"\nwrote {written} episodes to {args.root} (repo_id={args.repo_id})")
    if ds_val is not None:
        # Which raw episode each val episode came from. The val dataset
        # renumbers from 0, so without this the only way back to the source
        # episode -- needed for the pose the dataset deliberately drops -- is to
        # re-run the selection and hope the seed and manifest still match.
        with open(os.path.join(args.root + "_val", "source_episodes.json"), "w") as f:
            json.dump(sorted(holdout), f, indent=1)
        print(f"wrote {written_val} validation episodes to {args.root}_val "
              f"(repo_id={args.repo_id}_val)")


if __name__ == "__main__":
    main()
