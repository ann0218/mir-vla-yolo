#!/usr/bin/env python3
"""Ask a --box-action policy where it thinks the named person is, offline.

Held-out frames only, one forward pass each. This is the cheap check that has to
pass before a 1.5 h closed-loop batch is worth running: if the box dimensions did
not learn anything, the closed loop cannot show it either.

The observation is built exactly as eval_2p_instruction.py builds it -- the same
three streams from person_crops.py (marked frame, left crop, right crop), the
same 24-dim state, the same per-frame instruction -- because a policy shown
inputs it was not trained on scores nothing meaningful.

Ground truth for the box is to_lerobot.box_label(), i.e. the same labels the
model was trained against, recomputed here from the raw dump.

    python3 probe_box.py --ckpt train/smolvla_mswall_box/checkpoints/030000/pretrained_model
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import to_lerobot as T  # noqa: E402  (box_label, build_state)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402

NEAR_PX = 60.0
HALF_FOV = math.radians(37.6)
INSTR = {"purple": "follow the person in purple", "yellow": "follow the person in yellow"}


def state_width(pre):
    """Width the checkpoint was trained on, from the normaliser statistics.

    Not from config.input_features: finetuning leaves that at the base model's 6
    while training happily on our 24, so reading the config scores the model on
    an input layout it never saw.
    """
    for step in getattr(pre, "steps", []):
        stats = getattr(step, "stats", None) or {}
        s = stats.get("observation.state") or {}
        for k in ("mean", "std", "min", "max"):
            v = s.get(k)
            if v is not None and getattr(v, "shape", None):
                return int(v.shape[-1])
    return 24


def held_out(log_path):
    if not os.path.exists(log_path):
        return []
    for line in open(log_path, errors="ignore"):
        if "holding out" in line:
            return [p.split("(")[0].strip() for p in line.split(":", 1)[1].split(",")]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="train/smolvla_mswall_box/checkpoints/030000/pretrained_model")
    ap.add_argument("--raw", default="raw_msw_all")
    ap.add_argument("--regen-log", default="results_2p/regen_box.log")
    ap.add_argument("--episodes", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--samples", type=int, default=2,
                    help="draws per frame; the action head is stochastic")
    ap.add_argument("--only-distractor-centred", action="store_true",
                    help="score only frames where the OTHER person is nearer the\n"
                         "image centre than the named one. The expert keeps the\n"
                         "named person centred, so on ordinary frames a policy\n"
                         "that just boxes whoever is central scores the same as\n"
                         "one that reads the instruction; these frames separate\n"
                         "them.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    eps = a.episodes or held_out(a.regen_log)
    if not eps:
        sys.exit("no held-out episodes found; pass --episodes")
    print(f"held-out episodes: {', '.join(eps)}")

    ptype = json.load(open(os.path.join(a.ckpt, "config.json")))["type"]
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    policy = get_policy_class(ptype).from_pretrained(a.ckpt).eval().to(a.device)
    pre, post = make_pre_post_processors(cfg, a.ckpt)
    width = state_width(pre)
    print(f"policy {ptype} from {a.ckpt}; state width {width}")

    from person_crops import PersonCrops
    cropper = PersonCrops(device=a.device)

    n = 0
    say_vis = really_vis = agree = 0
    errs, on_named, on_other, on_neither = [], 0, 0, 0
    vs, ws = [], []
    for name in eps:
        ep = os.path.join(a.raw, name)
        meta_path = os.path.join(ep, "episode.json")
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path))
        scans_path = os.path.join(ep, "scan.npy")
        scans = np.load(scans_path) if os.path.exists(scans_path) else None
        cams = meta.get("cameras") or [{"name": "front", "dir": "frames"}]
        for i in range(0, len(meta["steps"]), a.stride):
            if n >= a.max_frames:
                break
            step = meta["steps"][i]
            im = cv2.imread(os.path.join(ep, cams[0]["dir"], f"{i:06d}.png"))
            if im is None:
                continue
            marked, crops, boxes = cropper(im)
            label = T.box_label(step, boxes)
            if a.only_distractor_centred:
                # BOTH in view, and the wrong one nearer the centre. Without
                # the "both in view" half this selects frames where the named
                # person is simply out of frame -- measured, only 11 of 76
                # such frames had anyone to box at all, which answers nothing.
                _pp = step.get("people") or {}
                _t = step.get("target")
                _o = [k for k in _pp if k != _t]
                if not _o or _t not in _pp:
                    continue
                _bt, _bo = abs(_pp[_t][1]), abs(_pp[_o[0]][1])
                if _bt > HALF_FOV or _bo > HALF_FOV or _bo >= _bt:
                    continue
            batch = {}
            for key, img in (("observation.images.camera1", marked),
                             ("observation.images.camera2", crops[0]),
                             ("observation.images.camera3", crops[1])):
                t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
                batch[key] = (t.float() / 255.0).unsqueeze(0).to(a.device)
            state = T.build_state(step, scans, i, 24, 10.0, None, True, True, width == 24)
            batch["observation.state"] = torch.from_numpy(
                np.asarray(state, dtype=np.float32)).unsqueeze(0).to(a.device)
            batch["task"] = [INSTR.get(step.get("target"), meta.get("instruction", ""))]

            draws = []
            for k in range(a.samples):
                torch.manual_seed(1000 + k)
                policy.reset()
                with torch.no_grad():
                    draws.append(post(policy.select_action(pre(batch))).squeeze(0).tolist())
            act = np.mean(np.asarray(draws, dtype=np.float32), axis=0)
            n += 1
            vs.append(float(act[0]))
            ws.append(float(act[1]))

            said = act[6] > 0.5
            real = label[4] > 0.5
            say_vis += bool(said)
            really_vis += bool(real)
            agree += bool(said == real)
            if not (said and real):
                continue
            cx_pred = (act[2] + act[4]) / 2.0
            cx_true = (label[0] + label[2]) / 2.0
            errs.append(abs(cx_pred - cx_true))
            # where does it point: the named person, the other one, or neither
            pp = step.get("people") or {}
            tgt = step.get("target")
            others = [k for k in pp if k != tgt]
            uo = T._box_u(pp[others[0]][0], pp[others[0]][1]) if others else None
            near_t = abs(cx_pred - cx_true) < NEAR_PX
            near_o = uo is not None and abs(cx_pred - uo) < NEAR_PX
            on_named += near_t and not near_o
            on_other += near_o and not near_t
            on_neither += not near_t and not near_o

    if not n:
        sys.exit("no frames scored")
    print(f"\nframes scored: {n}")
    print(f"visibility: policy says visible {100*say_vis/n:.1f} %, truth {100*really_vis/n:.1f} %, "
          f"agreement {100*agree/n:.1f} %")
    if errs:
        e = sorted(errs)
        print(f"box centre error on {len(e)} frames where both say visible: "
              f"median {st.median(e):.0f} px, p90 {e[int(.9*len(e))-1]:.0f} px, "
              f"within {NEAR_PX:.0f} px {100*sum(x < NEAR_PX for x in e)/len(e):.0f} %")
        print(f"box points at: named {on_named}, other {on_other}, neither {on_neither}")
    print(f"control dims still sane: v mean {st.mean(vs):+.2f} (range {min(vs):+.2f}..{max(vs):+.2f}), "
          f"w mean {st.mean(ws):+.2f}")


if __name__ == "__main__":
    main()
