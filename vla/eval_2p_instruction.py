#!/usr/bin/env python3
"""Does the policy actually READ its instruction?

This is the test the whole two-person scene was built to make possible, and it
is deliberately not a following-error metric. Following error can look fine
while the instruction is being ignored -- that is exactly what happened on the
single-person task, where the trained VLA scored well and, when probed, turned
out to move its output by 0.06 whether it was told "follow the person", "stop
immediately and do not move", or "banana telescope kettle", against a 0.34
spread across frames.

So the measurement here is a swap. The SAME frame is shown twice, once under
each instruction, and the question is whether the commanded turn follows the
person that was named:

    correct  =  sign(w_commanded) == sign(bearing to the named person)

Only frames where the two people are on OPPOSITE sides are scored. Where both
are to the left, "turn left" is right under either instruction and the frame
cannot tell an obedient policy from a deaf one -- including such frames would
inflate the score towards 50 % agreement for free.

Three numbers come out of it:

  purple / yellow accuracy   how often the turn goes towards the named person.
                             50 % is chance. A policy ignoring the text scores
                             ~50 % on one instruction and ~50 % on the other,
                             because it always turns the same way.
  swap rate                  how often changing ONLY the instruction flips the
                             sign of the commanded turn. This is the direct
                             measure of language sensitivity, and it needs no
                             ground truth at all.

Scored on the raw dump rather than the LeRobot dataset: the per-person bearings
are recorded there, and the conversion drops frames, so frame i of the dataset
is not frame i of the episode.

    python eval_2p_instruction.py --checkpoints 010000 020000 030000
"""
import argparse
import glob
import json
import math
import os

# A LoRA checkpoint names its base ("lerobot/smolvla_base") and loading it goes
# through the hub. The weights are already cached locally; without this the run
# still reaches out to check for updates and can sit there when the network is
# slow, which looks exactly like a hung evaluation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from to_lerobot import build_state

HERE = os.path.dirname(os.path.abspath(__file__))

INSTRUCTIONS = {
    "purple": "follow the person in purple",
    "yellow": "follow the person in yellow",
}
# Dataset key -> the key the finetuned policy asks for. Must match the
# --rename_map used at training, or the images arrive under names the policy
# ignores and it runs on blank inputs without complaining.
# This dataset records ONE camera and derives three at conversion time, so the
# evaluator has to derive them too -- with the same person_crops.py, or the
# model is scored on inputs it never saw.
CAM_TO_POLICY = {
    "front": "observation.images.camera1",
}
MAKE_CROPS = True


def state_width(pre):
    """How wide an observation.state this checkpoint expects.

    From the normaliser's statistics, which are computed from the dataset the
    model was trained on, so they cannot disagree with it. config.input_features
    is not reliable: finetuning from a pretrained checkpoint leaves the
    pretrained value in place.
    """
    for st in getattr(pre, "steps", []):
        stats = getattr(st, "stats", None) or {}
        sd = stats.get("observation.state") or {}
        for k in ("mean", "std", "min", "max"):
            v = sd.get(k)
            if v is not None and getattr(v, "shape", None):
                return int(v.shape[-1])
    raise SystemExit("cannot determine the state width from the checkpoint")


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, "episode.json")) as f:
        meta = json.load(f)
    scan_path = os.path.join(ep_dir, "scan.npy")
    scans = np.load(scan_path) if os.path.exists(scan_path) else None
    return meta, scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="train/smolvla_msw")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--raw", default=os.path.join(HERE, "raw_msw"))
    ap.add_argument("--val-root", default=os.path.join(HERE, "lerobot_msw_val"))
    ap.add_argument("--episodes", nargs="*", default=[],
                    help="raw episode names to score. Defaults to the held-out "
                         "ones the conversion set aside.")
    ap.add_argument("--min-sep", type=float, default=40.0,
                    help="degrees. Frames where the two people are closer "
                         "together than this are skipped even if they are on "
                         "opposite sides -- near the boundary the correct turn "
                         "is a few degrees either way and the sign is noise.")
    ap.add_argument("--min-off-centre", type=float, default=0.0,
                    help="degrees. Only score frames where BOTH people are at "
                         "least this far off the robot's centreline, i.e. it is "
                         "not already locked onto either. Isolates the frames "
                         "where the instruction is the only thing that can say "
                         "which way to turn.")
    ap.add_argument("--n", type=int, default=200, help="frames per episode cap")
    ap.add_argument("--samples", type=int, default=4,
                    help="draws per (frame, instruction), averaged. The action "
                         "head is stochastic; one draw each makes the swap rate "
                         "swing by more than the effect being measured.")
    ap.add_argument("--balance", action="store_true",
                    help="equalise purple-left and purple-right in the scored "
                         "set, so colour and side are decorrelated.")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth candidate frame, to spread the sample "
                         "over the episode rather than taking one run.")
    ap.add_argument("--control-task", default="",
                    help="a third, meaningless instruction. Reports how far it "
                         "moves the output compared with the real swap; if the "
                         "two are similar, nothing semantic is happening.")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    eps = a.episodes
    if not eps:
        src = os.path.join(a.val_root, "source_episodes.json")
        if not os.path.exists(src):
            raise SystemExit(f"no {src}; pass --episodes explicitly")
        eps = json.load(open(src))
    print(f"scoring {len(eps)} held-out episodes: {', '.join(eps)}")

    # Gather the frames worth scoring once, so every checkpoint sees the same
    # set and the comparison between them is not confounded by sampling.
    items = []
    for name in eps:
        ep_dir = os.path.join(a.raw, name)
        meta, scans = load_episode(ep_dir)
        cams = meta.get("cameras") or [{"name": "front", "dir": "frames"}]
        kept = 0
        for i, step in enumerate(meta["steps"]):
            if kept >= a.n:
                break
            # Stride, so the frames are spread across the episode instead of
            # being one contiguous run. At 10 Hz a run of 40 is four seconds of
            # nearly unchanged geometry -- 142 such frames were really only four
            # distinct configurations, which inflated the apparent sample size
            # roughly thirty-fold.
            if i % a.stride:
                continue
            pp = step.get("people") or {}
            if len(pp) != 2:
                continue
            bp, by = pp["purple"][1], pp["yellow"][1]
            # opposite sides, and far enough apart to have an unambiguous answer
            if bp * by >= 0:
                continue
            sep = abs((bp - by + math.pi) % (2 * math.pi) - math.pi)
            if math.degrees(sep) < a.min_sep:
                continue
            # Optionally require that the robot is not already locked onto
            # either person. Once the expert has acquired its target, the target
            # IS the person in the middle of the frame, so "keep the centred
            # person centred" reproduces the demonstration perfectly without
            # ever reading the instruction. Frames where someone is already
            # centred therefore cannot distinguish a policy that obeys from one
            # that centre-tracks -- and they are the majority of the dataset.
            if a.min_off_centre > 0.0:
                off = math.degrees(min(abs(bp), abs(by)))
                if off < a.min_off_centre:
                    continue
            items.append({"ep": ep_dir, "name": name, "i": i, "cams": cams,
                          "step": step, "scans": scans, "bp": bp, "by": by})
            kept += 1
    # Balance purple-left against purple-right. Without this the colour and the
    # side are correlated in the scored set, and a policy that merely maps the
    # word "purple" to one turn direction -- never looking at the image -- scores
    # as if it had understood: measured on the earlier set, 71.8 % purple-left
    # gave such a policy 143.7 %, above the 114.8 % actually observed, so the
    # observed number could not be attributed to colour at all.
    if a.balance:
        L = [it for it in items if it["bp"] > 0]
        R = [it for it in items if it["bp"] <= 0]
        k = min(len(L), len(R))
        rng = np.random.default_rng(0)
        if k == 0:
            raise SystemExit("cannot balance: every scored frame has purple on "
                             "the same side")
        items = ([L[i] for i in rng.choice(len(L), k, replace=False)] +
                 [R[i] for i in rng.choice(len(R), k, replace=False)])
        print(f"balanced to {k} + {k}")
    pl = sum(1 for it in items if it["bp"] > 0)
    neps = len({it["name"] for it in items})
    print(f"{len(items)} frames from {neps} episodes, purple-left "
          f"{100 * pl / max(len(items), 1):.0f}% "
          f"(a direction-only bias would score {200 * max(pl, len(items)-pl) / max(len(items),1):.0f}%)")
    if not items:
        raise SystemExit("nothing to score -- the two people were never on "
                         "opposite sides, so no frame can distinguish the "
                         "instructions")

    print(f"\n{'step':<8} {'purple acc':>11} {'yellow acc':>11} {'swap rate':>10} "
          f"{'|dw| mean':>10}")
    for ck_name in a.checkpoints:
        ck = os.path.join(a.model_dir, f"checkpoints/{ck_name}/pretrained_model")
        if not os.path.isdir(ck):
            print(f"{ck_name:<8} (missing)")
            continue
        ptype = json.load(open(os.path.join(ck, "config.json")))["type"]
        adapter = os.path.join(ck, "adapter_config.json")
        if os.path.exists(adapter):
            # A LoRA run saves only the adapter (19 MB), not a full
            # model.safetensors. Load the base the adapter names, then apply it
            # -- reading the base from adapter_config rather than assuming
            # smolvla_base, so an adapter trained on something else cannot be
            # silently mounted on the wrong weights.
            from peft import PeftModel
            from lerobot.configs.policies import PreTrainedConfig
            base = json.load(open(adapter))["base_model_name_or_path"]
            # Build from the base weights but with THIS run's config. The base
            # declares a 6-dim action (what smolvla_base was pretrained with);
            # this run's is 2. The action head itself is max_action_dim wide
            # either way and is restored in full from the adapter, but the
            # policy slices its output to config.output_features -- so leaving
            # the base config in place yields a 6-wide action against 2-wide
            # normaliser statistics, and inference dies in the unnormaliser.
            ckcfg = PreTrainedConfig.from_pretrained(ck)
            policy = get_policy_class(ptype).from_pretrained(base, config=ckcfg)
            policy = PeftModel.from_pretrained(policy, ck).eval().to(a.device)
            # The CHECKPOINT's config, not the base's. The base declares a
            # 6-dim action (what smolvla_base was pretrained with) while this
            # run's is 2, and the normaliser statistics saved next to the
            # adapter are 2-dim: pairing them with the base config fails on a
            # shape mismatch inside the unnormaliser.
            cfg = ckcfg
            print(f"  {ck_name}: LoRA adapter on {base} "
                  f"(action {cfg.output_features['action'].shape})")
        else:
            policy = get_policy_class(ptype).from_pretrained(ck).eval().to(a.device)
            cfg = policy.config
        pre, post = make_pre_post_processors(cfg, ck)
        # Which state layout this checkpoint was trained on, read from the
        # normaliser rather than assumed. Hard-coding it here once cost a run:
        # the conversion dropped the robot's own [v, w] -- 24 dims instead of
        # 26 -- and an evaluator still building 26 either crashes, as this one
        # did, or silently scores a model on inputs it never saw.
        want = state_width(pre)
        drop_vel = want == 24
        if want not in (24, 26):
            raise SystemExit(f"{ck_name}: unexpected state width {want}")

        from person_crops import PersonCrops
        cropper = PersonCrops()
        hits = {"purple": 0, "yellow": 0}
        swaps = 0
        good_swaps = 0
        ctrl = []
        # Per episode, because the scored frames are contiguous runs from a
        # handful of episodes and the geometry barely changes inside one. If the
        # correct swaps all land in the episodes where purple happens to be on
        # the left, the policy is following a fixed instruction-to-direction
        # bias, not the colour -- and the aggregate cannot tell the two apart.
        by_ep = {}
        dws = []
        for it in items:
            imgs = {}
            p = os.path.join(it["ep"], it["cams"][0]["dir"], f"{it['i']:06d}.png")
            im = cv2.imread(p)
            if im is None:
                continue
            marked, crops, _ = cropper(im)
            for key, img in (("observation.images.camera1", marked),
                             ("observation.images.camera2", crops[0]),
                             ("observation.images.camera3", crops[1])):
                t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
                imgs[key] = (t.float() / 255.0).unsqueeze(0).to(a.device)
            state = build_state(it["step"], it["scans"], it["i"], 24, 10.0,
                                None, True, True, drop_vel)
            st = torch.from_numpy(np.asarray(state, dtype=np.float32)) \
                .unsqueeze(0).to(a.device)

            # PAIRED, and averaged. SmolVLA's action head samples, so the same
            # frame and the same instruction do not give the same number twice:
            # measured, one checkpoint scored a 6.7 % swap rate on one pass and
            # 1.7 % on the next, which is the whole effect being looked for.
            #
            # Seeding identically before each instruction makes the two draws
            # use the same noise, so the difference between them is the
            # instruction and nothing else; averaging over --samples then damps
            # what is left.
            probe = dict(INSTRUCTIONS)
            if a.control_task:
                probe["control"] = a.control_task
            w = {}
            for who, instr in probe.items():
                vals = []
                for k in range(a.samples):
                    torch.manual_seed(1000 + k)
                    policy.reset()
                    batch = dict(imgs)
                    batch["observation.state"] = st
                    batch["task"] = [instr]
                    with torch.no_grad():
                        act = post(policy.select_action(pre(batch)))
                    vals.append(float(act.squeeze(0)[1]))
                w[who] = float(np.mean(vals))

            if np.sign(w["purple"]) == np.sign(it["bp"]):
                hits["purple"] += 1
            if np.sign(w["yellow"]) == np.sign(it["by"]):
                hits["yellow"] += 1
            e = by_ep.setdefault(it["name"], {"n": 0, "pl": 0, "sw": 0, "ok": 0})
            e["n"] += 1
            e["pl"] += 1 if it["bp"] > 0 else 0
            if np.sign(w["purple"]) != np.sign(w["yellow"]):
                swaps += 1
                e["sw"] += 1
                # A swap is only worth anything if it goes the RIGHT way. The
                # swap rate alone counts a policy that reverses at random as
                # language-sensitive, and separating the two is what says
                # whether the instruction is being understood or merely felt.
                if (np.sign(w["purple"]) == np.sign(it["bp"])
                        and np.sign(w["yellow"]) == np.sign(it["by"])):
                    good_swaps += 1
                    e["ok"] += 1
            dws.append(abs(w["purple"] - w["yellow"]))
            if "control" in w:
                ctrl.append(abs(w["purple"] - w["control"]))

        n = max(len(dws), 1)
        print(f"  {'episode':<12} {'n':>4} {'紫在左':>7} {'swap':>5} {'對':>4} {'錯':>4}")
        for nm, e in sorted(by_ep.items()):
            print(f"  {nm:<12} {e['n']:>4} {100*e['pl']/max(e['n'],1):>6.0f}% "
                  f"{e['sw']:>5} {e['ok']:>4} {e['sw']-e['ok']:>4}")
        if ctrl:
            # If swapping purple for a nonsense string moves the output as much
            # as swapping it for yellow, the model is reacting to the tokens
            # changing, not to what they mean.
            print(f"  |dw| purple vs yellow  {np.mean(dws):.4f}")
            print(f"  |dw| purple vs control {np.mean(ctrl):.4f}  "
                  f"({a.control_task!r})")
        bad = swaps - good_swaps
        print(f"{ck_name:<8} {100*hits['purple']/n:10.1f}% "
              f"{100*hits['yellow']/n:10.1f}% {100*swaps/n:9.1f}% "
              f"{np.mean(dws):10.4f}  |  swap 對 {100*good_swaps/n:5.1f}% "
              f"錯 {100*bad/n:5.1f}%")

    print("\n50% accuracy is chance. A policy that ignores the instruction "
          "turns the same way whatever it is told, so it scores about 50% on "
          "each and a swap rate near 0.")


if __name__ == "__main__":
    main()
