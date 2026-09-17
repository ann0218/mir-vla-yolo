#!/usr/bin/env python3
"""Serve a trained LeRobot policy (SmolVLA or ACT) over localhost HTTP.

Why a server at all: the policy needs torch + lerobot, which live in the HOST
conda env (env_lerobot); ROS2 lives in the my_isaac_sim container and has no
torch. Installing either side into the other is the thing the project README
warns against. The container uses host networking, so plain localhost HTTP
between the two is the cheapest bridge that keeps both environments intact.

Run on the host:

    /home/itri/anaconda3/envs/env_lerobot/bin/python policy_server.py \
        --ckpt /home/itri/mir_isaac_test/vla/train/act_red/checkpoints/last/pretrained_model

The policy family (smolvla / act / ...) is read from the checkpoint's own
config.json, so the same server drives either. ACT has no language input, so
the `task` field the runner still sends is simply dropped for it.

Then `policy_runner.py` inside the container drives the robot with it.
"""
import argparse
import base64
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors

STATE = {}


def state_dim(policy, pre):
    """How wide an observation.state this policy actually consumes.

    Not simply config.input_features, because finetuning from a pretrained
    checkpoint does not always rewrite it: smolvla_base declares a 6-dim state
    and lerobot-train left that in place while training happily on our 26-dim
    one (SmolVLA pads to max_state_dim internally, so nothing complains). The
    stale 6 then reaches the runner's /info check, which reports a mismatch on
    every run -- a safety check that cries wolf is worse than none, because it
    hides the real mismatch it exists to catch.

    The normaliser's statistics are computed from the dataset the model was
    trained on, so their width is the honest answer. Fall back to the config
    when no state statistics are present.
    """
    for st in getattr(pre, "steps", []):
        stats = getattr(st, "stats", None) or {}
        s = stats.get("observation.state") or {}
        for k in ("mean", "std", "min", "max"):
            v = s.get(k)
            if v is not None and getattr(v, "shape", None):
                return int(v.shape[-1])
    return policy.config.input_features["observation.state"].shape[0]


def build_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep stdout for our own logging
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # So `curl host:port/info` answers without a POST body. A stale
            # server left over from an earlier run keeps serving happily and
            # says nothing about which weights it holds -- this is how you find
            # out before trusting a closed-loop number.
            if self.path == "/info":
                self._send(200, STATE["info"])
            else:
                self._send(404, {"error": "GET /info"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")

            if self.path == "/reset":
                # SmolVLA emits action chunks and pops them from an internal
                # queue; without this the first actions of a new run are
                # leftovers aimed at the previous goal.
                STATE["policy"].reset()
                STATE["n"] = 0
                if STATE.get("cropper") is not None:
                    # New run, new tracker. Track ids only mean anything within
                    # one continuous sequence.
                    from person_crops import PersonCrops
                    old = STATE["cropper"]
                    STATE["cropper"] = PersonCrops(size=old.size,
                                                   device=old.device)
                self._send(200, {"ok": True})
                return

            if self.path == "/info":
                self._send(200, STATE["info"])
                return

            if self.path != "/act":
                self._send(404, {"error": "use /act, /reset or /info"})
                return

            t0 = time.time()

            def _decode(b64):
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                                   cv2.IMREAD_COLOR)
                # to_lerobot.py fed the policy RGB (it flipped cv2's BGR); match
                # it here or the policy sees colours it was never trained on.
                rgb = bgr[:, :, ::-1].copy()
                t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
                return t.unsqueeze(0).to(STATE["device"])

            batch = {
                "observation.state": torch.tensor(
                    req["state"], dtype=torch.float32).unsqueeze(0).to(STATE["device"]),
            }
            # Multi-camera policies take {"images": {key: b64}}; the older
            # single-camera clients send {"image": b64}. Accept both, and check
            # the multi-camera form actually names the keys this policy wants --
            # an unrecognised key is silently ignored by the processor, so a
            # typo would leave the model running on whatever cameras did match
            # and no error would say so.
            if STATE.get("cropper") is not None:
                # One image in, three out. The runner sends only the platform
                # camera; the tracker and the crops are built here, with the
                # same person_crops.py the dataset was converted with -- if the
                # two drifted apart the policy would be shown inputs it was
                # never trained on, and nothing would say so.
                # Take the platform camera BY NAME. This used to be "the first
                # image in the dict", and the runner fills its dict in the order
                # topic messages happen to arrive -- 5 of 5 fresh runner
                # processes put a corner camera first. All three are 640x480, so
                # nothing downstream noticed: the policy tracked from a view of
                # the whole room, a different camera each episode, and a full
                # paired batch had to be thrown away.
                if "image" in req:
                    b64 = req["image"]
                elif "observation.images.camera1" in req.get("images", {}):
                    b64 = req["images"]["observation.images.camera1"]
                else:
                    self._send(400, {"error": "crops mode needs the platform "
                                              "camera as 'image' or "
                                              "images['observation.images.camera1']",
                                     "received": sorted(req.get("images", {}))})
                    return
                bgr = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                                   cv2.IMREAD_COLOR)
                marked, crops, boxes = STATE["cropper"](bgr)
                for key, img in (("observation.images.camera1", marked),
                                 ("observation.images.camera2", crops[0]),
                                 ("observation.images.camera3", crops[1])):
                    t = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)
                    batch[key] = (t.float() / 255.0).unsqueeze(0).to(STATE["device"])
                STATE["boxes"] = len(boxes)
                # Box corners, so the runner can stop for a person too close
                # ahead. The lidar cannot do it: the characters have no
                # collision geometry, so the scan passes straight through them.
                STATE["box_list"] = [[round(float(c), 1) for c in b["xyxy"]]
                                     for b in boxes]
                # How often does the robot actually have both people in frame?
                # A blank crop is what the policy gets when YOLO sees nobody,
                # and the expert never put it in that state, so track it.
                h = STATE.setdefault("nhist", {0: 0, 1: 0, 2: 0})
                h[min(len(boxes), 2)] = h.get(min(len(boxes), 2), 0) + 1
                n = sum(h.values())
                if n % 100 == 0:
                    print(f"  detections over {n} frames: "
                          f"0 people {h[0]/n:.0%}, 1 person {h[1]/n:.0%}, "
                          f"2 people {h[2]/n:.0%}", flush=True)
            elif "images" in req:
                want = set(STATE["image_keys"])
                got = set(req["images"])
                if got != want:
                    self._send(400, {"error": "image keys do not match the "
                                              "policy",
                                     "expected": sorted(want),
                                     "received": sorted(got)})
                    return
                for k, b64 in req["images"].items():
                    batch[k] = _decode(b64)
            else:
                batch[STATE["image_key"]] = _decode(req["image"])
            # SmolVLA is language-conditioned; ACT is not and its processor has
            # no tokenizer step, so passing a task string would just be ignored.
            if STATE["uses_task"]:
                batch["task"] = [req["task"]]
            with torch.no_grad():
                act = STATE["post"](STATE["policy"].select_action(STATE["pre"](batch)))
            full = act.squeeze(0).tolist()
            # Only the first two dims drive the robot. A policy trained with
            # --box-action also predicts where it thinks the named person is;
            # those dims are passed through for logging and drawing, never
            # for control.
            v, w = full[:2]

            STATE["n"] = STATE.get("n", 0) + 1
            if STATE["n"] % 20 == 1:
                print(f"  [{STATE['n']:5d}] v={v:+.3f} w={w:+.3f} "
                      f"({1000 * (time.time() - t0):.0f} ms)", flush=True)
            # How many people the crop pipeline actually found. The caller needs
            # this to tell "the policy chose to turn away" from "the policy was
            # handed two blank crops", which look identical in the action alone.
            self._send(200, {"action": [v, w], "people": STATE.get("boxes", -1),
                             "boxes": STATE.get("box_list", []),
                             "action_full": [round(x, 4) for x in full]})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crops", action="store_true",
                    help="build the three streams here from one incoming image: "
                         "the frame with people boxed, plus the left and right "
                         "person cropped to full size. Use for a policy trained "
                         "on a dataset converted with to_lerobot.py "
                         "--make-crops; the runner then sends only the platform "
                         "camera.")
    ap.add_argument("--crop-size", type=int, default=512)
    ap.add_argument("--temporal-ensemble", type=float, default=None, metavar="COEFF",
                    help="enable ACT temporal ensembling with this coefficient "
                         "(the ACT paper uses 0.01; positive weights OLDER "
                         "predictions more, negative weights newer ones, 0 is a "
                         "flat average). Needs --n-action-steps 1. Inference "
                         "only -- no retraining. Aimed at the final approach, "
                         "where the measured failure is a ~0.23 m lateral offset "
                         "against a 0.10 m tolerance.")
    ap.add_argument("--n-action-steps", type=int, default=0,
                    help="override the policy's action-chunk horizon at inference. "
                         "SmolVLA plans a chunk and then runs it open-loop; at the "
                         "trained default of 10 that is 1 s (~0.5 m) of committed "
                         "motion, which is the same scale as the lateral correction "
                         "the robot fails to make. 1 = replan every tick.")
    args = ap.parse_args()

    with open(os.path.join(args.ckpt, "config.json")) as f:
        ptype = json.load(f)["type"]
    print(f"loading {args.ckpt}  (policy type: {ptype})")
    policy = get_policy_class(ptype).from_pretrained(args.ckpt).eval().to(args.device)
    if args.n_action_steps:
        policy.config.n_action_steps = args.n_action_steps
        print(f"n_action_steps overridden -> {args.n_action_steps}")

    if args.temporal_ensemble is not None:
        # ACT's own temporal ensembling (Algorithm 2 of the ACT paper): predict a
        # full chunk every tick and exponentially weight every past prediction
        # that covers the current step. It is pure inference -- nothing about the
        # trained weights changes -- but it is built in __init__ from the config,
        # so enabling it on a loaded policy means constructing the ensembler here
        # as well as setting the flag. reset() branches on the same flag, so both
        # must be in place before the first /reset.
        if ptype != "act":
            raise SystemExit("--temporal-ensemble is an ACT feature")
        if policy.config.n_action_steps != 1:
            raise SystemExit("--temporal-ensemble needs --n-action-steps 1: the "
                             "ensemble is formed by re-predicting every tick")
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble
        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble, policy.config.chunk_size)
        policy.reset()
        print(f"temporal ensembling on, coeff {args.temporal_ensemble} "
              f"over a {policy.config.chunk_size}-step chunk")
    pre, post = make_pre_post_processors(policy.config, args.ckpt)
    exp = state_dim(policy, pre)
    # the runner always sends observation.images.<something>; use whatever key
    # this policy was trained on (front for ACT, camera1 for the renamed SmolVLA)
    image_key = next(iter(policy.config.image_features))
    ck = os.path.abspath(args.ckpt)
    # checkpoints live at .../checkpoints/<step>/pretrained_model
    step = os.path.basename(os.path.dirname(ck))
    run = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ck))))
    info = {"policy": ptype, "run": run, "step": step, "ckpt": ck,
            "state_dim": exp, "image_key": image_key,
            "image_keys": list(policy.config.image_features),
            "n_action_steps": policy.config.n_action_steps,
            "temporal_ensemble": getattr(policy.config, "temporal_ensemble_coeff", None),
            "started": time.strftime("%F %T")}
    cropper = None
    if args.crops:
        from person_crops import PersonCrops
        cropper = PersonCrops(size=args.crop_size, device=args.device)
        print(f"crops on: one image in, three out ({args.crop_size}px)")
    STATE.update(cropper=cropper)
    STATE.update(policy=policy, pre=pre, post=post, device=args.device, n=0,
                 image_key=image_key, image_keys=list(policy.config.image_features),
                 uses_task=(ptype == "smolvla"), info=info)

    print(f"policy expects a {exp}-dim observation.state, image key {image_key}")
    print(f"serving {run}/{step}  (GET /info to confirm from the client side)")
    print(f"serving on http://{args.host}:{args.port}  (POST /act, POST /reset)")
    HTTPServer((args.host, args.port), build_handler()).serve_forever()


if __name__ == "__main__":
    main()
