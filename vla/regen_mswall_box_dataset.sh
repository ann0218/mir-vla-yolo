#!/usr/bin/env bash
# Convert the moving two-colour dump, generating the tracker's three streams.
#
# One camera was recorded; three come out. The reason is resolution: measured on
# the earlier build, a person occupied 1-7 of the 1024 patches SmolVLM2 splits a
# 512x512 input into, and the swap test never rose above 23 % before overfitting
# away. A crop blown up to full size carries 70,000-120,000 pixels of one colour
# and none of the other, so choosing between two candidates stops being a
# resolution problem and becomes what it should have been all along: reading the
# instruction.
#
#   --drop-vel     the expert's turn is smooth, so its own w predicts the next
#                  action with r = 0.895 and a policy handed it just continues
#                  turning. Leaving it in cost a full training run.
#   --min-separation
#                  frames where the two people are within 20 deg of each other
#                  are dropped: there both instructions call for the same turn,
#                  so the frame can neither teach nor test.
set -euo pipefail
cd /home/itri/mir_isaac_test/vla
PY=/home/itri/anaconda3/envs/env_lerobot/bin/python

RAW="${RAW:-raw_msw_all}"
ROOT="${ROOT:-lerobot_mswall_box}"
REPO="${REPO:-itri/mir_mswall_box}"
# 10, not 5. The significance that came out of the previous run (20 correct
# swaps against 5 wrong, p = 0.002) rested on 150 frames drawn from five
# episodes at a stride of three -- 0.3 s apart at 10 Hz, so consecutive scored
# frames are nearly the same scene and the effective sample is far below 150.
# Widening the held-out set is the cheapest way to find out whether that result
# survives.
HOLDOUT="${HOLDOUT:-10}"
MIN_SEP="${MIN_SEP:-20}"

rm -rf "$ROOT" "${ROOT}_val"
$PY to_lerobot.py \
  --raw "$RAW" --root "$ROOT" --repo-id "$REPO" \
  --make-crops --crop-size 512 --box-action \
  --scan-in-state 24 --scan-clip 10 --no-raw-scan \
  --drop-abs-xy --drop-yaw --drop-vel \
  --drop-lead-still \
  --min-separation "$MIN_SEP" \
  --holdout "$HOLDOUT"

echo
echo "=== what came out ==="
$PY - "$ROOT" <<'PY'
import json, sys, glob
import pandas as pd
root = sys.argv[1]
i = json.load(open(f"{root}/meta/info.json"))
print(f"  {root}: {i['total_episodes']} episodes, {i['total_frames']} frames")
print("  cameras:", [k.split('.')[-1] for k in i['features']
                     if k.startswith('observation.images')])
print("  state:", i['features']['observation.state']['shape'])
t = pd.read_parquet(f"{root}/meta/tasks.parquet")
print("  tasks:", list(t.index))
f = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
d = pd.concat([pd.read_parquet(x, columns=['task_index', 'episode_index'])
               for x in f])
per = d.groupby('episode_index').task_index.nunique()
sw = d.groupby('episode_index').task_index.apply(lambda s: (s.diff() != 0).sum() - 1)
print(f"  distinct tasks per episode: min {per.min()} max {per.max()}")
print(f"  instruction switches: {sw.sum()} across {len(per)} episodes")
print("  frames per task:")
for s, idx in t['task_index'].items():
    print(f"    {s!r}: {(d.task_index == idx).sum()}")
PY
