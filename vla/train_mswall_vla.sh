#!/usr/bin/env bash
# Train SmolVLA on the two-colour, language-conditioned task.
#
# Fifth attempt. The instruction switches mid-episode AND each candidate is
# handed to the model as a full-frame crop from a person tracker, so the colour
# that names them is no longer 1-7 patches out of 1024 but the subject of its
# own image.
#
# The first two failed for the same reason, which took a while to see: with one
# target per episode, "keep tracking whoever you are already tracking"
# reproduces the demonstration, so the instruction is never needed. Across five
# checkpoints and three configurations (including LoRA on the frozen towers)
# the instruction-swap rate never left 0-2.8 %. Nothing about the architecture
# was going to fix a dataset the task could be solved without reading.
#
# This dataset has 45 switches across 23 episodes. At each one the scene is
# unchanged, the robot state is unchanged, and the correct action reverses --
# verified in the dump: w goes from -0.60 to +0.60 across a single switch. A
# policy that ignores the text is wrong on one side of every switch.
#
# This is the run the whole two-person scene exists for. The single-person VLA
# was measured ignoring its text input completely -- swapping "follow the
# person" for "stop immediately and do not move" moved the commanded velocity by
# 0.06 against a 0.34 spread across frames, and the nonsense string "banana
# telescope kettle" moved it just as much. That was a property of the DATASET,
# not the architecture: one task means the instruction is a constant, and a
# constant input carries no gradient. Here the instruction alternates between
# two people who differ only in the colour of their clothes, so a policy that
# ignores the text can be right at best half the time.
#
#   3 cameras      front is the robot's own view and carries its heading;
#                  corner_a and corner_b look in from opposite corners of the
#                  room and always contain both people. Measured on this scene,
#                  the front camera alone had both in frame 15.7 % of ticks and
#                  the geometry caps that at FOV/360 -- even a 180 deg lens only
#                  reaches 48.7 %. Three is also exactly what smolvla_base
#                  declares, so all three map onto camera1/2/3 with nothing
#                  padded and nothing dropped.
#
#   --rename_map   smolvla_base was pretrained with camera1/2/3 and validates
#                  that the dataset's visual keys are a subset or a superset of
#                  its own. Descriptive names are kept in the dataset, where
#                  they are worth having, and mapped here.
#
#   batch 4        three 640x480 streams per sample against the single-camera
#                  run's one. Raise if memory allows; lower first if it OOMs.
#
#   30k steps      the dataset is 12k frames against the follow task's 19k, and
#                  the behaviour is simpler (turn towards one of two people).
#                  Pick the checkpoint on the held-out split afterwards, not
#                  `last` -- on every run so far the last checkpoint was not the
#                  best one.
set -euo pipefail
cd /home/itri/mir_isaac_test/vla
TRAIN=/home/itri/anaconda3/envs/env_lerobot/bin/lerobot-train

BATCH="${BATCH:-4}"
STEPS="${STEPS:-30000}"
ROOT="${ROOT:-lerobot_mswall}"
OUT="${OUT:-train/smolvla_mswall}"

# front carries WHERE the people are, left/right carry WHICH is which. The
# order matters only in that all three must be named, and must match what
# person_crops.py produces at inference.
RENAME='{"observation.images.front": "observation.images.camera1",
         "observation.images.left":  "observation.images.camera2",
         "observation.images.right": "observation.images.camera3"}'

echo "######## SmolVLA two-colour: batch=$BATCH steps=$STEPS  ($(date '+%F %T')) ########"
$TRAIN \
  --dataset.repo_id itri/mir_mswall \
  --dataset.root "$ROOT" \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --rename_map="$RENAME" \
  --output_dir "$OUT" \
  --job_name smolvla_mswall \
  --seed 1000 \
  --batch_size "$BATCH" \
  --steps "$STEPS" \
  --save_freq 5000 \
  --log_freq 250 \
  --num_workers 8 \
  --wandb.enable false
echo "TWO-COLOUR TRAINING DONE  ($(date '+%F %T'))"
