# MiR + YOLO + VLA — following the person the instruction names

A MiR100 in Isaac Sim follows one of two people. The two are the same character
model and differ only in the colour of their clothes; the instruction is
`follow the person in purple` or `follow the person in yellow`, and it switches
twice inside every episode. YOLO finds the people, a finetuned **SmolVLA** reads
the instruction, decides **which** of them to follow, and drives.

The policy never receives anyone's position. Ground truth is used by the scripted
expert during collection and for scoring afterwards — never as a model input.

---

## Division of labour

This is not a pure end-to-end VLA, and the split matters when reading the results.

| | does |
|---|---|
| **YOLO11n + ByteTrack** | detects people every frame; crops each to 512×512; draws its boxes on the full frame |
| **SmolVLA (450M, 100M trainable)** | reads the instruction, picks which person, outputs `[v, w]` — and, in the last experiment, where it thinks that person is |

The policy's three visual streams are all built by YOLO: the marked full frame
(`camera1`) and the two person crops sorted **left to right** (`camera2`,
`camera3`). Sorting by image column rather than by track id is deliberate: track
ids are issued in order of first appearance, so the target would land in the same
slot far more often than chance and the policy could learn the slot instead of
the instruction.

Why crops at all: SmolVLM2 splits a 512×512 input into 1024 patches, and a person
at following distance occupies 1–7 of them. At that size the colour is nearly
absent and the swap rate never rose above 23 %. A crop blown up to full size
carries 70,000–120,000 saturated pixels of one colour and none of the other, which
turns "which of these two" from a resolution problem into a reading-the-instruction
problem.

---

## Results

Every closed-loop batch below is **paired**: 12 seeds, each run twice from an
identical opening configuration (people and robot teleported back, same walk
seed), once per instruction. Odd seeds swap which side each person starts on, so
a policy that maps a colour word to a fixed turn direction scores exactly 100 %
on the summed metric and cannot fake a result.

| batch | what changed | instruction effect | notes |
|---|---|---|---|
| `cl_batch_balanced` | rotation only | on-target sum **+32.3**, 11/12 pairs, p = 0.002 | first valid result |
| `cl_batch_swapcol` | the two colours swapped | **+31.8**, 11/11, p < 0.001 | it follows the colour, not the prim |
| `cl_batch_redpurple` | purple + **red** (unseen word) | +18.7, 7/12, p = 0.039 | borderline |
| `cl_batch_redyellow` | yellow + **red** | +1.5, 2/11, p = 0.84 | no effect |
| `cl_batch_drive` | forward speed published | **+36.0**, 11/11, p < 0.001 | 1 tip-over, 9/23 runs needed a ground-truth stop |
| `cl_batch_drive_safe` | sensor-only safety layer | **+25.8**, 11/12, p = 0.001 | 0 tip-overs |
| `cl_batch_box` | policy also predicts a box | **+29.8**, 10/12, p = 0.003 | control unchanged within noise |

Two scores are reported per pair. `on_target` counts steps where the named person
is nearer the centreline than the other — a policy that ignores the instruction
scores 100 % summed over the pair. The stricter one is `view-diff`: how much more
of the episode the **named** person spends inside the 75.2° camera view than the
other one, summed over the pair, which is 0 for any rule that does not depend on
the words.

### What the policy learned, and did not

**It follows the named colour.** Swapping the two colours in the scene
(`--swap-colours`, a 24-byte change to the USD) leaves prim names, TF frames,
walker indices and gaits untouched. The policy followed the colour in 11/11 pairs,
which rules out every cue tied to the individual rather than to appearance.

**It does not generalise to an unseen colour word.** Replacing one person with red
and asking for `red`: scene A (purple + red) is borderline, scene B (yellow + red)
shows nothing. The decisive number is that in scene B the red person is in view
~59 % of the time **whether the word is "red" or "yellow"** — the unseen colour
draws the robot regardless of the instruction, and it overrides the trained word.
The pre-registered reading table (`results/cl_batch_redpurple/DESIGN.txt`) did not
anticipate that pattern; none of its four rows fits.

**It can say where it thinks the person is.** The action was extended to
`[v, w, x1, y1, x2, y2, visible]`; labels are free, because the crop pipeline
already computes the detections and the recorded ground truth says which person
the instruction names. On held-out frames the predicted box lands on the named
person in 87 of 92 frames (median centre error 11 px). Driving itself, it degrades
to 38 px and 67 % within 60 px — but of 9241 steps where it claims a box, only
483 (5 %) land on the **other** person; 3684 (40 %) land on nobody. So the failure
is vague localisation, not mistaken identity. Control was unchanged within noise
(paired differences p = 0.43–0.70).

`media/box_overlay.mp4` is 45 s of that policy driving, with its own predicted box
in green, YOLO's detections in grey, and each person's true position marked at the
bottom edge. `media/frame_*.png` are four stills: the box correct, on the wrong
person, on nobody, and the policy correctly reporting it cannot see them.

### Following quality

With driving enabled the policy keeps the named person at ~2.8–3.0 m against the
expert's 2.0 m standoff, and in view about half the time. Enabling forward speed
did **not** improve how often the named person is in view (paired difference
−15 pts, p = 0.20) — the earlier guess that the rotation-only protocol had
understated it was wrong.

---

## Safety

The people have **no collision geometry** — deliberately, so the robot cannot
shove them — and the PhysX lidar therefore passes straight through them: a person
at 5.35 m reads 8.00 m (the wall behind). Nothing in the robot's own sensing
except the camera can see a person at all.

`cl_batch_drive` used a ground-truth emergency stop and needed it in 9 of 23 runs.
`cl_batch_drive_safe` replaced it with sensor-only limits:

- forward acceleration capped at 1.0 m/s² (a tip-over followed a 0.35 → 0.87 m/s
  jump in one tick)
- wall slowdown on the **true** forward arc
- camera stop: no forward speed while a YOLO person box inside the central ±25°
  reaches image row 470, which measured out at ≈ 2.0 m

Result: 24/24 runs completed, no tip-overs, close approaches to the distractor
down from 184 to 42 steps. It is still not complete — the robot came within
0.59 m, and 56 of the 66 unguarded close steps involved a person **beside** the
robot, outside the camera stop's cone.

---

## Three bugs that invalidated results, and how they showed up

**The policy was fed a random camera.** `runner_2p.py` filled its image dict in
message-arrival order and the server took the first entry; 5 of 5 fresh runner
processes put a corner camera first. All three cameras are 640×480, so nothing
downstream complained: the policy tracked from a view of the whole room, a
different camera each episode. Every closed-loop number from 9/11–9/14 was
discarded. The tell was two numbers that cannot both be true — two people detected
in 67 % of frames while the target sat behind the robot in 4 of 12 runs.

**The expert's "front" lidar arc is the rear.** `/scan` is `virtual_laser_link`
with `angle_min` = −180°, so beam 0 points **backwards**, while
`follow_expert.front_clear()` treats beams around index 0 as ahead. The training
data was collected with a wall slowdown that reacted to what was behind the robot.
`runner_2p.py --true-front` selects by beam angle instead.

**A screenshot, not the scene, was the wrong colour.** An ad-hoc grab wrote the
`rgb8` camera message straight to `cv2.imwrite`, swapping red and blue, and a
report went out saying the "yellow" person was actually teal. The recorder and the
runner both convert through `cv_bridge`; the training frames show true purple and
yellow. `vla/grab_image.py` exists so that cannot recur.

---

## Layout

```
isaac_sim/
  mir_isaac_sim.py            shared simulator: --platform-camera, --walk-person,
                              --people, --person-home, --reset-file, --fold-arm
  gen_follow_scene_2p.py      builds the room; --wear / --swap-colours dress the people
vla/
  person_crops.py             YOLO + ByteTrack; one frame in, marked frame + two crops out
  follow_expert.py            scripted expert (TF only), --target-schedule switches mid-episode
  collect_2p_move_sw.py       expert + recorder per episode
  to_lerobot.py               conversion; --make-crops, --box-action
  train_mswall_vla.sh         SmolVLA, 2-dim action
  train_mswall_box_vla.sh     SmolVLA, 7-dim action with the box
  policy_server.py            serves the policy over localhost HTTP; --crops builds the streams
  runner_2p.py                closed loop; --drive, --true-front, --cam-stop-y2, --save-frames
  eval_closed_loop.sh         paired batch over seeds; WEARS/WORDS map colours to TF frames
  analyze_closed_loop.py      paired statistics
  probe_box.py                offline box scoring on held-out frames
  analyze_box.py              box scoring from closed-loop traces
  make_clip.py                frames -> mp4 (no ffmpeg on either side)
results/                      per batch: DESIGN.txt written before the runs, SUMMARY.txt after,
                              one CSV per run (per-tick bearings, distances, commands)
media/                        the clip, four stills, the label check, scene screenshots
```

Not included: datasets, checkpoints, raw dumps and the 450 PNGs the clip was made
from — all regenerable from the scripts here.

## Running it

Isaac Sim runs in a host conda environment, ROS 2 Humble and the runners in a
container, training and inference in a third environment on the host, talking to
the container over localhost HTTP.

```bash
# scene + stack  (the control stack MUST be restarted after Isaac, and the arm
# tucked through the controller afterwards, or it hangs in the camera's view)
bash vla/launch_2p_drive.sh
docker exec -d my_isaac_sim bash -c 'ros2 launch mir_description mir_isaac.launch.py ...'
docker exec my_isaac_sim bash /root/vla/tuck_arm.sh

# collect, convert, train
python3 vla/collect_2p_move_sw.py --out raw_msw_all -n 25
bash vla/regen_mswall_box_dataset.sh
bash vla/train_mswall_box_vla.sh

# serve and measure
python vla/policy_server.py --ckpt train/smolvla_mswall_box/checkpoints/030000/pretrained_model \
    --crops --host 0.0.0.0
RUNNER_ARGS="--drive --max-accel 1.0 --true-front --cam-stop-y2 470" \
  OUTNAME=cl_batch_box SEEDS="1 2 3 4 5 6 7 8 9 10 11 12" bash vla/eval_closed_loop.sh
python vla/analyze_closed_loop.py results/cl_batch_box
python vla/analyze_box.py results/cl_batch_box
```

## Caveats worth carrying

- Simulation only, one scene, one checkpoint per experiment, 12 paired seeds.
- The box labels come from YOLO plus ground-truth matching, so "the box is on the
  named person" measures agreement with that labelling; six frames were checked by
  eye (`media/label_check.png`) and that is the only independent evidence.
- The policy has no memory (`n_obs_steps` is not referenced anywhere in SmolVLA's
  model code) and no search behaviour: the expert always kept the target in frame,
  so 19–33 % of closed-loop steps see nobody at all and it cannot recover on
  purpose.
