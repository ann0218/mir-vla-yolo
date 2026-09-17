#!/usr/bin/env bash
# Paired closed-loop evaluation. Each seed is run twice, once per instruction,
# from an identical opening configuration -- same people, same robot heading,
# both people confirmed in frame before the policy takes over. Pairing is the
# point: a single run's on_target swings 44-58 % on the same settings, so an
# unpaired comparison measures the seed, not the instruction.
set -u
S="${S:-/home/itri/mir_isaac_test/vla/results_2p}"
SEEDS="${SEEDS:-1 2 3 4 5}"
# SWAPCOL=1: the scene is follow_room_2p_v2_swapcol.usd, where /World/PersonPurple
# wears the yellow tint and /World/PersonYellow the purple one. The instruction
# still names a colour, so the target is the prim now WEARING it -- the other TF
# frame. Scored this way a colour-reading policy looks the same as before; one
# that keyed on anything tied to the prim (walker index, gait, TF-side quirks)
# falls below 100 %.
SWAPCOL="${SWAPCOL:-0}"
# WEARS maps each TF frame to the colour that person wears in the loaded scene,
# e.g. WEARS="purple=purple yellow=red" for follow_room_2p_v2_redpurple.usd.
# WORDS are the colour words put in the instruction, defaulting to the colours
# worn. SWAPCOL=1 is kept as a shorthand for the colour-swap scene.
if [ "$SWAPCOL" = 1 ]; then
  WEARS="${WEARS:-purple=yellow yellow=purple}"; OUTNAME="${OUTNAME:-cl_batch_swapcol}"
else
  WEARS="${WEARS:-purple=purple yellow=yellow}"; OUTNAME="${OUTNAME:-cl_batch_balanced}"
fi
OUT="$S/$OUTNAME"; mkdir -p "$OUT"
WORDS="${WORDS:-$(for p in $WEARS; do printf '%s ' "${p#*=}"; done)}"
frame_wearing() { for p in $WEARS; do [ "${p#*=}" = "$1" ] && { echo "${p%%=*}"; return; }; done; }
other_frame()   { for p in $WEARS; do [ "${p%%=*}" != "$1" ] && { echo "${p%%=*}"; return; }; done; }
echo "WEARS=[$WEARS] WORDS=[$WORDS] -> $OUT"
for seed in $SEEDS; do
  for T in $WORDS; do
    # T = colour word in the instruction; TGT/O = TF frames of the person
    # wearing that colour and of the other one.
    TGT=$(frame_wearing "$T")
    [ -n "$TGT" ] || { echo "nobody wears '$T' in WEARS=[$WEARS]"; exit 1; }
    O=$(other_frame "$TGT")
    echo "$seed" > /tmp/mir_reset_people; sleep 4
    # --trace keeps every step's bearings. on_target alone only says the named
    # person was nearer the centreline than the other one, which is still true
    # with both behind the robot (98 % on_target at a 118 deg median bearing);
    # the trace lets "named person inside the 75 deg view" be scored instead.
    docker exec my_isaac_sim bash -lc "source /opt/ros/humble/setup.bash; cd /root/vla && \
      mkdir -p /root/vla/cl_trace && \
      timeout 400 python3 runner_2p.py --server 172.17.0.1:8770 \
      --instruction 'follow the person in $T' --target $TGT --other $O \
      --seconds 70 --warmup 25 --trace /root/vla/cl_trace/s${seed}_$T.csv ${RUNNER_ARGS:-} 2>&1" > "$OUT/s${seed}_$T.log" 2>&1
    docker cp "my_isaac_sim:/root/vla/cl_trace/s${seed}_$T.csv" "$OUT/" >/dev/null 2>&1
    line=$(grep -a "RESULT" "$OUT/s${seed}_$T.log" | tail -1)
    echo "seed $seed  $T  ${line#*RESULT }"
  done
done
