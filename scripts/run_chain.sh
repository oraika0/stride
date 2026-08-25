#!/usr/bin/env bash
# clean -> train -> clean -> test -> clean, in one command.
#
#   ./scripts/run_chain.sh                                  # 32node, stride
#   ./scripts/run_chain.sh geant_directed stride            # another env
#   STRIDE_CHUNK=4 ./scripts/run_chain.sh                   # with an override
#   ./scripts/run_chain.sh 32node_144tm_directed stride 18  # another seed
#
# Why a script rather than three commands: the test has to name the checkpoint
# the training just wrote, and that path contains a timestamp nobody knows in
# advance. This picks the newest run directory once training returns, so the
# pair cannot drift apart.
#
# Run it inside tmux. Training takes about eight hours, the test another hour,
# and the controller and the agent open windows of their own beside this one.
set -uo pipefail

ENV_NAME=${1:-32node_144tm_directed}
ALG_NAME=${2:-stride}
SEED=${3:-}

cd "$(dirname "$0")/.." || exit 1
REPO=$PWD
PY=${PY:-$HOME/miniconda3/envs/stride/bin/python}
[ -x "$PY" ] || { echo "[chain] no interpreter at $PY -- set PY=..." >&2; exit 1; }

# sudo -E does not reliably carry these, so pass them as explicit assignments.
# Empty ones are skipped, so an unset variable does not become VAR=''.
FORWARD=()
for v in STRIDE_VARIANT STRIDE_CHUNK STRIDE_PRECISION STRIDE_ATTN_KERNEL STRIDE_EVAL_SAMPLE; do
    [ -n "${!v:-}" ] && FORWARD+=("$v=${!v}")
done
SEED_ARG=()
[ -n "$SEED" ] && SEED_ARG=(--seed "$SEED")

# Label this window while the chain owns it. Run directly, the script lives in
# whichever window you happened to be in, which then reads as a shell next to the
# controller and drl windows the launcher names properly. Restored on the way out
# so the shell does not keep the name afterwards.
if [ -n "${TMUX:-}" ]; then
    WINDOW_WAS=$(tmux display-message -p '#{window_name}' 2>/dev/null || true)
    tmux rename-window chain 2>/dev/null || true   # restored by the EXIT trap below
fi

say() { echo; echo "=== [chain] $* ==="; echo; }
run_clean() { "$REPO/scripts/clean.sh" || echo "[chain] clean.sh returned $? -- continuing"; }

# One sudo prompt up front, refreshed in the background, so an eight-hour
# training run does not stop at a password prompt hours after you walked away.
sudo -v || { echo "[chain] sudo authentication failed" >&2; exit 1; }
while true; do sudo -n true; sleep 60; kill -0 $$ 2>/dev/null || exit; done 2>/dev/null &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null; [ -n "${WINDOW_WAS:-}" ] && tmux rename-window "$WINDOW_WAS" 2>/dev/null' EXIT

say "clean before training"
run_clean

# Which run directories exist now, so the one training creates can be identified
# by difference rather than by timestamp. Picking "the newest" would quietly test
# the wrong checkpoint if anything else touched an older directory, and a test
# against the wrong checkpoint looks exactly like a test against the right one.
RUNS_DIR="$REPO/results/$ALG_NAME/runs"
BEFORE=$(ls -1 "$RUNS_DIR" 2>/dev/null | sort)

say "train: env=$ENV_NAME alg=$ALG_NAME ${SEED:+seed=$SEED} ${FORWARD[*]:-}"
sudo -E "${FORWARD[@]}" "$PY" main.py --env "$ENV_NAME" --alg "$ALG_NAME" "${SEED_ARG[@]}" train
TRAIN_STATUS=$?
[ $TRAIN_STATUS -eq 0 ] || { echo "[chain] training exited $TRAIN_STATUS -- stopping before the test" >&2; run_clean; exit $TRAIN_STATUS; }

say "clean between training and test"
run_clean

AFTER=$(ls -1 "$RUNS_DIR" 2>/dev/null | sort)
NEW=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER"))
COUNT=$(echo "$NEW" | grep -c . || true)
if [ "$COUNT" -eq 1 ]; then
    RUN="$RUNS_DIR/$NEW/"
elif [ "$COUNT" -eq 0 ]; then
    echo "[chain] training created no new run directory under $RUNS_DIR" >&2
    exit 1
else
    echo "[chain] $COUNT new run directories appeared, cannot tell which to test:" >&2
    echo "$NEW" | sed 's/^/  /' >&2
    exit 1
fi
[ -d "${RUN}train/model" ] || { echo "[chain] $RUN has no train/model -- training did not archive a checkpoint" >&2; exit 1; }
echo "[chain] testing the checkpoint in ${RUN}train/model"

say "test"
sudo -E "${FORWARD[@]}" "$PY" test_single_tm.py --env "$ENV_NAME" --alg "$ALG_NAME" \
     "${SEED_ARG[@]}" --auto --model "${RUN}train/model"
TEST_STATUS=$?

say "clean after test"
run_clean

echo "[chain] done. train=$TRAIN_STATUS test=$TEST_STATUS"
echo "[chain] run archive: $RUN"
exit $TEST_STATUS
