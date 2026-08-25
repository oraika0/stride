#!/bin/bash
# clean.sh — Comprehensive cleanup for real Mininet training.
#
# Replaces the old top-level clean.sh which only killed iperf3 + ryu-manager
# + ran `mn -c`. The old one missed the most important case:
#
#   ⚠ When step 3000 reached, main.py often gets stuck for hours in
#   `net.stop()` / Ryu-Popen wait / wandb.finish. The old clean.sh
#   did NOT kill this stuck python — so the next launch couldn't
#   build Mininet topology (port 6633 still held by zombie python).
#
# This new version:
#   1. Kills stuck `main.py train` python BEFORE touching Mininet daemons
#   2. pkill -f ryu-manager (catches Ryu inside gnome-terminal — killall doesn't)
#   3. mn -c + pkill -9 mininet
#   4. Verifies controller port 6633/6653 released
#   5. Fixes file ownership (sudo training leaves root-owned results/)
#   6. Removes .drl_done sentinel
#
# Usage (no args, runs as current user with sudo):
#   ./scripts/clean.sh
#
# Auto-detects which user we're on via $HOME (works on dcnlab/dcnlab1/dcnlab2).

set -u

stamp() { date '+[%H:%M:%S]'; }
echo "$(stamp) ===== clean.sh start ====="

# One level up from scripts/, not two. This used to point at the repo's
# parent, so RESULTS_DIR and SENTINEL below named paths that do not exist
# and sections 7 and 8 were silently doing nothing at all.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# ALG only scopes the chown (§7) + sentinel removal (§8) tail -- the
# zombie-killing sections (1a-6) are process-pattern based and ALG-agnostic.
# Optional $1 lets stride real_test target results/stride; default keeps the
# default alg when called with no argument.
ALG="${1:-stride}"

# Refresh sudo (assume user did `sudo -v` before, or prompt)
sudo -v || { echo "[clean] sudo auth failed"; exit 1; }

# 1a. STUCK orchestrator python — either training (`main.py ... train`)
#     OR real test (`test_single_tm.py ... --auto`). Both are sudo-wrapped
#     under root, both build Mininet, both spawn Ryu+DRL gnome-terminals.
#     Kill FIRST so Mininet cleanup actually works.
#
#     2026-05-27: added test_single_tm.py pattern -- clean.sh used to miss
#     stuck real-test orchestrators (e.g. ckpt load crashed silently in
#     gnome-terminal, parent python sleeping in start_single_traffic's
#     360s iperf wait), leaving zombie root python that next launch's
#     `mn -c` couldn't displace.
MAIN_PIDS=$(pgrep -f "main\.py --env .* --alg .* train|test_single_tm\.py" 2>/dev/null || true)
if [ -n "$MAIN_PIDS" ]; then
    echo "$(stamp) found stuck train/test orchestrator python: $MAIN_PIDS — SIGKILL"
    sudo kill -9 $MAIN_PIDS 2>/dev/null || true
fi

# 1b. STUCK run_drl.py python (the actual DRL agent, spawned in gnome-terminal
#     subprocess). Pattern catches ANY --mode: train, test_single, test_sim_only.
#
#     2026-05-07: clean.sh used to skip this and only kill main.py, leaving
#     run_drl.py holding GPU + wandb lock → next launch couldn't bind
#     wandb-core port.
#     2026-05-27: broadened from --mode-train-only to any mode, so real-test
#     run_drl.py (--mode test_single) is also caught.
DRL_PIDS=$(pgrep -f "run_drl\.py" 2>/dev/null || true)
if [ -n "$DRL_PIDS" ]; then
    echo "$(stamp) found stuck run_drl.py: $DRL_PIDS — SIGKILL"
    sudo kill -9 $DRL_PIDS 2>/dev/null || true
fi

# 1c. Leftover wandb-core / gpu_stats (re-parent to init when run_drl dies).
#     They hold socket port that wandb.init() of next run will try to reuse.
WANDB_PIDS=$(pgrep -f "wandb/bin/(wandb-core|gpu_stats)" 2>/dev/null || true)
if [ -n "$WANDB_PIDS" ]; then
    echo "$(stamp) kill leftover wandb-core / gpu_stats: $WANDB_PIDS"
    sudo kill -9 $WANDB_PIDS 2>/dev/null || true
fi

# 1d. The bash launcher inside gnome-terminal that spawned run_drl.py.
#     Pattern: `bash -c "source ... && conda activate <env> && python run_drl.py ..."`.
#     Matched without naming the env, so renaming it cannot silently break cleanup.
#     After run_drl.py dies, the trailing `exec bash` keeps a login shell alive.
GT_BASH_PIDS=$(pgrep -f "bash.*conda activate.*run_drl\.py" 2>/dev/null || true)
if [ -n "$GT_BASH_PIDS" ]; then
    echo "$(stamp) kill gnome-terminal launcher bash: $GT_BASH_PIDS"
    sudo kill -9 $GT_BASH_PIDS 2>/dev/null || true
fi

if [ -z "$MAIN_PIDS" ] && [ -z "$DRL_PIDS" ] && [ -z "$WANDB_PIDS" ] && [ -z "$GT_BASH_PIDS" ]; then
    echo "$(stamp) no stuck main.py / run_drl.py / wandb-core"
fi
sleep 2

# 2. Ryu controller — pkill -f catches it inside gnome-terminal
#    (killall ryu-manager fails because process name is python, not ryu-manager)
# pgrep -cf prints count + returns 1 when 0 matches; old `|| echo 0` appended
# a second "0\n" → multi-line value broke `[ -gt 0 ]` (integer expression
# expected). Use $? branch instead: assign-or-zero.
# The bracket in '[r]yu-manager' keeps the pattern from matching this very
# command line. Without it, pkill -f matches its own `sudo pkill -9 -f
# ryu-manager` parent and SIGKILLs it: the real ryu survives, and sudo dies
# before restoring the tty it put into no-echo for the password prompt, which
# leaves the terminal printing newlines without carriage returns.
RYU_COUNT=$(pgrep -cf '[r]yu-manager' 2>/dev/null) || RYU_COUNT=0
if [ "$RYU_COUNT" -gt 0 ]; then
    echo "$(stamp) kill ryu-manager (×$RYU_COUNT)"
    sudo pkill -9 -f '[r]yu-manager' 2>/dev/null || true
    sleep 1
fi

# 3. iperf3
IPERF_COUNT=$(pgrep -c iperf3 2>/dev/null) || IPERF_COUNT=0
if [ "$IPERF_COUNT" -gt 0 ]; then
    echo "$(stamp) kill iperf3 (×$IPERF_COUNT)"
    sudo killall -9 iperf3 2>/dev/null || true
fi

# 4. Mininet topology + zombie processes
echo "$(stamp) mn -c"
sudo mn -c >/dev/null 2>&1
sudo pkill -9 -f '[m]ininet' 2>/dev/null || true
# ovs-vswitchd is deliberately left alone. It is the system openvswitch-switch
# daemon, shared with everyone else on the machine, and `mn -c` above already
# removes the bridges Mininet made. This line used to read `pkill -9 -f
# ovs-vswitchd`, which never actually killed anything -- it matched its own sudo
# and died first -- so not killing it is also the behaviour every run so far has
# had. If OVS really needs restarting, that is `systemctl restart
# openvswitch-switch`, on purpose, not as part of routine cleanup.
sleep 3

# 4b. tmux windows left behind by the controller and the agent.
#     Those windows get remain-on-exit so a crash stays readable, which means
#     nothing removes them either. Where they land depends on how the run was
#     started: inside tmux the launcher adds them to the current session, which
#     also holds your shell, so killing the session is not an option -- only the
#     dead panes are. Outside tmux they go to a detached 'stride' session that
#     is entirely ours, and that one goes as a whole. Both servers are swept,
#     because training runs under sudo and root's tmux is a different socket.
sweep_dead_panes() {
    $1 tmux list-panes -a -F '#{pane_id} #{pane_dead} #{window_name}' 2>/dev/null |
    while read -r pane dead wname; do
        [ "$dead" = "1" ] || continue
        case "$wname" in
            controller|drl)
                # Say where the output went before closing the pane. Sweeping
                # keeps the session usable for the next run, but it also takes
                # away the only copy anyone could still read -- unless the pane
                # was piped to a log, which is the point of that log.
                log=$(ls -1t "$REPO/results/_terminal_logs/${wname}_"*.log 2>/dev/null | head -1)
                if [ -n "$log" ]; then
                    echo "$(stamp)   closing $wname pane $pane -- output kept in $log"
                else
                    echo "$(stamp)   closing $wname pane $pane -- NO pane log; its output is gone"
                fi
                $1 tmux kill-pane -t "$pane" 2>/dev/null || true ;;
        esac
    done
}
DEAD=$(tmux list-panes -a -F '#{pane_dead} #{window_name}' 2>/dev/null |
       grep -cE '^1 (controller|drl)$') || DEAD=0
DEAD_ROOT=$(sudo tmux list-panes -a -F '#{pane_dead} #{window_name}' 2>/dev/null |
            grep -cE '^1 (controller|drl)$') || DEAD_ROOT=0
if [ "$DEAD" -gt 0 ] || [ "$DEAD_ROOT" -gt 0 ]; then
    echo "$(stamp) kill dead controller/drl panes (×$((DEAD + DEAD_ROOT)))"
    sweep_dead_panes ""
    sweep_dead_panes sudo
fi
if sudo tmux has-session -t stride 2>/dev/null; then
    echo "$(stamp) kill tmux session 'stride'"
    sudo tmux kill-session -t stride 2>/dev/null || true
fi

# 5. Verify controller port released — HARD GATE (2026-06-08).
#    Was WARN-only: if run1's ryu controller hadn't released 6633/6653 within
#    the sleeps above, the next chained run started dirty and connected to the
#    stale (frozen-monitor) controller -> constant-MLU-from-step-0 degenerate
#    runs. Reproduced 2/2 on PC2 (M12-s18 ff8mbg0y 0.895, statinit-s18 lbqw8syc
#    0.984); PC0/PC1 (faster release) unaffected. Section 2's `pkill -f
#    ryu-manager` evidently missed PC2's controller (cmdline mismatch), so kill
#    by PORT-HOLDER PID here (cmdline-agnostic) and re-verify, up to N tries;
#    fail loud (exit 1) if it never frees so the chain aborts instead of
#    dirty-running run2. Only one chain runs per host (lockfile), so the holder
#    is always stale -> safe to kill.
PORT_FREED=0
for try in 1 2 3 4 5; do
    PORT_BUSY=$(sudo netstat -tlnp 2>/dev/null | grep -E ":6633|:6653" || true)
    if [ -z "$PORT_BUSY" ]; then
        echo "$(stamp) controller port (6633/6653) free"
        PORT_FREED=1
        break
    fi
    HOLDER_PIDS=$(echo "$PORT_BUSY" | awk '{print $NF}' | cut -d/ -f1 \
                  | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
    echo "$(stamp) [try $try/5] controller port still busy -- holder PID(s): ${HOLDER_PIDS:-<none>}"
    echo "$PORT_BUSY"
    [ -n "$HOLDER_PIDS" ] && sudo kill -9 $HOLDER_PIDS 2>/dev/null || true
    sleep 2
done
if [ "$PORT_FREED" -ne 1 ]; then
    echo "$(stamp) [FATAL] controller port 6633/6653 never released after 5 tries --" >&2
    echo "$(stamp)         aborting clean to avoid a dirty (stale-controller) run." >&2
    exit 1
fi

# 6. Stale gnome-terminal windows holding shells
GT_PIDS=$(pgrep -f "gnome-terminal.*ryu\|gnome-terminal.*conda activate" 2>/dev/null || true)
if [ -n "$GT_PIDS" ]; then
    echo "$(stamp) close stale gnome-terminal: $GT_PIDS"
    kill $GT_PIDS 2>/dev/null || true
fi

# 7. Fix ownership (sudo training writes root-owned files)
RESULTS_DIR="$REPO/results/$ALG"
if [ -d "$RESULTS_DIR" ]; then
    NEEDS_CHOWN=$(find "$RESULTS_DIR" -maxdepth 2 -uid 0 -print -quit 2>/dev/null)
    if [ -n "$NEEDS_CHOWN" ]; then
        echo "$(stamp) chown root-owned files in $RESULTS_DIR back to $(id -un)"
        sudo chown -R "$(id -u):$(id -g)" "$RESULTS_DIR" 2>/dev/null || true
    fi
fi

# 8. Remove .drl_done sentinel (next training starts fresh)
SENTINEL="$REPO/results/$ALG/.drl_done"
[ -f "$SENTINEL" ] && { rm -f "$SENTINEL"; echo "$(stamp) removed sentinel $SENTINEL"; }

echo "$(stamp) ===== clean.sh done ====="
