#!/bin/bash
# launch_potential_pair_local.sh — the Q29 potential-pair diagnostic (run_potential_pair.py)
# on the shared workstation tycho, natively (no docker) and within a fixed budget.
#
# The wrapper re-executes itself inside ONE transient systemd user scope:
#   CPUQuota = threads × 100 %  (default threads = nproc/4)
#   MemoryMax = --memory-gb (default 8 GiB), MemoryHigh = 90 % of it, MemorySwapMax = 0
#   IOWeight=10, OOMPolicy=continue, nice -n 19, ionice -c 3
# No rclone, docker, sudo, shutdown, upload or deletion. Results stay in --output.
# Watchdogs only notify (ntfy, status only): nothing is stopped automatically.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
_THREADS_AUTO=$(( $(nproc) / 4 ))
[ "$_THREADS_AUTO" -lt 1 ] && _THREADS_AUTO=1
MODE=""
INPUT_ROOT=""
MODELS="models_potential_check.txt"
FIELD=""
INDEX=2
OUTPUT=""
REVIEW=""
THREADS="$_THREADS_AUTO"
MEMORY_GB=8
STALL_ALERT=21600
RUNTIME_ALERT=172800
NOTIFY=1
ORIG_ARGS=("$@")

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            printf '%s\n' \
                'Q29 potential-pair diagnostic on tycho: no docker, network sync, upload or deletion.' \
                'bash launch_potential_pair_local.sh --mode=validate|preflight|pilot|fields' \
                '  --input-root=DIR --models=RELATIVE_FILE --field-report=RELATIVE_REPORT' \
                '  --model-index=0|1|2 --output=NEW_DIRECTORY [--reviewed-freeq=REPORT]' \
                '  [--threads=nproc/4] [--memory-gb=8] [--stall-alert=21600] [--runtime-alert=172800] [--no-notify]' \
                'One systemd user scope: CPUQuota=threads*100%, MemoryMax=memory-gb, MemorySwapMax=0.' \
                'Use the same --threads for free-Q and Q1 (part of the pilot environment signature).' \
                'Refuses to start next to an active agama-tycho-*/agama-pair-* scope.' \
                'Watchdogs only notify: no report.json update for --stall-alert s, or total' \
                'runtime above --runtime-alert s, sends one status-only ntfy (0 disables).' \
                'ntfy (NTFY_TOPIC, NTFY_SERVER) also reports the final status; never result values.' \
                'Stop: Ctrl-C, systemctl --user kill --signal=TERM <unit>.scope, or touch OUTPUT/run/STOP.' \
                'validate: no AGAMA import. preflight: A/B and free-Q C, no IC/orbits/solve.' \
                'pilot: five free-Q or three Q1 integrations, seed 42; Q1 needs --reviewed-freeq.' \
                'fields: Step 1A with this binary, 192x17 probes, angular orders through 32.' \
                'Env: CONDA_ENV=agamaBoTorch ("" skips), PYTHON=python, PAIR_MIN_FREE_GB=20,' \
                '  PAIR_MONITOR_INTERVAL=30, PAIR_STOP_GRACE=600.'
            exit 0 ;;
        --mode=*) MODE="${arg#*=}" ;;
        --input-root=*) INPUT_ROOT="${arg#*=}" ;;
        --models=*) MODELS="${arg#*=}" ;;
        --field-report=*) FIELD="${arg#*=}" ;;
        --model-index=*) INDEX="${arg#*=}" ;;
        --output=*) OUTPUT="${arg#*=}" ;;
        --reviewed-freeq=*) REVIEW="${arg#*=}" ;;
        --threads=*) THREADS="${arg#*=}" ;;
        --memory-gb=*) MEMORY_GB="${arg#*=}" ;;
        --stall-alert=*) STALL_ALERT="${arg#*=}" ;;
        --runtime-alert=*) RUNTIME_ALERT="${arg#*=}" ;;
        --notify) NOTIFY=1 ;;
        --no-notify) NOTIFY=0 ;;
        --shutdown|--shutdown-delay=*|--image=*|--timeout=*|--stall=*)
            printf 'ERROR: %s is VM-only (launch_potential_pair.sh); tycho never stops or shuts down automatically\n' "$arg" >&2
            exit 2 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

PAIR_MEMINFO="${PAIR_MEMINFO:-/proc/meminfo}"
PAIR_PROC_CGROUP="${PAIR_PROC_CGROUP:-/proc/self/cgroup}"
PAIR_CGROUP_FS="${PAIR_CGROUP_FS:-/sys/fs/cgroup}"
_UID="$(id -u)"
PAIR_USER_CGROUP="${PAIR_USER_CGROUP:-${PAIR_CGROUP_FS}/user.slice/user-${_UID}.slice/user@${_UID}.service}"
PAIR_MIN_FREE_GB="${PAIR_MIN_FREE_GB:-20}"
PAIR_MONITOR_INTERVAL="${PAIR_MONITOR_INTERVAL:-30}"
PAIR_STOP_GRACE="${PAIR_STOP_GRACE:-600}"
CONDA_ENV="${CONDA_ENV-agamaBoTorch}"
PYTHON="${PYTHON:-python}"
NTFY_TOPIC="${NTFY_TOPIC:-GalaxySchwarzschildFornax}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
case "$MODE" in validate|preflight|pilot|fields) ;; *) fail 'Select an explicit --mode' ;; esac
for value in "$THREADS" "$MEMORY_GB"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'threads and memory-gb must be positive integers'
done
for value in "$STALL_ALERT" "$RUNTIME_ALERT" "$PAIR_MIN_FREE_GB" "$PAIR_MONITOR_INTERVAL" "$PAIR_STOP_GRACE"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail 'alerts, intervals and PAIR_* limits must be non-negative integers'
done
[[ "$INDEX" =~ ^[012]$ ]] || fail 'model-index must be 0, 1 or 2'
[[ -d "$INPUT_ROOT" && -n "$OUTPUT" ]] || fail 'Require existing input-root and new output'
INPUT_ROOT="$(realpath "$INPUT_ROOT")"
[[ -d "$(dirname "$OUTPUT")" ]] || fail 'Output parent must already exist'
OUTPUT="$(realpath -m "$OUTPUT")"
relative_input() {
    local path
    [[ "$1" != /* ]] || fail 'Input filenames must be relative to input-root'
    path="$(realpath -e "$INPUT_ROOT/$1")" || fail 'Input does not exist'
    [[ "$path" == "$INPUT_ROOT/"* && -f "$path" ]] || fail 'Input escapes input-root or is not a file'
    printf '%s' "$path"
}
HARNESS="${CODE_DIR}/Fornax_P21_PCA_w3Sersic_orblib_exp.py"
HERE="${CODE_DIR}/check_potential_convergence"
MODEL_PATH="$(relative_input "$MODELS")"
ARGS=("${HERE}/run_potential_pair.py" --models "$MODEL_PATH" --model-index "$INDEX"
      --harness "$HARNESS" --threads "$THREADS" --output "$OUTPUT/run")
if [ "$MODE" = fields ]; then
    [[ -z "$REVIEW" ]] || fail 'Review is only valid for Q1 pilot'
    ARGS=("${HERE}/check_potential_convergence.py" --models "$MODEL_PATH" --harness "$HARNESS"
          --threads 1 --output "$OUTPUT/run" --radial-nodes 23,46,92,184
          --angular-orders 4,8,12,16,24,32 --radii 192 --angles 17)
else
    [[ -n "$FIELD" ]] || fail 'field-report is required'
    ARGS+=(--field-report "$(relative_input "$FIELD")")
    case "$MODE" in
        validate) ARGS+=(--validate-pilot) ;;
        preflight) ARGS+=(--pilot-preflight) ;;
        pilot) ARGS+=(--pilot) ;;
    esac
fi
if [ "$MODE" = pilot ] && [ "$INDEX" != 2 ]; then
    [[ -f "$REVIEW" ]] || fail 'Q1 requires --reviewed-freeq=REPORT after manual review'
    ARGS+=(--reviewed-freeq "$(realpath "$REVIEW")")
elif [ -n "$REVIEW" ]; then
    fail 'reviewed-freeq is only valid for Q1 pilot'
fi
for f in "${ARGS[0]}" "$HARNESS" "${CODE_DIR}/run_single_model.py" "${CODE_DIR}/table3.dat" \
         "${HERE}/check_potential_convergence.py"; do
    [[ -f "$f" ]] || fail "Missing code file: $f"
done

MEM_MAX=$(( MEMORY_GB * 1024 * 1024 * 1024 ))
MEM_HIGH=$(( MEM_MAX / 10 * 9 ))
CPU_QUOTA="$(( THREADS * 100 ))%"
TIMESTAMP="${PAIR_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
UNIT="agama-pair-${MODE}-${INDEX}-${TIMESTAMP}"
LOG="$OUTPUT/launcher.log"

available_kb() { awk '/^MemAvailable:/ {print $2; exit}' "$PAIR_MEMINFO" 2>/dev/null || true; }
cgroup_dir() { printf '%s%s' "$PAIR_CGROUP_FS" "$(awk -F: '$1 == "0" {print $3}' "$PAIR_PROC_CGROUP" 2>/dev/null || true)"; }
print_limits() {
    local dir f
    dir="$(cgroup_dir)"
    printf 'cgroup=%s\n' "${dir#"$PAIR_CGROUP_FS"}"
    for f in memory.max memory.high memory.swap.max cpu.max; do
        printf '%s=%s\n' "$f" "$(tr '\n' ' ' < "${dir}/${f}" 2>/dev/null | sed 's/ $//' || true)"
    done
}
activate_conda() {
    [ -n "$CONDA_ENV" ] || return 0
    command -v conda > /dev/null 2>&1 || fail "conda not found (CONDA_ENV=$CONDA_ENV; '' skips activation)"
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" || { set -u; fail 'conda.sh not found'; }
    conda activate "$CONDA_ENV" || { set -u; fail "cannot activate $CONDA_ENV"; }
    set -u
}
notify() {
    [ "$NOTIFY" -eq 1 ] || return 0
    if curl -s -f --max-time 15 -H "Title: PotentialPair $(hostname)" -H "Priority: $2" \
            -d "$1" "${NTFY_SERVER}/${NTFY_TOPIC}" >> "$LOG" 2>&1; then
        printf '%s=sent\n' "$3" >> "$LOG"
    else
        printf '%s=failed\n' "$3" >> "$LOG"
        printf 'WARNING: ntfy notification failed; continuing\n' >&2
    fi
}

# ==============================================================
# Phase 1: outside the scope — checks, new output, re-exec in the scope
# ==============================================================
if [ "${PAIR_IN_SCOPE:-0}" != 1 ]; then
    [[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || fail 'Output already exists; no overwrite/resume'
    activate_conda
    command -v "$PYTHON" > /dev/null 2>&1 || fail "Python not found: $PYTHON"
    "$PYTHON" -c 'import numpy, scipy, agama; assert callable(agama.setNumThreads)' > /dev/null 2>&1 \
        || fail "$PYTHON lacks numpy/scipy/agama with setNumThreads"
    command -v systemd-run > /dev/null 2>&1 || fail 'systemd-run is required'
    controllers="$(cat "${PAIR_USER_CGROUP}/cgroup.controllers" 2>/dev/null || true)"
    for c in cpu memory; do
        [[ " $controllers " == *" $c "* ]] || fail "cgroup controller '$c' is not delegated to the user manager"
    done
    active="$(systemctl --user list-units --type=scope --state=active --no-legend --plain \
        'agama-tycho-*' 'agama-pair-*' 2>/dev/null || true)"
    [[ -z "${active//[[:space:]]/}" ]] || fail "Another Fornax job is active; run sequentially: ${active%%$'\n'*}"
    if [ -e "${CODE_DIR}/.tycho_launcher.lock" ]; then
        ( flock -n 9 ) 9<"${CODE_DIR}/.tycho_launcher.lock" || fail 'launch_tycho.sh is running; run sequentially'
    fi
    exec 8>"$(dirname "$OUTPUT")/.potential_pair.lock"
    flock -n 8 || fail 'Another potential-pair wrapper holds the lock'
    kb="$(available_kb)"
    [[ "$kb" =~ ^[0-9]+$ ]] && (( kb > (MEMORY_GB + 2) * 1024 * 1024 )) || fail 'Insufficient host memory headroom'
    if [ "$MODE" = pilot ]; then
        FREE_KB="$(df -Pk "$(dirname "$OUTPUT")" | awk 'NR==2 {print $4}')"
        [[ "$FREE_KB" =~ ^[0-9]+$ ]] && (( FREE_KB >= PAIR_MIN_FREE_GB * 1024 * 1024 )) \
            || fail "Pilot requires ${PAIR_MIN_FREE_GB} GiB free space"
    fi
    linger="$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo unknown)"
    if [ "$linger" != yes ] && [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
        printf 'WARNING: Linger=%s and not in tmux/screen: logout stops the scope\n' "$linger" >&2
    fi
    mkdir "$OUTPUT"
    printf 'mode=%s\nmodel_index=%s\nhost=%s\nunit=%s\npython=%s\nthreads=%s\nmemory_gb=%s\ncpu_quota=%s\nstall_alert_seconds=%s\nruntime_alert_seconds=%s\nnotify_enabled=%s\n' \
        "$MODE" "$INDEX" "$(hostname)" "$UNIT" "$(command -v "$PYTHON")" "$THREADS" "$MEMORY_GB" \
        "$CPU_QUOTA" "$STALL_ALERT" "$RUNTIME_ALERT" "$NOTIFY" > "$LOG"
    printf 'Scope %s.scope; stop: systemctl --user kill --signal=TERM %s.scope\n' "$UNIT" "$UNIT"
    export PAIR_IN_SCOPE=1 PAIR_TIMESTAMP="$TIMESTAMP"
    exec systemd-run --user --scope --quiet --collect --unit="$UNIT" \
        -p "CPUQuota=${CPU_QUOTA}" -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}" \
        -p "MemorySwapMax=0" -p "IOWeight=10" -p "OOMPolicy=continue" \
        bash "$SELF" "${ORIG_ARGS[@]}"
fi

# ==============================================================
# Phase 2: inside the scope
# ==============================================================
[[ -f "$LOG" ]] || fail 'Scope phase without a prepared output'
LIMITS="$(print_limits)"
printf '%s\n' "$LIMITS" >> "$LOG"
if grep -qE '^memory\.max=(max)?$' <<< "$LIMITS"; then
    printf 'exit_code=2\nerror=scope memory limit not applied\n' >> "$LOG"
    fail 'Scope memory.max is not set; the diagnostic is not started'
fi
CG_DIR="$(cgroup_dir)"

STOP_REASON=""
KILLER=""
PID=""
on_signal() {
    [ -z "$STOP_REASON" ] || return 0
    STOP_REASON=signal
    printf 'stop_reason=signal\n' >> "$LOG"
    [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null || true
    ( sleep "$PAIR_STOP_GRACE"; kill -KILL "$PID" 2>/dev/null ) &
    KILLER=$!
}
trap on_signal INT TERM

START=$(date +%s)
OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    nice -n 19 ionice -c 3 "$PYTHON" -B "${ARGS[@]}" > "$OUTPUT/runner.log" 2>&1 &
PID=$!
printf 'runner_pid=%s\nstart=%s\n' "$PID" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"

report_field() {
    [ -f "$OUTPUT/run/report.json" ] || return 0
    sed -n "s/.*\"$1\": *\"\([A-Za-z0-9_]*\)\".*/\1/p" "$OUTPUT/run/report.json" | head -1
}
sample() {
    local f
    printf '{"time": %s' "$(date +%s)"
    for f in memory.current memory.peak memory.swap.current; do
        printf ', "%s": "%s"' "$f" "$(cat "${CG_DIR}/${f}" 2>/dev/null || echo n/a)"
    done
    printf ', "cpu_usage_usec": "%s"' "$(awk '$1 == "usage_usec" {print $2}' "${CG_DIR}/cpu.stat" 2>/dev/null)"
    printf ', "psi_memory": "%s"' "$(head -1 /proc/pressure/memory 2>/dev/null)"
    printf ', "mem_available_kb": "%s"}\n' "$(available_kb)"
}
STALL_SENT=0
RUNTIME_SENT=0
while kill -0 "$PID" 2>/dev/null; do
    sample >> "$OUTPUT/resources.jsonl" 2>/dev/null || true
    NOW=$(date +%s)
    LAST=$START
    [ ! -f "$OUTPUT/run/report.json" ] || LAST=$(stat -c %Y "$OUTPUT/run/report.json")
    if [ "$STALL_ALERT" -gt 0 ] && [ "$STALL_SENT" -eq 0 ] && (( NOW - LAST > STALL_ALERT )); then
        STALL_SENT=1
        printf 'stall_alert=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
        notify "mode=${MODE} index=${INDEX}: report.json unchanged for $(( NOW - LAST )) s, stage=$(report_field stage); still running, not stopped; dir=$(basename "$OUTPUT")" high stall_notify
    fi
    if [ "$RUNTIME_ALERT" -gt 0 ] && [ "$RUNTIME_SENT" -eq 0 ] && (( NOW - START > RUNTIME_ALERT )); then
        RUNTIME_SENT=1
        printf 'runtime_alert=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
        notify "mode=${MODE} index=${INDEX}: running for $(( NOW - START )) s; still running, not stopped; dir=$(basename "$OUTPUT")" high runtime_notify
    fi
    sleep "$PAIR_MONITOR_INTERVAL" &
    wait $! 2>/dev/null || true
done
while true; do
    CODE=0
    wait "$PID" || CODE=$?
    kill -0 "$PID" 2>/dev/null || break
done
trap - INT TERM
[ -z "$KILLER" ] || kill "$KILLER" 2>/dev/null || true
{
    printf 'memory.peak=%s\n' "$(cat "${CG_DIR}/memory.peak" 2>/dev/null || echo n/a)"
    printf 'memory.max=%s\n' "$(cat "${CG_DIR}/memory.max" 2>/dev/null || echo n/a)"
    sed 's/^/memory.events./; s/ /=/' "${CG_DIR}/memory.events" 2>/dev/null || true
    sed 's/^/cpu.stat./; s/ /=/' "${CG_DIR}/cpu.stat" 2>/dev/null || true
} > "$OUTPUT/scope_state.txt"
OOM="$(awk '$1 == "oom_kill" {print $2}' "${CG_DIR}/memory.events" 2>/dev/null || true)"
RUNNER_CODE=$CODE
[ -z "$STOP_REASON" ] || CODE=124
printf 'runner_exit=%s\nexit_code=%s\nwall_seconds=%s\noom_kill=%s\n' \
    "$RUNNER_CODE" "$CODE" "$(( $(date +%s) - START ))" "${OOM:-n/a}" >> "$LOG"
[[ "${OOM:-0}" =~ ^[0-9]+$ ]] && (( ${OOM:-0} > 0 )) \
    && printf 'WARNING: OOM kill in the scope (memory.max=%s); see scope_state.txt\n' "$MEM_MAX" >&2
sync
STATUS=no_report
[ ! -f "$OUTPUT/run/report.json" ] || STATUS="$(report_field status)"
SUMMARY="mode=${MODE} index=${INDEX} exit=${CODE} status=${STATUS:-unparsed}"
[ -z "$STOP_REASON" ] || SUMMARY="${SUMMARY} stopped=${STOP_REASON}"
[[ "${OOM:-0}" =~ ^[1-9] ]] && SUMMARY="${SUMMARY} oom_kill=${OOM}"
SUMMARY="${SUMMARY} dir=$(basename "$OUTPUT") (results kept on tycho)"
printf 'Exit %s; results: %s\n' "$CODE" "$OUTPUT"
notify "$SUMMARY" "$([ "$CODE" -eq 0 ] && echo default || echo high)" notify
exit "$CODE"
