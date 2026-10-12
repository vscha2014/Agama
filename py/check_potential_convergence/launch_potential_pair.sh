#!/bin/bash
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE=""
INPUT_ROOT=""
MODELS="models_potential_check.txt"
FIELD=""
INDEX=2
OUTPUT=""
REVIEW=""
IMAGE="agama:latest"
THREADS=8
MEMORY_GB=8
TIMEOUT=21600
STALL=3600
GRACE=120
SHUTDOWN=0
SHUTDOWN_DELAY=2
NOTIFY=0
NTFY_TOPIC="${NTFY_TOPIC:-GalaxySchwarzschildFornax}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            printf '%s\n' \
                'Isolated VM diagnostic: no network, upload or automatic deletion.' \
                'bash launch_potential_pair.sh --mode=validate|preflight|pilot|fields' \
                '  --input-root=DIR --models=RELATIVE_FILE --field-report=RELATIVE_REPORT' \
                '  --model-index=0|1|2 --output=NEW_DIRECTORY [--reviewed-freeq=REPORT]' \
                '  --image=agama:latest --threads=8 --memory-gb=8 --timeout=21600 --stall=3600' \
                '  [--shutdown] [--shutdown-delay=2] [--notify]' \
                'Shutdown and ntfy are OFF by default and happen only after the container exits,' \
                'once logs/results are written and synced; the delay is just a cancellation window.' \
                'Results are NOT uploaded: they stay on this VM disk, so restart it to read them.' \
                '--notify sends status/exit code only (NTFY_TOPIC, NTFY_SERVER), never result values.' \
                'Cancel a scheduled shutdown with: sudo shutdown -c' \
                'validate: no AGAMA import. preflight: A/B and free-Q C, no IC/orbits/solve.' \
                'pilot: five free-Q integrations or three Q1 integrations, seed 42 only.' \
                'Q1 pilot requires explicit review of the free-Q output/run/report.json.' \
                'fields: Step 1A on this VM binary, 192x17 probes, angular orders through 32.' \
                'fields exit 3 can mean baseline needs refinement; inspect reference/B/C before proceeding.' \
                'Input/code mounts are read-only; only the new output and tmpfs are writable.' \
                'Containers are retained for inspection. No resume or seed-series expansion.'
            exit 0 ;;
        --mode=*) MODE="${arg#*=}" ;;
        --input-root=*) INPUT_ROOT="${arg#*=}" ;;
        --models=*) MODELS="${arg#*=}" ;;
        --field-report=*) FIELD="${arg#*=}" ;;
        --model-index=*) INDEX="${arg#*=}" ;;
        --output=*) OUTPUT="${arg#*=}" ;;
        --reviewed-freeq=*) REVIEW="${arg#*=}" ;;
        --image=*) IMAGE="${arg#*=}" ;;
        --threads=*) THREADS="${arg#*=}" ;;
        --memory-gb=*) MEMORY_GB="${arg#*=}" ;;
        --timeout=*) TIMEOUT="${arg#*=}" ;;
        --stall=*) STALL="${arg#*=}" ;;
        --shutdown) SHUTDOWN=1 ;;
        --shutdown-delay=*) SHUTDOWN_DELAY="${arg#*=}" ;;
        --notify) NOTIFY=1 ;;
        *) printf 'Unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
case "$MODE" in validate|preflight|pilot|fields) ;; *) fail 'Select an explicit --mode' ;; esac
for value in "$THREADS" "$MEMORY_GB" "$TIMEOUT" "$STALL"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail 'Resource limits must be positive integers'
done
[[ "$SHUTDOWN_DELAY" =~ ^[0-9]+$ ]] || fail 'shutdown-delay must be a non-negative integer'
[[ "$INDEX" =~ ^[012]$ ]] || fail 'model-index must be 0, 1 or 2'
[[ -d "$INPUT_ROOT" && -n "$OUTPUT" ]] || fail 'Require existing input-root and new output'
INPUT_ROOT="$(realpath "$INPUT_ROOT")"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || fail 'Output already exists; no overwrite/resume'
[[ -d "$(dirname "$OUTPUT")" ]] || fail 'Output parent must already exist'
OUTPUT="$(realpath -m "$OUTPUT")"
for value in "$CODE_DIR" "$INPUT_ROOT" "$OUTPUT" "$REVIEW"; do
    [[ "$value" != *,* && "$value" != *$'\n'* ]] || fail 'Mount paths cannot contain commas or newlines'
done
relative_input() {
    local path
    [[ "$1" != /* ]] || fail 'Input filenames must be relative to input-root'
    path="$(realpath -e "$INPUT_ROOT/$1")" || fail 'Input does not exist'
    [[ "$path" == "$INPUT_ROOT/"* && -f "$path" ]] || fail 'Input escapes input-root or is not a file'
    printf '/input/%s' "${path#"$INPUT_ROOT/"}"
}
MODEL_PATH="$(relative_input "$MODELS")"
ARGS=(-B /code/check_potential_convergence/run_potential_pair.py --models "$MODEL_PATH"
      --model-index "$INDEX" --harness /code/Fornax_P21_PCA_w3Sersic_orblib_exp.py
      --threads "$THREADS" --output /output/run)
if [ "$MODE" = fields ]; then
    [[ -z "$REVIEW" ]] || fail 'Review is only valid for Q1 pilot'
    ARGS=(-B /code/check_potential_convergence/check_potential_convergence.py --models "$MODEL_PATH"
          --harness /code/Fornax_P21_PCA_w3Sersic_orblib_exp.py --threads 1 --output /output/run
          --radial-nodes 23,46,92,184 --angular-orders 4,8,12,16,24,32 --radii 192 --angles 17)
else
    [[ -n "$FIELD" ]] || fail 'field-report is required'
    ARGS+=(--field-report "$(relative_input "$FIELD")")
    case "$MODE" in
        validate) ARGS+=(--validate-pilot) ;;
        preflight) ARGS+=(--pilot-preflight) ;;
        pilot) ARGS+=(--pilot) ;;
    esac
fi
MOUNTS=(--mount "type=bind,src=$CODE_DIR,dst=/code,readonly"
        --mount "type=bind,src=$INPUT_ROOT,dst=/input,readonly"
        --mount "type=bind,src=$OUTPUT,dst=/output")
if [ "$MODE" = pilot ] && [ "$INDEX" != 2 ]; then
    [[ -f "$REVIEW" ]] || fail 'Q1 requires --reviewed-freeq=REPORT after manual review'
    REVIEW="$(realpath "$REVIEW")"
    MOUNTS+=(--mount "type=bind,src=$REVIEW,dst=/review/report.json,readonly")
    ARGS+=(--reviewed-freeq /review/report.json)
elif [ -n "$REVIEW" ]; then
    fail 'reviewed-freeq is only valid for Q1 pilot'
fi
available_kb() {
    local name value rest
    while read -r name value rest; do
        if [ "$name" = MemAvailable: ]; then printf '%s\n' "$value"; return; fi
    done < /proc/meminfo
    return 1
}
[[ "$(docker info --format '{{.SwapLimit}}')" = true ]] || fail 'Docker swap accounting must be confirmed'
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")" || fail 'Local image missing; no automatic pull'
[[ -n "$IMAGE_ID" ]] || fail 'Empty image identity'
(( $(available_kb) > (MEMORY_GB + 2) * 1024 * 1024 )) || fail 'Insufficient host memory headroom'
if [ "$MODE" = pilot ]; then
    FREE_KB="$(df -Pk "$(dirname "$OUTPUT")" | awk 'NR==2 {print $4}')"
    [[ "$FREE_KB" =~ ^[0-9]+$ ]] && (( FREE_KB > 20 * 1024 * 1024 )) || fail 'Pilot requires 20 GiB free space'
fi
mkdir "$OUTPUT"
NAME="agama_pair_$(date +%Y%m%d_%H%M%S)_${INDEX}_$$"
printf 'mode=%s\nmodel_index=%s\nimage=%s\nimage_id=%s\ncontainer=%s\nthreads=%s\nmemory_gb=%s\nshutdown=%s\n' \
    "$MODE" "$INDEX" "$IMAGE" "$IMAGE_ID" "$NAME" "$THREADS" "$MEMORY_GB" "$SHUTDOWN" > "$OUTPUT/launcher.log"
CONTAINER_ARGS=(--name "$NAME" --network=none --read-only --cap-drop=ALL
    --security-opt=no-new-privileges --init --pids-limit=512 --cpus="$THREADS"
    --memory="${MEMORY_GB}g" --memory-swap="${MEMORY_GB}g" --stop-timeout="$GRACE"
    --user "$(id -u):$(id -g)" --entrypoint python3 --workdir /output
    --tmpfs /tmp:rw,nosuid,nodev,size=1g -e HOME=/tmp -e MPLBACKEND=Agg -e MPLCONFIGDIR=/tmp/mpl
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS="$THREADS"
    -e POTENTIAL_PAIR_IMAGE_ID="$IMAGE_ID"
    -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1
    "${MOUNTS[@]}" "$IMAGE_ID" "${ARGS[@]}")
CID="$(docker create "${CONTAINER_ARGS[@]}")"
printf '%s\n' "$CID" > "$OUTPUT/container_id.txt"
STOP_REASON=""
stop_container() {
    STOP_REASON="$1"
    printf 'stop_reason=%s\n' "$STOP_REASON" >> "$OUTPUT/launcher.log"
    docker stop --time "$GRACE" "$NAME" >> "$OUTPUT/launcher.log" 2>&1 || true
}
trap 'stop_container signal' INT TERM
START=$(date +%s)
docker start --attach "$NAME" > "$OUTPUT/container.log" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    NOW=$(date +%s)
    docker stats --no-stream --format '{{json .}}' "$NAME" >> "$OUTPUT/resources.jsonl" 2>&1 || true
    LAST=$START
    [ ! -f "$OUTPUT/run/report.json" ] || LAST=$(stat -c %Y "$OUTPUT/run/report.json")
    if (( NOW - START > TIMEOUT )); then stop_container timeout; break; fi
    if (( NOW - LAST > STALL )); then stop_container no_progress; break; fi
    if (( $(available_kb) < 2 * 1024 * 1024 )); then stop_container host_memory; break; fi
    sleep "${PAIR_MONITOR_INTERVAL:-10}"
done
set +e
wait "$PID"
CODE=$?
set -e
trap - INT TERM
docker inspect --format '{{json .State}}' "$NAME" > "$OUTPUT/container_state.json"
[ -z "$STOP_REASON" ] || CODE=124
printf 'exit_code=%s\n' "$CODE" >> "$OUTPUT/launcher.log"
printf 'Exit %s; results: %s; retained container: %s\n' "$CODE" "$OUTPUT" "$NAME"
sync
notify() {
    [ "$NOTIFY" -eq 1 ] || return 0
    if curl -s -f --max-time 15 -H "Title: PotentialPair $(hostname)" \
            -H "Priority: $2" -d "$1" "${NTFY_SERVER}/${NTFY_TOPIC}" \
            >> "$OUTPUT/launcher.log" 2>&1; then
        printf 'notify=sent\n' >> "$OUTPUT/launcher.log"
    else
        printf 'notify=failed\n' >> "$OUTPUT/launcher.log"
        printf 'WARNING: ntfy notification failed; continuing\n' >&2
    fi
}
STATUS=no_report
[ ! -f "$OUTPUT/run/report.json" ] || STATUS="$(sed -n 's/.*"status": *"\([a-z_]*\)".*/\1/p' \
    "$OUTPUT/run/report.json" | head -1)"
SUMMARY="mode=${MODE} index=${INDEX} exit=${CODE} status=${STATUS:-unparsed}"
[ -z "$STOP_REASON" ] || SUMMARY="${SUMMARY} stopped=${STOP_REASON}"
SUMMARY="${SUMMARY} dir=$(basename "$OUTPUT") (results kept on the VM disk)"
if [ "$SHUTDOWN" -eq 1 ]; then
    notify "${SUMMARY}; shutdown in ${SHUTDOWN_DELAY} min" \
        "$([ "$CODE" -eq 0 ] && echo default || echo high)"
    printf 'Results stay on this VM disk (no upload). Shutdown in %s min; cancel: sudo shutdown -c\n' \
        "$SHUTDOWN_DELAY"
    if sudo -n shutdown -h "+${SHUTDOWN_DELAY}" >> "$OUTPUT/launcher.log" 2>&1; then
        printf 'shutdown_scheduled=+%s\n' "$SHUTDOWN_DELAY" >> "$OUTPUT/launcher.log"
    else
        printf 'shutdown_scheduled=failed\n' >> "$OUTPUT/launcher.log"
        printf 'WARNING: shutdown could not be scheduled; the VM is still running and billing\n' >&2
        notify "${SUMMARY}; SHUTDOWN FAILED, VM still running" urgent
    fi
else
    notify "$SUMMARY" "$([ "$CODE" -eq 0 ] && echo default || echo high)"
fi
exit "$CODE"
