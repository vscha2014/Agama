#!/bin/bash
# launch_multi_model.sh — one-shot orchestrator: N DIFFERENT halo models in
# parallel containers, one integration + the standard Upsilon search each
# (run_single_model.py --protocols exp). No protocol comparison, no repeats,
# no search. Experimental harness, NOT production.
#
# Models come from a file (--models=FILE): one history row per container in any
# layout run_single_model.py understands (out_*/4Ups*, J_factor_*, Jcomputed_*),
# optional trailing "# label" (stored as reference.source in the report).
# Row penalty/Upsilon are the reference; parameters go to the runner explicitly,
# so no result numbers live in this repository.
#
# Memory protections are the ones of launch_single_model.sh (host swap,
# swapless container limit, resource sampler, MemAvailable/PSI and no-progress
# watchdogs, emergency log upload, verified shutdown); the lock
# orblib_single/.launcher.lock is shared with it.
#
# After the containers:
#   * summary = concatenated per-model text reports;
#   * history: out_<host>_d1_nb250_gh0_ser0_multi<TS>m<i>.txt uploaded AS-IS to
#     galAgama/ (unique names in the d1 pool); reports, logs, models file →
#     galAgama/single_model/<RUN_TAG>_<TS>/;
#   * every model's orbit library → shared catalog galAgama/orblib via
#     `orblib_storage.py prepare --resume` (size+MD5, receipt; the managed local
#     copy is removed after verification); skipped if the name has a receipt;
#   * VM shutdown unless --no-shutdown.
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="agama:latest"
N_VCPU=$(nproc)
RUNNER="run_single_model.py"
CALC_SCRIPT="Fornax_P21_PCA_w3Sersic_orblib_exp.py"

MODELS_FILE=""
DO_SHUTDOWN=1
DO_UPLOAD=1
PREFLIGHT=0

for arg in "$@"; do
    case $arg in
        --help|-h)
            printf '%s\n' \
                'Использование: bash launch_multi_model.sh --models=FILE [опции]' \
                '  --models=FILE    Файл моделей: по одной строке истории на контейнер' \
                '                   (форматы out_*/4Ups*, J_factor_*, Jcomputed_*; "# метка" в конце строки' \
                '                   допускается). incl и параметры берутся из строки, её penalty/Upsilon —' \
                '                   опорные. Одна интеграция + штатный поиск Upsilon (протокол exp).' \
                '  --no-shutdown    Не выключать VM после завершения.' \
                '  --no-upload      Не выгружать .npz в общий каталог galAgama/orblib.' \
                '  --preflight      Все модели без интегрирования: импорт, геометрия, имя библиотеки;' \
                '                   без выгрузки истории/.npz и без выключения.' \
                'Конфигурация d1_nb250_gh0_ser0. Число моделей ≤ nproc; ядра делятся поровну.' \
                'Память: ORBLIB_SWAPFILE=16G, ORBLIB_MEM_LIMIT, ORBLIB_SAVE_SLOTS=2, ORBLIB_MIN_AVAIL_MB=2048,' \
                '  ORBLIB_PSI_LIMIT=20, ORBLIB_PSI_SAMPLES=3, ORBLIB_STALL_TIMEOUT=3600, ORBLIB_STOP_GRACE=1800.' \
                'Диск: SINGLE_BYTES_PER_REPEAT=2000000000 на модель сверх ORBLIB_RESERVE_BYTES=2000000000.'
            exit 0
            ;;
        --models=*)      MODELS_FILE="${arg#*=}" ;;
        --no-shutdown)   DO_SHUTDOWN=0      ;;
        --no-upload)     DO_UPLOAD=0        ;;
        --preflight)     PREFLIGHT=1        ;;
        *)
            echo "ОШИБКА: неизвестный аргумент '$arg' (см. --help)" >&2
            exit 2
            ;;
    esac
done
if [ -z "$MODELS_FILE" ] || [ ! -f "$MODELS_FILE" ]; then
    echo "ОШИБКА: нужен существующий --models=FILE (получено: '${MODELS_FILE}')" >&2
    exit 2
fi
MODELS_FILE="$(cd "$(dirname "$MODELS_FILE")" && pwd)/$(basename "$MODELS_FILE")"
if ! _rows="$(python3 "${WORK_DIR}/${RUNNER}" --list-models "$MODELS_FILE")"; then
    echo "ОШИБКА: в ${MODELS_FILE} нет корректных строк моделей (или повторы параметров)" >&2
    exit 2
fi
mapfile -t MODEL_ROWS <<< "$_rows"
NMODELS=${#MODEL_ROWS[@]}
if [ "$PREFLIGHT" -eq 1 ]; then
    DO_UPLOAD=0
    DO_SHUTDOWN=0
fi

NTFY_TOPIC="${NTFY_TOPIC:-GalaxySchwarzschildFornax}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
RCLONE_REMOTE="${RCLONE_REMOTE:-yandex}"
RCLONE_CONF_DIR="${HOME}/.config/rclone"
HOSTNAME_ENV="$(hostname)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_DIR="galAgama"
ORBLIB_REMOTE_DIR="${REMOTE_DIR}/orblib"
ORBLIB_UPLOAD_ATTEMPTS="${ORBLIB_UPLOAD_ATTEMPTS:-3}"
ORBLIB_FILE_TIMEOUT="${ORBLIB_FILE_TIMEOUT:-900}"
ORBLIB_RESERVE_BYTES="${ORBLIB_RESERVE_BYTES:-2000000000}"
SINGLE_BYTES_PER_REPEAT="${SINGLE_BYTES_PER_REPEAT:-2000000000}"
ORBLIB_SWAPFILE="${ORBLIB_SWAPFILE:-16G}"
ORBLIB_SAVE_SLOTS="${ORBLIB_SAVE_SLOTS:-2}"
ORBLIB_MEM_LIMIT="${ORBLIB_MEM_LIMIT:-}"
ORBLIB_MIN_AVAIL_MB="${ORBLIB_MIN_AVAIL_MB:-2048}"
ORBLIB_PSI_LIMIT="${ORBLIB_PSI_LIMIT:-20}"
ORBLIB_PSI_SAMPLES="${ORBLIB_PSI_SAMPLES:-3}"
ORBLIB_STALL_TIMEOUT="${ORBLIB_STALL_TIMEOUT:-3600}"
ORBLIB_STOP_GRACE="${ORBLIB_STOP_GRACE:-1800}"
ORBLIB_MONITOR_INTERVAL="${ORBLIB_MONITOR_INTERVAL:-60}"
ORBLIB_WATCH_INTERVAL="${ORBLIB_WATCH_INTERVAL:-10}"
ORBLIB_MEMINFO="${ORBLIB_MEMINFO:-/proc/meminfo}"
ORBLIB_PSI_PATH="${ORBLIB_PSI_PATH:-/proc/pressure/memory}"

# Конфигурация штатного свободного поиска (как в проде: удвоение, n_bin=250).
EXP_ID="d1_nb250_gh0_ser0"
RUN_TAG="multi_${EXP_ID}_n${NMODELS}"
SINGLE_ROOT="${WORK_DIR}/orblib_single"
RESULTS_REMOTE="${REMOTE_DIR}/single_model/${RUN_TAG}_${TIMESTAMP}"
LOGFILE="${WORK_DIR}/launch_${RUN_TAG}_${TIMESTAMP}.log"
MONITOR_LOG="${WORK_DIR}/monitor_${RUN_TAG}_${TIMESTAMP}.log"
SUMMARY="${WORK_DIR}/summary_multi_${TIMESTAMP}.txt"

if [ "$NMODELS" -gt "$N_VCPU" ]; then
    echo "ОШИБКА: моделей ${NMODELS} > nproc=${N_VCPU}" >&2
    exit 1
fi

declare -a SUFFIXES ORBLIB_DIRS REPORTS CPU_RANGES THREADS_ARR NAMES
declare -a M_INCL M_Q M_GH M_RH M_RHO0 M_PEN M_UPS M_SRC
_base=$((N_VCPU / NMODELS))
_rem=$((N_VCPU % NMODELS))
_start=0
for ((i = 0; i < NMODELS; i++)); do
    IFS=$'\t' read -r M_INCL[i] M_Q[i] M_GH[i] M_RH[i] M_RHO0[i] M_PEN[i] M_UPS[i] M_SRC[i] \
        <<< "${MODEL_ROWS[$i]}"
    _size=$_base
    [ "$i" -lt "$_rem" ] && _size=$((_base + 1))
    _end=$((_start + _size - 1))
    SUFFIXES[i]="multi${TIMESTAMP}m${i}"
    ORBLIB_DIRS[i]="orblib_single/${TIMESTAMP}_m${i}"
    REPORTS[i]="report_multi_${TIMESTAMP}_m${i}.json"
    NAMES[i]="agama_multi_${HOSTNAME_ENV}_${TIMESTAMP}_m${i}"
    CPU_RANGES[i]="${_start}-${_end}"
    THREADS_ARR[i]=$_size
    _start=$((_end + 1))
done

MAIN_PID=$$
SHUTDOWN_DONE=0
MONITOR_PID=""
EMERGENCY_DONE=0
CONTAINERS_STARTED=0
declare -a PIDS=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

notify() {
    local msg="$1"
    local priority="${2:-default}"
    if curl -s -f --max-time 10 \
        -H "Title: MultiModel ${HOSTNAME_ENV}" \
        -H "Priority: ${priority}" \
        -d "$msg" \
        "${NTFY_SERVER}/${NTFY_TOPIC}" > /dev/null 2>>"$LOGFILE"; then
        :
    else
        log "  ~ ntfy: не удалось отправить уведомление ('${msg}')"
    fi
}

upload_file() {
    local src="$1"
    local dest="$2"
    [ -f "$src" ] || return 0
    if rclone copyto "$src" "${RCLONE_REMOTE}:${dest}" \
            --config "${RCLONE_CONF_DIR}/rclone.conf" \
            --timeout 30s --contimeout 10s --max-duration 120s \
            --cutoff-mode HARD --retries 1 --low-level-retries 1 \
            --stats-one-line 2>>"$LOGFILE"; then
        log "    ✓ $(basename "$src") → ${dest}"
        return 0
    fi
    log "    ✗ $(basename "$src") → ${dest}"
    return 1
}

schedule_shutdown() {
    local delay="$1"
    local reason="$2"
    if [ "$SHUTDOWN_DONE" -eq 1 ]; then
        return 0
    fi
    SHUTDOWN_DONE=1
    upload_file "$LOGFILE" "${RESULTS_REMOTE}/$(basename "$LOGFILE")" || true
    if [ "$DO_SHUTDOWN" -ne 1 ]; then
        log "Выключение пропущено (--no-shutdown/--preflight): ${reason}"
        return 0
    fi
    log "Выключение VM через ${delay} мин (${reason})..."
    local code=0
    sudo shutdown -h +"$delay" "AGAMA multi_model: ${reason}" || code=$?
    [ "$code" -eq 0 ] && return 0
    log "  ✗ НЕ УДАЛОСЬ запланировать выключение (код ${code}): VM продолжит работу — выключите вручную"
    notify "VM ${HOSTNAME_ENV}: sudo shutdown провалился (код ${code}); выключите вручную. Причина: ${reason}" "urgent"
    return 0
}

# SIGTERM → STOP в хранилище расчётного модуля → частичный отчёт (код 75)
# после выхода из текущего C-вызова; затем ограниченное ожидание и kill.
stop_containers() {
    [ "$CONTAINERS_STARTED" -eq 1 ] || return 0
    docker stop -t "$ORBLIB_STOP_GRACE" "${NAMES[@]}" >>"$LOGFILE" 2>&1 || true
    local waited=0 limit=$((ORBLIB_STOP_GRACE + 300)) pid running
    while [ "$waited" -lt "$limit" ]; do
        running=0
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && running=1
        done
        [ "$running" -eq 1 ] || break
        sleep 5
        waited=$((waited + 5))
    done
    docker kill "${NAMES[@]}" >>"$LOGFILE" 2>&1 || true
}

emergency_upload() {
    [ "$EMERGENCY_DONE" -eq 1 ] && return 0
    EMERGENCY_DONE=1
    local dest="${REMOTE_DIR}/galaxy_results_emergency/${RUN_TAG}_${TIMESTAMP}"
    log "  Аварийная выгрузка логов в ${RCLONE_REMOTE}:${dest}"
    local f
    for f in "$LOGFILE" "$MONITOR_LOG" "$MODELS_FILE" \
             "${WORK_DIR}/dockerlog_${RUN_TAG}_m"*"_${TIMESTAMP}.log" \
             "${WORK_DIR}/report_multi_${TIMESTAMP}_m"* \
             "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_multi${TIMESTAMP}m"*.txt
    do
        upload_file "$f" "${dest}/$(basename "$f")" || true
    done
}

on_exit() {
    local code=$?
    [ "${BASHPID:-$$}" = "$MAIN_PID" ] || return 0
    if [ "$SHUTDOWN_DONE" -eq 0 ] && [ "$code" -ne 0 ]; then
        log "Аварийное завершение скрипта (код ${code})"
        notify "Аварийное завершение multi_model на ${HOSTNAME_ENV} (код ${code})" "urgent"
        stop_containers
        stop_resource_sampler
        emergency_upload
        schedule_shutdown 1 "остановка после ошибки (код ${code})"
    fi
    stop_resource_sampler
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

die() {
    log "ОШИБКА: $*"
    notify "ОШИБКА multi_model на ${HOSTNAME_ENV}: $*" "urgent"
    exit 1
}

# --- Защита памяти: копия логики launch_single_model.sh (см. orblib_exp.md §3a) ---
ensure_host_swap() {
    if [ "$ORBLIB_SWAPFILE" = "0" ]; then
        log "  Swapfile отключён (ORBLIB_SWAPFILE=0)"
        return 0
    fi
    if [ -n "$(swapon --show --noheadings 2>/dev/null || true)" ]; then
        log "  Swap уже активен: $(swapon --show --noheadings 2>/dev/null | tr '\n' ' ')"
        return 0
    fi
    log "  Создание swapfile /swapfile размером ${ORBLIB_SWAPFILE} (защита host-процессов)"
    if sudo fallocate -l "$ORBLIB_SWAPFILE" /swapfile 2>>"$LOGFILE" \
            && sudo chmod 600 /swapfile 2>>"$LOGFILE" \
            && sudo mkswap /swapfile >>"$LOGFILE" 2>&1 \
            && sudo swapon /swapfile 2>>"$LOGFILE"; then
        sudo sysctl -w vm.swappiness=10 >>"$LOGFILE" 2>&1 || true
        log "  ✓ Swap включён, vm.swappiness=10"
    else
        log "  ~ ВНИМАНИЕ: swapfile не создан (нет sudo/места?) — host без защиты от давления памяти"
        sudo rm -f /swapfile 2>>"$LOGFILE" || true
    fi
}

check_swap_limit_support() {
    local supported
    supported=$(docker info --format '{{.SwapLimit}}' 2>/dev/null || true)
    if [ "$supported" = "true" ]; then
        log "  ✓ Учёт swap в cgroup доступен: --memory-swap будет соблюдён"
        return 0
    fi
    if [ "$supported" != "false" ] \
       && ! docker info 2>/dev/null | grep -qi 'No swap limit support'; then
        log "  ~ Поддержку swap-лимита определить не удалось; считаем её возможной"
        return 0
    fi
    log "  ~ ВНИМАНИЕ: cgroup без учёта swap — docker проигнорирует --memory-swap."
    sudo sysctl -w vm.swappiness=1 >>"$LOGFILE" 2>&1 || true
    [ "$ORBLIB_STALL_TIMEOUT" -gt 900 ] && ORBLIB_STALL_TIMEOUT=900
    return 0
}

container_mem_limit() {
    if [ -n "$ORBLIB_MEM_LIMIT" ]; then
        printf '%s' "$ORBLIB_MEM_LIMIT"
        return 0
    fi
    local total_kb
    total_kb=$(awk '/^MemTotal:/ {print $2}' "$ORBLIB_MEMINFO" 2>/dev/null || echo 0)
    local limit=$(( (total_kb * 1024 - 4 * 1024 * 1024 * 1024) / NMODELS ))
    [ "$limit" -lt $((2 * 1024 * 1024 * 1024)) ] && limit=$((2 * 1024 * 1024 * 1024))
    printf '%s' "$limit"
}

mem_available_mb() {
    local value
    value=$(awk '/^MemAvailable:/ {printf "%d", $2 / 1024}' "$ORBLIB_MEMINFO" 2>/dev/null || true)
    printf '%s' "${value:-999999}"
}

mem_pressure_avg60() {
    awk '/^some /{for(i=1;i<=NF;i++) if($i ~ /^avg60=/){sub("avg60=","",$i); print $i; exit}}' \
        "$ORBLIB_PSI_PATH" 2>/dev/null || true
}

start_resource_sampler() {
    (
        _sleep_pid=''
        trap 'kill "$_sleep_pid" 2>/dev/null; exit 0' TERM
        while :; do
            {
                echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
                free -m 2>/dev/null || true
                cat "$ORBLIB_PSI_PATH" 2>/dev/null || echo 'pressure: n/a'
                docker stats --no-stream --format '{{.Name}} {{.MemUsage}} {{.CPUPerc}}' 2>/dev/null || true
                df -h "$SINGLE_ROOT" 2>/dev/null || true
            } >> "$MONITOR_LOG" 2>&1
            sleep "$ORBLIB_MONITOR_INTERVAL" &
            _sleep_pid=$!
            wait "$_sleep_pid" || true
        done
    ) </dev/null >/dev/null 2>&1 &
    MONITOR_PID=$!
    log "  Сэмплер ресурсов: PID ${MONITOR_PID} → $(basename "$MONITOR_LOG")"
}

stop_resource_sampler() {
    [ -n "$MONITOR_PID" ] || return 0
    kill "$MONITOR_PID" 2>/dev/null || true
    MONITOR_PID=""
}

dump_diagnostics() {
    log "  --- Диагностика ($1) ---"
    {
        free -m 2>/dev/null || true
        cat /proc/pressure/memory 2>/dev/null || true
        ps -eo rss,pid,comm --sort=-rss 2>/dev/null | head -11 || true
        docker stats --no-stream 2>/dev/null || true
        df -h "$SINGLE_ROOT" 2>/dev/null || true
    } >> "$LOGFILE" 2>&1
}

handle_stall() {
    local trigger="$1"
    log "СРАБОТАЛ ВАТЧДОГ: ${trigger}"
    dump_diagnostics "$trigger"
    notify "Ватчдог multi_model на ${HOSTNAME_ENV}: ${trigger}" "urgent"
    stop_containers
    stop_resource_sampler
    emergency_upload
    notify "multi_model остановлен ватчдогом на ${HOSTNAME_ENV}; логи в galaxy_results_emergency/${RUN_TAG}_${TIMESTAMP}" "urgent"
    schedule_shutdown 1 "ватчдог: ${trigger}"
}

storage_cli() {
    local root="$1"
    shift
    # --reserve-bytes 0: выгрузка идёт после расчёта, резерв под checkpoint/логи не нужен.
    python3 "${WORK_DIR}/orblib_storage.py" "$@" \
        --root "$root" --remote "${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}" \
        --config "${RCLONE_CONF_DIR}/rclone.conf" \
        --attempts "$ORBLIB_UPLOAD_ATTEMPTS" --timeout "$ORBLIB_FILE_TIMEOUT" \
        --reserve-bytes 0
}

report_field() {
    [ -f "$1" ] || return 0
    python3 -c 'import json, sys
r = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    r = (r or {}).get(key)
print("" if r is None else r)' "$1" "$2" 2>/dev/null || true
}

run_container() {
    local i="$1"
    local sfx="${SUFFIXES[$i]}"
    local threads="${THREADS_ARR[$i]}"
    local proc_log="${WORK_DIR}/dockerlog_${RUN_TAG}_m${i}_${TIMESTAMP}.log"
    local extra=()
    [ "$PREFLIGHT" -eq 1 ] && extra+=(--preflight)
    log "  Контейнер m${i}: CPU=${CPU_RANGES[$i]} incl=${M_INCL[$i]} suffix=${sfx} orblib=${ORBLIB_DIRS[$i]}"
    set +e
    # --memory-swap == --memory: swap контейнеру запрещён, превышение → OOM-kill (137).
    # rclone-конфиг НЕ монтируется: раннер не выполняет облачных операций.
    docker run --rm \
        --name "${NAMES[$i]}" \
        --cpuset-cpus="${CPU_RANGES[$i]}" \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        -v "${WORK_DIR}:/workspace" \
        -e ORBLIB_RESERVE_BYTES="${ORBLIB_RESERVE_BYTES}" \
        -e ORBLIB_SAVE_SLOTS="${ORBLIB_SAVE_SLOTS}" \
        -e HOSTNAME_SUFFIX="${HOSTNAME_ENV}" \
        -e OMP_NUM_THREADS="${threads}" \
        -e OMP_PROC_BIND="close" \
        -e OMP_PLACES="cores" \
        -e MKL_NUM_THREADS="${threads}" \
        -e OPENBLAS_NUM_THREADS="${threads}" \
        -e NUMEXPR_NUM_THREADS="${threads}" \
        -w /workspace \
        "${IMAGE}" \
        python3 -u "/workspace/${RUNNER}" \
            --incl "${M_INCL[$i]}" \
            --Q "${M_Q[$i]}" --gh "${M_GH[$i]}" --rh "${M_RH[$i]}" --rho0 "${M_RHO0[$i]}" \
            --ref-penalty "${M_PEN[$i]}" --ref-upsilon "${M_UPS[$i]}" \
            --ref-source "${M_SRC[$i]}" \
            --protocols exp \
            --suffix "${sfx}" \
            --orblib-dir "/workspace/${ORBLIB_DIRS[$i]}" \
            --report "/workspace/${REPORTS[$i]}" \
            --n_threads "${threads}" \
            ${extra[@]+"${extra[@]}"} \
        2>&1 | tee "$proc_log"
    local -a codes=("${PIPESTATUS[@]}")
    set -e
    local code="${codes[0]}"
    if [ "${codes[1]}" -ne 0 ] && [ "$code" -eq 0 ]; then
        code=74
    fi
    if [ "$code" -eq 0 ]; then
        log "  ✓ m${i} завершён успешно"
    else
        log "  ✗ m${i} завершён с кодом ${code}"
        [ "$code" -eq 137 ] && log "  ! m${i} убит по лимиту памяти контейнера (OOM-kill, код 137); отчёт может быть частичным"
        notify "multi_model m${i} (${HOSTNAME_ENV}) завершился с кодом ${code}" "high"
    fi
    return "$code"
}

# Библиотека модели i → общий каталог (как r0 в launch_single_model.sh). 0 = доставлена или уже в каталоге.
deliver_library() {
    local i="$1"
    local status name root existing lsf_rc
    status="$(report_field "${WORK_DIR}/${REPORTS[$i]}" status)"
    name="$(report_field "${WORK_DIR}/${REPORTS[$i]}" orblib.name)"
    root="${WORK_DIR}/${ORBLIB_DIRS[$i]}"
    if [ "$status" != "ok" ] || [ -z "$name" ] || [ ! -f "${root}/${name}" ]; then
        log "  m${i}: нет проверенной библиотеки (status='${status}', name='${name}') — выгрузка пропущена"
        return 1
    fi
    set +e
    existing=$(rclone lsf "${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}/catalog" --files-only \
                   --include "${name}.json" --config "${RCLONE_CONF_DIR}/rclone.conf" \
                   --timeout 30s --contimeout 10s 2>>"$LOGFILE")
    lsf_rc=$?
    set -e
    if [ "$lsf_rc" -ne 0 ]; then
        log "  ✗ m${i}: каталог недоступен (rclone lsf код ${lsf_rc}) — ${name} оставлена на VM"
        notify "multi_model: каталог недоступен, ${name} оставлена на VM ${HOSTNAME_ENV}" "urgent"
        return 1
    elif [ -n "$existing" ]; then
        log "  ~ m${i}: ${name} уже в каталоге (archive-first-wins) — новая реализация оставлена на VM"
    elif storage_cli "$root" prepare --resume >>"$LOGFILE" 2>&1 \
            && storage_cli "$root" check >>"$LOGFILE" 2>&1; then
        log "  ✓ m${i}: ${name} доставлена в ${ORBLIB_REMOTE_DIR} (size+MD5, receipt); локальная копия удалена"
    else
        log "  ✗ m${i}: доставка ${name} не завершена; файл и очередь остаются в ${root}"
        notify "multi_model: доставка ${name} не завершена на ${HOSTNAME_ENV}" "urgent"
        return 1
    fi
    return 0
}

# ==============================================================
# ПРОВЕРКИ
# ==============================================================
log "======================================================"
log "AGAMA multi-model run (Docker)"
log "  hostname         = $HOSTNAME_ENV"
log "  runner           = $RUNNER (протокол exp)"
log "  EXP_ID           = $EXP_ID"
log "  файл моделей     = $MODELS_FILE"
log "  моделей          = $NMODELS (preflight=${PREFLIGHT})"
for ((i = 0; i < NMODELS; i++)); do
    log "    m${i}: incl=${M_INCL[$i]} Q=${M_Q[$i]} gh=${M_GH[$i]} rh=${M_RH[$i]} rho0=${M_RHO0[$i]}" \
        "| ref penalty=${M_PEN[$i]} Upsilon=${M_UPS[$i]} (${M_SRC[$i]})"
done
log "  orblib           = ${SINGLE_ROOT}/${TIMESTAMP}_m*"
log "  выгрузка .npz    = $DO_UPLOAD → ${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}"
log "  результаты       = ${RCLONE_REMOTE}:${RESULTS_REMOTE}"
log "  shutdown         = $DO_SHUTDOWN"
log "  Раскладка ядер   = ${CPU_RANGES[*]}"
log "======================================================"

[ -f "${RCLONE_CONF_DIR}/rclone.conf" ] \
    || die "rclone не настроен: ${RCLONE_CONF_DIR}/rclone.conf"
for f in "$RUNNER" "$CALC_SCRIPT" table3.dat orblib_storage.py; do
    [ -f "${WORK_DIR}/${f}" ] || die "не найден ${WORK_DIR}/${f}"
done
mkdir -p "$SINGLE_ROOT"
exec 201>"${SINGLE_ROOT}/.launcher.lock"
if ! flock -n 201; then
    SHUTDOWN_DONE=1
    log "Уже работает launch_single_model/launch_multi_model; запуск отменён без shutdown"
    exit 1
fi
# Параллельный штатный поиск делил бы память VM с этими контейнерами.
if [ -e "${WORK_DIR}/orblib/.launcher.lock" ]; then
    exec 202>>"${WORK_DIR}/orblib/.launcher.lock"
    if ! flock -n 202; then
        SHUTDOWN_DONE=1
        log "Работает launch_orblib_exp.sh (занят orblib/.launcher.lock); запуск отменён без shutdown"
        exit 1
    fi
    exec 202>&-
fi
avail_bytes=$(df --output=avail -B1 "$SINGLE_ROOT" 2>/dev/null | tail -1 | tr -d ' ' || echo 0)
need_bytes=$((NMODELS * SINGLE_BYTES_PER_REPEAT + ORBLIB_RESERVE_BYTES))
[ "${avail_bytes:-0}" -ge "$need_bytes" ] \
    || die "мало места: доступно ${avail_bytes} байт, нужно ${need_bytes}"
docker image inspect "$IMAGE" > /dev/null 2>&1 \
    || die "Docker-образ $IMAGE не найден"

notify "Старт multi_model на ${HOSTNAME_ENV}: моделей=${NMODELS}, preflight=${PREFLIGHT}"

# ==============================================================
# ШАГ 0: ЗАЩИТА ПАМЯТИ
# ==============================================================
log ""
log "ШАГ 0: Защита памяти"
ensure_host_swap
check_swap_limit_support
MEM_LIMIT="$(container_mem_limit)"
log "  Лимит памяти контейнера: ${MEM_LIMIT} байт (~$((MEM_LIMIT / 1024 / 1024 / 1024)) GiB) × ${NMODELS}"

# ==============================================================
# ШАГ 1: ЗАПУСК КОНТЕЙНЕРОВ
# ==============================================================
log ""
log "ШАГ 1: Запуск ${NMODELS} контейнеров"
start_resource_sampler
CONTAINERS_STARTED=1
for ((i = 0; i < NMODELS; i++)); do
    run_container "$i" &
    PIDS[i]=$!
    log "  PID ${PIDS[$i]} → m${i}"
done

# ==============================================================
# ШАГ 2: ОЖИДАНИЕ С ВАТЧДОГАМИ (как в launch_orblib_exp.sh)
# ==============================================================
log ""
log "ШАГ 2: Ожидание завершения"
PSI_HITS=0
LAST_PROGRESS=$(date +%s)
LAST_MTIME=0
WATCH_TICK=0
while :; do
    running=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then running=1; fi
    done
    [ "$running" -eq 1 ] || break
    if [ $((WATCH_TICK % ORBLIB_WATCH_INTERVAL)) -eq 0 ]; then
        avail=$(mem_available_mb)
        if [ "$avail" -lt "$ORBLIB_MIN_AVAIL_MB" ]; then
            handle_stall "MemAvailable=${avail} MB < ${ORBLIB_MIN_AVAIL_MB} MB"
            exit 75
        fi
        psi=$(mem_pressure_avg60)
        if [ -n "$psi" ] \
           && awk -v v="$psi" -v lim="$ORBLIB_PSI_LIMIT" 'BEGIN{exit !(v > lim)}'; then
            PSI_HITS=$((PSI_HITS + 1))
            log "  ~ Давление памяти: some avg60=${psi}% (${PSI_HITS}/${ORBLIB_PSI_SAMPLES})"
            if [ "$PSI_HITS" -ge "$ORBLIB_PSI_SAMPLES" ]; then
                handle_stall "memory pressure some avg60=${psi}% > ${ORBLIB_PSI_LIMIT}%"
                exit 75
            fi
        else
            PSI_HITS=0
        fi
        now=$(date +%s)
        newest=$(find "$WORK_DIR" "$SINGLE_ROOT" -maxdepth 2 \
                     \( -name "dockerlog_${RUN_TAG}_m*_${TIMESTAMP}.log" \
                        -o -name "report_multi_${TIMESTAMP}_m*" \
                        -o -name "out_${HOSTNAME_ENV}_${EXP_ID}_multi${TIMESTAMP}m*.txt" \
                        -o -path "*/${TIMESTAMP}_m*/*.npz*" \) \
                     -printf '%T@\n' 2>/dev/null \
                  | sort -n | tail -1 | cut -d. -f1)
        if [ -n "${newest:-}" ] && [ "$newest" -gt "$LAST_MTIME" ]; then
            LAST_MTIME="$newest"
            LAST_PROGRESS="$now"
        fi
        if [ $((now - LAST_PROGRESS)) -ge "$ORBLIB_STALL_TIMEOUT" ]; then
            handle_stall "no progress for $((now - LAST_PROGRESS))s (limit ${ORBLIB_STALL_TIMEOUT}s)"
            exit 75
        fi
    fi
    WATCH_TICK=$((WATCH_TICK + 1))
    sleep 1
done
stop_resource_sampler

FAILED=0
declare -a EXIT_CODES
for ((i = 0; i < NMODELS; i++)); do
    set +e; wait "${PIDS[$i]}"; EXIT_CODES[i]=$?; set -e
    [ "${EXIT_CODES[$i]}" -eq 0 ] || FAILED=$((FAILED + 1))
done
CONTAINERS_STARTED=0
log "  Успешно: $((NMODELS - FAILED))/${NMODELS}, ошибок: ${FAILED}"

# ==============================================================
# ШАГ 3: СВОДКА (отчёты моделей подряд)
# ==============================================================
log ""
log "ШАГ 3: Сводка отчётов"
{
    echo "multi-model summary: ${NMODELS} models, ${TIMESTAMP}, models file $(basename "$MODELS_FILE")"
    for ((i = 0; i < NMODELS; i++)); do
        txt="${WORK_DIR}/${REPORTS[$i]%.json}.txt"
        echo "--- m${i} (код ${EXIT_CODES[$i]})"
        if [ -f "$txt" ]; then cat "$txt"; else echo "  нет отчёта"; fi
    done
} > "$SUMMARY"
tee -a "$LOGFILE" < "$SUMMARY"

FINAL_RC=0
[ "$FAILED" -eq 0 ] || FINAL_RC=1

# ==============================================================
# ШАГ 4: ВЫГРУЗКА ИСТОРИИ И РЕЗУЛЬТАТОВ
# ==============================================================
log ""
log "ШАГ 4: Выгрузка истории и отчётов"
if [ "$PREFLIGHT" -eq 0 ]; then
    for ((i = 0; i < NMODELS; i++)); do
        pool="${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_${SUFFIXES[$i]}.txt"
        # Уникальное имя в пуле d1: без слияния в файл хоста, ничего не перезаписывается.
        upload_file "$pool" "${REMOTE_DIR}/$(basename "$pool")" || FINAL_RC=1
    done
fi
for f in "${WORK_DIR}/report_multi_${TIMESTAMP}_m"* \
         "$SUMMARY" "$MONITOR_LOG" "$MODELS_FILE" \
         "${WORK_DIR}/dockerlog_${RUN_TAG}_m"*"_${TIMESTAMP}.log"
do
    [ -f "$f" ] || continue
    upload_file "$f" "${RESULTS_REMOTE}/$(basename "$f")" || true
done

# ==============================================================
# ШАГ 5: БИБЛИОТЕКИ → ОБЩИЙ КАТАЛОГ
# ==============================================================
log ""
log "ШАГ 5: Библиотеки орбит"
if [ "$DO_UPLOAD" -ne 1 ]; then
    log "  Выгрузка отключена (--no-upload/--preflight)"
else
    for ((i = 0; i < NMODELS; i++)); do
        deliver_library "$i" || FINAL_RC=1
    done
fi

# ==============================================================
# ИТОГ И ВЫКЛЮЧЕНИЕ
# ==============================================================
log ""
log "======================================================"
log "ЗАВЕРШЕНО: $(date '+%Y-%m-%d %H:%M:%S')"
for ((i = 0; i < NMODELS; i++)); do
    log "    m${i}: код ${EXIT_CODES[$i]}"
done
log "  Итоговый код: ${FINAL_RC}"
log "======================================================"
notify "multi_model завершён на ${HOSTNAME_ENV}: ошибок ${FAILED}/${NMODELS}, код ${FINAL_RC}" \
    "$([ "$FINAL_RC" -eq 0 ] && echo high || echo urgent)"
if [ "$FINAL_RC" -eq 0 ]; then
    schedule_shutdown 1 "multi_model завершён успешно (${NMODELS} моделей)"
else
    schedule_shutdown 5 "multi_model завершён с ошибками (${NMODELS} моделей)"
fi
exit "$FINAL_RC"
