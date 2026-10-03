#!/bin/bash
# launch_single_model.sh — one-shot orchestrator for run_single_model.py
# (experimental single-model check, NOT production and NOT the search).
#
# Runs N independent realisations (default 4) of ONE halo model in parallel
# containers with the 2026-10-01 memory protections copied from
# launch_orblib_exp.sh (host swap, swapless container limit, resource sampler,
# MemAvailable/PSI and no-progress watchdogs, emergency log upload, verified
# shutdown). Each realisation has its own suffix and orbit-library directory
# orblib_single/<TS>_r<i>: the library name depends only on the parameters, so
# shared directories would collide.
#
# After the containers:
#   * summary of the JSON reports (host python3, standard library only);
#   * history: out_<host>_d1_nb250_gh0_ser0_single<TS>r<i>.txt (the exp row
#     only) uploaded AS-IS to galAgama/ — unique names in the d1 pool glob, no
#     merge into the per-host file; reports, side files, logs →
#     galAgama/single_model/<RUN_TAG>_<TS>/;
#   * orbit library of r0 → shared catalog galAgama/orblib via
#     `orblib_storage.py prepare --resume` (size+MD5, receipt; the managed local
#     copy is removed after verification). Skipped if the name already has a
#     receipt. r1..r(N-1) carry the same name and stay on the VM;
#   * VM shutdown unless --no-shutdown.
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="agama:latest"
N_VCPU=$(nproc)
RUNNER="run_single_model.py"
CALC_SCRIPT="Fornax_P21_PCA_w3Sersic_orblib_exp.py"

INCL="90.0"
REPEATS=4
DO_SHUTDOWN=1
DO_UPLOAD=1
PREFLIGHT=0
declare -a RUNNER_ARGS=()

for arg in "$@"; do
    case $arg in
        --help|-h)
            printf '%s\n' \
                'Использование: bash launch_single_model.sh [--repeats=4] [--incl=90.0] [опции]' \
                '  --repeats=N      Число независимых реализаций (контейнеров), по умолчанию 4.' \
                '  --no-shutdown    Не выключать VM после завершения.' \
                '  --no-upload      Не выгружать .npz r0 в общий каталог galAgama/orblib.' \
                '  --preflight      Один контейнер: геометрия, имя библиотеки, без расчёта,' \
                '                   без выгрузки и без выключения.' \
                '  --Q= --gh= --rh= --rho0= --ref-penalty= --ref-upsilon= --params-from=GLOB' \
                '  --protocols=exp,reuse,prod   Передаются в run_single_model.py.' \
                'По умолчанию: строка с минимальным penalty при --incl из 4UpsBoTorch_PCA_Sersic_*.txt' \
                '  (free-Q история прода) в каталоге запуска; конфигурация d1_nb250_gh0_ser0.' \
                'Память: ORBLIB_SWAPFILE=16G, ORBLIB_MEM_LIMIT, ORBLIB_SAVE_SLOTS=2, ORBLIB_MIN_AVAIL_MB=2048,' \
                '  ORBLIB_PSI_LIMIT=20, ORBLIB_PSI_SAMPLES=3, ORBLIB_STALL_TIMEOUT=3600, ORBLIB_STOP_GRACE=1800.' \
                'Диск: SINGLE_BYTES_PER_REPEAT=2000000000 сверх ORBLIB_RESERVE_BYTES=2000000000.'
            exit 0
            ;;
        --incl=*)        INCL="${arg#*=}"   ;;
        --repeats=*)     REPEATS="${arg#*=}" ;;
        --no-shutdown)   DO_SHUTDOWN=0      ;;
        --no-upload)     DO_UPLOAD=0        ;;
        --preflight)     PREFLIGHT=1        ;;
        --Q=*|--gh=*|--rh=*|--rho0=*|--ref-penalty=*|--ref-upsilon=*|--params-from=*|--protocols=*)
            RUNNER_ARGS+=("$arg") ;;
        *)
            echo "ОШИБКА: неизвестный аргумент '$arg' (см. --help)" >&2
            exit 2
            ;;
    esac
done
if [ "$PREFLIGHT" -eq 1 ]; then
    REPEATS=1
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
INCL_FMT="$(LC_ALL=C printf '%.1f' "$INCL")"
RUN_TAG="single_${EXP_ID}_i${INCL_FMT}"
SINGLE_ROOT="${WORK_DIR}/orblib_single"
RESULTS_REMOTE="${REMOTE_DIR}/single_model/${RUN_TAG}_${TIMESTAMP}"
LOGFILE="${WORK_DIR}/launch_${RUN_TAG}_${TIMESTAMP}.log"
MONITOR_LOG="${WORK_DIR}/monitor_${RUN_TAG}_${TIMESTAMP}.log"
SUMMARY="${WORK_DIR}/summary_single_${TIMESTAMP}.txt"
SUMMARY_JSON="${WORK_DIR}/summary_single_${TIMESTAMP}.json"

if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [ "$REPEATS" -lt 1 ] || [ "$REPEATS" -gt "$N_VCPU" ]; then
    echo "ОШИБКА: --repeats должно быть целым в [1, ${N_VCPU}] (получено: '$REPEATS')" >&2
    exit 1
fi

declare -a SUFFIXES ORBLIB_DIRS REPORTS CPU_RANGES THREADS_ARR NAMES
_base=$((N_VCPU / REPEATS))
_rem=$((N_VCPU % REPEATS))
_start=0
for ((i = 0; i < REPEATS; i++)); do
    _size=$_base
    [ "$i" -lt "$_rem" ] && _size=$((_base + 1))
    _end=$((_start + _size - 1))
    SUFFIXES[i]="single${TIMESTAMP}r${i}"
    ORBLIB_DIRS[i]="orblib_single/${TIMESTAMP}_r${i}"
    REPORTS[i]="report_single_${TIMESTAMP}_r${i}.json"
    NAMES[i]="agama_single_${HOSTNAME_ENV}_${TIMESTAMP}_r${i}"
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
        -H "Title: SingleModel ${HOSTNAME_ENV}" \
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
    sudo shutdown -h +"$delay" "AGAMA single_model: ${reason}" || code=$?
    [ "$code" -eq 0 ] && return 0
    log "  ✗ НЕ УДАЛОСЬ запланировать выключение (код ${code}): VM продолжит работу — выключите вручную"
    notify "VM ${HOSTNAME_ENV}: sudo shutdown провалился (код ${code}); выключите вручную. Причина: ${reason}" "urgent"
    return 0
}

# Остановка контейнеров: SIGTERM → обработчик расчётного модуля ставит STOP в
# своём хранилище, раннер дописывает частичный отчёт (код 75) после выхода из
# текущего C-вызова — отсюда grace-период; затем ограниченное ожидание и kill.
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
    for f in "$LOGFILE" "$MONITOR_LOG" \
             "${WORK_DIR}/dockerlog_${RUN_TAG}_r"*"_${TIMESTAMP}.log" \
             "${WORK_DIR}/report_single_${TIMESTAMP}_r"* \
             "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_single${TIMESTAMP}r"*.txt \
             "${WORK_DIR}/single_${HOSTNAME_ENV}_${EXP_ID}_single${TIMESTAMP}r"*.txt
    do
        upload_file "$f" "${dest}/$(basename "$f")" || true
    done
}

on_exit() {
    local code=$?
    [ "${BASHPID:-$$}" = "$MAIN_PID" ] || return 0
    if [ "$SHUTDOWN_DONE" -eq 0 ] && [ "$code" -ne 0 ]; then
        log "Аварийное завершение скрипта (код ${code})"
        notify "Аварийное завершение single_model на ${HOSTNAME_ENV} (код ${code})" "urgent"
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
    notify "ОШИБКА single_model на ${HOSTNAME_ENV}: $*" "urgent"
    exit 1
}

# --- Защита памяти: копия логики launch_orblib_exp.sh (см. orblib_exp.md §3a) ---
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
    local limit=$(( (total_kb * 1024 - 4 * 1024 * 1024 * 1024) / REPEATS ))
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
    notify "Ватчдог single_model на ${HOSTNAME_ENV}: ${trigger}" "urgent"
    stop_containers
    stop_resource_sampler
    emergency_upload
    notify "single_model остановлен ватчдогом на ${HOSTNAME_ENV}; логи в galaxy_results_emergency/${RUN_TAG}_${TIMESTAMP}" "urgent"
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
    local proc_log="${WORK_DIR}/dockerlog_${RUN_TAG}_r${i}_${TIMESTAMP}.log"
    local extra=()
    [ "$PREFLIGHT" -eq 1 ] && extra+=(--preflight)
    log "  Контейнер r${i}: CPU=${CPU_RANGES[$i]} suffix=${sfx} orblib=${ORBLIB_DIRS[$i]}"
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
            --incl "${INCL}" \
            --suffix "${sfx}" \
            --orblib-dir "/workspace/${ORBLIB_DIRS[$i]}" \
            --side-file "single_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt" \
            --report "/workspace/${REPORTS[$i]}" \
            --n_threads "${threads}" \
            ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"} \
            ${extra[@]+"${extra[@]}"} \
        2>&1 | tee "$proc_log"
    local -a codes=("${PIPESTATUS[@]}")
    set -e
    local code="${codes[0]}"
    if [ "${codes[1]}" -ne 0 ] && [ "$code" -eq 0 ]; then
        code=74
    fi
    if [ "$code" -eq 0 ]; then
        log "  ✓ r${i} завершён успешно"
    else
        log "  ✗ r${i} завершён с кодом ${code}"
        [ "$code" -eq 137 ] && log "  ! r${i} убит по лимиту памяти контейнера (OOM-kill, код 137); отчёт может быть частичным"
        notify "single_model r${i} (${HOSTNAME_ENV}) завершился с кодом ${code}" "high"
    fi
    return "$code"
}

# ==============================================================
# ПРОВЕРКИ
# ==============================================================
log "======================================================"
log "AGAMA single-model check (Docker)"
log "  hostname         = $HOSTNAME_ENV"
log "  runner           = $RUNNER"
log "  incl / EXP_ID    = $INCL / $EXP_ID"
log "  реализаций       = $REPEATS (preflight=${PREFLIGHT})"
log "  параметры        = ${RUNNER_ARGS[*]:-по умолчанию (min penalty при incl=${INCL} из 4UpsBoTorch_PCA_Sersic_*.txt)}"
log "  orblib           = ${SINGLE_ROOT}/${TIMESTAMP}_r*"
log "  выгрузка .npz r0 = $DO_UPLOAD → ${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}"
log "  результаты       = ${RCLONE_REMOTE}:${RESULTS_REMOTE}"
log "  shutdown         = $DO_SHUTDOWN"
log "  Раскладка ядер   = ${CPU_RANGES[*]}"
log "======================================================"

[ -f "${RCLONE_CONF_DIR}/rclone.conf" ] \
    || die "rclone не настроен: ${RCLONE_CONF_DIR}/rclone.conf"
for f in "$RUNNER" "$CALC_SCRIPT" table3.dat orblib_storage.py; do
    [ -f "${WORK_DIR}/${f}" ] || die "не найден ${WORK_DIR}/${f}"
done
_explicit=0
for a in ${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"}; do
    case "$a" in --Q=*|--gh=*|--rh=*|--rho0=*) _explicit=$((_explicit + 1)) ;; --params-from=*) _explicit=-9 ;; esac
done
if [ "$_explicit" -ge 0 ] && [ "$_explicit" -lt 4 ] \
        && ! compgen -G "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_*.txt" > /dev/null; then
    die "нет 4UpsBoTorch_PCA_Sersic_*.txt для параметров по умолчанию; задайте --Q= --gh= --rh= --rho0= или --params-from="
fi
mkdir -p "$SINGLE_ROOT"
exec 201>"${SINGLE_ROOT}/.launcher.lock"
if ! flock -n 201; then
    SHUTDOWN_DONE=1
    log "Другой launch_single_model уже работает; запуск отменён без shutdown"
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
need_bytes=$((REPEATS * SINGLE_BYTES_PER_REPEAT + ORBLIB_RESERVE_BYTES))
[ "${avail_bytes:-0}" -ge "$need_bytes" ] \
    || die "мало места: доступно ${avail_bytes} байт, нужно ${need_bytes}"
docker image inspect "$IMAGE" > /dev/null 2>&1 \
    || die "Docker-образ $IMAGE не найден"

notify "Старт single_model на ${HOSTNAME_ENV}: incl=${INCL}, реализаций=${REPEATS}, preflight=${PREFLIGHT}"

# ==============================================================
# ШАГ 0: ЗАЩИТА ПАМЯТИ
# ==============================================================
log ""
log "ШАГ 0: Защита памяти"
ensure_host_swap
check_swap_limit_support
MEM_LIMIT="$(container_mem_limit)"
log "  Лимит памяти контейнера: ${MEM_LIMIT} байт (~$((MEM_LIMIT / 1024 / 1024 / 1024)) GiB) × ${REPEATS}"

# ==============================================================
# ШАГ 1: ЗАПУСК КОНТЕЙНЕРОВ
# ==============================================================
log ""
log "ШАГ 1: Запуск ${REPEATS} контейнеров"
start_resource_sampler
CONTAINERS_STARTED=1
for ((i = 0; i < REPEATS; i++)); do
    run_container "$i" &
    PIDS[i]=$!
    log "  PID ${PIDS[$i]} → r${i}"
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
                     \( -name "dockerlog_${RUN_TAG}_r*_${TIMESTAMP}.log" \
                        -o -name "report_single_${TIMESTAMP}_r*" \
                        -o -name "out_${HOSTNAME_ENV}_${EXP_ID}_single${TIMESTAMP}r*.txt" \
                        -o -name "single_${HOSTNAME_ENV}_${EXP_ID}_single${TIMESTAMP}r*.txt" \
                        -o -path "*/${TIMESTAMP}_r*/*.npz*" \) \
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
for ((i = 0; i < REPEATS; i++)); do
    set +e; wait "${PIDS[$i]}"; EXIT_CODES[i]=$?; set -e
    [ "${EXIT_CODES[$i]}" -eq 0 ] || FAILED=$((FAILED + 1))
done
CONTAINERS_STARTED=0
log "  Успешно: $((REPEATS - FAILED))/${REPEATS}, ошибок: ${FAILED}"

# ==============================================================
# ШАГ 3: СВОДКА
# ==============================================================
log ""
log "ШАГ 3: Сводка отчётов"
declare -a EXISTING_REPORTS=()
for ((i = 0; i < REPEATS; i++)); do
    [ -f "${WORK_DIR}/${REPORTS[$i]}" ] && EXISTING_REPORTS+=("${WORK_DIR}/${REPORTS[$i]}")
done
if [ "${#EXISTING_REPORTS[@]}" -gt 0 ]; then
    python3 "${WORK_DIR}/${RUNNER}" --summarize "${EXISTING_REPORTS[@]}" \
        --summary-json "$SUMMARY_JSON" > "$SUMMARY" 2>>"$LOGFILE" || true
    tee -a "$LOGFILE" < "$SUMMARY"
else
    log "  Нет ни одного отчёта"
fi

FINAL_RC=0
[ "$FAILED" -eq 0 ] || FINAL_RC=1

# ==============================================================
# ШАГ 4: ВЫГРУЗКА ИСТОРИИ И РЕЗУЛЬТАТОВ
# ==============================================================
log ""
log "ШАГ 4: Выгрузка истории и отчётов"
if [ "$PREFLIGHT" -eq 0 ]; then
    for ((i = 0; i < REPEATS; i++)); do
        pool="${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_${SUFFIXES[$i]}.txt"
        # Уникальное имя в пуле d1: без слияния в файл хоста, ничего не перезаписывается.
        upload_file "$pool" "${REMOTE_DIR}/$(basename "$pool")" || FINAL_RC=1
    done
fi
for f in "${WORK_DIR}/report_single_${TIMESTAMP}_r"* \
         "${WORK_DIR}/single_${HOSTNAME_ENV}_${EXP_ID}_single${TIMESTAMP}r"*.txt \
         "$SUMMARY" "$SUMMARY_JSON" "$MONITOR_LOG" \
         "${WORK_DIR}/dockerlog_${RUN_TAG}_r"*"_${TIMESTAMP}.log"
do
    [ -f "$f" ] || continue
    upload_file "$f" "${RESULTS_REMOTE}/$(basename "$f")" || true
done

# ==============================================================
# ШАГ 5: БИБЛИОТЕКА r0 → ОБЩИЙ КАТАЛОГ
# ==============================================================
log ""
log "ШАГ 5: Библиотека орбит"
r0_status="$(report_field "${WORK_DIR}/${REPORTS[0]}" status)"
r0_name="$(report_field "${WORK_DIR}/${REPORTS[0]}" orblib.name)"
r0_root="${WORK_DIR}/${ORBLIB_DIRS[0]}"
if [ "$DO_UPLOAD" -ne 1 ]; then
    log "  Выгрузка отключена (--no-upload/--preflight)"
elif [ "$r0_status" != "ok" ] || [ -z "$r0_name" ] || [ ! -f "${r0_root}/${r0_name}" ]; then
    log "  r0 не дал проверенной библиотеки (status='${r0_status}', name='${r0_name}') — выгрузка пропущена"
    FINAL_RC=1
else
    set +e
    existing=$(rclone lsf "${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}/catalog" --files-only \
                   --include "${r0_name}.json" --config "${RCLONE_CONF_DIR}/rclone.conf" \
                   --timeout 30s --contimeout 10s 2>>"$LOGFILE")
    lsf_rc=$?
    set -e
    if [ "$lsf_rc" -ne 0 ]; then
        log "  ✗ Каталог недоступен (rclone lsf код ${lsf_rc}) — ${r0_name} оставлена на VM"
        notify "single_model: каталог недоступен, ${r0_name} оставлена на VM ${HOSTNAME_ENV}" "urgent"
        FINAL_RC=1
    elif [ -n "$existing" ]; then
        log "  ~ ${r0_name} уже в каталоге (archive-first-wins) — новая реализация оставлена на VM"
    elif storage_cli "$r0_root" prepare --resume >>"$LOGFILE" 2>&1 \
            && storage_cli "$r0_root" check >>"$LOGFILE" 2>&1; then
        log "  ✓ ${r0_name} доставлена в ${ORBLIB_REMOTE_DIR} (size+MD5, receipt); локальная копия удалена"
    else
        log "  ✗ Доставка ${r0_name} не завершена; файл и очередь остаются в ${r0_root}"
        notify "single_model: доставка ${r0_name} не завершена на ${HOSTNAME_ENV}" "urgent"
        FINAL_RC=1
    fi
fi
for ((i = 1; i < REPEATS; i++)); do
    for lib in "${WORK_DIR}/${ORBLIB_DIRS[$i]}"/*.npz; do
        [ -f "$lib" ] && log "  Оставлена на VM: ${ORBLIB_DIRS[$i]}/$(basename "$lib") ($(du -h "$lib" | cut -f1))"
    done
done

# ==============================================================
# ИТОГ И ВЫКЛЮЧЕНИЕ
# ==============================================================
log ""
log "======================================================"
log "ЗАВЕРШЕНО: $(date '+%Y-%m-%d %H:%M:%S')"
for ((i = 0; i < REPEATS; i++)); do
    log "    r${i}: код ${EXIT_CODES[$i]}"
done
log "  Итоговый код: ${FINAL_RC}"
log "======================================================"
notify "single_model завершён на ${HOSTNAME_ENV}: ошибок ${FAILED}/${REPEATS}, код ${FINAL_RC}" \
    "$([ "$FINAL_RC" -eq 0 ] && echo high || echo urgent)"
if [ "$FINAL_RC" -eq 0 ]; then
    schedule_shutdown 1 "single_model завершён успешно (incl=${INCL})"
else
    schedule_shutdown 5 "single_model завершён с ошибками (incl=${INCL})"
fi
exit "$FINAL_RC"
