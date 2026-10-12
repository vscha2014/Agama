#!/bin/bash
# launch_tycho.sh — run the experimental search (Fornax_P21_PCA_w3Sersic_orblib_exp.py)
# on a shared workstation (tycho) within a quarter of its resources.
#
# No docker, no swapfile, no sudo, no shutdown. The whole launcher re-executes
# itself inside ONE transient systemd user scope, so the workers, the history
# merge and rclone all share a single budget:
#   CPUQuota   = threads × 100 %   (default threads = nproc/4)
#   MemoryMax  = MemTotal/4,  MemoryHigh = 90 % of it,  MemorySwapMax = 0
#   IOWeight=10, nice -n 19, ionice -c 3, OOMPolicy=continue
# AGAMA gets an explicit --n_threads (the quota does not change the affinity
# mask, so the harness default of len(sched_getaffinity) would be 16 threads).
#
# Orbit libraries: --save-orblib --reuse-orblib, kept LOCALLY (no
# --stream-orblib, no cloud catalog). History: experimental d1 pool
# out_<host>_d1_nb250_gh0_ser0[_pN].txt (Q1d1_* with --Q1), merged and
# uploaded to galAgama/ like launch_orblib_exp.sh does.
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
N_VCPU=$(nproc)
_THREADS_AUTO=$(( N_VCPU / 4 ))
[ "$_THREADS_AUTO" -lt 1 ] && _THREADS_AUTO=1

INCL="90.0"
Q1=0
N_PROC=1
THREADS="${TYCHO_THREADS:-$_THREADS_AUTO}"
RESUME=0
PREFLIGHT=0
SAVE_ORBLIB=1
DO_SYNC=1
USE_SCOPE=1
PRINT_LIMITS=0
ORBLIB_DIR=""
CALC_SCRIPT="${CALC_SCRIPT:-Fornax_P21_PCA_w3Sersic_orblib_exp.py}"
EXTRA_ARGS=""
ORIG_ARGS=("$@")

for arg in "$@"; do
    case $arg in
        --help|-h)
            printf '%s\n' \
                'Использование: bash launch_tycho.sh [--incl=90.0] [--Q1] [опции]' \
                '  --nproc=1           Число процессов поиска (делят --threads и память).' \
                '  --threads=N         Всего потоков (по умолчанию nproc/4).' \
                '  --resume            Продолжить с checkpoint.' \
                '  --orblib-dir=DIR    Каталог библиотек орбит (по умолчанию ./orblib).' \
                '  --no-save-orblib    Не сохранять библиотеки (reuse локальных остаётся).' \
                '  --no-sync           Лаунчер не выгружает историю/логи (Python-скрипт' \
                '                      по-прежнему синхронизирует историю через rclone).' \
                '  --preflight         Только проверки окружения и пробный scope с лимитами.' \
                '  --no-scope          Без systemd-scope: БЕЗ лимита памяти (не рекомендуется).' \
                'Env: CONDA_ENV=agamaBoTorch ("" — не активировать), PYTHON=python,' \
                '  TYCHO_MEM_MAX (байт, по умолчанию MemTotal/4), TYCHO_MIN_FREE_GB=40,' \
                '  TYCHO_STOP_GRACE=1800, TYCHO_MONITOR_INTERVAL=60.' \
                'Остановка: Ctrl-C или systemctl --user kill --signal=TERM <unit> (см. лог).'
            exit 0
            ;;
        --incl=*)          INCL="${arg#*=}" ;;
        --Q1)              Q1=1 ;;
        --nproc=*)         N_PROC="${arg#*=}" ;;
        --threads=*)       THREADS="${arg#*=}" ;;
        --resume)          RESUME=1 ;;
        --orblib-dir=*)    ORBLIB_DIR="${arg#*=}" ;;
        --no-save-orblib)  SAVE_ORBLIB=0 ;;
        --no-sync)         DO_SYNC=0 ;;
        --preflight)       PREFLIGHT=1 ;;
        --no-scope)        USE_SCOPE=0 ;;
        --script=*)        CALC_SCRIPT="${arg#*=}" ;;
        --_print-limits)   PRINT_LIMITS=1 ;;
        --no-double|--n-bin=*|--gh-id=*|--ser-id=*|--stream-orblib|--save-orblib|--reuse-orblib|--n_threads*|--suffix*|--q1)
            echo "ОШИБКА: $arg не поддерживается launch_tycho.sh (d1_nb250, локальные библиотеки)" >&2
            exit 2
            ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

TYCHO_MEMINFO="${TYCHO_MEMINFO:-/proc/meminfo}"
TYCHO_PROC_CGROUP="${TYCHO_PROC_CGROUP:-/proc/self/cgroup}"
TYCHO_CGROUP_FS="${TYCHO_CGROUP_FS:-/sys/fs/cgroup}"
_UID="$(id -u)"
TYCHO_USER_CGROUP="${TYCHO_USER_CGROUP:-${TYCHO_CGROUP_FS}/user.slice/user-${_UID}.slice/user@${_UID}.service}"
TYCHO_MIN_FREE_GB="${TYCHO_MIN_FREE_GB:-40}"
TYCHO_STOP_GRACE="${TYCHO_STOP_GRACE:-1800}"
TYCHO_MONITOR_INTERVAL="${TYCHO_MONITOR_INTERVAL:-60}"
CONDA_ENV="${CONDA_ENV-agamaBoTorch}"
PYTHON="${PYTHON:-python}"
RCLONE_REMOTE="${RCLONE_REMOTE:-yandex}"
NTFY_TOPIC="${NTFY_TOPIC:-GalaxySchwarzschildFornax}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
REMOTE_DIR="galAgama"
HOSTNAME_ENV="${HOSTNAME_SUFFIX:-$(hostname)}"
TIMESTAMP="${TYCHO_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
[ -n "$ORBLIB_DIR" ] || ORBLIB_DIR="${WORK_DIR}/orblib"
case "$ORBLIB_DIR" in /*) ;; *) ORBLIB_DIR="${WORK_DIR}/${ORBLIB_DIR}" ;; esac

# --- Валидация ---
for _v in N_PROC THREADS TYCHO_MIN_FREE_GB TYCHO_STOP_GRACE TYCHO_MONITOR_INTERVAL; do
    if ! [[ "${!_v}" =~ ^[0-9]+$ ]]; then
        echo "ОШИБКА: $_v должно быть целым >= 0 (получено: '${!_v}')" >&2
        exit 2
    fi
done
if [ "$N_PROC" -lt 1 ] || [ "$THREADS" -lt "$N_PROC" ]; then
    echo "ОШИБКА: нужно 1 <= nproc <= threads (nproc=$N_PROC, threads=$THREADS)" >&2
    exit 2
fi
THREADS_PER_PROC=$(( THREADS / N_PROC ))
CPU_QUOTA="$(( THREADS_PER_PROC * N_PROC * 100 ))%"

_MEM_TOTAL_KB=$(awk '/^MemTotal:/ {print $2}' "$TYCHO_MEMINFO" 2>/dev/null || echo 0)
MEM_MAX="${TYCHO_MEM_MAX:-$(( _MEM_TOTAL_KB * 1024 / 4 ))}"
if ! [[ "$MEM_MAX" =~ ^[0-9]+$ ]] || [ "$MEM_MAX" -lt $((1024 * 1024 * 1024)) ]; then
    echo "ОШИБКА: лимит памяти TYCHO_MEM_MAX='$MEM_MAX' байт (< 1 GiB или не число)" >&2
    exit 2
fi
MEM_HIGH=$(( MEM_MAX / 10 * 9 ))

EXP_ID="d1_nb250_gh0_ser0"
[ "$Q1" -eq 1 ] && EXP_ID="Q1${EXP_ID}"
INCL_FMT="$(LC_ALL=C printf '%.1f' "$INCL")"
RUN_TAG="${EXP_ID}_i${INCL_FMT}"
UNIT="agama-tycho-${RUN_TAG//./p}-${TIMESTAMP}"
LOGFILE="${WORK_DIR}/launch_tycho_${RUN_TAG}_${TIMESTAMP}.log"
MONITOR_LOG="${WORK_DIR}/monitor_tycho_${RUN_TAG}_${TIMESTAMP}.log"
LOCK_FILE="${WORK_DIR}/.tycho_launcher.lock"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

die() {
    log "ОШИБКА: $*"
    exit 1
}

notify() {
    curl -s -f --max-time 10 -H "Title: Tycho ${HOSTNAME_ENV}" -H "Priority: ${2:-default}" \
        -d "$1" "${NTFY_SERVER}/${NTFY_TOPIC}" > /dev/null 2>>"$LOGFILE" \
        || log "  ~ ntfy: не удалось отправить уведомление"
}

# Лимиты cgroup текущего процесса (cgroup v2: одна строка '0::/path').
print_limits() {
    local path
    path=$(awk -F: '$1 == "0" {print $3}' "$TYCHO_PROC_CGROUP" 2>/dev/null || true)
    local dir="${TYCHO_CGROUP_FS}${path}"
    local f
    echo "cgroup=${path}"
    for f in memory.max memory.high memory.swap.max cpu.max; do
        printf '%s=%s\n' "$f" "$(cat "${dir}/${f}" 2>/dev/null | tr '\n' ' ' | sed 's/ $//' || true)"
    done
}

if [ "$PRINT_LIMITS" -eq 1 ]; then
    print_limits
    exit 0
fi

SCOPE_PROPS=(-p "CPUQuota=${CPU_QUOTA}" -p "MemoryHigh=${MEM_HIGH}" -p "MemoryMax=${MEM_MAX}"
             -p "MemorySwapMax=0" -p "IOWeight=10" -p "OOMPolicy=continue")

activate_conda() {
    [ -n "$CONDA_ENV" ] || return 0
    command -v conda > /dev/null 2>&1 || die "conda не найдена (CONDA_ENV=$CONDA_ENV; '' — не активировать)"
    # shellcheck disable=SC1091
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh" || { set -u; die "не найден conda.sh"; }
    conda activate "$CONDA_ENV" || { set -u; die "не удалось активировать $CONDA_ENV"; }
    set -u
}

# ==============================================================
# PREFLIGHT (вне scope): окружение, cgroup, диск, сеть, lock
# ==============================================================
preflight() {
    log "--- Preflight: ${RUN_TAG}, host=${HOSTNAME_ENV} ---"
    log "  Бюджет: процессов=${N_PROC} × потоков=${THREADS_PER_PROC} (vCPU хоста ${N_VCPU})," \
        "CPUQuota=${CPU_QUOTA}, MemoryMax=${MEM_MAX}, MemoryHigh=${MEM_HIGH}, MemorySwapMax=0"
    local f
    for f in "$CALC_SCRIPT" orblib_storage.py table3.dat; do
        [ -f "${WORK_DIR}/${f}" ] || die "нет файла ${WORK_DIR}/${f}"
    done
    activate_conda
    command -v "$PYTHON" > /dev/null 2>&1 || die "не найден $PYTHON"
    if ! "$PYTHON" -c "import agama, torch, botorch, sklearn, gpytorch, scipy, requests; assert callable(agama.setNumThreads)" \
            >>"$LOGFILE" 2>&1; then
        die "окружение $PYTHON неполное: нужны agama (с setNumThreads), torch, botorch, gpytorch, sklearn, scipy, requests"
    fi
    log "  ✓ Python: $(command -v "$PYTHON")"

    if [ "$USE_SCOPE" -eq 1 ]; then
        command -v systemd-run > /dev/null 2>&1 || die "нет systemd-run (или используйте --no-scope)"
        local controllers
        controllers=$(cat "${TYCHO_USER_CGROUP}/cgroup.controllers" 2>/dev/null || true)
        local c
        for c in cpu memory; do
            case " $controllers " in
                *" $c "*) ;;
                *) die "контроллер cgroup '$c' не делегирован user-менеджеру (${TYCHO_USER_CGROUP}/cgroup.controllers='${controllers}'). Нужен Delegate=cpu memory io pids в user@.service (настраивает администратор)." ;;
            esac
        done
        log "  ✓ cgroup-контроллеры: ${controllers}"
        local linger
        linger=$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || echo unknown)
        if [ "$linger" != "yes" ] && [ -z "${TMUX:-}" ] && [ -z "${STY:-}" ]; then
            log "  ~ ВНИМАНИЕ: Linger=${linger} и не tmux/screen — при выходе из сессии scope будет остановлен"
        fi
    else
        log "  ~ ВНИМАНИЕ: --no-scope — лимиты CPU/памяти НЕ применяются (только nice/ionice/OMP=${THREADS_PER_PROC})"
    fi

    mkdir -p "$ORBLIB_DIR"
    local free_kb need_kb
    free_kb=$(df -Pk "$ORBLIB_DIR" | awk 'NR == 2 {print $4}')
    need_kb=$(( TYCHO_MIN_FREE_GB * 1024 * 1024 ))
    if [ "$SAVE_ORBLIB" -eq 1 ] && [ "${free_kb:-0}" -lt "$need_kb" ]; then
        die "мало места в ${ORBLIB_DIR}: $(( ${free_kb:-0} / 1024 / 1024 )) GB < ${TYCHO_MIN_FREE_GB} GB (~0.6 GB .npz на модель; TYCHO_MIN_FREE_GB или --orblib-dir)"
    fi
    log "  ✓ Диск ${ORBLIB_DIR}: $(( ${free_kb:-0} / 1024 / 1024 )) GB свободно"

    if rclone lsf "${RCLONE_REMOTE}:${REMOTE_DIR}/" --max-depth 1 --timeout 30s --contimeout 10s \
            --retries 1 --low-level-retries 1 > /dev/null 2>>"$LOGFILE"; then
        log "  ✓ rclone ${RCLONE_REMOTE}:${REMOTE_DIR}/ доступен"
    else
        log "  ~ ВНИМАНИЕ: ${RCLONE_REMOTE}:${REMOTE_DIR}/ недоступен — общий пул не обновится"
    fi

    if ! ( flock -n 9 ) 9>"$LOCK_FILE"; then
        die "в ${WORK_DIR} уже работает launch_tycho.sh (${LOCK_FILE})"
    fi
}

# ==============================================================
# Фаза 1: вне scope → preflight → re-exec внутри scope
# ==============================================================
if [ "${TYCHO_IN_SCOPE:-0}" != "1" ]; then
    cd "$WORK_DIR"
    preflight
    if [ "$PREFLIGHT" -eq 1 ]; then
        if [ "$USE_SCOPE" -eq 1 ]; then
            log "  Пробный scope с лимитами:"
            systemd-run --user --scope --quiet --collect "${SCOPE_PROPS[@]}" \
                bash "$SELF" --_print-limits 2>&1 | tee -a "$LOGFILE"
        fi
        log "Preflight завершён (расчёт не запускался)"
        exit 0
    fi
    export TYCHO_IN_SCOPE=1 TYCHO_TIMESTAMP="$TIMESTAMP"
    if [ "$USE_SCOPE" -eq 1 ]; then
        log "Запуск в systemd scope ${UNIT}.scope"
        log "  Остановка: systemctl --user kill --signal=TERM ${UNIT}.scope"
        exec systemd-run --user --scope --quiet --collect --unit="$UNIT" "${SCOPE_PROPS[@]}" \
            bash "$SELF" "${ORIG_ARGS[@]}"
    fi
    exec bash "$SELF" "${ORIG_ARGS[@]}"
fi

# ==============================================================
# Фаза 2: внутри scope
# ==============================================================
cd "$WORK_DIR"
exec 9>"$LOCK_FILE"
flock -n 9 || die "в ${WORK_DIR} уже работает launch_tycho.sh (${LOCK_FILE})"
activate_conda

LIMITS="$(print_limits)"
log "======================================================"
log "launch_tycho: ${CALC_SCRIPT}, ${RUN_TAG}, host=${HOSTNAME_ENV}, resume=${RESUME}"
log "  процессов=${N_PROC} × потоков=${THREADS_PER_PROC}, orblib=${ORBLIB_DIR} (save=${SAVE_ORBLIB})"
while IFS= read -r _line; do log "  ${_line}"; done <<< "$LIMITS"
if [ "$USE_SCOPE" -eq 1 ] && grep -qE '^memory\.max=(max)?$' <<< "$LIMITS"; then
    die "лимит памяти scope не применён (memory.max не задан) — расчёт не запускается"
fi

CG_DIR="${TYCHO_CGROUP_FS}$(awk -F: '$1 == "0" {print $3}' "$TYCHO_PROC_CGROUP" 2>/dev/null || true)"

append_and_remove() {
    local src="$1" dst="$2" label="$3"
    [ -f "$src" ] || return 0
    local src_lines
    src_lines=$(grep -v "^#" "$src" | grep -v "^[[:space:]]*$" | grep -c "" 2>/dev/null || true)
    if [ "${src_lines:-0}" -eq 0 ]; then
        log "  Пропуск (нет строк данных): $src"
        rm -f "$src"
        return 0
    fi
    local src_hash
    src_hash=$(grep -v "^#" "$src" | grep -v "^[[:space:]]*$" | md5sum | cut -d' ' -f1)
    if [ -f "$dst" ] && grep -q "HASH:${src_hash}" "$dst" 2>/dev/null; then
        log "  Пропуск: $src уже добавлен (HASH:${src_hash})"
        rm -f "$src"
        return 0
    fi
    log "  $src → $dst ($src_lines строк данных, HASH:${src_hash})"
    if {
        echo ""
        echo "# ============================================================"
        echo "# Добавлено из: $src"
        echo "# Label:  ${label}"
        echo "# Time:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# Lines:  $src_lines"
        echo "# HASH:${src_hash}"
        echo "# ============================================================"
        cat "$src"
    } >> "$dst"; then
        rm -f "$src"
    else
        log "  ОШИБКА записи в $dst — $src сохранён!"
        return 1
    fi
}

proc_file()  { printf '%s/out_%s_%s_%s.txt' "$WORK_DIR" "$HOSTNAME_ENV" "$EXP_ID" "$1"; }
host_file()  { printf '%s/out_%s_%s.txt' "$WORK_DIR" "$HOSTNAME_ENV" "$EXP_ID"; }
checkpoint() { printf '%s/checkpoint_%s_%s_%s.pkl' "$WORK_DIR" "$HOSTNAME_ENV" "$EXP_ID" "$1"; }

RCLONE_OPTS=(--timeout 30s --contimeout 10s --retries 1 --low-level-retries 1 --stats-one-line)

upload() {
    [ "$DO_SYNC" -eq 1 ] && [ -f "$1" ] || return 0
    rclone copyto "$1" "${RCLONE_REMOTE}:${2:-$REMOTE_DIR}/$(basename "$1")" "${RCLONE_OPTS[@]}" 2>>"$LOGFILE" \
        && log "  ✓ Загружено: $(basename "$1")" || log "  ✗ Ошибка загрузки: $(basename "$1")"
}

delete_remote() {
    [ "$DO_SYNC" -eq 1 ] || return 0
    rclone deletefile "${RCLONE_REMOTE}:${REMOTE_DIR}/$1" "${RCLONE_OPTS[@]}" 2>>"$LOGFILE" \
        && log "  ✓ Удалено с Яндекс.Диска: $1" || log "  ~ Не найдено на диске: $1"
}

download_missing() {
    [ -f "${WORK_DIR}/$1" ] && { log "  Локально есть: $1"; return 0; }
    rclone copyto "${RCLONE_REMOTE}:${REMOTE_DIR}/$1" "${WORK_DIR}/$1" "${RCLONE_OPTS[@]}" 2>>"$LOGFILE" \
        && log "  ✓ Скачан: $1" || log "  ~ Не найден на Яндекс.Диске: $1"
}

merge_all() {
    local label="$1" sfx
    for sfx in "${SUFFIXES[@]}"; do
        append_and_remove "$(proc_file "$sfx")" "$(host_file)" "${label}, suffix=${sfx}" || true
    done
    upload "$(host_file)"
    for sfx in "${SUFFIXES[@]}"; do
        delete_remote "$(basename "$(proc_file "$sfx")")"
    done
}

# Фоновые циклы спят через `sleep & wait`: при kill подоболочки дочерний sleep
# иначе остался бы сиротой, держал бы stdout и не давал scope завершиться.
nap() {
    local s
    sleep "$1" &
    s=$!
    trap 'kill "$s" 2>/dev/null; exit 0' TERM
    wait "$s"
}

monitor_loop() {
    while true; do
        {
            echo "=== $(date '+%Y-%m-%d %H:%M:%S')"
            free -m 2>/dev/null || true
            cat /proc/pressure/memory 2>/dev/null || true
            for f in memory.current memory.peak memory.swap.current; do
                printf '%s=%s\n' "$f" "$(cat "${CG_DIR}/${f}" 2>/dev/null || echo n/a)"
            done
            grep -E '^(usage_usec|nr_throttled|throttled_usec)' "${CG_DIR}/cpu.stat" 2>/dev/null || true
            df -h "$ORBLIB_DIR" 2>/dev/null | tail -1 || true
        } >> "$MONITOR_LOG" 2>&1
        nap "$TYCHO_MONITOR_INTERVAL"
    done
}

declare -a SUFFIXES=() WPIDS=() WSTART=()
for ((i = 0; i < N_PROC; i++)); do SUFFIXES[i]="p${i}"; done
declare -A PROC_FLAGS

# --- Подготовка истории ---
if [ "$RESUME" -eq 1 ]; then
    log "Режим: возобновление с checkpoint"
    if [ "$DO_SYNC" -eq 1 ]; then
        for sfx in "${SUFFIXES[@]}"; do
            download_missing "$(basename "$(proc_file "$sfx")")"
            download_missing "$(basename "$(checkpoint "$sfx")")"
        done
    fi
    merge_all "RESUME-PRE: incl=${INCL}, exp=${EXP_ID}, host=${HOSTNAME_ENV}"
    for sfx in "${SUFFIXES[@]}"; do
        if [ -f "$(checkpoint "$sfx")" ]; then
            log "  ✓ Checkpoint найден: $sfx → resume"
            PROC_FLAGS[$sfx]=""
        else
            log "  ✗ Checkpoint не найден: $sfx → с нуля"
            PROC_FLAGS[$sfx]="--no-resume --delete-checkpoint"
        fi
    done
else
    log "Режим: чистый запуск"
    merge_all "CLEANUP: incl=${INCL}, exp=${EXP_ID}, host=${HOSTNAME_ENV}"
    for sfx in "${SUFFIXES[@]}"; do PROC_FLAGS[$sfx]="--no-resume --delete-checkpoint"; done
fi
rm -f "${WORK_DIR}/reservations_i${INCL_FMT}_${EXP_ID}/"*.resv 2>/dev/null || true

ORBLIB_FLAGS="--reuse-orblib --orblib-dir ${ORBLIB_DIR}"
[ "$SAVE_ORBLIB" -eq 1 ] && ORBLIB_FLAGS="--save-orblib ${ORBLIB_FLAGS}"
EXP_FLAGS=""
[ "$Q1" -eq 1 ] && EXP_FLAGS="--Q1"
export HOSTNAME_SUFFIX="$HOSTNAME_ENV" RCLONE_REMOTE NTFY_TOPIC NTFY_SERVER

# --- Остановка: сигналы доходят до Python (группа процессов/scope); пересылаем
#     ещё раз только тем, кто жив через 5 с, затем KILL после TYCHO_STOP_GRACE.
STOP_REQUESTED=0
KILLER_PID=""
on_signal() {
    [ "$STOP_REQUESTED" -eq 1 ] && return 0
    STOP_REQUESTED=1
    log "Получен сигнал остановки: workers пишут checkpoint (grace ${TYCHO_STOP_GRACE} с)"
    (
        nap 5
        for p in "${WPIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null; done
        nap "$TYCHO_STOP_GRACE"
        for p in "${WPIDS[@]}"; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
    ) &
    KILLER_PID=$!
}
trap on_signal TERM INT

monitor_loop &
MONITOR_PID=$!

log "Запуск ${N_PROC} процесс(ов) поиска"
for ((i = 0; i < N_PROC; i++)); do
    sfx="${SUFFIXES[i]}"
    proc_log="${WORK_DIR}/runlog_${sfx}_${RUN_TAG}_${TIMESTAMP}.log"
    # tee игнорирует INT/TERM: иначе Ctrl-C убил бы его раньше Python, и запись
    # checkpoint упала бы на BrokenPipe. $! — PID самого Python (nice/ionice exec).
    # shellcheck disable=SC2086
    OMP_NUM_THREADS="$THREADS_PER_PROC" MKL_NUM_THREADS="$THREADS_PER_PROC" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_PROC" NUMEXPR_NUM_THREADS="$THREADS_PER_PROC" \
    OMP_PROC_BIND=close OMP_PLACES=cores \
        nice -n 19 ionice -c 3 "$PYTHON" -u "${WORK_DIR}/${CALC_SCRIPT}" \
            --incl "$INCL" --suffix "$sfx" --n_threads "$THREADS_PER_PROC" \
            $ORBLIB_FLAGS $EXP_FLAGS ${PROC_FLAGS[$sfx]} $EXTRA_ARGS \
        > >(trap '' INT TERM; exec tee -a "$proc_log") 2>&1 &
    WPIDS[i]=$!
    WSTART[i]=$(date +%s)
    log "  $sfx: pid=${WPIDS[i]} threads=${THREADS_PER_PROC} flags='${PROC_FLAGS[$sfx]}' лог=$(basename "$proc_log")"
done

declare -a WCODES=()
for ((i = 0; i < N_PROC; i++)); do
    pid="${WPIDS[i]}"
    while true; do
        code=0
        wait "$pid" || code=$?
        kill -0 "$pid" 2>/dev/null || break
    done
    WCODES[i]=$code
    sfx="${SUFFIXES[i]}"
    elapsed=$(( $(date +%s) - WSTART[i] ))
    case "$code" in
        0)   log "  ✓ $sfx завершён (wall ${elapsed} с)" ;;
        75)  log "  ~ $sfx остановлен штатно, checkpoint записан (wall ${elapsed} с)" ;;
        137) log "  ✗ $sfx убит (код 137: OOM по MemoryMax или KILL) — checkpoint не записан; строки истории и .npz целы" ;;
        *)   log "  ✗ $sfx завершён с кодом $code (wall ${elapsed} с)" ;;
    esac
done
[ -n "$KILLER_PID" ] && kill "$KILLER_PID" 2>/dev/null || true
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

log "Итоги ресурсов scope: memory.peak=$(cat "${CG_DIR}/memory.peak" 2>/dev/null || echo n/a)," \
    "$(grep -E '^usage_usec' "${CG_DIR}/cpu.stat" 2>/dev/null || echo 'cpu.stat n/a')"

merge_all "RESULT(${WCODES[*]}): incl=${INCL}, exp=${EXP_ID}, host=${HOSTNAME_ENV}"

LOG_REMOTE="${REMOTE_DIR}/tycho_logs/${RUN_TAG}_${TIMESTAMP}"
for f in "$MONITOR_LOG" "${WORK_DIR}"/runlog_*_"${RUN_TAG}_${TIMESTAMP}".log \
         "${WORK_DIR}/log_${HOSTNAME_ENV}_${EXP_ID}"_p*.txt; do
    upload "$f" "$LOG_REMOTE"
done

FINAL=0
for code in "${WCODES[@]}"; do
    [ "$code" -ne 0 ] && FINAL=$code
done
notify "launch_tycho ${RUN_TAG} на ${HOSTNAME_ENV}: коды ${WCODES[*]}" "$([ "$FINAL" -eq 0 ] && echo default || echo high)"
log "Завершено, коды: ${WCODES[*]}"
upload "$LOGFILE" "$LOG_REMOTE"
exit "$FINAL"
