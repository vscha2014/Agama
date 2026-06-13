#!/bin/bash
# launch_docker_parallel.sh
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="agama:latest"
N_VCPU=$(nproc)
N_PROC=4
THREADS_PER_PROC=$((N_VCPU / N_PROC))

# --- Разбор аргументов ---
INCL="90.0"
RESUME=0
DO_SHUTDOWN=1
EXTRA_ARGS=""
CALC_SCRIPT="${CALC_SCRIPT:-Fornax_P21_symm_PCA_w3Sersic_yaVM.py}"

for arg in "$@"; do
    case $arg in
        --incl=*)      INCL="${arg#*=}"        ;;
        --resume)      RESUME=1                ;;
        --no-shutdown) DO_SHUTDOWN=0           ;;
        --script=*)    CALC_SCRIPT="${arg#*=}" ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

NTFY_TOPIC="${NTFY_TOPIC:-GalaxySchwarzschildFornax}"
NTFY_SERVER="${NTFY_SERVER:-https://ntfy.sh}"
RCLONE_REMOTE="${RCLONE_REMOTE:-yandex}"
RCLONE_CONF_DIR="${HOME}/.config/rclone"
HOSTNAME_ENV="$(hostname)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_DIR="galAgama"

SUFFIXES=("p0" "p1" "p2" "p3")
CPU_RANGES=("0-7" "8-15" "16-23" "24-31")

LOGFILE="${WORK_DIR}/launch_i${INCL}_${TIMESTAMP}.log"

# ==============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

die() {
    log "ОШИБКА: $*"
    notify "ОШИБКА на ${HOSTNAME_ENV}: $*" "urgent"
    [ $DO_SHUTDOWN -eq 1 ] && sudo shutdown -h +2 &
    exit 1
}

notify() {
    local msg="$1"
    local priority="${2:-default}"
    curl -s \
        -H "Title: Galaxy ${HOSTNAME_ENV}" \
        -H "Priority: ${priority}" \
        -d "$msg" \
        "${NTFY_SERVER}/${NTFY_TOPIC}" || true
}

# --------------------------------------------------------------
# append_and_remove: добавление файла-источника в файл-назначение
# с проверкой по MD5-хешу строк данных (не комментариев)
# --------------------------------------------------------------
append_and_remove() {
    local src="$1"
    local dst="$2"
    local label="$3"

    [ -f "$src" ] || return 0

    # Считаем только строки с данными (не комментарии, не пустые)
    local src_lines
    src_lines=$(grep -v "^#" "$src" \
                | grep -v "^[[:space:]]*$" \
                | grep -c "" 2>/dev/null || echo 0)

    if [ "$src_lines" -eq 0 ]; then
        log "  Пропуск (нет строк данных): $src"
        rm -f "$src"
        return 0
    fi

    # --- MD5-хеш строк данных ---
    local src_hash
    src_hash=$(grep -v "^#" "$src" \
               | grep -v "^[[:space:]]*$" \
               | md5sum | cut -d' ' -f1)

    # --- Проверка по хешу ---
    if [ -f "$dst" ] && grep -q "HASH:${src_hash}" "$dst" 2>/dev/null; then
        log "  Пропуск: $src уже добавлен (HASH:${src_hash})"
        rm -f "$src"
        return 0
    fi

    log "  $src → $dst ($src_lines строк данных, HASH:${src_hash})"

    {
        echo ""
        echo "# ============================================================"
        echo "# Добавлено из: $src"
        echo "# Label:  ${label}"
        echo "# Time:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "# Lines:  $src_lines"
        echo "# HASH:${src_hash}"
        echo "# ============================================================"
        cat "$src"
    } >> "$dst"

    if [ $? -eq 0 ]; then
        rm -f "$src"
        log "  ✓ Объединено и удалено: $src (HASH:${src_hash})"
    else
        log "  ОШИБКА записи в $dst — $src сохранён!"
        return 1
    fi
}

merge_proc_files() {
    local sfx="$1"
    local label="$2"
    append_and_remove \
        "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"     \
        "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"            \
        "$label"
    append_and_remove \
        "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt" \
        "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"        \
        "$label"
}

download_from_yadisk() {
    local fname="$1"
    local remote_path="${RCLONE_REMOTE}:${REMOTE_DIR}/${fname}"
    rclone copyto "$remote_path" "${WORK_DIR}/${fname}" \
        --config "${RCLONE_CONF_DIR}/rclone.conf" \
        --stats-one-line 2>>"$LOGFILE" \
        && log "  ✓ Скачан: $fname" \
        || { log "  ~ Не найден на Яндекс.Диске: $fname"; return 1; }
}

upload_to_yadisk() {
    local filepath="$1"
    [ -f "$filepath" ] || return 0
    local fname
    fname=$(basename "$filepath")
    rclone copyto "$filepath" \
        "${RCLONE_REMOTE}:${REMOTE_DIR}/${fname}" \
        --config "${RCLONE_CONF_DIR}/rclone.conf" \
        --stats-one-line 2>>"$LOGFILE" \
        && log "  ✓ Загружено: $fname" \
        || log "  ✗ Ошибка загрузки: $fname"
}

delete_from_yadisk() {
    local fname="$1"
    rclone deletefile \
        "${RCLONE_REMOTE}:${REMOTE_DIR}/${fname}" \
        --config "${RCLONE_CONF_DIR}/rclone.conf" \
        2>>"$LOGFILE" \
        && log "  ✓ Удалено с Яндекс.Диска: $fname" \
        || log "  ~ Не найдено на диске: $fname"
}

# --------------------------------------------------------------
# run_container: запуск одного Docker-контейнера
# # В run_container — только merge и upload:
# --------------------------------------------------------------
run_container() {
    local sfx="$1"
    local cpu_start="$2"
    local cpu_end="$3"
    local flags="${PROC_FLAGS[$sfx]}"
    local proc_log="${WORK_DIR}/log_${sfx}_i${INCL}_${TIMESTAMP}.log"
    local exit_code=0

    log "  Контейнер $sfx: CPU=${cpu_start}-${cpu_end} flags='$flags'"

    # --- Запуск контейнера ---
    docker run --rm \
        --name "agama_${HOSTNAME_ENV}_${sfx}" \
        --cpuset-cpus="${cpu_start}-${cpu_end}" \
        \
        -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" \
        \
        -v "${WORK_DIR}:/workspace" \
        -v "${RCLONE_CONF_DIR}:/workspace/.config/rclone:ro" \
        \
        -e RCLONE_CONFIG="/workspace/.config/rclone/rclone.conf" \
        -e RCLONE_REMOTE="${RCLONE_REMOTE}" \
        -e HOSTNAME_SUFFIX="${HOSTNAME_ENV}" \
        -e NTFY_TOPIC="${NTFY_TOPIC}" \
        -e NTFY_SERVER="${NTFY_SERVER}" \
        \
        -e OMP_NUM_THREADS="${THREADS_PER_PROC}" \
        -e OMP_PROC_BIND="close" \
        -e OMP_PLACES="cores" \
        -e MKL_NUM_THREADS="${THREADS_PER_PROC}" \
        -e OPENBLAS_NUM_THREADS="${THREADS_PER_PROC}" \
        -e NUMEXPR_NUM_THREADS="${THREADS_PER_PROC}" \
        \
        -w /workspace \
        "${IMAGE}" \
        python3 -u "/workspace/${CALC_SCRIPT}" \
            --incl "${INCL}" \
            --suffix "${sfx}" \
            $flags \
            $EXTRA_ARGS \
        2>&1 | tee "$proc_log" \
        || exit_code=$?

    # --- Немедленное объединение после завершения контейнера ---
    # (выполняется в фоновом подпроцессе, не блокирует другие контейнеры)
    local merge_label
    if [ $exit_code -eq 0 ]; then
        touch "${WORK_DIR}/.done_${sfx}"
        merge_label="RESULT-OK: incl=${INCL}, suffix=${sfx}, host=${HOSTNAME_ENV}"
        log "  ✓ Контейнер $sfx завершён успешно — объединяем файлы"
    else
        merge_label="RESULT-ERR(${exit_code}): incl=${INCL}, suffix=${sfx}, host=${HOSTNAME_ENV}"
        log "  ✗ Контейнер $sfx завершён с кодом $exit_code — объединяем частичные файлы"
    fi

    # Объединяем файлы этого процесса сразу после его завершения
    merge_proc_files "$sfx" "$merge_label"

    # Загружаем обновлённые общие файлы на Яндекс.Диск
    # (используем flock чтобы избежать одновременной записи от разных процессов)
    (
        flock -x 200
        upload_to_yadisk \
            "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
        upload_to_yadisk \
            "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
    ) 200>"${WORK_DIR}/.upload_lock"

#    # Удаляем _proc файлы с Яндекс.Диска (они уже в общем файле)
#    delete_from_yadisk \
#        "4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
#    delete_from_yadisk \
#        "4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"

    # Загружаем лог процесса
    upload_to_yadisk "$proc_log"

    return $exit_code
}

# ==============================================================
# ПРОВЕРКИ
# ==============================================================
log "======================================================"
log "AGAMA параллельный запуск (Docker)"
log "  hostname         = $HOSTNAME_ENV"
log "  script           = $CALC_SCRIPT"
log "  incl             = $INCL"
log "  resume           = $RESUME"
log "  shutdown         = $DO_SHUTDOWN"
log "  vCPU всего       = $N_VCPU"
log "  Процессов        = $N_PROC"
log "  Потоков/процесс  = $THREADS_PER_PROC"
log "======================================================"

[ -f "${RCLONE_CONF_DIR}/rclone.conf" ] \
    || die "rclone не настроен: ${RCLONE_CONF_DIR}/rclone.conf"

for f in "${CALC_SCRIPT}" table3.dat; do
    [ -f "${WORK_DIR}/${f}" ] || die "не найден ${WORK_DIR}/${f}"
done

docker image inspect "$IMAGE" > /dev/null 2>&1 \
    || die "Docker-образ $IMAGE не найден"

notify "Старт на ${HOSTNAME_ENV}, script=${CALC_SCRIPT}, incl=${INCL}"

# ==============================================================
# ШАГ 1: ПОДГОТОВКА ФАЙЛОВ
# ==============================================================
log ""
log "ШАГ 1: Подготовка файлов"

rm -f "${WORK_DIR}"/.done_*
rm -f "${WORK_DIR}"/.upload_lock

declare -A PROC_FLAGS

if [ $RESUME -eq 1 ]; then
    log "  Режим: возобновление с checkpoint"

    # Скачиваем недостающие файлы с Яндекс.Диска
    for sfx in "${SUFFIXES[@]}"; do
        for f in \
            "4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"     \
            "4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt" \
            "checkpoint_${HOSTNAME_ENV}_${sfx}.pkl"
        do
            [ ! -f "${WORK_DIR}/${f}" ] \
                && { download_from_yadisk "$f" || true; } \
                || log "  Локально есть: $f"
        done
    done

    # Объединяем _proc файлы прерванного запуска в общие
    log "  Объединение файлов прерванных процессов..."
    for sfx in "${SUFFIXES[@]}"; do
        merge_proc_files "$sfx" \
            "RESUME-PRE: incl=${INCL}, suffix=${sfx}, host=${HOSTNAME_ENV}"
    done

    # Загружаем обновлённые общие файлы
    upload_to_yadisk \
        "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
    upload_to_yadisk \
        "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"

    for sfx in "${SUFFIXES[@]}"; do
        delete_from_yadisk \
            "4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
        delete_from_yadisk \
            "4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
    done

    # Определяем флаги Python для каждого процесса
    for sfx in "${SUFFIXES[@]}"; do
        cp_file="${WORK_DIR}/checkpoint_${HOSTNAME_ENV}_${sfx}.pkl"
        if [ -f "$cp_file" ]; then
            log "  ✓ Checkpoint найден: $sfx → resume"
            PROC_FLAGS[$sfx]=""
        else
            log "  ✗ Checkpoint не найден: $sfx → с нуля"
            PROC_FLAGS[$sfx]="--no-resume --delete-checkpoint"
        fi
    done

else
    log "  Режим: чистый запуск с нуля"

    for sfx in "${SUFFIXES[@]}"; do
        merge_proc_files "$sfx" \
            "CLEANUP: incl=${INCL}, suffix=${sfx}, host=${HOSTNAME_ENV}"
        PROC_FLAGS[$sfx]="--no-resume --delete-checkpoint"
    done
fi

# ==============================================================
# ШАГ 2: ЗАПУСК КОНТЕЙНЕРОВ
# ==============================================================
log ""
log "ШАГ 2: Запуск $N_PROC контейнеров..."

declare -a PIDS
for i in $(seq 0 $((N_PROC - 1))); do
    sfx="${SUFFIXES[$i]}"
    cpu_s="${CPU_RANGES[$i]%%-*}"
    cpu_e="${CPU_RANGES[$i]##*-}"
    # run_container запускается в фоне
    run_container "$sfx" "$cpu_s" "$cpu_e" &
    PIDS[$i]=$!
    log "  PID ${PIDS[$i]} → процесс $sfx"
done

log "  Все контейнеры запущены: PIDs=${PIDS[*]}"

# ==============================================================
# ШАГ 3: ОЖИДАНИЕ ЗАВЕРШЕНИЯ
# ==============================================================
log ""
log "ШАГ 3: Ожидание завершения всех контейнеров..."

FAILED=0
declare -a EXIT_CODES
for i in $(seq 0 $((N_PROC - 1))); do
    sfx="${SUFFIXES[$i]}"
    set +e; wait "${PIDS[$i]}"; EXIT_CODES[$i]=$?; set -e
    if [ "${EXIT_CODES[$i]}" -eq 0 ]; then
        log "  ✓ $sfx завершён (код 0)"
    else
        log "  ✗ $sfx завершён (код ${EXIT_CODES[$i]})"
        FAILED=$((FAILED + 1))
    fi
done

DONE_COUNT=$(ls "${WORK_DIR}"/.done_* 2>/dev/null | wc -l)
log "  Успешно: ${DONE_COUNT}/${N_PROC}, ошибок: ${FAILED}"

# ==============================================================
# ШАГ 4: ФИНАЛЬНАЯ ПРОВЕРКА ОБЪЕДИНЕНИЯ
# ==============================================================
log ""
log "ШАГ 4: Финальная проверка объединения файлов..."
# run_container уже объединил файлы каждого процесса сразу после завершения.# Здесь объединяем файлы каждого процесса
for sfx in "${SUFFIXES[@]}"; do
    ups="${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
    res="${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
    if [ -f "$ups" ] || [ -f "$res" ]; then
        log "  Найдены необъединённые файлы $sfx — объединяем"
        merge_proc_files "$sfx" \
            "FINAL-CHECK: incl=${INCL}, suffix=${sfx}, host=${HOSTNAME_ENV}"
    fi
done

# ==============================================================
# ШАГ 5: ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ
# ==============================================================
log ""
log "ШАГ 5: Финальная синхронизация на Яндекс.Диск..."

# Финальное удаление _proc файлов с Яндекс.Диска
# К этому моменту все контейнеры завершены и finalize() отработал
(
    flock -x 200
    upload_to_yadisk \
        "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
    upload_to_yadisk \
        "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
) 200>"${WORK_DIR}/.upload_lock"

for sfx in "${SUFFIXES[@]}"; do
    delete_from_yadisk \
        "4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
    delete_from_yadisk \
        "4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${sfx}.txt"
    delete_from_yadisk "checkpoint_${HOSTNAME_ENV}_${sfx}.pkl"
done

upload_to_yadisk "$LOGFILE"
rm -f "${WORK_DIR}/.upload_lock"

# ==============================================================
# ИТОГ И УВЕДОМЛЕНИЕ
# ==============================================================
log ""
log "======================================================"
log "ЗАВЕРШЕНО: $(date '+%Y-%m-%d %H:%M:%S')"
log "  Успешно: ${DONE_COUNT}/${N_PROC}, ошибок: ${FAILED}"
for i in $(seq 0 $((N_PROC - 1))); do
    log "    ${SUFFIXES[$i]}: код ${EXIT_CODES[$i]}"
done
log "  Итоговые файлы:"
for f in \
    "${WORK_DIR}/4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt" \
    "${WORK_DIR}/4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
do
    if [ -f "$f" ]; then
        lines=$(grep -v "^#" "$f" | grep -c "" 2>/dev/null || echo 0)
        size=$(du -h "$f" | cut -f1)
        log "    $(basename $f): $lines строк данных, $size"
    fi
done
log "======================================================"

notify \
    "Завершено ${DONE_COUNT}/${N_PROC}, ошибок: ${FAILED}, incl=${INCL}" \
    "$([ $FAILED -eq 0 ] && echo high || echo urgent)"

# ==============================================================
# ШАГ 6: ВЫКЛЮЧЕНИЕ VM
# ==============================================================
if [ $DO_SHUTDOWN -eq 1 ]; then
    delay=$([ $FAILED -eq 0 ] && echo 1 || echo 5)
    log "Выключение VM через ${delay} мин..."
    upload_to_yadisk "$LOGFILE"
    sudo shutdown -h +"$delay" "AGAMA расчёт завершён"
else
    log "Выключение пропущено (--no-shutdown)"
fi

[ $FAILED -eq 0 ] && exit 0 || exit 1
