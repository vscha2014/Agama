#!/bin/bash
# launch_orblib_exp.sh — оркестратор ЭКСПЕРИМЕНТАЛЬНОГО скрипта
# Fornax_P21_PCA_w3Sersic_orblib_exp.py (НЕ прода!).
#
# Отличия от launch_docker_parallel.sh:
#   * скрипт по умолчанию: Fornax_P21_PCA_w3Sersic_orblib_exp.py;
#   * файлы истории/логов: out_*_{EXP_ID}_*.txt / log_*_{EXP_ID}_*.txt
#     (а не 4Ups*/4result*);
#   * схема tar-шардов библиотек орбит в galAgama/orblib на Яндекс.Диске:
#       ШАГ 0 (до запуска): потоково распаковать все orblib_${KEY}__*.tar
#         в ORBLIB_DIR (дедуп по имени .npz — контент-адресные), снять
#         snapshot before;
#       ФИНАЛ (после контейнеров): after − before = новые .npz → несколько
#         tar-частей допустимого размера в remote partial → проверка size+MD5
#         → атомарная публикация orblib_${KEY}__${HOST}_${TS}_partNNN.tar
#         без локальных копий tar. Существующие шарды НЕ трогаются.
#     Никакого синка орблибов ВО ВРЕМЯ расчёта (AGENTS §10: живая
#     видимость нужна истории out_*, её синкает сам Python-скрипт).
#
# Ключ эксперимента KEY = i{incl:.1f}_d{D}_nb{NB}_ser{SER} — БЕЗ gh_id:
# матрицы орбит не зависят от GH-реализации наблюдений.
#
# Консолидация шардов: когда шардов станет много, их можно оффлайн слить
# в один tar (union .npz по имени) и удалить исходные. В основной поток
# НЕ входит (политика консолидации — открытый вопрос к PI).
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="agama:latest"
N_VCPU=$(nproc)
# Число параллельных процессов на VM: авто ~1 процесс на 4 vCPU.
_NPROC_AUTO=$(( N_VCPU / 4 ))
[ "$_NPROC_AUTO" -lt 1 ] && _NPROC_AUTO=1
N_PROC="${N_PROC:-$_NPROC_AUTO}"

# --- Разбор аргументов ---
INCL="90.0"
RESUME=0
DO_SHUTDOWN=1
EXTRA_ARGS=""
CALC_SCRIPT="${CALC_SCRIPT:-Fornax_P21_PCA_w3Sersic_orblib_exp.py}"
# Параметры эксперимента (нужны оркестратору для EXP_ID/KEY;
# пробрасываются и в Python-скрипт).
DOUBLE=1
NBIN=250
GHID=0
SERID=0

for arg in "$@"; do
    case $arg in
        --incl=*)      INCL="${arg#*=}"        ;;
        --nproc=*)     N_PROC="${arg#*=}"      ;;
        --resume)      RESUME=1                ;;
        --no-shutdown) DO_SHUTDOWN=0           ;;
        --script=*)    CALC_SCRIPT="${arg#*=}" ;;
        --no-double)   DOUBLE=0                ;;
        --n-bin=*)     NBIN="${arg#*=}"        ;;
        --gh-id=*)     GHID="${arg#*=}"        ;;
        --ser-id=*)    SERID="${arg#*=}"       ;;
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
ORBLIB_REMOTE_DIR="${REMOTE_DIR}/orblib"
ORBLIB_UPLOAD_ATTEMPTS="${ORBLIB_UPLOAD_ATTEMPTS:-3}"
ORBLIB_PART_SIZE_GB="${ORBLIB_PART_SIZE_GB:-40}"
ORBLIB_UPLOAD_TIMEOUT="${ORBLIB_UPLOAD_TIMEOUT:-2h}"

# --- Идентификаторы эксперимента ---
# EXP_ID — как в Python-скрипте (входит в имена out_/log_/checkpoint_).
EXP_ID="d${DOUBLE}_nb${NBIN}_gh${GHID}_ser${SERID}"
# KEY — ключ библиотек орбит: incl с одним знаком, БЕЗ gh_id.
INCL_FMT="$(LC_ALL=C printf '%.1f' "$INCL")"
KEY="i${INCL_FMT}_d${DOUBLE}_nb${NBIN}_ser${SERID}"
SHARD_PATTERN="orblib_${KEY}__${HOSTNAME_ENV}_${TIMESTAMP}_partNNN.tar"

# Каталог библиотек орбит: внутри WORK_DIR → виден в контейнере как
# /workspace/orblib (WORK_DIR монтируется в /workspace), файлы на хосте.
ORBLIB_DIR="${WORK_DIR}/orblib"
SNAP_BEFORE="${ORBLIB_DIR}/.snapshot_before_${TIMESTAMP}.txt"

# Флаги эксперимента для Python-скрипта (сейв+реюз орблибов включены:
# управление библиотеками — смысл этого оркестратора).
EXP_FLAGS="--n-bin=${NBIN} --gh-id=${GHID} --ser-id=${SERID} --save-orblib --reuse-orblib"
[ "$DOUBLE" -eq 0 ] && EXP_FLAGS="--no-double $EXP_FLAGS"

# --- Валидация и авто-раскладка процессов по ядрам ---
if ! [[ "$N_PROC" =~ ^[0-9]+$ ]] || [ "$N_PROC" -lt 1 ]; then
    echo "ОШИБКА: N_PROC должно быть целым >= 1 (получено: '$N_PROC')" >&2
    exit 1
fi
if [ "$N_PROC" -gt "$N_VCPU" ]; then
    echo "ОШИБКА: N_PROC=$N_PROC превышает число vCPU=$N_VCPU" >&2
    exit 1
fi
if ! [[ "$ORBLIB_UPLOAD_ATTEMPTS" =~ ^[0-9]+$ ]] || [ "$ORBLIB_UPLOAD_ATTEMPTS" -lt 1 ]; then
    echo "ОШИБКА: ORBLIB_UPLOAD_ATTEMPTS должно быть целым >= 1" >&2
    exit 1
fi
if ! [[ "$ORBLIB_PART_SIZE_GB" =~ ^[0-9]+$ ]] || [ "$ORBLIB_PART_SIZE_GB" -lt 1 ]; then
    echo "ОШИБКА: ORBLIB_PART_SIZE_GB должно быть целым >= 1" >&2
    exit 1
fi
if ! [[ "$ORBLIB_UPLOAD_TIMEOUT" =~ ^[0-9]+(ms|s|m|h)$ ]]; then
    echo "ОШИБКА: ORBLIB_UPLOAD_TIMEOUT должен иметь вид 30m или 2h" >&2
    exit 1
fi

declare -a SUFFIXES CPU_RANGES THREADS_ARR
_base=$((N_VCPU / N_PROC))
_rem=$((N_VCPU % N_PROC))
_start=0
for ((i = 0; i < N_PROC; i++)); do
    _size=$_base
    [ "$i" -lt "$_rem" ] && _size=$((_base + 1))
    _end=$((_start + _size - 1))
    SUFFIXES[i]="p${i}"
    CPU_RANGES[i]="${_start}-${_end}"
    THREADS_ARR[i]=$_size
    _start=$((_end + 1))
done

LOGFILE="${WORK_DIR}/launch_orblib_${EXP_ID}_i${INCL}_${TIMESTAMP}.log"

MAIN_PID=$$
SHUTDOWN_DONE=0

# ==============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

notify() {
    local msg="$1"
    local priority="${2:-default}"
    if curl -s -f \
        -H "Title: OrblibExp ${HOSTNAME_ENV}" \
        -H "Priority: ${priority}" \
        -d "$msg" \
        "${NTFY_SERVER}/${NTFY_TOPIC}" > /dev/null 2>>"$LOGFILE"; then
        :
    else
        # Не даём set -e упасть на неудаче notify; фиксируем в лог, чтобы
        # было видно, что уведомление не дошло (сеть/ntfy недоступны).
        log "  ~ ntfy: не удалось отправить уведомление ('${msg}')"
    fi
}

schedule_shutdown() {
    local delay="$1"
    local reason="$2"
    # Идемпотентность (SHUTDOWN_DONE) ставим ПЕРВОЙ, до ветвления по
    # --no-shutdown — иначе on_exit не узнаёт, что штатное завершение уже
    # обработано, и шлёт дублирующее аварийное уведомление.
    if [ "$SHUTDOWN_DONE" -eq 1 ]; then
        return 0
    fi
    SHUTDOWN_DONE=1
    # Лог заливаем в любом случае (даже без реального выключения) — чтобы
    # результат был на Я.Диске независимо от --no-shutdown.
    upload_to_yadisk "$LOGFILE" || true
    if [ "$DO_SHUTDOWN" -ne 1 ]; then
        log "Выключение пропущено (--no-shutdown): ${reason}"
        return 0
    fi
    log "Выключение VM через ${delay} мин (${reason})..."
    sudo shutdown -h +"$delay" "AGAMA orblib_exp: ${reason}" || true
}

on_exit() {
    local code=$?
    [ "${BASHPID:-$$}" = "$MAIN_PID" ] || return 0
    if [ "$SHUTDOWN_DONE" -eq 0 ] && [ "$code" -ne 0 ]; then
        log "Аварийное завершение скрипта (код ${code})"
        notify "Аварийное завершение orblib_exp на ${HOSTNAME_ENV} (код ${code})" "urgent"
        # Попытаться сохранить наработанные библиотеки орбит перед выключением
        upload_orblib_shard || true
        schedule_shutdown 5 "аварийное завершение скрипта (код ${code})"
    fi
}
trap on_exit EXIT

die() {
    log "ОШИБКА: $*"
    notify "ОШИБКА orblib_exp на ${HOSTNAME_ENV}: $*" "urgent"
    schedule_shutdown 2 "критическая ошибка: $*"
    exit 1
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

    local src_lines
    src_lines=$(grep -v "^#" "$src" \
                | grep -v "^[[:space:]]*$" \
                | grep -c "" 2>/dev/null || echo 0)

    if [ "$src_lines" -eq 0 ]; then
        log "  Пропуск (нет строк данных): $src"
        rm -f "$src"
        return 0
    fi

    local src_hash
    src_hash=$(grep -v "^#" "$src" \
               | grep -v "^[[:space:]]*$" \
               | md5sum | cut -d' ' -f1)

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

# История эксперимента: per-proc → общий файл хоста (аналог 4Ups в проде,
# но с новыми именами out_*). log_* не объединяются (диагностика).
merge_proc_files() {
    local sfx="$1"
    local label="$2"
    append_and_remove \
        "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt" \
        "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}.txt"        \
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
# download_orblib_shards: ШАГ 0 — streaming download + unpack.
# Потоково читает ВСЕ шарды текущего KEY (любой хост/время) и распаковывает
# в ORBLIB_DIR без локальной копии tar. Дедуп по имени .npz ⇒
# совпадение имени = те же параметры модели, существующий не трогаем
# (tar --skip-old-files). Затем snapshot before.
# --------------------------------------------------------------
download_orblib_shards() {
    mkdir -p "$ORBLIB_DIR"

    local shards
    shards=$(rclone lsf "${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}" \
                --config "${RCLONE_CONF_DIR}/rclone.conf" \
                2>>"$LOGFILE" \
             | grep -E "^orblib_${KEY}__.*\.tar$" || true)

    if [ -z "$shards" ]; then
        log "  Шардов orblib_${KEY}__*.tar в ${ORBLIB_REMOTE_DIR} нет — старт с чистой библиотекой"
    else
        local n_shards
        n_shards=$(echo "$shards" | grep -c "")
        log "  Найдено шардов для KEY=${KEY}: ${n_shards}"
        local shard before after n_rollback
        while IFS= read -r shard; do
            log "  Потоковая распаковка шарда: $shard"
            before=$(mktemp "${ORBLIB_DIR}/.unpack_before_XXXXXX")
            after=$(mktemp "${ORBLIB_DIR}/.unpack_after_XXXXXX")
            (cd "$ORBLIB_DIR" && ls -1 -- *.npz 2>/dev/null || true) | sort > "$before"
            if rclone cat "${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}/${shard}" \
                    --config "${RCLONE_CONF_DIR}/rclone.conf" \
                    --stats-one-line 2>>"$LOGFILE" \
                 | tar -xf - -C "$ORBLIB_DIR" --skip-old-files \
                        2>>"$LOGFILE"; then
                log "    ✓ Потоково распакован: $shard"
            else
                (cd "$ORBLIB_DIR" && ls -1 -- *.npz 2>/dev/null || true) | sort > "$after"
                n_rollback=$(comm -13 "$before" "$after" | wc -l)
                while IFS= read -r name; do
                    [ -n "$name" ] && rm -f -- "${ORBLIB_DIR}/${name}"
                done < <(comm -13 "$before" "$after")
                log "    ✗ Ошибка потоковой распаковки: $shard; удалено новых файлов: ${n_rollback}"
            fi
            rm -f "$before" "$after"
        done <<< "$shards"
    fi

    # Snapshot: список .npz ДО запуска расчёта (для snapshot-diff в финале)
    (cd "$ORBLIB_DIR" && ls -1 -- *.npz 2>/dev/null || true) | sort > "$SNAP_BEFORE"
    local n_before
    n_before=$(wc -l < "$SNAP_BEFORE")
    log "  Библиотек .npz в ${ORBLIB_DIR} до запуска: ${n_before}"
}

# --------------------------------------------------------------
# upload_orblib_shard: ФИНАЛ — multipart snapshot-diff upload.
# Проверенный uploader разбивает новые .npz на потоковые tar-части, проверяет
# size+MD5 и атомарно публикует каждую часть без локальных копий tar.
# --------------------------------------------------------------
SHARD_UPLOADED=0
upload_orblib_shard() {
    [ "$SHARD_UPLOADED" -eq 1 ] && return 0
    [ -f "$SNAP_BEFORE" ] || { log "  Нет snapshot before — пропуск upload шардов"; return 0; }

    local uploader="${WORK_DIR}/upload_orblib_parts.sh"
    if [ ! -x "$uploader" ]; then
        log "  ✗ Multipart uploader не найден или не исполняемый: ${uploader}"
        return 1
    fi

    log "  Multipart upload: шаблон=${SHARD_PATTERN}, part=${ORBLIB_PART_SIZE_GB} GB, timeout=${ORBLIB_UPLOAD_TIMEOUT}"
    if "$uploader" \
            --snapshot="$SNAP_BEFORE" \
            --orblib-dir="$ORBLIB_DIR" \
            --remote-root="${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}" \
            --rclone-config="${RCLONE_CONF_DIR}/rclone.conf" \
            --host="$HOSTNAME_ENV" \
            --timestamp="$TIMESTAMP" \
            --part-size-gb="$ORBLIB_PART_SIZE_GB" \
            --attempts="$ORBLIB_UPLOAD_ATTEMPTS" \
            --timeout="$ORBLIB_UPLOAD_TIMEOUT" \
            --apply 2>&1 | tee -a "$LOGFILE"; then
        SHARD_UPLOADED=1
        log "  ✓ Все multipart-шарды библиотек орбит опубликованы"
        return 0
    fi

    log "  ✗ Multipart upload не завершён; .npz и snapshot сохранены для повторной попытки"
    return 1
}

# --------------------------------------------------------------
# run_container: запуск одного Docker-контейнера
# --------------------------------------------------------------
run_container() {
    local sfx="$1"
    local cpu_start="$2"
    local cpu_end="$3"
    local threads="$4"
    local flags="${PROC_FLAGS[$sfx]}"
    local proc_log="${WORK_DIR}/dockerlog_${sfx}_${EXP_ID}_i${INCL}_${TIMESTAMP}.log"
    local exit_code=0

    log "  Контейнер $sfx: CPU=${cpu_start}-${cpu_end} flags='$flags'"

    docker run --rm \
        --name "agama_orblib_${HOSTNAME_ENV}_${sfx}" \
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
        -e OMP_NUM_THREADS="${threads}" \
        -e OMP_PROC_BIND="close" \
        -e OMP_PLACES="cores" \
        -e MKL_NUM_THREADS="${threads}" \
        -e OPENBLAS_NUM_THREADS="${threads}" \
        -e NUMEXPR_NUM_THREADS="${threads}" \
        \
        -w /workspace \
        "${IMAGE}" \
        python3 -u "/workspace/${CALC_SCRIPT}" \
            --incl "${INCL}" \
            --suffix "${sfx}" \
            --orblib-dir /workspace/orblib \
            $EXP_FLAGS \
            $flags \
            $EXTRA_ARGS \
        2>&1 | tee "$proc_log" \
        || exit_code=$?

    # --- Немедленное объединение истории после завершения контейнера ---
    local merge_label
    if [ $exit_code -eq 0 ]; then
        touch "${WORK_DIR}/.done_orblib_${sfx}"
        merge_label="RESULT-OK: incl=${INCL}, exp=${EXP_ID}, suffix=${sfx}, host=${HOSTNAME_ENV}"
        log "  ✓ Контейнер $sfx завершён успешно — объединяем файлы"
    else
        merge_label="RESULT-ERR(${exit_code}): incl=${INCL}, exp=${EXP_ID}, suffix=${sfx}, host=${HOSTNAME_ENV}"
        log "  ✗ Контейнер $sfx завершён с кодом $exit_code — объединяем частичные файлы"
        # Немедленное уведомление о падении ЭТОГО контейнера — не ждём
        # финального summary (который может не наступить, если упадут все).
        local err_msg="Контейнер ${sfx} (${HOSTNAME_ENV}, exp=${EXP_ID}, incl=${INCL}) завершился с ошибкой (код ${exit_code}). Лог: dockerlog_${sfx}_${EXP_ID}_i${INCL}_${TIMESTAMP}.log"
        notify "$err_msg" "high"
    fi

    merge_proc_files "$sfx" "$merge_label"

    # Загружаем обновлённый общий файл истории на Яндекс.Диск
    (
        flock -x 200
        upload_to_yadisk "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}.txt"
    ) 200>"${WORK_DIR}/.upload_lock_orblib"

    # Диагностический лог процесса (log_*) и лог контейнера — заливаем
    # ВСЕГДА (в т.ч. при ошибке), чтобы traceback был на Я.Диске независимо
    # от исхода остальных контейнеров/финальных шагов оркестратора.
    upload_to_yadisk "${WORK_DIR}/log_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt"
    upload_to_yadisk "$proc_log"

    return $exit_code
}

# ==============================================================
# ПРОВЕРКИ
# ==============================================================
log "======================================================"
log "AGAMA orblib_exp параллельный запуск (Docker)"
log "  hostname         = $HOSTNAME_ENV"
log "  script           = $CALC_SCRIPT"
log "  incl             = $INCL"
log "  EXP_ID           = $EXP_ID"
log "  KEY (orblib)     = $KEY"
log "  shard pattern    = $SHARD_PATTERN"
log "  orblib dir       = $ORBLIB_DIR"
log "  orblib remote    = ${RCLONE_REMOTE}:${ORBLIB_REMOTE_DIR}"
log "  resume           = $RESUME"
log "  shutdown         = $DO_SHUTDOWN"
log "  vCPU всего       = $N_VCPU"
log "  Процессов        = $N_PROC"
log "  Раскладка ядер   = ${CPU_RANGES[*]}"
log "  Потоков/процесс  = ${THREADS_ARR[*]}"
log "  Размер tar-части = ${ORBLIB_PART_SIZE_GB} GB"
log "  Timeout upload   = ${ORBLIB_UPLOAD_TIMEOUT}"
log "  Попыток/часть    = ${ORBLIB_UPLOAD_ATTEMPTS}"
log "======================================================"

[ -f "${RCLONE_CONF_DIR}/rclone.conf" ] \
    || die "rclone не настроен: ${RCLONE_CONF_DIR}/rclone.conf"

for f in "${CALC_SCRIPT}" table3.dat; do
    [ -f "${WORK_DIR}/${f}" ] || die "не найден ${WORK_DIR}/${f}"
done
[ -x "${WORK_DIR}/upload_orblib_parts.sh" ] \
    || die "не найден исполняемый ${WORK_DIR}/upload_orblib_parts.sh"

docker image inspect "$IMAGE" > /dev/null 2>&1 \
    || die "Docker-образ $IMAGE не найден"

notify "Старт orblib_exp на ${HOSTNAME_ENV}, exp=${EXP_ID}, incl=${INCL}"

# ==============================================================
# ШАГ 0: PRE-DOWNLOAD ШАРДОВ БИБЛИОТЕК ОРБИТ
# ==============================================================
log ""
log "ШАГ 0: Pre-download tar-шардов библиотек орбит (KEY=${KEY})"
download_orblib_shards

# ==============================================================
# ШАГ 1: ПОДГОТОВКА ФАЙЛОВ
# ==============================================================
log ""
log "ШАГ 1: Подготовка файлов"

rm -f "${WORK_DIR}"/.done_orblib_*
rm -f "${WORK_DIR}/.upload_lock_orblib"
# Резервации точек прошлого запуска (от мёртвых процессов)
rm -rf "${WORK_DIR}"/reservations_i*

declare -A PROC_FLAGS

if [ $RESUME -eq 1 ]; then
    log "  Режим: возобновление с checkpoint"

    for sfx in "${SUFFIXES[@]}"; do
        for f in \
            "out_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt" \
            "checkpoint_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.pkl"
        do
            [ ! -f "${WORK_DIR}/${f}" ] \
                && { download_from_yadisk "$f" || true; } \
                || log "  Локально есть: $f"
        done
    done

    log "  Объединение файлов прерванных процессов..."
    for sfx in "${SUFFIXES[@]}"; do
        merge_proc_files "$sfx" \
            "RESUME-PRE: incl=${INCL}, exp=${EXP_ID}, suffix=${sfx}, host=${HOSTNAME_ENV}"
    done

    upload_to_yadisk "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}.txt"

    for sfx in "${SUFFIXES[@]}"; do
        delete_from_yadisk "out_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt"
    done

    for sfx in "${SUFFIXES[@]}"; do
        cp_file="${WORK_DIR}/checkpoint_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.pkl"
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
            "CLEANUP: incl=${INCL}, exp=${EXP_ID}, suffix=${sfx}, host=${HOSTNAME_ENV}"
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
    run_container "$sfx" "$cpu_s" "$cpu_e" "${THREADS_ARR[$i]}" &
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

# ВАЖНО: НЕ считать через `ls .../.done_orblib_* | wc -l` — под `set -e -o
# pipefail`, если ни один контейнер не создал .done_orblib_* (все упали),
# `ls` возвращает ненулевой код (glob не раскрылся), pipefail транслирует
# его в статус всего конвейера → `set -e` аварийно валит ВЕСЬ скрипт на
# этой строке, минуя ШАГ 4-6 (в т.ч. финальную заливку лога и уведомление).
# Считаем из уже отслеженного FAILED — без обращения к файловой системе.
DONE_COUNT=$((N_PROC - FAILED))
log "  Успешно: ${DONE_COUNT}/${N_PROC}, ошибок: ${FAILED}"
# Промежуточная заливка мастер-лога — чтобы результат был на Я.Диске уже
# здесь, даже если что-то в ШАГ 4-6 позже пойдёт не так.
upload_to_yadisk "$LOGFILE"

# ==============================================================
# ШАГ 4: ФИНАЛЬНАЯ ПРОВЕРКА ОБЪЕДИНЕНИЯ
# ==============================================================
log ""
log "ШАГ 4: Финальная проверка объединения файлов..."
for sfx in "${SUFFIXES[@]}"; do
    out_f="${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt"
    if [ -f "$out_f" ]; then
        log "  Найден необъединённый файл $sfx — объединяем"
        merge_proc_files "$sfx" \
            "FINAL-CHECK: incl=${INCL}, exp=${EXP_ID}, suffix=${sfx}, host=${HOSTNAME_ENV}"
    fi
done

# ==============================================================
# ШАГ 5: MULTIPART UPLOAD БИБЛИОТЕК ОРБИТ (snapshot-diff)
# ==============================================================
log ""
log "ШАГ 5: Multipart upload новых библиотек орбит..."
upload_orblib_shard

# ==============================================================
# ШАГ 6: ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ ИСТОРИИ
# ==============================================================
log ""
log "ШАГ 6: Финальная синхронизация на Яндекс.Диск..."

(
    flock -x 200
    upload_to_yadisk "${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}.txt"
) 200>"${WORK_DIR}/.upload_lock_orblib"

for sfx in "${SUFFIXES[@]}"; do
    delete_from_yadisk "out_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.txt"
    delete_from_yadisk "checkpoint_${HOSTNAME_ENV}_${EXP_ID}_${sfx}.pkl"
done

upload_to_yadisk "$LOGFILE"
rm -f "${WORK_DIR}/.upload_lock_orblib"
rm -f "$SNAP_BEFORE"

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
out_final="${WORK_DIR}/out_${HOSTNAME_ENV}_${EXP_ID}.txt"
if [ -f "$out_final" ]; then
    lines=$(grep -v "^#" "$out_final" | grep -c "" 2>/dev/null || echo 0)
    size=$(du -h "$out_final" | cut -f1)
    log "  Итоговый файл: $(basename "$out_final"): $lines строк данных, $size"
fi
n_orblib=$( (cd "$ORBLIB_DIR" 2>/dev/null && ls -1 -- *.npz 2>/dev/null || true) | wc -l)
log "  Библиотек .npz в ${ORBLIB_DIR}: ${n_orblib}"
log "======================================================"

notify \
    "orblib_exp завершено ${DONE_COUNT}/${N_PROC}, ошибок: ${FAILED}, exp=${EXP_ID}, incl=${INCL}" \
    "$([ $FAILED -eq 0 ] && echo high || echo urgent)"

# ==============================================================
# ШАГ 7: ВЫКЛЮЧЕНИЕ VM
# ==============================================================
if [ $FAILED -eq 0 ]; then
    schedule_shutdown 1 "orblib_exp завершён успешно (incl=${INCL}, exp=${EXP_ID})"
else
    schedule_shutdown 5 "orblib_exp завершён с ошибками: ${FAILED}/${N_PROC} (incl=${INCL}, exp=${EXP_ID})"
fi

[ $FAILED -eq 0 ] && exit 0 || exit 1
