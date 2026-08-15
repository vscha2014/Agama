#!/bin/bash
# run_single.sh — запуск одного потока с поддержкой resume
#
# Использование:
#   bash run_single.sh [--incl=71.85] [--suffix=p0] [--resume]

set -euo pipefail

# ==============================================================
# ПАРАМЕТРЫ
# ==============================================================
SCRIPT="Fornax_P21_symm_PCA_w3Sersic.py"
CONDA_ENV="agamaBoTorch"
REMOTE_DIR="galAgama"
RCLONE_REMOTE="${RCLONE_REMOTE:-yandex}"
HOSTNAME_ENV="${HOSTNAME_SUFFIX:-$(hostname)}"

# --- Разбор аргументов ---
INCL="71.85"
SUFFIX="p0"
RESUME=0          # по умолчанию — старт с нуля
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --incl=*)   INCL="${arg#*=}"            ;;
        --suffix=*) SUFFIX="${arg#*=}"          ;;
        --resume)   RESUME=1                    ;;
        *)          EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# --- Имена файлов ---
UPS_PROC="4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}_${SUFFIX}.txt"
RES_PROC="4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}_${SUFFIX}.txt"
UPS_HOST="4UpsBoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
RES_HOST="4result_BoTorch_PCA_Sersic_${HOSTNAME_ENV}.txt"
CHECKPOINT="checkpoint_${HOSTNAME_ENV}_${SUFFIX}.pkl"
LOGFILE="run_i${INCL}_${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

# ==============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

die() {
    log "ОШИБКА: $*"
    exit 1
}

show_file_status() {
    log "--- Состояние файлов ---"
    for f in "$UPS_PROC" "$RES_PROC" \
             "$UPS_HOST" "$RES_HOST" \
             "$CHECKPOINT"; do
        if [ -f "$f" ]; then
            lines=$(grep -c "" "$f" 2>/dev/null || echo "?")
            size=$(du -h "$f" 2>/dev/null | cut -f1)
            log "  ✓ $f  ($lines строк, $size)"
        else
            log "  ✗ $f  (не найден)"
        fi
    done
    log "------------------------"
}

download_from_yadisk() {
    local fname="$1"
    local remote_path="${RCLONE_REMOTE}:${REMOTE_DIR}/${fname}"
    log "  Скачиваем: $fname"
    if rclone copyto "$remote_path" "$fname" \
              --stats-one-line 2>>"$LOGFILE"; then
        log "  ✓ Скачан: $fname"
        return 0
    else
        log "  ~ Не найден на Яндекс.Диске: $fname"
        return 1
    fi
}

append_and_remove() {
    local src="$1"
    local dst="$2"
    local label="$3"

    [ -f "$src" ] || return 0

    local src_lines
    src_lines=$(grep -c "" "$src" 2>/dev/null || echo 0)
    if [ "$src_lines" -eq 0 ]; then
        log "  Пропуск (пустой): $src"
        rm -f "$src"
        return 0
    fi

    # --- Проверка по MD5-хешу содержимого ---
    local src_hash
    if ! src_hash=$(grep -v "^#" "$src" \
               | grep -v "^[[:space:]]*$" \
               | md5sum 2>/dev/null | cut -d' ' -f1); then
        log "  ОШИБКА: не удалось вычислить хеш для $src"
        return 1
    fi

    # Точный поиск хеша в формате HASH:xxx, только целые строки
    if grep -q "^# HASH:${src_hash}$" "$dst" 2>/dev/null; then
        log "  Пропуск: содержимое $src уже добавлено (hash=${src_hash})"
        rm -f "$src"
        return 0
    fi

    log "  $src → $dst ($src_lines строк, hash=${src_hash})"
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
        log "  ✓ Объединено и удалено: $src"
    else
        log "  ОШИБКА записи в $dst — $src сохранён!"
        return 1
    fi
}

upload_to_yadisk() {
    local filepath="$1"
    [ -f "$filepath" ] || { log "  Пропуск загрузки: $filepath"; return 0; }
    local remote_path="${RCLONE_REMOTE}:${REMOTE_DIR}/${filepath}"
    if rclone copyto "$filepath" "$remote_path" \
              --stats-one-line 2>>"$LOGFILE"; then
        local size
        size=$(du -h "$filepath" | cut -f1)
        log "  ✓ Загружено: $filepath ($size)"
    else
        log "  ✗ Ошибка загрузки: $filepath"
    fi
}

delete_from_yadisk() {
    local filepath="$1"
    local remote_path="${RCLONE_REMOTE}:${REMOTE_DIR}/${filepath}"
    rclone deletefile "$remote_path" 2>>"$LOGFILE" \
        && log "  ✓ Удалено с Яндекс.Диска: $filepath" \
        || log "  ~ Не найдено на Яндекс.Диске: $filepath"
}

# ==============================================================
# СТАРТ
# ==============================================================
log "======================================================"
log "ЗАПУСК: $SCRIPT"
log "  incl     = $INCL"
log "  suffix   = $SUFFIX"
log "  resume   = $RESUME"
log "  hostname = $HOSTNAME_ENV"
log "  лог      = $LOGFILE"
log "======================================================"

# ==============================================================
# ШАГ 1: ПОДГОТОВКА — зависит от режима запуска
# ==============================================================
log ""
log "ШАГ 1: Подготовка (resume=$RESUME)"

if [ $RESUME -eq 1 ]; then
    # ----------------------------------------------------------
    # РЕЖИМ RESUME: восстановление после прерывания
    # ----------------------------------------------------------
    log "  Режим: возобновление с checkpoint"
    log ""

    # 1а. Диагностика текущего состояния
    show_file_status

    # 1б. Скачиваем с Яндекс.Диска то, чего нет локально
    log "  Синхронизация с Яндекс.Диском..."
    for f in "$UPS_PROC" "$RES_PROC" "$CHECKPOINT"; do
        if [ ! -f "$f" ]; then
            download_from_yadisk "$f" || true
        else
            log "  Локально есть: $f"
        fi
    done

    # 1в. Объединяем _proc файлы в общие
    #     (частичные результаты прерванного запуска)
    log ""
    log "  Объединение файлов прерванного запуска..."
    LABEL="RESUME-PRE: incl=${INCL}, suffix=${SUFFIX}, host=${HOSTNAME_ENV}"
    append_and_remove "$UPS_PROC" "$UPS_HOST" "$LABEL"
    append_and_remove "$RES_PROC" "$RES_HOST" "$LABEL"

    # 1г. Загружаем обновлённые общие файлы на Яндекс.Диск
    log ""
    log "  Загрузка общих файлов на Яндекс.Диск..."
    upload_to_yadisk "$UPS_HOST"
    upload_to_yadisk "$RES_HOST"
    delete_from_yadisk "$UPS_PROC"
    delete_from_yadisk "$RES_PROC"

    # 1д. Определяем флаг для Python
    if [ -f "$CHECKPOINT" ]; then
        log ""
        log "  Checkpoint найден → resume=True для Python"
        PYTHON_RESUME_FLAG=""   # resume=True по умолчанию в Python
    else
        log ""
        log "  ВНИМАНИЕ: checkpoint не найден → запуск с нуля"
        PYTHON_RESUME_FLAG="--no-resume --delete-checkpoint"
    fi

    log ""
    log "  Состояние после подготовки:"
    show_file_status

else
    # ----------------------------------------------------------
    # РЕЖИМ FRESH START: чистый запуск с нуля
    # ----------------------------------------------------------
    log "  Режим: чистый запуск с нуля"

    # Удаляем старые _proc файлы если остались
    for f in "$UPS_PROC" "$RES_PROC"; do
        if [ -f "$f" ]; then
            log "  ВНИМАНИЕ: найден старый файл $f"
            # Сначала добавляем в общий (на случай если не был добавлен)
            LABEL="CLEANUP: incl=${INCL}, suffix=${SUFFIX}, host=${HOSTNAME_ENV}"
            if [ -f "$UPS_PROC" ]; then
                LABEL="CLEANUP: incl=${INCL}, suffix=${SUFFIX}, host=${HOSTNAME_ENV}"
                append_and_remove "$UPS_PROC" "$UPS_HOST" "$LABEL" || true
            fi
            if [ -f "$RES_PROC" ]; then
                LABEL="CLEANUP: incl=${INCL}, suffix=${SUFFIX}, host=${HOSTNAME_ENV}"
                append_and_remove "$RES_PROC" "$RES_HOST" "$LABEL" || true
            fi
        fi
    done

    PYTHON_RESUME_FLAG="--no-resume --delete-checkpoint"
fi

# ==============================================================
# ШАГ 2: АКТИВАЦИЯ CONDA
# ==============================================================
log ""
log "ШАГ 2: Активация conda..."

# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh" \
    || die "Не удалось найти conda"

conda activate "$CONDA_ENV" \
    || die "Не удалось активировать $CONDA_ENV"

log "  Python: $(which python)"

# ==============================================================
# ШАГ 3: ЗАПУСК PYTHON
# ==============================================================
log ""
log "ШАГ 3: Запуск расчёта..."
log "  Команда: nice -n 19 ionice -c 3 python $SCRIPT --incl=$INCL --suffix=$SUFFIX $PYTHON_RESUME_FLAG $EXTRA_ARGS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# shellcheck disable=SC2086
nice -n 19 ionice -c 3 python -u "$SCRIPT" \
    --incl="$INCL"    \
    --suffix="$SUFFIX" \
    $PYTHON_RESUME_FLAG \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOGFILE"

PYTHON_EXIT=${PIPESTATUS[0]}

if [ $PYTHON_EXIT -ne 0 ]; then
    log "ВНИМАНИЕ: Python завершился с кодом $PYTHON_EXIT"
    log "Продолжаем объединение файлов (частичные результаты)..."
else
    log "Python завершился успешно (код 0)"
fi

# ==============================================================
# ШАГ 4: ОБЪЕДИНЕНИЕ ФАЙЛОВ ПОСЛЕ РАСЧЁТА
# ==============================================================
log ""
log "ШАГ 4: Объединение файлов после расчёта..."

LABEL="RESULT: incl=${INCL}, suffix=${SUFFIX}, host=${HOSTNAME_ENV}"
append_and_remove "$UPS_PROC" "$UPS_HOST" "$LABEL"
append_and_remove "$RES_PROC" "$RES_HOST" "$LABEL"

# ==============================================================
# ШАГ 5: СИНХРОНИЗАЦИЯ НА ЯНДЕКС.ДИСК
# ==============================================================
log ""
log "ШАГ 5: Синхронизация на Яндекс.Диск..."

upload_to_yadisk "$UPS_HOST"
upload_to_yadisk "$RES_HOST"

delete_from_yadisk "$UPS_PROC"
delete_from_yadisk "$RES_PROC"
delete_from_yadisk "$CHECKPOINT"

# ==============================================================
# ИТОГ
# ==============================================================
log ""
log "======================================================"
log "ЗАВЕРШЕНО: $(date '+%Y-%m-%d %H:%M:%S')"
log "  Python exit code: $PYTHON_EXIT"
log "  Итоговые файлы:"
for f in "$UPS_HOST" "$RES_HOST"; do
    if [ -f "$f" ]; then
        lines=$(grep -c "" "$f" 2>/dev/null || echo 0)
        size=$(du -h "$f" | cut -f1)
        log "    $f: $lines строк, $size"
    fi
done
log "======================================================"

exit $PYTHON_EXIT
