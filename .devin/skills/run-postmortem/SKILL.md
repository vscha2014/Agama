---
name: run-postmortem
description: Diagnose a failed or partial Fornax run from launcher/docker logs — crashes, ENOSPC, lost .npz orbit libraries, rclone upload failures. Use when the user shares run logs or asks why a launch died or what was lost.
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Разбор аварийного/частичного прогона. Ничего не запускать в облаке и на VM,
не удалять файлы. Только чтение логов и сверка.

## 1. Собрать вход

Логи лежат в `results/<run_id>/raw/` либо присланы пользователем:
`launch_orblib_*.log` (оркестратор), `dockerlog_p*.log` (stdout контейнеров),
`log_<host>_<EXP_ID>_p*.txt` (лог расчёта), `out_<host>_<EXP_ID>.txt` (история),
`ls.txt` / `ls_orblib.txt` (листинги VM). Файлы под `.gitignore` читай через
`exec cat/grep`, а не инструментом read.

## 2. Чеклист причин (проверять в этом порядке)

- **Диск:** `No space left on device`, `Errno 28`; затем вторичные
  `ValueError: seek of closed file`. Сверить `ls_orblib.txt` с ожидаемым
  объёмом (~0.5 ГБ на библиотеку).
- **`set -euo pipefail` + пустой glob:** `ls .../.done_* | wc -l` валит весь
  launcher, если ни один контейнер не отметился (известный баг, см. Q20).
- **`tee` под pipefail:** расчётный лог дошёл до конца (`40/40`), а dockerlog
  оборван → падение записи лога, а не расчёта. Не считать такой процесс
  упавшим без проверки `out_*`.
- **Облако:** rclone 404/500 от Яндекс.Диска, таймауты (`ORBLIB_FILE_TIMEOUT`),
  исчерпание `ORBLIB_UPLOAD_ATTEMPTS` → STOP-маркер, недоставленная очередь.
- **Наука/данные:** `TargetLOSVD: datacube does not cover all apertures`
  (геометрия сетки), `Weights sum to zero` (PCA-веса), `NotImplementedError`
  для `gh_id>0`/`ser_id>0`.
- **Уведомления:** отсутствие ntfy само по себе — симптом падения оркестратора
  до финальной секции.

## 3. Сверка истории и библиотек

```bash
# ключи моделей из истории против фактически сохранённых .npz
grep -c '^[0-9]' results/<run_id>/raw/out_*.txt        # число строк-моделей
grep -o 'orblib_[^ ]*\.npz' results/<run_id>/raw/log_*.txt | sort -u | wc -l
```
Для потерянных моделей выдать таблицу `gh, rh, rho0, Upsilon, penalty, процесс`,
отсортированную по возрастанию `penalty` (`rho0` — в исходном виде, без
умножения на `Upsilon`). Проверить, уцелела ли библиотека лучшей модели.

## 4. Результат

Записать `results/<run_id>/postmortem.md`: symptom → root cause → damage
(числа: сколько моделей, сколько библиотек, диапазон penalty) → fixed since →
still open. Обновить строку прогона в `results/REGISTRY.md` (`status`) и, если
меняется картина проекта, `doc/ai/STATUS.md`.

Рекомендации по восстановлению формулировать как точечный пересчёт
потерянных параметров, а не повторный полный поиск; перед новым запуском —
`df -h`, `df -i`, размер каталога `orblib`. Найденный баг в **production**
launcher не исправлять: оформить вопрос в `doc/ai/questions_for_pi.md`.
