---
name: harness-dev
description: Modify the experimental Fornax harness — Fornax_P21_PCA_w3Sersic_orblib_exp.py, launch_orblib_exp.sh, orblib_storage.py and their tests. Use when the task edits experiment code, adds a run mode, or changes orbit-library storage.
allowed-tools:
  - read
  - edit
  - grep
  - glob
  - exec
---

Правки экспериментального харнесса. Полное описание поведения —
`doc/ai/harness/orblib_exp.md`; научные определения — `doc/ai/CONTRACT.md`.

## Границы

- Редактируемо: `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`,
  `py/launch_orblib_exp.sh`, `py/orblib_storage.py`, `tests/test_orblib_*.py`.
- Не трогать: `*_yaVM.py`, `J_factor_*.py`, `launch_docker_parallel.sh`,
  `src/**`, `schwarzlib.py`, `table3.dat`. Найденный в проде баг → вопрос PI.
- Не запускать: реальные расчёты AGAMA, Docker, rclone/Яндекс.Диск,
  `orblib_storage.py index`, `prune-verified --apply`, shutdown VM.

## Что легко сломать

- `bounds_original` объявлен **дважды** (глобально и в `run_pca_optimization`) —
  менять синхронно; сами значения без одобрения PI не менять.
- `EXP_ID` входит в `hostname_proc` и во все имена файлов: изменение схемы имён
  расщепляет пул PCA и историю. Fixed-Q пишет `Q1d1_*`, свободный — `d1_*`.
- Чтение истории: оба режима читают совместимые файлы всех хостов,
  Q1 фильтрует строки `Q=1`. Строки не удалять, не дедуплицировать.
- Веса PCA — везде `exp(-(penalty − penalty.min())/0.1)` (три места).
- Ключ кэша библиотеки включает `GEOM_HASH` и хэш параметров; сравнение
  метаданных допускает только 1e-12 roundoff.
- Checkpoint хранит физические параметры и репроецируется в перестроенный
  PCA-базис; старые free-Q checkpoint без параметров в streaming-режиме
  отвергаются намеренно.
- Доставка библиотек: при исчерпании попыток — STOP, дорасчёт начатых моделей,
  локальные checkpoint, затем выключение VM. Не добавлять бесконечных ретраев.
- Скрипт монолитный (>4000 строк) с кодом верхнего уровня: тесты извлекают
  функции через `ast`, импортировать модуль нельзя.

## Порядок работы

1. Прочитать релевантный участок и соответствующий раздел
   `doc/ai/harness/orblib_exp.md`; расхождение кода и документа — исправить
   документ в конце задачи.
2. Хирургический diff; ничего рядом не рефакторить.
3. Тест на поведение — в `tests/test_orblib_*.py` (маленькие массивы, mock
   docker/rclone/shutdown, без AGAMA и облака).
4. Проверка:

```bash
cd tests && ../.venv-ai/bin/python -m pytest -q test_orblib_q1.py \
    test_orblib_storage.py --rootdir=. --import-mode=importlib -p no:cacheprovider
cd .. && python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py py/orblib_storage.py
bash -n py/launch_orblib_exp.sh && git diff --check
git diff --stat   # прод-скрипты должны отсутствовать в списке
```

(`python -m pytest` из корня не работает: локальный `py/` перекрывает модуль
`py` самого pytest.)

5. Документировать новое поведение в `doc/ai/harness/orblib_exp.md`,
   решения — в `doc/ai/DECISIONS.md`. **В `AGENTS.md` не писать.**
