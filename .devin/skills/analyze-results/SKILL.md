---
name: analyze-results
description: Analyse a Fornax run history (out_*/4Ups* files) — best model, boundary hits, penalty statistics, parameter degeneracies, comparison between runs. Use when the user asks what a run found or whether the search space is sufficient.
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

Анализ истории оценок. Ничего не пересчитывать на AGAMA, файлы результатов не
менять и не дедуплицировать.

## 1. Данные

Строка истории: `incl Q gh rh rho0 Upsilon penalty timestamp` (строки с `#` —
комментарии, PCA-координаты, история Upsilon, тайминги). Файлы:
`results/<run_id>/raw/out_*.txt`, `py/out_*.txt`, `py/4UpsBoTorch_PCA_Sersic_*`
(канонические; архив `*_PA46.8_*` — отдельный, старая геометрия).
Большинство из них под `.gitignore` → читать только через `exec`:

```bash
.venv-ai/bin/python - <<'PY'
import numpy, glob
rows=[]
for f in glob.glob('results/<run_id>/raw/out_*.txt'):
    for ln in open(f):
        p=ln.split()
        if ln.startswith('#') or len(p)<7: continue
        try: rows.append([float(x) for x in p[:7]])
        except ValueError: pass
a=numpy.array(rows); print(len(a), a[a[:,6].argmin()])
PY
```

## 2. Что сообщать

- N моделей, набор `incl` и `Q`, лучшая модель (все 7 чисел) и `rho0*Upsilon`.
- **Упор в границы**: для каждого параметра доля строк на границе и доля среди
  лучших (Δpenalty ≤ 0.25 и ≤ 0.5). Минимум на границе = поиск не замкнут;
  сравнения penalty с другими прогонами при этом некорректны.
- Квантили penalty (min/p10/median/p90), по каждому `incl` отдельно.
- Вырождения: постоянство произведений (`rho0*Upsilon`), корреляции у лучших.
- Дубликаты параметров: считать их для *статистики* (они не независимые
  наблюдения), но не удалять из файлов.

## 3. Правила интерпретации

- `penalty` — не χ²; никаких «σ» и доверительных интервалов из Δpenalty.
- Penalty сравним между `incl`, но **не** между разными `double`/`n_bin`
  (иное число апертур/ограничений; при необходимости сравнивать нормировку на
  число ограничений).
- Внутренний поиск `Upsilon` в `orblib_exp` идёт на подвыборке орбит с финальным
  полным solve; при сравнении лучших точек учитывать этот источник расхождения.
- Расширение границ — **только рекомендация**: оформить как вопрос в
  `doc/ai/questions_for_pi.md`, не править `bounds_original`. Предлагать
  профильные прогоны при фиксированном параметре как дешёвую альтернативу
  широкому повторному поиску.

## 4. Запись

Итог — в `results/<run_id>/NOTES.md` (результат, диагноз границ, follow-ups) и
строкой в `results/REGISTRY.md`. Если меняется общая картина — обновить
`doc/ai/STATUS.md`. Для J-фактора использовать `/jfactor-report`.
