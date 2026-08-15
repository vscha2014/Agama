# Questions for the PI
Внимание!!! В коде Fornax_P21_symm_PCA_w3Sersic_yaVM.py обнаружена ошибка при
задании параметра posang    = 46.8 - не верно!!
должно быть: posang    = 42.3 # 4Sersic, Wang et al 2019  https://doi.org/10.3847/1538-4357/ab31a9 # 46.8 old Battaglia, G., Tolstoy, E., Helmi, A., et al. 2006, A&A, 459, 423
Также замени q_ap      = 0.7
на q_ap      = 1 - 0.31
В связи с этим переименованы все файлы с 4UpsBoTorch_PCA_Sersic_*.txt на 4UpsBoTorch_PCA_PA46.8_Sersic_*.txt и аналогично файлы результатов 4result.
В новых расчетах с правильным posang оставляем создание файлов с названиями  4UpsBoTorch_PCA_Sersic_*.txt и 4result_BoTorch_PCA_Sersic_*.txt
Для старта возможно использовать статистику 4UpsBoTorch_PCA_PA46.8_Sersic_*.txt для выбора начальных точек, но при этом все penalty надо пересчитывать
для построения PCA-модели.

Open questions encountered while reading the project (Stage-0). Do not guess
answers; record PI replies here when received.

> **Status (Stage-1, implemented):** Q1–Q14 answered by PI (see inline `Ответ:`).
> Applied in code: `posang=42.3`, `q_ap=1-0.31`, opt-in `--init-from-pa468`
> seeding with penalty recompute, production `pca_update_interval=12` (Q14).
> RNG policy: torch per-process seed (logged) + per-process `proc_rng` for
> optimizer seeding; **GH observation-error bootstrap restored to fixed `seed=42`**
> (identical realization across workers). Docs updated.
> **Goal 0 (duplicate parallel calculations):** investigated + partly fixed.
> Finding: for an unexplored incl, all workers recomputed identical initial
> points (deterministic `select_bootstrap_candidates` + LHS `seed=42`). Fix:
> per-process `proc_rng` decorrelates bootstrap selection + LHS (overlap 100%→~40%
> in a 40→12 test); main loop already decorrelated by torch seed. **Open
> recommendation:** a hard preventive reservation file (reserve-before-evaluate,
> incl. cross-host) for a strict guarantee — not built; needs PI go-ahead.
> **Two resolved corrections:**
> - Sersic index uncertainty is **±0.006** (per `Wang_2019_table1_apjab31a9t1_ascii.txt`),
>   not ±0.06 as typed in the Q10 answer below. Used ±0.006 in docs.
> - `check_imports.py` is now present in the repo root (Dockerfile step 13 OK).
>
> **Remaining (not blocking):** hard reservation harness (Goal 0, above) and any
> `bounds_original` extension (ask first).

## Penalty / statistics

1. **Exact interpretation of `penalty`** (col 6 of `4UpsBoTorch_*` files): it is
   `sum(penalties[1])` — the GH-kinematic term from `KinemDatasetGH.getPenalty`
   after solving orbit weights with `mult = sqrt(num_dof)*10` scaling. Can it be
   treated as `chi^2` (per what normalization), as `-2 log L`, or only as a
   relative ranking score? This determines how "80% confidence" boundaries and
   weighted J-factor histograms should be computed.

Ответ:  penalties coudn't be treated as chi^2

2. What weights are intended for the **weighted J-factor histogram** — uniform
   over good models, exp(-penalty/2), or something else?
   Ответ: как в коде J_factor_Sersic_Fornax_P21_symm.py
    это взвешенная KDE (kernel density estimation) плотности точек в пространстве двух параметров, где вес каждой точки 
      pen_min   = penalties.min()
    pen_sigma = max(penalties.std(), 1e-6)
    weights   = numpy.exp(-(penalties - pen_min) / pen_sigma)
    weights  /= weights.sum()

3. Is there a **prior on `incl`** (e.g. from photometric flattening limits
   `q_ap` vs `cos i`, or literature), or should `incl` be treated as uniform
   within its admissible range in the future incl-as-parameter analysis?

Ответ: внутреннее отношение осей видимой части галактики
axRZst  = (q_ap2 - cosbeta**2)**0.5/sinbeta, где q_ap2=q_ap^2, q_ap=1-0.31 (из наблюдений)
cosbeta и sinbeta - cos(incl) и sin(incl), т.е. cos(incl) < q_ap

## Result files

4. Which result files are **canonical**: the merged per-host files
   `4UpsBoTorch_PCA_Sersic_<host>.txt` (+ the Yandex.Disk copies), and are the
   `_p0.._p3` files purely **transient** (safe to ignore in analysis)? The
   committed `_pN` examples — keep as format references?

   Ответ: использовать все файлы. сейчас идет расчет на ya VM, в конце они будут
   соеденины в общий файл, но сейчас они также годны к исследованию

5. What is `4UpsBoTorch_Sersic.txt` (legacy, no `_PCA_`) referenced in
   `storage_patterns` — same column layout? Should new code still read it?

   Ответ: да, файл в том же формате, но создавался другим кодом

6. Are the `4result_BoTorch_*` log files used by any analysis, or
   monitoring-only?

   Ответ: monitoring-only

7. Readers concatenate all matching files without cross-file dedup; can the same
   data block legitimately exist in files of two hosts after Yandex.Disk syncing
   (i.e., should future analysis dedup by row hash)?

   Ответ: нет, не надо делать дедупликацию имеющихся данных, тем более с
   разных хостов. Это уникальные результаты эксперимента

8. Is a **metadata sidecar** (e.g. a small YAML/JSON next to each result file
   with seed, code version, run id, PCA-space id) acceptable for future tasks,
   provided the existing file formats stay untouched?
   
Ответ: да

## Parameters / bounds / inputs

9. Are `bounds_original` (`Q∈(0.05,2.5)`, `gh∈(0.0,1.6)`, `rh∈(0.5,3.5)`,
   `rho0∈(34,120)`, `Upsilon∈(0.1,1.6)`) the current **production bounds**?
   Goal 2 mentions possibly extending the explored space — documentation may
   recommend extensions, but who signs off changes in production code?

   Ответ: по согласованию с пользователем, спроси меня, перед тем, как предложить
   изменения

10. Which inputs are **frozen**: `py/table3.dat`, the Sersic parameters
    (`Sersic_m=0.80`, `massSt=14.0`, scale radius 16.4', `q_ap`, posang 46.8°),
    distance `D=143 kpc`? Goal 3 will vary Sersic parameters within literature
    errors — should that be done via CLI overrides in new code rather than
    edits to the script?

    Ответ: как указано вначале, надо сразу же заменить 46.8 на 42.3 в том же коде.
    massSt=14.0 не меняем, distance `D=143 kpc не меняем
     Далее код по влиянию Sersic-параметров создавай отдельно. Эти значения
    и границы изменения параметров есть в таблице в файле Wang_2019_table1_apjab31a9t1_ascii.txt
    в колонке для Sersic: Sersic_m=0.80 \pm 0.06, scale radius = 16.4 \pm 0.2
    q_ap = 1 - Ellipticity, Ellipticity = 0.31 \pm 0.002;  posang = 42.3 \pm 0.2

11. `numpy.random.seed(42)` is fixed in the production script: identical
    parallel processes could in principle produce correlated/duplicate orbit
    libraries. Intentional (reproducibility) or a known caveat for Goal 0?

Ответ: это лучше исправить, раздать параллельным скриптам разный seed

## Infrastructure

12. The J-factor script hardcodes `YADISK_DIR=/home/gala/Yandex.Disk/galAgama`
    — on which machine is the analysis normally run, and is a configurable path
    acceptable in a future version?

Ответ: анализ данных происходит на локальном компьютере, поэтому директория
с именем пользователя hard-corded

13. The Dockerfile for `agama:latest` is not in the repo — where is it
    maintained (needed to document exact production dependency versions)?

Ответ: в корень проекта положены файлы Dockerfile, check_agama.py, entrypoint.sh

14. `4result` headers say `Iterations planned: 40` while checkpoint messages use
    `n_iter` from the call site — confirm 40 iterations/process is the current
    production setting (and ~15 → one PCA update + buffer per run?).
    
Ответ: предложи оптимальные настройки для количества итераций и обновлений PCA

## Оптимизация Brent-поиска по Upsilon (требует одобрения PI — контракт §7)

15. **Ускорение минимизации `Upsilon`.** Профилирование (`4UpsBoTorch_PCA_Sersic_*_p0.txt`,
    incl=90, 6 оценок) показывает: `solveOpt` равномерно дорогой (3.6–5.1 с, среднее
    ~4.2 с), поэтому wall-время ≈ `N_проб × 4.2 с` (10–13 проб, ~44–52 с/оценку).
    Найденный `Upsilon*` очень устойчив: 0.604…0.633 (≈0.617 ± 0.011) по всему
    пространству гало-параметров. Источники потерь, видимые в каждой оценке:
    - фиксированные стартовые пробы золотого сечения на (0.1,1.6): `Upsilon=1.027`
      всегда даёт penalty 119–136, `Upsilon=0.454` — 31–50 (≈2 бесполезные пробы,
      ~9 с/оценку);
    - хвост: последние 2–4 пробы уточняют `Upsilon` в 3–4-м знаке при
      Δpenalty < 1e-3 (упор в `xatol=1e-3`).

    Текущее состояние (по запросу пользователя п.1, п.2, п.3 включены ПОСТОЯННО,
    без флагов, в `py/Fornax_P21_symm_PCA_w3Sersic_yaVM_timed.py`):

    - **п.1 — адаптивная узкая скобка** (`UPS_BRACKET_DELTA=0.1`, `UPS_BRACKET_NMED=8`):
      поиск в окне `[Ups0−δ, Ups0+δ]`, где `Ups0` — медиана недавних найденных
      `Upsilon*`; **откат к полному (0.1,1.6)**, если минимум упёрся в край узкой
      скобки (сохраняет корректность). Ожидается ~11 → ~5–6 проб.
    - **п.2 — толеранс Brent** (`UPS_XATOL=5e-3`, было 1e-3): у минимума penalty
      плоский (вторая производная ~1200 из лога), при `5e-3` максимальная ошибка
      записываемого penalty ~0.015 (<2% `pen_sigma` KDE), при этом срезаются 2–3
      хвостовые пробы. (1e-2 дало бы ошибку ~0.06 → отвергнуто.)
    - **п.3 — субдискретизация орбит** (`UPS_SUBSAMPLE_FRAC=0.25`): внутренний поиск
      `Upsilon` на подвыборке орбит, затем ОДИН продуктивный full-solve на полной
      библиотеке (100k) в найденной точке. **Записываемый penalty контрактно-точный**;
      финальный full-solve вызывается в обход `logger` → **не попадает в историю
      оптимизации `Upsilon`** (история содержит только субдискретизированные пробы).

    Совокупно (п.1+п.2+п.3): ожидается заметное ускорение Upsilon-поиска. **Перенос
    в основной production-скрипт `Fornax_P21_symm_PCA_w3Sersic_yaVM.py` — отдельно,
    по согласованию.**

    **Затрагивается контракт §7** (границы/толеранс Brent, penalty-path). Прошу
    согласовать включение в production. **План валидации до внедрения:** прогнать N
    контрольных точек с флагами и без, сравнить найденные `Upsilon*`, `penalty` и
    итоговый J-фактор (требование — совпадение `Upsilon*` в пределах `xatol`,
    `penalty` без значимого сдвига).

Ответ:

## Метод оценки J-фактора: взвешивание и плотность сэмплирования (контракт §7)

16. **Учитывать ли при оценке J не только качество (`penalty`), но и плотность
    сэмплирования точек оптимизатором?**

    Текущая схема (см. Q2) взвешивает точки только по качеству:
    `w = exp(-(penalty - pen_min)/pen_sigma)`. Но выборка `θ_i` получена
    **оптимизатором (BoTorch/TuRBO)**, а не равномерным/случайным сэмплированием:
    поисковая плотность `q(θ)` сама сгущается у оптимума. Корректная оценка
    апостериорного интеграла требует importance-веса
    `ω_i = π(θ_i)·L(θ_i)/q(θ_i)`; текущая схема неявно полагает `q ∝ π`
    (равномерно), что для оптимизатора неверно. В результате точки у оптимума
    «подавляют» соседние **и качеством, и количеством** — оценка J смещается к
    best-fit, а её разброс занижается.

    Подтверждение в данных: в `py/analise/J_factor_Sersic_incl*_theta0.5.txt`
    есть **точные дубликаты строк параметров** (один и тот же `θ`, а значит одно
    и то же `J`, входит во взвешенное среднее многократно — вес по частоте
    сэмплирования). Замечание: `AGENTS.md §7` «never dedup» — про **хранение**
    файлов как уникальных экспериментов; здесь же речь о **статистической
    обработке** для постериора по J (дубликаты не являются независимыми
    наблюдениями). Прошу подтвердить, что это не конфликтует.

    Кандидаты метода (затрагивают контракт §7 — определение взвешивания и оценки
    погрешности J):
    - **(A) Профиль по penalty / уровни Δ:** центр J — при `min penalty`,
      интервал — диапазон J на `{θ: penalty ≤ pen_min + Δ}`. Не зависит от
      плотности сэмплирования; но требует калибровки Δ (а PI указал в Q1, что
      `penalty` нельзя трактовать как χ² → Δ некалиброван).
    - **(B) Importance sampling с коррекцией на плотность:**
      `ω_i ∝ L(θ_i)/q̂(θ_i)`, где `q̂` — оценка плотности сэмплирования из самих
      точек (kNN в пространстве поисковых параметров `Q, gh, rh, rho0`). Это и
      есть «вес на плотность точек», формализованный в совместном пространстве.
    - **(C) Полноценный постериор (MCMC/nested) по суррогату penalty** — строго,
      но дорого.

    **Вопрос PI:** какой метод принять для итоговой оценки J и её погрешности
    (текущий «только-penalty», A, B или C)? До решения подготовлен **только
    диагностический** скрипт `py/diagnose_J_weighting.py`, который на
    тех же данных сравнивает текущий метод, A и B (со схлопыванием дублей перед
    статистикой) — он ничего не меняет в продакшене и не трогает исходные файлы.

Ответ:
