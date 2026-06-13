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
