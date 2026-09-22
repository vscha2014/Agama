---
name: jfactor-report
description: Compute or report the Fornax J-factor — estimators M0/M_A/M_B, penalty weighting, unit conversion, mass histograms and corner plots. Use when the user asks for J-factor values, uncertainties, or wording of the J result.
allowed-tools:
  - read
  - grep
  - glob
  - exec
---

J-фактор и связанные графики. Производственный скрипт
`py/J_factor_Sersic_Fornax_P21_symm.py` — **заморожен** (`doc/ai/CONTRACT.md`):
не править без одобрения PI; новый анализ — отдельным файлом.

## 1. Контракт расчёта

- Для каждой отобранной модели: гало `densitynorm = rho0*Upsilon`, звёзды
  `mass = massSt*Upsilon`, `axRZst = sqrt(q_ap² − cos²i)/sin i`
  (`q_ap = 1 − 0.31`, `massSt = 14.0`, `Sersic_m = 0.80`, `D = 143` кпк).
- Интеграл ρ² вдоль луча зрения по конусам θ ∈ {0.1, 0.2, 0.5, 1.0}°.
- Единицы: массы кода — в 10⁶ M☉;
  `rho_conv = 1e6*1.989e33/1.783e-24/kpc_to_cm**3`,
  `J = J_code * rho_conv**2 * kpc_to_cm`. Ошибка старой конверсии занижала J
  примерно на 40 порядков — проверять эту строку при любом переносе кода.
- Отбор моделей: адаптивный cutoff, лучшие `target_fraction = 0.30`
  (потолок `cutoff_start = 0.60`). Веса
  `w = exp(-(penalty − pen_min)/pen_sigma)`, `pen_sigma = max(std, 1e-6)`.
- Файлы вида `Jcomputed_from_raw_*.txt` с `cutoff_fraction=1.0` содержат и
  очень плохие модели — для итоговых чисел не использовать.

## 2. Оценки

- **M0** (основная): penalty-взвешенная медиана и разброс отобранных
  низко-penalty моделей при заданном `incl`. Устойчива к доле отбора 10–50 %.
- **M_A**: центр в модели с минимальным penalty + диапазон J по отобранным —
  cross-check.
- **M_B**: kNN-коррекция на плотность сэмплирования — эвристика, сдвигает
  результат на 0.03–0.05 dex и меняет знак между `incl`. Только как
  систематическая проверка/приложение, не headline.
- Корректный учёт плотности сэмплирования оптимизатора — открытый вопрос Q16;
  до ответа PI не менять схему взвешивания в продакшене.

## 3. Формулировки (обязательны)

Интервал = «penalty-взвешенный разброс отобранных моделей с малым penalty».
Нельзя называть его байесовским credible interval, 1σ или доверительным
интервалом: `penalty` — не χ². Текущее headline-значение и сравнение с
литературой (Hayashi et al. 2016, `log10 J(0.5°) = 17.90 +0.28/−0.16`,
D = 147 кпк, осесимметричные уравнения Джинса) — см. `doc/ai/STATUS.md` и
`results/legacy_d0_nb250_gh0_ser0/NOTES.md`.

## 4. Графики масс

Гистограммы `M(<1 кпк)` и полной массы: ось X — полная масса в 10⁷ M☉
(`M_code/10`), стек «тёмная материя / звёзды», без взвешивания по penalty
(взвешенный вариант по всем `incl` — `py/mass_histogram_all_incl.py`).
`totalMass()` включает гало до `outercutoffradius = 55` кпк — оговаривать это
при сравнении с `M(<1 кпк)`.

## 5. Запись

Числа и их `run_id` — в `results/<run_id>/NOTES.md` и `results/REGISTRY.md`.
Любое число, попадающее в статью, должно быть прослеживаемо до `run_id`.
