#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика взвешивания J-фактора (READ-ONLY, ничего не запускает в AGAMA).

Цель: на уже посчитанных файлах J_factor_Sersic*_theta0.5.txt сравнить
оценку log10(J) тремя способами и показать величину смещения, связанного с
плотностью сэмплирования оптимизатором:

  M0  "current"   : текущая схема — вес только по качеству
                    w = exp(-(penalty - pen_min)/pen_sigma).
                    Считается ДВАЖДЫ: на сырых точках (с дублями) и на
                    схлопнутых дублях — разница показывает вклад "подавления
                    количеством".
  M_A "profile"   : профиль по penalty. Центр = log10(J) при min(penalty);
                    интервал = [min,max] log10(J) на {penalty <= pen_min + dPen}
                    для dPen = k * pen_sigma. Не зависит от плотности точек.
  M_B "density"   : importance sampling с коррекцией на плотность сэмплирования
                    w_i ∝ L_i / q̂(θ_i), q̂ — kNN-плотность в пространстве
                    поисковых параметров (Q, gh, rh, rho0).

ВАЖНО: все статистики (кроме M0-raw) считаются ПОСЛЕ схлопывания точных дублей
строк параметров. Исходные файлы НЕ изменяются.

Это диагностический инструмент к вопросу PI #16 (см. doc/ai/questions_for_pi.md).
Метод итоговой оценки J — решение PI (контракт §7); здесь ничего не меняется.

Источники данных (--source):
  jfactor  (по умолчанию) — готовые файлы
                            J_factors/J_factor_Sersic*_theta0.5.txt
                            (J уже посчитан; AGAMA не нужна).
  raw                     — сырые файлы оптимизатора 4UpsBoTorch_PCA_Sersic*.txt
                            (по умолчанию в /home/gala/Yandex.Disk/galAgama)
                            (incl Q gh rh rho0 Upsilon penalty). J вычисляется
                            ЧЕРЕЗ AGAMA функцией compute_J_factor, скопированной
                            ВЕРБАТИМ из J_factor_Sersic_Fornax_P21_symm.py
                            (контракт §7). Дубли параметров схлопываются ДО
                            расчёта J (J не считается дважды). Результат
                            сохраняется в combined-файл (--save-j) для повторного
                            анализа без AGAMA.
  combined                — ранее сохранённый combined-файл (--combined-file):
                            incl + 9 колонок + count. AGAMA не нужна.

Запуск:
    python3 diagnose_J_weighting.py
    python3 diagnose_J_weighting.py --source raw            # нужен модуль agama
    python3 diagnose_J_weighting.py --source raw --incl 90 --cutoff-fraction 0.30
    python3 diagnose_J_weighting.py --source combined --combined-file Jcomputed_from_raw_theta0.5.txt
    python3 diagnose_J_weighting.py --plot out.pdf
"""

import argparse
import glob
import os
import re

import numpy

# Колонки файла:
# 0 Q  1 gh  2 rh  3 rho0  4 Upsilon  5 rho0_x_Ups  6 penalty
# 7 J_GeV2_cm5  8 log10_J
COL_PARAMS_DEDUP = [0, 1, 2, 3, 4]   # ключ дедупликации (полный набор параметров)
COL_PARAMS_DENSITY = [0, 1, 2, 3]    # поисковые параметры BoTorch (Q, gh, rh, rho0)
COL_PENALTY = 6
COL_LOGJ = 8

INCL_RE = re.compile(r'incl([0-9]+(?:\.[0-9]+)?)')
JFACTOR_EXP_RE = re.compile(
    r'^J_factor_Sersic(?:_(d[01]_nb\d+_gh\d+_ser\d+))?(?:_incl[^_]*)?_theta'
)


# ============================================================
#  ВВОД-ВЫВОД
# ============================================================

def parse_incl_from_name(path):
    m = INCL_RE.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1))


def parse_jfactor_experiment(path):
    match = JFACTOR_EXP_RE.match(os.path.basename(path))
    return match.group(1) if match is not None else None


def load_table(path):
    """Читает 9 старых либо 10 новых (incl + 9) числовых колонок."""
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            try:
                rows.append([float(x) for x in parts[:10]])
            except ValueError:
                continue
    if not rows:
        return numpy.empty((0, 9))
    return numpy.array(rows)


def load_combined(path):
    """
    Читает combined-файл, сохранённый режимом raw:
        incl Q gh rh rho0 Upsilon rho0_x_Ups penalty J_GeV2_cm5 log10_J [count]
    Если колонки count нет — считается count=1.
    """
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) < 10:
                continue
            try:
                vals = [float(x) for x in parts[:11]]
            except ValueError:
                continue
            if len(vals) == 10:
                vals.append(1.0)
            rows.append(vals)
    if not rows:
        return numpy.empty((0, 11))
    return numpy.array(rows)


# ============================================================
#  ЧТЕНИЕ СЫРЫХ ФАЙЛОВ ОПТИМИЗАТОРА + ВЫЧИСЛЕНИЕ J ЧЕРЕЗ AGAMA
#  J-факторная часть скопирована ВЕРБАТИМ из
#  J_factor_Sersic_Fornax_P21_symm.py — НЕ менять без согласования
#  PI (научный контракт §7: формула J и настройки интегрирования).
# ============================================================

# Контрактные константы расчёта J (идентичны J_factor_Sersic_Fornax_P21_symm.py)
ALPHAH = 2.0
BETAH = 3.0
HALO_CUTOFF = 55.0
HALO_CUTOFF_STRENGTH = 2.5
D_KPC = 143.0
N_LOS = 500
N_ANG = 200
N_PHI = 16
THETA_MAX_DEG = 0.5
# Адаптивный отбор точек (как в J_factor_*): доля лучших по penalty
TARGET_FRACTION = 0.30
CUTOFF_START = 0.60
MIN_POINTS = 10
PENALTY_MAX = 1e5          # отбраковка как в load_log_data (row[6] >= 1e5)

# Сырой файл 4UpsBoTorch_*: колонки строки (7 чисел)
#   0 incl  1 Q  2 gh  3 rh  4 rho0  5 Upsilon  6 penalty
RAW_COL_INCL = 0
RAW_COL_PARAMS = [1, 2, 3, 4, 5]   # Q gh rh rho0 Upsilon (ключ дедупликации)
RAW_COL_PENALTY = 6

# Директория с сырыми 4Ups-файлами (как YADISK_DIR в J_factor_Sersic_Fornax_P21_symm.py)
RAW_DIR_DEFAULT = "/home/gala/Yandex.Disk/galAgama"

# numpy.trapz удалён в NumPy >= 2.0 (как в J_factor_*)
if hasattr(numpy, 'trapezoid'):
    _trapz = numpy.trapezoid
else:
    _trapz = numpy.trapz


def adaptive_penalty_cutoff(penalties, target_fraction=TARGET_FRACTION,
                            cutoff_start=CUTOFF_START, min_points=MIN_POINTS):
    """Копия логики adaptive_penalty_cutoff из J_factor_* (контракт §7)."""
    penalties = numpy.asarray(penalties)
    n_total = len(penalties)
    if n_total <= min_points:
        return float(numpy.max(penalties))
    cutoff = float(numpy.percentile(penalties, target_fraction * 100))
    if cutoff > cutoff_start:
        if numpy.sum(penalties <= cutoff_start) >= min_points:
            cutoff = cutoff_start
    if numpy.sum(penalties <= cutoff) < min_points:
        sorted_pen = numpy.sort(penalties)
        cutoff = float(sorted_pen[min_points - 1])
    return cutoff


def compute_J_factor(Q, gh, rh, rho0, Upsilon,
                     alphah=ALPHAH, betah=BETAH,
                     cutoff=HALO_CUTOFF, cutoff_strength=HALO_CUTOFF_STRENGTH,
                     D_kpc=D_KPC, theta_max_deg=THETA_MAX_DEG,
                     n_los=N_LOS, n_ang=N_ANG, n_phi=N_PHI):
    """
    ВЕРБАТИМ-копия compute_J_factor из J_factor_Sersic_Fornax_P21_symm.py.
    НЕ менять без согласования PI (контракт §7).
    Требует модуль agama (импортируется лениво — только в режиме --source raw).
    """
    import agama
    kpc_to_cm = 3.0857e21
    rho_conv = 1e6 * 1.989e33 / 1.783e-24 / (kpc_to_cm**3)

    density_DM = agama.Density(
        type='spheroid',
        alpha=alphah,
        beta=betah,
        gamma=gh,
        axisratioz=Q,
        densitynorm=rho0 * Upsilon,
        scaleradius=rh,
        outercutoffradius=cutoff,
        cutoffstrength=cutoff_strength,
    )

    theta_max_rad = theta_max_deg * numpy.pi / 180.0
    R_max_kpc = D_kpc * numpy.tan(theta_max_rad)

    l_max = max(10.0 * rh, 3.0 * cutoff)
    l_grid = numpy.linspace(-l_max, l_max, n_los)

    b_edges = numpy.linspace(0.0, R_max_kpc, n_ang + 1)
    b_mids = 0.5 * (b_edges[:-1] + b_edges[1:])
    db = b_edges[1:] - b_edges[:-1]

    phi_grid = numpy.linspace(0.0, 2.0 * numpy.pi, n_phi, endpoint=False)

    J_total = 0.0
    for b_mid, dbi in zip(b_mids, db):
        J_los_phi = 0.0
        for phi in phi_grid:
            x_los = D_kpc - l_grid
            y_los = numpy.full(n_los, b_mid * numpy.cos(phi))
            z_los = numpy.full(n_los, b_mid * numpy.sin(phi))
            points = numpy.column_stack([x_los, y_los, z_los])
            rho_los = density_DM.density(points)
            J_los_phi += _trapz(rho_los**2, l_grid)
        J_los_phi /= n_phi
        dOmega = 2.0 * numpy.pi * b_mid * dbi / D_kpc**2
        J_total += J_los_phi * dOmega

    J_GeV2_cm5 = J_total * (rho_conv**2) * kpc_to_cm
    return J_GeV2_cm5, J_total


def load_raw_logs(paths, incl_filter, verbose=True):
    """
    Читает сырые 4Ups-файлы для ОДНОГО наклонения.
    Фильтры идентичны load_log_data из J_factor_*:
      строка: incl Q gh rh rho0 Upsilon penalty (>=7 чисел);
      отбраковка: nan/inf, rh<=0, rho0<=0, penalty>=1e5,
                  |incl - incl_filter| > 0.01.
    Возвращает массив (N, 7).
    """
    rows = []
    for p in paths:
        try:
            with open(p, 'r') as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.split()
                    if len(parts) < 7:
                        continue
                    try:
                        row = [float(x) for x in parts[:7]]
                    except ValueError:
                        continue
                    if any(numpy.isinf(v) or numpy.isnan(v) for v in row):
                        continue
                    if row[3] <= 0 or row[4] <= 0:
                        continue
                    if row[6] >= PENALTY_MAX:
                        continue
                    if abs(row[0] - incl_filter) > 0.01:
                        continue
                    rows.append(row)
        except FileNotFoundError:
            if verbose:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: файл не найден: {p}")
    if not rows:
        return numpy.empty((0, 7))
    return numpy.array(rows)


def discover_raw_inclinations(paths):
    """Уникальные наклонения (колонка 0) в сырых файлах, округл. до 0.01."""
    incls = set()
    for p in paths:
        try:
            with open(p, 'r') as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith('#'):
                        continue
                    parts = s.split()
                    if len(parts) < 7:
                        continue
                    try:
                        val = float(parts[0])
                    except ValueError:
                        continue
                    if numpy.isinf(val) or numpy.isnan(val):
                        continue
                    incls.add(round(val, 2))
        except FileNotFoundError:
            pass
    return sorted(incls)


# ============================================================
#  ВЗВЕШЕННЫЙ ПЕРЦЕНТИЛЬ (как в J_factor_Sersic_Fornax_P21_symm.py)
# ============================================================

def weighted_percentile(values, weights, percentiles):
    values = numpy.asarray(values, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    percentiles = numpy.asarray(percentiles, dtype=float)

    idx = numpy.argsort(values)
    v_sort = values[idx]
    w_sort = weights[idx]

    w_cumsum = numpy.cumsum(w_sort)
    if w_cumsum[-1] <= 0:
        return numpy.full_like(percentiles, numpy.nan, dtype=float)
    w_cumsum = w_cumsum / w_cumsum[-1]

    return numpy.interp(percentiles / 100.0, w_cumsum, v_sort)


# ============================================================
#  СХЛОПЫВАНИЕ ДУБЛЕЙ
# ============================================================

def collapse_duplicates(data, key_cols=COL_PARAMS_DEDUP, decimals=10):
    """
    Возвращает (data_unique, counts, n_raw, n_unique).
    Дубликатами считаются строки с совпадающим набором параметров key_cols
    (с округлением до `decimals` знаков для устойчивости к шуму записи).
    Сохраняется первое вхождение; counts[i] — кратность i-го уникума в исходных
    данных (для реконструкции "сырой" статистики с учётом частоты сэмплирования).
    Исходный массив не меняется.
    """
    if len(data) == 0:
        return data, numpy.empty(0), 0, 0
    key = numpy.round(data[:, key_cols], decimals)
    _, first_idx, counts = numpy.unique(
        key, axis=0, return_index=True, return_counts=True)
    order = numpy.argsort(first_idx)        # восстановить порядок первого вхождения
    sel = first_idx[order]
    counts = counts[order].astype(float)
    return data[sel], counts, len(data), len(sel)


# ============================================================
#  ОЦЕНКА ПЛОТНОСТИ СЭМПЛИРОВАНИЯ (kNN)
# ============================================================

def knn_density_factor(X, k=None):
    """
    Относительный фактор 1/q̂ для каждой точки, где q̂ — kNN-оценка плотности
    сэмплирования в стандартизованном пространстве X.

    q̂_i ∝ 1 / r_{k,i}^d   ⇒   1/q̂_i ∝ r_{k,i}^d,
    где r_{k,i} — расстояние до k-го соседа, d — размерность.

    Возвращает массив factor_i ∝ r_{k,i}^d (ненормированный),
    устойчиво обработанный (клиппинг по перцентилям 1..99).
    Если точек слишком мало (< 5), возвращает массив единиц.
    """
    n, d = X.shape
    if n < 5:
        return numpy.ones(n)

    if k is None:
        k = min(10, n - 1)
    k = max(1, min(k, n - 1))

    # Стандартизация (z-score); столбцы с нулевой дисперсией отбрасываем
    std = X.std(axis=0)
    keep = std > 0
    Xs = (X[:, keep] - X[:, keep].mean(axis=0)) / std[keep]
    d_eff = Xs.shape[1]
    if d_eff == 0:
        return numpy.ones(n)

    # kNN-расстояния
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Xs)
        dist, _ = tree.query(Xs, k=k + 1)   # +1: первый сосед — сама точка
        r_k = dist[:, -1]
    except Exception:
        # Брутфорс fallback
        r_k = numpy.empty(n)
        for i in range(n):
            dd = numpy.sqrt(((Xs - Xs[i]) ** 2).sum(axis=1))
            dd.sort()
            r_k[i] = dd[k]   # dd[0]==0 (сама точка)

    # Защита от нулевых расстояний (почти-дубли)
    pos = r_k[r_k > 0]
    floor = numpy.percentile(pos, 1) if len(pos) else 1.0
    r_k = numpy.maximum(r_k, floor)

    factor = r_k ** d_eff
    lo, hi = numpy.percentile(factor, [1, 99])
    factor = numpy.clip(factor, lo, hi)
    return factor


# ============================================================
#  МЕТОДЫ ОЦЕНКИ
# ============================================================

def quality_weights(penalties):
    pen_min = penalties.min()
    pen_sigma = max(penalties.std(), 1e-6)
    w = numpy.exp(-(penalties - pen_min) / pen_sigma)
    s = w.sum()
    if s > 0:
        w = w / s
    return w, pen_min, pen_sigma


def estimate_current(data, multiplicity=None):
    """
    M0: вес только по качеству w = exp(-(pen-pen_min)/pen_sigma).
    Если задан multiplicity (кратность точки в сырой выборке), вес умножается
    на неё — это воспроизводит "сырую" статистику с учётом частоты сэмплирования
    (без необходимости физически дублировать строки).
    Возвращает (median, p16, p84).
    """
    pen = data[:, COL_PENALTY]
    logJ = data[:, COL_LOGJ]
    w, _, _ = quality_weights(pen)
    if multiplicity is not None:
        w = w * numpy.asarray(multiplicity, dtype=float)
        s = w.sum()
        if s > 0:
            w = w / s
    med, p16, p84 = weighted_percentile(logJ, w, [50, 16, 84])
    return med, p16, p84


def estimate_profile(data, k_sigma=(1.0, 2.0)):
    """
    M_A: профиль по penalty.
    Центр = log10(J) при min(penalty).
    Для каждого k в k_sigma: интервал [min,max] log10(J) на
    {penalty <= pen_min + k*pen_sigma} и число точек в нём.
    """
    pen = data[:, COL_PENALTY]
    logJ = data[:, COL_LOGJ]
    pen_min = pen.min()
    pen_sigma = max(pen.std(), 1e-6)
    center = float(logJ[numpy.argmin(pen)])

    intervals = {}
    for k in k_sigma:
        thr = pen_min + k * pen_sigma
        sel = pen <= thr
        if sel.sum() >= 1:
            intervals[k] = (float(logJ[sel].min()),
                            float(logJ[sel].max()),
                            int(sel.sum()))
        else:
            intervals[k] = (numpy.nan, numpy.nan, 0)
    return center, intervals, pen_sigma


def estimate_density_corrected(data):
    """M_B: importance sampling с коррекцией на плотность сэмплирования."""
    pen = data[:, COL_PENALTY]
    logJ = data[:, COL_LOGJ]
    w_quality, _, _ = quality_weights(pen)

    X = data[:, COL_PARAMS_DENSITY]
    inv_q = knn_density_factor(X)          # ∝ 1/q̂
    w = w_quality * inv_q
    s = w.sum()
    if s > 0:
        w = w / s
    med, p16, p84 = weighted_percentile(logJ, w, [50, 16, 84])
    return med, p16, p84


# ============================================================
#  ОБРАБОТКА ОДНОГО НАКЛОНЕНИЯ
# ============================================================

def diagnose_dataset(incl, data, counts, n_raw):
    """
    Диагностика одного наклонения.
    `data`   — уникальные точки (9 колонок: Q gh rh rho0 Ups rho0xUps pen J log10J);
    `counts` — кратность каждой точки в сырой выборке;
    `n_raw`  — число сырых (с дублями) точек, вошедших в анализ.
    """
    if data is None or len(data) == 0:
        return None
    n_uniq = len(data)

    # M0: с учётом кратности ("сырая") и без (схлопнутые дубли)
    cur_raw = estimate_current(data, multiplicity=counts)
    cur_ded = estimate_current(data)

    # M_A профиль (на схлопнутых)
    prof_center, prof_int, pen_sigma = estimate_profile(data)

    # M_B плотностная коррекция (на схлопнутых)
    dens = estimate_density_corrected(data)

    return {
        'incl': incl,
        'n_raw': int(n_raw),
        'n_uniq': n_uniq,
        'pen_min': float(data[:, COL_PENALTY].min()),
        'pen_sigma': pen_sigma,
        'M0_raw': cur_raw,
        'M0_dedup': cur_ded,
        'MA_center': prof_center,
        'MA_int': prof_int,
        'MB_dedup': dens,
    }


# ============================================================
#  СБОРКА НАБОРОВ ДАННЫХ ПО ИСТОЧНИКАМ
#  Каждый билдер возвращает {incl: (data9_unique, counts, n_raw)}
# ============================================================

def build_dataset_jfactor(indir, pattern):
    files = sorted(glob.glob(os.path.join(indir, pattern)))
    if not files:
        raise SystemExit(
            f"Не найдено файлов по паттерну {pattern} в {indir}")
    experiments = {parse_jfactor_experiment(path) for path in files}
    if len(experiments) > 1:
        labels = ', '.join(sorted(value or 'legacy' for value in experiments))
        raise SystemExit(
            f"Паттерн одновременно выбрал разные эксперименты ({labels}); "
            "уточните --pattern."
        )
    print(f"[jfactor] Найдено файлов: {len(files)}")
    rows_by_incl = {}
    for f in files:
        table = load_table(f)
        if len(table) == 0:
            continue
        if table.shape[1] >= 10:
            for incl in numpy.unique(numpy.round(table[:, 0], 2)):
                selected = table[numpy.abs(table[:, 0] - incl) <= 0.01, 1:10]
                rows_by_incl.setdefault(float(incl), []).append(selected)
        else:
            incl = parse_incl_from_name(f)
            rows_by_incl.setdefault(incl, []).append(table[:, :9])

    out = {}
    for incl, chunks in rows_by_incl.items():
        data9 = numpy.vstack(chunks)
        du, counts, n_raw, _ = collapse_duplicates(data9)
        out[incl] = (du, counts, n_raw)
    return out


def build_dataset_combined(infile):
    if not os.path.isfile(infile):
        raise SystemExit(f"combined-файл не найден: {infile}")
    arr = load_combined(infile)
    if len(arr) == 0:
        raise SystemExit(f"Пустой combined-файл: {infile}")
    print(f"[combined] Прочитано точек: {len(arr)} из {infile}")
    out = {}
    incls = sorted(set(round(float(v), 2) for v in arr[:, 0]))
    for incl in incls:
        sel = numpy.abs(arr[:, 0] - incl) <= 0.01
        block = arr[sel]
        data9 = block[:, 1:10]
        counts = block[:, 10]
        n_raw = int(counts.sum())
        out[incl] = (data9, counts, n_raw)
    return out


def build_dataset_raw(raw_dir, raw_pattern, incl_only,
                      cutoff_fraction, save_path):
    """
    Читает сырые файлы, схлопывает дубли параметров, (опц.) применяет
    адаптивный cutoff, вычисляет J через AGAMA для каждой уникальной точки
    и пишет combined-файл (incl + 9 колонок + count) инкрементально.
    """
    paths = sorted(glob.glob(os.path.join(raw_dir, raw_pattern)))
    if not paths:
        raise SystemExit(
            f"Не найдено сырых файлов по паттерну {raw_pattern} в {raw_dir}")
    print(f"[raw] Найдено файлов: {len(paths)}")

    incls = discover_raw_inclinations(paths)
    if incl_only is not None:
        incls = [i for i in incls if abs(i - incl_only) <= 0.01]
    print(f"[raw] Наклонения к обработке: {incls}")
    if not incls:
        raise SystemExit("[raw] Нет подходящих наклонений.")

    # Шапка combined-файла (перезапись)
    with open(save_path, 'w') as fout:
        fout.write(f"# J computed from raw 4Ups files  {__import__('datetime').datetime.now()}\n")
        fout.write(f"# theta_max={THETA_MAX_DEG} deg, D={D_KPC} kpc, "
                   f"alphah={ALPHAH}, betah={BETAH}\n")
        fout.write(f"# cutoff_fraction={cutoff_fraction} "
                   f"(1.0 = все уникальные точки)\n")
        fout.write("# incl Q gh rh rho0 Upsilon rho0_x_Ups penalty "
                   "J_GeV2_cm5 log10_J count\n")

    out = {}
    for incl in incls:
        raw = load_raw_logs(paths, incl)
        if len(raw) == 0:
            continue
        # схлопывание дублей по параметрам (Q gh rh rho0 Upsilon) ДО расчёта J
        du_raw, counts, n_raw_total, n_uniq = collapse_duplicates(
            raw, key_cols=RAW_COL_PARAMS)

        # cutoff (по уникальным penalty)
        pen = du_raw[:, RAW_COL_PENALTY]
        if cutoff_fraction >= 1.0:
            mask = numpy.ones(len(du_raw), dtype=bool)
        else:
            cut = adaptive_penalty_cutoff(pen, target_fraction=cutoff_fraction)
            mask = pen <= cut
        kept = du_raw[mask]
        kept_counts = counts[mask]

        print(f"\n[raw] incl={incl}: сырых={n_raw_total}, уник={n_uniq}, "
              f"к J-расчёту={len(kept)} точек")

        data9_rows = []
        kept_mult = []
        for j in range(len(kept)):
            Q, gh, rh, rho0, Ups = (kept[j, 1], kept[j, 2], kept[j, 3],
                                    kept[j, 4], kept[j, 5])
            penalty = kept[j, 6]
            cnt = kept_counts[j]
            try:
                J_GeV, _ = compute_J_factor(Q=Q, gh=gh, rh=rh,
                                            rho0=rho0, Upsilon=Ups)
            except Exception as e:
                print(f"    ОШИБКА J (пропуск): {e}")
                continue
            l10 = float(numpy.log10(J_GeV))
            data9_rows.append([Q, gh, rh, rho0, Ups, rho0 * Ups,
                               penalty, J_GeV, l10])
            kept_mult.append(cnt)
            with open(save_path, 'a') as fout:
                fout.write(
                    f"{incl:.4f} {Q:.10f} {gh:.10f} {rh:.10f} {rho0:.10f} "
                    f"{Ups:.10f} {rho0*Ups:.10f} {penalty:.10f} "
                    f"{J_GeV:.8e} {l10:.8f} {int(cnt)}\n")
            if (j + 1) % 25 == 0:
                print(f"    [{j+1}/{len(kept)}] log10(J)={l10:.3f}")

        if not data9_rows:
            continue
        # n_raw для отчёта = сумма кратностей учтённых точек
        out[incl] = (numpy.array(data9_rows),
                     numpy.array(kept_mult, dtype=float),
                     int(sum(kept_mult)))

    print(f"\n[raw] Сохранён combined-файл: {save_path}")
    if not out:
        raise SystemExit("[raw] J не вычислен ни для одной точки.")
    return out


# ============================================================
#  ПЕЧАТЬ ОТЧЁТА
# ============================================================

def _fmt_med(t):
    med, p16, p84 = t
    return f"{med:7.4f} [{p16:7.4f},{p84:7.4f}]"


def print_report(results):
    print("=" * 100)
    print("ДИАГНОСТИКА ВЗВЕШИВАНИЯ J-ФАКТОРА  (log10 J, theta=0.5)")
    print("READ-ONLY; дубли схлопнуты перед статистикой; исходные файлы не изменены")
    print("=" * 100)

    # Таблица дублей
    print("\n[1] Схлопывание дублей по наклонению")
    print(f"{'incl':>6s}  {'N_raw':>6s}  {'N_uniq':>6s}  {'dup,%':>6s}  "
          f"{'pen_min':>8s}  {'pen_sigma':>9s}")
    print("-" * 60)
    for r in results:
        dup_pct = 100.0 * (r['n_raw'] - r['n_uniq']) / r['n_raw'] if r['n_raw'] else 0.0
        print(f"{r['incl']:6.2f}  {r['n_raw']:6d}  {r['n_uniq']:6d}  "
              f"{dup_pct:6.1f}  {r['pen_min']:8.4f}  {r['pen_sigma']:9.5f}")

    # Эффект дублей на M0
    print("\n[2] Эффект дублей на текущую схему (M0): сырые vs схлопнутые")
    print(f"{'incl':>6s}  {'M0_raw (median[16,84])':>30s}  "
          f"{'M0_dedup (median[16,84])':>30s}  {'Δmed':>8s}")
    print("-" * 84)
    for r in results:
        dmed = r['M0_dedup'][0] - r['M0_raw'][0]
        print(f"{r['incl']:6.2f}  {_fmt_med(r['M0_raw']):>30s}  "
              f"{_fmt_med(r['M0_dedup']):>30s}  {dmed:+8.4f}")

    # Сравнение методов (на схлопнутых)
    print("\n[3] Сравнение методов (на схлопнутых дублях)")
    print(f"{'incl':>6s}  {'M0 current':>28s}  {'MB density-corr':>28s}  "
          f"{'MA center':>9s}")
    print("-" * 80)
    for r in results:
        print(f"{r['incl']:6.2f}  {_fmt_med(r['M0_dedup']):>28s}  "
              f"{_fmt_med(r['MB_dedup']):>28s}  {r['MA_center']:9.4f}")

    # Профиль по уровням penalty
    print("\n[4] Профиль по penalty (M_A): интервал log10(J) на уровнях dPen=k*pen_sigma")
    print(f"{'incl':>6s}  {'center':>8s}  "
          f"{'k=1: [min,max] (N)':>30s}  {'k=2: [min,max] (N)':>30s}")
    print("-" * 82)
    for r in results:
        i1 = r['MA_int'].get(1.0, (numpy.nan, numpy.nan, 0))
        i2 = r['MA_int'].get(2.0, (numpy.nan, numpy.nan, 0))
        s1 = f"[{i1[0]:7.4f},{i1[1]:7.4f}] ({i1[2]})"
        s2 = f"[{i2[0]:7.4f},{i2[1]:7.4f}] ({i2[2]})"
        print(f"{r['incl']:6.2f}  {r['MA_center']:8.4f}  {s1:>30s}  {s2:>30s}")

    print("\nПримечания:")
    print("  - penalty НЕ калибруется как chi^2 (ответ PI Q1), поэтому уровни")
    print("    dPen=k*pen_sigma в M_A — относительные, НЕ доверительные.")
    print("  - M_B делит вес на kNN-оценку плотности сэмплирования в (Q,gh,rh,rho0);")
    print("    устойчивость: клиппинг 1/q̂ по перцентилям [1,99].")
    print("  - Метод итоговой оценки J — решение PI (вопрос #16, контракт §7).")
    print("=" * 100)


# ============================================================
#  ОПЦИОНАЛЬНЫЙ ГРАФИК
# ============================================================

def make_plot(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r['incl'])
    incl = numpy.array([r['incl'] for r in results])

    def col(method, idx):
        return numpy.array([r[method][idx] for r in results])

    fig, ax = plt.subplots(figsize=(9, 6))

    # M0 dedup
    m0 = col('M0_dedup', 0)
    ax.errorbar(incl - 0.15, m0,
                yerr=[m0 - col('M0_dedup', 1), col('M0_dedup', 2) - m0],
                fmt='o-', capsize=3, color='steelblue',
                label='M0 current (dedup)')

    # M_B density-corrected
    mb = col('MB_dedup', 0)
    ax.errorbar(incl + 0.15, mb,
                yerr=[mb - col('MB_dedup', 1), col('MB_dedup', 2) - mb],
                fmt='s-', capsize=3, color='crimson',
                label='M_B density-corrected')

    # M_A center (best-fit)
    ax.plot(incl, [r['MA_center'] for r in results],
            'D', color='black', markersize=5,
            label='M_A center (min penalty)')

    ax.set_xlabel('inclination [deg]')
    ax.set_ylabel(r'$\log_{10} J$  (theta=0.5)')
    ax.set_title('J-factor weighting diagnostic (duplicates collapsed)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nГрафик сохранён: {out_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_indir = os.path.join(RAW_DIR_DEFAULT, 'J_factors')
    default_savej = os.path.join(script_dir, 'Jcomputed_from_raw_theta0.5.txt')

    parser = argparse.ArgumentParser(
        description="Диагностика взвешивания J-фактора.")
    parser.add_argument(
        '--source', choices=['jfactor', 'raw', 'combined'], default='jfactor',
        help="Источник данных (см. докстринг). По умолчанию jfactor.")
    # jfactor
    parser.add_argument(
        '--indir', default=default_indir,
        help="[jfactor] Директория с J_factor_Sersic*_theta0.5.txt "
             "(по умолчанию Yandex.Disk/galAgama/J_factors).")
    parser.add_argument(
        '--pattern', default='J_factor_Sersic*_theta0.5.txt',
        help=("[jfactor] Glob-паттерн входных файлов; должен выбирать не более "
              "одного experiment ID."))
    # raw
    parser.add_argument(
        '--raw-dir', default=RAW_DIR_DEFAULT,
        help=f"[raw] Директория с сырыми 4Ups-файлами "
             f"(по умолчанию {RAW_DIR_DEFAULT}).")
    parser.add_argument(
        '--raw-pattern', default='4UpsBoTorch_PCA_Sersic*.txt',
        help="[raw] Glob-паттерн сырых файлов.")
    parser.add_argument(
        '--incl', type=float, default=None,
        help="[raw] Обработать только это наклонение (градусы).")
    parser.add_argument(
        '--cutoff-fraction', type=float, default=1.0,
        help="[raw] Доля лучших по penalty уникальных точек для расчёта J "
             "(1.0 = все; 0.30 = как в production J_factor_*).")
    parser.add_argument(
        '--save-j', default=default_savej,
        help="[raw] Куда сохранить посчитанный J (combined-файл).")
    # combined
    parser.add_argument(
        '--combined-file', default=None,
        help="[combined] Путь к combined-файлу (по умолчанию = --save-j).")
    parser.add_argument(
        '--plot', default=None,
        help="Путь для сохранения сравнительного графика (PDF/PNG). "
             "Если не задан — файл не создаётся.")
    args = parser.parse_args()

    if args.source == 'jfactor':
        datasets = build_dataset_jfactor(args.indir, args.pattern)
    elif args.source == 'raw':
        datasets = build_dataset_raw(
            args.raw_dir, args.raw_pattern, args.incl,
            args.cutoff_fraction, args.save_j)
    else:  # combined
        infile = args.combined_file or args.save_j
        datasets = build_dataset_combined(infile)

    results = []
    for incl in sorted(datasets, key=lambda x: (x if x is not None else 0)):
        du, counts, n_raw = datasets[incl]
        r = diagnose_dataset(incl, du, counts, n_raw)
        if r is not None:
            results.append(r)

    if not results:
        raise SystemExit("Нет данных для анализа.")

    print_report(results)

    if args.plot:
        make_plot(results, args.plot)


if __name__ == '__main__':
    main()
