#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J-factor computation for Fornax dSph galaxy
Based on Schwarzschild orbit modelling results (AGAMA/forstand)

Reads log files from:
  /home/gala/Yandex.Disk/galAgama/4UpsBoTorch_Sersic.txt
  /home/gala/Yandex.Disk/galAgama/4UpsBoTorch_PCA_Sersic_*.txt

Запуск в Spyder Console:
  %run compute_J_factor.py
  или выделить нужный блок и нажать F9
"""

import numpy
import glob
import os
import datetime
import agama

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# numpy.trapz удалён в NumPy >= 2.0, заменён на numpy.trapezoid
if hasattr(numpy, 'trapezoid'):
    _trapz = numpy.trapezoid
else:
    _trapz = numpy.trapz

# ============================================================
#  НАСТРОЙКИ — меняйте здесь
# ============================================================

# Директория с лог-файлами на Яндекс.Диске
# Входные И выходные файлы — в одной директории
YADISK_DIR = "/home/gala/Yandex.Disk/galAgama"

# Паттерны файлов (относительно YADISK_DIR)
LOG_PATTERNS_REL = [
    "4UpsBoTorch_Sersic.txt",
    "4UpsBoTorch_PCA_Sersic_*.txt",
]

# Параметры расчёта — редактируйте перед запуском
incl_target     = 90.0     # угол наклонения для фильтра (градусы)
penalty_cutoff  = None     # None = адаптивный (target_fraction лучших)
target_fraction = 0.30     # доля лучших точек при адаптивном cutoff
cutoff_start    = 0.60     # жёсткий потолок cutoff
D_kpc           = 143.0    # расстояние до галактики (kpc)
n_samples       = None     # None = все хорошие точки
alphah          = 2.0      # параметр профиля гало
betah           = 3.0      # параметр профиля гало
n_los           = 500      # точек по лучу зрения
n_ang           = 200      # точек по прицельному параметру
n_phi           = 16       # точек по азимутальному углу

# Угловые радиусы для расчёта J (градусы)
theta_list      = [0.1, 0.2, 0.5, 1.0]

# Строить corner-plot для theta=0.5°?
do_corner_plot  = True


# ============================================================
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: пути и имена файлов
# ============================================================

def make_output_filename(prefix, theta_max_deg, incl, ext='txt'):
    """
    Формирует имя выходного файла вида:
        {prefix}_Sersic_incl{incl:.2f}_theta{theta:.1f}.{ext}

    Примеры:
        J_factor_Sersic_incl90.00_theta0.5.txt
        corner_plot_Sersic_incl71.85_theta0.5.pdf
    """
    return (
        f"{prefix}_Sersic"
        f"_incl{incl:.2f}"
        f"_theta{theta_max_deg:.1f}"
        f".{ext}"
    )


def make_output_fullpath(prefix, theta_max_deg, incl, ext='txt',
                         yadisk_dir=YADISK_DIR):
    """
    Формирует полный путь к выходному файлу в директории yadisk_dir.

    Все выходные файлы (txt и pdf) сохраняются в ту же директорию,
    что и входные лог-файлы.

    Примеры результата:
        /home/gala/Yandex.Disk/galAgama/J_factor_Sersic_incl90.00_theta0.5.txt
        /home/gala/Yandex.Disk/galAgama/corner_plot_Sersic_incl90.00_theta0.5.pdf
    """
    fname = make_output_filename(prefix, theta_max_deg, incl, ext)
    return os.path.join(yadisk_dir, fname)


# ============================================================
#  СБОР ФАЙЛОВ
# ============================================================

def collect_log_files(yadisk_dir=YADISK_DIR,
                      patterns_rel=LOG_PATTERNS_REL,
                      verbose=True):
    """
    Собирает список файлов по glob-паттернам из директории yadisk_dir.
    """
    if not os.path.isdir(yadisk_dir):
        raise FileNotFoundError(
            f"Директория не найдена: {yadisk_dir}\n"
            f"Проверьте, что Яндекс.Диск смонтирован."
        )

    found = []
    for pattern_rel in patterns_rel:
        full_pattern = os.path.join(yadisk_dir, pattern_rel)
        matched      = sorted(glob.glob(full_pattern))
        for f in matched:
            if f not in found and os.path.isfile(f):
                found.append(f)

    if verbose:
        print(f"\nДиректория поиска: {yadisk_dir}")
        print(f"Найдено лог-файлов: {len(found)}")
        for f in found:
            size_kb = os.path.getsize(f) / 1024
            print(f"  {os.path.basename(f)}  ({size_kb:.1f} KB)")

    if not found:
        raise FileNotFoundError(
            f"Не найдено ни одного файла в {yadisk_dir}\n"
            f"по паттернам: {patterns_rel}"
        )

    return found


# ============================================================
#  ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_log_data(log_files, incl_filter=90.0, verbose=True):
    """
    Читает строки данных из лог-файлов.

    Формат строки (7 чисел):
        incl  Q  gh  rh  rho0  Upsilon  penalty
    """
    raw         = []
    file_counts = {}

    for filepath in log_files:
        count = 0
        try:
            with open(filepath, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        row = [float(p) for p in parts[:7]]
                        if any(numpy.isinf(v) or numpy.isnan(v)
                               for v in row):
                            continue
                        if row[3] <= 0 or row[4] <= 0:
                            continue
                        if row[6] >= 1e5:
                            continue
                        if abs(row[0] - incl_filter) > 0.01:
                            continue
                        raw.append(row)
                        count += 1
                    except ValueError:
                        continue
        except FileNotFoundError:
            if verbose:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: файл не найден: {filepath}")
        file_counts[os.path.basename(filepath)] = count

    if verbose:
        print(f"\nЗагружено строк (incl={incl_filter}):")
        for fname, cnt in file_counts.items():
            print(f"  {fname}: {cnt}")

    if not raw:
        return None, file_counts

    data = numpy.array(raw)

    if verbose:
        print(f"\nИтого точек: {len(data)}")
        print(f"Диапазон penalty: "
              f"[{data[:, 6].min():.4f}, {data[:, 6].max():.4f}]")

    return data, file_counts


# ============================================================
#  АДАПТИВНЫЙ CUTOFF
# ============================================================

def adaptive_penalty_cutoff(penalties, target_fraction=0.30,
                             cutoff_start=0.60, min_points=10):
    """
    Выбирает порог penalty так, чтобы оставить target_fraction
    лучших точек, но не менее min_points.
    """
    penalties = numpy.asarray(penalties)
    n_total   = len(penalties)

    if n_total <= min_points:
        return float(numpy.max(penalties))

    cutoff = float(numpy.percentile(penalties, target_fraction * 100))

    if cutoff > cutoff_start:
        if numpy.sum(penalties <= cutoff_start) >= min_points:
            cutoff = cutoff_start

    if numpy.sum(penalties <= cutoff) < min_points:
        sorted_pen = numpy.sort(penalties)
        cutoff     = float(sorted_pen[min_points - 1])

    return cutoff


# ============================================================
#  ВЗВЕШЕННЫЙ ПЕРЦЕНТИЛЬ
# ============================================================

def weighted_percentile(values, weights, percentiles):
    """
    Взвешенный перцентиль через кумулятивное распределение весов.
    """
    values      = numpy.asarray(values,  dtype=float)
    weights     = numpy.asarray(weights, dtype=float)
    percentiles = numpy.asarray(percentiles, dtype=float)

    idx      = numpy.argsort(values)
    v_sort   = values[idx]
    w_sort   = weights[idx]

    w_cumsum  = numpy.cumsum(w_sort)
    w_cumsum /= w_cumsum[-1]

    return numpy.interp(percentiles / 100.0, w_cumsum, v_sort)


# ============================================================
#  ВЫЧИСЛЕНИЕ J-ФАКТОРА
# ============================================================

def compute_J_factor(
    Q, gh, rh, rho0,
    alphah=2.0, betah=3.0,
    cutoff=55.0, cutoff_strength=2.5,
    D_kpc=143.0,
    theta_max_deg=0.5,
    n_los=500,
    n_ang=200,
    n_phi=16,
):
    """
    Вычисляет J-фактор для профиля ТМ типа spheroid (AGAMA).

    Returns
    -------
    J_GeV2_cm5  : float — J-фактор в GeV²/cm⁵
    J_Msun_kpc5 : float — J-фактор в Msun²/kpc⁵·sr (до конвертации)
    """
    kpc_to_cm = 3.0857e21
    rho_conv  = 1.989e33 / 1.602e-10 / (kpc_to_cm**3)

    density_DM = agama.Density(
        type              = 'spheroid',
        alpha             = alphah,
        beta              = betah,
        gamma             = gh,
        axisratioz        = Q,
        densitynorm       = rho0,
        scaleradius       = rh,
        outercutoffradius = cutoff,
        cutoffstrength    = cutoff_strength,
    )

    theta_max_rad = theta_max_deg * numpy.pi / 180.0
    R_max_kpc     = D_kpc * numpy.tan(theta_max_rad)

    l_max  = max(10.0 * rh, 3.0 * cutoff)
    l_grid = numpy.linspace(-l_max, l_max, n_los)

    b_edges = numpy.linspace(0.0, R_max_kpc, n_ang + 1)
    b_mids  = 0.5 * (b_edges[:-1] + b_edges[1:])
    db      = b_edges[1:] - b_edges[:-1]

    phi_grid = numpy.linspace(0.0, 2.0 * numpy.pi, n_phi, endpoint=False)

    J_total = 0.0

    for b_mid, dbi in zip(b_mids, db):
        J_los_phi = 0.0
        for phi in phi_grid:
            x_los = D_kpc - l_grid
            y_los = numpy.full(n_los, b_mid * numpy.cos(phi))
            z_los = numpy.full(n_los, b_mid * numpy.sin(phi))

            points    = numpy.column_stack([x_los, y_los, z_los])
            rho_los   = density_DM.density(points)
            J_los_phi += _trapz(rho_los**2, l_grid)

        J_los_phi /= n_phi

        dOmega  = 2.0 * numpy.pi * b_mid * dbi / D_kpc**2
        J_total += J_los_phi * dOmega

    J_GeV2_cm5 = J_total * (rho_conv**2) * kpc_to_cm

    return J_GeV2_cm5, J_total


# ============================================================
#  РАСЧЁТ J ДЛЯ НАБОРА ПАРАМЕТРОВ ИЗ ЛОГ-ФАЙЛОВ
# ============================================================

def compute_J_from_logs(
    theta_max_deg = 0.5,
    incl_filter   = None,
    pen_cutoff    = None,
    output_file   = None,
    yadisk_dir    = YADISK_DIR,
    patterns_rel  = LOG_PATTERNS_REL,
):
    """
    Читает лог-файлы из yadisk_dir, вычисляет J-фактор для каждой
    хорошей точки и строит взвешенную статистику.

    Выходной текстовый файл сохраняется в yadisk_dir.

    Returns
    -------
    results   : dict
    J_arr     : numpy.ndarray — J-значения (GeV²/cm⁵)
    data_good : numpy.ndarray — использованные строки данных
    """
    _incl   = incl_filter if incl_filter is not None else incl_target
    _cutoff = pen_cutoff  if pen_cutoff  is not None else penalty_cutoff

    # Выходной текстовый файл — в той же директории, что входные
    if output_file is not None:
        _outfile = output_file
    else:
        _outfile = make_output_fullpath(
            prefix        = 'J_factor',
            theta_max_deg = theta_max_deg,
            incl          = _incl,
            ext           = 'txt',
            yadisk_dir    = yadisk_dir,
        )

    print(f"\nВыходной файл: {_outfile}")

    # --- Сбор и загрузка файлов ---
    log_files = collect_log_files(
        yadisk_dir=yadisk_dir, patterns_rel=patterns_rel, verbose=True
    )
    data, file_counts = load_log_data(
        log_files, incl_filter=_incl, verbose=True
    )
    if data is None or len(data) == 0:
        raise ValueError(
            f"Нет данных для incl={_incl} в найденных файлах."
        )

    # --- Выбор cutoff ---
    if _cutoff is None:
        _cutoff = adaptive_penalty_cutoff(
            data[:, 6],
            target_fraction = target_fraction,
            cutoff_start    = cutoff_start,
        )
        print(f"\nАдаптивный penalty cutoff: {_cutoff:.4f} "
              f"(лучшие {target_fraction*100:.0f}%)")
    else:
        print(f"\nЗаданный penalty cutoff: {_cutoff:.4f}")

    mask_good = data[:, 6] <= _cutoff
    data_good = data[mask_good].copy()
    print(f"Хороших точек (penalty ≤ {_cutoff:.4f}): {len(data_good)}")

    if len(data_good) == 0:
        raise ValueError(
            f"Нет точек с penalty ≤ {_cutoff:.4f}."
        )

    # --- Взвешенная выборка ---
    if n_samples is not None and n_samples < len(data_good):
        w_samp  = numpy.exp(-data_good[:, 6] / 0.1)
        w_samp /= w_samp.sum()
        idx       = numpy.random.choice(
            len(data_good), size=n_samples, replace=False, p=w_samp
        )
        data_good = data_good[idx]
        print(f"Взвешенная выборка: {n_samples} точек")

    # --- Заголовок файла результатов ---
    with open(_outfile, 'w') as fout:
        fout.write(f"# J-factor computation  {datetime.datetime.now()}\n")
        fout.write(f"# theta_max={theta_max_deg} deg, D={D_kpc} kpc\n")
        fout.write(f"# incl_target={_incl}, "
                   f"penalty_cutoff={_cutoff:.6f}\n")
        fout.write(f"# alphah={alphah}, betah={betah}\n")
        fout.write(f"# Source directory: {yadisk_dir}\n")
        fout.write("# Source files:\n")
        for fname, cnt in file_counts.items():
            fout.write(f"#   {fname}: {cnt} rows\n")
        fout.write(
            "# Q gh rh rho0 Upsilon rho0_x_Ups penalty "
            "J_GeV2_cm5 log10_J\n"
        )

    # --- Основной цикл ---
    J_values = []
    J_Msun   = []

    for i, row in enumerate(data_good):
        _, Q, gh, rh, rho0, Upsilon, penalty = row

        print(f"  [{i+1:4d}/{len(data_good)}] "
              f"Q={Q:.3f} gh={gh:.3f} rh={rh:.3f} "
              f"rho0={rho0:.1f} Ups={Upsilon:.3f} "
              f"pen={penalty:.4f}",
              end="  →  ")

        try:
            J_GeV, J_M = compute_J_factor(
                Q=Q, gh=gh, rh=rh, rho0=rho0,
                alphah=alphah, betah=betah,
                D_kpc=D_kpc,
                theta_max_deg=theta_max_deg,
                n_los=n_los,
                n_ang=n_ang,
                n_phi=n_phi,
            )
            J_values.append(J_GeV)
            J_Msun.append(J_M)
            print(f"log10(J) = {numpy.log10(J_GeV):.3f}")

            with open(_outfile, 'a') as fout:
                fout.write(
                    f"{Q:.10f} {gh:.10f} {rh:.10f} {rho0:.10f} "
                    f"{Upsilon:.10f} {rho0*Upsilon:.10f} "
                    f"{penalty:.10f} "
                    f"{J_GeV:.8e} {numpy.log10(J_GeV):.8f}\n"
                )

        except Exception as e:
            print(f"ОШИБКА: {e}")
            continue

    if not J_values:
        raise RuntimeError("Не удалось вычислить J ни для одной точки.")

    J_arr = numpy.array(J_values)
    logJ  = numpy.log10(J_arr)

    # --- Взвешенная статистика ---
    pen_used  = data_good[:len(J_arr), 6]
    pen_min   = pen_used.min()
    pen_sigma = max(pen_used.std(), 1e-6)
    w_J       = numpy.exp(-(pen_used - pen_min) / pen_sigma)
    w_J      /= w_J.sum()

    logJ_wmean = float(numpy.sum(w_J * logJ))
    logJ_wstd  = float(
        numpy.sqrt(numpy.sum(w_J * (logJ - logJ_wmean)**2))
    )
    wp = weighted_percentile(
        logJ, w_J, [2.5, 16.0, 50.0, 84.0, 97.5]
    )
    logJ_w2p5, logJ_w16, logJ_w50, logJ_w84, logJ_w97p5 = wp

    results = {
        'J_median':       float(numpy.median(J_arr)),
        'J_wmean':        float(10**logJ_wmean),
        'J_mean':         float(numpy.mean(J_arr)),
        'J_std':          float(numpy.std(J_arr)),
        'logJ_median':    logJ_w50,
        'logJ_mean':      logJ_wmean,
        'logJ_std':       logJ_wstd,
        'logJ_16':        logJ_w16,
        'logJ_84':        logJ_w84,
        'logJ_2p5':       logJ_w2p5,
        'logJ_97p5':      logJ_w97p5,
        'n_points':       len(J_arr),
        'theta_max':      theta_max_deg,
        'D_kpc':          D_kpc,
        'incl':           _incl,
        'penalty_cutoff': _cutoff,
        'output_file':    _outfile,
        'pen_min':        float(pen_min),
        'pen_sigma':      float(pen_sigma),
    }

    # --- Вывод в консоль ---
    print("\n" + "=" * 60)
    print(f"J-ФАКТОР  (theta < {theta_max_deg}°, D={D_kpc} kpc,  "
          f"incl={_incl}°)")
    print("=" * 60)
    print(f"  log10(J) = {results['logJ_median']:.3f} "
          f"+ {results['logJ_84'] - results['logJ_median']:.3f} "
          f"- {results['logJ_median'] - results['logJ_16']:.3f}  (1σ)")
    print(f"  log10(J) = {results['logJ_median']:.3f} "
          f"+ {results['logJ_97p5'] - results['logJ_median']:.3f} "
          f"- {results['logJ_median'] - results['logJ_2p5']:.3f}  (2σ)")
    print(f"  Медиана J (взвеш.) = {10**logJ_w50:.3e} GeV²/cm⁵")
    print(f"  По {results['n_points']} точкам")
    print("=" * 60)

    # --- Запись итогов в файл ---
    with open(_outfile, 'a') as fout:
        fout.write("\n# ============ RESULTS ============\n")
        fout.write(
            "# Статистика ВЗВЕШЕННАЯ: "
            "w = exp(-(penalty - pen_min) / pen_sigma)\n"
        )
        fout.write(f"# pen_min             = {results['pen_min']:.6f}\n")
        fout.write(f"# pen_sigma           = {results['pen_sigma']:.6f}\n")
        fout.write(f"# n_points            = {results['n_points']}\n")
        fout.write(f"# penalty_cutoff      = {results['penalty_cutoff']:.6f}\n")
        fout.write(f"# log10(J) median     = {results['logJ_median']:.4f}\n")
        fout.write(f"# log10(J) mean       = {results['logJ_mean']:.4f}\n")
        fout.write(f"# log10(J) std        = {results['logJ_std']:.4f}\n")
        fout.write(
            f"# log10(J) 1sigma     = "
            f"[{results['logJ_16']:.4f}, {results['logJ_84']:.4f}]\n"
        )
        fout.write(
            f"# log10(J) 2sigma     = "
            f"[{results['logJ_2p5']:.4f}, {results['logJ_97p5']:.4f}]\n"
        )
        fout.write(
            f"# J median (weighted) = {10**logJ_w50:.4e} GeV2/cm5\n"
        )
        fout.write("# ==================================\n")

    print(f"Результаты записаны в: {_outfile}")

    return results, J_arr, data_good


# ============================================================
#  CORNER-PLOT
# ============================================================

def make_corner_plot(data_good, J_arr,
                     output_file=None,
                     theta_max_deg=0.5,
                     incl=None,
                     title=None,
                     yadisk_dir=YADISK_DIR):
    """
    Corner-plot: Q, gh, rh, rho0*Upsilon, Upsilon, log10(J).

    Диагональ          : взвешенная гистограмма + медиана (красный пунктир)
    Нижний треугольник : взвешенные KDE-контуры плотности
    Верхний треугольник: коэффициент корреляции Пирсона r
                         (цветной фон: зелёный=+, красный=−)

    Выходной PDF сохраняется в yadisk_dir.
    """
    _incl = incl if incl is not None else incl_target

    # Выходной PDF — в той же директории, что входные файлы
    if output_file is not None:
        _outfile = output_file
    else:
        _outfile = make_output_fullpath(
            prefix        = 'corner_plot',
            theta_max_deg = theta_max_deg,
            incl          = _incl,
            ext           = 'pdf',
            yadisk_dir    = yadisk_dir,
        )

    print(f"\nCorner-plot → {_outfile}")

    n_J = len(J_arr)

    # Колонки data_good: incl(0) Q(1) gh(2) rh(3) rho0(4) Upsilon(5) penalty(6)
    Q_arr        = data_good[:n_J, 1]
    gh_arr       = data_good[:n_J, 2]
    rh_arr       = data_good[:n_J, 3]
    rho0_arr     = data_good[:n_J, 4]
    Ups_arr      = data_good[:n_J, 5]
    penalties    = data_good[:n_J, 6]
    rho0_Ups_arr = rho0_arr * Ups_arr   # физическая нормировка плотности

    labels = [
        r'$q$',
        r'$\gamma_h$',
        r'$r_h$ (kpc)',
        r'$\rho_0 \cdot \Upsilon_*$',
        r'$\Upsilon_*$',
        r'$\log_{10}(J)$',
    ]

    X = numpy.column_stack([
        Q_arr,
        gh_arr,
        rh_arr,
        rho0_Ups_arr,       # rho0 * Upsilon вместо rho0
        Ups_arr,
        numpy.log10(J_arr),
    ])
    n_params = X.shape[1]

    # Веса по penalty
    pen_min   = penalties.min()
    pen_sigma = max(penalties.std(), 1e-6)
    weights   = numpy.exp(-(penalties - pen_min) / pen_sigma)
    weights  /= weights.sum()

    # --- Структура фигуры: только основная сетка, без colorbar ---
    fig, axes = plt.subplots(
        n_params, n_params,
        figsize=(13, 13),
    )
    fig.subplots_adjust(
        left=0.08, right=0.98,
        bottom=0.06, top=0.93,
        hspace=0.08, wspace=0.08,
    )

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # --------------------------------------------------
                # Диагональ: взвешенная гистограмма + медиана
                # --------------------------------------------------
                ax.hist(
                    X[:, i], bins=30,
                    weights=weights,
                    color='steelblue', alpha=0.75, density=True,
                    edgecolor='white', linewidth=0.3,
                )
                w_med = weighted_percentile(X[:, i], weights, [50])[0]
                ax.axvline(
                    w_med, color='crimson',
                    linewidth=1.2, linestyle='--', alpha=0.8,
                )
                ax.tick_params(labelsize=6)

            elif i > j:
                # --------------------------------------------------
                # Нижний треугольник: взвешенные KDE-контуры
                # Светлый = 20-й перцентиль плотности (внешний)
                # Средний = 50-й перцентиль плотности
                # Тёмный  = 80-й перцентиль плотности (ядро)
                # --------------------------------------------------
                try:
                    from scipy.stats import gaussian_kde
                    w_kde = numpy.exp(-(penalties - pen_min) / pen_sigma)
                    kde   = gaussian_kde(
                        numpy.vstack([X[:, j], X[:, i]]),
                        weights=w_kde,
                    )
                    xg = numpy.linspace(X[:, j].min(), X[:, j].max(), 60)
                    yg = numpy.linspace(X[:, i].min(), X[:, i].max(), 60)
                    XG, YG = numpy.meshgrid(xg, yg)
                    ZG = kde(
                        numpy.vstack([XG.ravel(), YG.ravel()])
                    ).reshape(XG.shape)

                    lev20, lev50, lev80 = numpy.percentile(
                        ZG, [20, 50, 80]
                    )
                    ax.contourf(
                        XG, YG, ZG,
                        levels=[lev20, lev50, lev80, ZG.max()],
                        colors=['#cce5ff', '#6baed6', '#2171b5'],
                        alpha=0.85,
                    )
                    ax.contour(
                        XG, YG, ZG,
                        levels=[lev20, lev50, lev80],
                        colors=['#2171b5'],
                        linewidths=0.8, alpha=0.9,
                    )

                except Exception:
                    # Fallback: простой scatter
                    ax.scatter(
                        X[:, j], X[:, i],
                        color='steelblue',
                        s=6, alpha=0.4,
                    )
                ax.tick_params(labelsize=6)

            else:
                # --------------------------------------------------
                # Верхний треугольник: r Пирсона
                # Фон: зелёный = положит. корреляция,
                #       красный = отрицат. корреляция
                # Насыщенность фона пропорциональна |r|
                # --------------------------------------------------
                corr = numpy.corrcoef(X[:, j], X[:, i])[0, 1]

                bg_alpha = min(abs(corr) * 0.6, 0.55)
                bg_color = '#ccffcc' if corr >= 0 else '#ffcccc'
                ax.set_facecolor(
                    (*matplotlib.colors.to_rgb(bg_color), bg_alpha)
                )

                color = 'crimson' if abs(corr) > 0.5 else 'black'
                fw    = 'bold'    if abs(corr) > 0.7 else 'normal'
                fs    = 11        if abs(corr) > 0.7 else 9
                ax.text(
                    0.5, 0.5, f'r = {corr:.2f}',
                    transform=ax.transAxes,
                    ha='center', va='center',
                    fontsize=fs, color=color, fontweight=fw,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            # Подписи осей — только по внешним краям
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=8)
            else:
                ax.set_ylabel('')
            if i == n_params - 1:
                ax.set_xlabel(labels[j], fontsize=8)
            else:
                ax.set_xlabel('')

            if i < n_params - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    # --- Заголовок ---
    _title = title or (
        f'Fornax dSph  |  incl={_incl:.2f}°  '
        f'|  theta<{theta_max_deg:.1f}°  '
        f'|  {n_J} models'
    )
    fig.suptitle(_title, fontsize=11)

    plt.savefig(_outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Corner-plot сохранён: {_outfile}")

    return _outfile


# ============================================================
#  СВОДНАЯ ТАБЛИЦА
# ============================================================

def print_summary_table(all_results):
    """
    Печатает сводную таблицу J-фактора для нескольких theta_max.
    """
    print("\n" + "=" * 80)
    print(f"{'СВОДНАЯ ТАБЛИЦА J-ФАКТОРА (взвешенная статистика)':^80}")
    print("=" * 80)
    print(f"{'theta':>8s}  {'log10(J)':>10s}  "
          f"{'+1σ':>7s}  {'-1σ':>7s}  "
          f"{'+2σ':>7s}  {'-2σ':>7s}  "
          f"{'N':>5s}  {'файл'}")
    print("-" * 80)
    for r in all_results:
        med = r['logJ_median']
        p1  = r['logJ_84']   - med
        m1  = med - r['logJ_16']
        p2  = r['logJ_97p5'] - med
        m2  = med - r['logJ_2p5']
        print(f"  {r['theta_max']:6.2f}°  {med:10.3f}  "
              f"{p1:+7.3f}  {m1:+7.3f}  "
              f"{p2:+7.3f}  {m2:+7.3f}  "
              f"{r['n_points']:5d}  "
              f"{os.path.basename(r['output_file'])}")
    print("=" * 80)


# ============================================================
#  ЗАПУСК (выполняется при %run или F9 в Spyder)
# ============================================================

print("=" * 60)
print("J-FACTOR COMPUTATION  Fornax dSph")
print(f"Start: {datetime.datetime.now()}")
print(f"incl = {incl_target}°,  D = {D_kpc} kpc")
print(f"Входные файлы  ← {YADISK_DIR}")
print(f"Выходные файлы → {YADISK_DIR}")
print("=" * 60)

all_results = []

# --- Расчёт для каждого theta ---
for _theta in theta_list:
    print(f"\n{'='*50}")
    print(f"theta_max = {_theta}°")
    print('='*50)

    try:
        _results, _J_arr, _data_used = compute_J_from_logs(
            theta_max_deg = _theta,
        )
        all_results.append(_results)

    except Exception as _e:
        print(f"  ОШИБКА для theta={_theta}: {_e}")
        continue

# --- Сводная таблица ---
if all_results:
    print_summary_table(all_results)

# --- Corner-plot для theta=0.5° ---
if do_corner_plot:
    _theta_corner = 0.5
    try:
        _res_c, _J_c, _data_c = compute_J_from_logs(
            theta_max_deg = _theta_corner,
        )
        make_corner_plot(
            data_good     = _data_c,
            J_arr         = _J_c,
            theta_max_deg = _theta_corner,
            incl          = incl_target,
            yadisk_dir    = YADISK_DIR,
            title         = (
                f'Fornax dSph  |  incl={incl_target:.2f}°  '
                f'|  theta<{_theta_corner:.1f}°  '
                f'|  D={D_kpc} kpc\n'
                f'{len(_J_c)} models  |  '
                f'log10(J) = '
                f'{_res_c["logJ_median"]:.2f} ± '
                f'{_res_c["logJ_std"]:.2f}'
            ),
        )
    except Exception as _e:
        print(f"  Ошибка corner-plot: {_e}")

print(f"\nГотово: {datetime.datetime.now()}")