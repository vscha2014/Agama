#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J-factor computation for Fornax dSph galaxy
Based on Schwarzschild orbit modelling results (AGAMA/forstand)

Reads log files from:
  /home/gala/Yandex.Disk/galAgama/4UpsBoTorch_Sersic.txt
  /home/gala/Yandex.Disk/galAgama/4UpsBoTorch_PCA_Sersic_*.txt
  /home/gala/Yandex.Disk/galAgama/out*.txt

Запуск в Spyder Console:
  %run J_factor_Sersic_Fornax_P21_symm.py
  %run J_factor_Sersic_Fornax_P21_symm.py --input-mask "out_*_d0_nb250_gh0_ser0*.txt"
"""

import argparse
import datetime
import glob
import os
import re

import numpy

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
YADISK_DIR = "/home/gala/Yandex.Disk/galAgama"
JFACTOR_DIR = os.path.join(YADISK_DIR, "J_factors")

# Паттерны файлов (относительно YADISK_DIR). --input-mask заменяет этот список.
LOG_PATTERNS_REL = [
    "4UpsBoTorch_Sersic.txt",
    "4UpsBoTorch_PCA_Sersic_*.txt",
    "out*.txt",
]

EXP_ID_RE = re.compile(r"(?:^|_)(d[01]_nb\d+_gh\d+_ser\d+)(?:_|\.txt$)")

# Параметры расчёта — редактируйте перед запуском
incl_target     = 90.0     # угол наклонения для фильтра (градусы)
penalty_cutoff  = None     # None = адаптивный (target_fraction лучших)
target_fraction = 0.30     # доля лучших точек при адаптивном cutoff
cutoff_start    = 0.60     # жёсткий потолок cutoff
D_kpc           = 143.0    # расстояние до галактики (kpc)
q_ap            = 1.0 - 0.31

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

def make_output_filename(prefix, theta_max_deg, incl, ext='txt',
                         experiment_id=None):
    """
    Формирует имя выходного файла вида:
        {prefix}_Sersic[_experiment]_incl{incl:.2f}_theta{theta:.1f}.{ext}
    Если theta_max_deg is None, суффикс _theta опускается.
    """
    base = f"{prefix}_Sersic"
    if experiment_id is not None:
        base += f"_{experiment_id}"
    base += f"_incl{incl:.2f}"
    if theta_max_deg is not None:
        base += f"_theta{theta_max_deg:.1f}"
    return f"{base}.{ext}"


def make_output_fullpath(prefix, theta_max_deg, incl, ext='txt',
                         yadisk_dir=YADISK_DIR, experiment_id=None):
    """
    Формирует полный путь к выходному файлу в директории yadisk_dir.

    Все выходные файлы (txt и pdf) сохраняются в ту же директорию,
    что и входные лог-файлы.

    Пример результата:
        /home/gala/Yandex.Disk/galAgama/corner_plot_Sersic_incl90.00_theta0.5.pdf
    """
    fname = make_output_filename(
        prefix, theta_max_deg, incl, ext, experiment_id
    )
    return os.path.join(yadisk_dir, fname)


def make_jfactor_output_filename(theta_max_deg, experiment_id=None, ext='txt'):
    base = 'J_factor_Sersic'
    if experiment_id is not None:
        base += f'_{experiment_id}'
    base += f'_theta{theta_max_deg:.1f}'
    return f'{base}.{ext}'


def make_jfactor_output_fullpath(theta_max_deg, experiment_id=None,
                                 yadisk_dir=YADISK_DIR):
    return os.path.join(
        yadisk_dir, 'J_factors',
        make_jfactor_output_filename(theta_max_deg, experiment_id),
    )


def experiment_id_from_filename(path):
    basename = os.path.basename(path)
    match = EXP_ID_RE.search(basename)
    if match is not None:
        return match.group(1)
    if basename.startswith('out'):
        raise ValueError(
            f"Не удалось извлечь experiment ID из файла {basename}; "
            "ожидается тег d<0|1>_nb<N>_gh<N>_ser<N>."
        )
    return None


def group_log_files(log_files):
    groups = {}
    for path in log_files:
        try:
            experiment_id = experiment_id_from_filename(path)
        except ValueError as error:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {error} Файл пропущен.")
            continue
        groups.setdefault(experiment_id, []).append(path)
    return {key: sorted(paths) for key, paths in groups.items()}


def compute_axRZst(incl_deg):
    beta = incl_deg * numpy.pi / 180.0
    sinb = numpy.sin(beta)
    cosb = numpy.cos(beta)
    return numpy.sqrt(q_ap**2 - cosb**2) / sinb


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


def discover_inclinations(log_files, verbose=True):
    """
    Сканирует лог-файлы и возвращает отсортированный список уникальных
    наклонений (колонка 0), встречающихся в 4Ups-файлах.

    Значения округляются до 0.01°, чтобы объединить идентичные наклонения,
    записанные с разной точностью.
    """
    incls = set()

    for filepath in log_files:
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
                        val = float(parts[0])
                    except ValueError:
                        continue
                    if numpy.isinf(val) or numpy.isnan(val):
                        continue
                    incls.add(round(val, 2))
        except FileNotFoundError:
            if verbose:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: файл не найден: {filepath}")

    result = sorted(incls)

    if verbose:
        print(f"\nНайдено уникальных наклонений: {len(result)}")
        print(f"  {result}")

    return result


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
    Q, gh, rh, rho0, Upsilon,
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
    rho0 — нормировка плотности ТМ в единицах кода (10⁶ M☉/кпк³).
    Upsilon — масштабный фактор массы; физическая нормировка = rho0 * Upsilon.

    Returns
    -------
    J_GeV2_cm5  : float — J-фактор в GeV²/cm⁵
    J_Msun_kpc5 : float — J-фактор в Msun²/kpc⁵·sr (до конвертации)
    """
    import agama

    kpc_to_cm = 3.0857e21
    rho_conv  = 1e6 * 1.989e33 / 1.783e-24 / (kpc_to_cm**3)

    density_DM = agama.Density(
        type              = 'spheroid',
        alpha             = alphah,
        beta              = betah,
        gamma             = gh,
        axisratioz        = Q,
        densitynorm       = rho0 * Upsilon,
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
#  ВЫЧИСЛЕНИЕ МАСС МОДЕЛИ
# ============================================================

def compute_model_masses(
    Q, gh, rh, rho0, Upsilon,
    alphah=2.0, betah=3.0,
    cutoff=55.0, cutoff_strength=2.5,
    massSt=14.0, scaleRst=None, Sersic_m=0.80, axRZst=1.0,
    enclosed_radii=(1.0,),
):
    import agama

    densityHalo = agama.Density(
        type='spheroid',
        alpha=alphah, beta=betah,
        gamma=gh, axisratioz=Q,
        densitynorm=rho0 * Upsilon, scaleradius=rh,
        outercutoffradius=cutoff, cutoffstrength=cutoff_strength,
    )
    densityStars = agama.Density(
        type='Sersic', sersicIndex=Sersic_m,
        mass=massSt * Upsilon, scaleRadius=scaleRst, axisRatioZ=axRZst,
    )
    densityTotal = agama.Density(densityStars, densityHalo)

    result = {}
    result['M_total_DM'] = float(densityHalo.totalMass())
    result['M_total_stars'] = float(densityStars.totalMass())
    result['M_total'] = float(densityTotal.totalMass())

    for r in enclosed_radii:
        m_dm = float(densityHalo.enclosedMass(r))
        m_st = float(densityStars.enclosedMass(r))
        result[f'M_DM_{r}'] = m_dm
        result[f'M_stars_{r}'] = m_st
        result[f'M_tot_{r}'] = m_dm + m_st

    return result


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
    log_files     = None,
    experiment_id = None,
    initialize_output = True,
    append_output = False,
):
    """
    Читает лог-файлы из yadisk_dir, вычисляет J-фактор для каждой
    хорошей точки и строит взвешенную статистику.

    Выходной текстовый файл сохраняется в yadisk_dir/J_factors.

    Returns
    -------
    results   : dict
    J_arr     : numpy.ndarray — J-значения (GeV²/cm⁵)
    data_good : numpy.ndarray — использованные строки данных
    """
    _incl   = incl_filter if incl_filter is not None else incl_target
    _cutoff = pen_cutoff  if pen_cutoff  is not None else penalty_cutoff

    if output_file is not None:
        _outfile = output_file
    else:
        _outfile = make_jfactor_output_fullpath(
            theta_max_deg, experiment_id, yadisk_dir
        )

    print(f"\nВыходной файл: {_outfile}")

    if log_files is None:
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

    if initialize_output:
        output_dir = os.path.dirname(_outfile)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        append_existing = append_output and os.path.isfile(_outfile)
        if os.path.isfile(_outfile) and not append_output:
            print(f"ВНИМАНИЕ: существующий файл будет перезаписан: {_outfile}")
        with open(_outfile, 'a' if append_existing else 'w') as fout:
            if append_existing:
                fout.write(f"\n# Appended J-factor computation  {datetime.datetime.now()}\n")
            else:
                fout.write(f"# J-factor computation  {datetime.datetime.now()}\n")
                fout.write(f"# theta_max={theta_max_deg} deg, D={D_kpc} kpc\n")
                if experiment_id is not None:
                    fout.write(f"# experiment_id={experiment_id}\n")
                fout.write(f"# alphah={alphah}, betah={betah}\n")
                fout.write(f"# Source directory: {yadisk_dir}\n")
                fout.write(
                    "# incl Q gh rh rho0 Upsilon rho0_x_Ups penalty "
                    "J_GeV2_cm5 log10_J\n"
                )
            fout.write("# Source files for this run:\n")
            for fname in file_counts:
                fout.write(f"#   {fname}\n")

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
                Q=Q, gh=gh, rh=rh, rho0=rho0, Upsilon=Upsilon,
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
                    f"{_incl:.2f} {Q:.10f} {gh:.10f} {rh:.10f} {rho0:.10f} "
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
        fout.write(f"\n# ============ RESULTS incl={_incl:.2f} ============\n")
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
                     yadisk_dir=YADISK_DIR,
                     experiment_id=None):
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
            experiment_id = experiment_id,
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
    print(f"{'incl':>7s}  {'theta':>8s}  {'log10(J)':>10s}  "
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
        print(f"  {r['incl']:5.2f}°  {r['theta_max']:6.2f}°  {med:10.3f}  "
              f"{p1:+7.3f}  {m1:+7.3f}  "
              f"{p2:+7.3f}  {m2:+7.3f}  "
              f"{r['n_points']:5d}  "
              f"{os.path.basename(r['output_file'])}")
    print("=" * 80)


# ============================================================
#  ГИСТОГРАММЫ МАССЫ МОДЕЛИ
# ============================================================

def make_mass_histograms(
    data_good, J_arr,
    enclosed_radii=(1.0,),
    massSt=14.0, scaleRst=None, Sersic_m=0.80, axRZst=1.0,
    alphah=2.0, betah=3.0,
    incl=None, yadisk_dir=YADISK_DIR, experiment_id=None,
):
    _incl = incl if incl is not None else incl_target
    n_J = len(J_arr)

    Q_arr     = data_good[:n_J, 1]
    gh_arr    = data_good[:n_J, 2]
    rh_arr    = data_good[:n_J, 3]
    rho0_arr  = data_good[:n_J, 4]
    Ups_arr   = data_good[:n_J, 5]

    _axRZst = compute_axRZst(_incl)

    all_radii = list(enclosed_radii) + ['total']
    mass_data = {r: [] for r in all_radii}
    dm_frac_data = {r: [] for r in all_radii}

    for i in range(n_J):
        try:
            masses = compute_model_masses(
                Q=Q_arr[i], gh=gh_arr[i], rh=rh_arr[i],
                rho0=rho0_arr[i], Upsilon=Ups_arr[i],
                alphah=alphah, betah=betah,
                massSt=massSt, scaleRst=scaleRst,
                Sersic_m=Sersic_m, axRZst=_axRZst,
                enclosed_radii=enclosed_radii,
            )
        except Exception:
            continue

        for r in enclosed_radii:
            m_tot = masses[f'M_tot_{r}']
            m_dm  = masses[f'M_DM_{r}']
            mass_data[r].append(m_tot)
            dm_frac_data[r].append(m_dm / m_tot if m_tot > 0 else 0)

        m_tot = masses['M_total']
        m_dm  = masses['M_total_DM']
        mass_data['total'].append(m_tot)
        dm_frac_data['total'].append(m_dm / m_tot if m_tot > 0 else 0)

    for r in all_radii:
        m_vals = numpy.array(mass_data[r]) / 10.0
        f_vals = numpy.array(dm_frac_data[r])
        if len(m_vals) == 0:
            continue

        is_total = (r == 'total')

        fig, ax = plt.subplots(figsize=(10, 6))

        if is_total:
            n_bins = 20
            bin_edges = numpy.linspace(550, 850, n_bins + 1)
        else:
            n_bins = 20
            pad = (m_vals.max() - m_vals.min()) * 0.1
            lo_auto = m_vals.min() - pad
            hi_auto = m_vals.max() + pad
            bin_edges = numpy.linspace(lo_auto, hi_auto, n_bins + 1)

        bin_widths = numpy.diff(bin_edges)

        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b+1]
            mask = (m_vals >= lo) & (m_vals < hi)
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue

            f_bin = f_vals[mask]

            groups = []
            remaining = list(range(n_in_bin))
            while remaining:
                idx0 = remaining.pop(0)
                group = [idx0]
                f_ref = f_bin[idx0]
                remaining2 = []
                for j in remaining:
                    if abs(f_bin[j] - f_ref) <= 0.10:
                        group.append(j)
                    else:
                        remaining2.append(j)
                remaining = remaining2
                avg_f = f_bin[group].mean()
                groups.append((len(group), avg_f))

            n_cols = len(groups)
            total_w = bin_widths[b] * 0.8
            col_w = total_w / n_cols if n_cols > 0 else total_w

            for col_idx, (cnt, avg_f_dm) in enumerate(groups):
                n_dm = cnt * avg_f_dm
                n_st = cnt * (1 - avg_f_dm)
                x_pos = lo + bin_widths[b] * 0.1 + col_idx * col_w + col_w / 2

                ax.bar(x_pos, n_dm, width=col_w * 0.85,
                       color='#2c3e50', alpha=0.85, linewidth=0)
                ax.bar(x_pos, n_st, width=col_w * 0.85, bottom=n_dm,
                       color='#bdc3c7', alpha=0.85, linewidth=0)

        if is_total:
            r_label = 'полная масса'
            ax.set_xlim(550, 850)
        else:
            r_label = f'r < {r} кпк'

        ax.set_xlabel(r'$M\;[10^7\;M_\odot]$', fontsize=11)
        ax.set_ylabel('Число моделей', fontsize=11)
        ax.set_title(f'Fornax dSph, incl={_incl:.2f}°, {r_label}, {len(m_vals)} моделей',
                     fontsize=12)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2c3e50', label='Тёмная материя'),
            Patch(facecolor='#bdc3c7', label='Звёзды'),
        ]
        if not is_total:
            ax.legend(handles=legend_elements, fontsize=8, loc='upper right')

        outfile = make_output_fullpath(
            prefix=f'mass_histogram_{r}',
            theta_max_deg=None, incl=_incl, ext='pdf', yadisk_dir=yadisk_dir,
            experiment_id=experiment_id,
        )
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Гистограмма массы сохранена: {outfile}")


# ============================================================
#  ЗАПУСК (выполняется при %run или из командной строки)
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='J-factor computation for Fornax dSph')
    parser.add_argument(
        '--input-mask', action='append', dest='input_masks', default=None,
        help=("Glob-маска входных файлов относительно YADISK_DIR. Можно указать "
              "несколько раз. Если задана, стандартные маски не используются."),
    )
    parser.add_argument(
        '--append', action='store_true',
        help=("Дописывать выбранные --input-mask расчёты в существующие J-файлы. "
              "Повторный запуск с той же маской добавит строки повторно."),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.append and not args.input_masks:
        raise ValueError("--append требует хотя бы одну --input-mask")
    patterns_rel = args.input_masks or LOG_PATTERNS_REL

    print("=" * 60)
    print("J-FACTOR COMPUTATION  Fornax dSph")
    print(f"Start: {datetime.datetime.now()}")
    print(f"D = {D_kpc} kpc")
    print(f"Входные файлы  ← {YADISK_DIR}")
    print(f"J-файлы        → {JFACTOR_DIR}")
    print(f"Маски входа    : {patterns_rel}")
    print("=" * 60)

    log_files = collect_log_files(
        yadisk_dir=YADISK_DIR, patterns_rel=patterns_rel, verbose=True
    )
    grouped_files = group_log_files(log_files)
    if not grouped_files:
        raise RuntimeError("Среди найденных файлов нет распознаваемых историй расчёта.")
    all_results = []

    theta_corner = 0.5
    massSt   = 14.0
    scaleRst = numpy.pi * D_kpc / 180 * 16.4 / 60
    Sersic_m = 0.80

    for experiment_id in sorted(grouped_files, key=lambda value: value or ''):
        experiment_files = grouped_files[experiment_id]
        experiment_label = experiment_id or 'legacy'
        print(f"\n{'#'*60}")
        print(f"# Эксперимент: {experiment_label}")
        print(f"# Файлов: {len(experiment_files)}")
        print('#'*60)

        incl_values = discover_inclinations(experiment_files, verbose=True)
        if not incl_values:
            print(f"  Нет наклонений для эксперимента {experiment_label}; пропуск.")
            continue

        corner_data = {}
        for theta in theta_list:
            output_file = make_jfactor_output_fullpath(
                theta, experiment_id, YADISK_DIR
            )
            initialize_output = True
            for incl in incl_values:
                print(f"\n{'='*50}")
                print(f"experiment={experiment_label}  incl={incl}°  theta_max={theta}°")
                print('='*50)

                try:
                    results, J_arr, data_used = compute_J_from_logs(
                        theta_max_deg=theta,
                        incl_filter=incl,
                        output_file=output_file,
                        yadisk_dir=YADISK_DIR,
                        log_files=experiment_files,
                        experiment_id=experiment_id,
                        initialize_output=initialize_output,
                        append_output=args.append,
                    )
                    all_results.append(results)
                    if abs(theta - theta_corner) < 1e-9:
                        corner_data[incl] = (data_used, J_arr, results)
                except Exception as error:
                    print(f"  ОШИБКА для experiment={experiment_label}, "
                          f"incl={incl}, theta={theta}: {error}")
                finally:
                    initialize_output = False

        if not do_corner_plot:
            continue
        for incl, (data_c, J_c, res_c) in corner_data.items():
            try:
                make_corner_plot(
                    data_good=data_c,
                    J_arr=J_c,
                    theta_max_deg=theta_corner,
                    incl=incl,
                    yadisk_dir=YADISK_DIR,
                    experiment_id=experiment_id,
                    title=(
                        f'Fornax dSph  |  {experiment_label}  |  incl={incl:.2f}°  '
                        f'|  theta<{theta_corner:.1f}°  |  D={D_kpc} kpc\n'
                        f'{len(J_c)} models  |  log10(J) = '
                        f'{res_c["logJ_median"]:.2f} ± {res_c["logJ_std"]:.2f}'
                    ),
                )
            except Exception as error:
                print(f"  Ошибка corner-plot (experiment={experiment_label}, "
                      f"incl={incl}): {error}")

            try:
                make_mass_histograms(
                    data_good=data_c,
                    J_arr=J_c,
                    enclosed_radii=(1.0,),
                    massSt=massSt,
                    scaleRst=scaleRst,
                    Sersic_m=Sersic_m,
                    axRZst=None,
                    incl=incl,
                    yadisk_dir=YADISK_DIR,
                    experiment_id=experiment_id,
                )
            except Exception as error:
                print(f"  Ошибка гистограмм массы (experiment={experiment_label}, "
                      f"incl={incl}): {error}")

    if all_results:
        print_summary_table(all_results)

    print(f"\nГотово: {datetime.datetime.now()}")


if __name__ == '__main__':
    main()