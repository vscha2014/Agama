#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гистограмма масс модели Fornax dSph — ПО ВСЕМ НАКЛОНЕНИЯМ СУММАРНО.

Отдельный (самостоятельный) вариант блока mass-histogram из
py/J_factor_Sersic_Fornax_P21_symm.py, но:
  * БЕЗ расчёта J-фактора (только массы модели через AGAMA);
  * ОДНА суммарная гистограмма по всем наклонениям (а не по каждому отдельно);
  * ВЗВЕШИВАНИЕ ПО penalty (в исходном коде гистограмма масс НЕ взвешивалась —
    каждая модель считалась как 1; по penalty там взвешивалась только
    статистика J). Здесь вес w = exp(-(penalty - pen_min) / pen_sigma),
    как для J-статистики в исходном коде. Опция --no-weight отключает вес.

Массы (M_tot, M_DM) считаются аналитически через agama.Density (spheroid-гало
+ Sersic-звёзды), поэтому это НЕ production-оптимизация и НЕ генерация орбит.
Скрипт только читает сырые 4Ups-файлы и строит рисунок; исходные данные не
изменяются.

Формат строки 4Ups-файла (7 чисел):
    incl  Q  gh  rh  rho0  Upsilon  penalty

Физические константы и модель звёзд/гало взяты из
py/J_factor_Sersic_Fornax_P21_symm.py (научный контракт §7):
    q_ap = 1 - 0.31,  D = 143 kpc,  massSt = 14.0,  Sersic_m = 0.80,
    scaleRst = pi*D/180*16.4/60 kpc,  alphah = 2.0, betah = 3.0.
"""

import os
import glob
import argparse
import datetime
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------
#  Константы модели (как в J_factor_Sersic_Fornax_P21_symm.py)
# ------------------------------------------------------------------
YADISK_DIR = "/home/gala/Yandex.Disk/galAgama"
LOG_PATTERNS_REL = [
    "4UpsBoTorch_Sersic.txt",
    "4UpsBoTorch_PCA_Sersic_*.txt",
]

D_kpc    = 143.0
q_ap     = 1.0 - 0.31
alphah   = 2.0
betah    = 3.0
cutoff_halo          = 55.0
cutoff_strength_halo = 2.5

massSt   = 14.0
Sersic_m = 0.80
scaleRst = numpy.pi * D_kpc / 180.0 * 16.4 / 60.0   # kpc (16.4 arcmin)


# ------------------------------------------------------------------
#  Загрузка данных ПО ВСЕМ наклонениям
# ------------------------------------------------------------------
def collect_log_files(yadisk_dir, patterns_rel, verbose=True):
    if not os.path.isdir(yadisk_dir):
        raise FileNotFoundError(f"Директория не найдена: {yadisk_dir}")
    found = []
    for pattern_rel in patterns_rel:
        for f in sorted(glob.glob(os.path.join(yadisk_dir, pattern_rel))):
            if f not in found and os.path.isfile(f):
                found.append(f)
    if verbose:
        print(f"\nДиректория поиска: {yadisk_dir}")
        print(f"Найдено лог-файлов: {len(found)}")
        for f in found:
            print(f"  {os.path.basename(f)}  "
                  f"({os.path.getsize(f)/1024:.1f} KB)")
    if not found:
        raise FileNotFoundError(
            f"Нет файлов в {yadisk_dir} по паттернам {patterns_rel}")
    return found


def load_combined_file(path, verbose=True):
    """Читает combined-файл (diagnose_J_weighting --source raw):
        incl Q gh rh rho0 Ups rho0xUps penalty J log10J count

    Возвращает data (N,7) [incl,Q,gh,rh,rho0,Ups,penalty] и mult (кратность
    из колонки count; если её нет — единицы). Кратность нужна, чтобы суммарная
    гистограмма совпадала с чтением сырых строк 4Ups (где дубликаты считаются
    отдельно), см. схлопывание дубликатов в diagnose_J_weighting.
    """
    rows, mult = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            p = s.split()
            if len(p) < 8:
                continue
            try:
                vals = [float(x) for x in p]
            except ValueError:
                continue
            incl, Q, gh, rh, rho0, Ups = vals[0:6]
            penalty = vals[7]
            if rh <= 0 or rho0 <= 0 or penalty >= 1e5:
                continue
            if any(numpy.isinf(v) or numpy.isnan(v)
                   for v in (incl, Q, gh, rh, rho0, Ups, penalty)):
                continue
            rows.append([incl, Q, gh, rh, rho0, Ups, penalty])
            mult.append(vals[10] if len(vals) >= 11 else 1.0)
    if not rows:
        raise SystemExit(f"Нет данных в {path}")
    data = numpy.array(rows)
    mult = numpy.array(mult, dtype=float)
    if verbose:
        incls, cnts = numpy.unique(numpy.round(data[:, 0], 2),
                                   return_counts=True)
        print(f"\nCombined-файл: {path}")
        print(f"Уникальных точек: {len(data)}  (суммарная кратность "
              f"{mult.sum():.0f})")
        print("Наклонения (incl: N_uniq):")
        for v, c in zip(incls, cnts):
            print(f"  {v:6.2f}: {c}")
        print(f"Диапазон penalty: "
              f"[{data[:, 6].min():.4f}, {data[:, 6].max():.4f}]")
    return data, mult


def load_all_data(log_files, verbose=True):
    """Читает ВСЕ строки (все наклонения). Возвращает массив (N,7)."""
    raw, file_counts = [], {}
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
                    except ValueError:
                        continue
                    if any(numpy.isinf(v) or numpy.isnan(v) for v in row):
                        continue
                    if row[3] <= 0 or row[4] <= 0:   # rh, rho0
                        continue
                    if row[6] >= 1e5:                # penalty
                        continue
                    raw.append(row)
                    count += 1
        except FileNotFoundError:
            if verbose:
                print(f"  ПРЕДУПРЕЖДЕНИЕ: файл не найден: {filepath}")
        file_counts[os.path.basename(filepath)] = count

    if verbose:
        print("\nЗагружено строк (все наклонения):")
        for fname, cnt in file_counts.items():
            print(f"  {fname}: {cnt}")
    if not raw:
        raise SystemExit("Нет данных в найденных файлах.")

    data = numpy.array(raw)
    if verbose:
        incls, cnts = numpy.unique(numpy.round(data[:, 0], 2),
                                   return_counts=True)
        print(f"\nИтого точек: {len(data)}")
        print("Наклонения (incl: N):")
        for v, c in zip(incls, cnts):
            print(f"  {v:6.2f}: {c}")
        print(f"Диапазон penalty: "
              f"[{data[:, 6].min():.4f}, {data[:, 6].max():.4f}]")
    return data, file_counts


# ------------------------------------------------------------------
#  Cutoff / веса / перцентиль (как в J_factor-скрипте)
# ------------------------------------------------------------------
def adaptive_penalty_cutoff(penalties, target_fraction=0.30,
                            cutoff_start=0.60, min_points=10):
    penalties = numpy.asarray(penalties)
    if len(penalties) <= min_points:
        return float(numpy.max(penalties))
    cutoff = float(numpy.percentile(penalties, target_fraction * 100))
    if cutoff > cutoff_start:
        if numpy.sum(penalties <= cutoff_start) >= min_points:
            cutoff = cutoff_start
    if numpy.sum(penalties <= cutoff) < min_points:
        cutoff = float(numpy.sort(penalties)[min_points - 1])
    return cutoff


def penalty_weights(penalties):
    """w = exp(-(penalty - pen_min) / pen_sigma), нормировка sum(w)=N."""
    pen = numpy.asarray(penalties, dtype=float)
    pen_min = pen.min()
    pen_sigma = max(pen.std(), 1e-6)
    w = numpy.exp(-(pen - pen_min) / pen_sigma)
    w *= len(w) / w.sum()          # сумма весов = числу моделей (N_eff)
    return w, float(pen_min), float(pen_sigma)


def weighted_percentile(values, weights, percentiles):
    values = numpy.asarray(values, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    idx = numpy.argsort(values)
    v_sort, w_sort = values[idx], weights[idx]
    w_cum = numpy.cumsum(w_sort)
    w_cum /= w_cum[-1]
    return numpy.interp(numpy.asarray(percentiles) / 100.0, w_cum, v_sort)


# ------------------------------------------------------------------
#  Массы модели (agama импортируется лениво)
# ------------------------------------------------------------------
def compute_axRZst(incl_deg):
    beta = incl_deg * numpy.pi / 180.0
    sinb, cosb = numpy.sin(beta), numpy.cos(beta)
    val = q_ap**2 - cosb**2
    if val <= 0 or sinb <= 0:
        return numpy.nan
    return numpy.sqrt(val) / sinb


def compute_model_masses(agama, Q, gh, rh, rho0, Upsilon, axRZst,
                         enclosed_radii):
    densityHalo = agama.Density(
        type='spheroid', alpha=alphah, beta=betah,
        gamma=gh, axisratioz=Q,
        densitynorm=rho0 * Upsilon, scaleradius=rh,
        outercutoffradius=cutoff_halo, cutoffstrength=cutoff_strength_halo,
    )
    densityStars = agama.Density(
        type='Sersic', sersicIndex=Sersic_m,
        mass=massSt * Upsilon, scaleRadius=scaleRst, axisRatioZ=axRZst,
    )
    result = {
        'M_total_DM': float(densityHalo.totalMass()),
        'M_total': float(densityHalo.totalMass())
                   + float(densityStars.totalMass()),
    }
    for r in enclosed_radii:
        m_dm = float(densityHalo.enclosedMass(r))
        m_st = float(densityStars.enclosedMass(r))
        result[f'M_DM_{r}'] = m_dm
        result[f'M_tot_{r}'] = m_dm + m_st
    return result


# ------------------------------------------------------------------
#  Построение одной суммарной гистограммы (стек ТМ+звёзды)
# ------------------------------------------------------------------
def draw_hist(ax, m_vals, f_vals, w_vals, nbins, weighted):
    """Стекированная гистограмма как в исходном коде, но с весами.

    В каждом бине по массе точки группируются по близкой доле ТМ (±0.10),
    высота субколонки = сумма весов группы; стек: ТМ снизу, звёзды сверху.
    """
    lo = numpy.percentile(m_vals, 0.5)
    hi = numpy.percentile(m_vals, 99.5)
    pad = (hi - lo) * 0.05 if hi > lo else max(abs(hi) * 0.05, 1e-3)
    bin_edges = numpy.linspace(lo - pad, hi + pad, nbins + 1)
    bin_widths = numpy.diff(bin_edges)

    for b in range(nbins):
        blo, bhi = bin_edges[b], bin_edges[b + 1]
        mask = (m_vals >= blo) & (m_vals < bhi)
        if not numpy.any(mask):
            continue
        f_bin = f_vals[mask]
        w_bin = w_vals[mask]

        # группировка по доле ТМ (в пределах 0.10)
        groups = []
        remaining = list(range(len(f_bin)))
        while remaining:
            idx0 = remaining.pop(0)
            group = [idx0]
            f_ref = f_bin[idx0]
            rest = []
            for j in remaining:
                if abs(f_bin[j] - f_ref) <= 0.10:
                    group.append(j)
                else:
                    rest.append(j)
            remaining = rest
            gw = w_bin[group].sum()
            avg_f = numpy.sum(w_bin[group] * f_bin[group]) / gw
            groups.append((gw, avg_f))

        n_cols = len(groups)
        total_w = bin_widths[b] * 0.8
        col_w = total_w / n_cols if n_cols > 0 else total_w
        for col_idx, (wsum, avg_f_dm) in enumerate(groups):
            n_dm = wsum * avg_f_dm
            n_st = wsum * (1 - avg_f_dm)
            x_pos = blo + bin_widths[b] * 0.1 + col_idx * col_w + col_w / 2
            ax.bar(x_pos, n_dm, width=col_w * 0.85,
                   color='#2c3e50', alpha=0.85, linewidth=0)
            ax.bar(x_pos, n_st, width=col_w * 0.85, bottom=n_dm,
                   color='#bdc3c7', alpha=0.85, linewidth=0)

    return bin_edges


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description="Суммарная (по всем наклонениям) гистограмма масс модели, "
                    "без расчёта J, взвешенная по penalty.")
    ap.add_argument('--yadisk-dir', default=YADISK_DIR,
                    help="Каталог с 4Ups-файлами (по умолчанию Yandex.Disk).")
    ap.add_argument('--combined-file', default=None,
                    help="Вместо сырых 4Ups читать combined-файл "
                         "(diagnose_J_weighting --source raw). Учитывает "
                         "кратность count. Удобно без доступа к Яндекс.Диску.")
    ap.add_argument('--out-dir', default=script_dir,
                    help="Каталог для PDF (по умолчанию каталог скрипта).")
    ap.add_argument('--radii', default='1.0',
                    help="Радиусы enclosed-массы (кпк) через запятую.")
    ap.add_argument('--cutoff', type=float, default=None,
                    help="Фиксированный penalty-cutoff (по умолч. адаптивный).")
    ap.add_argument('--target-fraction', type=float, default=0.30,
                    help="Доля лучших точек при адаптивном cutoff.")
    ap.add_argument('--cutoff-start', type=float, default=0.60,
                    help="Жёсткий потолок адаптивного cutoff.")
    ap.add_argument('--nbins', type=int, default=20,
                    help="Число бинов по массе.")
    ap.add_argument('--no-weight', action='store_true',
                    help="Отключить взвешивание по penalty (все веса = 1).")
    args = ap.parse_args()

    weighted = not args.no_weight
    enclosed_radii = tuple(float(x) for x in args.radii.split(',') if x.strip())

    # --- данные (сырые 4Ups или combined-файл) ---
    if args.combined_file is not None:
        data, mult = load_combined_file(args.combined_file)
    else:
        log_files = collect_log_files(args.yadisk_dir, LOG_PATTERNS_REL)
        data, _ = load_all_data(log_files)
        mult = numpy.ones(len(data))

    # --- глобальный cutoff по всем наклонениям ---
    if args.cutoff is None:
        cutoff = adaptive_penalty_cutoff(
            data[:, 6], target_fraction=args.target_fraction,
            cutoff_start=args.cutoff_start)
        print(f"\nАдаптивный penalty cutoff (глобальный): {cutoff:.4f} "
              f"(лучшие {args.target_fraction*100:.0f}%)")
    else:
        cutoff = args.cutoff
        print(f"\nЗаданный penalty cutoff: {cutoff:.4f}")

    good_mask = data[:, 6] <= cutoff
    good = data[good_mask].copy()
    mult = mult[good_mask]
    print(f"Хороших точек (penalty ≤ {cutoff:.4f}): {len(good)} "
          f"(суммарная кратность {mult.sum():.0f})")
    if len(good) == 0:
        raise SystemExit("Нет точек ниже cutoff.")

    # --- веса: penalty-вес * кратность, нормировка на суммарное число моделей ---
    if weighted:
        base, pen_min, pen_sigma = penalty_weights(good[:, 6])
        wtag, wlabel = 'weighted', 'взвеш. по penalty'
    else:
        base = numpy.ones(len(good))
        pen_min, pen_sigma = float(good[:, 6].min()), float('nan')
        wtag, wlabel = 'unweighted', 'без веса'
    w_all = base * mult
    w_all *= mult.sum() / w_all.sum()

    # --- массы модели (AGAMA) ---
    import agama  # ленивый импорт: нужен только для расчёта масс
    print(f"\nРасчёт масс модели для {len(good)} точек ...")

    all_radii = list(enclosed_radii) + ['total']
    mass_data = {r: [] for r in all_radii}
    fdm_data = {r: [] for r in all_radii}
    w_data = {r: [] for r in all_radii}

    n_bad = 0
    for i in range(len(good)):
        incl, Q, gh, rh, rho0, Ups, _pen = good[i]
        axRZst = compute_axRZst(incl)
        if not numpy.isfinite(axRZst):
            n_bad += 1
            continue
        try:
            m = compute_model_masses(agama, Q, gh, rh, rho0, Ups,
                                     axRZst, enclosed_radii)
        except Exception:
            n_bad += 1
            continue
        wi = w_all[i]
        for r in enclosed_radii:
            mt = m[f'M_tot_{r}']
            mass_data[r].append(mt)
            fdm_data[r].append(m[f'M_DM_{r}'] / mt if mt > 0 else 0.0)
            w_data[r].append(wi)
        mt = m['M_total']
        mass_data['total'].append(mt)
        fdm_data['total'].append(m['M_total_DM'] / mt if mt > 0 else 0.0)
        w_data['total'].append(wi)

    if n_bad:
        print(f"  пропущено (невалидный axRZst / ошибка): {n_bad}")

    # --- рисунки ---
    print("\n" + "=" * 64)
    print(f"СУММАРНАЯ ГИСТОГРАММА МАСС ({wlabel}); N={len(good)}, "
          f"наклонений={len(numpy.unique(numpy.round(good[:,0],2)))}")
    print("=" * 64)

    n_incl = len(numpy.unique(numpy.round(good[:, 0], 2)))
    for r in all_radii:
        m_vals = numpy.array(mass_data[r]) / 10.0     # -> 1e7 Msun
        f_vals = numpy.array(fdm_data[r])
        w_vals = numpy.array(w_data[r])
        if len(m_vals) == 0:
            continue

        # взвешенная статистика массы (полная, ТМ и звёзды раздельно)
        dm_vals = m_vals * f_vals            # масса ТМ [1e7 Msun]
        st_vals = m_vals * (1.0 - f_vals)    # масса звёзд [1e7 Msun]
        p = weighted_percentile(m_vals, w_vals, [16, 50, 84])
        pdm = weighted_percentile(dm_vals, w_vals, [16, 50, 84])
        pst = weighted_percentile(st_vals, w_vals, [16, 50, 84])
        r_label = 'полная масса' if r == 'total' else f'r < {r} кпк'
        print(f"  {r_label:>16}: M50={p[1]:.2f}  "
              f"[{p[0]:.2f}, {p[2]:.2f}] (1σ) [1e7 Msun]")
        print(f"  {'в т.ч. ТМ':>16}: M50={pdm[1]:.2f}  "
              f"[{pdm[0]:.2f}, {pdm[2]:.2f}] (1σ) [1e7 Msun]")
        print(f"  {'в т.ч. звёзды':>16}: M50={pst[1]:.2f}  "
              f"[{pst[0]:.2f}, {pst[2]:.2f}] (1σ) [1e7 Msun]")

        fig, ax = plt.subplots(figsize=(10, 6))
        draw_hist(ax, m_vals, f_vals, w_vals, args.nbins, weighted)

        ax.set_xlabel(r'$M\;[10^7\;M_\odot]$', fontsize=11)
        ax.set_ylabel('Взвеш. число моделей (Σw)' if weighted
                      else 'Число моделей', fontsize=11)
        ax.set_title(
            f'Fornax dSph — все наклонения ({n_incl}), {r_label}, '
            f'{len(m_vals)} моделей, {wlabel}', fontsize=12)
        ax.legend(handles=[
            Patch(facecolor='#2c3e50', label='Тёмная материя'),
            Patch(facecolor='#bdc3c7', label='Звёзды'),
        ], fontsize=9, loc='upper right')

        fname = f'mass_histogram_all_incl_{r}_{wtag}.pdf'
        outpath = os.path.join(args.out_dir, fname)
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    сохранено: {outpath}")

    print("=" * 64)
    print(f"Cutoff={cutoff:.4f}, pen_min={pen_min:.4f}, "
          f"pen_sigma={pen_sigma}")
    print("ПРИМЕЧАНИЕ: penalty-веса делают гистограмму профиль-ПОДОБНОЙ по "
          "качеству фита,\nно точки получены оптимизатором (BoTorch) — это НЕ "
          "постериор (см. вопрос PI #16).")


if __name__ == '__main__':
    main()
