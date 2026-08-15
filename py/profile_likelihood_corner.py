#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Профиль-правдоподобные 2D-карты + corner-plot (вариант C).

READ-ONLY. Строится по уже посчитанному combined-файлу
(diagnose_J_weighting.py --source raw), колонки которого:
    incl Q gh rh rho0 Upsilon rho0_x_Ups penalty J_GeV2_cm5 log10_J count

ИДЕЯ (профиль-правдоподобие, а не постериор):
  - точки получены БАЙЕСОВСКОЙ ОПТИМИЗАЦИЕЙ (BoTorch/TuRBO), поэтому их ПЛОТНОСТЬ
    не является постериором. Но для профиля нужен только МИНИМУМ penalty в каждом
    срезе параметра — это оптимизатор покрывает хорошо.
  - penalty — это chi^2 по GH-моментам (schwarzlib.getPenalty), НО с неопределённым
    числом эффективных степеней свободы (регуляризация весов орбит, симметризация),
    поэтому уровни Δpenalty здесь — НОМИНАЛЬНЫЕ Δchi^2, НЕ калиброванные
    доверительные. См. обсуждение в чате и вопрос PI #16.

ЧТО СТРОИТ:
  - диагональ: 1D профиль penalty(θ_k) = min_остальные penalty, как Δpenalty(θ_k),
    с номинальными уровнями Δchi^2 для 1 параметра (1.0, 3.84);
  - нижний треугольник: 2D карты Δpenalty(θ_j, θ_k) = min по ячейке, с контурами
    номинальных Δchi^2 для 2 параметров (2.30, 6.18, 11.83);
  - верхний треугольник: коэффициенты корреляции Пирсона по отобранным точкам
    (dpenalty<=--view-dpen). ЭТО ОПИСАТЕЛЬНАЯ корреляция ВЫБОРКИ ОПТИМИЗАТОРА,
    а НЕ постериорная корреляция параметров;
  - глобальный best-fit отмечается крестом.

Ось наклонения (incl) — дискретная сетка с неравномерным шагом (сгущение к 90°),
поэтому она рисуется в РАВНОМЕРНОМ (категориальном) масштабе: каждый узел сетки
занимает одинаковую ширину, тики подписаны реальными градусами. Так область
около 90° хорошо видна. Биннинг penalty при этом ведётся по реальным значениям.

Оси по умолчанию (как просил пользователь):
    incl, Q, gh (gamma_h), rh, rho0*Upsilon, Upsilon, log10(J)

Скрипт ничего не вычисляет через AGAMA и не меняет исходные файлы.
"""

import os
import argparse
import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ------------------------------------------------------------------
#  Колонки combined-файла
# ------------------------------------------------------------------
COL = {
    'incl': 0, 'Q': 1, 'gh': 2, 'rh': 3, 'rho0': 4, 'Ups': 5,
    'rho0xUps': 6, 'penalty': 7, 'J': 8, 'log10J': 9, 'count': 10,
}
COL_PENALTY = COL['penalty']

# Подписи осей
LABELS = {
    'incl': r'$i$ [deg]',
    'Q': r'$Q\ (a_{R/Z})$',
    'gh': r'$\gamma_h$',
    'rh': r'$r_h$ [kpc]',
    'rho0xUps': r'$\rho_0\cdot\Upsilon$',
    'Ups': r'$\Upsilon$',
    'log10J': r'$\log_{10} J$',
}

# Номинальные (НЕ калиброванные) уровни Δchi^2
LEVELS_1D = [1.0, 3.84]                 # 68%, 95% для 1 параметра
LEVELS_2D = [2.30, 6.18, 11.83]         # 68%, 95%, 99% для 2 параметров


def load_combined(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            p = s.split()
            if len(p) < 10:
                continue
            try:
                vals = [float(x) for x in p[:11]] if len(p) >= 11 \
                    else [float(x) for x in p[:10]] + [1.0]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise SystemExit(f"Нет данных в {path}")
    return numpy.array(rows)


def axis_spec(values_full, keep, nbins, discrete_max=12):
    """Спецификация оси для биннинга и отрисовки.

    Возвращает dict:
      edges        — рёбра бинов в РЕАЛЬНЫХ значениях (для digitize/profile);
      edges_plot   — рёбра в координатах ОТРИСОВКИ;
      centers_plot — центры бинов в координатах отрисовки;
      to_plot(v)   — функция: реальное значение -> координата отрисовки;
      ticks        — (позиции, подписи) или None (для дискретной оси);
      discrete     — bool.

    Для дискретной оси (мало уникальных значений, напр. incl) координата
    отрисовки — РАВНОМЕРНАЯ категориальная (0,1,2,...), чтобы неравномерная
    сетка (сгущение к 90°) не сжимала интересную область.
    """
    uniq = numpy.unique(values_full)
    if len(uniq) <= discrete_max:
        n = len(uniq)
        if n == 1:
            d = max(abs(uniq[0]) * 0.05, 1e-6)
            edges = numpy.array([uniq[0] - d, uniq[0] + d])
        else:
            mids = (uniq[:-1] + uniq[1:]) / 2.0
            left = uniq[0] - (mids[0] - uniq[0])
            right = uniq[-1] + (uniq[-1] - mids[-1])
            edges = numpy.concatenate(([left], mids, [right]))
        centers_plot = numpy.arange(n, dtype=float)
        edges_plot = numpy.arange(n + 1, dtype=float) - 0.5

        def to_plot(a, uniq=uniq):
            a = numpy.asarray(a, dtype=float)
            idx = numpy.abs(a.reshape(-1, 1) - uniq.reshape(1, -1)).argmin(axis=1)
            out = idx.astype(float)
            return out[0] if a.ndim == 0 else out.reshape(a.shape)

        ticks = (centers_plot, [f"{u:g}" for u in uniq])
        return dict(edges=edges, edges_plot=edges_plot,
                    centers_plot=centers_plot, to_plot=to_plot,
                    ticks=ticks, discrete=True)

    vals = values_full[keep] if numpy.any(keep) else values_full
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = numpy.linspace(lo, hi, nbins + 1)
    centers_plot = 0.5 * (edges[:-1] + edges[1:])
    return dict(edges=edges, edges_plot=edges, centers_plot=centers_plot,
                to_plot=lambda a: a, ticks=None, discrete=False)


def profile_1d(x, pen, edges):
    """min penalty в каждом бине оси x."""
    nb = len(edges) - 1
    idx = numpy.clip(numpy.digitize(x, edges) - 1, 0, nb - 1)
    prof = numpy.full(nb, numpy.inf)
    numpy.minimum.at(prof, idx, pen)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def profile_2d(x, y, pen, xedges, yedges):
    """min penalty в каждой 2D-ячейке (профиль по остальным параметрам)."""
    nbx, nby = len(xedges) - 1, len(yedges) - 1
    ix = numpy.clip(numpy.digitize(x, xedges) - 1, 0, nbx - 1)
    iy = numpy.clip(numpy.digitize(y, yedges) - 1, 0, nby - 1)
    flat = iy * nbx + ix
    grid = numpy.full(nbx * nby, numpy.inf)
    numpy.minimum.at(grid, flat, pen)
    return grid.reshape(nby, nbx)


def interval_from_profile(centers, dprof, level):
    """Границы интервала, где профиль Δpenalty <= level (по сетке бинов)."""
    ok = numpy.isfinite(dprof) & (dprof <= level)
    if not numpy.any(ok):
        return None, None
    return float(centers[ok].min()), float(centers[ok].max())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, 'Jcomputed_from_raw_theta0.5.txt')
    default_out = os.path.join(script_dir, 'profile_corner_theta0.5.pdf')

    ap = argparse.ArgumentParser(
        description="Профиль-правдоподобные 2D-карты + corner (READ-ONLY).")
    ap.add_argument('--combined-file', default=default_in,
                    help="combined-файл с posчитанным J (по умолчанию "
                         "py/Jcomputed_from_raw_theta0.5.txt).")
    ap.add_argument('--out', default=default_out,
                    help="Куда сохранить рисунок (PDF/PNG).")
    ap.add_argument('--params', default='incl,Q,gh,rh,rho0xUps,Ups,log10J',
                    help="Список осей через запятую (ключи COL/LABELS).")
    ap.add_argument('--nbins', type=int, default=40,
                    help="Число бинов для непрерывных осей.")
    ap.add_argument('--view-dpen', type=float, default=25.0,
                    help="Диапазоны осей строятся по точкам с Δpenalty<=этого "
                         "(отсекает плохие фиты для читаемости).")
    ap.add_argument('--vmax', type=float, default=11.83,
                    help="Верх цветовой шкалы Δpenalty на 2D-картах.")
    ap.add_argument('--incl', type=float, default=None,
                    help="Если задан — профиль по одному наклонению (incl "
                         "исключается из осей).")
    args = ap.parse_args()

    data = load_combined(args.combined_file)
    if args.incl is not None:
        sel = numpy.abs(data[:, COL['incl']] - args.incl) <= 0.01
        data = data[sel]
        if len(data) == 0:
            raise SystemExit(f"Нет точек для incl={args.incl}")

    pen = data[:, COL_PENALTY]
    pmin = float(pen.min())
    ibest = int(numpy.argmin(pen))
    dpen_all = pen - pmin

    params = [p.strip() for p in args.params.split(',') if p.strip()]
    if args.incl is not None and 'incl' in params:
        params.remove('incl')
    for p in params:
        if p not in COL:
            raise SystemExit(f"Неизвестная ось: {p} (есть: {list(COL)})")

    # точки для определения диапазонов осей (отсев плохих фитов)
    keep = dpen_all <= args.view_dpen

    # спецификации осей: биннинг penalty по РЕАЛЬНЫМ значениям, а отрисовка —
    # с равномерным (категориальным) масштабом для дискретных осей (incl),
    # чтобы область около 90° не сжималась.
    specs = {}
    for p in params:
        specs[p] = axis_spec(data[:, COL[p]], keep, args.nbins)

    N = len(params)
    fig, axes = plt.subplots(N, N, figsize=(2.3 * N, 2.3 * N))
    if N == 1:
        axes = numpy.array([[axes]])

    cmap = plt.get_cmap('viridis_r').copy()
    cmap.set_bad('white')
    norm = Normalize(vmin=0.0, vmax=args.vmax)
    mappable = None

    # печать интервалов
    print("=" * 78)
    print("ПРОФИЛЬ-ПРАВДОПОДОБИЕ (READ-ONLY)")
    if args.incl is not None:
        print(f"incl = {args.incl} (фиксировано)")
    print(f"Точек: {len(data)}   penalty_min = {pmin:.4f}")
    br = data[ibest]
    print(f"Best-fit: incl={br[COL['incl']]:.2f} Q={br[COL['Q']]:.4f} "
          f"gh={br[COL['gh']]:.4f} rh={br[COL['rh']]:.4f} "
          f"rho0={br[COL['rho0']]:.3f} Ups={br[COL['Ups']]:.4f} "
          f"rho0*Ups={br[COL['rho0xUps']]:.3f} log10J={br[COL['log10J']]:.4f}")
    print("-" * 78)
    print("1D профиль-интервалы (НОМИНАЛЬНЫЕ Δχ², НЕ калиброванные):")
    print(f"{'ось':>10}  {'best':>10}  {'Δ=1.0 (68%)':>26}  {'Δ=3.84 (95%)':>26}")

    # цветовая шкала для панели корреляций (верхний треугольник):
    # спокойные пастельные тона мята (r<0) — белый (r=0) — розовый (r>0)
    corr_cmap = LinearSegmentedColormap.from_list(
        'mint_pink', ['#7fcbb0', '#d6f0e6', '#f7f7f7', '#fbdfe9', '#f2a3c0'])
    corr_norm = Normalize(vmin=-1.0, vmax=1.0)
    kmask = keep if int(keep.sum()) >= 3 else numpy.ones(len(data), bool)

    def pearson(a, b):
        if len(a) < 3 or numpy.std(a) == 0 or numpy.std(b) == 0:
            return numpy.nan
        return float(numpy.corrcoef(a, b)[0, 1])

    for i, pi in enumerate(params):
        spec_i = specs[pi]
        for j, pj in enumerate(params):
            spec_j = specs[pj]
            ax = axes[i, j]

            if j > i:
                # верхний треугольник: коэффициент корреляции Пирсона
                r = pearson(data[kmask, COL[pj]], data[kmask, COL[pi]])
                if numpy.isfinite(r):
                    ax.set_facecolor(corr_cmap(corr_norm(r)))
                    ax.text(0.5, 0.5, f"{r:+.2f}", ha='center', va='center',
                            transform=ax.transAxes, color='#2b2b2b',
                            fontsize=9 + 13 * abs(r), fontweight='bold')
                else:
                    ax.text(0.5, 0.5, '—', ha='center', va='center',
                            transform=ax.transAxes, color='grey', fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor('#bbbbbb')
                continue

            if i == j:
                # 1D профиль
                x = data[:, COL[pi]]
                cen, prof = profile_1d(x, pen, spec_i['edges'])
                dprof = prof - pmin
                fin = numpy.isfinite(dprof)
                xpl = spec_i['centers_plot']
                ax.step(xpl[fin],
                        numpy.clip(dprof[fin], 0, max(LEVELS_1D) * 2.5),
                        where='mid', color='#1f3b73', lw=1.3)
                for lv in LEVELS_1D:
                    ax.axhline(lv, color='grey', ls=':', lw=0.8)
                ax.axvline(spec_i['to_plot'](br[COL[pi]]), color='crimson',
                           ls='-', lw=1.0, alpha=0.8)
                ax.set_ylim(0, max(LEVELS_1D) * 2.5)
                ax.set_xlim(spec_i['edges_plot'][0], spec_i['edges_plot'][-1])
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_ylabel(r'$\Delta$pen', fontsize=8)

                lo1, hi1 = interval_from_profile(cen, dprof, LEVELS_1D[0])
                lo2, hi2 = interval_from_profile(cen, dprof, LEVELS_1D[1])
                def _fmt(a, b):
                    if a is None:
                        return f"{'—':>26}"
                    return f"[{a:.4g}, {b:.4g}]".rjust(26)
                print(f"{pi:>10}  {br[COL[pi]]:>10.4g}  {_fmt(lo1, hi1)}  "
                      f"{_fmt(lo2, hi2)}")
            else:
                # 2D карта: x=params[j], y=params[i]
                xv, yv = data[:, COL[pj]], data[:, COL[pi]]
                grid = profile_2d(xv, yv, pen, spec_j['edges'], spec_i['edges'])
                dgrid = numpy.ma.masked_invalid(grid - pmin)
                mappable = ax.pcolormesh(spec_j['edges_plot'],
                                         spec_i['edges_plot'], dgrid,
                                         cmap=cmap, norm=norm, shading='flat')
                # контуры номинальных уровней
                xc = spec_j['centers_plot']
                yc = spec_i['centers_plot']
                if numpy.isfinite(grid).sum() > 4:
                    try:
                        ax.contour(xc, yc, dgrid, levels=LEVELS_2D,
                                   colors='white', linewidths=0.7,
                                   linestyles=['-', '--', ':'])
                    except Exception:
                        pass
                ax.plot(spec_j['to_plot'](br[COL[pj]]),
                        spec_i['to_plot'](br[COL[pi]]), 'x', color='crimson',
                        ms=7, mew=1.6)
                ax.set_xlim(spec_j['edges_plot'][0], spec_j['edges_plot'][-1])
                ax.set_ylim(spec_i['edges_plot'][0], spec_i['edges_plot'][-1])
                ax.tick_params(labelsize=7)

            # подписи и тики осей (для дискретных осей — реальные значения)
            if i == N - 1:
                ax.set_xlabel(LABELS.get(pj, pj), fontsize=9)
                if spec_j['ticks'] is not None:
                    ax.set_xticks(spec_j['ticks'][0])
                    ax.set_xticklabels(spec_j['ticks'][1], rotation=90,
                                       fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(LABELS.get(pi, pi), fontsize=9)
                if spec_i['ticks'] is not None:
                    ax.set_yticks(spec_i['ticks'][0])
                    ax.set_yticklabels(spec_i['ticks'][1], fontsize=6)
            elif j != 0:
                ax.set_yticklabels([])

    print("=" * 78)

    fig.suptitle(
        "Профиль-правдоподобие (Δpenalty); penalty = χ² GH-моментов, "
        "DOF не калиброван → уровни НОМИНАЛЬНЫЕ",
        fontsize=11, y=0.995)

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.90,
                        wspace=0.08, hspace=0.08)

    # общий colorbar
    if mappable is not None:
        cax = fig.add_axes([0.55, 0.94, 0.4, 0.012])
        cb = fig.colorbar(mappable, cax=cax, orientation='horizontal')
        cb.set_label(r'$\Delta$penalty (2D: уровни 2.30 / 6.18 / 11.83)',
                     fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Рисунок сохранён: {args.out}")
    print("ВНИМАНИЕ: точки получены оптимизатором (BoTorch), это профиль-")
    print("правдоподобие, НЕ постериор; уровни Δχ² номинальны (penalty не")
    print("калиброван как χ², см. вопрос PI #16).")
    print(f"Верхний треугольник: корреляция Пирсона по {int(kmask.sum())} точкам "
          "(Δpen<=view-dpen) — ОПИСАТЕЛЬНАЯ для выборки оптимизатора, НЕ постериор.")
    print("Ось incl нарисована в равномерном (категориальном) масштабе узлов сетки.")


if __name__ == '__main__':
    main()
