# #
## nice -n 19 ionice -c 3 python Fornax_P21_symm_PCA_w3Sersic.py --no-resume 
## nice -n 19 ionice -c 3 python Fornax_P21_symm_PCA_w3Sersic.py --no-resume --delete-checkpoint --incl=71.85
# --incl=71.85
# # Обычный запуск — с восстановлением из checkpoint (если есть):
# python Fornax_P21_symm_PCA_w3Sersic.py
#
# Запуск с нуля — checkpoint игнорируется, но файл остаётся:
# python Fornax_P21_symm_PCA_w3Sersic.py --no-resume
#
# Запуск с нуля — checkpoint удаляется физически:
# python Fornax_P21_symm_PCA_w3Sersic.py --no-resume --delete-checkpoint
#
import datetime
import requests
import subprocess
import os
import argparse
import numpy
import torch
torch.set_num_threads(1) # Ограничение intra-op потоков PyTorch
torch.set_num_interop_threads(1) # Ограничение inter-op потоков PyTorch
#import re
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import socket
import glob

# BoTorch / GPyTorch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input  import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

# Получаем имя хоста для суффикса файлов
# hostname_proc = socket.gethostname()
# В самом начале скрипта, после импортов и определения hostname_proc:
parser = argparse.ArgumentParser(description='Schwarzschild orbit modelling')
parser.add_argument('--no-resume', action='store_true',
                    help='Запустить расчёт с нуля, игнорируя checkpoint')
parser.add_argument('--delete-checkpoint', action='store_true',
                    help='Удалить checkpoint перед запуском')
parser.add_argument('--incl', type=float, default=None,
                    help='Угол наклонения галактики в градусах (например: --incl 90.0)')
# В блоке argparse добавить аргумент --suffix:
parser.add_argument('--suffix', type=str, default=None,
                    help='Суффикс для файлов результатов '
                         '(например: p0, p1). '
                         'По умолчанию используется hostname_p0.')
args = parser.parse_args()
# Формируем идентификатор процесса:
# hostname_proc берётся из переменной окружения (передаётся из контейнера)
# suffix переопределяет hostname_proc для параллельных процессов
_hostname_env = os.environ.get('HOSTNAME_SUFFIX', socket.gethostname())
if args.suffix is not None:
    hostname_proc = f"{_hostname_env}_{args.suffix}"
else:
    hostname_proc = f"{_hostname_env}_p0"

print(f"Идентификатор процесса: {hostname_proc}")

#files = ['4UpsBoTorch_Sersic.txt', '4UpsBoTorch_PCA_Sersic_gray.txt','4UpsBoTorch_PCA_Sersic_tycho.txt',
#         '4UpsBoTorch_Sersic_iota.txt']
# ###
host_patterns = [
    f"4UpsBoTorch_PCA_Sersic_{_hostname_env}.txt",
    f"4UpsBoTorch_PCA_Sersic_{_hostname_env}_p*.txt",
]

storage_patterns= [
    "4UpsBoTorch_Sersic.txt",
    "4UpsBoTorch_PCA_Sersic_*.txt"
]

fallback_patterns = [
    "4UpsBoTorch_PCA_PA46.8_Sersic*.txt",
]

# Добавлен суффикс hostname_proc к файлам
UpsFile = f"4UpsBoTorch_PCA_Sersic_{hostname_proc}.txt"
torchFile_result = f"4result_BoTorch_PCA_Sersic_{hostname_proc}.txt"

cutoff_start=2.0

do_resume = not args.no_resume

#TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')   # от @BotFather
#TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')    # ваш личный chat_id

# --- Настройка уведомлений ---
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'GalaxySchwarzschildFornax')  # ваш уникальный топик
NTFY_SERVER = os.environ.get('NTFY_SERVER', 'https://ntfy.sh')       # или свой сервер
RCLONE_REMOTE = os.environ.get('RCLONE_REMOTE', 'yandex')

import agama

def send_ntfy(message, title='Galaxy Calc', priority='default', tags=None):
    """
    Отправка push-уведомления через ntfy.sh.
    priority: min, low, default, high, urgent
    tags: список эмодзи-тегов, например ['rocket'], ['warning']
    Документация: https://docs.ntfy.sh
    """
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    headers = {
        'Title': title.encode('utf-8'),
        'Priority': priority,
        'Content-Type': 'text/plain; charset=utf-8',
    }
    if tags:
        headers['Tags'] = ','.join(tags)
    
    try:
        requests.post(
            url,
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )
    except Exception as e:
        print(f"ntfy недоступен: {e}")

def send_notification(message, title='Galaxy Calc',
                      priority='default', tags=None, silent=False):
    """
    Универсальная функция уведомлений.
    Пробует ntfy, при ошибке — пишет только в лог.
    """
    # ntfy не имеет режима silent, но можно снизить priority
    if silent:
        priority = 'min'
    
    send_ntfy(message, title=title, priority=priority, tags=tags)


# --- Угол наклонения: из командной строки или по умолчанию ---
_incl_default = 90.0
if args.incl is not None:
    # Проверка допустимого диапазона
    if not (0.0 < args.incl <= 90.0):
        raise ValueError(
            f"Недопустимое значение incl={args.incl}. "
            f"Допустимый диапазон: (0, 90] градусов."
        )
    incl = args.incl
    print(f"Угол наклонения задан из командной строки: incl={incl}")
else:
    incl = _incl_default
    print(f"Угол наклонения по умолчанию: incl={incl}")

# https://ui.adsabs.harvard.edu/abs/2022MNRAS.515L...1W/abstract
D_O22 = 143 # ± 3 kpc
D = D_O22

#incl      = 90.0
beta      = incl * numpy.pi/180
alpha     = 0.0 
posang    = 42.3 # 4Sersic, Wang et al 2019  https://doi.org/10.3847/1538-4357/ab31a9 # 46.8 old Battaglia, G., Tolstoy, E., Helmi, A., et al. 2006, A&A, 459, 423
gamma2 =  (posang - 90.0) * numpy.pi/180
q_ap      = 1 - 0.31  #  0.7 old !!!

sinbeta = numpy.sin(beta)
cosbeta = numpy.cos(beta)
singamma=numpy.sin(gamma2)
cosgamma=numpy.cos(gamma2)
q_ap2   = q_ap**2
axRZst  = (q_ap2 - cosbeta**2)**0.5/sinbeta

#P21 = numpy.loadtxt('/home/gala/Agama/py/table3.dat')
P21 = numpy.loadtxt(
    os.environ.get('AGAMA_TABLE3', 
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'table3.dat'))
)

P21_ra = 39.9583
P21_de = -34.4997
P21_vl = 54.7

vscale = (2*6.67/3.086)**0.5
P21_vl = P21_vl/vscale
P21[:,5] = P21[:,5]/vscale
P21[:,6] = P21[:,6]/vscale
P21=P21[numpy.where(P21[:,9]>0)[0],:]

sc=numpy.pi*D/180
xy_a2vleP_P21  = numpy.zeros(2*len(P21), 
    dtype=[('x','float64'),('y_','float64'),('a2','float64'),('vl','float64'),('err_v','float64'),('prob','float64')])
for i in range(len(P21)):
    X = sc*(P21[i,1]-P21_ra)*numpy.cos(P21_de*numpy.pi/180)
    Y = sc*(P21[i,2]-P21_de)
    xy_a2vleP_P21[2*i]['x'] = cosgamma*X - singamma*Y
    xy_a2vleP_P21[2*i]['y_'] = singamma*X + cosgamma*Y
    xy_a2vleP_P21[2*i]['a2'] = xy_a2vleP_P21[2*i]['x']**2 + xy_a2vleP_P21[2*i]['y_']**2/q_ap2
    xy_a2vleP_P21[2*i]['vl'] = P21[i,5]-P21_vl
    xy_a2vleP_P21[2*i]['err_v'] = P21[i,6] 
    xy_a2vleP_P21[2*i]['prob'] = P21[i,9]
    
    xy_a2vleP_P21[2*i+1]['x']  = - xy_a2vleP_P21[2*i]['x']
    xy_a2vleP_P21[2*i+1]['y_'] = - xy_a2vleP_P21[2*i]['y_'] 
    xy_a2vleP_P21[2*i+1]['a2'] = xy_a2vleP_P21[2*i]['a2'] 
    xy_a2vleP_P21[2*i+1]['vl'] = - xy_a2vleP_P21[2*i]['vl']
    xy_a2vleP_P21[2*i+1]['err_v'] = xy_a2vleP_P21[2*i]['err_v'] 
    xy_a2vleP_P21[2*i+1]['prob']  = xy_a2vleP_P21[2*i]['prob']

xy_a2vleP_P21.sort(order='a2')

max_r = 2.1
rm2 = max_r**2
xy_a2vleP_P21 = xy_a2vleP_P21[numpy.where(xy_a2vleP_P21['a2'][:]<rm2)]

xv_P21  = numpy.zeros((len(xy_a2vleP_P21), 6), dtype='float64')
err_P21 = numpy.zeros(len(xy_a2vleP_P21), dtype='float64')
prob_P21= numpy.zeros(len(xy_a2vleP_P21), dtype='float64')
for i in range(len(xy_a2vleP_P21)):
    xv_P21[i,0] = xy_a2vleP_P21[i]['x']
    xv_P21[i,1] = xy_a2vleP_P21[i]['y_'] * cosbeta
    xv_P21[i,2] = xy_a2vleP_P21[i]['y_'] * sinbeta
    xv_P21[i,3] = 0.0
    xv_P21[i,4] = - xy_a2vleP_P21[i]['vl']*sinbeta 
    xv_P21[i,5] =   xy_a2vleP_P21[i]['vl']*cosbeta 
    err_P21[i]  = xy_a2vleP_P21[i]['err_v']
    prob_P21[i] = xy_a2vleP_P21[i]['prob']

n_bin = 250
tg1 = q_ap*numpy.tan(numpy.pi/8)
ct2 = q_ap/numpy.tan(numpy.pi/8)
bound_circR      = [[],[],[],[],[],[],[],[],[]]
sectors_P21RvleP = [[],[],[],[],[],[],[],[],[]]
add_vleP         = [[],[],[],[],[],[],[],[],[]]
sect_i   = [0,0,0,0,0,0,0,0,0]
app_num_j= [0,0,0,0,0,0,0,0,0]
max_a2   = [0,0,0,0,0,0,0,0,0]
sector = 0
for j in range(len(xy_a2vleP_P21)):
    if(app_num_j[0]>n_bin-1) :
        if(xy_a2vleP_P21['a2'][j] > rm2) :
            break
        if(xy_a2vleP_P21['x'][j] >= 0) :
            if(numpy.abs(xy_a2vleP_P21['y_'][j]) <= tg1*xy_a2vleP_P21['x'][j]) :
                sector = 1
            else:
                if(numpy.abs(xy_a2vleP_P21['y_'][j]) < ct2*xy_a2vleP_P21['x'][j]) :
                    if(xy_a2vleP_P21['y_'][j]>0):
                       sector = 2
                    else:
                       sector = 8
                else:
                    if(xy_a2vleP_P21['y_'][j]>0):
                       sector = 3
                    else:
                       sector = 7
        else:
            if(numpy.abs(xy_a2vleP_P21['y_'][j]) <= tg1*numpy.abs(xy_a2vleP_P21['x'][j])) :
                sector = 5
            else :
                if(numpy.abs(xy_a2vleP_P21['y_'][j]) < ct2*numpy.abs(xy_a2vleP_P21['x'][j])) :
                    if(xy_a2vleP_P21['y_'][j]>0):
                        sector = 4
                    else:
                        sector = 6
                else:
                    if(xy_a2vleP_P21['y_'][j]>0):
                        sector = 3
                    else:
                        sector = 7

        max_a2[sector] = xy_a2vleP_P21['a2'][j]
        if(app_num_j[sector]>n_bin-1) :
            sectors_P21RvleP[sector].extend(numpy.array([add_vleP[sector]]))
            add_vleP[sector] = []
            add_vleP[sector].append(xy_a2vleP_P21[['vl', 'err_v', 'prob']][j])
            sect_i[sector] += 1
            app_num_j[sector] = xy_a2vleP_P21['prob'][j]
            bound_circR[sector].append((xy_a2vleP_P21['a2'][j-1]**0.5 + xy_a2vleP_P21['a2'][j]**0.5)/2)
        else:
            add_vleP[sector].append(xy_a2vleP_P21[['vl', 'err_v', 'prob']][j])
            app_num_j[sector] += xy_a2vleP_P21['prob'][j]
    else:
        add_vleP[0].append(xy_a2vleP_P21[['vl', 'err_v', 'prob']][j])
        app_num_j[0] += xy_a2vleP_P21['prob'][j]
        if(app_num_j[0]>n_bin-1) :
            sectors_P21RvleP[0].extend(numpy.array([add_vleP[0]]))
            for s in range(len(bound_circR)):
                bound_circR[s].append((xy_a2vleP_P21['a2'][j]**0.5 + xy_a2vleP_P21['a2'][j+1]**0.5)/2)

for s in range(1,len(sectors_P21RvleP)):
    sectors_P21RvleP[s].extend(numpy.array([add_vleP[s]]))
    bound_circR[s].append(max_a2[s]**0.5 + 0.01)

circ_points = [[],[],[],[],[],[],[],[],[]]
phit=numpy.linspace(0, 2*numpy.pi, 121)
circ_points[0] = numpy.column_stack(( numpy.cos(phit) , numpy.sin(phit)*q_ap ))
for i in (range(1,len(sectors_P21RvleP))) :
    phit = numpy.linspace((i-1.5)*numpy.pi/4, (i-0.5)*numpy.pi/4, 31)
    circ_points[i] = numpy.column_stack(( numpy.cos(phit) , numpy.sin(phit)*q_ap ))

sectAPP = []
addAPP = []
addAPP.append(numpy.vstack((circ_points[0]*bound_circR[0][0])) )
sectAPP.extend(addAPP)

for i in range(1,len(bound_circR)) : 
    for k in range(len(bound_circR[i])-1) : 
        addAPP = []
        addAPP.append(numpy.vstack((circ_points[i]*bound_circR[i][k+1],
                                list(reversed(circ_points[i]*bound_circR[i][k]) )
                                )) )
        sectAPP.extend(addAPP)


def find_nearest_incl_data_fallback(fallback_patterns,
                                     target_incl,
                                     min_points=5,
                                     timeout=300):
    """
    Fallback-поиск начальных точек в файлах другого posang
    (например 4UpsBoTorch_PCA_PA46.8_Sersic*).
    
    Penalty из этих файлов НЕ используется для PCA —
    только для выбора кандидатов на пересчёт.
    
    Возвращает:
        candidates : list of dict {'Q','gh','rh','rho0'}
                     отсортированных по penalty (для приоритизации)
        source_info: str — описание источника
    """
    print("\n  [fallback] Поиск в fallback-файлах:")
    for p in fallback_patterns:
        print(f"    {p}")

    # --- Скачиваем fallback-файлы с Яндекс.Диска ---
    try:
        result = subprocess.run(
            ['rclone', 'lsf', f"{RCLONE_REMOTE}:galAgama/"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            remote_files = [f.strip() for f in result.stdout.splitlines()]
            for remote_fname in remote_files:
                matches = any(
                    glob.fnmatch.fnmatch(remote_fname, os.path.basename(p))
                    for p in fallback_patterns
                )
                if not matches:
                    continue
                local_path = remote_fname
                remote_path = f"{RCLONE_REMOTE}:galAgama/{remote_fname}"
                cmd = ['rclone', 'copyto', remote_path, local_path,
                       '--stats-one-line', '--ignore-times']
                try:
                    subprocess.run(cmd, capture_output=True,
                                   text=True, timeout=timeout)
                    if os.path.exists(local_path):
                        size = os.path.getsize(local_path)
                        print(f"  [fallback] ↓ {remote_fname} "
                              f"({size/1024:.1f} KB)")
                except Exception as e:
                    print(f"  [fallback] ✗ {remote_fname}: {e}")
    except Exception as e:
        print(f"  [fallback] Ошибка доступа к яндексу: {e}")

    # --- Читаем все fallback-файлы ---
    all_files = []
    for pattern in fallback_patterns:
        for f in glob.glob(pattern):
            if f not in all_files:
                all_files.append(f)

    if not all_files:
        print("  [fallback] Файлы не найдены")
        return [], "нет fallback-файлов"

    print(f"  [fallback] Найдено файлов: {len(all_files)}")
    for f in all_files:
        print(f"    {f}")

    # --- Читаем строки ---
    all_rows = []
    file_counts = {}
    for filepath in all_files:
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
                        # incl, Q, gh, rh, rho0, Upsilon, penalty
                        if row[3] <= 0 or row[4] <= 0:
                            continue
                        if row[6] >= 1e5:
                            continue
                        all_rows.append(row)
                        count += 1
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        file_counts[filepath] = count
        print(f"    {filepath}: {count} строк")

    if not all_rows:
        print("  [fallback] Файлы пусты")
        return [], "fallback-файлы пусты"

    data_all = numpy.array(all_rows)
    print(f"  [fallback] Итого строк: {len(data_all)}")

    # --- Ищем ближайшее наклонение ---
    unique_incls = numpy.unique(numpy.round(data_all[:, 0], 2))
    print(f"  [fallback] Наклонения в файлах: {unique_incls}")

    dists = numpy.abs(unique_incls - target_incl)
    order = numpy.argsort(dists)

    selected_data = None
    nearest_incl  = None
    dist_val      = float('inf')

    for idx in order:
        candidate_incl = unique_incls[idx]
        mask           = numpy.abs(data_all[:, 0] - candidate_incl) < 0.01
        candidate_data = data_all[mask]
        if len(candidate_data) >= min_points:
            selected_data = candidate_data
            nearest_incl  = float(candidate_incl)
            dist_val      = float(dists[idx])
            break

    if selected_data is None and len(unique_incls) > 0:
        # Берём ближайшее даже если мало точек
        best_idx      = order[0]
        nearest_incl  = float(unique_incls[best_idx])
        dist_val      = float(dists[best_idx])
        mask          = numpy.abs(data_all[:, 0] - nearest_incl) < 0.01
        selected_data = data_all[mask]

    if selected_data is None or len(selected_data) == 0:
        print("  [fallback] Нет подходящих данных")
        return [], "нет данных в fallback-файлах"

    print(f"  [fallback] Выбрано incl={nearest_incl:.2f}° "
          f"(dist={dist_val:.2f}°, {len(selected_data)} точек)")

    # --- Сортируем по penalty и формируем кандидатов ---
    # ВАЖНО: penalty из fallback-файлов используется ТОЛЬКО
    # для сортировки/приоритизации, не для PCA!
    sorted_data = selected_data[numpy.argsort(selected_data[:, 6])]

    candidates = []
    for row in sorted_data:
        candidates.append({
            'Q':    float(row[1]),
            'gh':   float(row[2]),
            'rh':   float(row[3]),
            'rho0': float(row[4]),
            # penalty сохраняем только для информации
            '_penalty_ref': float(row[6]),
            '_incl_ref':    float(row[0]),
        })

    source_info = (f"fallback: {[os.path.basename(f) for f in all_files]}, "
                   f"incl={nearest_incl:.2f}° (dist={dist_val:.1f}°)")

    print(f"  [fallback] Кандидатов: {len(candidates)}")
    print(f"  [fallback] Источник: {source_info}")

    return candidates, source_info

def bootstrap_from_fallback(fallback_patterns,
                             target_incl,
                             bounds_original,
                             n_bootstrap=10,
                             penalty_cutoff_frac=0.5,
                             strategy='best_diverse',
                             output_file=None):
    """
    Bootstrap начальных точек из fallback-файлов (другой posang).
    
    Penalty из fallback-файлов используется ТОЛЬКО для выбора кандидатов.
    Для каждого кандидата penalty ПЕРЕСЧИТЫВАЕТСЯ для текущего incl/posang.
    
    Возвращает:
        bootstrap_results : list of dict
            [{'params': {...}, 'penalty': float}, ...]
    """
    if output_file is None:
        output_file = torchFile_result

    def _write(text):
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')

    _write("\n" + "=" * 60)
    _write("FALLBACK BOOTSTRAP из файлов другого posang")
    _write(f"target incl={target_incl:.2f}°")
    _write("=" * 60)
    _write("ВНИМАНИЕ: penalty из fallback-файлов используется")
    _write("          ТОЛЬКО для выбора кандидатов!")
    _write("          Для PCA penalty будет ПЕРЕСЧИТАН.")
    _write("=" * 60)

    # --- Получаем кандидатов ---
    candidates_all, source_info = find_nearest_incl_data_fallback(
        fallback_patterns = fallback_patterns,
        target_incl       = target_incl,
    )

    if not candidates_all:
        _write("  [fallback bootstrap] Нет кандидатов")
        return []

    _write(f"\n  Источник: {source_info}")
    _write(f"  Всего кандидатов: {len(candidates_all)}")

    # --- Выбираем n_bootstrap кандидатов ---
    # Стратегия: половина лучших по ref-penalty + половина равномерных
    n_pool = max(n_bootstrap,
                 int(len(candidates_all) * penalty_cutoff_frac))
    n_pool = min(n_pool, len(candidates_all))
    pool   = candidates_all[:n_pool]   # уже отсортированы по penalty

    if strategy == 'best_diverse':
        n_best = max(1, n_bootstrap // 2)
        n_div  = n_bootstrap - n_best
        best_part = pool[:n_best]
        if n_div > 0 and len(pool) > n_best:
            rest    = pool[n_best:]
            indices = numpy.linspace(0, len(rest) - 1,
                                     n_div, dtype=int)
            div_part = [rest[i] for i in indices]
            selected = best_part + div_part
        else:
            selected = best_part
    elif strategy == 'best':
        selected = pool[:n_bootstrap]
    else:
        indices  = numpy.linspace(0, len(pool) - 1,
                                  n_bootstrap, dtype=int)
        selected = [pool[i] for i in indices]

    _write(f"\n  Выбрано кандидатов: {len(selected)}")
    _write(f"  {'#':>3s}  {'Q':>8s}  {'gh':>8s}  "
           f"{'rh':>8s}  {'rho0':>8s}  {'ref_pen':>10s}  {'ref_incl':>8s}")
    _write(f"  {'-'*65}")
    for i, c in enumerate(selected):
        _write(f"  {i+1:3d}  {c['Q']:8.4f}  {c['gh']:8.4f}  "
               f"{c['rh']:8.4f}  {c['rho0']:8.4f}  "
               f"{c.get('_penalty_ref', 0):10.4f}  "
               f"{c.get('_incl_ref', 0):8.2f}")

    # --- Пересчёт penalty для текущего incl/posang ---
    _write(f"\n  Пересчёт penalty для incl={target_incl:.2f}°...")
    _write("  (используются текущие datasets и densityStars)")

    bootstrap_results = []
    n_success = 0
    n_failed  = 0

    for i, cand in enumerate(selected):
        params = {
            'Q':    cand['Q'],
            'gh':   cand['gh'],
            'rh':   cand['rh'],
            'rho0': cand['rho0'],
        }
        _write(f"\n  [{i+1}/{len(selected)}] "
               f"Q={params['Q']:.4f}, gh={params['gh']:.4f}, "
               f"rh={params['rh']:.4f}, rho0={params['rho0']:.4f} "
               f"(ref_pen={cand.get('_penalty_ref',0):.4f}, "
               f"ref_incl={cand.get('_incl_ref',0):.2f}°)")

        # Проверка границ
        in_bounds = all(
            bounds_original[name][0] <= params[name] <= bounds_original[name][1]
            for name in ['Q', 'gh', 'rh', 'rho0']
        )
        if not in_bounds:
            _write("    ✗ Параметры вне границ, пропускаем")
            n_failed += 1
            continue

        # Пересчёт penalty (direct_params — без PCA)
        try:
            y_val = halo_IC_lib_weights_pca_fixed(
                pc_coords     = numpy.array([params['Q'], params['gh'],
                                             params['rh'], params['rho0']]),
                model_data    = None,
                bounds_original = bounds_original,
                densityStars  = densityStars,
                datasets      = datasets,
                alphah        = alphah,
                betah         = betah,
                direct_params = params,   # ← прямая передача без PCA
            )
            penalty = -y_val

            if numpy.isfinite(penalty) and penalty < 1e5:
                bootstrap_results.append({
                    'params':       params,
                    'penalty':      penalty,
                    'penalty_ref':  cand.get('_penalty_ref', None),
                    'incl_ref':     cand.get('_incl_ref',    None),
                })
                n_success += 1
                ref_pen = cand.get('_penalty_ref', float('nan'))
                _write(f"    ✓ penalty={penalty:.6f} "
                       f"(ref={ref_pen:.4f}, "
                       f"delta={penalty-ref_pen:+.4f})")
            else:
                _write(f"    ✗ penalty={penalty:.6f} (невалидное)")
                n_failed += 1

        except Exception as e:
            _write(f"    ✗ Ошибка: {e}")
            n_failed += 1

    _write(f"\n  Fallback bootstrap завершён: "
           f"успешно={n_success}, ошибок={n_failed}")

    if bootstrap_results:
        best_pen = min(r['penalty'] for r in bootstrap_results)
        _write(f"  Лучший penalty: {best_pen:.6f}")

        send_notification(
            f"Fallback bootstrap завершён\n"
            f"incl={target_incl:.2f}°\n"
            f"Источник: {source_info}\n"
            f"Успешно: {n_success}/{len(selected)}\n"
            f"Лучший penalty: {best_pen:.6f}",
            title=f"Galaxy {hostname_proc}: Fallback Bootstrap",
            priority='default',
            tags=['white_check_mark']
        )

    return bootstrap_results

# ==============================================================
#  КЛАСС WeightedScaler
# ==============================================================
class WeightedScaler:
    def __init__(self, mean, std):
        self.mean_ = mean
        self.scale_ = std
    
    def transform(self, X):
        return (X - self.mean_) / self.scale_
    
    def inverse_transform(self, X):
        return self.mean_ + self.scale_ * X
    
    def fit_transform(self, X):
        return self.transform(X)

def pca_to_params_fixed(pc_coords, model_data, bounds_original):
    scaler = model_data['scaler']
    pca = model_data['pca']
    use_log_scale = model_data['use_log_scale']
    
    pc_coords = numpy.atleast_2d(pc_coords)
    n_input = pc_coords.shape[1]
    n_pca_components = pca.n_components_
    
    if n_input < n_pca_components:
        pc_full = numpy.zeros((pc_coords.shape[0], n_pca_components))
        pc_full[:, :n_input] = pc_coords
    else:
        pc_full = pc_coords[:, :n_pca_components]
    
    X_scaled = pca.inverse_transform(pc_full)
    X_transformed = scaler.inverse_transform(X_scaled)
    
    if use_log_scale:
        X_original = X_transformed.copy()
        X_original[:, 2] = 10**X_transformed[:, 2]
        X_original[:, 3] = 10**X_transformed[:, 3]
    else:
        X_original = X_transformed
    
    param_names = ['Q', 'gh', 'rh', 'rho0']
    result = {}
    
    for i, name in enumerate(param_names):
        val = X_original[0, i]
        lo, hi = bounds_original[name]
        val = numpy.clip(val, lo, hi)
        result[name] = float(val)
    
    return result

def params_to_pca_fixed(params_dict, model_data):
    scaler = model_data['scaler']
    pca = model_data['pca']
    use_log_scale = model_data['use_log_scale']
    
    param_names = ['Q', 'gh', 'rh', 'rho0']
    X = numpy.array([[params_dict[name] for name in param_names]])
    
    if use_log_scale:
        X_transformed = X.copy()
        X_transformed[0, 2] = numpy.log10(X[0, 2])
        X_transformed[0, 3] = numpy.log10(X[0, 3])
    else:
        X_transformed = X
    
    X_scaled = scaler.transform(X_transformed)
    pc_coords = pca.transform(X_scaled)
    return pc_coords[0, :pca.n_components_]

massSt    = 14.0
scaleRst  =  sc*16.4/60


Upsilon_start = [1.0]
Upsilon_lower = 0.1
Upsilon_upper = 1.6

# ============================================================================
#  ОПТИМИЗАЦИЯ Brent-поиска по Upsilon (включена постоянно).
#  п.1 — адаптивная узкая скобка вокруг предсказанного Upsilon* (медиана недавних
#        найденных значений) + откат к полному диапазону, если минимум на краю.
#  п.2 — толеранс Brent UPS_XATOL=5e-3 (был 1e-3; penalty у минимума плоский,
#        ошибка записываемого penalty ~0.015 < 2% pen_sigma KDE).
#  п.3 — субдискретизация орбит во ВНУТРЕННЕМ поиске + один продуктивный full-solve
#        на полной библиотеке в найденной точке: он даёт контрактно-точный penalty
#        и НЕ попадает в историю оптимизации Upsilon (вызывается в обход logger).
#        RNG подвыборки локальный: не трогает ни глобальный numpy seed, ни AGAMA
#        orbit RNG (контракт §7).
#  Обоснование на данных: doc/ai/questions_for_pi.md (Q15).
# ----------------------------------------------------------------------------
UPS_XATOL          = 5e-3   # толеранс Brent по Upsilon (M/L)
UPS_BRACKET_DELTA  = 0.1    # полуширина адаптивной скобки вокруг Upsilon*
UPS_BRACKET_NMED   = 8      # сколько последних Upsilon* усреднять (медиана)
UPS_SUBSAMPLE_FRAC = 0.25   # доля орбит для внутреннего поиска (0 < frac < 1)

#  Скользящая история найденных Upsilon* (для предсказания скобки, п.1).
_ups_recent = []

Sersic_m  = 0.80

NumStars  = 1000000
intTime   = 100.
regul     = 1.0
ghorder   = 6
degree    = 2
symmetry  = 't'
usehist   = 0
variant   = 'Hist' if usehist else 'GH'
numpy.random.seed(42)
numpy.set_printoptions(precision=8, linewidth=200, suppress=True)

densityParams = dict(
    type  = 'DensitySphHarm',
    gridr = numpy.linspace(0.0, 2.0, 21),
    lmax  = 4
)

gridv = numpy.linspace(-25, 25, 51)
velpsf = 0.0
hist_degree = 0
hist_gridv  = numpy.linspace(-50, 50, 50)

psf2   = 0.01
kinemParams2 = dict(
    type     = 'LOSVD',
    symmetry = symmetry,
    alpha    = alpha,
    beta     = beta,
    gamma    = gamma2,
    psf      = psf2,
    velpsf   = velpsf,
    degree   = degree,
    gridv    = gridv
)

n_grids = 51
n_grids_x_per_bin = n_grids / len(bound_circR[1])
gridx_min = bound_circR[0][0] / n_grids_x_per_bin
gridx_max = bound_circR[1][-1] + gridx_min
gridx = agama.nonuniformGrid(nnodes=n_grids+1,xmin=gridx_min,xmax=gridx_max)
gridx = numpy.hstack( (list(reversed(-gridx)),gridx[1:]) )
print(gridx)

n_grids_y_per_bin = n_grids /  len(bound_circR[3])
gridy_min = bound_circR[0][0] * q_ap / n_grids_y_per_bin
gridy_max = bound_circR[3][-1]* q_ap + gridy_min
gridy = agama.nonuniformGrid(nnodes=n_grids+1,xmin=gridy_min,xmax=gridy_max)
gridy = numpy.hstack( (list(reversed(-gridy)),gridy[1:]) )
print(gridy)
   
target       = agama.Target(apertures=sectAPP, gridx=gridx, gridy=gridy, **kinemParams2)
datacube_P21 = target((xv_P21,prob_P21)).reshape(len(sectAPP), -1)
ghm_moments_P21 = agama.ghMoments(degree=degree, gridv=gridv, matrix=datacube_P21, ghorder=ghorder)

n_boot = 100
bootxv_P21  = numpy.vstack([xv_P21] * n_boot)
diffbootVZ  = numpy.hstack([err_P21] * n_boot) * numpy.random.normal(size=len(bootxv_P21))
bootxv_P21[:,4] += -diffbootVZ*sinbeta
bootxv_P21[:,5] += diffbootVZ*cosbeta
bootcube_P21 = numpy.zeros((len(sectAPP)*n_boot, datacube_P21.shape[1]), dtype='float64')
for i in range(n_boot):
    add_cube = target((bootxv_P21[i*len(xv_P21):(i+1)*len(xv_P21),:],prob_P21)).reshape(len(sectAPP), -1)
    bootcube_P21[i*len(sectAPP):(i+1)*len(sectAPP),:] = add_cube
    
cube_errors = numpy.std(bootcube_P21.reshape(n_boot, -1), axis=0).reshape(datacube_P21.shape)
ghm_val_P21, ghm_err_P21 = agama.schwarzlib.ghMomentsErrors(degree=degree, gridv=gridv, values=datacube_P21, errors=cube_errors, ghorder=ghorder)
ind = (1,2,6,7,8,9)

print(ghm_val_P21)

datasets = []
densityStars = agama.Density(type='Sersic',sersicIndex=Sersic_m,
                             mass=massSt, scaleRadius=scaleRst, axisRatioZ=axRZst)
datasets.append(agama.schwarzlib.DensityDataset(
    density=densityStars,
    tolerance=0.0,
    **densityParams
) )

datasets.append(agama.schwarzlib.KinemDatasetGH(
    density   = densityStars,
    tolerance = 0.01,
    ghm_val   = ghm_val_P21[:,ind],
    ghm_err   = ghm_err_P21[:,ind],
    apertures = sectAPP,
    gridx=gridx, 
    gridy=gridy,
    **kinemParams2
) )

alphah    = 2.0
betah     = 3

numOrbits = 100000
trajsize = 1000

bounds_original = {
    'Q': (0.05, 2.5),
    'gh': (0.0, 1.6),
    'rh': (0.5, 3.5),
    'rho0': (34.0, 120.0),
    'Upsilon': (0.1, 1.6)
}

class FunctionLogger:
    def __init__(self, target_func):
        self.target_func = target_func
        self.history = []
    
    def __call__(self, x):
        result = self.target_func(x)
        x_save = numpy.atleast_1d(numpy.asarray(x, dtype=float)).copy()
        self.history.append((x_save, float(result)))
        return result
    
    def clear_history(self):
        self.history.clear()
    
    def save_history(self, filename):
        numpy.save(filename, numpy.array(self.history, dtype=object))

def sync_to_yadisk(local_dir='.', remote_dir='galAgama',
                   timeout=300):
    """
    Синхронизация результатов на Яндекс.Диск через rclone.
    Возвращает True при успехе.
    """
    # Список файлов для синхронизации
    files_to_sync = [
        UpsFile,
        torchFile_result,
        f"pca_model_weighted_{hostname_proc}.pkl",
        f"checkpoint_{hostname_proc}.pkl",
        f"diagnose_pca_space_{hostname_proc}.txt"
    ]
    
    success = True
    for filepath in files_to_sync:
        if not os.path.exists(filepath):
            print(f"  Пропуск (не найден): {filepath}")
            continue
        
        remote_path = f"{RCLONE_REMOTE}:{remote_dir}/{filepath}"
        cmd = ['rclone', 'copyto', filepath, remote_path,
               '--progress', '--stats-one-line']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                size = os.path.getsize(filepath)
                print(f"  ✓ {filepath} → Яндекс.Диск ({size/1024:.1f} KB)")
            else:
                print(f"  ✗ Ошибка {filepath}: {result.stderr[:100]}")
                success = False
        except subprocess.TimeoutExpired:
            print(f"  ✗ Таймаут при загрузке {filepath}")
            success = False
        except FileNotFoundError:
            print("  ✗ rclone не установлен!")
            return False
    
    return success

def load_from_yadisk(storage_patterns, host_patterns,
                     local_dir='.', remote_dir='galAgama',
                     timeout=300, 
                     force_update=False):
    """
    Скачивает файлы с Яндекс.Диска по паттернам.
    storage_patterns: паттерны для поиска в хранилище
    host_patterns:    паттерны файлов своего сервера (не скачиваем — они локальные)
    force_update:     True — принудительно перезаписывать локальные файлы
                      (используется при старте скрипта для получения свежих данных)
    """
    # Получаем список файлов в удалённой папке
    try:
        result = subprocess.run(
            ['rclone', 'lsf', f"{RCLONE_REMOTE}:{remote_dir}/"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            msg = f"  [yadisk] Ошибка получения списка: {result.stderr[:100]}"
            print(msg)
            send_notification(msg,
                title=f"Galaxy {hostname_proc}: Файлы не прочитаны с яндекса",
                priority='urgent',
                tags=['warning', 'rotating_light'])
            return
        remote_files = [f.strip() for f in result.stdout.splitlines()]
    except Exception as e:
        msg = f"  [yadisk] Недоступен: {e}"
        send_notification(msg,
            title=f"Galaxy {hostname_proc}: Файлы не прочитаны с яндекса",
            priority='urgent',
            tags=['warning', 'rotating_light'])
        print(msg)
        return

    # Файлы своего сервера — не перезаписываем НИКОГДА
    own_files = set()
    for pattern in host_patterns:
        for f in glob.glob(pattern):
            own_files.add(os.path.basename(f))
    # Защищаем UpsFile текущего процесса
    own_files.add(os.path.basename(UpsFile))

    if force_update:
        print(f"  [yadisk] force_update=True: "
              f"перезаписываем устаревшие файлы "
              f"(защищены свои: {own_files})")

    downloaded = 0
    skipped    = 0

    for remote_fname in remote_files:
        # Проверяем совпадение с паттернами хранилища
        matches_storage = any(
            glob.fnmatch.fnmatch(remote_fname, os.path.basename(p))
            for p in storage_patterns
        )
        if not matches_storage:
            continue

        # Свои файлы не перезаписываем никогда
        if remote_fname in own_files:
            skipped += 1
            print(f"  [yadisk] ПРОПУСК (свой файл): {remote_fname}")
            continue

        # Если файл уже есть локально и force_update=False — пропускаем
        local_path = os.path.join(local_dir, remote_fname)
        if os.path.exists(local_path) and not force_update:
            skipped += 1
            continue

        # Скачиваем
        remote_path = f"{RCLONE_REMOTE}:{remote_dir}/{remote_fname}"
        cmd = ['rclone', 'copyto', remote_path, remote_fname,
               '--stats-one-line']
        if force_update:
            # Перезаписать даже если время совпадает
            cmd.append('--ignore-times')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode == 0:
                size = (os.path.getsize(remote_fname)
                        if os.path.exists(remote_fname) else 0)
                action = "↓↓" if force_update else "↓"
                print(f"  [yadisk] {action} {remote_fname} "
                      f"({size/1024:.1f} KB)")
                downloaded += 1
            else:
                print(f"  [yadisk] ✗ {remote_fname}: "
                      f"{result.stderr[:80]}")
        except Exception as e:
            print(f"  [yadisk] ✗ {remote_fname}: {e}")

    msg = (f"  [yadisk] Скачано/обновлено: {downloaded}, "
           f"пропущено: {skipped}")
    print(msg)
    send_notification(msg,
        title=f"Galaxy {hostname_proc}: Файлы прочитаны с яндекса",
        priority='high',
        tags=['white_check_mark'])

def save_checkpoint(X_obs, Y_obs, turbo, iteration,
                                sync=True):
    local_file = f"checkpoint_{hostname_proc}.pkl"
    state = {
        'X_obs': X_obs.cpu().numpy(),
        'Y_obs': Y_obs.cpu().numpy(),
        'turbo_length': turbo.length,
        'turbo_success': turbo.success_count,
        'turbo_failure': turbo.failure_count,
        'iteration': iteration,
        'timestamp': datetime.datetime.now().isoformat(),
        'hostname_proc':       hostname_proc,
        # Глобальные счётчики
        'n_h_IC_lw':      number_of_h_IC_lw,
        'n_find_w_U':     number_of_find_w_U,
        'best_target':    best_overall_target,
        'best_Upsilon':   best_overall_Upsilon,
    }
   # Локальное сохранение
    tmp_file = local_file + '.tmp'
    with open(tmp_file, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp_file, local_file)   # атомарная замена
    print(f"  Checkpoint сохранён локально: итерация {iteration}")
    
    # Синхронизация на Яндекс.Диск
    if sync:
        sync_to_yadisk()

# def load_checkpoint(filename='checkpoint.pkl'):
#     if not os.path.exists(filename):
#         return None
#     with open(filename, 'rb') as f:
#         return pickle.load(f)

def finalize(best_params, best_Upsilon, best_penalty):
    """Финальные действия перед выключением."""
    
    print("\n" + "="*50)
    print("ФИНАЛИЗАЦИЯ: сохранение и синхронизация")
    print("="*50)
    
    # 1. Telegram: расчёт завершён
    send_notification(
        f"Расчёт на {hostname_proc} ЗАВЕРШЁН\n"
        f"incl:={incl:.2f}\n"
        f"penalty: {best_penalty:.4f}\n"
        f"Q={best_params['Q']:.4f}, gh={best_params['gh']:.4f}\n"
        f"rh={best_params['rh']:.4f}, rho0={best_params['rho0']:.4f}\n"
        f"Upsilon={best_Upsilon:.4f}",
        title=f"Galaxy {hostname_proc}: Готово",
        priority='high',
        tags=['white_check_mark']
        )
    
    # 2. Синхронизация
    print("\nСинхронизация на Яндекс.Диск...")
    sync_ok = sync_to_yadisk()
    
    # 3. Telegram: результат синхронизации
    if sync_ok:
        send_notification(f"Файлы {hostname_proc} загружены на Яндекс.Диск \n",
        title=f"Galaxy {hostname_proc}: Готово",
        priority='high',
        tags=['white_check_mark']                      
                          ) #VM выключается...")
    else:
        send_notification(f"Ошибка синхронизации!\nПроверьте файлы  {hostname_proc} на VM перед удалением!",
        title=f"Galaxy {hostname_proc}: Файлы не загружены",
        priority='urgent',
        tags=['warning', 'rotating_light']
       )
    
    # 4. Пауза чтобы убедиться что всё записано
#    print("Ожидание 30 сек для завершения записи...")
#    time.sleep(30)
    
    # 5. Выключение
#    print("Выключение VM...")
#    subprocess.run(['sudo', 'shutdown', '-h', 'now'])

# ==============================================================
#  ЦЕЛЕВАЯ ФУНКЦИЯ
# ==============================================================
def halo_IC_lib_weights_pca_fixed(pc_coords, model_data, bounds_original,
                                    densityStars, datasets, alphah, betah,
                                    Upsilon_lower=0.1, Upsilon_upper=1.6,
                                    numOrbits=100000, trajsize=1000, intTime=100.,
                                    regul=1.0,
                                    # НОВЫЙ параметр: прямые параметры без PCA
                                    direct_params=None):
    global best_overall_Upsilon, best_overall_target, number_of_h_IC_lw, number_of_find_w_U, hostname_proc, UpsFile, _ups_recent
    
   # --- РЕЖИМ БЕЗ PCA: direct_params передан напрямую ---
    if direct_params is not None:
        params = direct_params
        # Проверка границ
        param_names = ['Q', 'gh', 'rh', 'rho0']
        for name in param_names:
            lo, hi = bounds_original[name]
            params[name] = float(numpy.clip(params[name], lo, hi))
        print(f"  [direct] Q={params['Q']:.4f}, gh={params['gh']:.4f}, "
              f"rh={params['rh']:.4f}, rho0={params['rho0']:.4f}")
        # pc_coords используется только для логирования
        pc_coords_log = numpy.array([params['Q'], params['gh'],
                                      params['rh'], params['rho0']])
    else:
        # --- ОБЫЧНЫЙ РЕЖИМ: через PCA ---
        if model_data is None:
            print("  ОШИБКА: model_data=None и direct_params=None. Пропускаем.")
            return -1e6
        
        try:
            params = pca_to_params_fixed(pc_coords, model_data, bounds_original)
            pc_back = params_to_pca_fixed(params, model_data)
            error = numpy.abs(pc_coords - pc_back)
            if error.max() > 0.5:
                print(f"  ВНИМАНИЕ: большая ошибка PCA-преобразования: {error.max():.3f}")
        except Exception as e:
            print(f"  Ошибка при переводе в PCA-координаты: {e}. Пропускаем точку.")
            return -1e6
        
#        pc_coords_log = pc_coords
    
    Q    = params['Q']
    gh   = params['gh']
    rh   = params['rh']
    rho0 = params['rho0']
    
    print(f"  → Q={Q:.4f}, gh={gh:.4f}, rh={rh:.4f}, rho0={rho0:.4f}")
    
    try:
        densityHalo = agama.Density(
            type='spheroid', 
            alpha=alphah, 
            beta=betah, 
            gamma=gh, 
            axisratioz=Q,
            densitynorm=rho0, 
            scaleradius=rh, 
            outercutoffradius=55.0, 
            cutoffstrength=2.5
        )
        
        pot_gal = agama.Potential(
            type='Multipole',
            density=agama.Density(densityStars, densityHalo),
            lmax=4, mmax=0, gridSizeR=23
        )
        
        ic = numpy.vstack((
            densityStars.sample(int(numOrbits), potential=pot_gal)[0]
        ))
        
        matrices = agama.orbit(
            potential=pot_gal, 
            ic=ic, 
            time=pot_gal.Tcirc(ic) * intTime, 
            Omega=0.0,
            targets=[d.target for d in datasets], 
            trajsize=trajsize
        )
        matrices = matrices[:-1]
        
    except Exception as e:
        print(f"  Ошибка при создании модели: {e}")
        return -1e6
    
    num_dof = sum([sum(d.cons_err > 0) for d in datasets])
    mult = num_dof**0.5 * 10
    rhs = [d.cons_val / mult for d in datasets]
    pen_cons = [2 * d.cons_err**-2 for d in datasets]
    totalMass = 1.0
    pen_reg = 2. * regul * numpy.ones(numOrbits) * numOrbits / totalMass**2
    
    def _eval_penalty(Upsilon, mats, pen_reg_loc):
        global number_of_find_w_U
        try:
            matrix = [d.getOrbitMatrix(m, Upsilon).T for d, m in zip(datasets, mats)]
            weights = agama.solveOpt(matrix=matrix, rhs=rhs, rpenq=pen_cons, xpenq=pen_reg_loc) * mult
            superpositions = [weights.dot(m) for m in mats]
            penalties = [d.getPenalty(s, Upsilon) for d, s in zip(datasets, superpositions)]
            pen = numpy.sum(penalties[1])
            number_of_find_w_U += 1
            return pen
        except Exception as e:
            msg = f"Error with parameters: {params}, Upsilon: {Upsilon},\n Error: {e}"
            print(msg)
            with open(UpsFile, 'a') as f:
                f.write("#" + msg)
            return 1e6

    # --- п.3: субдискретизация орбит для ВНУТРЕННЕГО поиска (RNG локальный) ---
    if 0.0 < UPS_SUBSAMPLE_FRAC < 1.0:
        n_sub = min(numOrbits, max(1000, int(numOrbits * UPS_SUBSAMPLE_FRAC)))
        sub_idx = numpy.random.default_rng().choice(numOrbits, size=n_sub, replace=False)
        mats_search    = [m[sub_idx] for m in matrices]
        pen_reg_search = 2. * regul * numpy.ones(n_sub) * n_sub / totalMass**2
        print(f"  [subsample] внутренний поиск на {n_sub}/{numOrbits} орбитах")
    else:
        mats_search, pen_reg_search = matrices, pen_reg

    def find_weights_Ups(Upsilon):
        return _eval_penalty(Upsilon, mats_search, pen_reg_search)

    logger = FunctionLogger(find_weights_Ups)

    # --- п.1: адаптивная узкая скобка вокруг предсказанного Upsilon* ---
    if len(_ups_recent) > 0:
        ups0 = float(numpy.median(_ups_recent[-UPS_BRACKET_NMED:]))
        lo = max(Upsilon_lower, ups0 - UPS_BRACKET_DELTA)
        hi = min(Upsilon_upper, ups0 + UPS_BRACKET_DELTA)
        print(f"  [adaptive] скобка [{lo:.3f}, {hi:.3f}] вокруг Ups0={ups0:.3f}")
    else:
        lo, hi = Upsilon_lower, Upsilon_upper

    min_penalty_Ups = minimize_scalar(
        logger,
        bounds=(lo, hi),
        method='bounded',
        options={'xatol': UPS_XATOL, 'maxiter': 50}
    )
    # Откат к полному диапазону, если минимум упёрся в край узкой скобки.
    if (lo > Upsilon_lower or hi < Upsilon_upper) and \
            (min_penalty_Ups.x <= lo + 1e-6 or min_penalty_Ups.x >= hi - 1e-6):
        print("  [adaptive] минимум на краю узкой скобки → повтор на полном диапазоне")
        min_penalty_Ups = minimize_scalar(
            logger,
            bounds=(Upsilon_lower, Upsilon_upper),
            method='bounded',
            options={'xatol': UPS_XATOL, 'maxiter': 50}
        )

    min_Ups = float(min_penalty_Ups.x)

    # --- п.3: финальный ПРОДУКТИВНЫЙ solve на полной библиотеке орбит в найденной
    #     точке. Даёт контрактно-точный penalty (полные numOrbits). Вызывается
    #     напрямую, в обход logger → НЕ попадает в историю оптимизации Upsilon. ---
    if mats_search is not matrices:
        min_pen = float(_eval_penalty(min_Ups, matrices, pen_reg))
        print(f"  [subsample] финальный продуктивный full-solve: penalty={min_pen:.6f}")
    else:
        min_pen = float(min_penalty_Ups.fun)

    # Обновляем скользящую историю Upsilon* (для предсказания скобки, п.1).
    if numpy.isfinite(min_Ups):
        _ups_recent.append(min_Ups)
        if len(_ups_recent) > 64:
            del _ups_recent[0]
    
    number_of_h_IC_lw += 1
    print("number_of_h_IC_lw = ",number_of_h_IC_lw, "N_U = ", number_of_find_w_U)
    print(f"  → min_penalty={min_pen:.6f}, Upsilon={min_Ups:.4f}")
    
    if -min_pen > best_overall_target:
        best_overall_target = -min_pen
        best_overall_Upsilon = min_Ups
    
    with open(UpsFile, 'a') as f:
        f.write(f"# Server: {hostname_proc}\n")
        f.write(f"{incl:0.3f} {Q:0.15f} {gh:0.15f} {rh:0.15f} {rho0:0.15f} "
                f"{min_Ups:0.15f} {min_pen:0.15f} {datetime.datetime.now()}\n")
        pc_str = " ".join([f"{x:.6f}" for x in numpy.atleast_1d(pc_coords)])
        f.write(f"# PCA: {pc_str}\n")
        f.write("# Optimization history (Upsilon values -> function values):\n")
        for i, (upsilon, func_val) in enumerate(logger.history):
            if numpy.ndim(upsilon) == 0:
                upsilon_str = f"{float(upsilon):.15f}"
            else:
                upsilon_str = " ".join([f"{x:.15f}" for x in numpy.atleast_1d(upsilon)])
            f.write(f"# {i:4d}: [{upsilon_str}] -> {func_val:.15f}\n")
        f.write("# End of history\n\n")
    
    print("4UpsBoTorch writed")
    logger.clear_history()
    print("logger history cleaned")
    return -min_pen
# ==============================================================
#  ПОИСК БЛИЖАЙШЕГО НАКЛОНЕНИЯ И BOOTSTRAP НАЧАЛЬНЫХ ТОЧЕК
# ==============================================================

def find_nearest_incl_data(storage_patterns, host_patterns,
                            target_incl, min_points=10,
                            timeout=300):
    """
    Ищет в файлах результатов данные для наклонения,
    ближайшего к target_incl.
    
    Возвращает:
        data_nearest : numpy.ndarray shape (N, 7) или None
                       [incl, Q, gh, rh, rho0, Upsilon, penalty]
        nearest_incl : float — найденное ближайшее наклонение
        dist         : float — расстояние |target_incl - nearest_incl|
    """
    # --- Скачиваем свежие файлы ---
    load_from_yadisk(storage_patterns, host_patterns, timeout=timeout)

    # --- Собираем список файлов ---
    all_files = []
    for pattern in host_patterns + storage_patterns:
        for f in glob.glob(pattern):
            if f not in all_files:
                all_files.append(f)

    if not all_files:
        print("  [find_nearest_incl] Нет файлов для анализа")
        return None, None, float('inf')

    # --- Читаем все строки, собираем уникальные incl ---
    all_rows = []
    for filepath in all_files:
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
                        # Базовые проверки
                        if row[3] <= 0 or row[4] <= 0:
                            continue
                        if row[6] >= 1e5:
                            continue
                        all_rows.append(row)
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass

    if not all_rows:
        print("  [find_nearest_incl] Файлы пусты или нечитаемы")
        return None, None, float('inf')

    data_all = numpy.array(all_rows)

    # --- Уникальные наклонения (исключая target_incl) ---
    unique_incls = numpy.unique(numpy.round(data_all[:, 0], 2))
    # Убираем само целевое наклонение
    unique_incls = unique_incls[numpy.abs(unique_incls - target_incl) > 0.01]

    if len(unique_incls) == 0:
        print("  [find_nearest_incl] Нет других наклонений в файлах")
        return None, None, float('inf')

    print(f"  [find_nearest_incl] Найдены наклонения: {unique_incls}")

    # --- Ищем ближайшее с достаточным числом точек ---
    dists = numpy.abs(unique_incls - target_incl)
    order = numpy.argsort(dists)

    for idx in order:
        candidate_incl = unique_incls[idx]
        mask = numpy.abs(data_all[:, 0] - candidate_incl) < 0.01
        candidate_data = data_all[mask]

        if len(candidate_data) >= min_points:
            dist = float(dists[idx])
            print(f"  [find_nearest_incl] Выбрано incl={candidate_incl:.2f} "
                  f"(dist={dist:.2f}°, {len(candidate_data)} точек)")
            return candidate_data, float(candidate_incl), dist

    # Если ни одно не прошло порог min_points — берём ближайшее
    best_idx       = order[0]
    nearest_incl   = float(unique_incls[best_idx])
    dist           = float(dists[best_idx])
    mask           = numpy.abs(data_all[:, 0] - nearest_incl) < 0.01
    nearest_data   = data_all[mask]
    print(f"  [find_nearest_incl] Ближайшее (мало точек): "
          f"incl={nearest_incl:.2f} (dist={dist:.2f}°, "
          f"{len(nearest_data)} точек)")
    return nearest_data, nearest_incl, dist


def select_bootstrap_candidates(data_nearest, n_bootstrap,
                                 penalty_cutoff_frac=0.5,
                                 strategy='best_diverse'):
    """
    Выбирает n_bootstrap кандидатов из data_nearest для пересчёта.
    
    Стратегии:
      'best'        — просто лучшие по penalty
      'best_diverse'— лучшие + равномерное покрытие пространства параметров
      'uniform'     — равномерное покрытие без учёта penalty
    
    Возвращает: список словарей {'Q', 'gh', 'rh', 'rho0'}
    """
    # Сортируем по penalty
    data_sorted = data_nearest[numpy.argsort(data_nearest[:, 6])]

    # Отсечка: берём только лучшие penalty_cutoff_frac
    n_pool = max(n_bootstrap, int(len(data_sorted) * penalty_cutoff_frac))
    n_pool = min(n_pool, len(data_sorted))
    pool   = data_sorted[:n_pool]

    if strategy == 'best':
        selected = pool[:n_bootstrap]

    elif strategy == 'best_diverse':
        # Берём половину лучших, половину — равномерно из пула
        n_best = max(1, n_bootstrap // 2)
        n_div  = n_bootstrap - n_best
        best_part = pool[:n_best]

        # Равномерное покрытие: делим пул на n_div частей
        if n_div > 0 and len(pool) > n_best:
            rest    = pool[n_best:]
            indices = numpy.linspace(0, len(rest) - 1,
                                     n_div, dtype=int)
            div_part = rest[indices]
            selected = numpy.vstack([best_part, div_part])
        else:
            selected = best_part

    elif strategy == 'uniform':
        indices  = numpy.linspace(0, len(pool) - 1,
                                  n_bootstrap, dtype=int)
        selected = pool[indices]

    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    # Преобразуем в список словарей
    candidates = []
    for row in selected:
        candidates.append({
            'Q':    float(row[1]),
            'gh':   float(row[2]),
            'rh':   float(row[3]),
            'rho0': float(row[4]),
        })

    return candidates


def bootstrap_initial_points_from_nearest_incl(
        storage_patterns,
        host_patterns,
        target_incl,
        model_data_template,
        bounds_original,
        n_bootstrap=8,
        penalty_cutoff_frac=0.5,
        strategy='best_diverse',
        max_dist_warn=20.0,
):
    """
    Формирует начальные точки для нового target_incl,
    пересчитывая penalty для лучших параметров из ближайшего наклонения.
    
    Параметры:
        storage_patterns     : паттерны файлов хранилища
        host_patterns        : паттерны файлов своего сервера
        target_incl          : целевое наклонение (градусы)
        model_data_template  : PCA-модель (для обратного преобразования)
                               Если None — используются сырые параметры
        bounds_original      : словарь границ параметров
        n_bootstrap          : сколько точек пересчитать
        penalty_cutoff_frac  : доля лучших точек для пула кандидатов
        strategy             : стратегия выбора ('best', 'best_diverse',
                               'uniform')
        max_dist_warn        : предупреждение если расстояние > этого (°)
    
    Возвращает:
        bootstrap_results : list of dict
            [{'params': {...}, 'penalty': float, 'Upsilon': float}, ...]
        nearest_incl      : float
        dist              : float
    """
    print("\n" + "=" * 60)
    print(f"BOOTSTRAP: начальные точки для incl={target_incl:.2f}°")
    print("=" * 60)

    # --- Шаг 1: найти ближайшее наклонение ---
    data_nearest, nearest_incl, dist = find_nearest_incl_data(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        target_incl      = target_incl,
    )

    if data_nearest is None:
        print("  [bootstrap] Нет данных для bootstrap. "
              "Будут использованы случайные точки.")
        return [], None, float('inf')

    if dist > max_dist_warn:
        msg = (f"  [bootstrap] ВНИМАНИЕ: ближайшее наклонение "
               f"incl={nearest_incl:.2f}° далеко от целевого "
               f"incl={target_incl:.2f}° (dist={dist:.1f}°). "
               f"Начальные точки могут быть неточными.")
        print(msg)
        send_notification(
            msg,
            title=f"Galaxy {hostname_proc}: Bootstrap предупреждение",
            priority='low',
            tags=['warning']
        )

    print(f"  Ближайшее наклонение: incl={nearest_incl:.2f}° "
          f"(расстояние {dist:.2f}°)")
    print(f"  Доступно точек: {len(data_nearest)}")
    print(f"  Стратегия выбора: {strategy}, n_bootstrap={n_bootstrap}")

    # --- Шаг 2: выбрать кандидатов ---
    candidates = select_bootstrap_candidates(
        data_nearest,
        n_bootstrap          = n_bootstrap,
        penalty_cutoff_frac  = penalty_cutoff_frac,
        strategy             = strategy,
    )

    print(f"\n  Выбрано кандидатов: {len(candidates)}")
    print(f"  {'#':>3s}  {'Q':>8s}  {'gh':>8s}  "
          f"{'rh':>8s}  {'rho0':>8s}")
    print(f"  {'-'*45}")
    for i, c in enumerate(candidates):
        print(f"  {i+1:3d}  {c['Q']:8.4f}  {c['gh']:8.4f}  "
              f"{c['rh']:8.4f}  {c['rho0']:8.4f}")

    # --- Шаг 3: пересчёт penalty для target_incl ---
    bootstrap_results = []
    n_success = 0
    n_failed  = 0

    print(f"\n  Пересчёт penalty для incl={target_incl:.2f}°...")

    for i, params in enumerate(candidates):
        print(f"\n  [{i+1}/{len(candidates)}] "
              f"Q={params['Q']:.4f}, gh={params['gh']:.4f}, "
              f"rh={params['rh']:.4f}, rho0={params['rho0']:.4f}")

        # Переводим в PCA-координаты (если модель есть)
        if model_data_template is not None:
            try:
                pc_coords = params_to_pca_fixed(params, model_data_template)
            except Exception as e:
                print(f"    Ошибка params_to_pca: {e}, "
                      f"используем прямые параметры")
                # Создаём фиктивные PCA-координаты
                pc_coords = _params_to_dummy_pc(
                    params, model_data_template, bounds_original
                )
        else:
            # Без PCA-модели: создаём dummy-координаты
            pc_coords = _params_to_dummy_pc(
                params, None, bounds_original
            )

        # Вычисляем penalty для нового incl
        # (глобальные переменные densityStars, datasets, alphah, betah
        #  уже обновлены для target_incl при старте скрипта)
        try:
            y_val = halo_IC_lib_weights_pca_fixed(
                pc_coords,
                model_data_template,
                bounds_original,
                densityStars, datasets, alphah, betah,
                # НОВОЕ: передаём прямые параметры если модели нет
                direct_params=(params if model_data_template is None else None)
            )
            penalty = -y_val

            if numpy.isfinite(penalty) and penalty < 1e5:
                bootstrap_results.append({
                    'params':  params,
                    'penalty': penalty,
                    'pc':      pc_coords,
                })
                n_success += 1
                print(f"    ✓ penalty={penalty:.6f}")
            else:
                print(f"    ✗ penalty={penalty:.6f} (невалидное значение)")
                n_failed += 1

        except Exception as e:
            print(f"    ✗ Ошибка вычисления: {e}")
            n_failed += 1

    # --- Итог ---
    print(f"\n  Bootstrap завершён: "
          f"успешно={n_success}, ошибок={n_failed}")

    if bootstrap_results:
        best_pen = min(r['penalty'] for r in bootstrap_results)
        print(f"  Лучший penalty из bootstrap: {best_pen:.6f}")

        send_notification(
            f"Bootstrap для incl={target_incl:.2f}° завершён\n"
            f"Ближайшее incl={nearest_incl:.2f}° (dist={dist:.1f}°)\n"
            f"Успешно: {n_success}/{len(candidates)}\n"
            f"Лучший penalty: {best_pen:.6f}",
            title=f"Galaxy {hostname_proc}: Bootstrap",
            priority='default',
            tags=['white_check_mark']
        )

    return bootstrap_results, nearest_incl, dist


def _params_to_dummy_pc(params, model_data, bounds_original):
    """
    Вспомогательная функция: создаёт «нормализованные» координаты
    из словаря параметров без PCA-модели.
    Используется как fallback при отсутствии модели.
    """
    param_names = ['Q', 'gh', 'rh', 'rho0']
    coords = []
    for name in param_names:
        lo, hi = bounds_original[name]
        val    = params[name]
        # Нормализация в [-1, 1]
        coords.append(2.0 * (val - lo) / (hi - lo) - 1.0)
    return numpy.array(coords)


def build_initial_pca_from_bootstrap(bootstrap_results,
                                      bounds_original,
                                      n_components=3,
                                      use_log_scale=True,
                                      expand_pca_bounds=2.5,
                                      output_file=None):
    """
    Строит PCA-модель из результатов bootstrap.
    
    Возвращает model_data совместимый с остальным кодом.
    """
    if output_file is None:
        output_file = torchFile_result

    def _write(text):
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')

    _write(f"\n  [bootstrap PCA] Строим PCA из {len(bootstrap_results)} точек")

    # Собираем массив параметров
    param_names = ['Q', 'gh', 'rh', 'rho0']
    X_raw = numpy.array([
        [r['params'][name] for name in param_names]
        for r in bootstrap_results
    ])
    penalties = numpy.array([r['penalty'] for r in bootstrap_results])

    # Логарифмирование
    if use_log_scale:
        X_tr        = X_raw.copy()
        X_tr[:, 2]  = numpy.log10(numpy.maximum(X_raw[:, 2], 1e-10))
        X_tr[:, 3]  = numpy.log10(numpy.maximum(X_raw[:, 3], 1e-10))
    else:
        X_tr = X_raw

    # Взвешенное масштабирование
    weights       = numpy.exp(-penalties / 0.1)
    weighted_mean = numpy.average(X_tr, weights=weights, axis=0)
    weighted_std  = numpy.sqrt(
        numpy.average((X_tr - weighted_mean)**2,
                       weights=weights, axis=0)
    )
    weighted_std  = numpy.where(weighted_std < 1e-10, 1.0, weighted_std)
    X_scaled      = (X_tr - weighted_mean) / weighted_std

    # Если точек мало — уменьшаем n_components
    n_comp_actual = min(n_components, len(bootstrap_results) - 1,
                        X_raw.shape[1])
    if n_comp_actual < n_components:
        _write(f"  [bootstrap PCA] ВНИМАНИЕ: уменьшаем n_components "
               f"{n_components} → {n_comp_actual} "
               f"(мало точек: {len(bootstrap_results)})")

    pca = PCA(n_components=n_comp_actual)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    pca_bounds_lower = X_pca.min(axis=0) - expand_pca_bounds
    pca_bounds_upper = X_pca.max(axis=0) + expand_pca_bounds

    cumvar = numpy.cumsum(pca.explained_variance_ratio_)
    _write(f"  [bootstrap PCA] Объяснённая дисперсия: "
           f"{pca.explained_variance_ratio_}")
    _write(f"  [bootstrap PCA] Кумулятивная:          {cumvar}")

    # data_good в формате [incl, Q, gh, rh, rho0, Ups, penalty]
    data_good = numpy.hstack([
        numpy.full((len(X_raw), 1), incl),
        X_raw,
        numpy.zeros((len(X_raw), 1)),   # Upsilon (заглушка)
        penalties[:, numpy.newaxis]
    ])

    scaler     = WeightedScaler(weighted_mean, weighted_std)
    model_data = {
        'scaler':           scaler,
        'pca':              pca,
        'X_good':           X_tr,
        'X_raw':            X_raw,
        'use_log_scale':    use_log_scale,
        'pca_bounds_lower': pca_bounds_lower,
        'pca_bounds_upper': pca_bounds_upper,
        'data_good':        data_good,
        'weights':          weights,
    }

    _write(f"  [bootstrap PCA] Модель построена. "
           f"n_components={n_comp_actual}")

    return model_data
# ==============================================================
#  КЛАСС TuRBO ДЛЯ PCA-ПРОСТРАНСТВА
# ==============================================================
class TuRBO_PCA_Fixed:
    def __init__(
        self,
        model_data,
        output_file,
        bounds_original,
        noise_var=0.017**2,
        batch_size=1,
        length_init=0.8,
        length_min=0.5**7,
        length_max=1.6,
        success_tol=3,
        failure_tol=None,
        n_candidates=5000,
        device=torch.device('cpu'),
        dtype=torch.double,
    ):
        self.model_data = model_data
        self.pca = model_data['pca']
        self.bounds_original = bounds_original
        self.n_components = self.pca.n_components_
        
        self.noise_var = noise_var
        self.batch_size = batch_size
        self.length = length_init
        self.length_min = length_min
        self.length_max = length_max
        self.success_tol = success_tol
        self.failure_tol = failure_tol if failure_tol is not None else self.n_components
        self.n_candidates = n_candidates
        self.output_file = output_file 
        self.device = device
        self.dtype = dtype
        
        self.success_count = 0
        self.failure_count = 0
        
        self.pca_bounds_lower = model_data['pca_bounds_lower']
        self.pca_bounds_upper = model_data['pca_bounds_upper']
        self.pca_range = self.pca_bounds_upper - self.pca_bounds_lower
        
        print("\nTuRBO инициализирован {datetime.datetime.now()} :")
        print(f"  n_components = {self.n_components}")
        print(f"  length_init = {self.length}")
        print(f"  PCA bounds: [{self.pca_bounds_lower[0]:.2f}, {self.pca_bounds_upper[0]:.2f}] x ...")
        
        with open(self.output_file , 'a') as f:
            f.write("# TuRBO инициализирован:")
            f.write(f"#  n_components = {self.n_components}")
            f.write(f"#  length_init = {self.length}")
            f.write(f"#  PCA bounds: [{self.pca_bounds_lower[0]:.2f}, {self.pca_bounds_upper[0]:.2f}] x ...")
    
    def _fit_gp(self, X, Y):
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-8))
        likelihood.noise = torch.tensor(self.noise_var, dtype=self.dtype)
        
        model = SingleTaskGP(
            train_X=X,
            train_Y=Y,
            likelihood=likelihood,
            input_transform=Normalize(d=self.n_components),
            outcome_transform=Standardize(m=1),
        )
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model
    
    def _tr_bounds(self, x_center):
        x_center_norm = (x_center - self.pca_bounds_lower) / self.pca_range
        
        half = self.length / 2.0
        lo_norm = torch.clamp(x_center_norm - half, 0.0, 1.0)
        hi_norm = torch.clamp(x_center_norm + half, 0.0, 1.0)
        
        lo = lo_norm * self.pca_range + self.pca_bounds_lower
        hi = hi_norm * self.pca_range + self.pca_bounds_upper
        
        return torch.stack([lo, hi])
    
    def _update_tr(self, y_new_best, y_prev_best):
        rel_improvement = (y_new_best - y_prev_best) / (abs(y_prev_best) + 1e-8)
        
        if rel_improvement > 1e-4:
            self.success_count += 1
            self.failure_count = 0
        else:
            self.failure_count += 1
            self.success_count = 0
        
        if self.success_count >= self.success_tol:
            self.length = min(self.length * 2.0, self.length_max)
            self.success_count = 0
            print(f"  [TuRBO] TR расширена → length = {self.length:.4f}")    
            with open(self.output_file, 'a') as f:
                f.write(f"#  [TuRBO] TR расширена → length = {self.length:.4f}")
        
        if self.failure_count >= self.failure_tol:
            self.length = self.length / 2.0
            self.failure_count = 0
            print(f"  [TuRBO] TR сужена → length = {self.length:.4f}")   
            with open(self.output_file, 'a') as f:
                f.write(f"#  [TuRBO] TR сужена → length = {self.length:.4f}")
    
    def suggest(self, X_obs, Y_obs):
        best_idx = Y_obs.argmax()
        x_center = X_obs[best_idx]
        tr_bounds = self._tr_bounds(x_center)
        
        model = self._fit_gp(X_obs, Y_obs)
        model.eval()
        
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_obs,
            prune_baseline=True,
        )
        
        X_next, _ = optimize_acqf(
            acq_function=acqf,
            bounds=tr_bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=self.n_candidates,
        )
        return X_next

# ==============================================================
#  ОСНОВНАЯ ФУНКЦИЯ ОПТИМИЗАЦИИ
# ==============================================================
def adaptive_penalty_cutoff(data, target_fraction=0.3, min_points=10, cutoff_start=0.60):
    """
    Выбирает cutoff так, чтобы оставить target_fraction лучших точек.
    Гарантирует, что будет выбрано не менее min_points (чтобы PCA не падал).
    """
    penalties = data[:, 6]
    n_total = len(penalties)
    
    # Если всего данных меньше или равно min_points, берем их все
    if n_total <= min_points:
        return numpy.max(penalties)
        
    # Вычисляем порог по процентилю
    cutoff = numpy.percentile(penalties, target_fraction * 100)
    
    # Пытаемся ограничить порог значением cutoff_start, 
    # НО только если при этом останется хотя бы min_points точек
    if cutoff > cutoff_start:
        if numpy.sum(penalties <= cutoff_start) >= min_points:
            cutoff = cutoff_start
            
    # Финальная проверка: если даже текущий cutoff оставляет слишком мало точек,
    # принудительно берем значение penalty у min_points-ой по счету точки
    if numpy.sum(penalties <= cutoff) < min_points:
        sorted_penalties = numpy.sort(penalties)
        cutoff = sorted_penalties[min_points - 1]
        
    return cutoff

def load_fresh_data_from_files(storage_patterns, host_patterns, incl_filter, 
                                use_log_scale=True,
                                exclude_suffix=hostname_proc,
                                return_full=False):
    # --- Шаг 1: скачиваем свежие файлы с Яндекс.Диска ---
    load_from_yadisk(storage_patterns, host_patterns)
    
    # --- Шаг 2: собираем список файлов на диске ---
    all_files = []
    
    # ДИАГНОСТИКА: показываем что нашли по каждому паттерну
    print("  [load_fresh] Поиск файлов по паттернам:")
    print(f"    host_patterns:    {host_patterns}")
    print(f"    storage_patterns: {storage_patterns}")
    print(f"    exclude_suffix:   '{exclude_suffix}'")
    
    for pattern in host_patterns + storage_patterns:
        found = glob.glob(pattern)
        print(f"    паттерн '{pattern}': найдено {found}")
        for f in found:
            if f not in all_files:
                # ДИАГНОСТИКА: показываем решение по каждому файлу
                if exclude_suffix and exclude_suffix in f:
                    print(f"      ПРОПУСК (exclude_suffix): {f}")
                    continue
                all_files.append(f)
                print(f"      ДОБАВЛЕН: {f}")
    
    print(f"  [load_fresh] Итого файлов для чтения: {len(all_files)}")
    
    raw = []
    file_counts = {}
    
    for file in all_files:
        count = 0
        try:
            with open(file, 'r') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    try:
                        row = [float(p) for p in parts[:7]]
                        row = [1e6 if numpy.isinf(val) else val 
                               for val in row]
                        # ДИАГНОСТИКА: первые несколько строк
                        if count < 2:
                            print(f"      [{file}] строка: incl={row[0]:.2f}, "
                                  f"pen={row[6]:.4f}, "
                                  f"фильтр incl={incl_filter}: "
                                  f"{'OK' if abs(row[0]-incl_filter)<0.01 else 'SKIP'}")
                        # Фильтр по наклонению
                        if abs(row[0] - incl_filter) > 0.01:
                            continue
                        if row[3] <= 0 or row[4] <= 0:
                            continue
                        if row[6] >= 1e5:
                            continue
                        raw.append(row)
                        count += 1
                    except ValueError:
                        continue
        except FileNotFoundError:
            print(f"    ФАЙЛ НЕ НАЙДЕН: {file}")
        file_counts[file] = count
    
    print("  [PCA update] Загружено точек по файлам:")
    for fname, cnt in file_counts.items():
        print(f"    {fname}: {cnt}")
    
    if len(raw) == 0:
        if return_full:
            return None, file_counts
        else:
            return None, None, file_counts
    
    data = numpy.array(raw)
    print(f"  [load_fresh] Итого точек из файлов: {len(data)}")

    if return_full:
        return data, file_counts
    else:
        X_raw     = data[:, 1:5]
        penalties = data[:, 6]
        return X_raw, penalties, file_counts

def _update_pca_model(model_data, data_good, new_params, new_penalties,
                      bounds_original, use_log_scale, expand_pca_bounds,
                      X_obs, Y_obs, turbo,
                      output_file, dtype, device,
                      # Параметры для чтения файлов:
                      storage_patterns=None,  # паттерны всех файлов хранилища
                      host_patterns=None,     # паттерны файлов своего сервера
                      incl_filter=None,       # фильтр по incl
                      read_parallel=True,     # читать файлы параллельных процессов
                      current_suffix=None,    # суффикс текущего процесса
                      penalty_cutoff=2.0,     # отсечка по penalty
                      ):
    """
    Пересчёт PCA с учётом:
    1. Всех файлов с диска кроме своего (исторические + параллельные
       + другие серверы) — Вариант А
    2. Буфера новых точек текущего процесса
    Вариант Б (fallback): исторические из model_data + буфер
    """
    param_names = ['Q', 'gh', 'rh', 'rho0']

    def _write(text):
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')

    _write("\n  [PCA update] Источники данных:")

    # Флаг: удалось ли выполнить Вариант А
    variant_a_ok = False

    # -------------------------------------------------------
    # Вариант А: читаем все файлы заново (исторические +
    #            параллельные + другие серверы),
    #            НО НЕ свой файл (он в буфере new_params).
    #            Дубликаты не удаляем — разные запуски
    #            одних параметров дают разные penalty.
    # -------------------------------------------------------
    if (read_parallel
            and storage_patterns is not None
            and host_patterns    is not None
            and incl_filter      is not None):

        X_raw_files, pen_files, file_counts = load_fresh_data_from_files(
            storage_patterns = storage_patterns,
            host_patterns    = host_patterns,
            incl_filter      = incl_filter,
            use_log_scale    = False,          # сырые данные, логарифм ниже
            exclude_suffix   = current_suffix, # ← свой файл исключён
            return_full      = False,          # нужны только X_raw и penalties
        )

        _write("    Режим А: все файлы (кроме своего) + текущий буфер")
        _write(f"    Прочитано файлов: {len(file_counts)}")
        for fname, cnt in file_counts.items():
            _write(f"      {fname}: {cnt} строк")

        parts_X   = []
        parts_pen = []

        if X_raw_files is not None and len(X_raw_files) > 0:
            parts_X.append(X_raw_files)
            parts_pen.append(pen_files)
            _write(f"    Из файлов (все кроме своего): {len(pen_files)}")
        else:
            _write("    Из файлов: 0 (нет данных)")

        if len(new_params) > 0:
            X_raw_new = numpy.array(
                [[p[name] for name in param_names] for p in new_params]
            )
            pen_new   = numpy.array(new_penalties)
            parts_X.append(X_raw_new)
            parts_pen.append(pen_new)
            _write(f"    Из буфера своего процесса:    {len(pen_new)}")
        else:
            _write("    Из буфера своего процесса:    0")

        if len(parts_X) > 0:
            X_raw_all = numpy.vstack(parts_X)
            pen_all   = numpy.hstack(parts_pen)
            # Дубликаты НЕ удаляем — разные запуски одних параметров
            # могут давать разные значения penalty (статистический шум)
            _write(f"    Итого точек для PCA:          {len(pen_all)}")

            if len(pen_all) >= 5:
                variant_a_ok = True
            else:
                _write("    Мало точек, переключаемся на вариант Б")
        else:
            _write("    Нет данных ни из файлов, ни из буфера, "
                   "переключаемся на вариант Б")
    else:
        missing = []
        if storage_patterns is None: missing.append("storage_patterns")
        if host_patterns    is None: missing.append("host_patterns")
        if incl_filter      is None: missing.append("incl_filter")
        if not read_parallel:        missing.append("read_parallel=False")
        _write(f"    Вариант А недоступен: {', '.join(missing)}")

    # -------------------------------------------------------
    # Вариант Б: только исторические из model_data + буфер
    # (fallback если Вариант А не удался)
    # -------------------------------------------------------
    if not variant_a_ok:
        _write("    Режим Б: исторические из model_data + текущий буфер")

        X_raw_hist = model_data['X_raw']
        pen_hist   = model_data['data_good'][:, 6]
        _write(f"    Исторических точек: {len(pen_hist)}")

        if len(new_params) > 0:
            X_raw_new = numpy.array(
                [[p[name] for name in param_names] for p in new_params]
            )
            pen_new   = numpy.array(new_penalties)
            X_raw_all = numpy.vstack([X_raw_hist, X_raw_new])
            pen_all   = numpy.hstack([pen_hist,   pen_new  ])
            _write(f"    Из буфера своего процесса: {len(pen_new)}")
        else:
            X_raw_all = X_raw_hist
            pen_all   = pen_hist
            _write("    Из буфера своего процесса: 0")

        _write(f"    Всего точек: {len(pen_all)}")

    # -------------------------------------------------------
    # Фильтр по penalty_cutoff
    # -------------------------------------------------------
    mask      = pen_all <= penalty_cutoff
    X_raw_all = X_raw_all[mask]
    pen_all   = pen_all[mask]
    _write(f"    После фильтра penalty≤{penalty_cutoff}: {len(pen_all)} точек")

    if len(pen_all) < 5:
        _write("    ВНИМАНИЕ: мало точек для PCA, пропускаем обновление")
        return model_data, X_obs, Y_obs, turbo

    # -------------------------------------------------------
    # Логарифмирование
    # -------------------------------------------------------
    if use_log_scale:
        X_tr_all        = X_raw_all.copy()
        X_tr_all[:, 2]  = numpy.log10(numpy.maximum(X_raw_all[:, 2], 1e-10))
        X_tr_all[:, 3]  = numpy.log10(numpy.maximum(X_raw_all[:, 3], 1e-10))
    else:
        X_tr_all = X_raw_all

    # -------------------------------------------------------
    # Взвешенное масштабирование и PCA
    # -------------------------------------------------------
    weights       = numpy.exp(-pen_all / 0.1)
    weighted_mean = numpy.average(X_tr_all, weights=weights, axis=0)
    weighted_std  = numpy.sqrt(
        numpy.average((X_tr_all - weighted_mean)**2,
                       weights=weights, axis=0)
    )
    weighted_std  = numpy.where(weighted_std < 1e-10, 1.0, weighted_std)
    X_scaled_all  = (X_tr_all - weighted_mean) / weighted_std

    n_components  = model_data['pca'].n_components_
    pca_new       = PCA(n_components=n_components)
    pca_new.fit(X_scaled_all)
    X_pca_all     = pca_new.transform(X_scaled_all)

    # -------------------------------------------------------
    # Новые границы PCA
    # -------------------------------------------------------
    pca_bounds_lower = X_pca_all.min(axis=0) - expand_pca_bounds
    pca_bounds_upper = X_pca_all.max(axis=0) + expand_pca_bounds

    # -------------------------------------------------------
    # Логируем изменение PCA
    # -------------------------------------------------------
    cumvar = numpy.cumsum(pca_new.explained_variance_ratio_)
    _write("\n  [PCA update] Обновлённая дисперсия:")
    for i, (ev, cv) in enumerate(
            zip(pca_new.explained_variance_ratio_, cumvar)):
        _write(f"    PC{i+1}: {ev:.4f}  cumul={cv:.4f}")

    _write("\n  [PCA update] Изменение границ PCA:")
    for i in range(n_components):
        old_lo = model_data['pca_bounds_lower'][i]
        old_hi = model_data['pca_bounds_upper'][i]
        _write(f"    PC{i+1}: [{old_lo:.3f}, {old_hi:.3f}]"
               f" → [{pca_bounds_lower[i]:.3f}, {pca_bounds_upper[i]:.3f}]")

    # -------------------------------------------------------
    # Формируем data_good_new
    # -------------------------------------------------------
    _incl = incl_filter if incl_filter is not None else 90.0
    data_good_new = numpy.hstack([
        numpy.full((len(X_raw_all), 1), _incl),
        X_raw_all,
        numpy.zeros((len(X_raw_all), 1)),   # Upsilon (заглушка)
        pen_all[:, numpy.newaxis]
    ])

    scaler_new     = WeightedScaler(weighted_mean, weighted_std)
    model_data_new = {
        'scaler':           scaler_new,
        'pca':              pca_new,
        'X_good':           X_tr_all,
        'X_raw':            X_raw_all,
        'use_log_scale':    use_log_scale,
        'pca_bounds_lower': pca_bounds_lower,
        'pca_bounds_upper': pca_bounds_upper,
        'data_good':        data_good_new,
        'weights':          weights,
    }

    # -------------------------------------------------------
    # Пересчёт X_obs в новом PCA-пространстве
    # -------------------------------------------------------
    X_obs_np  = X_obs.cpu().numpy()
    X_obs_new = numpy.zeros_like(X_obs_np)
    n_failed  = 0

    for i in range(len(X_obs_np)):
        try:
            params_i     = pca_to_params_fixed(
                X_obs_np[i], model_data, bounds_original
            )
            X_obs_new[i] = params_to_pca_fixed(params_i, model_data_new)
        except Exception:
            X_obs_new[i] = X_obs_np[i]   # fallback: оставляем как есть
            n_failed     += 1

    if n_failed > 0:
        _write(f"  [PCA update] ВНИМАНИЕ: не удалось перепроецировать "
               f"{n_failed}/{len(X_obs_np)} точек (использован fallback)")

    X_obs_new_t = torch.tensor(X_obs_new, dtype=dtype, device=device)

    # -------------------------------------------------------
    # Обновляем TuRBO
    # -------------------------------------------------------
    turbo.pca_bounds_lower = pca_bounds_lower
    turbo.pca_bounds_upper = pca_bounds_upper
    turbo.pca_range        = pca_bounds_upper - pca_bounds_lower
    turbo.model_data       = model_data_new

    _write(f"  [PCA update] Готово. "
           f"Вариант: {'А' if variant_a_ok else 'Б'}, "
           f"точек: {len(pen_all)}")

    return model_data_new, X_obs_new_t, Y_obs, turbo

def _generate_random_initial_points(bounds_original, n_points,
                                     output_file=None):
    """
    Последний резерв: вычисляет penalty для случайных точек
    из пространства параметров (Latin Hypercube Sampling).
    
    Возвращает:
        data            : numpy.ndarray shape (N, 7)
        bootstrap_results: list of dict
    """
    if output_file is None:
        output_file = torchFile_result

    def _write(text):
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')

    _write(f"\n  [random init] Генерация {n_points} случайных точек "
           f"(Latin Hypercube)")

    param_names = ['Q', 'gh', 'rh', 'rho0']

    # Latin Hypercube Sampling
    rng      = numpy.random.default_rng(seed=42)
    n_params = len(param_names)
    # Разбиваем [0,1]^n на n_points равных ячеек по каждому измерению
    lhs = numpy.zeros((n_points, n_params))
    for j in range(n_params):
        perm       = rng.permutation(n_points)
        lhs[:, j]  = (perm + rng.random(n_points)) / n_points

    # Масштабируем в пространство параметров
    candidates = []
    for i in range(n_points):
        params = {}
        for j, name in enumerate(param_names):
            lo, hi       = bounds_original[name]
            params[name] = lo + lhs[i, j] * (hi - lo)
        candidates.append(params)

    _write(f"  {'#':>3s}  {'Q':>8s}  {'gh':>8s}  "
           f"{'rh':>8s}  {'rho0':>8s}")
    _write(f"  {'-'*45}")
    for i, c in enumerate(candidates):
        _write(f"  {i+1:3d}  {c['Q']:8.4f}  {c['gh']:8.4f}  "
               f"  {c['rh']:8.4f}  {c['rho0']:8.4f}")

    # Вычисляем penalty
    bootstrap_results = []
    for i, params in enumerate(candidates):
        _write(f"\n  [{i+1}/{n_points}] Вычисление penalty...")
        # Dummy PCA-координаты (нормализованные параметры)
        pc_coords = _params_to_dummy_pc(params, None, bounds_original)
        try:
            y_val   = halo_IC_lib_weights_pca_fixed(
                pc_coords, None, bounds_original,
                densityStars, datasets, alphah, betah,
                # НОВОЕ: прямые параметры
                direct_params=params,
            )
            penalty = -y_val
            if numpy.isfinite(penalty) and penalty < 1e5:
                bootstrap_results.append({
                    'params':  params,
                    'penalty': penalty,
                    'pc':      pc_coords,
                })
                _write(f"    ✓ penalty={penalty:.6f}")
        except Exception as e:
            _write(f"    ✗ Ошибка: {e}")

    if not bootstrap_results:
        return numpy.empty((0, 7)), []

    rows = numpy.array([
        [incl,
         r['params']['Q'],  r['params']['gh'],
         r['params']['rh'], r['params']['rho0'],
         0.0,               r['penalty']]
        for r in bootstrap_results
    ])
    _write(f"\n  [random init] Успешно: {len(rows)}/{n_points}")
    return rows, bootstrap_results
def run_pca_optimization(
    storage_patterns=None,   # паттерны всех файлов из хранилища
    host_patterns=None,      # паттерны файлов своего сервера,
    fallback_patterns=None,    # ← НОВЫЙ параметр
    n_components=4,
    n_iter=30,
    target_fraction=0.3,
    cutoff_start=0.6,
    length_init=0.8,
    use_log_scale=True,
    expand_pca_bounds=2.5,
    output_file=None,
    resume=True,
    pca_update_interval=12
):
    if fallback_patterns is None:
        fallback_patterns = ["4UpsBoTorch_PCA_PA46.8_Sersic*.txt"]
        
    global best_overall_Upsilon, best_overall_target, number_of_find_w_U
    global number_of_h_IC_lw, hostname_proc
    global densityStars, datasets, incl, alphah, betah

    if output_file is None:
        output_file = torchFile_result
    if storage_patterns is None:
        storage_patterns = ["4UpsBoTorch_Sersic.txt",
                            "4UpsBoTorch_PCA_Sersic_*.txt"]
    if host_patterns is None:
        host_patterns = [
            f"4UpsBoTorch_PCA_Sersic_{_hostname_env}.txt",
            f"4UpsBoTorch_PCA_Sersic_{_hostname_env}_p*.txt",
        ]

    def _write(text):
        """Вспомогательная функция: печать + запись в файл."""
        print(text)
        with open(output_file, 'a') as f:
            f.write(text + '\n')

    # --- Заголовок лога ---
    _write("# TuRBO-PCA Optimization Log")
    _write(f"# Server: {hostname_proc}")
    _write(f"# Start: {datetime.datetime.now()}")
    _write(f"# storage_patterns: {storage_patterns}")
    _write(f"# host_patterns:    {host_patterns}")
    _write(f"# Iterations planned: {n_iter}")
    _write(f"# PCA components: {n_components}")
    _write(f"# Target fraction: {target_fraction}")
    _write(f"# Expand bounds by: {expand_pca_bounds}")
    _write("")

    best_overall_Upsilon = None
    best_overall_target  = -float('inf')
    number_of_find_w_U   = 0
    number_of_h_IC_lw    = 0

    bounds_original = {
        'Q':   (0.05, 2.5),
        'gh':  (0.0,  1.6),
        'rh':  (0.5,  3.5),
        'rho0':(34.0, 120.0),
    }
# ==============================================================
    # ШАГ 0: ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ ФАЙЛОВ С ЯНДЕКС.ДИСКА
    # ==============================================================
    _write("\n" + "=" * 60)
    _write("ОБНОВЛЕНИЕ ФАЙЛОВ С ЯНДЕКС.ДИСКА (force_update)")
    _write("=" * 60)
    
    load_from_yadisk(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        force_update     = True,   # перезаписать устаревшие локальные копии
    )
   # ==============================================================
    # ШАГ 1: ЗАГРУЗКА ДАННЫХ
    # ==============================================================
    _write("\n" + "=" * 60)
    _write("ЗАГРУЗКА ДАННЫХ И РАСЧЁТ АДАПТИВНОГО CUTOFF")
    _write("=" * 60)

    MIN_POINTS_FOR_PCA = 10  # минимум точек для построения PCA

    data, file_counts = load_fresh_data_from_files(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        incl_filter      = incl,
        use_log_scale    = False,
        exclude_suffix   = None,
        return_full      = True,
    )

    _write(f"\nПрочитано файлов: {len(file_counts)}")
    for fname, cnt in file_counts.items():
        _write(f"  {fname}: {cnt} строк")

    # --- Проверяем достаточность данных ---
    n_have = len(data) if data is not None else 0
    data_sufficient = (n_have >= MIN_POINTS_FOR_PCA)

       # ==============================================================
    # ШАГ 1b: BOOTSTRAP если данных нет или мало
    # ==============================================================
    bootstrap_results   = []   # результаты bootstrap (могут быть пустыми)
    nearest_incl_used   = None
    dist_used           = float('inf')

    if not data_sufficient:
        _write(f"\nДанных для incl={incl} недостаточно "
               f"({n_have} < {MIN_POINTS_FOR_PCA}).")

        # Сколько точек пересчитать:
        # если данных совсем нет — берём 12,
        # иначе добираем до MIN_POINTS_FOR_PCA + 3 запасных
        n_boot = (12 if n_have == 0
                  else max(8, MIN_POINTS_FOR_PCA - n_have + 3))

        # ------------------------------------------------------
        # Попытка 1: bootstrap из ближайшего наклонения
        #            в файлах 4UpsBoTorch_PCA_Sersic*
        # ------------------------------------------------------
        _write("\nПопытка 1: bootstrap из ближайшего наклонения "
               "(файлы 4UpsBoTorch_PCA_Sersic*)...")

        send_notification(
            f"Bootstrap для incl={incl:.2f}°\n"
            f"Данных: {n_have} (нужно ≥ {MIN_POINTS_FOR_PCA})\n"
            f"Попытка 1: ближайшее наклонение из PCA_Sersic*...",
            title=f"Galaxy {hostname_proc}: Bootstrap",
            priority='default',
            tags=['hourglass_flowing_sand']
        )

        bootstrap_results, nearest_incl_used, dist_used = \
            bootstrap_initial_points_from_nearest_incl(
                storage_patterns    = storage_patterns,
                host_patterns       = host_patterns,
                target_incl         = incl,
                model_data_template = None,   # модели ещё нет
                bounds_original     = bounds_original,
                n_bootstrap         = n_boot,
                penalty_cutoff_frac = 0.5,
                strategy            = 'best_diverse',
                max_dist_warn       = 20.0,
            )

        if bootstrap_results:
            _write(f"\n  Попытка 1 успешна: {len(bootstrap_results)} точек "
                   f"(из incl={nearest_incl_used:.2f}°, "
                   f"dist={dist_used:.1f}°)")

        # ------------------------------------------------------
        # Попытка 2: fallback из файлов другого posang
        #            4UpsBoTorch_PCA_PA46.8_Sersic*
        #            penalty из этих файлов — только для выбора!
        #            penalty ПЕРЕСЧИТЫВАЕТСЯ для текущего incl
        # ------------------------------------------------------
        if not bootstrap_results:
            _write("\n  Попытка 1 не дала результатов.")
            _write("Попытка 2: fallback из файлов другого posang "
                   "(4UpsBoTorch_PCA_PA46.8_Sersic*)...")
            _write("  ВНИМАНИЕ: penalty из fallback-файлов используется")
            _write("            ТОЛЬКО для выбора кандидатов!")
            _write("            Для PCA penalty будет ПЕРЕСЧИТАН.")

            send_notification(
                f"Bootstrap (попытка 1) не дал результатов\n"
                f"incl={incl:.2f}°\n"
                f"Попытка 2: fallback из PA46.8_Sersic*...",
                title=f"Galaxy {hostname_proc}: Fallback",
                priority='default',
                tags=['hourglass_flowing_sand']
            )

            bootstrap_results = bootstrap_from_fallback(
                fallback_patterns   = fallback_patterns,
                target_incl         = incl,
                bounds_original     = bounds_original,
                n_bootstrap         = n_boot,
                penalty_cutoff_frac = 0.5,
                strategy            = 'best_diverse',
                output_file         = output_file,
            )

            if bootstrap_results:
                _write(f"\n  Попытка 2 успешна: "
                       f"{len(bootstrap_results)} точек "
                       f"(penalty пересчитан для incl={incl:.2f}°)")
                send_notification(
                    f"Fallback bootstrap успешен\n"
                    f"incl={incl:.2f}°\n"
                    f"Точек: {len(bootstrap_results)}\n"
                    f"Лучший penalty: "
                    f"{min(r['penalty'] for r in bootstrap_results):.4f}",
                    title=f"Galaxy {hostname_proc}: Fallback OK",
                    priority='default',
                    tags=['white_check_mark']
                )

        # ------------------------------------------------------
        # Попытка 3: Latin Hypercube Sampling
        #            последний резерв — случайные точки
        # ------------------------------------------------------
        if not bootstrap_results:
            _write("\n  Попытка 2 не дала результатов.")
            _write("Попытка 3: случайные начальные точки (LHS)...")

            send_notification(
                f"Fallback (попытка 2) не дал результатов\n"
                f"incl={incl:.2f}°\n"
                f"Попытка 3: Latin Hypercube Sampling...",
                title=f"Galaxy {hostname_proc}: LHS",
                priority='default',
                tags=['hourglass_flowing_sand']
            )

            data_lhs, bootstrap_results = _generate_random_initial_points(
                bounds_original = bounds_original,
                n_points        = MIN_POINTS_FOR_PCA,
                output_file     = output_file,
            )

            if bootstrap_results:
                _write(f"\n  Попытка 3 успешна: "
                       f"{len(bootstrap_results)} точек (LHS)")
                send_notification(
                    f"LHS bootstrap успешен\n"
                    f"incl={incl:.2f}°\n"
                    f"Точек: {len(bootstrap_results)}",
                    title=f"Galaxy {hostname_proc}: LHS OK",
                    priority='default',
                    tags=['white_check_mark']
                )
            else:
                _write("  Попытка 3 не дала результатов.")
                send_notification(
                    f"ВСЕ попытки bootstrap провалились!\n"
                    f"incl={incl:.2f}°",
                    title=f"Galaxy {hostname_proc}: ОШИБКА",
                    priority='urgent',
                    tags=['warning', 'rotating_light']
                )

        # ------------------------------------------------------
        # Объединяем результаты bootstrap с имеющимися данными
        # ------------------------------------------------------
        if bootstrap_results:
            # Формируем строки [incl, Q, gh, rh, rho0, Ups, penalty]
            boot_rows = numpy.array([
                [incl,
                 r['params']['Q'],  r['params']['gh'],
                 r['params']['rh'], r['params']['rho0'],
                 0.0,               r['penalty']]
                for r in bootstrap_results
            ])

            if data is not None and len(data) > 0:
                data = numpy.vstack([data, boot_rows])
                _write(f"\n  Объединено: {len(data)} точек "
                       f"(исходные + bootstrap)")
            else:
                data = boot_rows
                _write(f"\n  Только bootstrap: {len(data)} точек")

        else:
            # bootstrap_results пуст — но data_lhs могла быть заполнена
            # в попытке 3 (LHS возвращает data_lhs отдельно)
            if 'data_lhs' in dir() and data_lhs is not None and len(data_lhs) > 0:
                if data is not None and len(data) > 0:
                    data = numpy.vstack([data, data_lhs])
                else:
                    data = data_lhs
                _write(f"\n  Использованы LHS-точки: {len(data_lhs)}")

        # Обновляем флаг после всех попыток
        n_have          = len(data) if data is not None else 0
        data_sufficient = (n_have >= MIN_POINTS_FOR_PCA)

    if not data_sufficient:
        raise ValueError(
            f"Не удалось набрать достаточно начальных точек "
            f"для incl={incl} "
            f"(есть {n_have}, нужно ≥ {MIN_POINTS_FOR_PCA})."
        )

    # --- Фильтр на корректность для логарифмирования ---
    if use_log_scale:
        mask_valid = (data[:, 3] > 0) & (data[:, 4] > 0)
        n_dropped  = numpy.sum(~mask_valid)
        if n_dropped > 0:
            _write(f"  Отброшено точек с rh<=0 или rho0<=0: {n_dropped}")
        data = data[mask_valid]

    _write(f"Итого строк для построения модели (incl={incl}): {len(data)}")

    # --- Сортировка по penalty ---
    data_sort = data[numpy.argsort(data[:, 6])]
    _write(f"Диапазон penalty: [{data_sort[:, 6].min():.4f}, "
           f"{data_sort[:, 6].max():.4f}]")

    # --- Адаптивный cutoff ---
    penalty_cutoff = adaptive_penalty_cutoff(
        data_sort,
        target_fraction = target_fraction,
        cutoff_start    = cutoff_start
    )
    _write(f"Адаптивный penalty cutoff: {penalty_cutoff:.4f}"
           f" (оставляем {target_fraction*100:.0f}% лучших точек)")

    # ==============================================================
    # ШАГ 2: ПОСТРОЕНИЕ PCA-МОДЕЛИ
    # ==============================================================
    _write("\n" + "=" * 60)
    _write("ПОСТРОЕНИЕ PCA-МОДЕЛИ")
    _write("=" * 60)

    mask_good = data_sort[:, 6] <= penalty_cutoff
    data_good = data_sort[mask_good]
    _write(f"Точек с penalty ≤ {penalty_cutoff}: {len(data_good)}")

    # X_raw всегда определяется здесь — независимо от пути (bootstrap / обычный)
    X_raw = data_good[:, 1:5].copy()   # Q, gh, rh, rho0

    if use_log_scale:
        X_transformed       = X_raw.copy()
        X_transformed[:, 2] = numpy.log10(X_raw[:, 2])
        X_transformed[:, 3] = numpy.log10(X_raw[:, 3])
        _write("Используется логарифмическое масштабирование для rh и rho0")
    else:
        X_transformed = X_raw

    weights       = numpy.exp(-data_good[:, 6] / 0.1)
    weighted_mean = numpy.average(X_transformed, weights=weights, axis=0)
    weighted_std  = numpy.sqrt(
        numpy.average((X_transformed - weighted_mean)**2,
                       weights=weights, axis=0)
    )
    weighted_std  = numpy.where(weighted_std < 1e-10, 1.0, weighted_std)
    X_scaled      = (X_transformed - weighted_mean) / weighted_std

    # Если точек мало (bootstrap) — уменьшаем n_components
    n_comp_actual = min(n_components, len(data_good) - 1, X_raw.shape[1])
    if n_comp_actual < n_components:
        _write(f"  ВНИМАНИЕ: уменьшаем n_components "
               f"{n_components} → {n_comp_actual} "
               f"(мало точек: {len(data_good)})")

    pca = PCA(n_components=n_comp_actual)
    pca.fit(X_scaled)

    _write("\n" + "=" * 55)
    if bootstrap_results:

        if nearest_incl_used is not None:
            _write(f"PCA-модель построена на основе bootstrap "
                   f"(incl={nearest_incl_used:.2f}° → {incl:.2f}°):")
        else:
            _write(f"PCA-модель построена на основе bootstrap "
                   f"(incl={incl:.2f}°):")
    else:
        _write("PCA-модель построена (взвешенная, 4 параметра):")
    _write("=" * 55)

    cumvar = numpy.cumsum(pca.explained_variance_ratio_)
    for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        bar = '█' * int(ev * 40)
        _write(f"  PC{i+1}: {ev:6.3f}  cumul={cv:6.3f}  {bar}")
    _write(f"  Итого объяснено: {cumvar[-1]:.4f}")

    X_pca            = pca.transform(X_scaled)
    pca_bounds_lower = X_pca.min(axis=0) - expand_pca_bounds
    pca_bounds_upper = X_pca.max(axis=0) + expand_pca_bounds

    _write("\nГраницы в PCA-пространстве:")
    for i in range(n_comp_actual):
        _write(f"  PC{i+1}: [{pca_bounds_lower[i]:.3f}, "
               f"{pca_bounds_upper[i]:.3f}]")

    # --- Проверка обратного преобразования ---
    X_check_scaled = pca.inverse_transform(X_pca)
    X_check        = weighted_mean + weighted_std * X_check_scaled
    if use_log_scale:
        X_check[:, 2] = 10**X_check[:, 2]
        X_check[:, 3] = 10**X_check[:, 3]

    _write("\nПроверка обратного преобразования:")
    param_names = ['Q', 'gh', 'rh', 'rho0']
    for i, name in enumerate(param_names):
        _write(f"  {name:5s}: [{X_check[:, i].min():.4f}, "
               f"{X_check[:, i].max():.4f}]")

    # --- Сохранение модели ---
    scaler     = WeightedScaler(weighted_mean, weighted_std)
    model_data = {
        'scaler':           scaler,
        'pca':              pca,
        'X_good':           X_transformed,
        'X_raw':            X_raw,
        'use_log_scale':    use_log_scale,
        'pca_bounds_lower': pca_bounds_lower,
        'pca_bounds_upper': pca_bounds_upper,
        'data_good':        data_good,
        'weights':          weights,
    }
    pkl_file = f"pca_model_weighted_{hostname_proc}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(model_data, f)
    _write(f"\nМодель сохранена в {pkl_file}")

    # --- Уведомление о bootstrap ---
    if bootstrap_results:
        if nearest_incl_used is not None:
            _src_txt = (f"incl={incl:.2f}° (из {nearest_incl_used:.2f}°, "
                        f"dist={dist_used:.1f}°)\n")
        else:
            _src_txt = f"incl={incl:.2f}°\n"
        
        send_notification(
            f"PCA-модель построена на bootstrap\n"
            f"{_src_txt}"
            f"Точек: {len(data_good)}, "
            f"n_components={n_comp_actual}",
            title=f"Galaxy {hostname_proc}: PCA готова",
            priority='default',
            tags=['white_check_mark']
        )

    # ==============================================================
    # ШАГ 3: ИНИЦИАЛИЗАЦИЯ GP И TuRBO
    # ==============================================================
    Y_pca  = -data_good[:, 6:7]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.double

    X_obs = torch.tensor(X_pca,  dtype=dtype, device=device)
    Y_obs = torch.tensor(Y_pca,  dtype=dtype, device=device)

    _write(f"\nИсторических точек для GP: {len(X_obs)}")
    _write(f"Y (target): min={Y_obs.min().item():.4f}, "
           f"max={Y_obs.max().item():.4f}")

    # Априорная точка — лучшая из истории
    best_idx        = data_good[:, 6].argmin()
    best_historical = data_good[best_idx]
    prior_params    = {
        'Q':    best_historical[1],
        'gh':   best_historical[2],
        'rh':   best_historical[3],
        'rho0': best_historical[4],
    }

    turbo = TuRBO_PCA_Fixed(
        model_data      = model_data,
        output_file     = output_file,
        bounds_original = bounds_original,
        noise_var       = 0.017**2,
        batch_size      = 1,
        length_init     = length_init,
        success_tol     = 3,
        n_candidates    = 5000,
        device          = device,
        dtype           = dtype,
    )

    # ==============================================================
    # ШАГ 4: CHECKPOINT / RESUME
    # ==============================================================
    checkpoint_file = f"checkpoint_{hostname_proc}.pkl"
    start_iter      = 1
    do_prior        = True

    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                state = pickle.load(f)

            X_obs = torch.tensor(state['X_obs'], dtype=dtype, device=device)
            Y_obs = torch.tensor(state['Y_obs'], dtype=dtype, device=device)

            turbo.length        = state['turbo_length']
            turbo.success_count = state['turbo_success']
            turbo.failure_count = state['turbo_failure']

            number_of_h_IC_lw    = state.get('n_h_IC_lw',    0)
            number_of_find_w_U   = state.get('n_find_w_U',   0)
            best_overall_target  = state.get('best_target',  -float('inf'))
            best_overall_Upsilon = state.get('best_Upsilon',  None)

            start_iter = state['iteration'] + 1
            do_prior   = False

            msg  = f"\n{'='*60}\n"
            msg += f"ВОЗОБНОВЛЕНИЕ С CHECKPOINT на {hostname_proc}\n"
            msg += f"{'='*60}\n"
            msg += f"  Файл:            {checkpoint_file}\n"
            msg += f"  Сохранён:        {state.get('timestamp','?')}\n"
            msg += f"  Итерация старта: {start_iter}/{n_iter}\n"
            msg += f"  Точек в истории: {len(X_obs)}\n"
            msg += f"  Лучший target:   {Y_obs.max().item():.6f}\n"
            msg += f"  TR length:       {turbo.length:.4f}\n"
            msg += f"  Вычислений halo: {number_of_h_IC_lw}\n"
            msg += f"{'='*60}"
            _write(msg)

            send_notification(
                f"Возобновление на {hostname_proc}\n"
                f"Итерация {start_iter}/{n_iter}\n"
                f"Лучший target: {Y_obs.max().item():.6f}",
                title=f"Galaxy {hostname_proc}: restart",
                priority='high',
                tags=['rocket']
            )

        except Exception as e:
            msg = (f"ПРЕДУПРЕЖДЕНИЕ: checkpoint повреждён ({e}),\n"
                   f"  стартуем с нуля")
            _write("# " + msg)
            start_iter = 1
            do_prior   = True
    else:
        _write("# Checkpoint не найден, запуск с нуля")

    # ==============================================================
    # ШАГ 5: АПРИОРНАЯ ТОЧКА
    # ==============================================================
    if do_prior:
        _write("\nДобавление априорной точки (лучшая из исторических):")
        _write(f"  Q={prior_params['Q']}, gh={prior_params['gh']}, "
               f"rh={prior_params['rh']}, rho0={prior_params['rho0']}")

        prior_pc = params_to_pca_fixed(prior_params, model_data)
        _write(f"  PCA-координаты априорной точки: {prior_pc}")
        _write("  Вычисление penalty для априорной точки...")

        prior_y = halo_IC_lib_weights_pca_fixed(
            prior_pc, model_data, bounds_original,
            densityStars, datasets, alphah, betah
        )

        X_obs = torch.cat([
            X_obs,
            torch.tensor([prior_pc],  dtype=dtype, device=device)
        ], dim=0)
        Y_obs = torch.cat([
            Y_obs,
            torch.tensor([[prior_y]], dtype=dtype, device=device)
        ], dim=0)

        _write(f"  Априорная точка добавлена. Penalty={-prior_y:.6f}")

    # ==============================================================
    # ШАГ 6: ОСНОВНОЙ ЦИКЛ TuRBO
    # ==============================================================
    _write("\n" + "=" * 60)
    _write(f"ЗАПУСК TuRBO-PCA: {n_iter} итераций")
    _write("=" * 60)

    y_best_prev = Y_obs.max().item()
    y_last      = Y_obs.max().item()

    new_points_params  = []   # буфер параметров текущего запуска
    new_points_penalty = []   # буфер penalty текущего запуска

    for iteration in range(start_iter, n_iter + 1):

        # --- Заголовок итерации ---
        y_last_source = (
            "из предыдущей итерации"
            if iteration > start_iter
            else "лучшее из истории"
        )
        _write(f"\n--- Итерация {iteration}/{n_iter} на {hostname_proc} ---")
        _write(f"    Лучший target     = {Y_obs.max().item():.6f}")
        _write(f"    Предыдущий target = {y_last:.6f}  ({y_last_source})")
        _write(f"    Размер TR         = {turbo.length:.4f}")

        # --- Предложение новой точки ---
        X_next    = turbo.suggest(X_obs, Y_obs)
        pc_coords = X_next[0].cpu().numpy()

        # --- Вычисление целевой функции ---
        y_next = halo_IC_lib_weights_pca_fixed(
            pc_coords, model_data, bounds_original,
            densityStars, datasets, alphah, betah
        )

        # --- Сохраняем в буфер ---
        new_params_i = pca_to_params_fixed(pc_coords, model_data, bounds_original)
        new_points_params.append(new_params_i)
        new_points_penalty.append(-y_next)   # penalty = -y_next

        # --- Обновление TuRBO и истории ---
        Y_next      = torch.tensor([[y_next]], dtype=dtype, device=device)
        turbo._update_tr(y_next, y_best_prev)
        y_best_prev = max(y_best_prev, y_next)
        X_obs       = torch.cat([X_obs, X_next],  dim=0)
        Y_obs       = torch.cat([Y_obs, Y_next],  dim=0)
        y_last      = y_next

        # --- Перезапуск TR если схлопнулась ---
        if turbo.length < turbo.length_min:
            _write("  [TuRBO-PCA] TR схлопнулась — перезапуск")
            turbo.length        = length_init
            turbo.success_count = 0
            turbo.failure_count = 0

        # --- Периодическое обновление PCA ---
        if (pca_update_interval is not None
                and iteration % pca_update_interval == 0
                and len(new_points_params) >= 5):

            _write(f"\n  [PCA] Обновление PCA на итерации {iteration}...")

            model_data, X_obs, Y_obs, turbo = _update_pca_model(
                model_data        = model_data,
                data_good         = data_good,
                new_params        = new_points_params,
                new_penalties     = new_points_penalty,
                bounds_original   = bounds_original,
                use_log_scale     = use_log_scale,
                expand_pca_bounds = expand_pca_bounds,
                X_obs             = X_obs,
                Y_obs             = Y_obs,
                turbo             = turbo,
                output_file       = output_file,
                dtype             = dtype,
                device            = device,
                # Параметры для чтения параллельных файлов:
                storage_patterns  = storage_patterns,
                host_patterns     = host_patterns,
                incl_filter       = incl,
                read_parallel     = True,
                current_suffix    = hostname_proc,
                penalty_cutoff    = cutoff_start,
            )

            # Сохраняем обновлённую модель
            with open(pkl_file, 'wb') as f:
                pickle.dump(model_data, f)

            _write(f"  [PCA] Обновление завершено. "
                   f"Точек в буфере: {len(new_points_params)}")

        # --- Checkpoint и уведомления ---
        if iteration % 3 == 0 or iteration == n_iter:
            save_checkpoint(X_obs, Y_obs, turbo, iteration)
            send_notification(
                f"Итерация {iteration}/{n_iter} на {hostname_proc}\n"
                f"incl={incl:.2f}\n"
                f"Лучший penalty:  {-Y_obs.max().item():.4f}\n"
                f"Текущий penalty: {-y_next:.4f}\n"
                f"TR length: {turbo.length:.4f}",
                title=f"Galaxy {hostname_proc}: Прогресс",
                priority='min',
                tags=['chart_with_upwards_trend']
            )

    # ==============================================================
    # ШАГ 7: РЕЗУЛЬТАТ
    # ==============================================================
    best_idx    = Y_obs.argmax().item()
    best_pc     = X_obs[best_idx].cpu().numpy()
    best_target = Y_obs[best_idx].item()
    best_params = pca_to_params_fixed(best_pc, model_data, bounds_original)

    _write("\n" + "=" * 60)
    _write("РЕЗУЛЬТАТ TuRBO-PCA:")
    _write("=" * 60)

    for line in [
        f"  PC-координаты: {best_pc}",
        f"  incl    = {incl:.2f}",
        f"  Q       = {best_params['Q']:.6f}",
        f"  gh      = {best_params['gh']:.6f}",
        f"  rh      = {best_params['rh']:.6f}",
        f"  rho0    = {best_params['rho0']:.6f}",
        f"  Upsilon = {best_overall_Upsilon:.6f}",
        f"  penalty = {-best_target:.6f}",
        f"\nВсего вычислений целевой функции: {number_of_h_IC_lw}",
        f"Всего оптимизаций Upsilon: {number_of_find_w_U}",
        "=" * 60,
    ]:
        _write(line)

    _write(f"# End: {datetime.datetime.now()}")

    return best_params, best_overall_Upsilon, -best_target

def compare_good_vs_acceptable(data, cutoff1=0.60, cutoff2=0.75,
                                incl_filter=None, diag_file=None):
    """
    Сравнение хороших и приемлемых точек.
    incl_filter: если не None — фильтровать по наклонению (столбец 0)
    diag_file:   если не None — дублировать вывод в файл
    """
    # --- Фильтр по наклонению ---
    if incl_filter is not None:
        mask = numpy.abs(data[:, 0] - incl_filter) < 0.01
        data = data[mask]
        incl_msg = f" (incl={incl_filter})"
    else:
        incl_msg = " (все наклонения)"

    good       = data[data[:, 6] <= cutoff1]
    acceptable = data[(data[:, 6] > cutoff1) & (data[:, 6] <= cutoff2)]

    lines = []
    lines.append(f"\nСравнение хороших и приемлемых точек{incl_msg}:")
    lines.append(f"  Хороших     (≤{cutoff1}):          {len(good)}")
    lines.append(f"  Приемлемых  ({cutoff1}–{cutoff2}): {len(acceptable)}")

    if len(good) == 0:
        lines.append("  ВНИМАНИЕ: нет хороших точек для сравнения.")
        for line in lines:
            print(line)
        if diag_file:
            with open(diag_file, 'a') as f:
                f.write('\n'.join(lines) + '\n')
        return

    lines.append(f"\n  {'Параметр':10s} {'Хорошие':>12s} {'Приемлемые':>12s} {'Разница':>12s}")
    lines.append(f"  {'-'*48}")

    names = ['Q', 'gh', 'rh', 'rho0', 'Upsilon']
    for i, name in enumerate(names):
        m1 = good[:, i + 1].mean()
        if len(acceptable) > 0:
            m2       = acceptable[:, i + 1].mean()
            diff_str = f"{abs(m1 - m2):12.4f}"
            m2_str   = f"{m2:12.4f}"
        else:
            m2_str   = f"{'—':>12s}"
            diff_str = f"{'—':>12s}"
        lines.append(f"  {name:10s} {m1:12.4f} {m2_str} {diff_str}")

    for line in lines:
        print(line)
    if diag_file:
        with open(diag_file, 'a') as f:
            f.write('\n'.join(lines) + '\n')


# ==============================================================
#  ДИАГНОСТИКА: Сравнение PCA-пространств
# ==============================================================
def diagnose_pca_space(storage_patterns, host_patterns,
                       cutoff_start=0.60, incl_filter=None):

    diag_file = f"diagnose_pca_space_{hostname_proc}.txt"

    if incl_filter is None:
        incl_filter = incl
    incl_msg = f"incl={incl_filter}"

    def _write(text):
        print(text)
        with open(diag_file, 'a') as f:
            f.write(text + '\n')

    _write("\n" + "=" * 70)
    _write(f"ДИАГНОСТИКА PCA-ПРОСТРАНСТВА  ({incl_msg})")
    _write(f"Файл диагностики: {diag_file}")
    _write("=" * 70)

    # --- Загрузка данных с force_update ---
    _write("\nЗагрузка данных...")
    load_from_yadisk(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        force_update     = True,
    )

    data, file_counts = load_fresh_data_from_files(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        incl_filter      = incl_filter,
        use_log_scale    = False,
        exclude_suffix   = None,
        return_full      = True,
    )

    # --- Отчёт по файлам ---
    _write(f"\nПрочитано файлов: {len(file_counts)}")
    for fname, cnt in file_counts.items():
        _write(f"  {fname}: {cnt} строк")

    if data is None or len(data) == 0:
        _write(f"Нет данных для incl={incl_filter}. Диагностика пропущена.")
        return None

    _write(f"Загружено строк (incl={incl_filter}): {len(data)}")

    # --- Минимум точек для диагностики ---
    MIN_POINTS_DIAG = 5   # меньше этого — диагностика бессмысленна

    # --- Выбор cutoff ---
    penalty_cutoff = cutoff_start
    mask_good      = data[:, 6] <= penalty_cutoff
    data_good      = data[mask_good]

    if len(data_good) == 0:
        _write(f"\nВНИМАНИЕ: Нет точек с penalty <= {penalty_cutoff}.")
        adaptive_cutoff = numpy.percentile(data[:, 6], 30)
        _write(f"Используем адаптивный cutoff (лучшие 30%): {adaptive_cutoff:.4f}")
        penalty_cutoff = adaptive_cutoff
        mask_good      = data[:, 6] <= penalty_cutoff
        data_good      = data[mask_good]

    if len(data_good) == 0:
        _write("Ошибка: не удалось отобрать точки. Пропускаем диагностику.")
        return None

    _write(f"\nВсего точек ({incl_msg}):              {len(data)}")
    _write(f"Точек с penalty <= {penalty_cutoff:.4f}: {len(data_good)}")

    # --- Распределение параметров ---
    _write("\nРаспределение параметров (хорошие точки):")
    _write("-" * 50)
    param_names = ['Q', 'gh', 'rh', 'rho0', 'Upsilon']
    for i, name in enumerate(param_names):
        vals = data_good[:, i + 1]
        _write(f"  {name:8s}: min={vals.min():8.3f}, max={vals.max():8.3f}, "
               f"mean={vals.mean():8.3f}, std={vals.std():8.3f}")

    # --- Проверка достаточности точек для PCA и корреляций ---
    if len(data_good) < MIN_POINTS_DIAG:
        _write(f"\nВНИМАНИЕ: Мало точек для полной диагностики "
               f"({len(data_good)} < {MIN_POINTS_DIAG}).")
        _write("Пропускаем корреляционную матрицу и PCA-анализ.")
        _write("Доступна только статистика параметров (см. выше).")
        _write("\nРекомендация: запустите bootstrap или добавьте данные.")

        # Минимальная диагностика: статистика penalty
        _write("\nСтатистика penalty:")
        _write(f"  min={data[:, 6].min():.4f}, "
               f"max={data[:, 6].max():.4f}, "
               f"mean={data[:, 6].mean():.4f}")
        _write(f"\nКоличество точек по порогам penalty:")
        cutoffs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                   1.0,  1.5,  2.0,  3.0]
        for cutoff in cutoffs:
            n_pts = numpy.sum(data[:, 6] <= cutoff)
            _write(f"  penalty <= {cutoff:.2f}: "
                   f"{n_pts:5d} точек ({100 * n_pts / len(data):.1f}%)")

        sync_to_yadisk()
        return data_good

    # --- Корреляционная матрица (только если точек достаточно) ---
    _write("\nКорреляционная матрица:")
    _write("-" * 50)
    X    = data_good[:, 1:6]
    
    # Защита от вырожденных столбцов (std=0)
    stds = X.std(axis=0)
    if numpy.any(stds == 0):
        _write(f"  ВНИМАНИЕ: вырожденные столбцы (std=0): "
               f"{[param_names[i] for i, s in enumerate(stds) if s == 0]}")
        _write("  Корреляционная матрица не вычисляется.")
    else:
        corr = numpy.corrcoef(X.T)
        _write("        Q      gh      rh    rho0     Ups")
        for i, name in enumerate(param_names):
            row_str = (f"{name:5s} " +
                       "".join(f"{corr[i,j]:7.3f} " for j in range(5)))
            _write(row_str)

    # --- PCA-анализ (только если точек достаточно) ---
    _write("\nСравнение PCA с Upsilon и без:")
    _write("-" * 50)

    # Максимально возможное число компонент
    n_comp_max_with    = min(4, len(data_good) - 1, 5)   # 5 параметров с Ups
    n_comp_max_without = min(4, len(data_good) - 1, 4)   # 4 параметра без Ups

    if n_comp_max_with < 1 or n_comp_max_without < 1:
        _write(f"  Недостаточно точек для PCA "
               f"(нужно ≥ 2, есть {len(data_good)}). Пропускаем.")
    else:
        X_with_ups = data_good[:, 1:6]
        scaler1    = StandardScaler()
        pca1       = PCA(n_components=n_comp_max_with)
        try:
            pca1.fit(scaler1.fit_transform(X_with_ups))
            _write(f"С Upsilon (n_components={n_comp_max_with}):\n"
                   f"  Explained variance: "
                   f"{pca1.explained_variance_ratio_}\n"
                   f"  Cumulative:         "
                   f"{numpy.cumsum(pca1.explained_variance_ratio_)}")
        except Exception as e:
            _write(f"  PCA с Upsilon: ошибка — {e}")

        X_no_ups = data_good[:, 1:5]
        scaler2  = StandardScaler()
        pca2     = PCA(n_components=n_comp_max_without)
        try:
            pca2.fit(scaler2.fit_transform(X_no_ups))
            _write(f"\nБез Upsilon (n_components={n_comp_max_without}):\n"
                   f"  Explained variance: "
                   f"{pca2.explained_variance_ratio_}\n"
                   f"  Cumulative:         "
                   f"{numpy.cumsum(pca2.explained_variance_ratio_)}")
        except Exception as e:
            _write(f"  PCA без Upsilon: ошибка — {e}")

        # --- PCA с логарифмическим масштабированием ---
        _write("\nС логарифмическим масштабированием:")
        X_log       = X_no_ups.copy()
        # Защита от log(0)
        mask_pos    = (X_log[:, 2] > 0) & (X_log[:, 3] > 0)
        if numpy.sum(mask_pos) < 2:
            _write("  Недостаточно точек с rh>0 и rho0>0. Пропускаем.")
        else:
            X_log_valid       = X_log[mask_pos].copy()
            X_log_valid[:, 2] = numpy.log10(X_log_valid[:, 2])
            X_log_valid[:, 3] = numpy.log10(X_log_valid[:, 3])
            n_comp_log        = min(n_comp_max_without,
                                    len(X_log_valid) - 1)
            scaler3           = StandardScaler()
            pca3              = PCA(n_components=n_comp_log)
            try:
                pca3.fit(scaler3.fit_transform(X_log_valid))
                _write(f"  n_components={n_comp_log}:\n"
                       f"  Explained variance: "
                       f"{pca3.explained_variance_ratio_}\n"
                       f"  Cumulative:         "
                       f"{numpy.cumsum(pca3.explained_variance_ratio_)}")
            except Exception as e:
                _write(f"  PCA с log: ошибка — {e}")

    # --- Проверка обратного преобразования ---
    _write("\nПроверка обратного преобразования:")
    _write("-" * 50)
    idx        = numpy.argmin(data_good[:, 6])
    best_point = data_good[idx]
    _write(f"Лучшая точка из данных:\n"
           f"  Q={best_point[1]:.4f}, gh={best_point[2]:.4f}, "
           f"rh={best_point[3]:.4f}, rho0={best_point[4]:.4f}, "
           f"Ups={best_point[5]:.4f}, pen={best_point[6]:.4f}")

    if len(data_good) >= 2:
        params     = {'Q': best_point[1], 'gh': best_point[2],
                      'rh': best_point[3], 'rho0': best_point[4]}
        X_test     = numpy.array([[params['Q'], params['gh'],
                                   numpy.log10(max(params['rh'], 1e-10)),
                                   numpy.log10(max(params['rho0'], 1e-10))]])

        try:
            X_t_scaled    = scaler3.transform(X_test)
            pc_coords     = pca3.transform(X_t_scaled)
            _write(f"\nPCA-координаты: {pc_coords[0]}")

            X_back_scaled = pca3.inverse_transform(pc_coords)
            X_back        = scaler3.inverse_transform(X_back_scaled)
            _write(f"Обратное преобразование:\n"
                   f"  Q={X_back[0,0]:.4f}, gh={X_back[0,1]:.4f}, "
                   f"rh={10**X_back[0,2]:.4f}, rho0={10**X_back[0,3]:.4f}")
            _write(f"\nОшибка обратного преобразования:\n"
                   f"  Q:    {abs(X_back[0,0] - params['Q']):.6f}\n"
                   f"  gh:   {abs(X_back[0,1] - params['gh']):.6f}\n"
                   f"  rh:   {abs(10**X_back[0,2] - params['rh']):.6f}\n"
                   f"  rho0: {abs(10**X_back[0,3] - params['rho0']):.6f}")
        except Exception as e:
            _write(f"  Проверка обратного преобразования: ошибка — {e}")
    else:
        _write("  Недостаточно точек для проверки обратного преобразования.")

    # --- Статистика по cutoff ---
    _write("\nКоличество точек по порогам penalty:")
    cutoffs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
               1.0,  1.5,  2.0,  3.0]
    for cutoff in cutoffs:
        n_pts = numpy.sum(data[:, 6] <= cutoff)
        _write(f"  penalty <= {cutoff:.2f}: "
               f"{n_pts:5d} точек ({100 * n_pts / len(data):.1f}%)")

    # --- Сравнение хороших и приемлемых ---
    if len(data_good) >= MIN_POINTS_DIAG:
        compare_good_vs_acceptable(
            data,
            cutoff1     = cutoff_start,
            cutoff2     = min(cutoff_start * 1.25, cutoff_start + 0.15),
            incl_filter = incl_filter,
            diag_file   = diag_file
        )

    sync_to_yadisk()
    return data_good

# ==============================================================
#  ЗАПУСК
# ==============================================================


# print("Запуск диагностики...")
# diagnose_pca_space(files, cutoff_start=cutoff_start)

# best_params, best_Upsilon, best_penalty = run_pca_optimization(
#             files=files,
#             cutoff_start=cutoff_start,
#             n_components=3,
#             n_iter=40,
#             length_init=0.6,
#             use_log_scale=True,resume= False
#         ) 

# ============================================================
# ЗАПУСК (в самом конце скрипта)
# ============================================================
if __name__ == '__main__':
    
    send_notification(
    f"Старт расчёта на {hostname_proc}\nincl={incl}\nresume={do_resume}",
    title=f"Galaxy {hostname_proc}: Старт",
    priority='high',
    tags=['rocket']
    )
    
    # Watchdog в фоне (опционально)
    # subprocess.Popen(['bash', 'watchdog.sh'])
    
    try:
        # Диагностика: читает все файлы через load_fresh_data_from_files
        diagnose_pca_space(
            storage_patterns = storage_patterns,
            host_patterns    = host_patterns,
            cutoff_start     = cutoff_start,
            incl_filter      = incl,
        )
        
        best_params, best_Upsilon, best_penalty = run_pca_optimization(
            storage_patterns = storage_patterns,
            host_patterns    = host_patterns,
        fallback_patterns = fallback_patterns, 
            cutoff_start     = cutoff_start,
            n_components     = 3,
            n_iter           = 40,
            length_init      = 0.6,
            use_log_scale    = True,
            resume           = do_resume,
        )
        
        finalize(best_params, best_Upsilon, best_penalty)
        
    except Exception as e:
        # При любой ошибке — сохранить что есть и уведомить
        send_notification(
            f"ОШИБКА на {hostname_proc}:\n{str(e)[:300]}",
            title=f"Galaxy {hostname_proc}: ОШИБКА",
            priority='urgent',    # срочное, со звуком
            tags=['warning', 'rotating_light']
            )
        print(f"Критическая ошибка: {e}")
        
        # Попытка сохранить хоть что-то
        sync_to_yadisk(remote_dir='galaxy_results_emergency')
        
        # Выключение через 2 минуты
        #subprocess.Popen(['sudo', 'shutdown', '-h', '+2'])
        raise
