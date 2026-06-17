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
import time

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
parser.add_argument('--n_threads', type=int, default=None,
                    help='Число потоков OpenMP для AGAMA')
parser.add_argument('--init-from-pa468', dest='seed_from_pa468',
                    action='store_true',
                    help='Использовать архивные файлы 4UpsBoTorch_PCA_PA46.8_Sersic_* '
                         'как источник НАЧАЛЬНЫХ точек (penalty пересчитывается '
                         'с корректной геометрией). Penalty из этих файлов НЕ '
                         'попадают в PCA-модель напрямую.')
args = parser.parse_args()

if args.n_threads is not None:
    # Ограничиваем OpenMP ДО импорта agama
    os.environ['OMP_NUM_THREADS']      = str(args.n_threads)
    os.environ['MKL_NUM_THREADS']      = str(args.n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.n_threads)

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

# Архивные файлы со СТАРОЙ (неверной) геометрией posang=46.8, q_ap=0.7.
# Используются ТОЛЬКО как источник кандидатов-параметров для начальных точек
# (penalty пересчитывается с корректной геометрией). Эти паттерны намеренно
# отделены от storage_patterns/host_patterns, чтобы устаревшие penalty никогда
# не попадали в PCA-модель напрямую. Активируется флагом --init-from-pa468.
seed_patterns = [
    "4UpsBoTorch_PCA_PA46.8_Sersic_*.txt",
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
posang    = 42.3   # Sersic, Wang et al. 2019 (https://doi.org/10.3847/1538-4357/ab31a9), 42.3 +/- 0.2 deg; old 46.8 = Battaglia et al. 2006
gamma2 =  (posang - 90.0) * numpy.pi/180
q_ap      = 1 - 0.31   # 1 - Ellipticity, Wang et al. 2019 (Ellipticity = 0.31 +/- 0.002); old 0.7

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

# ==============================================================
#  РЕЗЕРВИРОВАНИЕ РАСЧЁТНЫХ ТОЧЕК (превентивная защита от дублей)
# ==============================================================
# Параллельные процессы одной VM делят общий /workspace. Перед дорогой
# оценкой penalty процесс «резервирует» точку в стабильном физическом
# пространстве параметров (Q, gh, rh, rho0) — единственном, общем для всех
# процессов (PCA у каждого своё и дрейфует). Резервация = маленький файл в
# каталоге reservations_i<incl>/. Активность определяется по mtime + TTL:
# если процесс умер во время расчёта, его резервация протухает и точку снова
# можно занять. Дедупликации РЕЗУЛЬТАТОВ не происходит — это только защита
# от одновременного пересчёта одной точки (AGENTS §10: превентивно).
_RESV_PARAM_NAMES = ['Q', 'gh', 'rh', 'rho0']

def reservation_dir(incl):
    d = f"reservations_i{incl}"
    os.makedirs(d, exist_ok=True)
    return d

def reservation_dist_norm(p1, p2, bounds_original):
    """Евклидово расстояние между точками в нормированном по bounds
    пространстве параметров (каждая ось масштабируется своим диапазоном)."""
    s = 0.0
    for name in _RESV_PARAM_NAMES:
        lo, hi = bounds_original[name]
        rng = (hi - lo) or 1.0
        s += ((p1[name] - p2[name]) / rng) ** 2
    return s ** 0.5

def read_active_reservations(resv_dir, ttl_sec, exclude_suffix=None):
    """Возвращает список активных (не протухших) резерваций других процессов:
    [{'params': {...}, 'suffix': str, 'mtime': float, 'path': str}, ...]."""
    now = time.time()
    out = []
    for fp in glob.glob(os.path.join(resv_dir, '*.resv')):
        try:
            mt = os.path.getmtime(fp)
        except OSError:
            continue
        if now - mt > ttl_sec:
            continue
        try:
            with open(fp) as f:
                parts = f.read().split()
        except OSError:
            continue
        if len(parts) < 5:
            continue
        sfx = parts[4]
        if exclude_suffix is not None and sfx == exclude_suffix:
            continue
        try:
            params = {
                'Q':    float(parts[0]),
                'gh':   float(parts[1]),
                'rh':   float(parts[2]),
                'rho0': float(parts[3]),
            }
        except ValueError:
            continue
        out.append({'params': params, 'suffix': sfx,
                    'mtime': mt, 'path': fp})
    return out

def reserve_point(resv_dir, suffix, params):
    """Атомарно создаёт файл-резервацию. Имя уникально (suffix+pid+время),
    поэтому коллизий имён нет. Возвращает путь к файлу резервации."""
    fname = f"{suffix}_{os.getpid()}_{int(time.time() * 1e6)}.resv"
    fp = os.path.join(resv_dir, fname)
    fd = os.open(fp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, (
            f"{params['Q']:.15g} {params['gh']:.15g} "
            f"{params['rh']:.15g} {params['rho0']:.15g} {suffix}\n"
        ).encode())
    finally:
        os.close(fd)
    return fp

def release_reservation(fp):
    if not fp:
        return
    try:
        os.remove(fp)
    except OSError:
        pass

massSt    = 14.0
scaleRst  =  sc*16.4/60


Upsilon_start = [1.0]
Upsilon_lower = 0.1
Upsilon_upper = 1.6

Sersic_m  = 0.80

NumStars  = 1000000
intTime   = 100.
regul     = 1.0
ghorder   = 6
degree    = 2
symmetry  = 't'
usehist   = 0
variant   = 'Hist' if usehist else 'GH'
# RNG policy:
#  - torch (BoTorch acquisition) gets an INDEPENDENT per-process seed, drawn from
#    OS entropy mixed with the process id (hostname_proc) and logged below. This
#    decorrelates the candidate proposals of parallel workers (helps avoid
#    duplicate evaluations — see Goal 0).
#  - the numpy generator is NOT seeded per-process here; the only numpy-random
#    consumer that must be reproducible (the GH-moment error bootstrap below) is
#    explicitly seeded with seed=42 right before it, so the observation-error
#    realization is IDENTICAL in every process (comparable penalties).
#  - NB: neither seed affects AGAMA orbit-IC sampling: densityStars.sample() uses
#    AGAMA's own internal RNG (agama.setRandomSeed, not called) → orbit libraries
#    are identical across processes regardless of these seeds.
_seed_seq    = numpy.random.SeedSequence(entropy=None,
                                         spawn_key=(abs(hash(hostname_proc)) % (2**32),))
process_seed = int(_seed_seq.generate_state(1)[0])
torch.manual_seed(process_seed)
# Per-process RNG for OPTIMIZER SEEDING (initial points only, not physics):
# distinct per process AND per host (process_seed mixes in hostname_proc), so
# parallel workers pick DIFFERENT initial points for a not-yet-explored incl
# instead of recomputing identical ones (Goal 0: avoid duplicate calculations).
proc_rng = numpy.random.default_rng(process_seed)
print(f"[seed] process={hostname_proc} torch seed={process_seed} "
      f"(numpy GH-bootstrap fixed at 42; AGAMA orbit RNG не затронут)")
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
# Fixed seed=42: the observation-error realization (Monte-Carlo resampling of the
# measured velocities) must be IDENTICAL in every parallel process, so that the
# resulting GH-moment errors — and hence penalties — are comparable between
# workers. (This is the only numpy-random consumer here; LHS uses its own RNG,
# torch/BoTorch is decorrelated per-process — see RNG policy comment above.)
numpy.random.seed(42)
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
            tags=['warning', 'rotating_light'] )
            return
        remote_files = [f.strip() for f in result.stdout.splitlines()]
    except Exception as e:
        msg = f"  [yadisk] Недоступен: {e}"
        send_notification(msg,
        title=f"Galaxy {hostname_proc}: Файлы не прочитаны с яндекса",
        priority='urgent',
        tags=['warning', 'rotating_light'] )
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
    
    # Скачиваем файлы по паттернам storage_patterns
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
                                    regul=1.,
                                    # НОВЫЙ параметр: прямые параметры без PCA
                                    direct_params=None):
    global best_overall_Upsilon, best_overall_target, number_of_h_IC_lw, number_of_find_w_U, hostname_proc, UpsFile
    
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
#        pc_coords_log = numpy.array([params['Q'], params['gh'],
#                                      params['rh'], params['rho0']])
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
 
    Q = params['Q']
    gh = params['gh']
    rh = params['rh']
    rho0 = params['rho0']
    
#    print(f"PCA coords: {pc_coords}")
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
    
    # Времена solveOpt по каждой пробе Upsilon. Копятся в памяти и пишутся
    # на диск ОДИН раз после минимизации (как и история Upsilon), чтобы не
    # дёргать диск на каждой итерации solveOpt.
    solveopt_times = []
    
    def find_weights_Ups(Upsilon):
        global number_of_find_w_U
        _ups_val = float(numpy.ravel(numpy.asarray(Upsilon))[0])
        try:
            matrix = [d.getOrbitMatrix(m, Upsilon).T for d, m in zip(datasets, matrices)]
            _t_solve = time.perf_counter()
            weights = agama.solveOpt(matrix=matrix, rhs=rhs, rpenq=pen_cons, xpenq=pen_reg) * mult
            solveopt_times.append((_ups_val, time.perf_counter() - _t_solve))
            superpositions = [weights.dot(m) for m in matrices]
            penalties = [d.getPenalty(s, Upsilon) for d, s in zip(datasets, superpositions)]
            pen = numpy.sum(penalties[1])
            number_of_find_w_U += 1
            return pen
        except Exception as e:
            solveopt_times.append((_ups_val, float('nan')))
            msg = f"Error with parameters: {params}, Upsilon: {Upsilon},\n Error: {e}"
            print(msg)
            with open(UpsFile, 'a') as f:
                f.write("#" + msg)
            return 1e6
    
    logger = FunctionLogger(find_weights_Ups)
    min_penalty_Ups = minimize_scalar(
        logger,
        bounds=(Upsilon_lower, Upsilon_upper),
        method='bounded',
        options={'xatol': 1e-3, 'maxiter': 50}
    )
    
    min_pen = float(min_penalty_Ups.fun)
    min_Ups = float(min_penalty_Ups.x)
    
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
        # --- Времена solveOpt (батч-запись, всё в комментариях) ---
        f.write("# solveOpt times (s) per Upsilon probe:\n")
        for i, (ups_val, dt) in enumerate(solveopt_times):
            f.write(f"# T {i:4d}: Upsilon={ups_val:.15f} solveOpt_s={dt:.6f}\n")
        _valid = numpy.array([dt for _, dt in solveopt_times if dt == dt])
        if _valid.size:
            f.write(f"# solveOpt summary: n={_valid.size} "
                    f"total_s={_valid.sum():.6f} mean_s={_valid.mean():.6f} "
                    f"median_s={numpy.median(_valid):.6f} "
                    f"min_s={_valid.min():.6f} max_s={_valid.max():.6f}\n")
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
                                 strategy='best_diverse',
                                 rng=None):
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

    if rng is not None and len(pool) > n_bootstrap:
        # Per-process decorrelation (Goal 0): each parallel worker draws a
        # DIFFERENT subset from the good pool (weighted toward lower penalty),
        # so processes don't recompute identical initial points. Falls back to
        # the deterministic strategies below when rng is None or the pool is
        # too small to differentiate workers.
        ranks    = numpy.arange(len(pool))
        weights  = numpy.exp(-ranks / max(1.0, len(pool) / 3.0))
        weights /= weights.sum()
        sel_idx  = numpy.sort(rng.choice(len(pool), size=n_bootstrap,
                                         replace=False, p=weights))
        selected = pool[sel_idx]

    elif strategy == 'best':
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
        rng=None,
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
        rng                  = rng,
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


def seed_points_from_patterns(seed_patterns, target_incl, bounds_original,
                              n_seed=12, penalty_cutoff_frac=0.5,
                              strategy='best_diverse', rng=None):
    """
    Формирует начальные точки из АРХИВНЫХ файлов (seed_patterns, например
    PA46.8) для того же target_incl, ПЕРЕСЧИТЫВАЯ penalty с текущей
    (корректной) геометрией.

    Отличие от bootstrap_initial_points_from_nearest_incl:
      - читает строки на ТОМ ЖЕ наклонении target_incl (а не ближайшем),
      - источник — отдельные seed_patterns (не storage/host), поэтому
        устаревшие penalty не попадают в PCA-модель напрямую;
        в модель идут только пересчитанные значения.

    Возвращает: bootstrap_results — list of {'params','penalty','pc'}.
    """
    print("\n" + "=" * 60)
    print(f"SEED из архива (PA46.8) для incl={target_incl:.2f}°")
    print("=" * 60)

    # --- Шаг 1: собрать локальные архивные файлы (без скачивания) ---
    all_files = []
    for pattern in seed_patterns:
        for f in glob.glob(pattern):
            if f not in all_files:
                all_files.append(f)

    if not all_files:
        print("  [seed-pa468] Архивные файлы не найдены.")
        return []

    print(f"  [seed-pa468] Файлов: {len(all_files)}")

    # --- Шаг 2: прочитать строки на ТОМ ЖЕ наклонении ---
    rows = []
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
                    except ValueError:
                        continue
                    if abs(row[0] - target_incl) > 0.01:
                        continue
                    if row[3] <= 0 or row[4] <= 0:
                        continue
                    if row[6] >= 1e5:
                        continue
                    rows.append(row)
        except FileNotFoundError:
            pass

    if not rows:
        print(f"  [seed-pa468] Нет точек на incl={target_incl:.2f}° в архиве.")
        return []

    data_seed = numpy.array(rows)
    print(f"  [seed-pa468] Найдено архивных точек: {len(data_seed)}")

    # --- Шаг 3: выбрать кандидатов (ранжируем по СТАРОМУ penalty) ---
    candidates = select_bootstrap_candidates(
        data_seed,
        n_bootstrap         = n_seed,
        penalty_cutoff_frac = penalty_cutoff_frac,
        strategy            = strategy,
        rng                 = rng,
    )
    print(f"  [seed-pa468] Выбрано кандидатов: {len(candidates)}")

    # --- Шаг 4: пересчёт penalty с текущей геометрией ---
    seed_results = []
    n_success = 0
    for i, params in enumerate(candidates):
        pc_coords = _params_to_dummy_pc(params, None, bounds_original)
        try:
            y_val = halo_IC_lib_weights_pca_fixed(
                pc_coords,
                None,                       # модели ещё нет
                bounds_original,
                densityStars, datasets, alphah, betah,
                direct_params=params,
            )
            penalty = -y_val
            if numpy.isfinite(penalty) and penalty < 1e5:
                seed_results.append({
                    'params':  params,
                    'penalty': penalty,
                    'pc':      pc_coords,
                })
                n_success += 1
                print(f"    [{i+1}/{len(candidates)}] ✓ penalty={penalty:.6f}")
            else:
                print(f"    [{i+1}/{len(candidates)}] ✗ penalty={penalty:.6f}")
        except Exception as e:
            print(f"    [{i+1}/{len(candidates)}] ✗ ошибка: {e}")

    print(f"  [seed-pa468] Пересчитано успешно: {n_success}/{len(candidates)}")
    return seed_results


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
    
    def suggest(self, X_obs, Y_obs, X_pending=None):
        best_idx = Y_obs.argmax()
        x_center = X_obs[best_idx]
        tr_bounds = self._tr_bounds(x_center)
        
        model = self._fit_gp(X_obs, Y_obs)
        model.eval()
        
        # X_pending — точки, которые другие процессы уже считают (резервации).
        # q-acquisition штатно уводит предложение в сторону от них.
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_obs,
            prune_baseline=True,
            X_pending=X_pending,
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

def load_fresh_data_from_files(storage_patterns,host_patterns, incl_filter, 
                                use_log_scale=True,
                                exclude_suffix=hostname_proc,
                                return_full=False):
    """
    Читает все доступные файлы результатов прямо сейчас,
    включая файлы параллельных процессов.
    
    files:          базовые файлы (исторические)
    incl_filter:    фильтр по наклонению
    use_log_scale:  логарифмировать rh и rho0
    exclude_suffix: суффикс файлов текущего процесса 
                    (чтобы не читать дважды, если уже в model_data)
    
    Возвращает: X_raw, penalties — все доступные точки
    """
    # --- Шаг 1: скачиваем свежие файлы с Яндекс.Диска ---
    load_from_yadisk(storage_patterns,host_patterns)
    
    # --- Шаг 2: собираем список файлов на диске ---
    all_files = []
    
    for pattern in host_patterns + storage_patterns:
        found = glob.glob(pattern)
        for f in found:
            if f not in all_files:
                # Пропускаем файл текущего процесса если указано
                if exclude_suffix and exclude_suffix in f:
                    continue
                all_files.append(f)
    
    print(f"  [load_fresh] Файлов для чтения: {len(all_files)}")
    for f in all_files:
        print(f"    {f}")
    
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
                        # Фильтр по наклонению
                        if abs(row[0] - incl_filter) > 0.01:
                            continue
                        # Фильтр на корректность
                        if row[3] <= 0 or row[4] <= 0:
                            continue
                        if row[6] >= 1e5:   # пропускаем failed runs
                            continue
                        raw.append(row)
                        count += 1
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
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

    # --- Возврат ---
    if return_full:
        # Полный массив: все 7 столбцов [incl, Q, gh, rh, rho0, Ups, penalty]
        return data, file_counts
    else:
        # Только параметры и penalty
        X_raw     = data[:, 1:5]   # Q, gh, rh, rho0
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

    # Latin Hypercube Sampling.
    # Per-process RNG (proc_rng) instead of a fixed seed → each parallel worker
    # generates a DIFFERENT random fallback set (Goal 0: no duplicate init points).
    rng      = proc_rng
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
    host_patterns=None,      # паттерны файлов своего сервера
    n_components=4,
    n_iter=30,
    target_fraction=0.3,
    cutoff_start=0.6,
    length_init=0.8,
    use_log_scale=True,
    expand_pca_bounds=2.5,
    output_file=None,
    resume=True,
    pca_update_interval=15,
    seed_patterns=None,      # архивные паттерны (PA46.8) для начальных точек
    seed_from_pa468=False,   # включить seed из архива (penalty пересчитывается)
    reserve_points=True,     # резервировать точки (защита от дублей внутри VM)
    reserve_eps=0.02,        # порог «одинаковости» в нормированном простр-ве
    reserve_ttl_sec=7200,    # TTL резервации (протухание после смерти процесса)
    reserve_max_retries=8    # сколько раз возмущать кандидат при коллизии
):
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

    # --- Каталог резерваций (общий для процессов одной VM) ---
    resv_dir = reservation_dir(incl) if reserve_points else None
    if reserve_points:
        _write(f"\nРезервирование точек включено: dir={resv_dir}, "
               f"eps={reserve_eps}, ttl={reserve_ttl_sec}s")

    # ==============================================================
    # ШАГ 1: ЗАГРУЗКА ДАННЫХ через load_fresh_data_from_files
    # ==============================================================
    _write("\n" + "=" * 60)
    _write("ЗАГРУЗКА ДАННЫХ И РАСЧЁТ АДАПТИВНОГО CUTOFF")
    _write("=" * 60)

    MIN_POINTS_FOR_PCA = 10  # минимум точек для построения PCA

    # Читаем ВСЕ файлы включая свой (exclude_suffix=None),
    # так как это начальная загрузка — нужна полная история
    data, file_counts = load_fresh_data_from_files(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        incl_filter      = incl,
        use_log_scale    = False,      # сырые данные, логарифм применим позже
        exclude_suffix   = None,       # читаем все файлы включая свой
        return_full      = True,       # нужен полный массив [7 столбцов]
    )

    # --- Отчёт по загруженным файлам ---
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

        # --- Источник 0 (опц.): SEED из архива PA46.8 на том же incl ---
        # Penalty пересчитывается с корректной геометрией; устаревшие penalty
        # в PCA-модель не попадают.
        if seed_from_pa468 and seed_patterns:
            _write("Источник начальных точек: архив PA46.8 "
                   "(penalty пересчитывается)...")
            n_seed = (12 if n_have == 0
                      else max(8, MIN_POINTS_FOR_PCA - n_have + 3))
            seed_results = seed_points_from_patterns(
                seed_patterns       = seed_patterns,
                target_incl         = incl,
                bounds_original     = bounds_original,
                n_seed              = n_seed,
                penalty_cutoff_frac = 0.5,
                strategy            = 'best_diverse',
                rng                 = proc_rng,
            )
            if seed_results:
                bootstrap_results = seed_results
                seed_rows = numpy.array([
                    [incl,
                     r['params']['Q'],  r['params']['gh'],
                     r['params']['rh'], r['params']['rho0'],
                     0.0,               r['penalty']]
                    for r in seed_results
                ])
                if data is not None and len(data) > 0:
                    data = numpy.vstack([data, seed_rows])
                else:
                    data = seed_rows
                _write(f"  SEED PA46.8 дал {len(seed_results)} точек, "
                       f"всего {len(data)}")
                n_have          = len(data) if data is not None else 0
                data_sufficient = (n_have >= MIN_POINTS_FOR_PCA)

    if not data_sufficient:
        _write("Запускаем bootstrap из ближайшего наклонения...")

        send_notification(
            f"Bootstrap для incl={incl:.2f}°\n"
            f"Данных: {n_have} (нужно ≥ {MIN_POINTS_FOR_PCA})\n"
            f"Ищем ближайшее наклонение...",
            title=f"Galaxy {hostname_proc}: Bootstrap",
            priority='default',
            tags=['hourglass_flowing_sand']
        )

        # Сколько точек пересчитать:
        # если данных совсем нет — берём 12,
        # иначе добираем до MIN_POINTS_FOR_PCA + 3 запасных
        n_boot = (12 if n_have == 0
                  else max(8, MIN_POINTS_FOR_PCA - n_have + 3))

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
                rng                 = proc_rng,
            )

        if bootstrap_results:
            _write(f"\n  Bootstrap дал {len(bootstrap_results)} точек "
                   f"(из incl={nearest_incl_used:.2f}°, "
                   f"dist={dist_used:.1f}°)")

            # Формируем строки в формате [incl, Q, gh, rh, rho0, Ups, penalty]
            boot_rows = numpy.array([
                [incl,
                 r['params']['Q'],  r['params']['gh'],
                 r['params']['rh'], r['params']['rho0'],
                 0.0,               r['penalty']]
                for r in bootstrap_results
            ])

            # Объединяем с имеющимися данными (если есть)
            if data is not None and len(data) > 0:
                data = numpy.vstack([data, boot_rows])
                _write(f"  Объединено: {len(data)} точек "
                       f"(исходные + bootstrap)")
            else:
                data = boot_rows
                _write(f"  Только bootstrap: {len(data)} точек")

        else:
            _write("  Bootstrap не дал результатов.")
            _write("  Генерируем случайные начальные точки (LHS)...")

            # Последний резерв: Latin Hypercube Sampling
            data_lhs, bootstrap_results = _generate_random_initial_points(
                bounds_original = bounds_original,
                n_points        = MIN_POINTS_FOR_PCA,
                output_file     = output_file,
            )
            if data is not None and len(data) > 0:
                data = numpy.vstack([data, data_lhs])
            else:
                data = data_lhs

        # Обновляем флаг после всех попыток
        n_have          = len(data) if data is not None else 0
        data_sufficient = (n_have >= MIN_POINTS_FOR_PCA)

    if not data_sufficient:
        raise ValueError(
            f"Не удалось набрать достаточно начальных точек "
            f"для incl={incl} "
            f"(есть {n_have}, нужно ≥ {MIN_POINTS_FOR_PCA})."
        )

    # Фильтр на корректность для логарифмирования
    if use_log_scale:
        mask_valid = (data[:, 3] > 0) & (data[:, 4] > 0)
        n_dropped  = numpy.sum(~mask_valid)
        if n_dropped > 0:
            _write(f"  Отброшено точек с rh<=0 или rho0<=0: {n_dropped}")
        data = data[mask_valid]

    _write(f"Загружено строк (incl={incl}): {len(data)}")

    # --- Сортировка по penalty ---
    data_sort = data[numpy.argsort(data[:, 6])]
    _write(f"Диапазон penalty: [{data_sort[:, 6].min():.4f}, "
           f"{data_sort[:, 6].max():.4f}]")

    # --- Адаптивный cutoff ---
    penalty_cutoff = adaptive_penalty_cutoff(
        data_sort,
        target_fraction=target_fraction,
        cutoff_start=cutoff_start
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

    X_raw = data_good[:, 1:5].copy()   # Q, gh, rh, rho0

    if use_log_scale:
        X_transformed        = X_raw.copy()
        X_transformed[:, 2]  = numpy.log10(X_raw[:, 2])
        X_transformed[:, 3]  = numpy.log10(X_raw[:, 3])
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
        # --- Резервации других процессов (точки «под оценкой») ---
        active_resv = (read_active_reservations(
                           resv_dir, reserve_ttl_sec,
                           exclude_suffix=hostname_proc)
                       if reserve_points else [])

        # Переводим чужие резервации в текущее PCA-пространство → X_pending,
        # чтобы acquisition штатно уводил предложение в сторону от них.
        X_pending = None
        if active_resv:
            pend = []
            for r in active_resv:
                try:
                    pend.append(params_to_pca_fixed(r['params'], model_data))
                except Exception:
                    pass
            if pend:
                X_pending = torch.tensor(numpy.array(pend),
                                         dtype=dtype, device=device)

        # --- Предложение новой точки ---
        X_next    = turbo.suggest(X_obs, Y_obs, X_pending=X_pending)
        pc_coords = X_next[0].cpu().numpy()
        new_params_i = pca_to_params_fixed(pc_coords, model_data, bounds_original)

        # --- Жёсткая страховка: если предложение всё же слишком близко к
        #     чужой резервации, возмущаем кандидат в пределах TR ---
        if reserve_points and active_resv:
            attempt = 0
            while (attempt < reserve_max_retries
                   and any(reservation_dist_norm(new_params_i, r['params'],
                                                  bounds_original) < reserve_eps
                           for r in active_resv)):
                attempt += 1
                jitter    = proc_rng.normal(scale=0.1 * turbo.length,
                                            size=pc_coords.shape)
                pc_coords = pc_coords + jitter
                new_params_i = pca_to_params_fixed(pc_coords, model_data,
                                                   bounds_original)
            if attempt > 0:
                _write(f"    [reserve] кандидат сдвинут за {attempt} попыток "
                       f"(избегаем дублей с {len(active_resv)} активными "
                       f"резервациями)")
            # Синхронизируем тензор X_next с возможным возмущением
            X_next = torch.tensor([pc_coords], dtype=dtype, device=device)

        # --- Резервируем выбранную точку до дорогого расчёта ---
        resv_fp = (reserve_point(resv_dir, hostname_proc, new_params_i)
                   if reserve_points else None)

        # --- Вычисление целевой функции ---
        try:
            y_next = halo_IC_lib_weights_pca_fixed(
                pc_coords, model_data, bounds_original,
                densityStars, datasets, alphah, betah
            )
        finally:
            # Результат уже записан в UpsFile — резервацию можно снять
            release_reservation(resv_fp)

        # --- Сохраняем в буфер ---
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
    """
    storage_patterns: паттерны для поиска в хранилище (все серверы)
    host_patterns:    паттерны файлов своего сервера
    cutoff_start:     порог отсечки по penalty
    incl_filter:      если не None — фильтровать данные по наклонению.
                      Если None — используется глобальная переменная incl.
    """
    # --- Имя файла диагностики ---
    diag_file = f"diagnose_pca_space_{hostname_proc}.txt"

    # --- Фильтр по наклонению ---
    if incl_filter is None:
        incl_filter = incl          # глобальная переменная
    incl_msg = f"incl={incl_filter}"

    def _write(text):
        """Вспомогательная функция: печать + запись в файл."""
        print(text)
        with open(diag_file, 'a') as f:
            f.write(text + '\n')

    _write("\n" + "=" * 70)
    _write(f"ДИАГНОСТИКА PCA-ПРОСТРАНСТВА  ({incl_msg})")
    _write(f"Файл диагностики: {diag_file}")
    _write("=" * 70)

    # --- Загрузка данных ---
   # --- Загрузка данных через load_fresh_data_from_files ---
    # Читаем ВСЕ файлы включая свой (exclude_suffix=None),
    # так как диагностика должна видеть полную картину
    _write("\nЗагрузка данных...")
    data, file_counts = load_fresh_data_from_files(
        storage_patterns = storage_patterns,
        host_patterns    = host_patterns,
        incl_filter      = incl_filter,
        use_log_scale    = False,      # сырые данные, без логарифмирования
        exclude_suffix   = None,       # читаем все файлы включая свой
        return_full      = True,       # нужен полный массив
    )

    # --- Отчёт по файлам ---
    _write(f"\nПрочитано файлов: {len(file_counts)}")
    for fname, cnt in file_counts.items():
        _write(f"  {fname}: {cnt} строк")

    if data is None or len(data) == 0:
        _write("Нет данных для диагностики.")
        return None

    _write(f"Загружено строк (incl={incl_filter}): {len(data)}")

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

    # --- Корреляционная матрица ---
    _write("\nКорреляционная матрица:")
    _write("-" * 50)
    X    = data_good[:, 1:6]
    corr = numpy.corrcoef(X.T)
    _write("        Q      gh      rh    rho0     Ups")
    for i, name in enumerate(param_names):
        row_str = f"{name:5s} " + "".join(f"{corr[i,j]:7.3f} " for j in range(5))
        _write(row_str)

    # --- PCA без Upsilon ---
    _write("\nСравнение PCA с Upsilon и без:")
    _write("-" * 50)

    X_with_ups = data_good[:, 1:6]
    scaler1    = StandardScaler()
    pca1       = PCA(n_components=4)
    pca1.fit(scaler1.fit_transform(X_with_ups))
    _write(f"С Upsilon:\n"
           f"  Explained variance: {pca1.explained_variance_ratio_}\n"
           f"  Cumulative:         {numpy.cumsum(pca1.explained_variance_ratio_)}")

    X_no_ups = data_good[:, 1:5]
    scaler2  = StandardScaler()
    pca2     = PCA(n_components=4)
    pca2.fit(scaler2.fit_transform(X_no_ups))
    _write(f"\nБез Upsilon:\n"
           f"  Explained variance: {pca2.explained_variance_ratio_}\n"
           f"  Cumulative:         {numpy.cumsum(pca2.explained_variance_ratio_)}")

    # --- PCA с логарифмическим масштабированием ---
    _write("\nС логарифмическим масштабированием:")
    X_log       = X_no_ups.copy()
    X_log[:, 2] = numpy.log10(X_no_ups[:, 2])
    X_log[:, 3] = numpy.log10(X_no_ups[:, 3])
    scaler3     = StandardScaler()
    pca3        = PCA(n_components=4)
    pca3.fit(scaler3.fit_transform(X_log))
    _write(f"  Explained variance: {pca3.explained_variance_ratio_}\n"
           f"  Cumulative:         {numpy.cumsum(pca3.explained_variance_ratio_)}")

    # --- Проверка обратного преобразования ---
    _write("\nПроверка обратного преобразования:")
    _write("-" * 50)
    idx        = numpy.argmin(data_good[:, 6])
    best_point = data_good[idx]
    _write(f"Лучшая точка из данных:\n"
           f"  Q={best_point[1]:.4f}, gh={best_point[2]:.4f}, "
           f"rh={best_point[3]:.4f}, rho0={best_point[4]:.4f}, "
           f"Ups={best_point[5]:.4f}, pen={best_point[6]:.4f}")

    params     = {'Q': best_point[1], 'gh': best_point[2],
                  'rh': best_point[3], 'rho0': best_point[4]}
    X_test     = numpy.array([[params['Q'], params['gh'],
                               numpy.log10(params['rh']),
                               numpy.log10(params['rho0'])]])
    X_t_scaled = scaler3.transform(X_test)
    pc_coords  = pca3.transform(X_t_scaled)
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

    # --- Статистика по cutoff ---
    _write("\nКоличество точек по порогам penalty:")
    cutoffs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
               1.0,  1.5,  2.0,  3.0]
    for cutoff in cutoffs:
        n_pts = numpy.sum(data[:, 6] <= cutoff)
        _write(f"  penalty <= {cutoff:.2f}: "
               f"{n_pts:5d} точек ({100 * n_pts / len(data):.1f}%)")

    # --- Сравнение хороших и приемлемых ---
    compare_good_vs_acceptable(
        data,
        cutoff1=cutoff_start,
        cutoff2=min(cutoff_start * 1.25, cutoff_start + 0.15),
        incl_filter=incl_filter,
        diag_file=diag_file
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
            cutoff_start     = cutoff_start,
            n_components     = 3,
            n_iter           = 40,
            length_init      = 0.6,
            use_log_scale    = True,
            resume           = do_resume,
            pca_update_interval = 12,
            seed_patterns    = seed_patterns,
            seed_from_pa468  = args.seed_from_pa468,
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