#!/usr/bin/env python3
"""Penalty profile along the halo flattening Q (diagnostic, not production).

For every Q of a fixed grid the halo parameters (gh, rh, rho0) are optimised in
the whole bounds_original box (no PCA) by a trust-region Bayesian optimiser
with common random numbers: agama.setRandomSeed(42) before every evaluation,
so all points share one orbit-IC realisation (DECISIONS 2026-10-28). Branches:
`compact` is continued upwards from the lowest Q, `extended` downwards from
Q=1; Sobol points at every Q may spawn a third, `global`, branch. Known seed-42
evaluations (--prior-points) enter the GP at their Q without recomputation.

Rows go to qprofile_* history files that no pool glob reads. Orbit libraries of
evaluations with penalty <= --save-below (Q != 1) are written afterwards into
the shared library directory under their standard names, so a later
--reuse-orblib search does not integrate those points again.

profile_state.json is rewritten after every evaluation; --resume continues an
interrupted run. --dry-run and --summarize need only the standard library.
See doc/ai/harness/orblib_exp.md, section "Q profile".
"""
import argparse
import datetime
import functools
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
import traceback
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_single_model as rsm  # noqa: E402

PARAMS = ('gh', 'rh', 'rho0')
LOG_PARAMS = ('rh', 'rho0')
DEFAULT_Q_GRID = ('low', 0.30, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0)
DEFAULTS = dict(opt_seed=42, n_global=6, local_budget=10, global_budget=10,
                tr_init=0.4, tr_min=0.05, tr_max=0.8, success_tol=2, failure_tol=3,
                improve=0.002, merge_dist=0.05, merge_dp=0.01, global_dist=0.25,
                global_ratio=1.5, smooth_offsets=[0.97, 0.985, 1.015, 1.03],
                sobol_seed=20261028, subsample_seed=20261001)
SECONDS_PER_EVAL = 540.0
STATE_NAME = 'profile_state.json'
TABLE_NAME = 'profile.tsv'
REPORT_NAME = 'report_qprofile'
TABLE_COLUMNS = ('id', 'Q', 'branch', 'kind', 'gh', 'rh', 'rho0', 'seed', 'status', 'penalty',
                 'upsilon', 'wall_s', 'orbit_s', 'library', 'source', 'finished')
POT_RANK = {'l24': 0, 'l4': 1}


class ConfigError(ValueError):
    pass


class StopRun(Exception):
    """STOP/TERM: the evaluation was not recorded; the state on disk is consistent."""


class RunFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Coordinates (unit cube: gh linear, rh and rho0 logarithmic)
# --------------------------------------------------------------------------
def to_unit(params, bounds):
    u = []
    for key in PARAMS:
        lo, hi = bounds[key]
        value = params[key]
        if key in LOG_PARAMS:
            lo, hi, value = math.log(lo), math.log(hi), math.log(value)
        u.append((value - lo) / (hi - lo))
    return u


def from_unit(u, bounds):
    params = {}
    for key, x in zip(PARAMS, u):
        lo, hi = bounds[key]
        x = float(x)
        if x <= 0.0 or x >= 1.0:
            params[key] = float(lo if x <= 0.0 else hi)
            continue
        value = (math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if key in LOG_PARAMS
                 else lo + x * (hi - lo))
        params[key] = min(hi, max(lo, value))
    return params


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def same_params(a, b):
    return all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12) for k in PARAMS)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def parse_q_grid(spec, bounds, low):
    """Comma list ending at 1; the token `low` stands for `low` (see build_config)."""
    tokens = [t.strip() for t in str(spec).split(',') if t.strip()] if spec else list(DEFAULT_Q_GRID)
    try:
        grid = [float(low) if str(t) == 'low' else float(t) for t in tokens]
    except (TypeError, ValueError):
        raise ConfigError(f'bad --q-grid {spec!r}')
    lo, hi = bounds['Q']
    if len(grid) < 2 or grid != sorted(set(grid)):
        raise ConfigError('--q-grid needs >= 2 distinct increasing values')
    if not all(lo <= q <= hi for q in grid):
        raise ConfigError(f'--q-grid outside bounds_original Q {bounds["Q"]}')
    if abs(grid[-1] - 1.0) > 1e-12:
        raise ConfigError('--q-grid must end at Q=1 (start of the extended branch)')
    return grid


def read_endpoints(path, bounds, incl):
    """compact start: lowest-penalty row with Q < 1; extended start: with Q = 1.
    Only (gh, rh, rho0) are used; Q comes from the grid."""
    rows = [r for r in rsm.read_models(path, bounds) if abs(r['incl'] - incl) <= 1e-6]
    flattened = [r for r in rows if r['Q'] < 1 - 1e-12]
    spherical = [r for r in rows if abs(r['Q'] - 1) <= 1e-12]
    if not flattened or not spherical:
        raise ConfigError(f'{path}: need a row with Q<1 and a row with Q=1 at incl={incl}')
    starts = {}
    for name, candidates in (('compact', flattened), ('extended', spherical)):
        row = min(candidates, key=lambda r: r['penalty'])
        starts[name] = dict(params={k: row[k] for k in PARAMS}, Q=row['Q'],
                            penalty=row['penalty'], source=row['source'])
    return starts


def read_prior_rows(path, bounds, incl):
    """Known seed-42 evaluations (any history layout). Label tokens `pot=l24|l4`,
    `protocol=…`, `src=…`. Rows at another incl or outside the bounds are skipped."""
    rows, skipped = [], []
    with open(path, errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            text, _, label = line.partition('#')
            row = rsm.parse_reference_line(text)
            if row is None:
                continue
            tags = dict(t.split('=', 1) for t in label.split() if '=' in t)
            where = f'{os.path.basename(path)}:{number}'
            if abs(row['incl'] - incl) > 1e-6 or rsm.bounds_violations(row, bounds):
                skipped.append(where)
                continue
            rows.append(dict(Q=row['Q'], params={k: row[k] for k in PARAMS}, penalty=row['penalty'],
                             upsilon=row['upsilon'], pot=tags.get('pot', 'unknown'),
                             protocol=tags.get('protocol', 'unknown'), source=tags.get('src', where)))
    return rows, skipped


def assign_priors(rows, grid):
    """Rows whose Q is not on the grid are skipped; repeated parameters at one Q
    keep the better potential tag (l24 > l4 > unknown)."""
    found, skipped = [], []
    for row in rows:
        qi = next((i for i, q in enumerate(grid) if abs(q - row['Q']) <= 1e-12), None)
        if qi is None:
            skipped.append(row['source'] + ' (Q off the grid)')
        else:
            found.append(dict(row, qi=qi, Q=grid[qi]))
    found.sort(key=lambda p: POT_RANK.get(p['pot'], 2))
    priors = []
    for prior in found:
        if any(p['qi'] == prior['qi'] and same_params(p['params'], prior['params']) for p in priors):
            skipped.append(prior['source'] + ' (duplicate)')
        else:
            priors.append(prior)
    priors.sort(key=lambda p: (p['qi'], p['source']))
    return priors, skipped


def build_config(args, bounds):
    for name in ('opt_seed', 'local_budget', 'global_budget'):
        if getattr(args, name) < 1:
            raise ConfigError(f'--{name.replace("_", "-")} must be >= 1')
    if args.n_global < 0:
        raise ConfigError('--n-global must be >= 0')
    starts = read_endpoints(args.endpoints, bounds, args.incl)
    rows, skipped = read_prior_rows(args.prior_points, bounds, args.incl) if args.prior_points else ([], [])
    lows = [r['Q'] for r in rows if r['pot'] == 'l24' and r['Q'] < 1 - 1e-12]
    grid = parse_q_grid(args.q_grid, bounds, min(lows) if lows else starts['compact']['Q'])
    priors, off_grid = assign_priors(rows, grid)
    return dict(DEFAULTS, incl=float(args.incl), q_grid=grid, pot='l24',
                bounds={k: list(v) for k, v in bounds.items()}, starts=starts,
                priors=priors, prior_skipped=skipped + off_grid, opt_seed=args.opt_seed,
                n_global=args.n_global, local_budget=args.local_budget,
                global_budget=args.global_budget)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def budget(config):
    n = len(config['q_grid'])
    base = n * (2 * config['local_budget'] + config['n_global'])
    return dict(per_q=2 * config['local_budget'] + config['n_global'], total=base,
                total_max=base + n * config['global_budget'],
                hours=base * SECONDS_PER_EVAL / 3600,
                hours_max=(base + n * config['global_budget']) * SECONDS_PER_EVAL / 3600)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def atomic_json(path, data):
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w') as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# Default numerical back-ends (imported lazily: --dry-run/--summarize are stdlib)
# --------------------------------------------------------------------------
def sobol_points(n, seed):
    import warnings
    from scipy.stats import qmc
    if n <= 0:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return qmc.Sobol(d=len(PARAMS), scramble=True, seed=seed).random(n).tolist()


def _random_in_box(lo, hi, seed):
    rng = random.Random(seed)
    return [rng.uniform(a, b) for a, b in zip(lo, hi)]


def propose_bo(points, values, baseline, center, length, seed):
    """qLogNEI inside the trust region [center ± length/2] ∩ [0,1]^3; GP on
    y = -log(penalty) of all points at this Q, incumbent from the branch."""
    lo = [max(0.0, c - length / 2) for c in center]
    hi = [min(1.0, c + length / 2) for c in center]
    if len(points) < 3:
        return _random_in_box(lo, hi, seed)
    try:
        import torch
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.constraints import GreaterThan
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.manual_seed(seed)
        dtype = torch.double
        X = torch.tensor(points, dtype=dtype)
        Y = -torch.log(torch.tensor(values, dtype=dtype)).unsqueeze(-1)
        unit = torch.tensor([[0.0] * len(PARAMS), [1.0] * len(PARAMS)], dtype=dtype)
        model = SingleTaskGP(X, Y, likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
                             input_transform=Normalize(d=len(PARAMS), bounds=unit),
                             outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=X[list(baseline)], prune_baseline=True)
        candidate, _ = optimize_acqf(acq_function=acqf, bounds=torch.tensor([lo, hi], dtype=dtype),
                                     q=1, num_restarts=10, raw_samples=2048)
        return [float(v) for v in candidate[0]]
    except Exception as error:
        print(f'[qprofile] BO proposal failed ({error!r}); random point in the trust region', flush=True)
        return _random_in_box(lo, hi, seed)


def smooth_check(records):
    """Quadratic fit of penalty vs log rho0 over the diagnostic cut (logged only)."""
    import numpy
    x = numpy.log([r['params']['rho0'] for r in records])
    y = numpy.array([r['penalty'] for r in records])
    if len(records) < 4:
        return dict(n=len(records), rms=None, max=None, verdict='insufficient points')
    residual = y - numpy.polyval(numpy.polyfit(x, y, 2), x)
    return dict(n=len(records), rms=float(numpy.sqrt(numpy.mean(residual ** 2))),
                max=float(numpy.max(numpy.abs(residual))),
                points=[[r['params']['rho0'], r['penalty']] for r in records], verdict='logged')


# --------------------------------------------------------------------------
# Profile driver (back-ends injected: tests run it without AGAMA/torch)
# --------------------------------------------------------------------------
class Profile:
    def __init__(self, config, state_dir, evaluator, proposer=propose_bo, sobol=sobol_points,
                 smooth=smooth_check, log=print, max_failures=5):
        self.cfg = config
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, STATE_NAME)
        self.table = os.path.join(state_dir, TABLE_NAME)
        self.evaluator = evaluator
        self.proposer = proposer
        self.sobol = sobol
        self.smooth = smooth
        self.log = log
        self.max_failures = max_failures
        self.failures = 0
        self.state = None
        self.last = len(config['q_grid']) - 1

    # ---- state -----------------------------------------------------------
    def open(self, resume, run_info=None):
        os.makedirs(self.state_dir, exist_ok=True)
        if os.path.exists(self.path):
            if not resume:
                raise ConfigError(f'{self.path} exists: pass --resume or choose another --state-dir')
            with open(self.path) as stream:
                self.state = json.load(stream)
            if self.state['config_hash'] != config_hash(self.cfg):
                raise ConfigError('configuration differs from the state file (grid/starts/priors/budgets)')
        else:
            if resume:
                raise ConfigError(f'--resume: no {self.path}')
            self.state = dict(version=1, config=self.cfg, config_hash=config_hash(self.cfg),
                              created=now(), status='running', evaluations=[], branches={},
                              merged={}, globals_checked=[], smooth_check=None, runs=[])
            for prior in self.cfg['priors']:
                self._append(dict(prior, kind='prior', branch=None, seed=None, status='ok',
                                  u=to_unit(prior['params'], self.cfg['bounds']), finished=None))
        if run_info:
            self.state['runs'].append(dict(run_info, started=now()))
        self.state['status'] = 'running'
        self.save()

    def save(self):
        self.state['updated'] = now()
        atomic_json(self.path, self.state)

    def _append(self, record):
        record['id'] = len(self.state['evaluations'])
        self.state['evaluations'].append(record)
        new = not os.path.exists(self.table)
        with open(self.table, 'a') as stream:
            if new:
                stream.write('\t'.join(TABLE_COLUMNS) + '\n')
            flat = dict(record, **record['params'])
            flat['library'] = (record.get('library') or {}).get('name') or (record.get('library') or {}).get('skipped')
            stream.write('\t'.join('' if flat.get(k) is None else str(flat.get(k)) for k in TABLE_COLUMNS) + '\n')
        return record

    # ---- queries ---------------------------------------------------------
    def records(self, qi, ok=True):
        return [r for r in self.state['evaluations']
                if r['qi'] == qi and (not ok or r['status'] == 'ok')]

    def find(self, qi, params):
        return next((r for r in self.records(qi, ok=False) if same_params(r['params'], params)), None)

    def branch_best(self, name, qi):
        branch = self.state['branches'].get(f'{name}@{qi}')
        if branch is None:
            return None
        members = [self.state['evaluations'][i] for i in branch['members']]
        members = [r for r in members if r['status'] == 'ok']
        return min(members, key=lambda r: r['penalty']) if members else None

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, qi, params, kind, branch=None, extra=None):
        Q = self.cfg['q_grid'][qi]
        meta = dict(id=len(self.state['evaluations']), qi=qi, Q=Q, branch=branch, kind=kind)
        self.log(f'[qprofile] eval {meta["id"]}: Q={Q:.6g} {kind} branch={branch} '
                 + ' '.join(f'{k}={params[k]:.6g}' for k in PARAMS))
        result = self.evaluator(Q, dict(params), self.cfg['opt_seed'], meta)
        record = dict(qi=qi, Q=Q, params=dict(params), u=to_unit(params, self.cfg['bounds']),
                      kind=kind, branch=branch, seed=self.cfg['opt_seed'], status=result.get('status', 'failed'),
                      penalty=result.get('penalty'), upsilon=result.get('upsilon'),
                      wall_s=result.get('wall_s'), orbit_s=result.get('orbit_s'),
                      library=result.get('library'), error=result.get('error'),
                      source='qprofile', finished=now(), **(extra or {}))
        if record['status'] == 'ok' and not (record['penalty'] and record['penalty'] > 0):
            record['status'] = 'failed'
        self.failures = 0 if record['status'] == 'ok' else self.failures + 1
        return record

    def _record(self, record, branch=None):
        self._append(record)
        if branch is not None:
            branch['members'].append(record['id'])
            branch['n_evals'] += 1
        self.save()
        self.log(f'[qprofile]   → status={record["status"]} penalty={record["penalty"]} '
                 f'upsilon={record["upsilon"]} library={record.get("library")}')
        if self.failures >= self.max_failures:
            raise RunFailed(f'{self.failures} consecutive failed evaluations')
        return record

    # ---- schedule --------------------------------------------------------
    def stages(self):
        yield functools.partial(self.branch_stage, 'compact', 0)
        yield functools.partial(self.sobol_stage, 0)
        yield functools.partial(self.branch_stage, 'extended', self.last)
        yield functools.partial(self.sobol_stage, self.last)
        for qi in range(1, self.last + 1):
            yield functools.partial(self.sobol_stage, qi)
            yield functools.partial(self.branch_stage, 'compact', qi)
        for qi in range(self.last - 1, -1, -1):
            yield functools.partial(self.branch_stage, 'extended', qi)
        yield self.global_stage

    def run(self):
        for stage in self.stages():
            stage()
        self.state['status'] = 'done'
        self.save()

    def sobol_stage(self, qi):
        for j, u in enumerate(self.sobol(self.cfg['n_global'], self.cfg['sobol_seed'] + qi)):
            if any(r['kind'] == 'sobol' and r['qi'] == qi and r.get('sobol_index') == j
                   for r in self.state['evaluations']):
                continue
            self._record(self.evaluate(qi, from_unit(u, self.cfg['bounds']), 'sobol',
                                       extra=dict(sobol_index=j)))

    def start_params(self, name, qi):
        origin = 0 if name == 'compact' else self.last
        if qi == origin:
            return dict(self.cfg['starts'][name]['params'])
        best = self.branch_best(name, qi - 1 if name == 'compact' else qi + 1)
        return dict(best['params']) if best else None

    def new_branch(self, name, qi, start, budget_, members=(), status='active', reason=None):
        branch = dict(name=name, qi=qi, Q=self.cfg['q_grid'][qi], start=start, budget=budget_,
                      length=self.cfg['tr_init'], succ=0, fail=0, n_evals=0, members=list(members),
                      duplicates=0, status=status, reason=reason, merged_into=None)
        self.state['branches'][f'{name}@{qi}'] = branch
        self.save()
        return branch

    def branch_stage(self, name, qi):
        branch = self.state['branches'].get(f'{name}@{qi}')
        if branch is None:
            if name in self.state['merged']:
                self.new_branch(name, qi, None, 0, status='skipped',
                                reason=f'merged at Q={self.cfg["q_grid"][self.state["merged"][name]]}')
                return
            start = self.start_params(name, qi)
            if start is None:
                self.new_branch(name, qi, None, 0, status='skipped', reason='no start point')
                return
            branch = self.new_branch(name, qi, start, self.cfg['local_budget'])
        if branch['status'] == 'active':
            self.advance(branch)

    def advance(self, branch):
        qi = branch['qi']
        if not branch['members']:
            existing = self.find(qi, branch['start'])
            if existing is not None:
                branch['members'].append(existing['id'])
                self.save()
            else:
                self._record(self.evaluate(qi, branch['start'], 'start', branch['name']), branch)
        if branch['name'] == 'compact' and qi == 0 and self.state['smooth_check'] is None:
            self.diagnostic(branch)
        while branch['status'] == 'active':
            if branch['n_evals'] >= branch['budget']:
                branch['status'] = 'budget'
            elif branch['length'] < self.cfg['tr_min']:
                branch['status'] = 'converged'
            elif self.branch_best(branch['name'], qi) is None:
                branch['status'] = 'failed'
            else:
                self.step(branch)
                continue
            self.save()
        self.check_merge(branch)

    def diagnostic(self, branch):
        start, bounds = branch['start'], self.cfg['bounds']
        for factor in self.cfg['smooth_offsets']:
            params = dict(start, rho0=min(bounds['rho0'][1], max(bounds['rho0'][0], start['rho0'] * factor)))
            if self.find(0, params) is None and branch['n_evals'] < branch['budget']:
                self._record(self.evaluate(0, params, 'diag', branch['name']), branch)
        cut = [r for r in self.records(0) if r['params']['gh'] == start['gh'] and r['params']['rh'] == start['rh']
               and r['source'] == 'qprofile']
        self.state['smooth_check'] = self.smooth(sorted(cut, key=lambda r: r['params']['rho0']))
        self.save()
        self.log(f'[qprofile] smoothness check (logged only): {self.state["smooth_check"]}')

    def step(self, branch):
        qi, name = branch['qi'], branch['name']
        best = self.branch_best(name, qi)
        points = self.records(qi)
        index = {r['id']: i for i, r in enumerate(points)}
        baseline = [index[i] for i in branch['members'] if i in index]
        seed = (self.cfg['sobol_seed'] + 1000 * qi + 17 * branch['n_evals'] + branch['duplicates']
                + zlib.crc32(name.encode())) % (2 ** 31)
        u = self.proposer([r['u'] for r in points], [r['penalty'] for r in points], baseline,
                          best['u'], branch['length'], seed)
        params = from_unit(u, self.cfg['bounds'])
        u = to_unit(params, self.cfg['bounds'])
        if any(distance(u, r['u']) < 1e-6 for r in self.records(qi, ok=False)):
            branch['duplicates'] += 1
            self.update_tr(branch, improved=False)
            self.save()
            return
        record = self._record(self.evaluate(qi, params, 'bo', name), branch)
        self.update_tr(branch, record['status'] == 'ok' and record['penalty'] < best['penalty'] - self.cfg['improve'])
        self.save()

    def update_tr(self, branch, improved):
        if improved:
            branch['succ'], branch['fail'] = branch['succ'] + 1, 0
        else:
            branch['succ'], branch['fail'] = 0, branch['fail'] + 1
        if branch['succ'] >= self.cfg['success_tol']:
            branch['length'], branch['succ'] = min(2 * branch['length'], self.cfg['tr_max']), 0
        if branch['fail'] >= self.cfg['failure_tol']:
            branch['length'], branch['fail'] = branch['length'] / 2, 0

    def check_merge(self, branch):
        if branch['name'] not in ('compact', 'extended') or branch['status'] not in ('converged', 'budget'):
            return
        other = 'extended' if branch['name'] == 'compact' else 'compact'
        theirs = self.state['branches'].get(f'{other}@{branch["qi"]}')
        if theirs is None or theirs['status'] not in ('converged', 'budget'):
            return
        mine, best = self.branch_best(branch['name'], branch['qi']), self.branch_best(other, branch['qi'])
        if (mine and best and distance(mine['u'], best['u']) < self.cfg['merge_dist']
                and abs(mine['penalty'] - best['penalty']) <= self.cfg['merge_dp']):
            branch['merged_into'] = other
            self.state['merged'][branch['name']] = branch['qi']
            self.save()
            self.log(f'[qprofile] {branch["name"]} merged into {other} at Q={branch["Q"]:.6g}')

    def global_stage(self):
        for qi in range(self.last + 1):
            key = f'global@{qi}'
            if key in self.state['branches']:
                if self.state['branches'][key]['status'] == 'active':
                    self.advance(self.state['branches'][key])
                continue
            if qi in self.state['globals_checked']:
                continue
            sobol = [r for r in self.records(qi) if r['kind'] == 'sobol']
            self.state['globals_checked'].append(qi)
            if not sobol:
                self.save()
                continue
            candidate = min(sobol, key=lambda r: r['penalty'])
            bests = [b for b in (self.branch_best(n, qi) for n in ('compact', 'extended')) if b]
            far = all(distance(candidate['u'], b['u']) > self.cfg['global_dist'] for b in bests)
            good = not bests or candidate['penalty'] < self.cfg['global_ratio'] * min(b['penalty'] for b in bests)
            if far and good:
                self.log(f'[qprofile] Q={self.cfg["q_grid"][qi]:.6g}: Sobol point {candidate["id"]} '
                         'is far from both branches — global branch')
                self.advance(self.new_branch('global', qi, dict(candidate['params']),
                                             self.cfg['global_budget'], members=[candidate['id']]))
            else:
                self.save()


# --------------------------------------------------------------------------
# Summary (standard library only)
# --------------------------------------------------------------------------
def _boundary(u):
    hits = []
    for key, x in zip(PARAMS, u):
        if x <= 1e-3:
            hits.append(f'{key}=lo')
        elif x >= 1 - 1e-3:
            hits.append(f'{key}=hi')
    return hits


def _point(record):
    if record is None:
        return None
    upsilon = record.get('upsilon')
    return dict(id=record['id'], penalty=record['penalty'], upsilon=upsilon, kind=record['kind'],
                params=record['params'], rho0_upsilon=record['params']['rho0'] * upsilon if upsilon else None,
                boundary=_boundary(record['u']), source=record.get('source'), pot=record.get('pot'))


def summarize(state):
    cfg = state['config']
    evaluations = state['evaluations']
    rows = []
    for qi, Q in enumerate(cfg['q_grid']):
        ok = [r for r in evaluations if r['qi'] == qi and r['status'] == 'ok']
        row = dict(qi=qi, Q=Q, n_evaluations=sum(1 for r in evaluations if r['qi'] == qi and r['kind'] != 'prior'),
                   n_priors=sum(1 for r in ok if r['kind'] == 'prior'), branches={},
                   sobol_best=_point(min((r for r in ok if r['kind'] == 'sobol'), key=lambda r: r['penalty'],
                                         default=None)),
                   profile=_point(min(ok, key=lambda r: r['penalty'], default=None)))
        for branch in state['branches'].values():
            if branch['qi'] != qi:
                continue
            members = [evaluations[i] for i in branch['members'] if evaluations[i]['status'] == 'ok']
            best = min(members, key=lambda r: r['penalty'], default=None)
            row['branches'][branch['name']] = dict(status=branch['status'], reason=branch.get('reason'),
                                                   n_evals=branch['n_evals'], length=branch['length'],
                                                   merged_into=branch.get('merged_into'), best=_point(best))
        compact = (row['branches'].get('compact') or {}).get('best')
        extended = (row['branches'].get('extended') or {}).get('best')
        row['hysteresis'] = bool(compact and extended and
                                 distance(to_unit(compact['params'], cfg['bounds']),
                                          to_unit(extended['params'], cfg['bounds'])) > 0.1)
        row['lower_branch'] = (None if not (compact or extended) else
                               'compact' if extended is None or (compact and compact['penalty'] <= extended['penalty'])
                               else 'extended')
        rows.append(row)
    profile = [r['profile']['penalty'] if r['profile'] else None for r in rows]
    barrier = None
    if len(profile) > 2 and all(p is not None for p in profile):
        top = max(range(1, len(profile) - 1), key=lambda i: profile[i])
        barrier = dict(value=profile[top] - max(profile[0], profile[-1]), Q=cfg['q_grid'][top])
    switches = [rows[i]['Q'] for i in range(1, len(rows))
                if rows[i]['lower_branch'] and rows[i - 1]['lower_branch']
                and rows[i]['lower_branch'] != rows[i - 1]['lower_branch']]
    return dict(status=state.get('status'), updated=state.get('updated'), opt_seed=cfg['opt_seed'],
                n_evaluations=sum(1 for r in evaluations if r['kind'] != 'prior'),
                n_failed=sum(1 for r in evaluations if r['status'] != 'ok'),
                libraries_saved=sum(1 for r in evaluations if (r.get('library') or {}).get('name')),
                smooth_check=state.get('smooth_check'), merged=state.get('merged'),
                rows=rows, barrier=barrier, branch_switch_Q=switches,
                note='one IC realisation (seed %d): shapes are reliable between neighbouring points, '
                     'differences between distant parameters carry the realisation scatter of two '
                     'independent evaluations; penalty is not chi^2' % cfg['opt_seed'])


def format_summary(summary):
    lines = [f"Q profile: status={summary['status']} evaluations={summary['n_evaluations']} "
             f"failed={summary['n_failed']} libraries saved={summary['libraries_saved']} "
             f"(seed {summary['opt_seed']}, updated {summary['updated']})"]
    smooth = summary.get('smooth_check')
    if smooth:
        lines.append(f"  smoothness check (logged): n={smooth.get('n')} rms={smooth.get('rms')} max={smooth.get('max')}")
    lines.append(f"  {'Q':>8} {'P_prof':>9} {'kind':>7} {'branch':>8}  compact (P, status)        extended (P, status)       sobol P   notes")
    for row in summary['rows']:
        prof = row['profile'] or {}
        cells = []
        for name in ('compact', 'extended'):
            b = row['branches'].get(name)
            cells.append('-' if b is None else
                         f"{(b['best'] or {}).get('penalty', float('nan')):.5f} {b['status']}"
                         + (f"→{b['merged_into']}" if b.get('merged_into') else ''))
        notes = []
        if row['hysteresis']:
            notes.append('hysteresis')
        if 'global' in row['branches']:
            g = row['branches']['global']
            notes.append(f"global {(g['best'] or {}).get('penalty')}")
        if prof.get('boundary'):
            notes.append('bound:' + ','.join(prof['boundary']))
        sob = (row['sobol_best'] or {}).get('penalty')
        sob_text = '-' if sob is None else format(sob, '.4g')
        lines.append(f"  {row['Q']:8.5f} {prof.get('penalty', float('nan')):9.5f} {str(prof.get('kind')):>7} "
                     f"{str(row['lower_branch']):>8}  {cells[0]:26} {cells[1]:26} "
                     f"{sob_text:>8}  {' '.join(notes)}")
        if prof:
            p = prof['params']
            lines.append(f"  {'':8} gh={p['gh']:.4g} rh={p['rh']:.5g} rho0={p['rho0']:.5g} "
                         f"Upsilon={prof.get('upsilon')} rho0*Ups={prof.get('rho0_upsilon')}")
    if summary['barrier']:
        lines.append(f"  barrier B = {summary['barrier']['value']:+.4f} at Q={summary['barrier']['Q']:.4g}")
    lines.append(f"  lower branch switches at Q: {summary['branch_switch_Q']}; merged: {summary['merged']}")
    lines.append(f"  note: {summary['note']}")
    return '\n'.join(lines) + '\n'


def write_summary(state_dir):
    with open(os.path.join(state_dir, STATE_NAME)) as stream:
        summary = summarize(json.load(stream))
    atomic_json(os.path.join(state_dir, REPORT_NAME + '.json'), summary)
    text = format_summary(summary)
    with open(os.path.join(state_dir, REPORT_NAME + '.txt'), 'w') as stream:
        stream.write(text)
    return text


def format_dry_run(config):
    plan = budget(config)
    lines = [f"Q profile plan: incl={config['incl']} pot={config['pot']} seed={config['opt_seed']}",
             f"  Q grid: {', '.join(repr(q) for q in config['q_grid'])}"]
    for name, start in config['starts'].items():
        p = start['params']
        lines.append(f"  start {name}: gh={p['gh']:.6g} rh={p['rh']:.6g} rho0={p['rho0']:.6g} "
                     f"(row Q={start['Q']:.6g}, penalty={start['penalty']:.6g}, {start['source']})")
    for qi, Q in enumerate(config['q_grid']):
        priors = [p for p in config['priors'] if p['qi'] == qi]
        if priors:
            lines.append(f"  priors at Q={Q:.6g}: {len(priors)} ({', '.join(sorted({p['pot'] for p in priors}))})")
    if config['prior_skipped']:
        lines.append(f"  priors skipped (Q not on the grid, incl, bounds or duplicates): {len(config['prior_skipped'])}")
    lines.append(f"  budget per Q: 2 branches x {config['local_budget']} + {config['n_global']} Sobol "
                 f"(+{config['global_budget']} if a global branch appears)")
    lines.append(f"  evaluations: {plan['total']} (max {plan['total_max']}), "
                 f"~{plan['hours']:.0f} h (max ~{plan['hours_max']:.0f} h) at {SECONDS_PER_EVAL:.0f} s/evaluation")
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
# AGAMA evaluation (module import as in run_single_model) and library saving
# --------------------------------------------------------------------------
class OrbitCapture:
    """Proxy for the harness's `agama`: keeps the last orbit() inputs/outputs so
    the library can be saved after the penalty is known."""

    def __init__(self, agama):
        self._agama = agama
        self.last = None

    def __getattr__(self, name):
        return getattr(self._agama, name)

    def orbit(self, *args, **kwargs):
        result = self._agama.orbit(*args, **kwargs)
        self.last = dict(ic=kwargs.get('ic'), time=kwargs.get('time'), matrices=result)
        return result


def append_line(path, text):
    with open(path, 'a+b') as stream:
        stream.seek(0, os.SEEK_END)
        lead = b''
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            lead = b'' if stream.read(1) == b'\n' else b'\n'
        stream.write(lead + (text + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())


def save_library(mod, storage, capture, Q, params, penalty, shared_dir, save_below, min_free_gb):
    """Write the captured library to shared_dir/<orblib_key> unless a file (or a
    .building claim) of that name exists; never overwrites. Q=1 names carry no
    potential tag, so L24 matrices there would be mistaken for N23/l4 ones."""
    import inspect
    import numpy
    import shutil
    if penalty is None or penalty > save_below:
        return dict(skipped=f'penalty>{save_below}')
    if abs(Q - 1.0) <= 1e-12:
        return dict(skipped='Q=1 (untagged library name)')
    if capture.last is None or capture.last.get('ic') is None or capture.last.get('time') is None:
        return dict(skipped='no captured orbit library')
    name, _ = mod.orblib_key(Q, params['gh'], params['rh'], params['rho0'])
    path = os.path.join(shared_dir, name)
    if os.path.exists(path) or os.path.exists(path + '.building'):
        return dict(skipped='exists', name_existing=name)
    os.makedirs(shared_dir, exist_ok=True)
    if shutil.disk_usage(shared_dir).free < min_free_gb * 1e9:
        return dict(skipped=f'free disk < {min_free_gb} GB')
    defaults = inspect.signature(mod.halo_IC_lib_weights_pca_fixed).parameters
    ic, inttime, matrices = capture.last['ic'], capture.last['time'], capture.last['matrices']
    scalars = dict(Q=Q, gh=params['gh'], rh=params['rh'], rho0=params['rho0'], incl=mod.incl,
                   numOrbits=len(ic), trajsize=mod.TRAJSIZE_STORED, intTime=defaults['intTime'].default,
                   gridv=mod.gridv, degree=mod.degree, ghorder=mod.ghorder, n_bin=mod.N_BIN,
                   double=int(mod.DOUBLE), ser_id=mod.SER_ID)
    expected = {k: v for k, v in scalars.items() if k not in ('gridv', 'trajsize')}
    expected.update(gridv_md5=hashlib.md5(numpy.asarray(mod.gridv).tobytes()).hexdigest(),
                    gridv_dtype=numpy.asarray(mod.gridv).dtype.str)
    temporary = f'{path}.{os.getpid()}.qprofile.tmp'
    started = time.perf_counter()
    try:
        with open(temporary, 'wb') as stream:
            hashed = storage.HashingWriter(stream)
            mod.write_orblib_npz(hashed, dict(matrix_dens=matrices[0], matrix_kinem=matrices[1]),
                                 ic, inttime, scalars)
            stream.flush()
            os.fsync(stream.fileno())
        if not storage.compatible_metadata(storage.metadata(temporary), expected):
            raise ValueError('metadata of the written library differ from the evaluation')
        try:
            os.link(temporary, path)
        except FileExistsError:
            return dict(skipped='exists', name_existing=name)
        storage.fsync_directory(shared_dir)
        return dict(name=name, size=hashed.size, md5=hashed.checksum.hexdigest(),
                    save_s=round(time.perf_counter() - started, 1))
    except Exception as error:
        traceback.print_exc()
        return dict(skipped=f'save failed: {error!r}')
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def build_evaluator(args, state_dir):
    import importlib.util
    import numpy
    sys.argv = [rsm.EXP_SCRIPT, '--incl', repr(float(args.incl)), '--suffix', args.suffix,
                '--orblib-dir', args.orblib_dir, '--no-resume', '--l24']
    if args.n_threads is not None:
        sys.argv += ['--n_threads', str(args.n_threads)]
    spec = importlib.util.spec_from_file_location('orblib_exp', rsm.EXP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['orblib_exp'] = mod
    spec.loader.exec_module(mod)
    import orblib_storage as storage
    store = storage.Store(mod.ORBLIB_DIR, int(os.environ.get('ORBLIB_RESERVE_BYTES', '2000000000')),
                          save_slots=int(os.environ.get('ORBLIB_SAVE_SLOTS', '2')))
    rsm.prepare_module(mod, store)
    capture = OrbitCapture(mod.agama)
    mod.agama = capture
    history = os.path.join(state_dir, f'qprofile_{mod.hostname_proc}.txt')
    files = dict(pool=history, side=history)

    def evaluate(Q, params, seed, meta):
        try:
            store.check_stop()
            append_line(history, f"# qprofile: eval={meta['id']} Q={Q!r} branch={meta['branch']} kind={meta['kind']}")
            rsm.mark_seed(history, seed)
            capture.last = None
            try:
                run = rsm.run_protocol(mod, 'exp', dict(params, Q=Q), files, args.subsample_seed,
                                       numpy.random.default_rng, ic_seed=seed)
            except storage.StorageStop:
                raise
            except Exception as error:
                traceback.print_exc()
                return dict(status='failed', error=repr(error))
            result = dict(status='ok', penalty=run['penalty'], upsilon=run['upsilon'],
                          wall_s=run['wall_s'], orbit_s=run['orbit_s'])
            result['library'] = save_library(mod, storage, capture, Q, params, run['penalty'],
                                             args.shared_orblib, args.save_below, args.min_free_gb)
            return result
        except storage.StorageStop as error:
            raise StopRun(error.reason)
        finally:
            capture.last = None

    return evaluate, mod, store


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], allow_abbrev=False)
    parser.add_argument('--incl', type=float, default=90.0)
    parser.add_argument('--q-grid', default=None, help='Comma list ending at 1 (default: '
                        + ','.join(str(q) for q in DEFAULT_Q_GRID) + '; low = lowest Q<1 of the pot=l24 '
                        'priors, else the Q<1 finalist)')
    parser.add_argument('--endpoints', default=None, help='Rows of the two finalists (Q<1 and Q=1)')
    parser.add_argument('--prior-points', default=None, help='Known seed-42 evaluations (label pot=l24|l4)')
    parser.add_argument('--state-dir', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--orblib-dir', default=None, help='Isolated store directory of the harness')
    parser.add_argument('--shared-orblib', default=None, help='Shared library directory (saved .npz)')
    parser.add_argument('--n_threads', type=int, default=None)
    parser.add_argument('--save-below', type=float, default=3.5)
    parser.add_argument('--min-free-gb', type=float, default=40.0)
    parser.add_argument('--opt-seed', type=int, default=DEFAULTS['opt_seed'])
    parser.add_argument('--n-global', type=int, default=DEFAULTS['n_global'])
    parser.add_argument('--local-budget', type=int, default=DEFAULTS['local_budget'])
    parser.add_argument('--global-budget', type=int, default=DEFAULTS['global_budget'])
    parser.add_argument('--subsample-seed', type=int, default=DEFAULTS['subsample_seed'])
    parser.add_argument('--max-failures', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='Print the plan (standard library only)')
    parser.add_argument('--summarize', default=None, metavar='STATE_DIR',
                        help='Write report_qprofile.json/.txt of a state directory (standard library only)')
    args = parser.parse_args(argv)
    if args.summarize is None:
        if not args.endpoints:
            parser.error('--endpoints is required')
        if not args.dry_run and not (args.state_dir and args.suffix and args.orblib_dir and args.shared_orblib):
            parser.error('--state-dir, --suffix, --orblib-dir and --shared-orblib are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summarize is not None:
        print(write_summary(args.summarize), end='')
        return 0
    try:
        config = build_config(args, rsm.script_bounds())
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    if args.dry_run:
        print(format_dry_run(config), end='')
        return 0
    os.makedirs(args.state_dir, exist_ok=True)
    stop_marker = os.path.join(args.orblib_dir, '.storage', 'STOP')
    if os.path.exists(stop_marker):
        os.remove(stop_marker)
    evaluate, mod, _ = build_evaluator(args, args.state_dir)
    profile = Profile(config, args.state_dir, evaluate, max_failures=args.max_failures)
    run_info = dict(host=os.environ.get('HOSTNAME_SUFFIX', socket.gethostname()), suffix=args.suffix,
                    hostname_proc=mod.hostname_proc, exp_id=mod.EXP_ID, geom_hash=mod.GEOM_HASH,
                    evaluation_context=mod.EVALUATION_CONTEXT,
                    harness_sha256=sha256_file(rsm.EXP_SCRIPT), driver_sha256=sha256_file(__file__),
                    omp_threads=getattr(mod, 'AGAMA_ORBIT_THREADS', None))
    try:
        profile.open(args.resume, run_info)
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    code = 0
    try:
        profile.run()
    except StopRun as error:
        profile.state['status'], code = f'stopped: {error}', 75
    except RunFailed as error:
        profile.state['status'], code = f'failed: {error}', 1
    except Exception as error:
        traceback.print_exc()
        profile.state['status'], code = f'failed: {error!r}', 1
    profile.save()
    print(write_summary(args.state_dir), end='', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
