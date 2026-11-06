#!/usr/bin/env python3
"""Local TuRBO search in a full 4D PCA basis with common random numbers (diagnostic, not production).

All four halo parameters (Q, gh, rh, rho0) are searched inside a local box (--box,
within bounds_original) by a trust-region Bayesian optimiser in the coordinates of
a 4-component PCA of x = (Q, gh, log10 rh, log10 rho0), built as in the harness
(best 30 % of the points, at least 10; weights exp(-(P - Pmin)/0.1); weighted
mean/std with a floor of 0.05 x the box width) and refitted every 6 evaluations.
The box is imposed as linear constraints in PCA coordinates (no clipping).
A box item NAME=v:v fixes that parameter: PCA, GP and TuRBO then work on the
remaining free parameters (at least two), e.g. gh=1:1 for a fixed-cusp search.
Every evaluation calls agama.setRandomSeed(42) first (DECISIONS 2026-10-28,
2026-11-03), exactly as run_q_profile.py, so known seed-42 evaluations of Q-profile
states (--prior-states) enter the GP and the PCA without recomputation.

Schedule: --n-sobol scrambled Sobol points in the box, then TuRBO steps until
--budget new evaluations. Rows go to localpca_* history files that no pool glob
reads; orbit libraries with penalty <= --save-below (Q != 1) are saved into the
shared directory as by run_q_profile.py. search_state.json is rewritten after every
evaluation; --resume continues, a larger --budget extends. --dry-run and
--summarize need only the standard library. See doc/ai/harness/orblib_exp.md,
section "Local PCA search".
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import socket
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_q_profile as rqp  # noqa: E402

rsm = rqp.rsm
ConfigError, StopRun, RunFailed = rqp.ConfigError, rqp.StopRun, rqp.RunFailed

NAMES = ('Q', 'gh', 'rh', 'rho0')
LOG_NAMES = ('rh', 'rho0')
DEFAULTS = dict(opt_seed=42, budget=60, n_sobol=8, sobol_seed=20261103, subsample_seed=20261001,
                tr_init=0.6, tr_min=0.5 ** 7, tr_max=1.6, success_tol=3, failure_tol=4, improve_rel=1e-4,
                refit_every=6, target_fraction=0.3, min_good=10, cutoff_start=0.6, weight_temp=0.1,
                std_floor_frac=0.05, dup_tol=1e-6, max_dup_run=50)
SECONDS_PER_EVAL = 550.0
PREFIX = 'localpca'
STATE_NAME = 'search_state.json'
TABLE_NAME = 'search.tsv'
REPORT_NAME = 'report_localpca'
TABLE_COLUMNS = ('id', 'kind', 'Q', 'gh', 'rh', 'rho0', 'seed', 'status', 'penalty', 'upsilon', 'wall_s',
                 'orbit_s', 'tr_length', 'basis', 'library', 'source', 'finished')
CONTEXT_KEYS = ('exp_id', 'geom_hash', 'evaluation_context', 'harness_sha256')


# --------------------------------------------------------------------------
# Coordinates and the local box (standard library)
# --------------------------------------------------------------------------
def to_x(params):
    return [float(params['Q']), float(params['gh']), math.log10(params['rh']), math.log10(params['rho0'])]


def from_x(x):
    return dict(Q=float(x[0]), gh=float(x[1]), rh=10 ** float(x[2]), rho0=10 ** float(x[3]))


def box_x(box):
    return (to_x({k: box[k][0] for k in NAMES}), to_x({k: box[k][1] for k in NAMES}))


def free_names(box):
    return tuple(k for k in NAMES if box[k][1] > box[k][0])


def free_index(box):
    return [i for i, k in enumerate(NAMES) if box[k][1] > box[k][0]]


def fixed_params(box):
    return {k: box[k][0] for k in NAMES if box[k][1] == box[k][0]}


def parse_box(spec, bounds):
    """'Q=a:b,gh=a:b,rh=a:b,rho0=a:b' inside bounds_original; a == b fixes that parameter."""
    box = {}
    for item in str(spec or '').split(','):
        item = item.strip()
        if not item:
            continue
        name, sep, span = item.partition('=')
        lo, sep2, hi = span.partition(':')
        name = name.strip()
        if not sep or not sep2 or name not in NAMES:
            raise ConfigError(f'bad --box item {item!r} (expected NAME=lo:hi, NAME in {", ".join(NAMES)})')
        if name in box:
            raise ConfigError(f'--box: {name} given twice')
        try:
            box[name] = [float(lo), float(hi)]
        except ValueError:
            raise ConfigError(f'bad --box item {item!r}')
    if set(box) != set(NAMES):
        raise ConfigError(f'--box needs all of {", ".join(NAMES)}')
    for name, (lo, hi) in box.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ConfigError(f'--box {name}: need finite lo <= hi (lo == hi fixes the parameter)')
        blo, bhi = bounds[name]
        if lo < blo or hi > bhi:
            raise ConfigError(f'--box {name} [{lo:g}, {hi:g}] outside bounds_original [{blo:g}, {bhi:g}]')
    box = {k: box[k] for k in NAMES}
    if len(free_names(box)) < 2:
        raise ConfigError('--box: at least two parameters must have lo < hi')
    return box


def in_box(params, box, tol=1e-9):
    return all(box[k][0] - tol * max(1.0, abs(box[k][0])) <= params[k] <= box[k][1] + tol * max(1.0, abs(box[k][1]))
               for k in NAMES)


def clip_to_box(params, box):
    return {k: min(box[k][1], max(box[k][0], float(params[k]))) for k in NAMES}


def same_params(a, b):
    return all(math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12) for k in NAMES)


def box_distance(a, b, box):
    lo, hi = box_x(box)
    return math.sqrt(sum(((p - q) / (h - l)) ** 2 for p, q, l, h in zip(to_x(a), to_x(b), lo, hi) if h > l))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def read_prior_states(paths, box, bounds, incl):
    """Seed-42 evaluations of Q-profile states inside the box. Every state must be a
    seed-42 L24 profile at this incl with these bounds and one evaluation context."""
    priors, skipped, outside, context, sources = [], [], 0, None, []
    reference = {k: list(v) for k, v in bounds.items()}
    for path in paths:
        file = os.path.join(path, rqp.STATE_NAME) if os.path.isdir(path) else path
        try:
            with open(file, 'rb') as stream:
                raw = stream.read()
            state = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ConfigError(f'{file}: unreadable prior state ({error})')
        cfg = state.get('config') or {}
        name = os.path.basename(os.path.dirname(os.path.abspath(file)))
        problems = []
        if cfg.get('opt_seed') != DEFAULTS['opt_seed']:
            problems.append(f"opt_seed={cfg.get('opt_seed')}")
        if cfg.get('pot') != 'l24':
            problems.append(f"pot={cfg.get('pot')}")
        if not isinstance(cfg.get('incl'), (int, float)) or abs(cfg['incl'] - incl) > 1e-6:
            problems.append(f"incl={cfg.get('incl')}")
        if {k: list(v) for k, v in (cfg.get('bounds') or {}).items()} != reference:
            problems.append('bounds differ from bounds_original')
        contexts = {json.dumps({k: r.get(k) for k in CONTEXT_KEYS}, sort_keys=True) for r in state.get('runs') or []}
        ctx = json.loads(next(iter(contexts))) if len(contexts) == 1 else None
        if ctx is None or any(v is None for v in ctx.values()):
            problems.append('runs without one complete evaluation context')
        if problems:
            raise ConfigError(f'{file}: incompatible prior state: ' + '; '.join(problems))
        if context is None:
            context = ctx
        elif ctx != context:
            raise ConfigError(f'{file}: evaluation context differs from the other prior states')
        sources.append(dict(name=name, file=os.path.abspath(file), sha256=hashlib.sha256(raw).hexdigest()))
        for record in state.get('evaluations', []):
            where = f"{name}:{record.get('id')}"
            if record.get('kind') == 'prior' or record.get('source') != 'qprofile':
                continue
            params = dict(record['params'], Q=record['Q'])
            if not in_box(params, box):
                outside += 1
                continue
            status, penalty = record.get('status'), record.get('penalty')
            if status != 'ok':
                penalty = rqp.rejected_penalty(record)
                if penalty is None or not rqp.is_rejected_value(penalty):
                    skipped.append(f'{where} ({status})')
                    continue
                status = 'rejected'
            if any(same_params(p['params'], params) for p in priors):
                skipped.append(f'{where} (duplicate)')
                continue
            priors.append(dict(params={k: float(params[k]) for k in NAMES}, penalty=float(penalty),
                               upsilon=record.get('upsilon'), status=status, source=where))
    return priors, skipped, outside, context, sources


def read_initial_points(path, box, bounds, priors):
    if path is None:
        return [], None
    try:
        with open(path, 'rb') as stream:
            raw = stream.read()
        lines = raw.decode('utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        raise ConfigError(f'{path}: unreadable initial points ({error})')
    points = []
    for number, line in enumerate(lines, 1):
        tokens = line.partition('#')[0].split()
        if not tokens:
            continue
        where = f'{path}:{number}'
        if len(tokens) != len(NAMES):
            raise ConfigError(f'{where}: expected four coordinates: Q gh rh rho0')
        try:
            params = dict(zip(NAMES, map(float, tokens)))
        except ValueError:
            raise ConfigError(f'{where}: initial coordinates must be numeric')
        if not all(math.isfinite(v) for v in params.values()):
            raise ConfigError(f'{where}: initial coordinates must be finite')
        if not in_box(params, bounds, tol=0):
            raise ConfigError(f'{where}: initial point outside bounds_original')
        if not in_box(params, box, tol=0):
            raise ConfigError(f'{where}: initial point outside --box')
        if any(same_params(params, p) for p in points):
            raise ConfigError(f'{where}: duplicate initial point')
        if any(same_params(params, p['params']) for p in priors):
            raise ConfigError(f'{where}: initial point duplicates a prior observation')
        points.append(params)
    if not points:
        raise ConfigError(f'{path}: no points in --initial-points')
    return points, dict(name=os.path.basename(path), file=os.path.abspath(path), sha256=hashlib.sha256(raw).hexdigest())


def build_config(args, bounds):
    box = parse_box(args.box, bounds)
    budget = DEFAULTS['budget'] if args.budget is None else args.budget
    if budget < 1:
        raise ConfigError('--budget must be >= 1')
    if not 0 <= args.n_sobol <= budget:
        raise ConfigError('--n-sobol must be between 0 and --budget')
    priors, skipped, outside, context, sources = read_prior_states(args.prior_states or [], box, bounds, args.incl)
    initial, initial_source = read_initial_points(args.initial_points, box, bounds, priors)
    if len(initial) + args.n_sobol > budget:
        raise ConfigError('initial points + --n-sobol must not exceed --budget')
    return dict(DEFAULTS, incl=float(args.incl), pot='l24', box=box, bounds={k: list(v) for k, v in bounds.items()},
                budget=budget, n_sobol=args.n_sobol, sobol_seed=args.sobol_seed, subsample_seed=args.subsample_seed,
                priors=priors, prior_skipped=skipped, prior_outside=outside, prior_context=context,
                prior_sources=sources, initial_points=initial, initial_source=initial_source)


def check_prior_context(config, live):
    """Priors are usable only with the evaluation context they were computed in."""
    context = config.get('prior_context')
    if not context:
        return
    differ = sorted(k for k in CONTEXT_KEYS if k in live and live[k] != context.get(k))
    if differ:
        raise ConfigError('prior states were computed in another evaluation context: ' + ', '.join(
            f'{k} {context.get(k)} != {live[k]}' for k in differ))


def identity(config):
    result = {k: v for k, v in config.items() if k != 'budget'}
    result.setdefault('initial_points', [])
    result.setdefault('initial_source', None)
    return result


# --------------------------------------------------------------------------
# PCA basis and box constraints (numpy, imported lazily)
# --------------------------------------------------------------------------
def adaptive_cutoff(penalties, target_fraction, min_points, cutoff_start):
    """Port of the harness's adaptive_penalty_cutoff."""
    penalties = sorted(penalties)
    n = len(penalties)
    if n <= min_points:
        return penalties[-1]
    cutoff = penalties[min(n, max(min_points, math.ceil(target_fraction * n))) - 1]
    if cutoff > cutoff_start and sum(p <= cutoff_start for p in penalties) >= min_points:
        cutoff = cutoff_start
    if sum(p <= cutoff for p in penalties) < min_points:
        cutoff = penalties[min_points - 1]
    return cutoff


def fit_basis(records, config, at):
    """Full-rank weighted PCA of the ok points over the free parameters, as in the harness."""
    import numpy
    free = free_index(config['box'])
    x_fixed = box_x(config['box'])[0]
    lo, hi = (numpy.array(v)[free] for v in box_x(config['box']))
    d = len(free)
    ok = [r for r in records if r['status'] == 'ok']
    if len(ok) >= 2:
        P = numpy.array([r['penalty'] for r in ok])
        X = numpy.array([to_x(r['params']) for r in ok])[:, free]
        cutoff = adaptive_cutoff(P.tolist(), config['target_fraction'], config['min_good'], config['cutoff_start'])
        good = P <= cutoff
        Xg, Pg = X[good], P[good]
        weights = numpy.exp(-(Pg - Pg.min()) / config['weight_temp'])
        mean = numpy.average(Xg, weights=weights, axis=0)
        std = numpy.sqrt(numpy.average((Xg - mean) ** 2, weights=weights, axis=0))
        std = numpy.maximum(std, config['std_floor_frac'] * (hi - lo))
        scaled = (Xg - mean) / std
        center = scaled.mean(axis=0)
        _, singular, components = numpy.linalg.svd(scaled - center, full_matrices=True)
        variance = numpy.zeros(d)
        variance[:len(singular)] = singular ** 2
        n_good = int(good.sum())
    else:
        cutoff, n_good = None, len(ok)
        mean, std = (lo + hi) / 2, hi - lo
        center, components, variance = numpy.zeros(d), numpy.eye(d), numpy.zeros(d)
    rows = numpy.arange(d)
    sign = numpy.sign(components[rows, numpy.argmax(numpy.abs(components), axis=1)])
    components = components * numpy.where(sign == 0, 1.0, sign)[:, None]
    corners = numpy.array(list(itertools.product(*zip(lo, hi))))
    Z = ((corners - mean) / std - center) @ components.T
    explained = variance / variance.sum() if variance.sum() > 0 else numpy.full(d, 1.0 / d)
    return dict(at=at, n_points=len(ok), n_good=n_good, cutoff=None if cutoff is None else float(cutoff),
                mean=mean.tolist(), std=std.tolist(), center=center.tolist(), components=components.tolist(),
                explained=explained.tolist(), zlo=Z.min(axis=0).tolist(), zhi=Z.max(axis=0).tolist(),
                free=free, x_fixed=x_fixed)


def _arrays(basis):
    """Stored bases of all-free searches (before fixed parameters) have no free/x_fixed keys."""
    import numpy
    a = {k: numpy.array(basis[k]) for k in ('mean', 'std', 'center', 'components', 'zlo', 'zhi')}
    a['free'] = basis.get('free', list(range(len(NAMES))))
    a['x_fixed'] = numpy.array(basis.get('x_fixed', [0.0] * len(NAMES)), dtype=float)
    return a


def to_u(params, basis):
    """Physical parameters -> normalised PCA coordinates ([0,1]^d = PCA box of the local box)."""
    import numpy
    a = _arrays(basis)
    z = ((numpy.array(to_x(params))[a['free']] - a['mean']) / a['std'] - a['center']) @ a['components'].T
    return ((z - a['zlo']) / (a['zhi'] - a['zlo'])).tolist()


def x_from_u(u, basis):
    """Normalised PCA coordinates -> full x; fixed parameters take their box value."""
    import numpy
    a = _arrays(basis)
    z = a['zlo'] + numpy.asarray(u, dtype=float) * (a['zhi'] - a['zlo'])
    x = a['x_fixed'].copy()
    x[a['free']] = a['mean'] + a['std'] * (z @ a['components'] + a['center'])
    return x.tolist()


def box_constraints(basis, box):
    """Rows A, rhs b with A u >= b  <=>  the point lies in the local box (x linear in u)."""
    import numpy
    a = _arrays(basis)
    lo, hi = (numpy.array(v)[a['free']] for v in box_x(box))
    M = ((a['zhi'] - a['zlo'])[:, None] * a['components']) * a['std'][None, :]
    offset = a['mean'] + a['std'] * (a['zlo'] @ a['components'] + a['center'])
    return numpy.vstack([M.T, -M.T]).tolist(), numpy.concatenate([lo - offset, offset - hi]).tolist()


def feasible(u, A, b, lo, hi, tol=1e-9):
    import numpy
    u = numpy.asarray(u, dtype=float)
    return bool(numpy.all(u >= numpy.asarray(lo) - tol) and numpy.all(u <= numpy.asarray(hi) + tol)
                and numpy.all(numpy.asarray(A) @ u >= numpy.asarray(b) - tol * numpy.maximum(1.0, numpy.abs(b))))


def feasible_point(target, A, b, lo, hi, seed, fallback):
    """Feasible point of the trust region nearest to `target` among uniform samples; `fallback` if none."""
    import numpy
    rng = numpy.random.default_rng(seed)
    lo, hi, A, b = (numpy.asarray(v, dtype=float) for v in (lo, hi, A, b))
    target = numpy.asarray(target, dtype=float)
    for _ in range(20):
        U = lo + rng.random((4096, len(lo))) * (hi - lo)
        ok = numpy.all(U @ A.T >= b - 1e-9 * numpy.maximum(1.0, numpy.abs(b)), axis=1)
        if ok.any():
            U = U[ok]
            return U[numpy.argmin(numpy.sum((U - target) ** 2, axis=1))].tolist()
    return list(fallback)


# --------------------------------------------------------------------------
# Default numerical back-ends (torch/BoTorch only on the run host)
# --------------------------------------------------------------------------
def sobol_points(n, seed, d=len(NAMES)):
    import warnings
    from scipy.stats import qmc
    if n <= 0:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return qmc.Sobol(d=d, scramble=True, seed=seed).random(n).tolist()


def propose_bo(points, values, center, lo, hi, A, b, seed):
    """qLogNEI in the trust region ∩ local box (linear constraints); GP on y = -log(penalty).
    None => the caller falls back to a feasible random point."""
    if len(points) < 3:
        return None
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
        d = len(lo)
        unit = torch.tensor([[0.0] * d, [1.0] * d], dtype=dtype)
        model = SingleTaskGP(X, Y, likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
                             input_transform=Normalize(d=d, bounds=unit),
                             outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acqf = qLogNoisyExpectedImprovement(model=model, X_baseline=X, prune_baseline=True)
        index = torch.arange(d)
        constraints = [(index, torch.tensor(row, dtype=dtype), float(rhs)) for row, rhs in zip(A, b)]
        candidate, _ = optimize_acqf(acq_function=acqf, bounds=torch.tensor([lo, hi], dtype=dtype), q=1,
                                     num_restarts=10, raw_samples=2048, inequality_constraints=constraints)
        return [float(v) for v in candidate[0]]
    except Exception as error:
        print(f'[localpca] BO proposal failed ({error!r}); feasible random point in the trust region', flush=True)
        return None


def self_test(box, proposer=propose_bo, n=30, seed=1):
    """Constrained proposal on a synthetic objective (no AGAMA): the run host's BoTorch must
    return a feasible acquisition point, otherwise the search would degrade to random points."""
    import random
    rng = random.Random(seed)
    lo, hi = box_x(box)
    target = [a + 0.3 * (c - a) for a, c in zip(lo, hi)]
    records = []
    for _ in range(n):
        x = [rng.uniform(a, c) for a, c in zip(lo, hi)]
        penalty = 1.5 + sum(((v - t) / (c - a)) ** 2 for v, t, a, c in zip(x, target, lo, hi) if c > a)
        records.append(dict(params=from_x(x), status='ok', penalty=penalty))
    config = dict(DEFAULTS, box=box)
    basis = fit_basis(records, config, 0)
    A, b = box_constraints(basis, box)
    best = min(records, key=lambda r: r['penalty'])
    center = to_u(best['params'], basis)
    tr_lo = [max(0.0, c - 0.3) for c in center]
    tr_hi = [min(1.0, c + 0.3) for c in center]
    u = proposer([to_u(r['params'], basis) for r in records], [r['penalty'] for r in records],
                 center, tr_lo, tr_hi, A, b, seed)
    ok = u is not None and feasible(u, A, b, tr_lo, tr_hi)
    return ok, (None if u is None else clip_to_box(from_x(x_from_u(u, basis)), box))


# --------------------------------------------------------------------------
# Search driver (back-ends injected: tests run it without AGAMA/torch)
# --------------------------------------------------------------------------
class Search:
    def __init__(self, config, state_dir, evaluator, proposer=propose_bo, sobol=sobol_points, log=print,
                 max_failures=5):
        self.cfg = config
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, STATE_NAME)
        self.table = os.path.join(state_dir, TABLE_NAME)
        self.evaluator = evaluator
        self.proposer = proposer
        self.sobol = sobol
        self.log = log
        self.max_failures = max_failures
        self.failures = 0
        self.state = None

    # ---- state -----------------------------------------------------------
    def open(self, resume, run_info=None):
        os.makedirs(self.state_dir, exist_ok=True)
        if os.path.exists(self.path):
            if not resume:
                raise ConfigError(f'{self.path} exists: pass --resume or choose another --state-dir')
            with open(self.path) as stream:
                self.state = json.load(stream)
            self._check(self.state['config'])
        else:
            if resume:
                raise ConfigError(f'--resume: no {self.path}')
            self.state = dict(version=1, config=self.cfg, config_hash=rqp.config_hash(self.cfg), created=rqp.now(),
                              status='running', evaluations=[], bases=[], duplicates=0, dup_run=0,
                              tr=dict(length=self.cfg['tr_init'], succ=0, fail=0, restarts=0), runs=[])
            for prior in self.cfg['priors']:
                self._append(dict(prior, kind='prior', seed=None, finished=None))
        if run_info:
            self.state['runs'].append(dict(run_info, started=rqp.now()))
        self.state['status'] = 'running'
        self.save()

    def _check(self, stored):
        mine, theirs = identity(self.cfg), identity(stored)
        differ = sorted(k for k in set(mine) | set(theirs)
                        if json.dumps(mine.get(k), sort_keys=True) != json.dumps(theirs.get(k), sort_keys=True))
        if differ:
            raise ConfigError(f'configuration differs from the state file: {", ".join(differ)}')
        if self.cfg['budget'] < stored['budget']:
            raise ConfigError(f"--budget {self.cfg['budget']} < stored {stored['budget']}: the budget can only grow")
        if self.cfg['budget'] > stored['budget']:
            self.log(f"[localpca] budget extended: {stored['budget']} -> {self.cfg['budget']}")
        self.state['config'], self.state['config_hash'] = self.cfg, rqp.config_hash(self.cfg)

    def save(self):
        self.state['updated'] = rqp.now()
        rqp.atomic_json(self.path, self.state)

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
    def new_records(self):
        return [r for r in self.state['evaluations'] if r['kind'] != 'prior']

    def best(self):
        ok = [r for r in self.state['evaluations'] if r['status'] == 'ok']
        return min(ok, key=lambda r: r['penalty']) if ok else None

    def gp_records(self):
        return [r for r in self.state['evaluations'] if r['status'] in ('ok', 'rejected')]

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, params, kind, extra=None):
        meta = dict(id=len(self.state['evaluations']), branch=None, kind=kind)
        self.log(f'[localpca] eval {meta["id"]}: {kind} ' + ' '.join(f'{k}={params[k]:.6g}' for k in NAMES))
        result = self.evaluator(params['Q'], dict(params), self.cfg['opt_seed'], meta)
        record = dict(params=dict(params), kind=kind, seed=self.cfg['opt_seed'],
                      status=result.get('status', 'failed'), penalty=result.get('penalty'),
                      upsilon=result.get('upsilon'), wall_s=result.get('wall_s'), orbit_s=result.get('orbit_s'),
                      library=result.get('library'), error=result.get('error'), source=PREFIX,
                      finished=rqp.now(), **(extra or {}))
        if record['status'] == 'ok' and not (record['penalty'] and record['penalty'] > 0):
            record['status'] = 'failed'
        if record['status'] == 'rejected' and not rqp.is_rejected_value(record['penalty']):
            record['status'] = 'failed'
        self.failures = 0 if record['status'] in ('ok', 'rejected') else self.failures + 1
        return record

    def _record(self, record):
        self._append(record)
        self.save()
        self.log(f'[localpca]   → status={record["status"]} penalty={record["penalty"]} '
                 f'upsilon={record["upsilon"]} library={record.get("library")}')
        if self.failures >= self.max_failures:
            raise RunFailed(f'{self.failures} consecutive failed evaluations')
        return record

    # ---- schedule --------------------------------------------------------
    def run(self):
        self.initial_stage()
        self.sobol_stage()
        while len(self.new_records()) < self.cfg['budget']:
            if self.state['dup_run'] >= self.cfg['max_dup_run']:
                self.state['status'] = 'stalled'
                self.save()
                self.log(f"[localpca] {self.state['dup_run']} duplicate proposals in a row: stopped")
                return
            self.step()
        self.state['status'] = 'done'
        self.save()

    def initial_stage(self):
        for j, params in enumerate(self.cfg.get('initial_points', [])):
            if len(self.new_records()) >= self.cfg['budget']:
                return
            if any(r['kind'] == 'initial' and r.get('initial_index') == j for r in self.state['evaluations']):
                continue
            self._record(self.evaluate(params, 'initial', extra=dict(initial_index=j)))

    def sobol_stage(self):
        lo, hi = box_x(self.cfg['box'])
        free = free_index(self.cfg['box'])
        for j, s in enumerate(self.sobol(self.cfg['n_sobol'], self.cfg['sobol_seed'], len(free))):
            if len(self.new_records()) >= self.cfg['budget']:
                return
            if any(r['kind'] == 'sobol' and r.get('sobol_index') == j for r in self.state['evaluations']):
                continue
            x = list(lo)
            for i, t in zip(free, s):
                x[i] = lo[i] + t * (hi[i] - lo[i])
            params = clip_to_box(from_x(x), self.cfg['box'])
            self._record(self.evaluate(params, 'sobol', extra=dict(sobol_index=j)))

    def basis(self):
        current = self.state['bases'][-1] if self.state['bases'] else None
        n_new = len(self.new_records())
        if current is None or n_new - current['at'] >= self.cfg['refit_every']:
            current = fit_basis(self.state['evaluations'], self.cfg, n_new)
            self.state['bases'].append(current)
            self.save()
            self.log(f"[localpca] PCA basis {len(self.state['bases']) - 1} at {n_new} new evaluations: "
                     f"{current['n_good']} good points, explained "
                     + ', '.join(f'{v:.3f}' for v in current['explained']))
        return current

    def step(self):
        best = self.best()
        if best is None:
            raise RunFailed('no successful evaluation to centre the trust region')
        basis = self.basis()
        box, tr = self.cfg['box'], self.state['tr']
        A, b = box_constraints(basis, box)
        records = self.gp_records()
        center = to_u(best['params'], basis)
        lo = [max(0.0, c - tr['length'] / 2) for c in center]
        hi = [min(1.0, c + tr['length'] / 2) for c in center]
        n_new = len(self.new_records())
        seed = (self.cfg['sobol_seed'] + 17 * n_new + self.state['duplicates']) % (2 ** 31)
        u = self.proposer([to_u(r['params'], basis) for r in records],
                          [min(r['penalty'], rsm.FAILED_PENALTY) for r in records], center, lo, hi, A, b, seed)
        proposal = 'acquisition'
        if u is None or not feasible(u, A, b, lo, hi):
            proposal = 'fallback' if u is None else 'projected'
            u = feasible_point(center if u is None else u, A, b, lo, hi, seed, center)
        params = clip_to_box(from_x(x_from_u(u, basis)), box)
        if any(box_distance(params, r['params'], box) < self.cfg['dup_tol'] for r in self.state['evaluations']):
            self.state['duplicates'] += 1
            self.state['dup_run'] += 1
            self.update_tr(False)
            self.save()
            return
        self.state['dup_run'] = 0
        record = self._record(self.evaluate(params, 'bo', extra=dict(tr_length=tr['length'], proposal=proposal,
                                                                      basis=len(self.state['bases']) - 1)))
        self.update_tr(record['status'] == 'ok'
                       and record['penalty'] < best['penalty'] * (1 - self.cfg['improve_rel']))
        self.save()

    def update_tr(self, improved):
        tr = self.state['tr']
        if improved:
            tr['succ'], tr['fail'] = tr['succ'] + 1, 0
        else:
            tr['succ'], tr['fail'] = 0, tr['fail'] + 1
        if tr['succ'] >= self.cfg['success_tol']:
            tr['length'], tr['succ'] = min(2 * tr['length'], self.cfg['tr_max']), 0
        if tr['fail'] >= self.cfg['failure_tol']:
            tr['length'], tr['fail'] = tr['length'] / 2, 0
        if tr['length'] < self.cfg['tr_min']:
            tr['length'], tr['restarts'] = self.cfg['tr_init'], tr['restarts'] + 1
            self.log(f"[localpca] trust region collapsed: restart {tr['restarts']} around the best point")


# --------------------------------------------------------------------------
# Summary (standard library only)
# --------------------------------------------------------------------------
def _edges(params, limits, rel=1e-3):
    hits = []
    for k in NAMES:
        lo, hi = limits[k]
        if hi <= lo:
            continue
        if k in LOG_NAMES:
            lo, hi, value = math.log10(lo), math.log10(hi), math.log10(params[k])
        else:
            value = params[k]
        width = hi - lo
        if value <= lo + rel * width:
            hits.append(f'{k}=lo')
        elif value >= hi - rel * width:
            hits.append(f'{k}=hi')
    return hits


def _point(record, cfg):
    if record is None:
        return None
    upsilon = record.get('upsilon')
    return dict(id=record['id'], kind=record['kind'], source=record.get('source'), penalty=record['penalty'],
                upsilon=upsilon, params=record['params'],
                rho0_upsilon=record['params']['rho0'] * upsilon if upsilon else None,
                box_edges=_edges(record['params'], cfg['box']), bounds_edges=_edges(record['params'], cfg['bounds']))


def summarize(state):
    cfg, evaluations = state['config'], state['evaluations']
    ok = sorted((r for r in evaluations if r['status'] == 'ok'), key=lambda r: r['penalty'])
    new = [r for r in evaluations if r['kind'] != 'prior']
    best = ok[0] if ok else None
    best_prior = next((r for r in ok if r['kind'] == 'prior'), None)
    best_new = next((r for r in ok if r['kind'] != 'prior'), None)
    near = {}
    for delta in (0.25, 0.5):
        rows = [r for r in ok if best and r['penalty'] <= best['penalty'] + delta]
        counts = {}
        for r in rows:
            for edge in _edges(r['params'], cfg['box']):
                counts[edge] = counts.get(edge, 0) + 1
        near[str(delta)] = dict(n=len(rows), box_edges=counts)
    return dict(status=state.get('status'), updated=state.get('updated'), opt_seed=cfg['opt_seed'], box=cfg['box'],
                fixed=fixed_params(cfg['box']), budget=cfg['budget'], n_new=len(new), n_priors=len(evaluations) - len(new),
                evaluation_kinds={k: sum(r['kind'] == k for r in new) for k in ('initial', 'sobol', 'bo')},
                n_failed=sum(1 for r in new if r['status'] == 'failed'),
                n_rejected=sum(1 for r in new if r['status'] == 'rejected'),
                libraries_saved=sum(1 for r in new if (r.get('library') or {}).get('name')),
                duplicates=state.get('duplicates', 0), tr=state.get('tr'),
                proposals={k: sum(1 for r in new if r.get('proposal') == k)
                           for k in ('acquisition', 'projected', 'fallback')},
                best=_point(best, cfg), best_prior=_point(best_prior, cfg), best_new=_point(best_new, cfg),
                improvement=(best_prior['penalty'] - best_new['penalty']) if best_prior and best_new else None,
                near_best=near, top=[_point(r, cfg) for r in ok[:10]],
                bases=[dict(at=b['at'], n_good=b['n_good'], cutoff=b['cutoff'], explained=b['explained'],
                            components=b['components']) for b in state.get('bases', [])],
                note='one IC realisation (seed %d); penalty is not chi^2; differences between distant '
                     'parameters carry the realisation scatter of two independent evaluations' % cfg['opt_seed'])


def _fmt_point(label, point):
    if not point:
        return [f'  {label}: -']
    p = point['params']
    return [f"  {label}: P={point['penalty']:.6f} ({point['kind']}, id {point['id']}, {point['source']}) "
            f"Q={p['Q']:.6g} gh={p['gh']:.6g} rh={p['rh']:.6g} rho0={p['rho0']:.6g} "
            f"Upsilon={point['upsilon']} rho0*Ups={point['rho0_upsilon']}",
            f"    box edges: {','.join(point['box_edges']) or '-'}; bounds_original edges: "
            f"{','.join(point['bounds_edges']) or '-'}"]


def _fmt_fixed(box):
    fixed = fixed_params(box)
    return ''.join(f', fixed {k}={v:g}' for k, v in fixed.items())


def format_summary(summary):
    tr = summary.get('tr') or {}
    lines = [f"Local PCA search: status={summary['status']} new evaluations={summary['n_new']}/{summary['budget']} "
             f"priors={summary['n_priors']} failed={summary['n_failed']} rejected={summary['n_rejected']} "
             f"duplicates={summary['duplicates']} libraries saved={summary['libraries_saved']} "
             f"(seed {summary['opt_seed']}, updated {summary['updated']})",
             '  box: ' + ', '.join(f'{k} {lo:g}..{hi:g}' for k, (lo, hi) in summary['box'].items())
             + _fmt_fixed(summary['box']),
             f"  trust region: length={tr.get('length')} restarts={tr.get('restarts')}; proposals "
             + ', '.join(f'{k}={v}' for k, v in (summary.get('proposals') or {}).items())]
    lines.append('  new evaluations by kind: ' + ' '.join(
        f'{k}={v}' for k, v in summary.get('evaluation_kinds', {}).items()))
    lines += _fmt_point('best', summary['best'])
    lines += _fmt_point('best prior', summary['best_prior'])
    lines += _fmt_point('best new', summary['best_new'])
    if summary['improvement'] is not None:
        lines.append(f"  best prior - best new = {summary['improvement']:+.6f}")
    for delta, row in summary['near_best'].items():
        lines.append(f"  within +{delta}: {row['n']} points; box edges {row['box_edges'] or '-'}")
    for b in summary['bases']:
        lines.append(f"  PCA basis at {b['at']}: n_good={b['n_good']} explained="
                     + ', '.join(f'{v:.3f}' for v in b['explained']))
    lines.append('  top: ' + '; '.join(f"{t['penalty']:.4f}@Q={t['params']['Q']:.4g}" for t in summary['top']))
    lines.append(f"  note: {summary['note']}")
    return '\n'.join(lines) + '\n'


def write_summary(state_dir):
    with open(os.path.join(state_dir, STATE_NAME)) as stream:
        summary = summarize(json.load(stream))
    rqp.atomic_json(os.path.join(state_dir, REPORT_NAME + '.json'), summary)
    text = format_summary(summary)
    with open(os.path.join(state_dir, REPORT_NAME + '.txt'), 'w') as stream:
        stream.write(text)
    return text


def format_dry_run(config):
    priors = config['priors']
    lines = [f"Local PCA search plan: incl={config['incl']} pot={config['pot']} seed={config['opt_seed']} "
             f"(agama.setRandomSeed before every evaluation)",
             '  box: ' + ', '.join(f'{k} {lo:g}..{hi:g}' for k, (lo, hi) in config['box'].items())
             + ' (Upsilon searched internally)',
             f"  PCA: {len(free_names(config['box']))} components of ("
             + ', '.join(k if k not in LOG_NAMES else f'log10 {k}' for k in free_names(config['box']))
             + f"){_fmt_fixed(config['box'])}, refit every {config['refit_every']} evaluations"]
    lines.append(f"  prior states: {len(config['prior_sources'])} "
                 f"({', '.join(s['name'] for s in config['prior_sources']) or '-'})")
    if priors:
        best = min((p for p in priors if p['status'] == 'ok'), key=lambda p: p['penalty'], default=None)
        lines.append(f"  priors in the box: {len(priors)} (rejected {sum(p['status'] == 'rejected' for p in priors)}); "
                     f"outside: {config['prior_outside']}; skipped: {len(config['prior_skipped'])}")
        if best:
            p = best['params']
            lines.append(f"  best prior: P={best['penalty']:.6g} Q={p['Q']:.6g} gh={p['gh']:.6g} rh={p['rh']:.6g} "
                         f"rho0={p['rho0']:.6g} ({best['source']})")
        lines.append(f"  prior context: harness {config['prior_context']['harness_sha256'][:12]}, "
                     f"evaluation_context {config['prior_context']['evaluation_context'][:12]}")
    else:
        lines.append('  priors in the box: 0')
    initial = config.get('initial_points', [])
    if initial:
        source = config['initial_source']
        lines.append(f"  initial points: {len(initial)} ({source['file']}, sha256 {source['sha256'][:12]})")
        for j, params in enumerate(initial):
            lines.append(f'    initial {j}: ' + ' '.join(f'{k}={params[k]:.12g}' for k in NAMES))
    n_bo = config['budget'] - len(initial) - config['n_sobol']
    start = f'{len(initial)} initial + ' if initial else ''
    lines.append(f"  evaluations: {config['budget']} ({start}{config['n_sobol']} Sobol + {n_bo} TuRBO), "
                 f"~{config['budget'] * SECONDS_PER_EVAL / 3600:.1f} h at {SECONDS_PER_EVAL:.0f} s/evaluation")
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], allow_abbrev=False)
    parser.add_argument('--incl', type=float, default=90.0)
    parser.add_argument('--box', default=None,
                        help='Q=lo:hi,gh=lo:hi,rh=lo:hi,rho0=lo:hi inside bounds_original (lo == hi fixes it)')
    parser.add_argument('--budget', type=int, default=None,
                        help=f"New evaluations (default {DEFAULTS['budget']}; on --resume: stored, larger extends)")
    parser.add_argument('--n-sobol', type=int, default=DEFAULTS['n_sobol'])
    parser.add_argument('--initial-points', default=None,
                        help='File of Q gh rh rho0 rows to evaluate before Sobol (counts towards --budget)')
    parser.add_argument('--prior-states', nargs='*', default=[],
                        help='Q-profile state directories (or profile_state.json) whose seed-42 points enter the GP')
    parser.add_argument('--state-dir', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--orblib-dir', default=None, help='Isolated store directory of the harness')
    parser.add_argument('--shared-orblib', default=None, help='Shared library directory (saved .npz)')
    parser.add_argument('--n_threads', type=int, default=None)
    parser.add_argument('--save-below', type=float, default=3.5)
    parser.add_argument('--min-free-gb', type=float, default=40.0)
    parser.add_argument('--sobol-seed', type=int, default=DEFAULTS['sobol_seed'])
    parser.add_argument('--subsample-seed', type=int, default=DEFAULTS['subsample_seed'])
    parser.add_argument('--max-failures', type=int, default=5)
    parser.add_argument('--dry-run', action='store_true', help='Print the plan (standard library only)')
    parser.add_argument('--self-test', action='store_true',
                        help='Check the constrained BoTorch proposal on a synthetic objective (no AGAMA)')
    parser.add_argument('--summarize', default=None, metavar='STATE_DIR',
                        help='Write report_localpca.json/.txt of a state directory (standard library only)')
    args = parser.parse_args(argv)
    if args.summarize is None:
        if not args.box:
            parser.error('--box is required')
        if not (args.dry_run or args.self_test) and not (args.state_dir and args.suffix and args.orblib_dir and args.shared_orblib):
            parser.error('--state-dir, --suffix, --orblib-dir and --shared-orblib are required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.summarize is not None:
        print(write_summary(args.summarize), end='')
        return 0
    if args.self_test:
        try:
            box = parse_box(args.box, rsm.script_bounds())
        except ConfigError as error:
            print(f'ERROR: {error}', file=sys.stderr)
            return 2
        ok, params = self_test(box)
        print(f"[localpca] self-test of the constrained proposal: {'ok' if ok else 'FAILED'} {params}")
        return 0 if ok else 1
    harness_sha256 = rqp.sha256_file(rsm.EXP_SCRIPT)
    try:
        config = build_config(args, rsm.script_bounds())
        check_prior_context(config, dict(harness_sha256=harness_sha256))
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    if args.dry_run:
        print(format_dry_run(config), end='')
        return 0
    os.makedirs(args.state_dir, exist_ok=True)
    state_path = os.path.join(args.state_dir, STATE_NAME)
    if args.resume and args.budget is None and os.path.exists(state_path):
        with open(state_path) as stream:
            config['budget'] = json.load(stream)['config']['budget']
    stop_marker = os.path.join(args.orblib_dir, '.storage', 'STOP')
    if os.path.exists(stop_marker):
        os.remove(stop_marker)
    evaluate, mod, _ = rqp.build_evaluator(args, args.state_dir, prefix=PREFIX)
    live = dict(exp_id=mod.EXP_ID, geom_hash=mod.GEOM_HASH, evaluation_context=mod.EVALUATION_CONTEXT,
                harness_sha256=harness_sha256)
    try:
        check_prior_context(config, live)
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    search = Search(config, args.state_dir, evaluate, max_failures=args.max_failures)
    run_info = dict(host=os.environ.get('HOSTNAME_SUFFIX', socket.gethostname()), suffix=args.suffix,
                    hostname_proc=mod.hostname_proc, driver_sha256=rqp.sha256_file(__file__),
                    omp_threads=getattr(mod, 'AGAMA_ORBIT_THREADS', None), **live)
    try:
        search.open(args.resume, run_info)
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    code = 0
    try:
        search.run()
    except StopRun as error:
        search.state['status'], code = f'stopped: {error}', 75
    except RunFailed as error:
        search.state['status'], code = f'failed: {error}', 1
    except Exception as error:
        traceback.print_exc()
        search.state['status'], code = f'failed: {error!r}', 1
    search.save()
    print(write_summary(args.state_dir), end='', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
