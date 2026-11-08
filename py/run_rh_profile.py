#!/usr/bin/env python3
"""Diagnostic rh profiles at beta=3,4,7,10, with Q/gh/rho0/alpha re-optimised.

Reuses the full-rank local PCA/TuRBO engine (four free coordinates, no discarded
PC), the L24 evaluator and seed-42 CRN protocol. Each (beta, rh) cell has its own
search_state.json; the manifest locks inputs, evaluation context and code hashes.
No previously measured penalties are imported. --dry-run/--summarize use stdlib;
--self-test tests a constrained BoTorch proposal without importing AGAMA.
"""
import argparse
import json
import math
import os
import socket
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_local_pca as lp  # noqa: E402

rqp, rsm = lp.rqp, lp.rsm
ConfigError, StopRun, RunFailed = lp.ConfigError, lp.StopRun, lp.RunFailed
BETAS = (3.0, 4.0, 7.0, 10.0)
RH_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
STATE_NAME = 'rh_profile_state.json'
DEFAULT_BUDGET = 24
DEFAULT_SOBOL = 8


def parse_grid(spec, bounds, name, choices=None):
    try:
        values = [float(x) for x in spec.split(',')]
    except ValueError:
        raise ConfigError(f'{name}: expected comma-separated numbers')
    if not values or any(not math.isfinite(x) or not bounds[0] <= x <= bounds[1] for x in values):
        raise ConfigError(f'{name}: finite values in [{bounds[0]:g},{bounds[1]:g}] required')
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ConfigError(f'{name}: values must be strictly increasing')
    if choices and any(x not in choices for x in values):
        raise ConfigError(f'{name}: choose from {",".join(f"{x:g}" for x in choices)}')
    return values


def build_config(args, bounds):
    bounds = lp.search_bounds(bounds, True)
    spec = args.profile_box or ','.join(f'{k}={bounds[k][0]}:{bounds[k][1]}' for k in lp.names_of(bounds))
    box = lp.parse_box(spec, bounds)
    grid = parse_grid(args.rh_grid, box['rh'], '--rh-grid')
    betas = parse_grid(args.beta_grid, (3, 10), '--beta-grid', BETAS)
    if not math.isfinite(args.incl) or not math.degrees(math.acos(0.69)) < args.incl <= 90:
        raise ConfigError('--incl must satisfy cos(incl)<0.69 and incl<=90 degrees')
    initial, source = lp.read_initial_points(args.profile_starts, box, bounds, [])
    if not initial:
        # A generic core start; real run handoffs may supply established finalists.
        p = lp.from_x([(a + b) / 2 for a, b in zip(*lp.box_x(box))])
        p['gh'] = box['gh'][0]
        initial = [p]
    budget = DEFAULT_BUDGET if args.cell_budget is None else args.cell_budget
    if args.cell_sobol < 0 or budget <= len(initial) + 1 + args.cell_sobol:
        raise ConfigError('--cell-budget must exceed starts + one continuation + --cell-sobol')
    if args.max_failures < 1:
        raise ConfigError('--max-failures must be >=1')
    return dict(version=1, incl=float(args.incl), box=box,
                bounds={k: list(v) for k, v in bounds.items()}, rh_grid=grid, beta_grid=betas,
                cell_budget=budget, cell_sobol=args.cell_sobol,
                sobol_seed=args.sobol_seed, subsample_seed=args.subsample_seed,
                initial_points=initial, initial_source=source, opt_seed=42, pot='l24')


def identity(config):
    return {k: v for k, v in config.items() if k != 'cell_budget'}


def check_config(config, stored):
    if identity(config) != identity(stored):
        raise ConfigError('profile configuration differs from the manifest')
    if config['cell_budget'] < stored['cell_budget']:
        raise ConfigError('cell budget can only grow')


def cell_name(beta, rh):
    # Full float repr avoids colliding custom grid values.
    return f'beta_{beta!r}_rh_{rh!r}'


def cells(config):
    # Start at the large-rh end; continue towards smaller radii in each branch.
    return [(beta, rh) for beta in config['beta_grid'] for rh in reversed(config['rh_grid'])]


def projected_starts(config, rh, continuation=None):
    starts = []
    for p in ([continuation] if continuation else []) + config['initial_points']:
        point = {k: float(p[k]) for k in lp.names_of(config['box'])}
        point['rh'] = rh
        if not any(lp.same_params(point, old) for old in starts):
            starts.append(point)
    return starts


def cell_config(config, beta, rh, continuation=None):
    box = {k: list(v) for k, v in config['box'].items()}
    box['rh'] = [rh, rh]
    return dict(lp.DEFAULTS, **lp.ALPHA_DEFAULTS, incl=config['incl'], pot='l24', box=box,
                bounds=config['bounds'], fit_alpha=True, pca_components=4, fixed_beta=beta,
                budget=config['cell_budget'], n_sobol=config['cell_sobol'],
                sobol_seed=config['sobol_seed'], subsample_seed=config['subsample_seed'],
                priors=[], prior_skipped=[], prior_outside=0, prior_context=None, prior_sources=[],
                initial_points=projected_starts(config, rh, continuation),
                initial_source=config['initial_source'])


class CellSearch(lp.Search):
    def evaluate(self, params, kind, extra=None):
        params = dict(params, beta=self.cfg['fixed_beta'])
        if params['rh'] != self.cfg['box']['rh'][0]:
            raise ConfigError('candidate changed the fixed rh')
        return super().evaluate(params, kind, extra)


class Profile:
    def __init__(self, config, state_dir, evaluator, search_factory=CellSearch, log=print, max_failures=5):
        self.cfg, self.state_dir, self.evaluator = config, state_dir, evaluator
        self.search_factory, self.log, self.max_failures = search_factory, log, max_failures
        self.path = os.path.join(state_dir, STATE_NAME)
        self.state = None

    def open(self, resume, context):
        os.makedirs(self.state_dir, exist_ok=True)
        if os.path.exists(self.path):
            if not resume:
                raise ConfigError(f'{self.path} exists; use --resume')
            with open(self.path) as stream:
                self.state = json.load(stream)
            check_config(self.cfg, self.state['config'])
            if context != self.state['context']:
                raise ConfigError('profile evaluation context/code differs; use a separate run')
            self.state['config'] = self.cfg
        else:
            if resume:
                raise ConfigError(f'--resume: no {self.path}')
            self.state = dict(config=self.cfg, context=context, created=rqp.now(), runs=[])
        self.state['runs'].append(dict(started=rqp.now(), host=socket.gethostname(),
                                       cell_budget=self.cfg['cell_budget']))
        self.state['status'] = 'running'
        self.save()

    def save(self):
        self.state['updated'] = rqp.now()
        rqp.atomic_json(self.path, self.state)

    def cell_dir(self, beta, rh):
        return os.path.join(self.state_dir, cell_name(beta, rh))

    def best(self, beta, rh):
        path = os.path.join(self.cell_dir(beta, rh), lp.STATE_NAME)
        if not os.path.exists(path):
            return None
        with open(path) as stream:
            ok = [r for r in json.load(stream)['evaluations'] if r['status'] == 'ok']
        return min(ok, key=lambda r: r['penalty'])['params'] if ok else None

    def run(self):
        for beta, rh in cells(self.cfg):
            directory = self.cell_dir(beta, rh)
            path = os.path.join(directory, lp.STATE_NAME)
            resuming = os.path.exists(path)
            if resuming:
                with open(path) as stream:
                    stored = json.load(stream)
                # Continuation coordinates are fixed when a cell is first created.
                # Extending a neighbour's budget must not replace that provenance.
                cfg = cell_config(self.cfg, beta, rh)
                cfg['initial_points'] = stored['config']['initial_points']
            else:
                larger = [r for r in self.cfg['rh_grid'] if r > rh]
                previous = [b for b in self.cfg['beta_grid'] if b < beta]
                continuation = (self.best(beta, min(larger)) if larger else
                                self.best(max(previous), rh) if previous else None)
                cfg = cell_config(self.cfg, beta, rh, continuation)
            self.state['current_cell'] = dict(beta=beta, rh=rh)
            self.save()
            self.log(f'[rhprofile] beta={beta:g} rh={rh:g}: budget {cfg["budget"]}, full 4D nuisance search')

            def evaluate(Q, params, seed, meta):
                return self.evaluator(Q, params, seed, dict(meta, branch=cell_name(beta, rh)))

            search = self.search_factory(cfg, directory, evaluate, log=self.log, max_failures=self.max_failures)
            search.open(resuming, self.state['context'])
            try:
                search.initial_stage()
                search.sobol_stage()
                if search.best() is None:
                    search.state['status'] = 'no_valid_model'
                    self.log('[rhprofile] no successful start/Sobol model; cell has no measured minimum')
                else:
                    search.run()
            except StopRun as error:
                search.state['status'] = f'stopped: {error}'
                raise
            except Exception as error:
                search.state['status'] = f'failed: {error!r}'
                raise
            finally:
                search.save()
                write_summary(self.state_dir)
        self.state['status'] = 'done'
        self.save()


def summarize(state_dir):
    with open(os.path.join(state_dir, STATE_NAME)) as stream:
        manifest = json.load(stream)
    config, rows = manifest['config'], []
    for beta in config['beta_grid']:
        for rh in config['rh_grid']:
            row = dict(beta=beta, rh=rh, status='pending', n_eval=0, n_failed=0, n_rejected=0,
                       penalty=None, params=None, upsilon=None, rho0_upsilon=None,
                       boundary=[], last_improvement=None)
            path = os.path.join(state_dir, cell_name(beta, rh), lp.STATE_NAME)
            if os.path.exists(path):
                with open(path) as stream:
                    cell = json.load(stream)
                records = cell['evaluations']
                ok = [r for r in records if r['status'] == 'ok']
                row.update(status=cell['status'], n_eval=len(records),
                           n_failed=sum(r['status'] == 'failed' for r in records),
                           n_rejected=sum(r['status'] == 'rejected' for r in records))
                if ok:
                    best = min(ok, key=lambda r: r['penalty'])
                    row.update(penalty=best['penalty'], params=best['params'], upsilon=best['upsilon'],
                               rho0_upsilon=best['params']['rho0'] * best['upsilon'],
                               boundary=lp._edges(best['params'], cell['config']['box']),
                               last_improvement=best['id'])
            rows.append(row)
    minima = {}
    for beta in config['beta_grid']:
        valid = [r for r in rows if r['beta'] == beta and r['penalty'] is not None]
        best = min(valid, key=lambda r: r['penalty']) if valid else None
        minima[str(beta)] = None if best is None else dict(
            rh=best['rh'], penalty=best['penalty'],
            at_grid_edge=best['rh'] in (config['rh_grid'][0], config['rh_grid'][-1]))
    return dict(status=manifest['status'], config=config, rows=rows, minima=minima,
                interpretation='Budget-limited upper estimates of minima; penalty is not chi-squared. '
                               'Grid-edge or nuisance-boundary minima are not closed constraints.')


def write_summary(state_dir):
    summary = summarize(state_dir)
    columns = ('beta', 'rh', 'status', 'n_eval', 'n_failed', 'n_rejected', 'penalty',
               'Q', 'gh', 'rho0', 'alpha', 'upsilon', 'rho0_upsilon', 'boundary', 'last_improvement')
    with open(os.path.join(state_dir, 'rh_profile.tsv'), 'w') as stream:
        stream.write('\t'.join(columns) + '\n')
        for row in summary['rows']:
            flat = dict(row, **(row['params'] or {}))
            flat['boundary'] = ','.join(row['boundary'])
            stream.write('\t'.join('' if flat.get(k) is None else str(flat[k]) for k in columns) + '\n')
    rqp.atomic_json(os.path.join(state_dir, 'report_rhprofile.json'), summary)
    lines = ['rh profiles: ' + summary['status'], summary['interpretation']]
    for beta, best in summary['minima'].items():
        lines.append(f'beta={beta}: ' + ('no valid model' if best is None else
                     f'best rh={best["rh"]:g}, penalty={best["penalty"]:.8g}, grid-edge={best["at_grid_edge"]}'))
    lines.append('Full nuisance parameters, counts and boundaries: rh_profile.tsv')
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(state_dir, 'report_rhprofile.txt'), 'w') as stream:
        stream.write(text)
    return text


def format_plan(config):
    total = len(cells(config)) * config['cell_budget']
    return (f"rh profiles: beta={config['beta_grid']}, rh={config['rh_grid']} kpc\n"
            f"Box: {config['box']}; fixed rh/beta per cell, full 4D PCA (no dropped direction)\n"
            f"Initial coordinates: {len(config['initial_points'])} + at most one continuation, "
            f"Sobol: {config['cell_sobol']}, then TuRBO up to {config['cell_budget']} evaluations/cell\n"
            f"Budget: <= {total} new evaluations; seed 42; L24/N184; no imported penalties\n"
            "All beta branches use the same nuisance Sobol sequence. Libraries and history carry beta.\n"
            "Per-cell checkpoints; larger cell budget extends all cells; no data-supported J bound implied.\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rh-grid', default=','.join(f'{x:g}' for x in RH_GRID))
    parser.add_argument('--beta-grid', default=','.join(f'{x:g}' for x in BETAS))
    parser.add_argument('--profile-box', help='optional five-parameter box inside existing bounds; all spans free')
    parser.add_argument('--profile-starts', help='initial coordinates file: Q gh rh rho0 alpha; rh projected to grid')
    parser.add_argument('--cell-budget', type=int)
    parser.add_argument('--cell-sobol', type=int, default=DEFAULT_SOBOL)
    parser.add_argument('--sobol-seed', type=int, default=20261108)
    parser.add_argument('--subsample-seed', type=int, default=lp.DEFAULTS['subsample_seed'])
    parser.add_argument('--incl', type=float, default=90.0)
    parser.add_argument('--state-dir', default='rhprofile')
    parser.add_argument('--suffix', default='rhprofile')
    parser.add_argument('--orblib-dir', default='orblib_rhprofile')
    parser.add_argument('--shared-orblib', default='orblib')
    parser.add_argument('--save-below', type=float, default=3.5)
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    parser.add_argument('--n_threads', type=int)
    parser.add_argument('--max-failures', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--summarize', metavar='DIR')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.summarize:
        print(write_summary(args.summarize), end='')
        return 0
    try:
        stored = None
        if args.resume:
            with open(os.path.join(args.state_dir, STATE_NAME)) as stream:
                stored = json.load(stream)
            if args.cell_budget is None:
                args.cell_budget = stored['config']['cell_budget']
        config = build_config(args, rsm.script_bounds())
        if stored is not None:
            check_config(config, stored['config'])
        elif os.path.exists(os.path.join(args.state_dir, STATE_NAME)) and not (args.dry_run or args.self_test):
            raise ConfigError(f'{args.state_dir} already contains a profile; use --resume')
        if args.dry_run:
            print(format_plan(config), end='')
            return 0
        if args.self_test:
            cfg = cell_config(config, config['beta_grid'][0], config['rh_grid'][0])
            ok, point = lp.self_test(cfg['box'], options=dict(fit_alpha=True, pca_components=4, **lp.ALPHA_DEFAULTS))
            print(f'[rhprofile] constrained full-4D self-test: {"ok" if ok else "FAILED"} {point}')
            return 0 if ok else 1
    except (ConfigError, OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    context = {}
    for name, path in (('harness', rsm.EXP_SCRIPT), ('driver', __file__), ('search', lp.__file__),
                       ('evaluator', rqp.__file__), ('protocol', rsm.__file__),
                       ('storage', os.path.join(HERE, 'orblib_storage.py'))):
        context[name + '_sha256'] = rqp.sha256_file(path)
    if stored is not None and any(stored['context'].get(k) != v for k, v in context.items()):
        print('ERROR: profile code differs; use a separate run', file=sys.stderr)
        return 2
    os.makedirs(args.state_dir, exist_ok=True)
    # This is an isolated diagnostic store; allow a stopped run to continue.
    stop = os.path.join(args.orblib_dir, '.storage', 'STOP')
    if os.path.exists(stop):
        os.remove(stop)
    evaluator, mod, _ = rqp.build_evaluator(args, args.state_dir, prefix='rhprofile')
    context.update(exp_id=mod.EXP_ID, geom_hash=mod.GEOM_HASH, evaluation_context=mod.EVALUATION_CONTEXT)
    profile = Profile(config, args.state_dir, evaluator, max_failures=args.max_failures)
    try:
        if tuple(mod.ALPHA_FIT_BOUNDS) != lp.ALPHA_BOUNDS or tuple(mod.BETA_PROFILE_BOUNDS) != (3., 10.):
            raise ConfigError('harness diagnostic alpha/beta domains differ')
        profile.open(args.resume, context)
    except ConfigError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    code = 0
    try:
        profile.run()
    except StopRun as error:
        profile.state['status'], code = f'stopped: {error}', 75
    except Exception as error:
        traceback.print_exc()
        profile.state['status'], code = f'failed: {error!r}', 1
    profile.save()
    print(write_summary(args.state_dir), end='', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
