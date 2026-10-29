"""Q profile driver py/run_q_profile.py (no AGAMA, torch or cloud).

The schedule, trust regions, merging, global branches, resume and the
library-saving policy run against synthetic objectives and fake back-ends.
"""
import ast
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
DRIVER = ROOT / 'py/run_q_profile.py'
TREE = ast.parse(SCRIPT.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rqp = load('run_q_profile', DRIVER)
storage = load('orblib_storage', ROOT / 'py/orblib_storage.py')
BOUNDS = {k: list(v) for k, v in rqp.rsm.script_bounds().items()}
GRID = [0.27, 0.30, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]
COMPACT = dict(gh=0.0, rh=2.4, rho0=90.0)
EXTENDED = dict(gh=0.0, rh=28.0, rho0=43.0)


def make_config(**override):
    config = dict(rqp.DEFAULTS, incl=90.0, q_grid=list(GRID), pot='l24', bounds=BOUNDS,
                  starts=dict(compact=dict(params=dict(COMPACT), Q=0.27, penalty=1.44, source='m0'),
                              extended=dict(params=dict(EXTENDED), Q=1.0, penalty=1.53, source='m1')),
                  priors=[], prior_skipped=[])
    config.update(override)
    return config


def two_basins(Q, params):
    u = rqp.to_unit(params, BOUNDS)
    c, e = rqp.to_unit(COMPACT, BOUNDS), rqp.to_unit(EXTENDED, BOUNDS)
    compact = 1.4 + 3.0 * (Q - GRID[0]) + 5.0 * sum((a - b) ** 2 for a, b in zip(u, c))
    extended = 1.5 + 2.0 * (1.0 - Q) + 5.0 * sum((a - b) ** 2 for a, b in zip(u, e))
    return min(compact, extended)


class Recorder:
    def __init__(self, objective=two_basins, stop_at=None, fail=False):
        self.objective, self.stop_at, self.fail = objective, stop_at, fail
        self.calls = []

    def __call__(self, Q, params, seed, meta):
        if self.stop_at is not None and len(self.calls) == self.stop_at:
            raise rqp.StopRun('test stop')
        self.calls.append(dict(meta, params=dict(params), seed=seed))
        if self.fail:
            return dict(status='failed', error='boom')
        return dict(status='ok', penalty=self.objective(Q, params), upsilon=0.6, wall_s=1.0, orbit_s=1.0)


def local_proposer(points, values, baseline, center, length, seed):
    rng = random.Random(seed)
    return [min(1.0, max(0.0, c + rng.uniform(-length / 2, length / 2))) for c in center]


def projection_proposer(target):
    def propose(points, values, baseline, center, length, seed):
        return [min(c + length / 2, max(c - length / 2, t)) for c, t in zip(center, target)]
    return propose


def fake_sobol(n, seed):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(3)] for _ in range(n)]


def profile(tmp_path, config=None, evaluator=None, proposer=local_proposer, sobol=fake_sobol, **kw):
    return rqp.Profile(config or make_config(), str(tmp_path / 'state'), evaluator or Recorder(),
                       proposer=proposer, sobol=sobol, log=lambda *a: None, **kw)


# --------------------------------------------------------------------------
# Coordinates and inputs
# --------------------------------------------------------------------------
def test_unit_coordinates_round_trip_log_axes_and_clipping():
    rng = random.Random(1)
    for _ in range(50):
        params = dict(gh=rng.uniform(*BOUNDS['gh']), rh=math.exp(rng.uniform(*map(math.log, BOUNDS['rh']))),
                      rho0=math.exp(rng.uniform(*map(math.log, BOUNDS['rho0']))))
        u = rqp.to_unit(params, BOUNDS)
        assert all(0 <= x <= 1 for x in u)
        back = rqp.from_unit(u, BOUNDS)
        assert all(math.isclose(back[k], params[k], rel_tol=1e-12) for k in rqp.PARAMS)
    mid = rqp.from_unit([0.5, 0.5, 0.5], BOUNDS)
    assert math.isclose(mid['rh'], math.sqrt(BOUNDS['rh'][0] * BOUNDS['rh'][1]))
    assert math.isclose(mid['gh'], sum(BOUNDS['gh']) / 2)
    edge = rqp.from_unit([-0.3, 1.7, 1.0], BOUNDS)
    assert edge['gh'] == BOUNDS['gh'][0] and edge['rh'] == BOUNDS['rh'][1] and edge['rho0'] == BOUNDS['rho0'][1]


def test_q_grid_defaults_and_validation():
    assert rqp.parse_q_grid(None, BOUNDS, 0.27) == GRID
    assert rqp.parse_q_grid('low,0.6,1', BOUNDS, 0.2) == [0.2, 0.6, 1.0]
    assert rqp.parse_q_grid('0.3,0.6,1', BOUNDS, 0.27) == [0.3, 0.6, 1.0]
    for bad in ('0.3,0.6', '0.6,0.3,1', '1', '0.01,1', '0.3,0.3,1', 'x,1'):
        with pytest.raises(rqp.ConfigError):
            rqp.parse_q_grid(bad, BOUNDS, 0.27)


def test_endpoints_pick_compact_and_extended_rows(tmp_path):
    path = tmp_path / 'finalists.txt'
    path.write_text(
        '# header\n'
        '90.000 0.27 0.0 2.4 90.0 0.66 1.40 2026-10-22 22:04:15  # m0\n'
        '90.000 0.300000000000000 0.0 2.0 80.0 0.66 1.9 2026-10-22 22:04:15  # worse\n'
        '90.000 1.0 0.0 28.0 43.0 0.47 1.50 2026-10-16 04:19:35  # m1\n')
    starts = rqp.read_endpoints(str(path), BOUNDS, 90.0)
    assert starts['compact']['params'] == COMPACT and starts['compact']['source'] == 'm0'
    assert starts['extended']['params'] == EXTENDED and starts['extended']['Q'] == 1.0
    path.write_text('90.000 1.0 0.0 28.0 43.0 0.47 1.53 2026-10-16 04:19:35\n')
    with pytest.raises(rqp.ConfigError):
        rqp.read_endpoints(str(path), BOUNDS, 90.0)


def test_priors_keep_grid_rows_and_prefer_l24_duplicates(tmp_path):
    path = tmp_path / 'priors.txt'
    path.write_text(
        '90.000 1.0 0.0 25.0 35.0 0.6 1.80 2026-10-14 00:00:00  # pot=l24 protocol=strict src=q29\n'
        '90.000 1.000000000000000 0.0 25.000000000000000 35.000000000000000 0.6 1.81 2026-10-06 14:42:52  # pot=l4 protocol=exp src=r2\n'
        '90.000 1.0 0.0 7.0 40.0 0.5 2.20 2026-10-06 14:37:47  # pot=l4 protocol=exp src=r2b\n'
        '90.000 0.27 0.0 2.4 90.0 0.66 1.60 2026-10-14 00:00:00  # pot=l24 src=fq\n'
        '90.000 0.55 0.0 7.0 40.0 0.5 2.0 2026-10-06 14:37:47  # pot=l4 off-grid\n'
        '85.000 1.0 0.0 7.0 40.0 0.5 2.0 2026-10-06 14:37:47  # pot=l4 other incl\n')
    rows, skipped = rqp.read_prior_rows(str(path), BOUNDS, 90.0)
    assert len(rows) == 5 and len(skipped) == 1
    priors, dropped = rqp.assign_priors(rows, GRID)
    assert [(p['qi'], p['source'], p['pot']) for p in priors] == [(0, 'fq', 'l24'), (7, 'q29', 'l24'), (7, 'r2b', 'l4')]
    assert sorted(dropped) == ['priors.txt:5 (Q off the grid)', 'r2 (duplicate)']


def test_default_low_q_comes_from_l24_priors_else_the_finalist(tmp_path):
    finalists = tmp_path / 'finalists.txt'
    finalists.write_text('90.000 0.27 0.0 2.4 90.0 0.66 1.40 2026-10-22 22:04:15  # m0\n'
                         '90.000 1.0 0.0 28.0 43.0 0.47 1.50 2026-10-16 04:19:35  # m1\n')
    priors = tmp_path / 'priors.txt'
    priors.write_text('90.000 0.26 0.0 2.4 90.0 0.66 1.60 2026-10-14 00:00:00  # pot=l24 src=fq\n'
                      '90.000 0.20 0.0 2.4 90.0 0.66 1.90 2026-10-14 00:00:00  # pot=l4 src=old\n')
    args = rqp.parse_args(['--dry-run', '--endpoints', str(finalists)])
    assert rqp.build_config(args, BOUNDS)['q_grid'][0] == 0.27
    args = rqp.parse_args(['--dry-run', '--endpoints', str(finalists), '--prior-points', str(priors)])
    config = rqp.build_config(args, BOUNDS)
    assert config['q_grid'][0] == 0.26 and [p['source'] for p in config['priors']] == ['fq']


def test_budget_of_variant_a():
    plan = rqp.budget(make_config())
    assert plan['per_q'] == 26 and plan['total'] == 208 and plan['total_max'] == 288


def test_config_hash_changes_with_scientific_settings():
    assert rqp.config_hash(make_config()) == rqp.config_hash(make_config())
    assert rqp.config_hash(make_config()) != rqp.config_hash(make_config(local_budget=8))
    assert rqp.config_hash(make_config()) != rqp.config_hash(make_config(opt_seed=7))


# --------------------------------------------------------------------------
# Schedule
# --------------------------------------------------------------------------
def test_schedule_continues_both_branches_over_the_whole_grid_with_seed_42(tmp_path):
    recorder = Recorder()
    run = profile(tmp_path, evaluator=recorder)
    run.open(resume=False)
    run.run()
    state = json.loads((tmp_path / 'state/profile_state.json').read_text())
    assert state['status'] == 'done'
    assert {c['seed'] for c in recorder.calls} == {42}
    last = len(GRID) - 1
    blocks = []
    for call in recorder.calls:
        key = (call['qi'], call['branch'] if call['kind'] != 'sobol' else 'sobol')
        if not blocks or blocks[-1] != key:
            blocks.append(key)
    expected = [(0, 'compact'), (0, 'sobol'), (last, 'extended'), (last, 'sobol')]
    for qi in range(1, last):
        expected += [(qi, 'sobol'), (qi, 'compact')]
    expected += [(last, 'compact')] + [(qi, 'extended') for qi in range(last - 1, -1, -1)]
    assert [b for b in blocks if b[1] != 'global'][:len(expected)] == expected
    assert recorder.calls[0]['kind'] == 'start' and [c['kind'] for c in recorder.calls[1:5]] == ['diag'] * 4
    assert [round(c['params']['rho0'] / COMPACT['rho0'], 3) for c in recorder.calls[1:5]] == [0.97, 0.985, 1.015, 1.03]
    for key, branch in state['branches'].items():
        assert branch['n_evals'] <= branch['budget'], key
    for qi in range(1, last + 1):
        previous = run.branch_best('compact', qi - 1)
        if previous and state['branches'][f'compact@{qi}']['status'] != 'skipped':
            assert state['branches'][f'compact@{qi}']['start'] == previous['params']
    for qi in range(last - 1, -1, -1):
        previous = run.branch_best('extended', qi + 1)
        if previous and state['branches'][f'extended@{qi}']['status'] != 'skipped':
            assert state['branches'][f'extended@{qi}']['start'] == previous['params']
    assert len(recorder.calls) <= rqp.budget(make_config())['total_max']
    assert sum(1 for c in recorder.calls if c['kind'] == 'sobol') == 6 * len(GRID)
    assert state['smooth_check']['verdict'] == 'logged' and state['smooth_check']['n'] == 5
    table = (tmp_path / 'state/profile.tsv').read_text().splitlines()
    assert table[0].split('\t') == list(rqp.TABLE_COLUMNS) and len(table) == len(recorder.calls) + 1


def test_trust_region_update_rules(tmp_path):
    run = profile(tmp_path)
    branch = dict(length=0.4, succ=0, fail=0)
    run.update_tr(branch, True)
    run.update_tr(branch, True)
    assert branch['length'] == 0.8 and branch['succ'] == 0
    run.update_tr(branch, True)
    run.update_tr(branch, True)
    assert branch['length'] == 0.8
    for _ in range(3):
        run.update_tr(branch, False)
    assert branch['length'] == 0.4 and branch['fail'] == 0


def test_branches_merge_and_the_merged_branch_stops(tmp_path):
    target = rqp.to_unit(dict(gh=0.0, rh=10.0, rho0=60.0), BOUNDS)

    def one_basin(Q, params):
        return 1.5 + sum((a - b) ** 2 for a, b in zip(rqp.to_unit(params, BOUNDS), target))

    config = make_config(local_budget=12, n_global=0)
    run = profile(tmp_path, config=config, evaluator=Recorder(one_basin), proposer=projection_proposer(target))
    run.open(resume=False)
    run.run()
    state = run.state
    last = len(GRID) - 1
    assert state['merged'].get('extended') == last - 1
    assert state['branches'][f'extended@{last - 1}']['merged_into'] == 'compact'
    for qi in range(last - 1):
        branch = state['branches'][f'extended@{qi}']
        assert branch['status'] == 'skipped' and 'merged' in branch['reason'] and branch['n_evals'] == 0


def test_far_good_sobol_point_spawns_a_global_branch(tmp_path):
    third = [0.9, 0.1, 0.9]

    def three_basins(Q, params):
        u = rqp.to_unit(params, BOUNDS)
        return min(two_basins(Q, params), 1.3 + 5 * sum((a - b) ** 2 for a, b in zip(u, third)))

    sobol = lambda n, seed: [list(third)] + fake_sobol(n - 1, seed)  # noqa: E731
    run = profile(tmp_path, evaluator=Recorder(three_basins), sobol=sobol)
    run.open(resume=False)
    run.run()
    branch = run.state['branches']['global@0']
    first = run.state['evaluations'][branch['members'][0]]
    assert first['kind'] == 'sobol' and first['u'] == pytest.approx(third)
    assert branch['n_evals'] <= run.cfg['global_budget']
    assert all(qi in run.state['globals_checked'] for qi in range(len(GRID)))


def test_resume_after_stop_reproduces_the_uninterrupted_run(tmp_path):
    target = rqp.to_unit(dict(gh=0.1, rh=5.0, rho0=50.0), BOUNDS)
    proposer = projection_proposer(target)
    full = profile(tmp_path / 'a', evaluator=Recorder(), proposer=proposer)
    full.open(resume=False)
    full.run()
    reference = [(r['qi'], r['kind'], r['branch'], r['params']) for r in full.state['evaluations']]

    first = profile(tmp_path / 'b', evaluator=Recorder(stop_at=23), proposer=proposer)
    first.open(resume=False)
    with pytest.raises(rqp.StopRun):
        first.run()
    assert len(first.state['evaluations']) == 23
    second = profile(tmp_path / 'b', evaluator=Recorder(), proposer=proposer)
    second.open(resume=True, run_info=dict(suffix='r2'))
    second.run()
    assert [(r['qi'], r['kind'], r['branch'], r['params']) for r in second.state['evaluations']] == reference
    assert second.state['runs'][-1]['suffix'] == 'r2'


def test_state_guards_on_open(tmp_path):
    run = profile(tmp_path)
    with pytest.raises(rqp.ConfigError):
        run.open(resume=True)
    run.open(resume=False)
    with pytest.raises(rqp.ConfigError):
        profile(tmp_path).open(resume=False)
    with pytest.raises(rqp.ConfigError):
        profile(tmp_path, config=make_config(local_budget=8)).open(resume=True)


def test_priors_enter_the_gp_without_budget_and_replace_an_identical_start(tmp_path):
    priors = [dict(qi=7, Q=1.0, params=dict(EXTENDED), penalty=1.58, upsilon=0.47, pot='l4', protocol='exp', source='r4'),
              dict(qi=7, Q=1.0, params=dict(gh=0.0, rh=7.0, rho0=40.0), penalty=2.23, upsilon=0.52, pot='l4',
                   protocol='exp', source='r2')]
    recorder = Recorder()
    run = profile(tmp_path, config=make_config(priors=priors), evaluator=recorder)
    run.open(resume=False)
    assert [r['kind'] for r in run.state['evaluations']] == ['prior', 'prior']
    run.run()
    branch = run.state['branches']['extended@7']
    assert branch['members'][0] == 0
    assert not any(c['qi'] == 7 and c['kind'] == 'start' and c['branch'] == 'extended' for c in recorder.calls)
    assert branch['n_evals'] == sum(1 for c in recorder.calls if c['qi'] == 7 and c['branch'] == 'extended')


def test_consecutive_failures_abort(tmp_path):
    run = profile(tmp_path, evaluator=Recorder(fail=True), max_failures=3)
    run.open(resume=False)
    with pytest.raises(rqp.RunFailed):
        run.run()
    assert len(run.state['evaluations']) == 3


def test_smooth_check_measures_the_quadratic_residual():
    records = [dict(params=dict(rho0=r), penalty=1.5 + 0.3 * math.log(r / 90) ** 2) for r in (87, 88.6, 90, 91.4, 93)]
    check = rqp.smooth_check(records)
    assert check['rms'] < 1e-10 and check['verdict'] == 'logged'
    noisy = [dict(r, penalty=r['penalty'] + d) for r, d in zip(records, (0.02, -0.03, 0.04, -0.02, 0.01))]
    assert rqp.smooth_check(noisy)['rms'] > 0.005


# --------------------------------------------------------------------------
# Summary and CLI
# --------------------------------------------------------------------------
def _record(i, qi, kind, branch, params, penalty):
    return dict(id=i, qi=qi, Q=GRID[qi], kind=kind, branch=branch, params=params, u=rqp.to_unit(params, BOUNDS),
                status='ok', penalty=penalty, upsilon=0.5, source='qprofile')


def test_summary_reports_barrier_switch_and_hysteresis():
    config = make_config(q_grid=[GRID[0], 0.5, 1.0])
    evaluations = [_record(0, 0, 'start', 'compact', COMPACT, 1.45), _record(1, 0, 'start', 'extended', EXTENDED, 1.80),
                   _record(2, 1, 'start', 'compact', COMPACT, 1.90), _record(3, 1, 'start', 'extended', EXTENDED, 1.95),
                   _record(4, 2, 'start', 'compact', dict(COMPACT, gh=1.6), 2.40),
                   _record(5, 2, 'start', 'extended', EXTENDED, 1.55)]
    branches = {f'{b}@{qi}': dict(name=b, qi=qi, status='converged', n_evals=1, length=0.04, members=[i])
                for i, (qi, b) in enumerate([(0, 'compact'), (0, 'extended'), (1, 'compact'), (1, 'extended'),
                                             (2, 'compact'), (2, 'extended')])}
    summary = rqp.summarize(dict(config=config, evaluations=evaluations, branches=branches, status='done',
                                 merged={}, smooth_check=None))
    assert summary['barrier']['value'] == pytest.approx(1.90 - 1.55) and summary['barrier']['Q'] == 0.5
    assert summary['branch_switch_Q'] == [1.0]
    assert [r['hysteresis'] for r in summary['rows']] == [True, True, True]
    assert summary['rows'][2]['profile']['boundary'] == ['gh=lo']
    assert summary['rows'][2]['profile']['rho0_upsilon'] == pytest.approx(EXTENDED['rho0'] * 0.5)
    text = rqp.format_summary(summary)
    assert 'barrier B = +0.3500' in text and 'hysteresis' in text


def test_dry_run_and_summarize_need_only_the_standard_library(tmp_path):
    finalists = tmp_path / 'finalists.txt'
    finalists.write_text(
        '90.000 0.27 0.0 2.4 90.0 0.66 1.40 2026-10-22 22:04:15  # m0\n'
        '90.000 1.0 0.0 28.0 43.0 0.47 1.50 2026-10-16 04:19:35  # m1\n')
    env = dict(os.environ, PYTHONPATH='')
    python = '/usr/bin/python3' if Path('/usr/bin/python3').exists() else sys.executable
    result = subprocess.run([python, '-S', str(DRIVER), '--dry-run', '--endpoints', str(finalists)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'evaluations: 208 (max 288)' in result.stdout and 'start extended: gh=0 rh=28 rho0=43' in result.stdout
    bad = subprocess.run([python, '-S', str(DRIVER), '--dry-run', '--endpoints', str(finalists), '--q-grid', '0.3,0.5'],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and 'end at Q=1' in bad.stderr

    run = profile(tmp_path, config=make_config(q_grid=[GRID[0], 0.5, 1.0], n_global=1, local_budget=2))
    run.open(resume=False)
    run.run()
    result = subprocess.run([python, '-S', str(DRIVER), '--summarize', str(tmp_path / 'state')],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'Q profile: status=done' in result.stdout
    report = json.loads((tmp_path / 'state/report_qprofile.json').read_text())
    assert len(report['rows']) == 3 and report['opt_seed'] == 42


def test_history_names_are_not_read_by_any_pool_glob():
    names = ['qprofile_tycho_d1_nb250_gh0_l24n184_ser0_qprofile20261028_120000.txt',
             'profile_state.json', 'profile.tsv', 'report_qprofile.json']
    tags = ['d1_nb250_gh0_l24n184_ser0', 'Q1d1_nb250_gh0_ser0', 'd1_nb250_gh0_ser0', 'Q1d1_nb250_gh0_l24n184_ser0']
    patterns = ([f'out_*_{t}{s}.txt' for t in tags for s in ('_*', '')]
                + [f'out_tycho_{t}{s}.txt' for t in tags for s in ('', '_p*')]
                + ['4UpsBoTorch_PCA_Sersic*.txt', '4UpsBoTorch_PCA_PA46.8_Sersic_*.txt'])
    assert not [(n, p) for n in names for p in patterns if fnmatch.fnmatch(n, p)]


# --------------------------------------------------------------------------
# Harness contract and library saving
# --------------------------------------------------------------------------
def _module_level_names(tree):
    names = set()
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            stack.extend(node.body + getattr(node, 'orelse', []) + getattr(node, 'finalbody', []))
    return names


def test_every_harness_name_the_driver_relies_on_exists():
    needed = {'write_orblib_npz', 'TRAJSIZE_STORED', 'gridv', 'degree', 'ghorder', 'incl', 'N_BIN', 'DOUBLE',
              'SER_ID', 'orblib_key', 'halo_IC_lib_weights_pca_fixed', 'hostname_proc', 'EXP_ID', 'GEOM_HASH',
              'EVALUATION_CONTEXT', 'AGAMA_ORBIT_THREADS', 'ORBLIB_DIR'}
    assert needed <= _module_level_names(TREE), sorted(needed - _module_level_names(TREE))
    halo = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'halo_IC_lib_weights_pca_fixed')
    assert 'intTime' in [a.arg for a in halo.args.args]
    orbit = [n for n in ast.walk(halo) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'agama.orbit']
    assert orbit and {'ic', 'time', 'potential', 'targets'} <= {k.arg for k in orbit[0].keywords}
    flags = {n.args[0].value for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'parser.add_argument'}
    assert {'--l24', '--orblib-dir', '--suffix', '--no-resume', '--n_threads'} <= flags


def test_orbit_capture_forwards_attributes_and_keeps_the_last_call():
    real = SimpleNamespace(orbit=lambda **kw: ['dens', 'kinem'], setRandomSeed=lambda k: ('seeded', k))
    capture = rqp.OrbitCapture(real)
    assert capture.setRandomSeed(42) == ('seeded', 42)
    assert capture.orbit(potential='p', ic='IC', time='T', targets=[]) == ['dens', 'kinem']
    assert capture.last == dict(ic='IC', time='T', matrices=['dens', 'kinem'])


def _writer_namespace():
    namespace = dict(numpy=numpy, zipfile=zipfile)
    for node in TREE.body:
        if (isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == '_NPZ_BLOCK_BYTES' for t in node.targets)) \
                or (isinstance(node, ast.FunctionDef) and node.name == 'write_orblib_npz'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    return namespace


def fake_module(tmp_path):
    def halo_IC_lib_weights_pca_fixed(pc, model_data, bounds, densityStars, datasets, alphah, betah,
                                      Upsilon_lower=0.1, Upsilon_upper=1.6, numOrbits=100000, intTime=100.):
        pass

    def orblib_key(Q, gh, rh, rho0):
        digest = hashlib.md5(f'{Q:.6g}_{gh:.6g}_{rh:.6g}_{rho0:.6g}'.encode()).hexdigest()[:10]
        name = f'orblib_i90.0_d1_nb250_ser0_l24n184_geomabcdef12_{digest}.npz'
        return name, str(tmp_path / 'isolated' / name)

    return SimpleNamespace(orblib_key=orblib_key, write_orblib_npz=_writer_namespace()['write_orblib_npz'],
                           halo_IC_lib_weights_pca_fixed=halo_IC_lib_weights_pca_fixed, incl=90.0,
                           TRAJSIZE_STORED=0, gridv=numpy.linspace(-25, 25, 51), degree=2, ghorder=6,
                           N_BIN=250, DOUBLE=True, SER_ID=0)


def _capture(n=5):
    capture = rqp.OrbitCapture(None)
    rng = numpy.random.default_rng(0)
    capture.last = dict(ic=rng.normal(size=(n, 6)), time=rng.uniform(1, 2, n),
                        matrices=[rng.normal(size=(n, 4)).astype(numpy.float32),
                                  rng.normal(size=(n, 7)).astype(numpy.float32)])
    return capture


def test_save_library_writes_a_valid_standard_library_once(tmp_path):
    mod, shared = fake_module(tmp_path), tmp_path / 'orblib'
    params = dict(gh=0.0, rh=5.0, rho0=50.0)
    info = rqp.save_library(mod, storage, _capture(), 0.5, params, 1.9, str(shared), 3.5, 0)
    name = mod.orblib_key(0.5, **params)[0]
    path = shared / name
    assert info['name'] == name and path.exists()
    assert info['md5'] == storage.file_hash(path) and info['size'] == path.stat().st_size
    meta = storage.metadata(path)
    assert meta['Q'] == 0.5 and meta['rho0'] == 50.0 and meta['numOrbits'] == 5 and meta['intTime'] == 100.0
    assert not list(shared.glob('*.tmp'))
    before = path.read_bytes()
    again = rqp.save_library(mod, storage, _capture(), 0.5, params, 1.9, str(shared), 3.5, 0)
    assert again['skipped'] == 'exists' and path.read_bytes() == before


@pytest.mark.parametrize('case, expected', [
    ('penalty', 'penalty>3.5'), ('q1', 'Q=1 (untagged library name)'), ('building', 'exists'),
    ('disk', 'free disk < 1000000000000.0 GB'), ('nocapture', 'no captured orbit library')])
def test_save_library_skips_without_writing(tmp_path, case, expected):
    mod, shared = fake_module(tmp_path), tmp_path / 'orblib'
    shared.mkdir()
    params, Q, penalty, min_free, capture = dict(gh=0.0, rh=5.0, rho0=50.0), 0.5, 2.0, 0, _capture()
    if case == 'penalty':
        penalty = 3.6
    elif case == 'q1':
        Q = 1.0
    elif case == 'building':
        (shared / (mod.orblib_key(Q, **params)[0] + '.building')).write_text('busy')
    elif case == 'disk':
        min_free = 1e12
    elif case == 'nocapture':
        capture.last = None
    info = rqp.save_library(mod, storage, capture, Q, params, penalty, str(shared), 3.5, min_free)
    assert info['skipped'] == expected
    assert not list(shared.glob('*.npz')) and not list(shared.glob('*.tmp'))
