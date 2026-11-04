"""Mocked tests for py/run_local_pca.py (no AGAMA, torch or BoTorch)."""
import ast
import fnmatch
import importlib.util
import inspect
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
DRIVER = ROOT / 'py/run_local_pca.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lp = load('run_local_pca', DRIVER)
BOUNDS = {k: list(v) for k, v in lp.rsm.script_bounds().items()}
BOX = dict(Q=[0.90, 1.05], gh=[0.0, 0.20], rh=[10.0, 30.0], rho0=[15.0, 60.0])
BOX_SPEC = 'Q=0.90:1.05,gh=0:0.20,rh=10:30,rho0=15:60'
CONTEXT = dict(exp_id='d1_nb250_gh0_l24n184_ser0', geom_hash='b99b4c93', evaluation_context='ctx0',
               harness_sha256='sha0')
TARGET = dict(Q=0.96, gh=0.0, rh=27.0, rho0=26.0)


def objective(Q, params):
    lo, hi = lp.box_x(BOX)
    return 1.45 + 4.0 * sum(((a - b) / (h - l)) ** 2 for a, b, l, h in zip(lp.to_x(params), lp.to_x(TARGET), lo, hi))


class Recorder:
    def __init__(self, stop_at=None, fail=False, reject=False):
        self.stop_at, self.fail, self.reject = stop_at, fail, reject
        self.calls = []

    def __call__(self, Q, params, seed, meta):
        if self.stop_at is not None and len(self.calls) == self.stop_at:
            raise lp.StopRun('test stop')
        self.calls.append(dict(meta, Q=Q, params=dict(params), seed=seed))
        if self.fail:
            return dict(status='failed', error='boom')
        if self.reject:
            return dict(status='rejected', penalty=3e7, error='cusp')
        return dict(status='ok', penalty=objective(Q, params), upsilon=0.7, wall_s=1.0, orbit_s=1.0)


def random_proposer(points, values, center, lo, hi, A, b, seed):
    rng = random.Random(seed)
    return [rng.uniform(a, c) for a, c in zip(lo, hi)]


def outside_proposer(points, values, center, lo, hi, A, b, seed):
    return [2.0, -1.0, 2.0, -1.0]


def center_proposer(points, values, center, lo, hi, A, b, seed):
    return list(center)


def fake_sobol(n, seed):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(4)] for _ in range(n)]


def prior(params, penalty, status='ok', source='p'):
    return dict(params=dict(params), penalty=penalty, upsilon=0.7, status=status, source=source)


PRIORS = [prior(dict(Q=0.95, gh=0.0, rh=29.26, rho0=26.84), 1.4793, source='a:24'),
          prior(dict(Q=1.0, gh=0.0, rh=30.0, rho0=41.76), 1.5634, source='b:133'),
          prior(dict(Q=0.9, gh=0.0, rh=30.0, rho0=35.43), 1.8481, source='b:145'),
          prior(dict(Q=1.0, gh=0.16, rh=30.0, rho0=55.7), 190.18, source='b:20')]


def make_config(**override):
    config = dict(lp.DEFAULTS, incl=90.0, pot='l24', box={k: list(v) for k, v in BOX.items()}, bounds=BOUNDS,
                  budget=20, n_sobol=4, priors=[dict(p) for p in PRIORS], prior_skipped=[], prior_outside=0,
                  prior_context=dict(CONTEXT), prior_sources=[])
    config.update(override)
    return config


def search(tmp_path, config=None, evaluator=None, proposer=random_proposer, sobol=fake_sobol, **kw):
    return lp.Search(config or make_config(), str(tmp_path / 'state'), evaluator or Recorder(),
                     proposer=proposer, sobol=sobol, log=lambda *a: None, **kw)


# --------------------------------------------------------------------------
# Box, coordinates, inputs
# --------------------------------------------------------------------------
def test_box_parsing_and_validation():
    assert lp.parse_box(BOX_SPEC, BOUNDS) == BOX
    assert lp.parse_box(' rho0=15:60, rh=10:30,gh=0:0.2,Q=0.9:1.05 ', BOUNDS) == BOX
    for bad in ('', 'Q=0.9:1.05,gh=0:0.2,rh=10:30', BOX_SPEC + ',Q=0.9:1', 'Q=1.05:0.9,gh=0:0.2,rh=10:30,rho0=15:60',
                'Q=0.9:1.05,gh=0:0.2,rh=10:31,rho0=15:60', 'Q=0.9:1.05,gh=-0.1:0.2,rh=10:30,rho0=15:60',
                'Q=0.9-1.05,gh=0:0.2,rh=10:30,rho0=15:60', 'X=0:1,' + BOX_SPEC, 'Q=a:1,gh=0:0.2,rh=10:30,rho0=15:60'):
        with pytest.raises(lp.ConfigError):
            lp.parse_box(bad, BOUNDS)


def test_coordinates_round_trip_and_clip():
    params = dict(Q=0.97, gh=0.05, rh=21.0, rho0=33.0)
    back = lp.from_x(lp.to_x(params))
    assert all(math.isclose(back[k], params[k], rel_tol=1e-12) for k in lp.NAMES)
    assert lp.in_box(params, BOX) and not lp.in_box(dict(params, rh=30.5), BOX)
    assert lp.clip_to_box(dict(params, rh=31.0, gh=-0.1), BOX) == dict(params, rh=30.0, gh=0.0)


def test_adaptive_cutoff_matches_the_harness():
    tree = ast.parse(SCRIPT.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'adaptive_penalty_cutoff')
    namespace = dict(numpy=numpy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    rng = random.Random(3)
    for n in (3, 10, 11, 30, 57):
        penalties = [rng.uniform(0.3, 4.0) for _ in range(n)]
        assert lp.adaptive_cutoff(penalties, 0.3, 10, 0.6) == namespace['adaptive_penalty_cutoff'](
            numpy.array(penalties), 0.3, 10, 0.6)


def _records(n=30, seed=1):
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        params = dict(Q=rng.uniform(0.9, 1.05), gh=rng.choice([0.0, rng.uniform(0, 0.2)]),
                      rh=10 ** rng.uniform(1, math.log10(30)), rho0=10 ** rng.uniform(math.log10(15), math.log10(60)))
        records.append(dict(params=params, status='ok', penalty=objective(params['Q'], params)))
    return records


def test_basis_is_full_rank_and_maps_the_box_to_the_unit_cube():
    basis = lp.fit_basis(_records(), make_config(), 0)
    C = numpy.array(basis['components'])
    assert C.shape == (4, 4) and numpy.allclose(C @ C.T, numpy.eye(4))
    assert math.isclose(sum(basis['explained']), 1.0) and basis['n_good'] == 10
    for record in _records(10, seed=7):
        u = lp.to_u(record['params'], basis)
        assert all(-1e-12 <= v <= 1 + 1e-12 for v in u)
        back = lp.from_x(lp.x_from_u(u, basis))
        assert all(math.isclose(back[k], record['params'][k], rel_tol=1e-9, abs_tol=1e-12) for k in lp.NAMES)


def test_box_constraints_in_pca_coordinates_are_the_physical_box():
    basis = lp.fit_basis(_records(), make_config(), 0)
    A, b = lp.box_constraints(basis, BOX)
    assert len(A) == 8
    rng = numpy.random.default_rng(5)
    inside = outside = 0
    for u in rng.random((4000, 4)):
        params = lp.from_x(lp.x_from_u(u, basis))
        margin = min(min((params[k] - BOX[k][0]), (BOX[k][1] - params[k])) / (BOX[k][1] - BOX[k][0]) for k in lp.NAMES)
        if abs(margin) < 1e-6:
            continue
        assert lp.feasible(u, A, b, [0] * 4, [1] * 4) == (margin > 0)
        inside, outside = inside + (margin > 0), outside + (margin < 0)
    assert inside > 50 and outside > 50


def test_std_floor_and_identity_basis_without_points():
    records = [dict(r, params=dict(r['params'], gh=0.0, rh=30.0)) for r in _records()]
    basis = lp.fit_basis(records, make_config(), 0)
    assert math.isclose(basis['std'][1], 0.05 * 0.2) and math.isclose(basis['std'][2], 0.05 * math.log10(3))
    empty = lp.fit_basis([], make_config(), 0)
    assert numpy.allclose(empty['components'], numpy.eye(4)) and empty['cutoff'] is None
    assert numpy.allclose(lp.to_u(dict(Q=0.9, gh=0.0, rh=10.0, rho0=15.0), empty), 0)


def write_state(path, records, config=None, runs=None):
    cfg = dict(opt_seed=42, pot='l24', incl=90.0, bounds=BOUNDS)
    cfg.update(config or {})
    path.mkdir(parents=True, exist_ok=True)
    (path / 'profile_state.json').write_text(json.dumps(dict(
        config=cfg, runs=[dict(CONTEXT)] if runs is None else runs, evaluations=records)))
    return path


def qrecord(i, Q, params, penalty, status='ok', kind='bo', source='qprofile', error=None):
    return dict(id=i, Q=Q, params=params, penalty=penalty, status=status, kind=kind, source=source,
                upsilon=0.7, error=error)


def test_prior_states_keep_seed_42_points_inside_the_box(tmp_path):
    good = dict(gh=0.0, rh=29.26, rho0=26.84)
    records = [qrecord(0, 0.95, good, 1.4793),
               qrecord(1, 0.28, dict(gh=0.0, rh=2.5, rho0=100.0), 1.6),
               qrecord(2, 1.0, dict(gh=0.0, rh=30.0, rho0=41.76), 1.56),
               qrecord(3, 0.95, dict(gh=0.1, rh=20.0, rho0=30.0), None, status='failed', error='boom'),
               qrecord(4, 0.95, dict(gh=0.2, rh=12.0, rho0=59.0), None, status='failed',
                       error='ProtocolError: exp: evaluation failed (penalty=3e7)'),
               qrecord(5, 1.0, dict(gh=0.0, rh=25.0, rho0=33.0), 1.75, kind='prior', source='q29'),
               qrecord(6, 1.05, dict(gh=0.0, rh=30.0, rho0=40.0), 3.0, status='rejected')]
    one = write_state(tmp_path / 'one', records)
    two = write_state(tmp_path / 'two', [qrecord(0, 0.95, good, 1.4793)])
    priors, skipped, outside, context, sources = lp.read_prior_states([str(one), str(two / 'profile_state.json')],
                                                                      BOX, BOUNDS, 90.0)
    assert [p['source'] for p in priors] == ['one:0', 'one:2', 'one:4']
    assert priors[2]['status'] == 'rejected' and priors[2]['penalty'] == 3e7
    assert priors[1]['params'] == dict(Q=1.0, gh=0.0, rh=30.0, rho0=41.76)
    assert outside == 1 and context == CONTEXT and [s['name'] for s in sources] == ['one', 'two']
    assert skipped == ['one:3 (failed)', 'one:6 (rejected)', 'two:0 (duplicate)']


@pytest.mark.parametrize('config, runs, message', [
    (dict(opt_seed=7), None, 'opt_seed=7'),
    (dict(pot='l4'), None, 'pot=l4'),
    (dict(incl=60.0), None, 'incl=60.0'),
    (dict(bounds=dict(BOUNDS, rh=[0.5, 7.0])), None, 'bounds differ'),
    (None, [], 'evaluation context'),
    (None, [dict(CONTEXT), dict(CONTEXT, harness_sha256='other')], 'evaluation context'),
    (None, [dict(CONTEXT, geom_hash=None)], 'evaluation context'),
])
def test_incompatible_prior_states_are_refused(tmp_path, config, runs, message):
    path = write_state(tmp_path / 's', [], config=config, runs=runs)
    with pytest.raises(lp.ConfigError, match=message):
        lp.read_prior_states([str(path)], BOX, BOUNDS, 90.0)


def test_prior_states_must_share_one_context_and_match_the_live_module(tmp_path):
    one = write_state(tmp_path / 'one', [])
    two = write_state(tmp_path / 'two', [], runs=[dict(CONTEXT, evaluation_context='ctx1')])
    with pytest.raises(lp.ConfigError, match='differs from the other'):
        lp.read_prior_states([str(one), str(two)], BOX, BOUNDS, 90.0)
    config = make_config()
    lp.check_prior_context(config, dict(CONTEXT))
    lp.check_prior_context(make_config(prior_context=None), dict(harness_sha256='x'))
    with pytest.raises(lp.ConfigError, match='harness_sha256'):
        lp.check_prior_context(config, dict(harness_sha256='changed'))


# --------------------------------------------------------------------------
# Search schedule
# --------------------------------------------------------------------------
def test_sobol_then_turbo_with_seed_42_inside_the_box(tmp_path):
    recorder = Recorder()
    run = search(tmp_path, evaluator=recorder)
    run.open(resume=False)
    run.run()
    evaluations = run.state['evaluations']
    assert [r['kind'] for r in evaluations[:4]] == ['prior'] * 4
    new = run.new_records()
    assert len(new) == 20 == len(recorder.calls) and run.state['status'] == 'done'
    assert [r['kind'] for r in new] == ['sobol'] * 4 + ['bo'] * 16
    assert {r['proposal'] for r in new if r['kind'] == 'bo'} == {'acquisition', 'projected'}
    assert {c['seed'] for c in recorder.calls} == {42}
    assert all(lp.in_box(c['params'], BOX, tol=0) and c['Q'] == c['params']['Q'] for c in recorder.calls)
    assert [b['at'] for b in run.state['bases']] == [4, 10, 16]
    assert {r['basis'] for r in new if r['kind'] == 'bo'} == {0, 1, 2}
    assert min(r['penalty'] for r in new) < 1.4793 + 0.5
    table = (tmp_path / 'state' / lp.TABLE_NAME).read_text().splitlines()
    assert table[0].split('\t') == list(lp.TABLE_COLUMNS) and len(table) == 25


def test_infeasible_proposals_are_replaced_by_feasible_points(tmp_path):
    recorder = Recorder()
    run = search(tmp_path, config=make_config(budget=10, n_sobol=0), evaluator=recorder, proposer=outside_proposer)
    run.open(resume=False)
    run.run()
    assert len(recorder.calls) == 10 and all(lp.in_box(c['params'], BOX, tol=0) for c in recorder.calls)


def test_proposal_failure_falls_back_to_a_feasible_point(tmp_path):
    recorder = Recorder()
    run = search(tmp_path, config=make_config(budget=5, n_sobol=0), evaluator=recorder,
                 proposer=lambda *a: None)
    run.open(resume=False)
    run.run()
    assert len(recorder.calls) == 5 and all(lp.in_box(c['params'], BOX, tol=0) for c in recorder.calls)
    assert lp.summarize(run.state)['proposals'] == dict(acquisition=0, projected=0, fallback=5)


def feasible_proposer(points, values, center, lo, hi, A, b, seed):
    return lp.feasible_point(center, A, b, lo, hi, seed, center)


def test_self_test_requires_a_feasible_acquisition_point():
    ok, params = lp.self_test(BOX, proposer=feasible_proposer)
    assert ok and lp.in_box(params, BOX, tol=0)
    assert lp.self_test(BOX, proposer=lambda *a: None) == (False, None)
    assert not lp.self_test(BOX, proposer=outside_proposer)[0]


def test_trust_region_rules_and_restart(tmp_path):
    run = search(tmp_path)
    run.open(resume=False)
    tr = run.state['tr']
    for _ in range(3):
        run.update_tr(True)
    assert math.isclose(tr['length'], 1.2)
    run.update_tr(True), run.update_tr(True), run.update_tr(True)
    assert math.isclose(tr['length'], 1.6)
    for _ in range(4):
        run.update_tr(False)
    assert math.isclose(tr['length'], 0.8)
    for _ in range(4 * 7):
        run.update_tr(False)
    assert math.isclose(tr['length'], 0.6) and tr['restarts'] == 1


def test_duplicate_proposals_cost_no_evaluation_and_stop_a_stalled_search(tmp_path):
    recorder = Recorder()
    run = search(tmp_path, config=make_config(budget=10, n_sobol=2, max_dup_run=5), evaluator=recorder,
                 proposer=center_proposer)
    run.open(resume=False)
    run.run()
    assert len(recorder.calls) == 2 and run.state['status'] == 'stalled'
    assert run.state['duplicates'] == 5 and run.state['tr']['fail'] == 1


def test_resume_after_stop_reproduces_the_uninterrupted_run(tmp_path):
    full = search(tmp_path / 'a')
    full.open(resume=False)
    full.run()
    reference = [(r['kind'], r['params']) for r in full.state['evaluations']]
    first = search(tmp_path / 'b', evaluator=Recorder(stop_at=9))
    first.open(resume=False)
    with pytest.raises(lp.StopRun):
        first.run()
    second = search(tmp_path / 'b')
    second.open(resume=True, run_info=dict(suffix='r2'))
    second.run()
    assert [(r['kind'], r['params']) for r in second.state['evaluations']] == reference
    assert second.state['runs'][-1]['suffix'] == 'r2'


def test_state_guards_and_budget_extension(tmp_path):
    with pytest.raises(lp.ConfigError):
        search(tmp_path).open(resume=True)
    run = search(tmp_path, config=make_config(budget=6))
    run.open(resume=False)
    run.run()
    with pytest.raises(lp.ConfigError, match='exists'):
        search(tmp_path, config=make_config(budget=6)).open(resume=False)
    with pytest.raises(lp.ConfigError, match='box'):
        search(tmp_path, config=make_config(budget=6, box=dict(BOX, Q=[0.9, 1.1]))).open(resume=True)
    with pytest.raises(lp.ConfigError, match='only grow'):
        search(tmp_path, config=make_config(budget=5)).open(resume=True)
    more = search(tmp_path, config=make_config(budget=9))
    more.open(resume=True)
    more.run()
    assert len(more.new_records()) == 9 and more.state['config']['budget'] == 9


def test_failures_abort_but_rejected_points_do_not(tmp_path):
    run = search(tmp_path / 'f', evaluator=Recorder(fail=True), max_failures=3)
    run.open(resume=False)
    with pytest.raises(lp.RunFailed):
        run.run()
    assert len(run.new_records()) == 3
    rejected = search(tmp_path / 'r', config=make_config(budget=8, n_sobol=2), evaluator=Recorder(reject=True),
                      max_failures=3)
    rejected.open(resume=False)
    rejected.run()
    assert [r['status'] for r in rejected.new_records()] == ['rejected'] * 8
    assert rejected.best()['source'] == 'a:24'


def test_without_any_successful_point_the_trust_region_has_no_centre(tmp_path):
    run = search(tmp_path, config=make_config(priors=[], n_sobol=1), evaluator=Recorder(reject=True))
    run.open(resume=False)
    with pytest.raises(lp.RunFailed, match='centre'):
        run.run()


def test_summary_reports_best_edges_and_improvement(tmp_path):
    run = search(tmp_path, config=make_config(budget=8, n_sobol=2))
    run.open(resume=False)
    run.run()
    summary = lp.summarize(run.state)
    assert summary['n_new'] == 8 and summary['n_priors'] == 4 and summary['best_prior']['source'] == 'a:24'
    assert summary['best_prior']['box_edges'] == ['gh=lo'] and summary['best_prior']['bounds_edges'] == ['gh=lo']
    assert math.isclose(summary['improvement'], 1.4793 - summary['best_new']['penalty'])
    assert summary['near_best']['0.5']['n'] >= 1 and [b['at'] for b in summary['bases']] == [2]
    edge = lp._point(run.state['evaluations'][1], run.cfg)
    assert edge['box_edges'] == ['gh=lo', 'rh=hi'] and edge['bounds_edges'] == ['gh=lo', 'rh=hi']
    assert 'Local PCA search: status=done' in lp.format_summary(summary)


# --------------------------------------------------------------------------
# CLI, history names, harness contract
# --------------------------------------------------------------------------
def test_dry_run_and_summarize_need_only_the_standard_library(tmp_path):
    state = write_state(tmp_path / 'qprofile_x', [qrecord(0, 0.95, dict(gh=0.0, rh=29.26, rho0=26.84), 1.4793)],
                        runs=[dict(CONTEXT, harness_sha256=lp.rqp.sha256_file(str(SCRIPT)))])
    env = dict(os.environ, PYTHONPATH='')
    python = '/usr/bin/python3' if Path('/usr/bin/python3').exists() else sys.executable
    result = subprocess.run([python, '-S', str(DRIVER), '--dry-run', '--box', BOX_SPEC, '--prior-states', str(state)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert 'priors in the box: 1 (rejected 0)' in result.stdout and 'best prior: P=1.4793' in result.stdout
    assert 'evaluations: 60 (8 Sobol + 52 TuRBO)' in result.stdout
    other = write_state(tmp_path / 'qprofile_y', [], runs=[dict(CONTEXT)])
    bad = subprocess.run([python, '-S', str(DRIVER), '--dry-run', '--box', BOX_SPEC, '--prior-states', str(other)],
                         capture_output=True, text=True, env=env)
    assert bad.returncode == 2 and 'harness_sha256' in bad.stderr
    for args in (['--box', 'Q=0.9:1.05'], ['--box', BOX_SPEC, '--budget', '0'], ['--box', BOX_SPEC, '--n-sobol', '61']):
        bad = subprocess.run([python, '-S', str(DRIVER), '--dry-run', *args], capture_output=True, text=True, env=env)
        assert bad.returncode == 2, bad.stdout
    missing = subprocess.run([python, '-S', str(DRIVER), '--dry-run'], capture_output=True, text=True, env=env)
    assert missing.returncode == 2 and '--box is required' in missing.stderr

    run = search(tmp_path, config=make_config(budget=6, n_sobol=2))
    run.open(resume=False)
    run.run()
    result = subprocess.run([python, '-S', str(DRIVER), '--summarize', str(tmp_path / 'state')],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'state' / (lp.REPORT_NAME + '.json')).read_text())
    assert report['n_new'] == 6 and report['opt_seed'] == 42


def test_history_names_are_not_read_by_any_pool_glob():
    names = ['localpca_tycho_d1_nb250_gh0_l24n184_ser0_localpca20261103_120000.txt',
             lp.STATE_NAME, lp.TABLE_NAME, lp.REPORT_NAME + '.json']
    tags = ['d1_nb250_gh0_l24n184_ser0', 'Q1d1_nb250_gh0_ser0', 'd1_nb250_gh0_ser0', 'Q1d1_nb250_gh0_l24n184_ser0']
    patterns = ([f'out_*_{t}{s}.txt' for t in tags for s in ('_*', '')]
                + [f'out_tycho_{t}{s}.txt' for t in tags for s in ('', '_p*')]
                + ['4UpsBoTorch_PCA_Sersic*.txt', '4UpsBoTorch_PCA_PA46.8_Sersic_*.txt', 'qprofile_*'])
    assert not [(n, p) for n in names for p in patterns if fnmatch.fnmatch(n, p)]


def test_the_evaluator_is_shared_with_the_q_profile():
    assert inspect.signature(lp.rqp.build_evaluator).parameters['prefix'].default == 'qprofile'
    assert lp.PREFIX == 'localpca' and lp.DEFAULTS['opt_seed'] == 42
    assert lp.DEFAULTS['subsample_seed'] == lp.rqp.DEFAULTS['subsample_seed']
