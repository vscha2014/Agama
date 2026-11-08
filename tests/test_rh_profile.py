"""Diagnostic rh profiles with synthetic evaluations; no AGAMA/BoTorch/cloud."""
import copy
import importlib.util
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'py/run_rh_profile.py'
spec = importlib.util.spec_from_file_location('run_rh_profile', DRIVER)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
CONTEXT = dict(geom_hash='geometry-a', evaluation_context='seed42-L24', harness_sha256='mock')


def config(tmp_path, *args):
    starts = tmp_path / 'starts.txt'
    starts.write_text('0.95 0 20 30 2\n0.4 0 2 80 3\n')
    return rp.build_config(rp.parse_args([
        '--profile-starts', str(starts), '--beta-grid', '3,7', '--rh-grid', '1,30',
        '--cell-budget', '12', '--cell-sobol', '3', *args]), rp.rsm.script_bounds())


def fake_sobol(n, seed, d=4):
    rng = random.Random(seed)
    return [[rng.random() for _ in range(d)] for _ in range(n)]


def proposer(points, values, center, lo, hi, A, b, seed):
    rng = random.Random(seed)
    return [rng.uniform(a, z) for a, z in zip(lo, hi)]


def search_factory(*args, **kwargs):
    return rp.CellSearch(*args, proposer=proposer, sobol=fake_sobol, **kwargs)


class Recorder:
    def __init__(self, stop_at=None, status='ok'):
        self.calls, self.stop_at, self.status = [], stop_at, status

    def __call__(self, Q, params, seed, meta):
        if len(self.calls) == self.stop_at:
            raise rp.StopRun('synthetic stop')
        self.calls.append((dict(params), seed, dict(meta)))
        if self.status == 'failed':
            return dict(status='failed', error='synthetic failure')
        if self.status == 'rejected':
            return dict(status='rejected', penalty=3e7)
        penalty = 1 + math.log(params['rh'] / params['beta']) ** 2
        penalty += .01 * (Q ** 2 + params['gh'] ** 2 + math.log10(params['rho0']) ** 2 + params['alpha'] ** 2)
        return dict(status='ok', penalty=penalty, upsilon=.6, wall_s=1, orbit_s=.5)


def profile(cfg, path, recorder, resume=False, context=CONTEXT, **kwargs):
    run = rp.Profile(cfg, str(path), recorder, search_factory=search_factory, log=lambda *a: None, **kwargs)
    run.open(resume, context)
    return run


def cell_states(run):
    return [json.loads((Path(run.cell_dir(b, r)) / rp.lp.STATE_NAME).read_text()) for b, r in rp.cells(run.cfg)]


def test_profiles_fix_rh_beta_keep_all_four_nuisance_directions_and_common_random_numbers(tmp_path):
    cfg, rec = config(tmp_path), Recorder()
    run = profile(cfg, tmp_path / 'run', rec)
    run.run()
    assert len(rec.calls) == 4 * 12
    assert {seed for _, seed, _ in rec.calls} == {42}
    assert {p['beta'] for p, _, _ in rec.calls} == {3., 7.}
    sobol_nuisance = []
    for (beta, rh), state in zip(rp.cells(cfg), cell_states(run)):
        assert state['status'] == 'done'
        assert all(e['params']['beta'] == beta and e['params']['rh'] == rh for e in state['evaluations'])
        assert all(e['source'] == 'rhprofile' for e in state['evaluations'])
        assert all(b['free'] == [0, 1, 3, 4] and numpy.shape(b['components']) == (4, 4)
                   and 'dropped' not in b for b in state['bases'])
        for key in ('Q', 'gh', 'rho0', 'alpha'):
            assert len({e['params'][key] for e in state['evaluations']}) > 3
        sobol_nuisance.append([[e['params'][k] for k in ('Q', 'gh', 'rho0', 'alpha')]
                              for e in state['evaluations'] if e['kind'] == 'sobol'])
        header = (Path(run.cell_dir(beta, rh)) / 'search.tsv').read_text().splitlines()[0].split('\t')
        assert header[header.index('rho0') + 1:header.index('rho0') + 3] == ['alpha', 'beta']
    assert all(s == sobol_nuisance[0] for s in sobol_nuisance)
    summary = rp.summarize(str(tmp_path / 'run'))
    assert summary['status'] == 'done' and len(summary['rows']) == 4
    assert all(r['params']['beta'] == r['beta'] and r['params']['rh'] == r['rh'] for r in summary['rows'])
    assert all('rh=lo' not in r['boundary'] and 'rh=hi' not in r['boundary'] for r in summary['rows'])


def test_stop_resume_reproduces_an_uninterrupted_profile_and_extension_keeps_all_rows(tmp_path):
    cfg = config(tmp_path)
    reference = profile(cfg, tmp_path / 'reference', Recorder())
    reference.run()
    first = Recorder(stop_at=17)
    stopped = profile(cfg, tmp_path / 'resumed', first)
    with pytest.raises(rp.StopRun):
        stopped.run()
    resumed_calls = Recorder()
    resumed = profile(cfg, tmp_path / 'resumed', resumed_calls, resume=True)
    resumed.run()
    assert len(first.calls) + len(resumed_calls.calls) == 48
    for expected, actual in zip(cell_states(reference), cell_states(resumed)):
        assert [(e['params'], e['penalty'], e['kind']) for e in actual['evaluations']] == [
            (e['params'], e['penalty'], e['kind']) for e in expected['evaluations']]
    old = cell_states(resumed)
    extended_cfg, extension = dict(cfg, cell_budget=14), Recorder()
    extended = profile(extended_cfg, tmp_path / 'resumed', extension, resume=True)
    extended.run()
    assert len(extension.calls) == 8
    for before, after in zip(old, cell_states(extended)):
        assert after['evaluations'][:12] == before['evaluations']
        assert after['config']['initial_points'] == before['config']['initial_points']
        assert len(after['evaluations']) == 14


@pytest.mark.parametrize('change', ['grid', 'beta', 'box', 'seed', 'budget', 'geometry'])
def test_resume_refuses_changed_scientific_inputs_without_rewriting_the_manifest(tmp_path, change):
    cfg = config(tmp_path)
    original = profile(cfg, tmp_path / 'run', Recorder())
    before = Path(original.path).read_bytes()
    edited, context = copy.deepcopy(cfg), dict(CONTEXT)
    if change == 'grid':
        edited['rh_grid'] = [1., 20., 30.]
    elif change == 'beta':
        edited['beta_grid'] = [3., 4.]
    elif change == 'box':
        edited['box']['gh'] = [0., .2]
    elif change == 'seed':
        edited['opt_seed'] = 43
    elif change == 'budget':
        edited['cell_budget'] = 11
    else:
        context['geom_hash'] = 'different'
    with pytest.raises(rp.ConfigError):
        profile(edited, tmp_path / 'run', Recorder(), resume=True, context=context)
    assert Path(original.path).read_bytes() == before


def test_rejected_cells_have_no_fake_minimum_and_failed_evaluations_stop(tmp_path):
    cfg = config(tmp_path)
    run = profile(cfg, tmp_path / 'rejected', Recorder(status='rejected'))
    run.run()
    report = rp.summarize(str(tmp_path / 'rejected'))
    assert all(r['status'] == 'no_valid_model' and r['penalty'] is None for r in report['rows'])
    assert all(best is None for best in report['minima'].values())
    failed = profile(cfg, tmp_path / 'failed', Recorder(status='failed'), max_failures=2)
    with pytest.raises(rp.RunFailed, match='consecutive failed'):
        failed.run()
    row = rp.summarize(str(tmp_path / 'failed'))['rows'][-2]
    assert row['status'] == 'pending'  # later beta branch was not silently attempted
    states = list((tmp_path / 'failed').glob('beta_*/search_state.json'))
    assert len(states) == 1 and len(json.loads(states[0].read_text())['evaluations']) == 2


@pytest.mark.parametrize('args, match', [
    (['--rh-grid', '1,31'], 'finite values'),
    (['--rh-grid', '1,1,3'], 'increasing'),
    (['--beta-grid', '3,5,10'], 'choose from'),
    (['--beta-grid', '3,nan'], 'finite values'),
    (['--cell-budget', '6'], 'must exceed'),
    (['--incl', '20'], 'cos'),
])
def test_bad_plans_are_rejected(tmp_path, args, match):
    with pytest.raises(rp.ConfigError, match=match):
        config(tmp_path, *args)


def test_dry_run_is_stdlib_only_and_does_not_create_outputs(tmp_path):
    out = tmp_path / 'output'
    result = subprocess.run([sys.executable, '-S', str(DRIVER), '--dry-run', '--state-dir', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '<= 672 new evaluations' in result.stdout and 'no dropped direction' in result.stdout
    assert "'Q': [0.05, 2.5]" in result.stdout and "'gh': [0.0, 1.6]" in result.stdout
    assert not out.exists()


def test_main_rejects_incompatible_resume_before_importing_the_harness(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    profile(cfg, tmp_path / 'run', Recorder())
    monkeypatch.setattr(rp.rqp, 'build_evaluator', lambda *a, **kw: pytest.fail('must reject before import'))
    assert rp.main(['--resume', '--state-dir', str(tmp_path / 'run'), '--beta-grid', '3,4']) == 2


def test_resume_dry_run_inherits_extended_budget_and_refuses_changed_input(tmp_path, capsys):
    cfg = config(tmp_path, '--cell-budget', '32')
    profile(cfg, tmp_path / 'run', Recorder())
    args = ['--resume', '--dry-run', '--state-dir', str(tmp_path / 'run'),
            '--profile-starts', str(tmp_path / 'starts.txt'), '--beta-grid', '3,7',
            '--rh-grid', '1,30', '--cell-sobol', '3']
    assert rp.main(args) == 0
    assert '<= 128 new evaluations' in capsys.readouterr().out
    assert rp.main(args + ['--cell-budget', '24']) == 2
