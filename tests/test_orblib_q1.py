import argparse
import ast
import datetime
import fnmatch
import glob
import hashlib
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
LAUNCHER = ROOT / 'py/launch_orblib_exp.sh'
TREE = ast.parse(SCRIPT.read_text())
BOUNDS = dict(Q=(0.05, 2.5), gh=(0.0, 1.6), rh=(0.5, 7.0), rho0=(10.0, 120.0))


@pytest.mark.parametrize('q1', [False, True])
def test_expanded_search_bounds(q1):
    assignments = [node for node in ast.walk(TREE) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'bounds_original'
                           for target in node.targets)]
    assert len(assignments) == 2
    identity = SimpleNamespace(inverse_transform=lambda x: x, n_components_=4)
    model = dict(scaler=identity, pca=identity, use_log_scale=False)
    convert = functions('pca_to_params_fixed', q1=q1)['pca_to_params_fixed']
    for node in assignments:
        bounds = ast.literal_eval(node.value)
        assert {name: bounds[name] for name in BOUNDS} == BOUNDS
        for rh, rho0 in [(0.5, 120.0), (5.0, 20.0), (7.0, 10.0)]:
            params = convert([0.8, 0.4, rh, rho0], model, bounds)
            assert params == dict(Q=1.0 if q1 else 0.8, gh=0.4, rh=rh, rho0=rho0)


def functions(*names, q1=True, **extra):
    namespace = dict(numpy=numpy, os=os, glob=glob, hashlib=hashlib,
                     datetime=datetime, pickle=pickle, Q1=q1,
                     hostname_proc='test_Q1d1_nb250_gh0_ser0_p0', incl=90.0,
                     EXP_ID='Q1d1_nb250_gh0_ser0', **extra)
    nodes = [node for node in TREE.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), 'exec'), namespace)
    return namespace


def configuration(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), *args])
    start = next(i for i, node in enumerate(TREE.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'parser')
    end = next(i for i, node in enumerate(TREE.body)
               if isinstance(node, ast.Import) and node.names[0].name == 'agama')
    namespace = dict(argparse=argparse, numpy=numpy, os=os, socket=socket, hashlib=hashlib)
    exec(compile(ast.Module(body=TREE.body[start:end], type_ignores=[]), str(SCRIPT), 'exec'),
         namespace)
    return namespace


def test_cli_and_shared_history_with_separate_writers(monkeypatch):
    free = configuration(monkeypatch)
    fixed = configuration(monkeypatch, '--Q1')
    assert fixed['DOUBLE'] is True
    assert fixed['EXP_ID'] == 'Q1d1_nb250_gh0_ser0'
    assert free['EXP_ID'] == 'd1_nb250_gh0_ser0'
    assert fixed['storage_patterns'] == free['storage_patterns']
    assert fixed['host_patterns'] == free['host_patterns']
    for suffix in ['', '_p0', '_p12']:
        for tag in [fixed['EXP_ID'], free['EXP_ID']]:
            name = f'out_other_{tag}{suffix}.txt'
            assert any(fnmatch.fnmatch(name, p) for p in fixed['storage_patterns'])
    for tag in ['d0_nb250_gh0_ser0', 'Q1d0_nb250_gh0_ser0', 'd1_nb2500_gh0_ser0',
                'd1_nb250_gh1_ser0', 'd1_nb250_gh0_ser1', 'Q1d1_nb200_gh0_ser0']:
        assert not any(fnmatch.fnmatch(f'out_other_{tag}_p0.txt', p)
                       for p in fixed['storage_patterns'])
    undoubled = configuration(monkeypatch, '--no-double')
    assert not any(fnmatch.fnmatch('out_other_Q1d0_nb250_gh0_ser0_p0.txt', p)
                   for p in undoubled['storage_patterns'])
    assert fixed['UpsFile'] != free['UpsFile']
    assert fixed['hostname_proc'] != free['hostname_proc']
    for args in [('--Q1', '--no-double'), ('--q1',)]:
        with pytest.raises(SystemExit) as error:
            configuration(monkeypatch, *args)
        assert error.value.code == 2


@pytest.mark.parametrize('q1', [False, True])
def test_seed_candidates_are_fixed_before_reservation(q1):
    ns = functions('select_bootstrap_candidates', q1=q1)
    rows = numpy.array([[90, 0.2, 0.4, 1.5, 70, 0.6, 1],
                        [90, 2.0, 0.8, 2.0, 90, 0.6, 2]])
    candidates = ns['select_bootstrap_candidates'](rows, 2)
    assert [p['Q'] for p in candidates] == ([1.0, 1.0] if q1 else [0.2, 2.0])
    numpy.testing.assert_array_equal(rows[:, 1], [0.2, 2.0])


@pytest.mark.parametrize('q1', [False, True])
def test_lhs_reserves_and_evaluates_same_parameters(tmp_path, q1):
    reserved, evaluated = [], []

    def reserve(params, *args):
        reserved.append(params.copy())
        return None, False

    def evaluate(*args, direct_params):
        evaluated.append(direct_params.copy())
        return -0.5

    ns = functions('_generate_random_initial_points', '_params_to_dummy_pc', q1=q1,
                   proc_rng=numpy.random.default_rng(15),
                   _try_reserve_candidate=reserve, halo_IC_lib_weights_pca_fixed=evaluate,
                   release_reservation=lambda fp: None,
                   _periodic_bootstrap_sync=lambda *args: None,
                   densityStars=None, datasets=None, alphah=2, betah=3)
    rows, results, _ = ns['_generate_random_initial_points'](
        BOUNDS, 10, output_file=str(tmp_path / 'lhs.log'))
    assert reserved == evaluated
    assert len(results) == 10
    assert numpy.isfinite(rows).all()
    assert numpy.all(rows[:, 1] == 1) if q1 else numpy.ptp(rows[:, 1]) > 1
    for name in ('gh', 'rh', 'rho0'):
        values = [p[name] for p in evaluated]
        assert BOUNDS[name][0] <= min(values) < max(values) <= BOUNDS[name][1]


class NumpyPCA:
    def __init__(self, n_components):
        self.n_components_ = n_components

    def fit(self, x):
        if self.n_components_ > min(x.shape):
            raise ValueError('Too few samples or features for PCA')
        self.mean_ = x.mean(axis=0)
        _, singular, vt = numpy.linalg.svd(x - self.mean_, full_matrices=False)
        self.components_ = vt[:self.n_components_]
        self.explained_variance_ratio_ = (singular**2 / (singular**2).sum())[:self.n_components_]
        return self

    def transform(self, x):
        return (x - self.mean_) @ self.components_.T

    def inverse_transform(self, x):
        return x @ self.components_ + self.mean_


@pytest.mark.parametrize('penalty', [0.5, 107.125773968194466, 15922.86406406542])
def test_pca_has_three_directions_and_roundtrips(tmp_path, penalty):
    ns = functions('WeightedScaler', 'build_initial_pca_from_bootstrap',
                   'pca_to_params_fixed', 'params_to_pca_fixed', '_update_pca_model',
                   'adaptive_penalty_cutoff', PCA=NumpyPCA,
                   torch=SimpleNamespace(tensor=ArrayTensor))
    rng = numpy.random.default_rng(28)
    results = [dict(params=dict(Q=1.0, gh=rng.uniform(0, 1.6), rh=rng.uniform(0.5, 3.5),
                                rho0=rng.uniform(34, 120)), penalty=penalty) for _ in range(20)]
    model = ns['build_initial_pca_from_bootstrap'](
        results, BOUNDS, n_components=4, output_file=str(tmp_path / 'pca.log'))
    assert model['pca'].n_components_ == 3
    for result in results:
        params = result['params']
        pc = ns['params_to_pca_fixed'](params, model)
        recovered = ns['pca_to_params_fixed'](pc, model, BOUNDS)
        assert recovered['Q'] == 1.0
        numpy.testing.assert_allclose(list(recovered.values()), list(params.values()))
        projected = ns['params_to_pca_fixed'](dict(params, Q=2.3), model)
        numpy.testing.assert_allclose(pc, projected)
    for pc in rng.normal(size=(20, 3)) * 10:
        assert ns['pca_to_params_fixed'](pc, model, BOUNDS)['Q'] == 1.0
    params = results[0]['params']
    pc = ns['params_to_pca_fixed'](params, model)
    model_new, observed, _, turbo = ns['_update_pca_model'](
        model, model['data_good'], [r['params'] for r in results[:5]], [penalty - 0.1] * 5,
        BOUNDS, True, 2.5, ArrayTensor([pc]), ArrayTensor([[-0.5]]), SimpleNamespace(),
        str(tmp_path / 'update.log'), None, None, read_parallel=False, incl_filter=90.0)
    assert model_new['pca'].n_components_ == 3
    assert turbo.model_data is model_new
    recovered = ns['pca_to_params_fixed'](observed.numpy()[0], model_new, BOUNDS)
    numpy.testing.assert_allclose(list(recovered.values()), list(params.values()))
    assert recovered['Q'] == 1.0


@pytest.mark.parametrize('q1', [False, True])
def test_history_reads_both_modes_without_dedup(tmp_path, monkeypatch, q1):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOSTNAME_SUFFIX', 'testhost')
    config = configuration(monkeypatch, *(['--Q1'] if q1 else []))
    calls = []
    ns = functions('load_fresh_data_from_files', q1=q1,
                   load_from_yadisk=lambda *a, **kw: calls.append(kw))
    for host, suffix in [('testhost', '_p0'), ('other', '')]:
        for prefix in ['', 'Q1']:
            name = f'out_{host}_{prefix}d1_nb250_gh0_ser0{suffix}.txt'
            content = '90 1 0.4 1.5 70 0.6 1\n80 1 0.4 1.5 70 0.6 0.1\n'
            if not prefix:
                content += '90 0.5 0.4 1.5 70 0.6 0.1\n'
            (tmp_path / name).write_text(content)
    patterns, host_patterns = config['storage_patterns'], config['host_patterns']
    read = ns['load_fresh_data_from_files']
    data, counts = read(patterns, host_patterns, 90, return_full=True, exclude_suffix=None)
    expected = 4 if q1 else 6
    assert len(data) == expected
    assert len(counts) == 4
    assert numpy.all(data[:, 1] == 1) if q1 else numpy.any(data[:, 1] == 0.5)
    assert calls[-1]['force_update'] is True
    params, penalties, _ = read(patterns, host_patterns, 90, exclude_suffix=None)
    numpy.testing.assert_array_equal(params, data[:, 1:5])
    numpy.testing.assert_array_equal(penalties, data[:, 6])
    own_excluded, _ = read(patterns, host_patterns, 90, return_full=True,
                           exclude_suffix=config['hostname_proc'])
    assert len(own_excluded) == expected - (1 if q1 else 2)
    with (tmp_path / 'out_other_Q1d1_nb250_gh0_ser0.txt').open('a') as stream:
        stream.write('90 1 0.8 2 90 0.6 0.7\n')
    fresh, _ = read(patterns, host_patterns, 90, return_full=True, exclude_suffix=None)
    assert len(fresh) == expected + 1


@pytest.mark.parametrize('q1', [False, True])
def test_shared_yadisk_refresh_protects_both_local_modes(tmp_path, monkeypatch, q1):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOSTNAME_SUFFIX', 'testhost')
    config = configuration(monkeypatch, *(['--Q1'] if q1 else []))
    remote, own, calls = {}, [], []
    for host in ['testhost', 'other']:
        for prefix in ['', 'Q1']:
            for suffix in ['', '_p0']:
                name = f'out_{host}_{prefix}d1_nb250_gh0_ser0{suffix}.txt'
                (tmp_path / name).write_text('90 1 0.4 1.5 70 0.6 1\n')
                remote[name] = '90 1 0.4 1.5 70 0.6 0.8\n'
                if host == 'testhost':
                    own.append(name)
    remote['out_other_d0_nb250_gh0_ser0.txt'] = 'incompatible experiment'

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ['rclone', 'lsf']:
            return SimpleNamespace(returncode=0, stdout='\n'.join(remote), stderr='')
        assert command[:2] == ['rclone', 'copyto']
        name = command[3]
        assert name not in own
        Path(name).write_text(remote[name])
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    ns = functions('load_fresh_data_from_files', 'load_from_yadisk', q1=q1,
                   subprocess=SimpleNamespace(run=run), RCLONE_REMOTE='mock',
                   UpsFile=config['UpsFile'])
    read = ns['load_fresh_data_from_files']
    read(config['storage_patterns'], config['host_patterns'], 90,
         exclude_suffix=None, notify=False)
    copied = [cmd[3] for cmd in calls if cmd[1] == 'copyto']
    assert set(copied) == set(remote) - set(own) - {'out_other_d0_nb250_gh0_ser0.txt'}
    assert all(Path(name).read_text().endswith('0.6 1\n') for name in own)
    assert not Path('out_other_d0_nb250_gh0_ser0.txt').exists()
    name = 'out_other_Q1d1_nb250_gh0_ser0.txt'
    remote[name] += '90 1 0.8 2 90 0.6 0.7\n'
    data, _ = read(config['storage_patterns'], config['host_patterns'], 90,
                   return_full=True, exclude_suffix=None, notify=False)
    assert len(data) == 9
    assert Path(name).read_text() == remote[name]


@pytest.mark.parametrize('q1', [False, True])
def test_nearest_incl_uses_shared_history_with_q_filter(tmp_path, monkeypatch, q1):
    monkeypatch.chdir(tmp_path)
    config = configuration(monkeypatch, *(['--Q1'] if q1 else []))
    Path('out_other_d1_nb250_gh0_ser0.txt').write_text(
        '89 0.5 0.4 1.5 70 0.6 0.1\n88 1 0.4 1.5 70 0.6 0.8\n')
    Path('out_another_Q1d1_nb250_gh0_ser0_p0.txt').write_text(
        '88 1 0.8 2 90 0.6 0.7\n')
    calls = []
    ns = functions('find_nearest_incl_data', q1=q1,
                   load_from_yadisk=lambda *a, **kw: calls.append(kw))
    data, nearest, _ = ns['find_nearest_incl_data'](
        config['storage_patterns'], config['host_patterns'], 90, min_points=1)
    assert nearest == (88 if q1 else 89)
    assert len(data) == (2 if q1 else 1)
    assert calls[-1]['force_update'] is True


def test_launcher_configuration_matches_python(monkeypatch):
    source = LAUNCHER.read_text().split('MAIN_PID=$$')[0]
    probe = source + '\nprintf "%s\\n" "$EXP_ID" "$KEY" "$EXP_FLAGS"\n'
    for args in [[], ['--Q1'], ['--Q1', '--incl=60', '--n-bin=200']]:
        result = subprocess.run(['bash', '-s', '--', *args], input=probe,
                                text=True, capture_output=True, check=True)
        exp_id, key, flags = result.stdout.strip().splitlines()[-3:]
        ns = configuration(monkeypatch, *args)
        assert exp_id == ns['EXP_ID']
        assert key.startswith('i60.0_d1_' if '--incl=60' in args else 'i90.0_d1_')
        assert ('--Q1' in flags) == ('--Q1' in args)
        assert '--save-orblib' in flags and '--reuse-orblib' in flags
        assert '--no-double' not in flags
    result = subprocess.run(['bash', '-s', '--', '--Q1', '--no-double'], input=probe,
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert '--no-double' in result.stderr
    result = subprocess.run(['bash', '-s', '--', '--q1'], input=probe,
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert '--Q1' in result.stderr


class ArrayTensor:
    def __init__(self, values, **kwargs):
        self.values = numpy.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __len__(self):
        return len(self.values)

    def min(self):
        return ArrayTensor(self.values.min())

    def max(self):
        return ArrayTensor(self.values.max())

    def item(self):
        return self.values.item()

    def argmax(self):
        return ArrayTensor(self.values.argmax())

    def __getitem__(self, index):
        return ArrayTensor(self.values[index])


def test_q1_checkpoint_reprojects_physical_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = functions('save_checkpoint', 'q1_checkpoint_coords',
                   'pca_to_params_fixed', 'params_to_pca_fixed', 'WeightedScaler',
                   number_of_h_IC_lw=2, number_of_find_w_U=3,
                   best_overall_target=-0.5, best_overall_Upsilon=0.6)
    pca = NumpyPCA(3).fit(numpy.random.default_rng(4).normal(size=(10, 4)))
    model = dict(scaler=ns['WeightedScaler'](numpy.array([1, 0.5, 0.2, 1.9]),
                                           numpy.array([1, 0.1, 0.1, 0.1])),
                 pca=pca, use_log_scale=True)
    params = dict(Q=1.0, gh=0.5, rh=1.5, rho0=80.0)
    coordinates = ns['params_to_pca_fixed'](params, model)
    evaluated = ns['pca_to_params_fixed'](coordinates, model, BOUNDS)
    turbo = SimpleNamespace(length=0.6, success_count=0, failure_count=0,
                            model_data=model, bounds_original=BOUNDS)
    ns['save_checkpoint'](ArrayTensor([coordinates]), ArrayTensor([[-0.5]]),
                          turbo, 3, sync=False)
    state = pickle.loads(next(tmp_path.glob('checkpoint_*.pkl')).read_bytes())
    assert state['q1'] is True
    assert state['incl'] == 90.0
    assert state['params_obs'] == [evaluated]
    pca.components_ = -pca.components_
    restored = ns['q1_checkpoint_coords'](state, model)
    expected = ns['params_to_pca_fixed'](evaluated, model)
    numpy.testing.assert_allclose(restored, [expected])
    assert not numpy.allclose(restored, state['X_obs'])
    for bad_state in [dict(state, incl=60.0), dict(state, q1=False),
                      dict(state, params_obs=[dict(evaluated, Q=0.9)])]:
        with pytest.raises(ValueError):
            ns['q1_checkpoint_coords'](bad_state, model)


def test_direct_evaluation_and_orbit_key_use_q1():
    class StopEvaluation(BaseException):
        pass

    received = []

    def density(**kwargs):
        received.append(kwargs)
        raise StopEvaluation

    ns = functions('halo_IC_lib_weights_pca_fixed', 'orblib_key', 'OrblibBusyError',
                   release_reservation=lambda fp: None,
                   agama=SimpleNamespace(Density=density),
                   DOUBLE=True, N_BIN=250, SER_ID=0, GEOM_HASH='test', ORBLIB_DIR='orblib')
    params = dict(Q=0.2, gh=0.5, rh=1.5, rho0=80.0)
    with pytest.raises(StopEvaluation):
        ns['halo_IC_lib_weights_pca_fixed'](numpy.zeros(3), None, BOUNDS,
                                            None, None, 2, 3, direct_params=params)
    assert received[0]['axisratioz'] == 1.0
    assert params['Q'] == 1.0
    fixed_key = ns['orblib_key'](1, 0.5, 1.5, 80)
    ns['Q1'] = False
    assert ns['orblib_key'](1, 0.5, 1.5, 80) == fixed_key
    assert ns['orblib_key'](0.2, 0.5, 1.5, 80) != fixed_key


@pytest.mark.parametrize('resume,exit_code', [(False, 0), (True, 0), (False, 1)])
def test_launcher_with_mocked_external_commands(tmp_path, resume, exit_code):
    launcher = tmp_path / LAUNCHER.name
    launcher.write_text(LAUNCHER.read_text())
    (tmp_path / SCRIPT.name).touch()
    (tmp_path / 'table3.dat').touch()
    (tmp_path / 'orblib_storage.py').touch()
    uploader = tmp_path / 'upload_orblib_parts.sh'
    uploader.write_text('#!/bin/sh\nexit 0\n')
    uploader.chmod(0o755)
    config = tmp_path / '.config/rclone'
    config.mkdir(parents=True)
    (config / 'rclone.conf').touch()
    exp_id = 'Q1d1_nb250_gh0_ser0'
    checkpoint = tmp_path / f'checkpoint_testhost_{exp_id}_p0.pkl'
    if resume:
        checkpoint.write_bytes(b'mock checkpoint, never unpickled')
    other = tmp_path / 'reservations_i90.0_d1_nb250_gh0_ser0'
    other.mkdir()
    untouched = other / 'active.resv'
    untouched.write_text('another experiment')
    markers = [tmp_path / '.done_orblib_other_p0', tmp_path / '.upload_lock_orblib_other']
    for marker in markers:
        marker.touch()
    prefix = '''
hostname() { printf 'testhost'; }
nproc() { printf '4'; }
rclone() { printf 'rclone %s\\n' "$*" >> "$HOME/calls.txt"; }
curl() { return 0; }
python3() {
    printf 'storage %s\\n' "$*" >> "$HOME/calls.txt"
    mkdir -p "$WORK_DIR/orblib/.storage"
    case "$2" in
        watch)
            while [ ! -f "$WORK_DIR/orblib/.storage/FINISH" ] && [ ! -f "$WORK_DIR/orblib/.storage/STOP" ]; do sleep 0.02; done
            [ ! -f "$WORK_DIR/orblib/.storage/STOP" ] ;;
        finish) touch "$WORK_DIR/orblib/.storage/FINISH" ;;
        stop) touch "$WORK_DIR/orblib/.storage/STOP" ;;
    esac
    return 0
}
sudo() { printf 'UNEXPECTED sudo\\n' >> "$HOME/calls.txt"; return 1; }
docker() {
    printf 'docker %s\\n' "$*" >> "$HOME/calls.txt"
    if [ "$1" = image ]; then return 0; fi
    printf '90 1 0.5 1.5 80 0.6 0.5\\n' > "$WORK_DIR/out_${HOSTNAME_ENV}_${EXP_ID}_p0.txt"
    printf 'mock diagnostic\\n' > "$WORK_DIR/log_${HOSTNAME_ENV}_${EXP_ID}_p0.txt"
    return "$MOCK_EXIT_CODE"
}
source "$0" "$@"
'''
    args = ['bash', '-c', prefix, str(launcher), '--Q1', '--nproc=1', '--no-shutdown']
    if resume:
        args.append('--resume')
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, timeout=15,
                            env=dict(os.environ, HOME=str(tmp_path), MOCK_EXIT_CODE=str(exit_code)))
    assert result.returncode == exit_code, result.stdout + result.stderr
    calls = (tmp_path / 'calls.txt').read_text()
    assert '--Q1' in calls and '--no-double' not in calls
    assert ('--no-resume --delete-checkpoint' in calls) != resume
    assert f'agama_orblib_testhost_{exp_id}_i90.0_p0' in calls
    assert f'out_testhost_{exp_id}.txt' in calls
    assert (f'checkpoint_testhost_{exp_id}_p0.pkl' in calls) == (resume or exit_code == 0)
    assert 'UNEXPECTED' not in calls
    assert untouched.read_text() == 'another experiment'
    assert all(marker.exists() for marker in markers)
    merged = (tmp_path / f'out_testhost_{exp_id}.txt').read_text()
    assert merged.count('90 1 0.5 1.5 80 0.6 0.5') == 1


def test_q1_diagnostics_accept_partial_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = numpy.array([[90, 1, 0.4, 1.5, 70, 0.6, 0.5],
                        [90, 1, 0.8, 2.0, 90, 0.7, 0.6],
                        [90, 1, 1.0, 2.5, 80, 0.8, 0.7]])
    ns = functions('diagnose_pca_space', 'compare_good_vs_acceptable', PCA=NumpyPCA,
                   StandardScaler=lambda: SimpleNamespace(fit_transform=lambda x: x, transform=lambda x: x,
                                                          inverse_transform=lambda x: x),
                   load_fresh_data_from_files=lambda **kwargs: (rows, {}),
                   sync_to_yadisk=lambda: None)
    with numpy.errstate(invalid='ignore', divide='ignore'):
        result = ns['diagnose_pca_space']([], [], cutoff_start=2.0)
    numpy.testing.assert_array_equal(result, rows)


@pytest.mark.parametrize('function_name', [
    'build_initial_pca_from_bootstrap', '_update_pca_model', 'run_pca_optimization'])
def test_pca_weight_blocks_handle_observed_large_penalties(function_name):
    function = next(node for node in TREE.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    start = next(i for i, node in enumerate(function.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'weights')
    block = ast.Module(body=function.body[start:start + 3], type_ignores=[])
    code = compile(block, str(SCRIPT), 'exec')
    observed = numpy.array([107.125773968194466, 4069.903388131755,
                           1404.9730911971842, 458.20959966310744, 15922.86406406542,
                           13000.52767838662, 125.47926431460684, 2599.1494481986774,
                           565.2609135826413, 147.68411731146472])
    x = numpy.random.default_rng(18).normal(size=(10, 4))
    for penalties in [observed, numpy.linspace(0.5, 0.7, 10), numpy.full(10, 1000.0)]:
        data = numpy.zeros((10, 7))
        data[:, 6] = penalties
        ns = dict(numpy=numpy, penalties=penalties, pen_all=penalties, data_good=data,
                  X_tr=x, X_tr_all=x, X_transformed=x)
        exec(code, ns)
        assert ns['weights'].max() == 1.0
        assert numpy.isfinite(ns['weighted_mean']).all()
        assert numpy.isfinite(ns['weighted_std']).all()
        expected = numpy.exp(-(penalties - penalties.min()) / 0.1)
        numpy.testing.assert_allclose(ns['weighted_mean'], numpy.average(x, weights=expected, axis=0))
        if penalties.max() < 1:
            original = numpy.exp(-penalties / 0.1)
            numpy.testing.assert_allclose(ns['weighted_mean'], numpy.average(x, weights=original, axis=0))


def seed_namespace(tmp_path, monkeypatch, outcome=107.0, extra_functions=()):
    monkeypatch.chdir(tmp_path)
    evaluated, reserved, released, syncs = [], [], [], []
    history = tmp_path / 'out_test_Q1d1_nb250_gh0_ser0_p0.txt'
    ns = functions('seed_q1_from_nearest_q', 'load_prior_candidates_from_patterns',
                   '_params_to_dummy_pc', 'orblib_key', 'OrblibBusyError', *extra_functions,
                   DOUBLE=True, N_BIN=250, SER_ID=0, GEOM_HASH='test', ORBLIB_DIR='orblib',
                   densityStars=None, datasets=None, alphah=2, betah=3,
                   proc_rng=numpy.random.default_rng(12),
                   load_from_yadisk=lambda *a, **kw: None,
                   sync_to_yadisk=lambda: syncs.append(True),
                   _periodic_bootstrap_sync=lambda *a: None)

    def reserve(params, *args):
        reserved.append(params.copy())
        return str(len(reserved)), False

    def evaluate(*args, direct_params, **kwargs):
        assert args[1] is None
        assert direct_params['Q'] == 1.0
        evaluated.append(direct_params.copy())
        penalty = outcome(direct_params) if callable(outcome) else outcome
        if -penalty > ns.get('best_overall_target', -numpy.inf):
            ns.update(best_overall_target=-penalty, best_overall_Upsilon=0.6)
        if numpy.isfinite(penalty) and penalty < 1e5:
            with history.open('a') as stream:
                stream.write(f"90 1 {direct_params['gh']} {direct_params['rh']} "
                             f"{direct_params['rho0']} 0.6 {penalty}\n")
        return -penalty

    ns.update(_try_reserve_candidate=reserve, halo_IC_lib_weights_pca_fixed=evaluate,
              release_reservation=released.append)
    return ns, evaluated, reserved, released, syncs


def write_seed_history(path, count=40, incl=90):
    rows = numpy.array([[incl, 0.8 + i / 200, 0.1 + i / 100, 1 + i / 50,
                         50 + i, 0.6, 0.01 + i / 1000] for i in range(count)])
    numpy.savetxt(path, rows)
    return rows


@pytest.mark.parametrize('existing_count', [0, 10])
def test_nearest_q_seed_is_bounded_with_large_penalties(tmp_path, monkeypatch, existing_count):
    ns, evaluated, reserved, released, _ = seed_namespace(tmp_path, monkeypatch)
    rows = write_seed_history(tmp_path / 'prior.txt')
    existing = numpy.array([[90, 1, 1.5, 3, 115, 0.6, 107]] * existing_count).reshape(-1, 7)
    data, results = ns['seed_q1_from_nearest_q'](
        existing, [], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert len(evaluated) == 10
    assert reserved == evaluated
    assert len(released) == 10
    assert len(results) == 10
    assert len(data) == existing_count + 10
    assert numpy.all(data[:, 1] == 1.0)
    assert numpy.all(data[:, 6] == 107.0)
    nearest = rows[numpy.argsort(numpy.abs(rows[:, 1] - 1))[:24]]
    assert all(any(numpy.allclose([p['gh'], p['rh'], p['rho0']], row[2:5])
                   for row in nearest) for p in evaluated)


def test_nearest_q_skips_completed_projected_duplicates(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch)
    rows = numpy.array([[90, 0.99, 0.4, 1.5, 70, 0.6, 0.1],
                        [90, 1.01, 0.4, 1.5, 70, 0.6, 0.2],
                        [90, 0.98, 0.8, 2.0, 80, 0.6, 0.3],
                        [90, 1.02, 0.8, 2.0, 80, 0.6, 0.4]])
    numpy.savetxt(tmp_path / 'prior.txt', rows)
    known = rows[:1].copy()
    known[0, 1] = 1
    known[0, 6] = 107
    data, _ = ns['seed_q1_from_nearest_q'](
        known, [], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert len(evaluated) == 1
    assert evaluated[0]['gh'] == 0.8
    assert len(data) == 2


@pytest.mark.parametrize('failure', ['invalid', 'busy', 'exception', 'orblib_busy'])
def test_nearest_q_failures_terminate_and_release_claims(tmp_path, monkeypatch, failure):
    ns, evaluated, reserved, released, _ = seed_namespace(tmp_path, monkeypatch, outcome=1e6)
    write_seed_history(tmp_path / 'prior.txt')
    if failure == 'busy':
        ns['_try_reserve_candidate'] = lambda *args: (None, True)
    elif failure in ('exception', 'orblib_busy'):
        def fail(*args, **kwargs):
            error = ns['OrblibBusyError'] if failure == 'orblib_busy' else RuntimeError
            raise error('mock failed solve')
        ns['halo_IC_lib_weights_pca_fixed'] = fail
    data, results = ns['seed_q1_from_nearest_q'](
        None, [], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert len(data) == 0
    assert not results
    assert len(reserved) == (0 if failure == 'busy' else 10)
    assert len(released) == len(reserved)


def test_nearest_q_prefers_current_incl_and_keeps_sources_separate(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch)
    write_seed_history(tmp_path / 'out.txt', incl=80)
    rows = write_seed_history(tmp_path / 'prior.txt', incl=90)
    rows[:, 2] += 0.6
    rows[:, 6] += 1000
    numpy.savetxt(tmp_path / 'prior.txt', rows)
    ns['seed_q1_from_nearest_q'](
        None, ['out.txt'], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert len(evaluated) == 10
    assert all(p['gh'] >= 0.7 for p in evaluated)


def test_nearest_q_parallel_results_end_seeding_early(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch)
    write_seed_history(tmp_path / 'prior.txt')
    shared = numpy.array([[90, 1, 0.8, 2, 80, 0.6, 150]] * 10)
    ns['_periodic_bootstrap_sync'] = lambda n, *args: shared if n % 4 == 0 else None
    data, _ = ns['seed_q1_from_nearest_q'](
        None, [], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert len(evaluated) == 4
    numpy.testing.assert_array_equal(data, shared)


def test_nearest_q_accepts_no_sources_or_existing_good_history(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch)
    empty, results = ns['seed_q1_from_nearest_q'](
        None, [], [], [], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert empty.shape == (0, 7)
    assert not results
    write_seed_history(tmp_path / 'prior.txt')
    good = numpy.array([[90, 1, 0.8, 2, 80, 0.6, 5]] * 10)
    data, results = ns['seed_q1_from_nearest_q'](
        good, [], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    numpy.testing.assert_array_equal(data, good)
    assert not evaluated and not results


@pytest.mark.parametrize('initial_penalty,prior_available,resume,q1,expected_evaluations', [
    (None, True, False, True, 10),
    (107.0, True, False, True, 10),
    (5.0, True, False, True, 0),
    (None, False, False, True, 10),
    (107.0, True, True, True, 0),
    (107.0, True, False, False, 0),
])
def test_initialization_reaches_turbo_without_low_penalties(
        tmp_path, monkeypatch, initial_penalty, prior_available, resume, q1, expected_evaluations):
    class ReadyForTurbo(BaseException):
        pass

    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch, extra_functions=(
        'run_pca_optimization', 'WeightedScaler', 'adaptive_penalty_cutoff',
        'load_fresh_data_from_files', '_generate_random_initial_points'))
    if prior_available:
        write_seed_history(tmp_path / 'prior.txt')
    if initial_penalty is not None:
        rows = write_seed_history(tmp_path / 'out_previous.txt', count=10)
        rows[:, 1] = 1.0
        rows[:, 6] = initial_penalty
        numpy.savetxt(tmp_path / 'out_previous.txt', rows)
    if resume:
        Path(f"checkpoint_{ns['hostname_proc']}.pkl").touch()

    def turbo(**kwargs):
        data = kwargs['model_data']['data_good']
        assert len(data) >= 10
        assert numpy.all(data[:, 1] == 1)
        assert numpy.all(data[:, 6] >= (initial_penalty or 107))
        assert numpy.isfinite(kwargs['model_data']['pca'].components_).all()
        assert len(evaluated) == expected_evaluations
        raise ReadyForTurbo

    ns.update(Q1=q1, PCA=NumpyPCA, TuRBO_PCA_Fixed=turbo,
              torch=SimpleNamespace(tensor=ArrayTensor, double='double', device=lambda x: x,
                                    cuda=SimpleNamespace(is_available=lambda: False)),
              send_notification=lambda *a, **kw: None)
    with pytest.raises(ReadyForTurbo):
        ns['run_pca_optimization'](
            storage_patterns=['out_*.txt'], host_patterns=[], prior_patterns=['prior.txt'],
            output_file=str(tmp_path / 'run.log'), resume=resume, reserve_points=False,
            n_components=3, n_iter=0)


def test_q1_run_finishes_without_repeating_prior_or_requiring_penalty_ten(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch, outcome=200.0, extra_functions=(
        'run_pca_optimization', 'WeightedScaler', 'adaptive_penalty_cutoff',
        'load_fresh_data_from_files', 'pca_to_params_fixed'))
    write_seed_history(tmp_path / 'prior.txt')
    rows = write_seed_history(tmp_path / 'out_previous.txt', count=10)
    rows[:, 1] = 1.0
    rows[:, 6] = 107.0
    rows[:, 5] = 0.75
    numpy.savetxt(tmp_path / 'out_previous.txt', rows)
    ns.update(PCA=NumpyPCA, TuRBO_PCA_Fixed=lambda **kwargs: SimpleNamespace(length=0.6),
              torch=SimpleNamespace(tensor=ArrayTensor, double='double', device=lambda x: x,
                                    cuda=SimpleNamespace(is_available=lambda: False)),
              send_notification=lambda *a, **kw: None)
    params, upsilon, penalty = ns['run_pca_optimization'](
        storage_patterns=['out_*.txt'], host_patterns=[], prior_patterns=['prior.txt'],
        output_file=str(tmp_path / 'run.log'), resume=False, reserve_points=False,
        n_components=3, n_iter=0)
    assert len(evaluated) == 10
    assert params['Q'] == 1.0
    assert penalty == 107.0
    assert upsilon == 0.75


def test_nearest_q_rechecks_completion_after_claim(tmp_path, monkeypatch):
    ns, evaluated, _, released, _ = seed_namespace(tmp_path, monkeypatch)
    row = numpy.array([[90, 0.99, 0.4, 1.5, 70, 0.6, 0.1]])
    numpy.savetxt(tmp_path / 'prior.txt', row)

    def reserve(params, *args):
        row[0, 1] = 1
        row[0, 6] = 107
        numpy.savetxt(tmp_path / 'out_finished.txt', row)
        return 'claim', False

    ns['_try_reserve_candidate'] = reserve
    ns['seed_q1_from_nearest_q'](
        None, ['out_*.txt'], [], ['prior.txt'], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert not evaluated
    assert released == ['claim']


def test_nearest_q_bounds_and_archive_are_respected(tmp_path, monkeypatch):
    ns, evaluated, _, _, _ = seed_namespace(tmp_path, monkeypatch)
    rows = numpy.array([[90, 0.99, 0.4, 1.5, 70, 0.6, 0.1],
                        [90, 0.99, -0.1, 1.5, 70, 0.6, 0.1],
                        [90, 0.99, 0.4, 1.5, 150, 0.6, 0.1],
                        [90, 0.99, 0.4, 0, 70, 0.6, 0.1],
                        [90, 0.99, 0.4, 1.5, 70, 0.6, numpy.nan]])
    numpy.savetxt(tmp_path / 'archive.txt', rows)
    ns['seed_q1_from_nearest_q'](None, [], [], [], BOUNDS, output_file=str(tmp_path / 'seed.log'))
    assert not evaluated
    data, _ = ns['seed_q1_from_nearest_q'](
        None, [], [], [], BOUNDS, seed_patterns=['archive.txt'], output_file=str(tmp_path / 'seed.log'))
    assert len(evaluated) == len(data) == 1
    assert data[0, 6] == 107.0


def test_failed_history_seeds_fall_back_to_bounded_lhs(tmp_path, monkeypatch):
    class ReadyForTurbo(BaseException):
        pass

    outcomes = iter([1e6] * 10 + [150.0] * 10)
    ns, evaluated, _, _, _ = seed_namespace(
        tmp_path, monkeypatch, outcome=lambda params: next(outcomes), extra_functions=(
            'run_pca_optimization', 'WeightedScaler', 'adaptive_penalty_cutoff',
            'load_fresh_data_from_files', '_generate_random_initial_points'))
    write_seed_history(tmp_path / 'prior.txt')

    def turbo(**kwargs):
        assert len(evaluated) == 20
        data = kwargs['model_data']['data_good']
        assert len(data) == 10
        assert numpy.all(data[:, 6] == 150)
        raise ReadyForTurbo

    ns.update(PCA=NumpyPCA, TuRBO_PCA_Fixed=turbo,
              torch=SimpleNamespace(tensor=ArrayTensor, double='double', device=lambda x: x,
                                    cuda=SimpleNamespace(is_available=lambda: False)),
              send_notification=lambda *a, **kw: None)
    with pytest.raises(ReadyForTurbo):
        ns['run_pca_optimization'](
            storage_patterns=['out_*.txt'], host_patterns=[], prior_patterns=['prior.txt'],
            output_file=str(tmp_path / 'run.log'), resume=False, reserve_points=False,
            n_components=3, n_iter=0)


@pytest.mark.parametrize('q1', [False, True])
@pytest.mark.parametrize('initial_resume', [False, True])
@pytest.mark.parametrize('before_first', [False, True])
def test_stream_stop_saves_last_completed_evaluation_and_resumes(tmp_path, monkeypatch, q1, initial_resume, before_first):
    class Stop(SystemExit):
        pass

    stopped = [False]

    def check():
        if stopped[0]:
            raise Stop(75)

    def atomic(path, data):
        Path(path).write_bytes(data)

    ns, _, _, _, syncs = seed_namespace(tmp_path, monkeypatch, extra_functions=(
        'run_pca_optimization', 'WeightedScaler', 'adaptive_penalty_cutoff',
        'load_fresh_data_from_files', 'pca_to_params_fixed', 'params_to_pca_fixed',
        'save_checkpoint', 'q1_checkpoint_coords', 'save_initial_checkpoint', 'check_storage_stop'))
    rows = write_seed_history(tmp_path / 'out_previous.txt', count=10)
    rows[:, 1] = 1.0 if q1 else 0.8
    rows[:, 6] = 5.0
    numpy.savetxt(tmp_path / 'out_previous.txt', rows)
    evaluated = []
    constructed = []

    def turbo_factory(**kwargs):
        if before_first and not constructed:
            stopped[0] = True
        constructed.append(True)
        return SimpleNamespace(length=0.6, length_min=0.01, success_count=0, failure_count=0,
                               model_data=kwargs['model_data'], bounds_original=BOUNDS,
                               suggest=lambda *a, **k: ArrayTensor([[0.1, 0.2, 0.3]]),
                               _update_tr=lambda *a: None)

    def evaluate(pc, model, bounds, *args):
        evaluated.append(ns['pca_to_params_fixed'](pc, model, bounds))
        ns.update(best_overall_target=-4.0, best_overall_Upsilon=0.6)
        stopped[0] = True
        return -4.0

    ns.update(Q1=q1, PCA=NumpyPCA, TuRBO_PCA_Fixed=turbo_factory,
              torch=SimpleNamespace(tensor=ArrayTensor, double='double', device=lambda x: x,
                                    cat=lambda tensors, dim=0: ArrayTensor(numpy.concatenate([t.values for t in tensors], axis=dim)),
                                    cuda=SimpleNamespace(is_available=lambda: False)),
              orblib_store=SimpleNamespace(check_stop=check, stopped=lambda: stopped[0], control=tmp_path),
              atomic_bytes=atomic, EVALUATION_CONTEXT='same', _ups_recent=[], do_resume=False,
              UpsFile=str(tmp_path / 'out_current.txt'),
              read_active_reservations=lambda *a, **k: [],
              halo_IC_lib_weights_pca_fixed=evaluate, send_notification=lambda *a, **k: None)
    kwargs = dict(storage_patterns=['out_*.txt'], host_patterns=[], prior_patterns=[],
                  output_file=str(tmp_path / 'run.log'), reserve_points=False, n_components=3)
    if initial_resume:
        ns['save_initial_checkpoint']()
        ns['do_resume'] = True
    with pytest.raises(Stop):
        ns['run_pca_optimization'](**kwargs, resume=initial_resume, n_iter=5)
    ns['checkpoint_for_stop']()
    checkpoint = next(tmp_path.glob('checkpoint_*.pkl'))
    state = pickle.loads(checkpoint.read_bytes())
    expected_count = 0 if before_first else 1
    assert state['iteration'] == expected_count
    assert state['phase'] == 'turbo'
    if not before_first:
        assert state['params_obs'][-1] == evaluated[0]
    assert state['Y_obs'][-1, 0] == (-5.0 if before_first else -4.0)
    assert state['best_Upsilon'] == 0.6
    assert len(evaluated) == expected_count
    assert not syncs
    stopped[0] = False
    ns['run_pca_optimization'](**kwargs, resume=True, n_iter=expected_count)
    assert len(evaluated) == expected_count


def test_completed_points_use_compatible_context_and_keep_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'out_history.txt'
    content = '# storage-context other\n90 1 0.4 7 10 0.6 1\n'
    path.write_text(content)
    ns = functions('completed_point', orblib_store=object(), storage_patterns=['out_*.txt'],
                   host_patterns=[], EVALUATION_CONTEXT='current')
    params = dict(Q=1.0, gh=0.4, rh=7.0, rho0=10.0)
    assert not ns['completed_point'](params)
    path.write_text(content.replace('other', 'current'))
    assert ns['completed_point'](params)
    assert not ns['completed_point'](dict(params, Q=0.8))
    assert path.read_text() == content.replace('other', 'current')


def test_initial_checkpoint_is_not_a_turbo_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = tmp_path / 'out_test.txt'
    history.write_text('90 1 0.4 7 10 0.6 5\n')
    ns = functions('save_initial_checkpoint', UpsFile=str(history), do_resume=False,
                   proc_rng=numpy.random.default_rng(7), EVALUATION_CONTEXT='test',
                   atomic_bytes=lambda path, data: Path(path).write_bytes(data))
    ns['save_initial_checkpoint']()
    state = pickle.loads(next(tmp_path.glob('checkpoint_*.pkl')).read_bytes())
    assert state['phase'] == 'initial'
    assert state['history_rows'] == [history.read_text()]
    assert 'X_obs' not in state
    assert 'rng_state' in state


def test_launcher_help_is_side_effect_free():
    result = subprocess.run(['bash', str(LAUNCHER), '--help'],
                            text=True, capture_output=True, check=True)
    assert '--Q1' in result.stdout
    assert '--no-shutdown' in result.stdout
