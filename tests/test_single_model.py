"""Single-model check: run_single_model.py and launch_single_model.sh.

No AGAMA, docker or cloud: the runner is driven against the real objective
function of the experimental script (extracted through ast) with fake AGAMA
objects, and the orchestrator against stubbed external commands.
"""
import argparse
import ast
import contextlib
import datetime
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile
from types import SimpleNamespace

import numpy
import pytest
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
RUNNER_PATH = ROOT / 'py/run_single_model.py'
LAUNCHER = ROOT / 'py/launch_single_model.sh'
TREE = ast.parse(SCRIPT.read_text())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('run_single_model', RUNNER_PATH)
storage = load('orblib_storage', ROOT / 'py/orblib_storage.py')


class ModuleView:
    """Attribute view of an exec namespace, so the runner patches the very
    globals the extracted functions read."""

    def __init__(self, namespace):
        object.__setattr__(self, '_ns', namespace)

    def __getattr__(self, name):
        try:
            return self._ns[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self._ns[name] = value


HISTORY_ROW = '90.000 0.3 0.1 2.5 80.0 0.6 1.30 2026-06-16 19:18:09.9\n'


def write_history(directory):
    """Synthetic production history (made-up numbers) for the default reference."""
    (directory / '4UpsBoTorch_PCA_Sersic_test_p0.txt').write_text('# Server: test\n' + HISTORY_ROW)


def args_for(tmp_path, **overrides):
    argv = ['--suffix', 'singleTr0', '--orblib-dir', str(tmp_path / 'orblib_single/T_r0'),
            '--report', str(tmp_path / 'report.json')]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return runner.parse_args(argv)


# --------------------------------------------------------------------------
# Reference rows
# --------------------------------------------------------------------------
def test_reference_rows_in_all_layouts(tmp_path):
    (tmp_path / '4Ups_a.txt').write_text(
        '# header\n'
        '90.000 0.3 0.0 2.4 90.0 0.65 1.30 2026-06-16 19:18:09.9\n'
        '90.000 0.3 0.0 2.4 90.0 0.65 -1.0 2026-06-16 19:18:09.9\n'      # negative penalty
        '90.000 0.3 0.0 2.4 90.0 0.65 nan 2026-06-16 19:18:09.9\n'       # non-finite
        '85.000 0.3 0.0 2.4 90.0 0.65 0.50 2026-06-16 19:18:09.9\n'      # other incl
        'Error with parameters\n')
    (tmp_path / 'J_factor_x_theta0.5.txt').write_text(
        '90.0 0.27 0.0 2.41 90.4 0.66 59.6 1.25 3e18 18.5\n')
    (tmp_path / 'Jcomputed_from_raw_theta0.5.txt').write_text(
        '90.0 0.26 0.0 2.42 91.0 0.64 58.0 1.24 3e18 18.6 2\n'
        '90.0 0.26 0.0 2.42 91.0 0.0 58.0 1.00 3e18 18.6 2\n')           # Upsilon 0
    best = runner.best_reference(glob.glob(str(tmp_path / '*.txt')), 90.0)
    assert best['penalty'] == 1.24 and best['upsilon'] == 0.64 and best['Q'] == 0.26
    assert best['source'].startswith('Jcomputed_from_raw_theta0.5.txt:1')
    assert runner.best_reference([str(tmp_path / '4Ups_a.txt')], 85.0)['penalty'] == 0.5
    assert runner.best_reference([str(tmp_path / '4Ups_a.txt')], 70.0) is None


def test_default_reference_comes_from_the_production_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):                       # no history, no explicit point
        runner.resolve_reference(args_for(tmp_path))
    write_history(tmp_path)
    (tmp_path / '4UpsBoTorch_PCA_Sersic_other_p3.txt').write_text(
        '90.000 0.5 0.2 3.0 70.0 0.7 1.90 2026-06-17 10:00:00.0\n'
        '85.000 0.5 0.2 3.0 70.0 0.7 0.90 2026-06-17 10:00:00.0\n')
    ref = runner.resolve_reference(args_for(tmp_path))
    assert (ref['Q'], ref['penalty'], ref['upsilon']) == (0.3, 1.3, 0.6)
    assert ref['source'] == '4UpsBoTorch_PCA_Sersic_test_p0.txt:2'
    partial = runner.resolve_reference(args_for(tmp_path, rh=3.0, ref_penalty=2.0))
    assert partial['rh'] == 3.0 and partial['penalty'] == 2.0 and partial['Q'] == 0.3
    assert partial['source'].endswith('(overridden: rh)')
    explicit = runner.resolve_reference(args_for(tmp_path, Q=1.0, gh=0.1, rh=2.0, rho0=50.0))
    assert explicit['source'] == 'command line' and explicit['penalty'] is None
    assert runner.resolve_reference(args_for(tmp_path, incl=85.0))['penalty'] == 0.9


# --------------------------------------------------------------------------
# Configuration handed to the experimental script
# --------------------------------------------------------------------------
def script_configuration(monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', argv)
    start = next(i for i, node in enumerate(TREE.body)
                 if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'parser')
    end = next(i for i, node in enumerate(TREE.body)
               if isinstance(node, ast.Import) and node.names[0].name == 'agama')
    namespace = dict(argparse=argparse, numpy=numpy, os=os, socket=socket, hashlib=hashlib)
    exec(compile(ast.Module(body=TREE.body[start:end], type_ignores=[]), str(SCRIPT), 'exec'),
         namespace)
    return namespace


def test_runner_gives_the_script_the_free_q_d1_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('HOSTNAME_SUFFIX', 'testhost')
    args = runner.parse_args(['--suffix', 'single20261001_101500r2', '--orblib-dir',
                              'orblib_single/x_r2', '--n_threads', '8'])
    argv = runner.module_argv(args)
    assert '--stream-orblib' not in argv and '--Q1' not in argv and '--no-double' not in argv
    config = script_configuration(monkeypatch, argv)
    assert config['DOUBLE'] is True and config['Q1'] is False and config['N_BIN'] == 250
    assert config['SAVE_ORBLIB'] and config['REUSE_ORBLIB'] and not config['args'].stream_orblib
    assert config['ORBLIB_DIR'] == 'orblib_single/x_r2'
    assert config['EXP_ID'] == 'd1_nb250_gh0_ser0'
    pool = config['UpsFile']
    assert pool == 'out_testhost_d1_nb250_gh0_ser0_single20261001_101500r2.txt'
    patterns = config['storage_patterns'] + config['host_patterns']
    import fnmatch
    assert any(fnmatch.fnmatch(pool, p) for p in config['storage_patterns'])
    side = f"single_{config['hostname_proc']}.txt"
    assert not any(fnmatch.fnmatch(side, p) for p in patterns)
    assert args.report == 'report_single20261001_101500r2.json'


def test_protocol_list_validation(tmp_path):
    assert args_for(tmp_path).protocols == ['exp', 'reuse', 'prod']
    for bad in ('reuse,exp', 'exp,exp', 'exp,foo', ''):
        with pytest.raises(SystemExit):
            args_for(tmp_path, protocols=bad)
    with pytest.raises(SystemExit):
        runner.parse_args(['--suffix', 'x'])


def test_every_name_the_runner_relies_on_exists_in_the_script():
    defined = set()
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    needed = {'orblib_store', 'completed_point', 'checkpoint_for_stop', 'save_initial_checkpoint',
              'UpsFile', 'UPS_XATOL', 'UPS_SUBSAMPLE_FRAC', '_ups_recent', 'proc_rng',
              '_params_to_dummy_pc', 'halo_IC_lib_weights_pca_fixed', 'bounds_original',
              'densityStars', 'datasets', 'alphah', 'betah', 'hostname_proc', 'gridx_max',
              'gridy_max', 'bound_circR', 'gridx_min', 'gridy_min', 'q_ap', 'EXP_ID', 'GEOM_HASH',
              'EVALUATION_CONTEXT', 'sectAPP', 'DOUBLE', 'N_BIN', 'ORBLIB_DIR', 'orblib_key'}
    assert needed <= defined, sorted(needed - defined)
    halo = next(n for n in TREE.body if isinstance(n, ast.FunctionDef)
                and n.name == 'halo_IC_lib_weights_pca_fixed')
    declared = {name for n in ast.walk(halo) if isinstance(n, ast.Global) for name in n.names}
    assert {'best_overall_target', 'best_overall_Upsilon', 'number_of_h_IC_lw',
            'number_of_find_w_U', 'UpsFile', '_ups_recent'} <= declared
    flags = {n.args[0].value for n in ast.walk(TREE) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'parser.add_argument'}
    assert {'--incl', '--suffix', '--save-orblib', '--reuse-orblib', '--orblib-dir',
            '--no-resume', '--n_threads'} <= flags


def test_prod_formula_grid_comparison():
    mod = SimpleNamespace(gridx_max=2.0 + 0.01, gridy_max=1.38 + 0.02, bound_circR=[[0.1], [2.0], [0], [2.0]],
                          gridx_min=0.01, gridy_min=0.02, q_ap=0.69, EXP_ID='d1', hostname_proc='h',
                          GEOM_HASH='g', EVALUATION_CONTEXT='c', sectAPP=[0] * 25, DOUBLE=True, N_BIN=250,
                          datasets=[SimpleNamespace(cons_err=numpy.array([1.0, 0.0])),
                                    SimpleNamespace(cons_err=numpy.array([1.0, 2.0]))])
    context, grid = runner.static_context(mod)
    assert grid['identical'] and context['num_dof'] == 3 and context['n_apertures'] == 25
    mod.gridx_max = 2.2
    assert not runner.static_context(mod)[1]['identical']


# --------------------------------------------------------------------------
# End to end against the real objective function (fake AGAMA)
# --------------------------------------------------------------------------
N_ORBITS = 8000   # sub-sample = max(1000, 25 %) = exactly a quarter


def real_halo_module(tmp_path, store):
    names = ['halo_IC_lib_weights_pca_fixed', 'orblib_key', 'write_orblib_npz', 'log_mem',
             'note_archived_rebuild', '_params_to_dummy_pc', 'FunctionLogger', 'OrblibBusyError',
             'save_initial_checkpoint']
    nodes = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    calls = SimpleNamespace(orbit=0, subsamples=[])

    def orbit(**kwargs):
        assert 'trajsize' not in kwargs
        calls.orbit += 1
        rows = numpy.arange(N_ORBITS, dtype=numpy.float32)[:, None]
        return [numpy.hstack([rows, rows + 1]), numpy.hstack([rows * 0.5, rows, rows + 2])]

    class Dataset:
        cons_val = numpy.ones(4)
        cons_err = numpy.ones(4)
        target = [0, 1, 2]

        def getOrbitMatrix(self, matrix, upsilon):
            calls.subsamples.append(len(matrix))
            return matrix

        def getPenalty(self, superposition, upsilon):
            return (float(upsilon) - 0.62) ** 2 * 10 + 1.25

    ns = dict(numpy=numpy, os=os, time=time, datetime=datetime, contextlib=contextlib,
              zipfile=zipfile, hashlib=hashlib, minimize_scalar=minimize_scalar,
              HashingWriter=storage.HashingWriter, StorageStop=storage.StorageStop,
              LibraryBusy=storage.LibraryBusy, _NPZ_BLOCK_BYTES=4096, TRAJSIZE_STORED=0,
              Q1=False, DOUBLE=True, N_BIN=250, SER_ID=0, GEOM_HASH='abcdef12', incl=90.0,
              ORBLIB_DIR=str(tmp_path / 'orblib_single/T_r0'), SAVE_ORBLIB=True, REUSE_ORBLIB=True,
              hostname_proc='testhost_d1_nb250_gh0_ser0_singleTr0',
              UpsFile='out_testhost_d1_nb250_gh0_ser0_singleTr0.txt',
              EXP_ID='d1_nb250_gh0_ser0', EVALUATION_CONTEXT='ctx',
              _ORBLIB_BUILD_TTL_SEC=7200, _claim_file=lambda *a: True,
              release_reservation=lambda *a: None, orblib_counter=0, orblib_archived_rebuilds=0,
              gridv=numpy.linspace(-25, 25, 51), degree=2, ghorder=6, AGAMA_ORBIT_THREADS=8,
              UPS_XATOL=5e-3, UPS_BRACKET_DELTA=0.1, UPS_BRACKET_NMED=8, UPS_SUBSAMPLE_FRAC=0.25,
              _ups_recent=[0.3], proc_rng=numpy.random.default_rng(1),
              completed_point=lambda params: pytest.fail('completed_point must be neutralised'),
              bounds_original=dict(Q=(0.05, 2.5), gh=(0.0, 1.6), rh=(0.5, 7.0), rho0=(10.0, 120.0)),
              densityStars=SimpleNamespace(sample=lambda n, potential: [numpy.zeros((n, 6))]),
              datasets=[Dataset(), Dataset()], alphah=2.0, betah=3,
              agama=SimpleNamespace(
                  Density=lambda *a, **kw: None, orbit=orbit,
                  setNumThreads=lambda n: contextlib.nullcontext(),
                  Potential=lambda **kw: SimpleNamespace(Tcirc=lambda ic: numpy.ones(len(ic))),
                  solveOpt=lambda matrix, **kw: numpy.ones(matrix[0].shape[1])))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SCRIPT), 'exec'), ns)
    ns['checkpoint_for_stop'] = ns['save_initial_checkpoint']
    # The objective's default numOrbits is 100000; keep the test small.
    halo = ns['halo_IC_lib_weights_pca_fixed']
    ns['halo_IC_lib_weights_pca_fixed'] = lambda *a, **kw: halo(*a, numOrbits=N_ORBITS, **kw)
    mod = ModuleView(ns)
    runner.prepare_module(mod, store)
    return mod, calls


def test_protocols_on_one_integration_with_the_real_objective(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_history(tmp_path)
    args = args_for(tmp_path)
    store = storage.Store(args.orblib_dir, 0, save_slots=2)
    mod, calls = real_halo_module(tmp_path, store)
    report = runner.base_report(args, runner.resolve_reference(args))
    name, path = mod.orblib_key(*(report['params'][k] for k in ('Q', 'gh', 'rh', 'rho0')))
    report['orblib'] = dict(name=name, path=path)
    code = runner.execute(mod, args, report, store, numpy.random.default_rng, storage)
    assert code == 0, report
    assert calls.orbit == 1                                 # one integration for three protocols
    saved = json.loads(Path(args.report).read_text())
    assert saved['status'] == 'ok'
    exp, reuse, prod = (saved['protocols'][p] for p in ('exp', 'reuse', 'prod'))
    assert exp['saved'] and not exp['reused'] and reuse['reused'] and prod['reused']
    assert exp['settings'] == dict(xatol=5e-3, subsample=0.25) == reuse['settings']
    assert prod['settings'] == dict(xatol=1e-3, subsample=0.0)
    for result in (exp, reuse, prod):
        assert result['penalty'] == pytest.approx(1.25, abs=1e-3)
        assert result['upsilon'] == pytest.approx(0.62, abs=0.01)
    assert saved['deltas']['reuse_minus_exp'] == pytest.approx(0.0, abs=1e-12)
    assert set(saved['deltas']) >= {'exp_minus_ref', 'prod_minus_ref', 'prod_minus_exp'}
    # Cleared before every protocol (the stale 0.3 is gone), then one Upsilon* appended.
    assert len(mod._ups_recent) == 1 and mod._ups_recent[0] == pytest.approx(0.62, abs=0.01)
    # exp and reuse use the 25 % sub-sample, prod the full library.
    assert N_ORBITS // 4 in calls.subsamples and N_ORBITS in calls.subsamples
    pool = (tmp_path / mod.UpsFile).read_text()
    side = (tmp_path / 'single_testhost_d1_nb250_gh0_ser0_singleTr0.txt').read_text()
    data = lambda text: [l for l in text.splitlines() if l and not l.startswith('#')]
    assert len(data(pool)) == 1 and '# orblib saved:' in pool and '# storage-context ctx' in pool
    assert len(data(side)) == 2 and side.count('# orblib reused:') == 2
    library = saved['orblib']
    assert library['metadata_ok'] and library['store_consistent'] and library['store_state'] == 'ready'
    assert library['metadata']['trajsize'] == 0 and library['metadata']['numOrbits'] == N_ORBITS
    assert not list(tmp_path.glob('checkpoint_*'))          # no search checkpoint written
    assert mod.UPS_XATOL == 5e-3 and mod.UPS_SUBSAMPLE_FRAC == 0.25 and mod.UpsFile.startswith('out_')
    assert (tmp_path / 'report.txt').read_text().startswith('single-model check singleTr0 status=ok')


# --------------------------------------------------------------------------
# Driver failure paths (stub objective)
# --------------------------------------------------------------------------
def stub_module(tmp_path, behaviour):
    ns = dict(UPS_XATOL=5e-3, UPS_SUBSAMPLE_FRAC=0.25, UpsFile='out_h_d1_nb250_gh0_ser0_s.txt',
              _ups_recent=[0.5], proc_rng=None, hostname_proc='h_d1_nb250_gh0_ser0_s',
              bounds_original={}, densityStars=None, datasets=None, alphah=2, betah=3,
              _params_to_dummy_pc=lambda params, model, bounds: numpy.zeros(4))
    mod = ModuleView(ns)
    seen = []

    def halo(pc, model, bounds, *rest, direct_params=None):
        assert pc is not None and model is None
        seen.append(dict(file=mod.UpsFile, xatol=mod.UPS_XATOL, sub=mod.UPS_SUBSAMPLE_FRAC,
                         recent=list(mod._ups_recent), rng=mod.proc_rng))
        direct_params['Q'] = -1                       # the objective clips in place
        return behaviour(len(seen), mod)

    ns['halo_IC_lib_weights_pca_fixed'] = halo
    return mod, seen


def write_block(mod, penalty, reused):
    with open(mod.UpsFile, 'a') as stream:
        stream.write(f'# Server: {mod.hostname_proc}\n90.000 0.2 0 2 90 0.6 {penalty} 2026-10-01 10:00:00\n'
                     f"# orbitlib times (s): sample_s=1.0 orbit_s=2.0 total_s=3.0 (numOrbits=10\n"
                     + (f'# orblib reused: lib.npz load_s=0.5\n' if reused else
                        f'# orblib saved: lib.npz size_MB=1.000 save_s=0.300\n')
                     + '# End of history\n\n')
    mod.number_of_find_w_U += 3
    return -penalty


class FakeStore:
    def __init__(self):
        self.stop = False

    def stopped(self):
        return self.stop

    def row(self, name):
        return None


@pytest.mark.parametrize('scenario', ['ok', 'failed_eval', 'not_reused', 'stop'])
def test_driver_routes_rows_and_reports_failures(tmp_path, monkeypatch, scenario):
    monkeypatch.chdir(tmp_path)
    write_history(tmp_path)
    store = FakeStore()

    def behaviour(call, mod):
        if scenario == 'failed_eval' and call == 2:
            return -1e6
        if scenario == 'stop' and call == 1:
            store.stop = True
        return write_block(mod, 1.2 + call * 0.01, reused=call > 1 and scenario != 'not_reused')

    mod, seen = stub_module(tmp_path, behaviour)
    runner.prepare_module(mod, store)
    args = args_for(tmp_path, subsample_seed=7)
    (tmp_path / 'lib.npz').write_bytes(b'x')
    report = runner.base_report(args, runner.resolve_reference(args))
    report['orblib'] = dict(name='lib.npz', path=str(tmp_path / 'lib.npz'))
    checks = SimpleNamespace(StorageStop=storage.StorageStop, file_hash=lambda path: 'md5',
                             metadata=lambda path, deep: dict(numOrbits=10))
    code = runner.execute(mod, args, report, store, lambda seed: ('rng', seed), checks)
    saved = json.loads(Path(args.report).read_text())
    expected = dict(ok=(0, 'ok', 3), failed_eval=(1, 'failed', 1), not_reused=(1, 'failed', 1),
                    stop=(75, 'stopped', 1))[scenario]
    assert (code, saved['status'], len(saved['protocols'])) == expected
    assert 'exp' in saved['protocols'] and saved['protocols']['exp']['probes'] == 3
    assert seen[0]['file'].startswith('out_') and all(s['file'].startswith('single_') for s in seen[1:])
    assert all(s['recent'] == [] and s['rng'] == ('rng', 7) for s in seen)
    assert [s['xatol'] for s in seen][:3] == [5e-3, 5e-3, 1e-3][:len(seen)]
    assert report['params']['Q'] == 0.3                       # never clipped through
    assert mod.UpsFile.startswith('out_') and mod.UPS_XATOL == 5e-3
    assert mod.number_of_h_IC_lw == 0 and mod.best_overall_target == -float('inf')


def test_block_parser_reads_the_last_block_of_the_server(tmp_path):
    path = tmp_path / 'out.txt'
    path.write_text('# Server: other\n90 1 0 2 9 0.5 9.9 d t\n# End of history\n'
                    '# storage-context abc\n# Server: me\n90 1 0 2 9 0.5 1.5 d t\n'
                    '# orblib reused: orblib_x.npz load_s=2.500\n# End of history\n'
                    '# Server: me\n90 1 0 2 9 0.7 1.4 d t\n# solveOpt summary: n=5 total_s=10.0\n'
                    '# orblib saved: orblib_x.npz size_MB=590.000 save_s=40.000\n# End of history\n')
    block = runner.parse_last_block(path, 'me')
    assert block['row'][5:7] == [0.7, 1.4]
    assert block['saved'] == dict(name='orblib_x.npz', size_MB=590.0, save_s=40.0)
    assert block['solveopt']['total_s'] == 10.0 and 'reused' not in block
    assert runner.parse_last_block(path, 'nobody') is None
    assert runner.parse_last_block(tmp_path / 'missing.txt', 'me') is None


# --------------------------------------------------------------------------
# Host-side summary
# --------------------------------------------------------------------------
def fake_report(path, suffix, exp, prod, peak, status='ok'):
    report = dict(status=status, suffix=suffix, reference=dict(penalty=1.25),
                  grid=dict(identical=True), orblib=dict(name='orblib_x.npz'),
                  protocols=dict(exp=dict(penalty=exp, peak_rss_mb=peak, save_s=30.0),
                                 reuse=dict(penalty=exp, peak_rss_mb=peak),
                                 prod=dict(penalty=prod, peak_rss_mb=peak + 500)),
                  deltas=dict(reuse_minus_exp=0.0, prod_minus_exp=prod - exp, prod_minus_ref=prod - 1.25))
    path.write_text(json.dumps(report))
    return path


def test_summary_statistics_and_verdict(tmp_path):
    paths = [fake_report(tmp_path / f'r{i}.json', f'r{i}', 1.26 + 0.01 * i, 1.25 + 0.01 * i, 1500)
             for i in range(4)]
    reports, summary = runner.summarize(paths)
    assert summary['prod']['mean'] == pytest.approx(1.265)
    assert summary['prod_minus_ref_in_scatter'] == pytest.approx(0.015 / summary['prod']['std'])
    assert summary['peak_rss_mb']['max'] == 2000 and summary['memory_verdict'] == 'within expected'
    assert summary['grid_identical'] == [True] and summary['orblib_names'] == ['orblib_x.npz']
    assert 'r3: status=ok exp=1.290000' in runner.format_summary(reports, summary)
    fake_report(tmp_path / 'r9.json', 'r9', 1.3, 1.3, 3000, status='failed')
    assert runner.main(['--summarize', *map(str, paths)]) == 0
    assert runner.main(['--summarize', str(tmp_path / 'r9.json'),
                        '--summary-json', str(tmp_path / 's.json')]) == 1
    assert 'above expected' in json.loads((tmp_path / 's.json').read_text())['memory_verdict']


def test_summary_needs_only_the_standard_library(tmp_path):
    path = fake_report(tmp_path / 'r0.json', 'r0', 1.26, 1.25, 1500)
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import run_single_model as r; '
            'r.main(["--summarize", sys.argv[2]]); print("numpy" in sys.modules)')
    result = subprocess.run([sys.executable, '-c', code, str(ROOT / 'py'), str(path)],
                            capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('False')


# --------------------------------------------------------------------------
# Orchestrator with stubbed external commands
# --------------------------------------------------------------------------
PREFIX = r'''
hostname() { printf 'testhost'; }
nproc() { printf '8'; }
curl() { printf 'notify %s\n' "$*" >> "$HOME/events"; return 0; }
swapon() { return 0; }
sudo() {
    if [ "$1" = shutdown ]; then printf 'shutdown %s\n' "$3" >> "$HOME/events"; return "${SHUTDOWN_RC:-0}"; fi
    return 0
}
rclone() {
    printf 'rclone %s\n' "$*" >> "$HOME/events"
    if [ "$1" = lsf ] && [ -n "${RECEIPT_EXISTS:-}" ]; then printf '%s\n' "$RECEIPT_EXISTS"; fi
    return 0
}
python3() {
    case "$1" in
        */orblib_storage.py) printf 'storage %s\n' "$*" >> "$HOME/events"; return 0 ;;
    esac
    command python3 "$@"
}
docker() {
    case "$1" in
        image|info|stats|kill) return 0 ;;
        stop) printf 'docker_stop %s\n' "$*" >> "$HOME/events"; touch "$HOME/stopped"; return 0 ;;
    esac
    printf 'run %s\n' "$*" >> "$HOME/events"
    local prev='' sfx='' dir='' report='' side=''
    for a in "$@"; do
        case "$prev" in
            --suffix) sfx="$a" ;; --orblib-dir) dir="${a#/workspace/}" ;;
            --report) report="${a#/workspace/}" ;; --side-file) side="$a" ;;
        esac
        prev="$a"
    done
    if [ -n "${HANG:-}" ]; then
        while [ ! -f "$HOME/stopped" ]; do sleep 0.05; done
        return 143
    fi
    mkdir -p "$WORK_DIR/$dir"
    printf 'lib\n' > "$WORK_DIR/$dir/orblib_i90.0_d1_nb250_ser0_geomabcdef12_0123456789.npz"
    printf '90.000 0.267 0 2.41 90.4 0.66 1.25 d t\n' > "$WORK_DIR/out_testhost_d1_nb250_gh0_ser0_${sfx}.txt"
    printf '90.000 0.267 0 2.41 90.4 0.66 1.26 d t\n' > "$WORK_DIR/$side"
    printf '{"status": "%s", "suffix": "%s", "reference": {"penalty": 1.24}, "grid": {"identical": true}, "orblib": {"name": "orblib_i90.0_d1_nb250_ser0_geomabcdef12_0123456789.npz"}, "protocols": {"exp": {"penalty": 1.25, "peak_rss_mb": 1500}}, "deltas": {}}\n' \
        "${RUN_STATUS:-ok}" "$sfx" > "$WORK_DIR/$report"
    [ "${RUN_STATUS:-ok}" = ok ] && return 0 || return 1
}
source "$0" "$@"
'''


def launcher_workdir(tmp_path, avail_mb=8000, history=True):
    launcher = tmp_path / LAUNCHER.name
    launcher.write_text(LAUNCHER.read_text())
    (tmp_path / 'run_single_model.py').write_text(RUNNER_PATH.read_text())
    for name in (SCRIPT.name, 'table3.dat', 'orblib_storage.py'):
        (tmp_path / name).touch()
    if history:
        write_history(tmp_path)
    config = tmp_path / '.config/rclone'
    config.mkdir(parents=True)
    (config / 'rclone.conf').touch()
    (tmp_path / 'meminfo').write_text(
        f'MemTotal:       32000000 kB\nMemAvailable:   {avail_mb * 1024} kB\n')
    (tmp_path / 'psi').write_text('some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n')
    return launcher


def run_launcher(tmp_path, *args, avail_mb=8000, history=True, **env):
    launcher = launcher_workdir(tmp_path, avail_mb, history)
    environment = dict(os.environ, HOME=str(tmp_path), ORBLIB_SWAPFILE='0',
                       ORBLIB_MEMINFO=str(tmp_path / 'meminfo'), ORBLIB_PSI_PATH=str(tmp_path / 'psi'),
                       ORBLIB_WATCH_INTERVAL='1', ORBLIB_STOP_GRACE='1', ORBLIB_MONITOR_INTERVAL='60',
                       SINGLE_BYTES_PER_REPEAT='0', ORBLIB_RESERVE_BYTES='0', **env)
    result = subprocess.run(['bash', '-c', PREFIX, str(launcher), *args], cwd=tmp_path, text=True,
                            capture_output=True, timeout=90, env=environment)
    events_file = tmp_path / 'events'
    events = events_file.read_text().splitlines() if events_file.exists() else []
    return result, events


def test_launcher_runs_four_isolated_realisations_and_delivers_r0(tmp_path):
    result, events = run_launcher(tmp_path)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    runs = [e.split() for e in events if e.startswith('run ')]
    assert len(runs) == 4
    suffixes = {r[r.index('--suffix') + 1] for r in runs}
    dirs = {r[r.index('--orblib-dir') + 1] for r in runs}
    assert len(suffixes) == 4 and len(dirs) == 4
    for run in runs:
        limit = next(p for p in run if p.startswith('--memory='))
        assert f"--memory-swap={limit.split('=')[1]}" in run
        assert not any('rclone' in p for p in run)
        assert '--stream-orblib' not in run and '--Q1' not in run and '--preflight' not in run
    uploads = [e for e in events if e.startswith('rclone copyto')]
    pool = [e for e in uploads if ' yandex:galAgama/out_testhost_d1_nb250_gh0_ser0_single' in e]
    assert len(pool) == 4
    assert any('galAgama/single_model/single_d1_nb250_gh0_ser0_i90.0_' in e and 'summary_single_' in e
               for e in uploads)
    storage_calls = [e for e in events if e.startswith('storage ')]
    assert len(storage_calls) == 2
    assert ' prepare --resume ' in storage_calls[0] and ' check ' in storage_calls[1]
    for call in storage_calls:
        root = call.split('--root ')[1].split()[0]
        assert root.endswith('_r0') and '/orblib_single/' in root
        assert '--remote yandex:galAgama/orblib ' in call
    shutdown = events.index(next(e for e in events if e.startswith('shutdown')))
    assert shutdown > max(i for i, e in enumerate(events) if e.startswith('rclone copyto'))
    assert events[shutdown] == 'shutdown +1'
    for i in (1, 2, 3):
        assert list(tmp_path.glob(f'orblib_single/*_r{i}/*.npz'))   # r1..r3 stay on the VM
    log = next(tmp_path.glob('launch_single_*.log')).read_text()
    assert 'single-model summary: 4 reports' in log and 'Оставлена на VM' in log


def test_launcher_keeps_library_when_name_is_already_archived(tmp_path):
    result, events = run_launcher(
        tmp_path, '--repeats=1',
        RECEIPT_EXISTS='orblib_i90.0_d1_nb250_ser0_geomabcdef12_0123456789.npz.json')
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert not any(e.startswith('storage ') for e in events)
    assert list(tmp_path.glob('orblib_single/*_r0/*.npz'))
    assert 'уже в каталоге' in next(tmp_path.glob('launch_single_*.log')).read_text()


def test_launcher_no_upload_no_shutdown_and_failure_code(tmp_path):
    result, events = run_launcher(tmp_path, '--repeats=2', '--no-upload', '--no-shutdown',
                                  RUN_STATUS='failed')
    assert result.returncode == 1, result.stdout[-4000:] + result.stderr[-4000:]
    assert not any(e.startswith('storage ') or e.startswith('shutdown') for e in events)
    assert sum(e.startswith('run ') for e in events) == 2


def test_launcher_preflight_is_one_container_without_side_effects(tmp_path):
    result, events = run_launcher(tmp_path, '--preflight', '--rh=3.0')
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    runs = [e.split() for e in events if e.startswith('run ')]
    assert len(runs) == 1 and '--preflight' in runs[0] and '--rh=3.0' in runs[0]
    assert not any(e.startswith(('storage ', 'shutdown')) for e in events)
    assert not any(' yandex:galAgama/out_' in e for e in events)


def test_launcher_reports_failed_shutdown(tmp_path):
    result, events = run_launcher(tmp_path, '--repeats=1', SHUTDOWN_RC='1')
    assert result.returncode == 0
    assert 'НЕ УДАЛОСЬ запланировать выключение' in next(tmp_path.glob('launch_single_*.log')).read_text()
    assert any(e.startswith('notify ') and 'Priority: urgent' in e and 'shutdown' in e for e in events)


def test_launcher_watchdog_stops_containers_and_uploads_before_shutdown(tmp_path):
    result, events = run_launcher(tmp_path, '--repeats=2', avail_mb=500,
                                  ORBLIB_MIN_AVAIL_MB='2048', HANG='1')
    assert result.returncode == 75, result.stdout[-4000:] + result.stderr[-4000:]
    stop = events.index(next(e for e in events if e.startswith('docker_stop')))
    emergency = max(i for i, e in enumerate(events) if 'galaxy_results_emergency' in e)
    assert stop < emergency < events.index('shutdown +1')
    assert not any(e.startswith('storage ') for e in events)
    assert 'СРАБОТАЛ ВАТЧДОГ' in next(tmp_path.glob('launch_single_*.log')).read_text()


def test_launcher_requires_a_parameter_source(tmp_path):
    result, events = run_launcher(tmp_path, '--repeats=1', '--rh=3.0', history=False)
    assert result.returncode == 1
    assert not any(e.startswith('run ') for e in events)
    assert 'нет 4UpsBoTorch_PCA_Sersic_' in next(tmp_path.glob('launch_single_*.log')).read_text()


def test_launcher_accepts_an_explicit_point_without_history(tmp_path):
    result, events = run_launcher(tmp_path, '--repeats=1', '--no-upload', '--no-shutdown',
                                  '--Q=1.0', '--gh=0.1', '--rh=2.0', '--rho0=50.0', history=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    run = next(e for e in events if e.startswith('run ')).split()
    assert {'--Q=1.0', '--gh=0.1', '--rh=2.0', '--rho0=50.0'} <= set(run)


def test_launcher_refuses_while_the_search_launcher_runs(tmp_path):
    (tmp_path / 'orblib').mkdir()
    with open(tmp_path / 'orblib/.launcher.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result, events = run_launcher(tmp_path)
    assert result.returncode == 1
    assert not any(e.startswith(('run ', 'shutdown')) for e in events)


def test_launcher_help_and_bad_arguments_are_side_effect_free(tmp_path):
    launcher = tmp_path / LAUNCHER.name
    launcher.write_text(LAUNCHER.read_text())
    before = sorted(p.name for p in tmp_path.iterdir())
    result = subprocess.run(['bash', str(launcher), '--help'], capture_output=True, text=True,
                            timeout=15, cwd=tmp_path, env=dict(os.environ, PATH='/usr/bin:/bin'))
    assert result.returncode == 0 and '--repeats' in result.stdout
    bad = subprocess.run(['bash', str(launcher), '--stream-orblib'], capture_output=True, text=True,
                         timeout=15, cwd=tmp_path)
    assert bad.returncode == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == before


# --------------------------------------------------------------------------
# Several different models in parallel (launch_multi_model.sh)
# --------------------------------------------------------------------------
MULTI_LAUNCHER = ROOT / 'py/launch_multi_model.sh'
# Made-up numbers in the three layouts; the last row carries a label.
MODELS_TEXT = (
    '# models\n'
    '89.5000 0.27 0.0 2.46 92.3 0.637 58.8 1.241 3e18 18.58 1\n'
    '89.5000 0.26 0.0 2.41 90.3 0.660 59.6 1.245 3e18 18.57 1\n'
    '90.000 0.28 0.01 2.50 94.0 0.617 1.250 2026-06-16 19:18:09.9\n'
    '88.5 0.29 0.0 2.30 85.0 0.70 59.5 1.260 3e18 18.50   # src.txt:7\n')


def test_models_file_rows_labels_and_duplicates(tmp_path):
    models = tmp_path / 'models.txt'
    models.write_text(MODELS_TEXT + 'garbage line\n')
    rows = runner.read_models(str(models))
    assert [r['incl'] for r in rows] == [89.5, 89.5, 90.0, 88.5]
    assert [r['penalty'] for r in rows] == [1.241, 1.245, 1.25, 1.26]
    assert rows[0]['source'] == 'models.txt:2' and rows[3]['source'] == 'src.txt:7'
    printed = subprocess.run([sys.executable, str(RUNNER_PATH), '--list-models', str(models)],
                             capture_output=True, text=True, timeout=30, check=True)
    lines = [line.split('\t') for line in printed.stdout.splitlines()]
    assert len(lines) == 4 and all(len(cells) == 8 for cells in lines)
    assert lines[2][:7] == ['90.0', '0.28', '0.01', '2.5', '94.0', '1.25', '0.617']
    models.write_text(MODELS_TEXT + '89.5 0.27 0.0 2.46 92.3 0.6 58.8 1.3 3e18 18.58 1\n')
    with pytest.raises(SystemExit):
        runner.read_models(str(models))
    (tmp_path / 'empty.txt').write_text('# nothing\n')
    assert subprocess.run([sys.executable, str(RUNNER_PATH), '--list-models',
                           str(tmp_path / 'empty.txt')], capture_output=True).returncode == 1


def test_reference_source_label_overrides_command_line(tmp_path):
    ref = runner.resolve_reference(args_for(tmp_path, Q=1.0, gh=0.1, rh=2.0, rho0=50.0,
                                            ref_penalty=1.5, ref_source='J.txt:12'))
    assert ref['source'] == 'J.txt:12' and ref['penalty'] == 1.5


MULTI_PREFIX = PREFIX.replace(
    """printf '90.000 0.267 0 2.41 90.4 0.66 1.26 d t\\n' > "$WORK_DIR/$side\"""",
    """[ -z "$side" ] || exit 9""")


def run_multi(tmp_path, *args, models=MODELS_TEXT, **env):
    tmp_path.mkdir(exist_ok=True)
    launcher = tmp_path / MULTI_LAUNCHER.name
    launcher.write_text(MULTI_LAUNCHER.read_text())
    launcher_workdir(tmp_path, history=False)
    if models is not None:
        (tmp_path / 'models.txt').write_text(models)
    environment = dict(os.environ, HOME=str(tmp_path), ORBLIB_SWAPFILE='0',
                       ORBLIB_MEMINFO=str(tmp_path / 'meminfo'), ORBLIB_PSI_PATH=str(tmp_path / 'psi'),
                       ORBLIB_WATCH_INTERVAL='1', ORBLIB_STOP_GRACE='1', ORBLIB_MONITOR_INTERVAL='60',
                       SINGLE_BYTES_PER_REPEAT='0', ORBLIB_RESERVE_BYTES='0', **env)
    result = subprocess.run(['bash', '-c', MULTI_PREFIX, str(launcher), *args], cwd=tmp_path,
                            text=True, capture_output=True, timeout=90, env=environment)
    events_file = tmp_path / 'events'
    events = events_file.read_text().splitlines() if events_file.exists() else []
    return result, events


def test_multi_launcher_runs_each_model_once_and_delivers_every_library(tmp_path):
    assert MULTI_PREFIX != PREFIX
    result, events = run_multi(tmp_path, '--models=models.txt')
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    runs = [e.split() for e in events if e.startswith('run ')]
    assert len(runs) == 4

    def value(run, flag):
        return run[run.index(flag) + 1]
    assert sorted(value(r, '--incl') for r in runs) == ['88.5', '89.5', '89.5', '90.0']
    assert {(value(r, '--Q'), value(r, '--rho0'), value(r, '--ref-penalty')) for r in runs} == {
        ('0.27', '92.3', '1.241'), ('0.26', '90.3', '1.245'), ('0.28', '94.0', '1.25'),
        ('0.29', '85.0', '1.26')}
    assert len({value(r, '--suffix') for r in runs}) == 4
    assert len({value(r, '--orblib-dir') for r in runs}) == 4
    assert {value(r, '--ref-source') for r in runs} >= {'src.txt:7', 'models.txt:2'}
    for run in runs:
        assert value(run, '--protocols') == 'exp' and value(run, '--n_threads') == '2'
        limit = next(p for p in run if p.startswith('--memory='))
        assert f"--memory-swap={limit.split('=')[1]}" in run
        assert '--side-file' not in run and '--preflight' not in run and '--stream-orblib' not in run
    uploads = [e for e in events if e.startswith('rclone copyto')]
    assert sum(' yandex:galAgama/out_testhost_d1_nb250_gh0_ser0_multi' in e for e in uploads) == 4
    assert any('galAgama/single_model/multi_d1_nb250_gh0_ser0_n4_' in e and 'models.txt' in e
               for e in uploads)
    storage_calls = [e for e in events if e.startswith('storage ')]
    assert len(storage_calls) == 8
    roots = sorted({call.split('--root ')[1].split()[0][-3:] for call in storage_calls})
    assert roots == ['_m0', '_m1', '_m2', '_m3']
    shutdown = events.index(next(e for e in events if e.startswith('shutdown')))
    assert shutdown > max(i for i, e in enumerate(events) if e.startswith(('rclone ', 'storage ')))
    assert events[shutdown] == 'shutdown +1'
    log = next(tmp_path.glob('launch_multi_*.log')).read_text()
    assert 'multi-model summary: 4 models' in log and '(src.txt:7)' in log


def test_multi_launcher_preflight_runs_all_models_without_side_effects(tmp_path):
    result, events = run_multi(tmp_path, '--models=models.txt', '--preflight')
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    runs = [e.split() for e in events if e.startswith('run ')]
    assert len(runs) == 4 and all('--preflight' in r for r in runs)
    assert not any(e.startswith(('storage ', 'shutdown')) for e in events)
    assert not any(' yandex:galAgama/out_' in e for e in events)


def test_multi_launcher_skips_archived_and_reports_failures(tmp_path):
    result, events = run_multi(
        tmp_path / 'a', '--models=models.txt', '--no-shutdown',
        RECEIPT_EXISTS='orblib_i90.0_d1_nb250_ser0_geomabcdef12_0123456789.npz.json')
    assert result.returncode == 0
    assert not any(e.startswith(('storage ', 'shutdown')) for e in events)
    result, events = run_multi(tmp_path / 'b', '--models=models.txt', '--no-shutdown',
                               RUN_STATUS='failed')
    assert result.returncode == 1
    assert not any(e.startswith('storage ') for e in events)


def test_multi_launcher_rejects_bad_models_without_side_effects(tmp_path):
    for args, models in (((), MODELS_TEXT), (('--models=missing.txt',), None),
                         (('--models=models.txt',), '# only a comment\n'),
                         (('--models=models.txt',), MODELS_TEXT + MODELS_TEXT.splitlines()[1] + '\n'),
                         (('--models=models.txt', '--repeats=2'), MODELS_TEXT)):
        work = tmp_path / f'case{len(list(tmp_path.iterdir()))}'
        result, events = run_multi(work, *args, models=models)
        assert result.returncode == 2, (args, result.stdout + result.stderr)
        assert not events and not list(work.glob('launch_multi_*.log'))
    help_run = subprocess.run(['bash', str(MULTI_LAUNCHER), '--help'], capture_output=True,
                              text=True, timeout=15, cwd=tmp_path)
    assert help_run.returncode == 0 and '--models=FILE' in help_run.stdout
