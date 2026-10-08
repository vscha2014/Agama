import ast
import contextlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / 'Fornax_P21_PCA_w3Sersic_orblib_exp.py'
SCRIPT = HERE / 'run_potential_pair.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair():
    return load(SCRIPT, 'potential_pair_test')


@pytest.fixture
def solver():
    tree = ast.parse(HARNESS.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'solve_orbit_library')
    import time
    agama = SimpleNamespace(solveOpt=lambda **kw: np.ones(kw['matrix'][0].shape[1]) /
                            kw['matrix'][0].shape[1] / 20)
    ns = dict(numpy=np, time=time, agama=agama)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HARNESS), 'exec'), ns)
    return ns['solve_orbit_library']


class Dataset:
    cons_val = np.ones(2)
    cons_err = np.ones(2)
    target = [0, 1]

    def getOrbitMatrix(self, matrix, upsilon):
        return matrix

    def getPenalty(self, prediction, upsilon):
        return np.array([1 + (upsilon - 0.62)**2 + (prediction[0] - 1)**2])


class Potential:
    def __init__(self, scale=1):
        self.scale = scale

    def potential(self, xyz):
        return np.full(len(xyz), -10.0)

    def Tcirc(self, ic):
        return np.full(len(ic), 2.0 / self.scale)

    def export(self, filename):
        Path(filename).write_text(f'synthetic scale={self.scale}\n')


class FakeAgama:
    def __init__(self):
        self.seeds = []
        self.calls = []
        self.mutate = False
        self.fail = False

    def setRandomSeed(self, seed):
        self.seeds.append(seed)

    def setNumThreads(self, threads):
        assert threads >= 1
        return contextlib.nullcontext()

    def orbit(self, **kw):
        assert 'trajsize' not in kw
        assert kw['Omega'] == 0
        self.calls.append(dict(ic=kw['ic'].copy(), time=kw['time'].copy(),
                               potential=kw['potential']))
        if self.fail:
            raise RuntimeError('synthetic orbit failure')
        if self.mutate:
            kw['ic'][0, 0] += 1
        return [np.full((len(kw['ic']), 2), kw['potential'].scale, dtype=np.float32)
                for _ in kw['targets']]


class Stars:
    def __init__(self):
        self.calls = 0

    def sample(self, n, potential):
        self.calls += 1
        ic = np.ones((n, 6)) * 0.1
        ic[:, 0] = potential.scale
        return ic, np.ones(n) / n


@pytest.mark.parametrize('spec', ['0', '-1', '1,1', '3-1', '', 'nan', '2147483648'])
def test_bad_seeds_rejected(pair, spec):
    with pytest.raises(ValueError):
        pair.parse_seeds(spec)


def test_seeds_preserve_prespecified_order(pair):
    assert pair.parse_seeds('42,1-7') == [42, 1, 2, 3, 4, 5, 6, 7]


def test_array_identity_includes_shape_dtype_and_exact_values(pair):
    a = np.arange(12, dtype=np.float64).reshape(2, 6)
    assert pair.array_identity(a) != pair.array_identity(a.astype(np.float32))
    assert pair.array_identity(a) != pair.array_identity(a.reshape(3, 4))
    b = a.copy()
    b[0, 0] = 1e-14
    assert pair.array_identity(a) != pair.array_identity(b)
    assert pair.array_identity(a) == pair.array_identity(a.copy())


@pytest.mark.parametrize('bad', ['shape', 'nan', 'time_zero', 'time_nan', 'count'])
def test_orbit_inputs_fail_closed(pair, bad):
    ic, times = np.ones((4, 6)), np.ones(4)
    if bad == 'shape':
        ic = ic[:, :5]
    elif bad == 'nan':
        ic[1, 2] = np.nan
    elif bad == 'time_zero':
        times[0] = 0
    elif bad == 'time_nan':
        times[0] = np.nan
    else:
        times = times[:-1]
    with pytest.raises(ValueError):
        pair.validate_inputs(ic, times, 4)


def test_unbound_ic_not_clipped_or_resampled(pair):
    ic = np.ones((4, 6)) * 100
    before = ic.copy()
    with pytest.raises(ValueError, match='bound'):
        pair.check_bound({'A': Potential(), 'B': Potential()}, ic)
    np.testing.assert_array_equal(ic, before)


def test_library_identity_prevents_wrong_variant_and_modified_payload(pair, tmp_path):
    mats = [np.ones((4, 2), dtype=np.float32)] * 2
    manifest = dict(schema=pair.SCHEMA, variant='A', seed=42, context={'Q': 1.0})
    path = tmp_path / 'library'
    pair.save_library(path, mats, manifest)
    loaded = pair.load_library(path, manifest)
    assert loaded[0].dtype == mats[0].dtype
    np.testing.assert_array_equal(loaded[0], mats[0])
    with pytest.raises(ValueError, match='manifest'):
        pair.load_library(path, dict(manifest, variant='B'))
    with pytest.raises(FileExistsError):
        pair.save_library(path, mats, manifest)
    with (path / 'matrix_0.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        pair.load_library(path, manifest)


def test_legacy_library_is_not_a_diagnostic_cache(pair, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        pair.load_library(tmp_path, dict(schema=pair.SCHEMA))


def test_common_inputs_and_native_inputs_are_separate(pair, solver, tmp_path):
    agama, stars = FakeAgama(), Stars()
    pots = {'A': Potential(), 'B': Potential(1.01)}
    report = pair.evaluate_pair(agama, pots, stars, [Dataset(), Dataset()], solver,
                                tmp_path / 'pair', {'model': {'Q': 1.0}}, 42, 0.6,
                                num_orbits=12, native=True)
    assert report['status'] == 'ok'
    assert stars.calls == 2 and agama.seeds == [42, 42]
    assert len(agama.calls) == 3
    a, b, native = agama.calls
    np.testing.assert_array_equal(a['ic'], b['ic'])
    np.testing.assert_array_equal(a['time'], b['time'])
    assert not np.array_equal(b['ic'], native['ic'])
    assert not np.array_equal(b['time'], native['time'])
    assert report['variants']['A']['manifest_id'] != report['variants']['B']['manifest_id']
    assert report['variants']['B_native']['manifest_id'] != report['variants']['B']['manifest_id']
    assert report['delta_penalty'] == pytest.approx(0.0001, abs=1e-7)
    assert report['variants']['A']['upsilon'] == pytest.approx(0.62, abs=1e-3)
    assert set(report['cross_penalties']) == {'A', 'B'}
    assert not list(tmp_path.rglob('out_*')) and not list(tmp_path.rglob('4Ups*'))
    assert not list(tmp_path.rglob('checkpoint*')) and not list(tmp_path.rglob('.storage'))
    assert (tmp_path / 'pair/A/weights.npy').is_file()
    assert (tmp_path / 'pair/A/residual_1.npy').is_file()
    assert (tmp_path / 'pair/ic_A.npy').is_file()
    assert (tmp_path / 'pair/time_A.npy').is_file()


def test_orbit_input_mutation_fails_with_partial_report(pair, solver, tmp_path):
    agama = FakeAgama()
    agama.mutate = True
    report = pair.evaluate_pair(agama, {'A': Potential(), 'B': Potential()}, Stars(),
                                [Dataset(), Dataset()], solver, tmp_path / 'pair', {}, 42, 0.6,
                                num_orbits=4)
    assert report['status'] == 'failed'
    assert 'mutat' in report['error'].lower()
    assert (tmp_path / 'pair/report.json').is_file()
    assert (tmp_path / 'pair/ic_A.npy').is_file()


def test_solver_failure_is_not_a_penalty_sentinel(pair, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic solver failure')
    report = pair.evaluate_pair(FakeAgama(), {'A': Potential(), 'B': Potential()}, Stars(),
                                [Dataset(), Dataset()], fail, tmp_path / 'pair', {}, 42, 0.6,
                                num_orbits=4)
    assert report['status'] == 'failed'
    assert 'synthetic solver failure' in report['error']
    assert 'delta_penalty' not in report


def test_stop_leaves_partial_report(pair, solver, tmp_path):
    def stop():
        raise InterruptedError('STOP')
    agama = FakeAgama()
    report = pair.evaluate_pair(agama, {'A': Potential(), 'B': Potential()}, Stars(),
                                [Dataset(), Dataset()], solver, tmp_path / 'pair', {}, 42, 0.6,
                                num_orbits=4, check_stop=stop)
    assert report['status'] == 'stopped'
    assert not agama.calls and not agama.seeds


def test_full_search_and_optimizer_failure(pair, solver, monkeypatch):
    mats = [np.ones((12, 2), dtype=np.float32)] * 2
    fit = pair.fit_library(solver, [Dataset(), Dataset()], mats, 0.6)
    assert fit['upsilon'] == pytest.approx(0.62, abs=1e-3)
    assert fit['penalty'] == pytest.approx(1)
    assert len(fit['final']['weights']) == 12
    monkeypatch.setattr(pair, 'minimize_scalar', lambda *a, **k:
                        SimpleNamespace(success=False, message='maxiter reached'))
    with pytest.raises(ValueError, match='maxiter reached'):
        pair.fit_library(solver, [Dataset(), Dataset()], mats, 0.6)


def test_summary_keeps_missing_failed_and_duplicate_pairs(pair):
    reports = [dict(seed=1, status='ok', delta_penalty=0.1, delta_upsilon=0.01),
               dict(seed=2, status='failed', error='failed')]
    result = pair.summarize_pairs(reports, [1, 2, 3])
    assert result['complete'] == 1 and result['missing'] == [3]
    assert result['failed'] == [2] and result['mean_delta_penalty'] == 0.1
    assert result['sem_delta_penalty'] is None
    duplicate = pair.summarize_pairs(reports + [reports[0]], [1, 2, 3])
    assert duplicate['duplicate_seeds'] == [1] and duplicate['complete'] == 0
    assert len(duplicate['records']) == 3


def test_cli_requires_nonorbital_mode_and_rejects_execution_flag(pair):
    with pytest.raises(SystemExit):
        pair.parse_args([])
    with pytest.raises(SystemExit):
        pair.parse_args(['--execute'])


def test_help_does_not_import_agama_or_torch():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--validate-inputs' in result.stdout and '--preflight' in result.stdout
    assert '--execute' not in result.stdout


def test_no_cloud_or_shell_calls_in_diagnostic_source():
    tree = ast.parse(SCRIPT.read_text())
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not any('subprocess' in call or call.startswith(('requests.', 'os.system')) for call in calls)


def write_ini(path, grid):
    path.write_text('gridSizeR=' + str(len(grid)) + '\n#Phi\n' +
                    ''.join(f'{x:.14g} 0\n' for x in grid) + '#dPhi/dr\n' +
                    ''.join(f'{x:.14g} 0\n' for x in grid))


@pytest.fixture
def evidence(pair, tmp_path):
    recipe = pair.fields.read_recipe(HARNESS)
    model = dict(incl=90., Q=1., gh=0., rh=3., rho0=50., upsilon=0.6)
    parent = tmp_path / 'fields'
    directory = parent / 'model_000'
    directory.mkdir(parents=True)
    coeffs = []
    for tag, n, lmax in [('baseline', 23, 4), ('angular_l24', 184, 24),
                         ('stars_baseline', 23, 4), ('halo_baseline', 23, 4)]:
        grid = np.geomspace(0.001, 100, n)
        path = directory / (tag + '.ini')
        write_ini(path, grid)
        actual = pair.fields.read_grid(path).tolist()
        settings = dict(type='Multipole', gridSizeR=n, lmax=lmax, mmax=0)
        if tag == 'angular_l24':
            settings.update(rmin=actual[0], rmax=actual[-1])
        coeffs.append(dict(tag=tag, actual_grid=actual, requested=settings,
                           coefficient_file=path.name))
    scores = {name: dict(rms=1e-7, max=1e-6) for name in pair.fields.REGIONS}
    data = dict(model=model, reference={'converged': True}, coefficients=coeffs,
                comparisons={'angular_l24': dict(scores=scores, status='pass', main_status='pass')},
                thresholds=dict(rms=1e-4, maximum=1e-3), penalty_checked=False)
    path = directory / 'report.json'
    path.write_text(json.dumps(data))
    top = dict(recipe=recipe, options=dict(radii=192, angles=17),
               models=[dict(directory='model_000')], agama=dict(files=[]))
    (parent / 'report.json').write_text(json.dumps(top))
    return path, model, recipe


def test_evidence_requires_converged_reference_and_exact_model(pair, evidence):
    path, model, recipe = evidence
    proof = pair.read_evidence(path, model, recipe)
    assert proof['coefficients']['B']['requested']['gridSizeR'] == 184
    with pytest.raises(ValueError, match='model'):
        pair.read_evidence(path, dict(model, rho0=50.0000001), recipe)
    data = json.loads(path.read_text())
    data['reference']['converged'] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='reference'):
        pair.read_evidence(path, model, recipe)


@pytest.mark.parametrize('bad', ['regional_failure', 'wrong_grid', 'traversal', 'sparse', 'recipe'])
def test_evidence_rejects_incompatible_checks(pair, evidence, bad):
    path, model, recipe = evidence
    data = json.loads(path.read_text())
    if bad == 'regional_failure':
        data['comparisons']['angular_l24']['scores']['main']['rms'] = 2e-4
    elif bad == 'wrong_grid':
        data['coefficients'][1]['requested']['rmin'] *= 2
    elif bad == 'traversal':
        data['coefficients'][1]['coefficient_file'] = '../outside.ini'
    elif bad == 'sparse':
        top_path = path.parent.parent / 'report.json'
        top = json.loads(top_path.read_text())
        top['options']['radii'] = 96
        top_path.write_text(json.dumps(top))
    else:
        recipe = dict(recipe, stellar=dict(recipe['stellar'], mass=15))
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        pair.read_evidence(path, model, recipe)


def test_validation_never_imports_harness_or_agama(pair, evidence, tmp_path, monkeypatch):
    path, model, recipe = evidence
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n')
    def forbidden(*a, **k):
        pytest.fail('runtime harness imported during validation')
    monkeypatch.setattr(pair, 'load_harness', forbidden)
    output = tmp_path / 'unused'
    args = pair.parse_args(['--models', str(models), '--field-report', str(path),
                           '--harness', str(HARNESS), '--output', str(output), '--validate-inputs'])
    assert pair.main(args) == 0
    assert not output.exists()


def test_preflight_dependency_failure_is_reported_without_orbits(pair, evidence, tmp_path, monkeypatch):
    path, model, recipe = evidence
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n')
    def unavailable(*a, **k):
        raise ModuleNotFoundError('synthetic missing AGAMA')
    monkeypatch.setattr(pair, 'load_harness', unavailable)
    output = tmp_path / 'preflight'
    args = pair.parse_args(['--models', str(models), '--field-report', str(path),
                           '--harness', str(HARNESS), '--output', str(output), '--preflight'])
    assert pair.main(args) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed' and report['penalty_checked'] is False
    assert report['orbit_calls'] == 0 and report['ic_sampled'] is False
    assert 'missing AGAMA' in report['error']
    assert pair.main(args) == 2


@pytest.mark.parametrize('failure', [None, 'binary', 'grid', 'force', 'force_A', 'lossy', 'nonfinite', 'hash'])
def test_real_preflight_path_uses_fields_only(pair, evidence, tmp_path, monkeypatch, failure):
    path, model, recipe = evidence
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n')
    top_path = path.parent.parent / 'report.json'
    top = json.loads(top_path.read_text())
    binary = dict(version='synthetic', files=[dict(path='/synthetic/agama.so', sha256='synthetic')],
                  binary_fingerprint_present=True)
    top['agama'] = binary
    top_path.write_text(json.dumps(top))
    evidence_calls, density_calls, snapshots = [], [], []
    output = tmp_path / 'preflight'

    class FieldPotential:
        def __init__(self, source, name='A', changed=False, reloaded=False):
            self.source, self.name = Path(source), name
            self.changed, self.reloaded = changed, reloaded

        def export(self, filename):
            if self.changed and self.name == 'B' and failure == 'grid':
                write_ini(Path(filename), np.geomspace(0.001, 101, 184))
            else:
                content = self.source.read_bytes()
                if self.changed and self.name == 'A' and failure == 'hash':
                    content += b'\n'
                Path(filename).write_bytes(content)

        def force(self, points):
            snapshots.append(json.loads((output / 'report.json').read_text()))
            changed_force = self.changed and (failure == 'force' and self.name == 'B' or
                                             failure == 'force_A' and self.name == 'A')
            scale = 1.01 if changed_force else 0.99 if self.reloaded and failure == 'lossy' else 1
            result = -scale * points / (1 + np.sum(points**2, axis=1))[:, None]**1.5
            if self.changed and self.name == 'A' and failure == 'nonfinite':
                result[0, 0] = np.nan
            return result

    def density(*args, **kwargs):
        if kwargs:
            density_calls.append(kwargs)
            return SimpleNamespace(origin='evidence')
        return SimpleNamespace(origin=getattr(args[0], 'origin', 'harness'))

    def potential(*args, **kwargs):
        if args:
            return FieldPotential(args[0], reloaded=True)
        evidence_calls.append(kwargs)
        name = 'A' if kwargs['gridSizeR'] == 23 else 'B'
        source = 'baseline.ini' if name == 'A' else 'angular_l24.ini'
        return FieldPotential(path.parent / source, name=name,
                              changed=kwargs['density'].origin == 'harness')

    def forbidden(*args, **kwargs):
        pytest.fail('Preflight must not sample, integrate, fit or seed')

    agama = SimpleNamespace(Potential=potential, Density=density, orbit=forbidden,
                            solveOpt=forbidden, setRandomSeed=forbidden)
    mod = SimpleNamespace(agama=agama, SAVE_ORBLIB=False, REUSE_ORBLIB=False, orblib_store=None,
                          gridx=np.ones(3), gridy=np.ones(3), gridv=np.ones(4),
                          sectAPP=[np.ones((3, 2))], densityStars=SimpleNamespace(sample=forbidden, origin='harness'),
                          datasets=[Dataset(), Dataset()], bounds_original=recipe['bounds'], alphah=2, betah=3)

    def capture_entry(*args, diagnostic, direct_params):
        return diagnostic(baseline=FieldPotential(path.parent / 'baseline.ini', changed=True),
                          density_stars=mod.densityStars, density_halo=object(), num_orbits=100000,
                          int_time=100.0, regul=1.0, upsilon_bounds=(0.1, 1.6))

    mod.halo_IC_lib_weights_pca_fixed = capture_entry
    monkeypatch.setattr(pair, 'load_harness', lambda *a: mod)
    monkeypatch.setattr(pair.single, 'static_context', lambda mod: ({'geom_hash': 'synthetic'}, {}))
    current = dict(binary, files=[dict(path='/synthetic/agama.so', sha256='different')]) if failure == 'binary' else binary
    monkeypatch.setattr(pair.fields, 'module_identity', lambda agama: current)
    args = pair.parse_args(['--models', str(models), '--field-report', str(path),
                           '--harness', str(HARNESS), '--output', str(output), '--preflight'])
    successful = failure in (None, 'lossy')
    assert pair.main(args) == (0 if successful else 1)
    report = json.loads((output / 'report.json').read_text())
    assert report['orbit_calls'] == 0 and report['ic_sampled'] is False
    assert report['solver_checked'] is False and report['library_created'] is False
    assert not list(output.rglob('*.npy')) and not list(output.rglob('out_*'))
    assert report['context']['python'] == sys.executable
    assert report['context']['numpy'] == np.__version__
    assert report['context']['scipy'] == pair.fields.scipy.__version__
    assert report['context']['agama'] == current
    if failure == 'binary':
        assert report['status'] == 'failed'
        assert not (output / 'A.ini').exists()
        return
    assert set(report['potentials']) == {'A', 'B'}
    assert density_calls[0]['mass'] == recipe['stellar']['mass']
    assert density_calls[1]['densitynorm'] == model['rho0']
    assert any(c.get('rmin') == 0.001 and c.get('rmax') == 100 for c in evidence_calls)
    assert all('context' in snap and 'potentials' in snap for snap in snapshots)
    if successful:
        assert report['status'] == 'preflight_pass'
        assert report['manifest_id']
        assert report['serialization_warnings'] == (['A', 'B'] if failure == 'lossy' else [])
        for record in report['potentials'].values():
            assert record['status'] == 'pass' and record['export_matches_evidence'] is True
            assert record['live_reproduction']['max'] == 0
            assert record['live_reproduction']['tolerance'] == 1e-9
            assert record['serialized_reproduction']['max'] == 0
            assert record['export_roundtrip']['scores']['main']['worst_xyz']
            if failure == 'lossy':
                assert record['export_roundtrip']['max'] == pytest.approx(0.01)
                assert record['export_roundtrip']['within_reproduction_tolerance'] is False
                assert record['export_roundtrip']['gates_preflight'] is False
            else:
                assert record['export_roundtrip']['max'] == 0
    else:
        assert report['status'] == 'failed' and 'manifest_id' not in report
        failed = 'B' if failure in ('force', 'grid') else 'A'
        assert report['potentials'][failed]['status'] == 'failed'
        assert report['potentials']['B' if failed == 'A' else 'A']['status'] == 'pass'
        if failure in ('force', 'force_A'):
            assert report['potentials'][failed]['live_reproduction']['max'] == pytest.approx(0.01)
        if failure in ('force_A', 'nonfinite', 'hash'):
            assert any(snap['potentials']['A']['status'] == 'failed' and
                       snap['potentials']['B']['status'] == 'running' for snap in snapshots)


def test_summary_marks_malformed_results_and_refuses_mixed_models(pair):
    malformed = dict(seed=1, status='ok', delta_penalty=None, delta_upsilon=0)
    assert pair.summarize_pairs([malformed], [1])['failed'] == [1]
    models = [dict(seed=i, status='ok', delta_penalty=0.1, delta_upsilon=0,
                   context={'model': {'Q': q}}) for i, q in [(1, 1.0), (2, 0.5)]]
    with pytest.raises(ValueError, match='model'):
        pair.summarize_pairs(models, [1, 2])


def test_pair_refuses_different_matrix_dtypes(pair, solver, tmp_path):
    agama = FakeAgama()
    orbit = agama.orbit
    def changed_dtype(**kw):
        mats = orbit(**kw)
        return [m.astype(np.float64) for m in mats] if len(agama.calls) == 2 else mats
    agama.orbit = changed_dtype
    report = pair.evaluate_pair(agama, {'A': Potential(), 'B': Potential()}, Stars(),
                                [Dataset(), Dataset()], solver, tmp_path / 'pair', {}, 42, 0.6,
                                num_orbits=4)
    assert report['status'] == 'failed' and 'dtype' in report['error']
    assert 'A' in report['variants'] and 'B' not in report['variants']


def test_stop_between_variants_preserves_completed_A(pair, solver, tmp_path):
    calls = []
    def stop():
        calls.append(None)
        if len(calls) == 4:
            raise InterruptedError('STOP between variants')
    report = pair.evaluate_pair(FakeAgama(), {'A': Potential(), 'B': Potential()}, Stars(),
                                [Dataset(), Dataset()], solver, tmp_path / 'pair', {}, 42, 0.6,
                                num_orbits=4, check_stop=stop)
    assert report['status'] == 'stopped'
    assert 'A' in report['variants'] and 'B' not in report['variants']
    assert (tmp_path / 'pair/A/library.json').is_file()


def test_evidence_checksums_rechecked_before_field_build(pair, evidence, tmp_path):
    path, model, recipe = evidence
    proof = pair.read_evidence(path, model, recipe)
    with (path.parent / 'angular_l24.ini').open('a') as stream:
        stream.write('\nchanged\n')
    with pytest.raises(ValueError, match='checksum'):
        pair.field_preflight(None, None, None, None, proof, tmp_path)


def test_production_harness_is_rejected_before_import(pair):
    with pytest.raises(ValueError, match='never production'):
        pair.validate_harness_interface(HERE.parent / 'Fornax_P21_symm_PCA_w3Sersic_yaVM.py')


def test_outdated_experimental_harness_is_rejected_statically(pair, tmp_path):
    path = tmp_path / HARNESS.name
    path.write_text(HARNESS.read_text().replace('allow_orblib_reuse=True, diagnostic=None',
                                               'allow_orblib_reuse=True'))
    with pytest.raises(ValueError, match='updated experimental copy'):
        pair.validate_harness_interface(path)


def test_force_comparison_records_regions_and_worst_position(pair):
    points = np.array([[0.01, 0, 0], [1, 0, 0], [50, 0, 0]])
    reference = np.tile([-2.0, 0, 0], (3, 1))
    measured = reference.copy()
    measured[1, 0] -= 0.2
    result = pair.force_comparison(measured, reference, points)
    assert result['max'] == pytest.approx(0.1)
    assert result['rms'] == pytest.approx(0.1 / np.sqrt(3))
    assert result['scores']['main']['worst_xyz'] == [1.0, 0.0, 0.0]
    assert result['scores']['all']['worst_radius'] == 1
    assert result['scores']['outer_orbits'] is None
    assert result['tolerance'] == 1e-9 and result['within_reproduction_tolerance'] is False


@pytest.mark.parametrize('failure', ['shape', 'nan', 'zero_reference'])
def test_force_comparison_rejects_invalid_force(pair, failure):
    points = np.array([[1.0, 0, 0]])
    measured, reference = np.ones((1, 3)), np.ones((1, 3))
    if failure == 'shape':
        measured = measured[:, :2]
    elif failure == 'nan':
        measured[0, 0] = np.nan
    else:
        reference[:] = 0
    with pytest.raises(ValueError):
        pair.force_comparison(measured, reference, points)


def test_pilot_plan_has_exactly_eleven_integrations(pair):
    assert pair.pilot_variants({'Q': 0.3}) == ['A', 'B', 'A_repeat', 'C', 'B_native']
    assert pair.pilot_variants({'Q': 1.0}) == ['A', 'B', 'B_native']
    assert sum(len(pair.pilot_variants({'Q': q})) for q in (0.3, 1.0, 1.0)) == 11


def test_pilot_cli_is_explicit_and_seed42_only(pair):
    base = ['--models', 'models', '--field-report', 'report', '--output', 'new']
    assert pair.parse_args(base + ['--pilot']).pilot
    assert pair.parse_args(base + ['--pilot-preflight']).pilot_preflight
    with pytest.raises(SystemExit):
        pair.parse_args(base + ['--pilot', '--seeds', '1'])
    with pytest.raises(SystemExit):
        pair.parse_args(base + ['--pilot', '--preflight'])


def test_freeq_evidence_includes_exact_C(pair, evidence):
    path, model, recipe = evidence
    data = json.loads(path.read_text())
    grid = np.geomspace(5e-5, 2200., 399)
    write_ini(path.parent / 'common_fine.ini', grid)
    data['coefficients'].append(dict(tag='common_fine', coefficient_file='common_fine.ini',
        actual_grid=pair.fields.read_grid(path.parent / 'common_fine.ini').tolist(),
        requested=dict(type='Multipole', gridSizeR=399, lmax=40, mmax=0, rmin=5e-5, rmax=2200.)))
    data['comparisons']['common_fine'] = data['comparisons']['angular_l24']
    path.write_text(json.dumps(data))
    proof = pair.read_evidence(path, model, recipe, include_control=True)
    assert proof['coefficients']['C']['requested']['lmax'] == 40
    data['comparisons']['common_fine'] = dict(scores={})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='control'):
        pair.read_evidence(path, model, recipe, include_control=True)


def test_pilot_checks_preserve_primary_fit_and_all_controls(pair, solver, tmp_path):
    agama = FakeAgama()
    a = Potential()
    pots = dict(A=a, B=Potential(1.01), A_repeat=a, C=Potential(1.02))
    events = []
    report = pair.evaluate_pair(agama, pots, Stars(), [Dataset(), Dataset()], solver,
        tmp_path / 'pilot', {}, 42, 0.6, num_orbits=12, native=True,
        pilot_checks=True, progress=events.append)
    assert report['status'] == 'ok'
    assert report['orbit_calls'] == 5
    assert len(agama.calls) == 5
    assert report['technical_checks']['passed']
    assert report['technical_checks']['budget'] == 0.001
    assert report['control_deltas']['A_repeat_minus_A'] == 0
    assert set(report['variants']) == set(pots) | {'B_native'}
    assert all(v['settings']['xatol'] == 1e-3 for v in report['variants'].values())
    assert all(v['strict_search']['settings']['xatol'] == 1e-4 for v in report['variants'].values())
    assert any(e['stage'] == 'integrate' for e in events)
    assert all(e['resources']['peak_rss_bytes'] > 0 for e in events)
    assert report['ic_radius']['count_outside_field_probes'] == 0


def test_pilot_disk_gate_precedes_seed_or_orbits(pair, solver, tmp_path, monkeypatch):
    monkeypatch.setattr(pair.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    agama = FakeAgama()
    report = pair.evaluate_pair(agama, dict(A=Potential(), B=Potential()), Stars(),
        [Dataset(), Dataset()], solver, tmp_path / 'pilot', {}, 42, 0.6,
        num_orbits=4, pilot_checks=True)
    assert report['status'] == 'failed' and 'disk' in report['error'].lower()
    assert not agama.seeds and not agama.calls


def test_full_search_checks_stop_between_solver_calls(pair, solver):
    def stop():
        raise InterruptedError('requested stop')
    with pytest.raises(InterruptedError):
        pair.fit_library(solver, [Dataset(), Dataset()], [np.ones((4, 2))] * 2, 0.6,
                         check_stop=stop)


@pytest.mark.parametrize('failure', [None, 'binary', 'control_force', 'solver', 'stop'])
def test_pilot_cli_uses_checked_live_objects_and_persists_failures(pair, evidence, solver, tmp_path, monkeypatch, failure):
    path, model, recipe = evidence
    model.update(Q=0.5, rh=2.0)
    data = json.loads(path.read_text())
    data['model'] = model
    write_ini(path.parent / 'common_fine.ini', np.geomspace(5e-5, 2200., 399))
    data['coefficients'].append(dict(tag='common_fine', coefficient_file='common_fine.ini',
        actual_grid=pair.fields.read_grid(path.parent / 'common_fine.ini').tolist(),
        requested=dict(type='Multipole', gridSizeR=399, lmax=40, mmax=0, rmin=5e-5, rmax=2200.)))
    data['comparisons']['common_fine'] = data['comparisons']['angular_l24']
    path.write_text(json.dumps(data))
    binary = dict(files=[dict(path='/fake/agama.so', sha256='test')])
    top_path = path.parent.parent / 'report.json'
    top = json.loads(top_path.read_text())
    top['agama'] = binary
    top_path.write_text(json.dumps(top))
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n90 1 0 25 50 0.6\n90 0.5 0 2 50 0.6\n')
    built = []

    class FieldPotential(Potential):
        def __init__(self, name, loaded=False):
            super().__init__({'A': 1, 'B': 1.01, 'C': 1.02}[name])
            self.name, self.loaded = name, loaded
            self.source = path.parent / {'A': 'baseline.ini', 'B': 'angular_l24.ini', 'C': 'common_fine.ini'}[name]

        def export(self, filename):
            Path(filename).write_bytes(self.source.read_bytes())

        def force(self, points):
            changed = (failure == 'control_force' and self.name == 'C' and
                       self is next(p for p in built if p.name == 'C'))
            return -points * self.scale * (1.1 if changed else 1)

    def potential(*args, **kw):
        if args:
            name = 'C' if Path(args[0]).name in ('C.ini', 'common_fine.ini') else (
                'B' if Path(args[0]).name in ('B.ini', 'angular_l24.ini') else 'A')
            return FieldPotential(name, loaded=True)
        name = {23: 'A', 184: 'B', 399: 'C'}[kw['gridSizeR']]
        value = FieldPotential(name)
        built.append(value)
        return value

    agama = FakeAgama()
    agama.Density = lambda *a, **kw: object()
    agama.Potential = potential
    original_orbit = agama.orbit
    def live_orbit(**kw):
        assert not kw['potential'].loaded
        return original_orbit(**kw)
    agama.orbit = live_orbit
    def selected_solver(*args, **kw):
        if failure == 'solver':
            raise RuntimeError('pilot solver failed')
        if failure == 'stop':
            raise InterruptedError('pilot STOP')
        return solver(*args, **kw)
    mod = SimpleNamespace(agama=agama, SAVE_ORBLIB=False, REUSE_ORBLIB=False, orblib_store=None,
        gridx=np.ones(3), gridy=np.ones(3), gridv=np.ones(4), sectAPP=[np.ones((3, 2))],
        densityStars=Stars(), datasets=[Dataset(), Dataset()], bounds_original=recipe['bounds'], alphah=2, betah=3)
    def capture(*args, diagnostic, direct_params):
        return diagnostic(baseline=FieldPotential('A'), density_stars=mod.densityStars,
            density_halo=object(), datasets=mod.datasets, num_orbits=100000, int_time=100., regul=1.,
            upsilon_bounds=(0.1, 1.6), solve_library=selected_solver)
    mod.halo_IC_lib_weights_pca_fixed = capture
    monkeypatch.setattr(pair, 'load_harness', lambda *args: mod)
    monkeypatch.setattr(pair.single, 'static_context', lambda mod: ({'geom_hash': 'test'}, {}))
    monkeypatch.setattr(pair.fields, 'module_identity', lambda agama:
        dict(files=[dict(path='/fake/agama.so', sha256='changed')]) if failure == 'binary' else binary)
    evaluate = pair.evaluate_pair
    monkeypatch.setattr(pair, 'evaluate_pair', lambda *a, **kw: evaluate(*a, **dict(kw, num_orbits=12)))
    output = tmp_path / 'pilot'
    args = pair.parse_args(['--models', str(models), '--model-index', '2', '--field-report', str(path),
        '--harness', str(HARNESS), '--output', str(output), '--pilot'])
    code = pair.main(args)
    report = json.loads((output / 'report.json').read_text())
    if failure in ('binary', 'control_force'):
        assert code == 1 and not agama.calls and not agama.seeds
        assert report['status'] == 'failed'
    elif failure in ('solver', 'stop'):
        assert code == 1 and len(agama.calls) == 1
        assert report['status'] == ('stopped' if failure == 'stop' else 'failed')
        assert (output / 'seed_42/A/library.json').exists()
    else:
        assert code == 0 and report['status'] == 'pilot_pass'
        assert report['orbit_calls'] == 5 and report['penalty_checked'] is True
        assert report['plan']['num_orbits'] == 100000
        assert report['evaluation']['context']['construction_mode'] == 'live_density'
        assert set(report['potentials']) == {'A', 'B', 'C'}
        assert (output / 'pilot.tsv').exists()
        reviewed = pair.reviewed_freeq(output / 'report.json', report['plan'])
        assert reviewed['environment_signature'] == report['environment_signature']
        bad = dict(report['plan'], models_sha256='different')
        with pytest.raises(ValueError, match='incompatible'):
            pair.reviewed_freeq(output / 'report.json', bad)
        report['evaluation']['technical_checks']['passed'] = False
        (output / 'report.json').write_text(json.dumps(report))
        with pytest.raises(ValueError, match='technical'):
            pair.reviewed_freeq(output / 'report.json', report['plan'])
    assert pair.main(args) == 2


def test_q1_pilot_requires_explicit_review_before_import(pair, evidence, tmp_path, monkeypatch):
    path, model, recipe = evidence
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n90 1 0 25 50 0.6\n90 0.5 0 2 50 0.6\n')
    monkeypatch.setattr(pair, 'load_harness', lambda *a: pytest.fail('No runtime before review'))
    output = tmp_path / 'pilot'
    args = pair.parse_args(['--models', str(models), '--field-report', str(path), '--pilot',
                           '--harness', str(HARNESS), '--output', str(output)])
    assert pair.main(args) == 2
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['missing', 'order', 'incl'])
def test_pilot_refuses_changed_model_plan(pair, mutation):
    models = [dict(Q=1., rh=3., incl=90), dict(Q=1., rh=25., incl=90), dict(Q=0.3, rh=2., incl=90)]
    if mutation == 'missing':
        models.pop()
    elif mutation == 'order':
        models.reverse()
    else:
        models[2]['incl'] = 85
    with pytest.raises(ValueError, match='three fixed rows'):
        pair.validate_pilot_models(models, 0)


@pytest.mark.parametrize('mode,index', [('validate', 2), ('preflight', 2), ('pilot', 2), ('pilot', 0), ('fields', 2)])
def test_vm_launcher_is_offline_readonly_and_propagates_exit(tmp_path, mode, index):
    import os
    bins = tmp_path / 'bin'
    bins.mkdir()
    log = tmp_path / 'docker.jsonl'
    docker = bins / 'docker'
    docker.write_text('#!' + sys.executable + '\n' + '''import json, os, sys, time
with open(os.environ['MOCK_DOCKER_LOG'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
command = sys.argv[1]
if command == 'info': print('true')
elif command == 'image': print('sha256:existing-image')
elif command == 'create': print('mock-container')
elif command == 'start': time.sleep(0.1); sys.exit(3)
elif command == 'inspect': print('{"ExitCode":3,"OOMKilled":false}')
elif command == 'stats': print('{}')
else: sys.exit(9)
''')
    docker.chmod(0o755)
    df = bins / 'df'
    df.write_text('#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted\\nmock 90000000 1 89999999 1%% /mock\\n"\n')
    df.chmod(0o755)
    inputs = tmp_path / 'input with spaces'
    inputs.mkdir()
    (inputs / 'models.txt').write_text('synthetic')
    (inputs / 'report.json').write_text('{}')
    output = tmp_path / 'new output'
    args = ['bash', str(HERE / 'launch_potential_pair.sh'), f'--mode={mode}',
        f'--input-root={inputs}', '--models=models.txt', '--field-report=report.json',
        f'--model-index={index}', f'--output={output}', '--memory-gb=1']
    if mode == 'pilot' and index == 0:
        args.append(f'--reviewed-freeq={inputs / "report.json"}')
    env = dict(os.environ, PATH=str(bins) + ':' + os.environ['PATH'],
               MOCK_DOCKER_LOG=str(log), PAIR_MONITOR_INTERVAL='0.01')
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == 3, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    create = next(call for call in calls if call[0] == 'create')
    for option in ('--network=none', '--read-only', '--memory=1g', '--memory-swap=1g'):
        assert option in create
    assert '--entrypoint' in create and 'python3' in create
    assert 'sha256:existing-image' in create and '--rm' not in create
    assert any('dst=/code,readonly' in arg for arg in create)
    assert any('dst=/input,readonly' in arg for arg in create)
    assert not any(call[0] in ('pull', 'rm', 'kill') for call in calls)
    assert (output / 'container_state.json').exists()
    before = log.read_text()
    repeat = subprocess.run(args, env=env, capture_output=True, text=True)
    assert repeat.returncode == 2 and log.read_text() == before


def test_manifest_roundtrip_preserves_tuple_recipe_identity(pair, tmp_path):
    manifest = dict(schema=pair.SCHEMA, context=dict(bounds={'rh': (0.5, 30.0)}))
    directory = tmp_path / 'library'
    pair.save_library(directory, [np.ones((4, 2))] * 2, manifest)
    assert len(pair.load_library(directory, manifest)) == 2
    changed = dict(manifest, context=dict(bounds={'rh': (0.5, 31.0)}))
    with pytest.raises(ValueError, match='manifest'):
        pair.load_library(directory, changed)


def test_large_field_penalty_shift_is_not_a_technical_failure(pair, solver, tmp_path):
    a = Potential()
    report = pair.evaluate_pair(FakeAgama(), dict(A=a, B=Potential(2), A_repeat=a, C=Potential(2.1)),
        Stars(), [Dataset(), Dataset()], solver, tmp_path / 'pilot', {}, 42, 0.6,
        num_orbits=12, native=True, pilot_checks=True)
    assert report['delta_penalty'] > 0.01
    assert report['technical_checks']['passed']
    assert 'C_minus_B' in report['sensitivity_flags']


def test_strict_search_diagnostic_does_not_replace_primary_penalty(pair, solver, tmp_path, monkeypatch):
    original = pair.fit_library
    def fit(*args, **kw):
        result = original(*args, **kw)
        if kw.get('xatol') == 1e-4:
            result['penalty'] += 0.01
        return result
    monkeypatch.setattr(pair, 'fit_library', fit)
    report = pair.evaluate_pair(FakeAgama(), dict(A=Potential(), B=Potential()), Stars(),
        [Dataset(), Dataset()], solver, tmp_path / 'pilot', {}, 42, 0.6,
        num_orbits=4, pilot_checks=True)
    assert report['status'] == 'ok' and not report['technical_checks']['passed']
    assert report['variants']['A']['penalty'] == pytest.approx(1)


def test_pilot_environment_identity_excludes_model_but_includes_threads_and_construction_code(pair):
    plan = dict(implementation_sha256='a', fields_implementation_sha256='b', single_implementation_sha256='c',
                recipe=dict(sha256='h', source='s', halo='fixed'), threads=8)
    context = dict(python_version='p', numpy='n', scipy='s', agama={'binary': 'g'},
                   container_image_id='image', catalogue_sha256='table', geometry_arrays=[],
                   observations=[], model={'Q': 1})
    base = pair.pilot_environment(plan, context)
    assert pair.pilot_environment(plan, dict(context, model={'Q': 0.3})) == base
    assert pair.pilot_environment(dict(plan, threads=4), context) != base
    assert pair.pilot_environment(plan, dict(context, container_image_id='another-image')) != base
    assert pair.pilot_environment(dict(plan, implementation_sha256='changed'), context) != base


@pytest.mark.parametrize('fail_import', [False, True])
def test_harness_import_cannot_install_its_checkpoint_signal_handler(pair, tmp_path, monkeypatch, fail_import):
    import signal
    numbers = (signal.SIGTERM, signal.SIGINT)
    before = {n: signal.getsignal(n) for n in numbers}
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    def fake_import(*args):
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        assert all(n in blocked for n in numbers)
        for n in numbers:
            signal.signal(n, lambda *args: pytest.fail('Harness checkpoint handler escaped import'))
        if fail_import:
            raise RuntimeError('import failed')
        return 'loaded'
    monkeypatch.setattr(pair, 'load_module', fake_import)
    if fail_import:
        with pytest.raises(RuntimeError, match='import failed'):
            pair.load_harness(HARNESS, {'incl': 90}, 1, tmp_path)
    else:
        assert pair.load_harness(HARNESS, {'incl': 90}, 1, tmp_path) == 'loaded'
    assert {n: signal.getsignal(n) for n in numbers} == before
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == mask
