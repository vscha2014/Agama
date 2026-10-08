import ast
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / 'check_potential_convergence.py'
HARNESS = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
spec = importlib.util.spec_from_file_location('potential_check', SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture
def recipe():
    return check.read_recipe(HARNESS)


@pytest.fixture
def model():
    return dict(incl=90.0, Q=1.0, gh=0.0, rh=3.0, rho0=50.0, upsilon=0.6,
                source='synthetic fixture')


def test_recipe_matches_harness_without_importing_it(recipe):
    assert recipe['baseline'] == dict(type='Multipole', gridSizeR=23, lmax=4, mmax=0)
    assert recipe['stellar'] == dict(type='Sersic', sersicIndex=0.8, mass=14.0,
                                    scaleRadius=math.pi * 143 / 180 * 16.4 / 60)
    assert recipe['halo'] == dict(type='spheroid', alpha=2.0, beta=3,
                                 outercutoffradius=55.0, cutoffstrength=2.5)
    assert recipe['q_ap'] == 0.69
    assert recipe['bounds']['rh'] == (0.5, 30.0)
    assert recipe['upsilon_bounds'] == [0.1, 1.6]
    assert 'orblib_exp' not in sys.modules
    assert 'torch' not in sys.modules


def test_recipe_refuses_unknown_expression(tmp_path):
    path = tmp_path / 'harness.py'
    path.write_text(HARNESS.read_text().replace('massSt    = 14.0', 'massSt = unknown_call()'))
    with pytest.raises(ValueError, match='expression'):
        check.read_recipe(path)


def test_recipe_refuses_changed_geometry(tmp_path):
    path = tmp_path / 'harness.py'
    path.write_text(HARNESS.read_text().replace(
        'axRZst  = (q_ap2 - cosbeta**2)**0.5/sinbeta', 'axRZst = q_ap'))
    with pytest.raises(ValueError, match='axRZst'):
        check.read_recipe(path)


def test_density_parameters_do_not_apply_upsilon_twice(recipe, model):
    stars, halo = check.density_parameters(model, recipe)
    assert stars['mass'] == 14.0
    assert stars['axisRatioZ'] == pytest.approx(0.69)
    assert halo['densitynorm'] == 50.0
    assert halo['scaleradius'] == 3.0
    assert halo['axisratioz'] == 1.0
    assert recipe['stellar']['mass'] == 14.0


@pytest.mark.parametrize('tail', ['', ' 1.4 2026-01-01 00:00:00',
                                   ' 30 1.4 1e18 18', ' 30 1.4 1e18 18 2'])
def test_model_formats_and_provenance(tmp_path, recipe, tail):
    path = tmp_path / 'models.txt'
    path.write_text('# header\n90 1 0 3 50 0.6' + tail + ' # control\n')
    rows = check.read_models(path, recipe)
    assert len(rows) == 1 and rows[0]['rho0'] == 50
    assert rows[0]['label'] == 'control'
    assert rows[0]['source'].endswith('models.txt:2')
    assert rows[0]['line'].endswith('# control')


@pytest.mark.parametrize('line', ['garbage', '90 1 0 31 50 0.6', '40 1 0 3 50 0.6',
                                  '90 1 nan 3 50 0.6', '90 1 0 3 50 0',
                                  '90 0 0 3 50 0.6', '90 1 0 3 50 0.6 -1 d t'])
def test_bad_model_rows_fail_instead_of_being_skipped(tmp_path, recipe, line):
    path = tmp_path / 'models.txt'
    path.write_text(line + '\n')
    with pytest.raises(ValueError):
        check.read_models(path, recipe)


def test_duplicate_rows_are_preserved(tmp_path, recipe):
    path = tmp_path / 'models.txt'
    path.write_text('90 1 0 3 50 0.6\n' * 2)
    assert len(check.read_models(path, recipe)) == 2


def test_spherical_field_uncut_analytic_mass(recipe, model):
    recipe['halo']['outercutoffradius'] = math.inf
    radii = np.array([0.001, 0.05, 0.7, 3., 12., 100.])
    points = np.column_stack((radii, np.zeros((len(radii), 2))))
    field = check.spherical_field(points, model, recipe)
    x = radii / model['rh']
    expected_mass = 4 * math.pi * model['rho0'] * model['rh']**3 * (
        np.arcsinh(x) - x / np.sqrt(1 + x*x))
    assert np.allclose(-field['acc'][:, 0] * radii**2, expected_mass, rtol=2e-8)
    assert np.all(field['acc'][:, 1:] == 0)
    assert np.all(np.diff(field['pot']) > 0)


def test_spherical_field_harmonic_limit_and_scaling(recipe, model):
    model['rh'] = 30.0
    points = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])
    field = check.spherical_field(points, model, recipe)
    assert np.allclose(field['acc'], -4*math.pi/3*model['rho0']*points, rtol=1e-8)
    doubled = check.spherical_field(points, dict(model, rho0=100), recipe)
    assert np.allclose(doubled['acc'], 2 * field['acc'], rtol=1e-10)
    assert np.allclose(doubled['pot'], 2 * field['pot'], rtol=1e-10)


def test_spherical_field_cusp_cutoff_and_gradient(recipe, model):
    model['gh'] = 1.2
    r = np.array([0.01, 0.2, 2., 20., 110.])
    points = np.column_stack((r, np.zeros((len(r), 2))))
    loose = check.spherical_field(points, model, recipe, rtol=1e-8)
    tight = check.spherical_field(points, model, recipe, rtol=1e-11)
    assert np.allclose(loose['acc'], tight['acc'], rtol=1e-8)
    step = 1e-4
    plus = check.spherical_field(points * (1 + step), model, recipe)
    minus = check.spherical_field(points * (1 - step), model, recipe)
    assert np.allclose(-(plus['pot'] - minus['pot']) / (2 * step * r),
                       tight['acc'][:, 0], rtol=1e-6)
    assert check.halo_density(110., model, recipe) < check.halo_density(55., model, recipe)


def test_spherical_reference_rejects_flattening_and_origin(recipe, model):
    with pytest.raises(ValueError):
        check.spherical_field(np.array([[1., 0, 0]]), dict(model, Q=0.5), recipe)
    with pytest.raises(ValueError):
        check.spherical_field(np.zeros((1, 3)), model, recipe)


def test_mass_integrator_with_plummer():
    radii = np.array([0.001, 0.2, 1., 10., 100.])
    density = lambda r: 3 / (4 * math.pi) * (1 + r*r)**-2.5
    measured = np.array([check.enclosed_mass(density, r) for r in radii])
    assert np.allclose(measured, radii**3 / (1+radii**2)**1.5, rtol=1e-9)


def test_grid_parser(tmp_path):
    path = tmp_path / 'potential.ini'
    path.write_text('[Potential]\ntype=Multipole\ngridSizeR=3\nlmax=0\n'
                    'Coefficients\n#Phi\n#radius l=0,m=0\n0.01 -3\n1 -2\n100 -1\n'
                    '\n#dPhi/dr\n#radius l=0,m=0\n0.01 1\n1 1\n100 1\n')
    assert np.array_equal(check.read_grid(path), [0.01, 1, 100])
    path.write_text(path.read_text().replace('100 -1', '0.001 -1'))
    with pytest.raises(ValueError):
        check.read_grid(path)


def test_grid_extensions_keep_log_spacing():
    original = dict(gridSizeR=23, rmin=0.001, rmax=100.)
    wide = check.expand_grid(original, 2.)
    assert wide['rmin'] == 0.0005 and wide['rmax'] == 200
    assert math.log(wide['rmax']/wide['rmin'])/(wide['gridSizeR']-1) <= (
        math.log(original['rmax']/original['rmin'])/(original['gridSizeR']-1))


def test_metrics_ignore_potential_zero_and_component_zeros():
    points = np.array([[0.1, 0, 0], [0, 1, 0], [0, 0, 2.]])
    ref = dict(acc=-points, pot=np.array([0.005, 0.5, 2.]), pivot=0.5)
    shifted = dict(acc=-points, pot=ref['pot'] + 1e5, pivot=1e5+0.5)
    scores, errors = check.compare_fields(shifted, ref, points)
    assert scores['main']['max'] == 0
    assert np.max(np.abs(errors['potential_difference'])) < 1e-10
    biased = dict(ref, acc=-points*1.001)
    scores, errors = check.compare_fields(biased, ref, points)
    assert scores['main']['max'] == pytest.approx(0.001)
    assert np.all(np.isfinite(errors['component_scaled']))
    assert scores['main']['worst_xyz'] in points.tolist()


def test_nonfinite_field_is_not_a_pass():
    points = np.array([[1., 0, 0]])
    ref = dict(acc=-points, pot=np.array([-1.]), pivot=-1.)
    with pytest.raises(ValueError, match='finite'):
        check.compare_fields(dict(ref, acc=points*np.nan), ref, points)


def test_unconverged_reference_cannot_pass():
    good = {'rms': 0., 'max': 0.}
    bad = {'rms': 1., 'max': 1.}
    assert check.verdict(False, good, 1e-4, 1e-3) == 'reference_unconverged'
    assert check.verdict(True, good, 1e-4, 1e-3) == 'pass'
    assert check.verdict(True, bad, 1e-4, 1e-3) == 'needs_refinement'


def test_output_refuses_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        check.create_output(tmp_path)
    target = tmp_path / 'new'
    check.create_output(target)
    assert target.is_dir()
    with pytest.raises(FileExistsError):
        check.create_output(target)


def test_probe_grid_has_off_node_points_and_axes():
    grid = np.array([0.01, 0.1, 1., 10.])
    points = check.probe_points([grid], 24, 7)
    radii = np.linalg.norm(points, axis=1)
    assert np.any(np.isclose(radii, math.sqrt(0.1)))
    assert np.any(points[:, 0] == 0)
    assert np.any(points[:, 2] == 0)
    assert np.all(radii > 0)


def test_cli_validation_does_not_import_agama(tmp_path):
    models = tmp_path / 'models.txt'
    models.write_text('90 1 0 3 50 0.6\n')
    output = tmp_path / 'not_created'
    result = subprocess.run([sys.executable, str(SCRIPT), '--models', str(models),
                             '--validate-inputs', '--output', str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'validated' in result.stdout
    assert not output.exists()


def test_no_orbits_solver_storage_or_network_calls():
    tree = ast.parse(SCRIPT.read_text())
    forbidden = {'orbit', 'solveOpt', 'sample', 'setRandomSeed', 'setUnits', 'system', 'Popen',
                 'run', 'call', 'check_call', 'check_output', 'urlopen'}
    calls = [node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not (set(calls) & forbidden)


class FakeDensity:
    def __init__(self, *components, **settings):
        self.components = components
        self.settings = settings

    def field(self, points):
        if self.components:
            return check.add_fields(*(component.field(points) for component in self.components))
        mass = self.settings.get('mass', self.settings.get('densitynorm', 1.))
        scale = self.settings.get('scaleRadius', self.settings.get('scaleradius', 1.))
        points = np.asarray(points, dtype=float)
        r2 = np.sum(points**2, axis=1)
        return dict(acc=-mass*points/(scale*scale+r2)[:, None]**1.5,
                    pot=-mass/np.sqrt(scale*scale+r2), pivot=-mass/math.sqrt(scale*scale+1))


class FakePotential:
    def __init__(self, owner, density=None, **settings):
        self.owner = owner
        self.settings = settings
        self.density = density or FakeDensity(**settings)
        self.bias = 0.
        if settings.get('type') == 'Multipole':
            if owner.baseline_bad and settings['gridSizeR'] == 23:
                self.bias += 0.01
            if owner.reference_bad:
                self.bias += 0.05/(settings['lmax']+1)

    def field(self, points):
        field = self.density.field(points)
        return {key: value*(1+self.bias) for key, value in field.items()}

    def force(self, points):
        return self.field(np.atleast_2d(points))['acc']

    def potential(self, points):
        value = self.field(np.atleast_2d(points))['pot']
        return value if np.ndim(points) == 2 else float(value[0])

    def export(self, name):
        settings = self.settings
        grid = np.geomspace(settings.get('rmin', 0.01), settings.get('rmax', 100.), settings['gridSizeR'])
        with open(name, 'w') as stream:
            stream.write(f'gridSizeR={len(grid)}\nlmax={settings["lmax"]}\nCoefficients\n')
            for tag in ('#Phi', '#dPhi/dr'):
                stream.write(tag + '\n#radius l=0,m=0\n')
                for radius in grid:
                    stream.write(f'{radius:.15g} -1\n')
        self.owner.saved[name] = self


class FakeAgama:
    __version__ = 'mock-not-a-real-agama-run'

    def __init__(self, baseline_bad=False, reference_bad=False):
        self.baseline_bad = baseline_bad
        self.reference_bad = reference_bad
        self.saved = {}
        self.calls = []

    def Density(self, *components, **settings):
        self.calls.append(('density', settings))
        return FakeDensity(*components, **settings)

    def Potential(self, **settings):
        self.calls.append(('potential', settings))
        if 'file' in settings:
            return self.saved[settings['file']]
        return FakePotential(self, **settings)

    def setNumThreads(self, threads):
        self.calls.append(('threads', threads))
        return contextlib.nullcontext()

    def __getattr__(self, name):
        if name in ('orbit', 'solveOpt', 'sample', 'setRandomSeed', 'setUnits'):
            pytest.fail(f'Forbidden AGAMA operation: {name}')
        raise AttributeError(name)


def fake_run(monkeypatch, tmp_path, fake, *options):
    original = check.importlib.import_module
    monkeypatch.setattr(check.importlib, 'import_module',
                        lambda name, *a, **kw: fake if name == 'agama' else original(name, *a, **kw))
    models = tmp_path / 'models.txt'
    models.write_text('90 0.6 0 3 50 0.6 # synthetic mock model\n')
    output = tmp_path / 'output'
    code = check.main(['--models', str(models), '--output', str(output), '--radii', '16',
                       '--angles', '3', '--radial-nodes', '23,46', '--angular-orders', '4,8', *options])
    return code, output


@pytest.mark.parametrize('bad_baseline,bad_reference,expected', [
    (False, False, 'pass'), (True, False, 'needs_refinement'),
    (False, True, 'reference_unconverged')])
def test_full_mocked_run_and_statuses(monkeypatch, tmp_path, bad_baseline, bad_reference, expected):
    fake = FakeAgama(baseline_bad=bad_baseline, reference_bad=bad_reference)
    if bad_reference:
        monkeypatch.setattr(check, 'agama_self_test', lambda *args: {'status': 'mocked'})
    code, output = fake_run(monkeypatch, tmp_path, fake, '--no-plots')
    top = json.loads((output / 'report.json').read_text())
    assert code == (0 if expected == 'pass' else 3), top
    assert top['status'] == expected
    result = json.loads((output / 'model_000/report.json').read_text())
    assert result['status'] == expected
    assert result['penalty_checked'] is False
    assert result['units']['upsilon_applied'] is False
    assert result['density_parameters']['stars']['mass'] == 14
    assert result['density_parameters']['halo']['densitynorm'] == 50
    assert result['reference']['halo']['reference_kind'].startswith('numerically converged')
    assert (result['field_only_candidate'] is None) == bad_reference
    coefficients = {item['tag']: item for item in result['coefficients']}
    baseline_grid = coefficients['baseline']['actual_grid']
    for tag in ('radial_n23', 'radial_n46'):
        requested = coefficients[tag]['requested']
        assert requested['lmax'] == 4
        assert requested['rmin'] == baseline_grid[0] and requested['rmax'] == baseline_grid[-1]
    assert coefficients['angular_l8']['requested']['gridSizeR'] == 46
    assert 'rmin' not in coefficients['auto_n46']['requested']
    assert coefficients['range_x2']['requested']['gridSizeR'] > 46
    assert (output / 'model_000/errors.tsv').stat().st_size > 100
    assert not list(output.rglob('*.npz'))
    assert not list(output.rglob('out_*'))
    assert not list(output.rglob('checkpoint*'))


def test_mocked_spherical_run_uses_independent_force(monkeypatch, tmp_path, recipe, model):
    directory = check.create_output(tmp_path / 'model')
    args = check.parse_args(['--models', 'unused', '--no-plots', '--radii', '16', '--angles', '3',
                             '--radial-nodes', '23,46', '--angular-orders', '4,8'])
    result = check.check_model(FakeAgama(), model, recipe, args, directory)
    assert result['reference']['converged']
    assert result['reference']['halo']['reference_kind'].startswith('independent spherical')
    assert result['status'] == 'needs_refinement'
    assert result['comparisons']['halo_baseline']['scores']['main']['max'] > 0.01


def test_plot_outputs_in_mocked_run(monkeypatch, tmp_path):
    code, output = fake_run(monkeypatch, tmp_path, FakeAgama())
    assert code == 0
    for filename in ('force_errors.png', 'error_map.png', 'radial_grids.png'):
        assert (output / 'model_000' / filename).stat().st_size > 100


def test_runtime_failure_preserves_partial_report(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('simulated potential failure')

    monkeypatch.setattr(check, 'check_model', fail)
    code, output = fake_run(monkeypatch, tmp_path, FakeAgama(), '--no-plots')
    report = json.loads((output / 'report.json').read_text())
    assert code == 1 and report['status'] == 'failed'
    assert report['smoke_test']['status'] == 'pass'
    assert report['models'][0]['status'] == 'failed'
    assert 'simulated potential failure' in report['error']
    assert (output / 'smoke_plummer.ini').exists()


def test_smoke_test_requires_no_harness_or_models(monkeypatch, tmp_path):
    fake = FakeAgama()
    monkeypatch.setattr(check.importlib, 'import_module', lambda name: fake)
    output = tmp_path / 'smoke'
    assert check.main(['--self-test', '--harness', '/nonexistent/harness', '--output', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['models'] == [] and report['recipe'] is None
    assert report['smoke_test']['status'] == 'pass'


def test_missing_agama_produces_actionable_error(monkeypatch, tmp_path, capsys):
    def unavailable(name):
        raise ModuleNotFoundError("No module named 'agama'", name='agama')

    monkeypatch.setattr(check.importlib, 'import_module', unavailable)
    output = tmp_path / 'missing'
    assert check.main(['--self-test', '--output', str(output)]) == 1
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed'
    assert 'user gala' in capsys.readouterr().err


def test_existing_output_is_never_modified_by_cli(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text('preserve me')
    assert check.main(['--self-test', '--output', str(tmp_path)]) == 2
    assert path.read_text() == 'preserve me'


@pytest.mark.parametrize('options', [['--threads', '0'], ['--radii', '2'],
                                     ['--rms-tol', 'nan'], ['--max-tol', '-1'],
                                     ['--radial-nodes', '23,23'], ['--radial-nodes', '1,23'],
                                     ['--angular-orders', '4,7'], ['--angular-orders', '4,64']])
def test_cli_rejects_invalid_sweep_options(options):
    with pytest.raises(SystemExit):
        check.parse_args(['--self-test', *options])


@pytest.mark.parametrize('rh', [0.5, 30.])
@pytest.mark.parametrize('gh', [0., 0.2, 1.6])
def test_spherical_reference_at_parameter_boundaries(recipe, model, rh, gh):
    model = dict(model, rh=rh, gh=gh, rho0=120.)
    radii = np.array([0.001, 0.05, 2.1, 10., 110.])
    points = np.column_stack((radii, np.zeros((len(radii), 2))))
    field = check.spherical_field(points, model, recipe, rtol=1e-12)
    generic = np.array([check.enclosed_mass(lambda r: check.halo_density(r, model, recipe), r)
                        for r in radii])
    assert np.allclose(-field['acc'][:, 0]*radii**2, generic, rtol=1e-8)
    assert np.all(np.isfinite(field['pot']))


def test_binary_and_wrapper_fingerprints(monkeypatch, tmp_path):
    wrapper = ModuleType('agama')
    wrapper.__file__ = str(tmp_path / '__init__.py')
    wrapper.__version__ = 'test'
    binary = ModuleType('agama.agama')
    binary.__file__ = str(tmp_path / 'agama.so')
    Path(wrapper.__file__).write_text('mock wrapper')
    Path(binary.__file__).write_bytes(b'mock binary, never imported')
    monkeypatch.setitem(sys.modules, 'agama', wrapper)
    monkeypatch.setitem(sys.modules, 'agama.agama', binary)
    identity = check.module_identity(wrapper)
    assert identity['binary_fingerprint_present']
    assert identity['version'] == 'test'
    entries = {entry['path']: entry['sha256'] for entry in identity['files']}
    assert entries[binary.__file__] == check.sha256_file(binary.__file__)
    assert entries[wrapper.__file__] != entries[binary.__file__]


def test_smoke_test_rejects_wrong_force_units(tmp_path):
    fake = FakeAgama()
    original = fake.Potential

    def wrong_units(**settings):
        potential = original(**settings)
        potential.bias = 1.
        return potential

    fake.Potential = wrong_units
    with pytest.raises(ValueError, match='G=1'):
        check.agama_self_test(fake, tmp_path)
