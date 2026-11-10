"""Inclination fan driver: plan, exactness of the per-inclination constraints, fan bookkeeping.

No AGAMA, docker or cloud. The fan/diagnose/benchmark code paths run against a fake AGAMA whose
Target reproduces the two properties the scheme relies on: an aperture mass depends on the
PROJECTED axis ratio only, and the observed datacube is recovered by the inverse rotation. A
fake that pairs the wrong density with the wrong beta therefore fails these tests.
"""
import json
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'py/run_incl_fan.py'
spec = importlib.util.spec_from_file_location('run_incl_fan', DRIVER)
fan = importlib.util.module_from_spec(spec)
sys.modules['run_incl_fan'] = fan
spec.loader.exec_module(fan)

MODEL = dict(Q=0.98, gh=0.0, rh=20.0, rho0=40.0)
GRID = [70.0, 80.0, 90.0]


# --------------------------------------------------------------------------
# Fake AGAMA
# --------------------------------------------------------------------------
class FakeDensity:
    def __init__(self, axis_ratio_z=None, parts=()):
        self.axisRatioZ = axis_ratio_z
        self.parts = list(parts)

    def totalMass(self):
        return 2.0

    def sample(self, count, potential=None):
        self.sampled = (count, potential)
        return [numpy.tile(numpy.array([self.axisRatioZ, 0., 0., 0., 1., 0.]), (count, 1))], None


class FakePotential:
    def __init__(self, density, lmax, gridSizeR):
        self.stars = next((p for p in density.parts if p.axisRatioZ is not None), None)
        self.lmax, self.ngrid = lmax, gridSizeR

    def force(self, points):
        points = numpy.asarray(points, dtype=float)
        radius = numpy.linalg.norm(points, axis=1)[:, None]
        return -points / radius**3 * (1.0 + 0.1 * self.stars.axisRatioZ)

    def Tcirc(self, ic):
        return numpy.ones(len(ic))


class FakeTarget:
    """Aperture masses follow the projected axis ratio; the datacube inverts the rotation."""
    NBINS = 4
    NAPER = 25

    def __init__(self, kind, beta=0.0):
        self.kind, self.beta = kind, beta
        self.size = 8 if kind == 'density' else self.NAPER * self.NBINS

    def __len__(self):
        return self.size

    def __call__(self, argument):
        if isinstance(argument, FakeDensity):
            if self.kind == 'density':      # intrinsic grid: no projection involved
                return numpy.linspace(1.0, 2.0, self.size) * argument.axisRatioZ
            q = fan.projected_ratio(argument.axisRatioZ, math.degrees(self.beta))
            return numpy.array([q**(k + 1) for k in range(self.NAPER)])
        points, weights = argument
        points = numpy.asarray(points, dtype=float)
        line_of_sight = -points[:, 4] * math.sin(self.beta) + points[:, 5] * math.cos(self.beta)
        sky = points[:, 1] * math.cos(self.beta) + points[:, 2] * math.sin(self.beta)
        cube = numpy.zeros((self.NAPER, self.NBINS))
        for index, (value, position, weight) in enumerate(zip(line_of_sight, sky, weights)):
            cube[index % self.NAPER, index % self.NBINS] += weight * (value + position)
        return cube.reshape(-1)


class FakeDataset:
    def __init__(self, target, density, beta):
        self.target, self.density, self.beta = target, density, beta
        self.num_bsplines = FakeTarget.NBINS
        self.cons_val = numpy.linspace(1.0, 2.0, len(target))
        self.cons_err = numpy.full(len(target), 0.1)


class FakeSchwarzlib:
    @staticmethod
    def DensityDataset(density, tolerance, **params):
        return FakeDataset(FakeTarget('density'), density, None)

    @staticmethod
    def KinemDatasetGH(density, tolerance, ghm_val, ghm_err, apertures, gridx, gridy, **params):
        return FakeDataset(FakeTarget('losvd', params['beta']), density, params['beta'])

    @staticmethod
    def ghMomentsErrors(degree, gridv, values, errors, ghorder):
        moments = numpy.column_stack([numpy.asarray(values).sum(axis=1)] * (ghorder + 4))
        spread = numpy.column_stack([numpy.asarray(errors).sum(axis=1)] * (ghorder + 4))
        return moments, spread + 1.0


class FakeThreads:
    def __init__(self, count):
        self.count = count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeAgama:
    schwarzlib = FakeSchwarzlib

    def __init__(self):
        self.orbit_calls, self.seeds, self.threads = [], [], []

    def Density(self, *parts, **kwargs):
        if parts:
            return FakeDensity(parts=parts)
        return FakeDensity(axis_ratio_z=kwargs.get('axisRatioZ', kwargs.get('axisratioz')))

    def Potential(self, type, density, lmax, mmax, gridSizeR):
        return FakePotential(density, lmax, gridSizeR)

    def Target(self, apertures, gridx, gridy, **params):
        return FakeTarget('losvd', params['beta'])

    def ghMoments(self, degree, gridv, matrix, ghorder):
        return numpy.column_stack([numpy.asarray(matrix).sum(axis=1)] * 10)

    def setNumThreads(self, count):
        self.threads.append(count)
        return FakeThreads(count)

    def setRandomSeed(self, seed):
        self.seeds.append(int(seed))

    def orbit(self, potential, ic, time, Omega, targets):
        self.orbit_calls.append(dict(potential=potential, targets=list(targets), n=len(ic)))
        # The i-th matrix is tagged with i, so the fit can report which matrix it received.
        return [numpy.full((len(ic), len(t)), float(index))
                for index, t in enumerate(targets)]


class FakeModule:
    """The harness attributes run_incl_fan reads, plus a recording solve_orbit_library."""

    def __init__(self, incl=80.0, n_stars=12):
        self.agama = FakeAgama()
        self.incl = incl
        self.q_ap = fan.Q_AP
        self.Sersic_m, self.massSt, self.scaleRst = 0.8, 14.0, 0.68
        self.densityParams = dict(type='DensitySphHarm', lmax=4)
        self.kinemParams2 = dict(type='LOSVD', alpha=0.0, beta=math.radians(incl), gamma=0.1)
        self.sectAPP = [numpy.zeros((3, 2))] * 25
        self.gridx = self.gridy = numpy.linspace(-1, 1, 5)
        self.gridv = numpy.linspace(-25, 25, 51)
        self.degree, self.ghorder, self.n_boot = 2, 6, 3
        self.ind = (1, 2, 6, 7, 8, 9)
        self.ghm_val_P21 = numpy.ones((25, 10))
        self.ghm_err_P21 = numpy.ones((25, 10))
        self.intTime, self.alphah, self.betah = 100.0, 2.0, 3
        self.EXP_ID, self.GEOM_HASH = 'd1_nb250_gh0_ser0', 'geom1234'
        self.hostname_proc = f'testhost_{self.EXP_ID}_p0'
        self.SAVE_ORBLIB = self.REUSE_ORBLIB = False
        self.orblib_store = None
        self.POT_LMAX, self.POT_NGRID = 4, 23
        self.solves = []
        self.xy_a2vleP_P21 = numpy.zeros(n_stars, dtype=[('x', 'f8'), ('y_', 'f8'), ('a2', 'f8'),
                                                         ('vl', 'f8'), ('err_v', 'f8'), ('prob', 'f8')])
        self.xy_a2vleP_P21['x'] = numpy.linspace(-1, 1, n_stars)
        self.xy_a2vleP_P21['y_'] = numpy.linspace(0.5, -0.5, n_stars)
        self.xy_a2vleP_P21['vl'] = numpy.linspace(-3, 3, n_stars)
        self.prob_P21 = numpy.ones(n_stars)
        self.err_P21 = numpy.full(n_stars, 0.5)
        self.datacube_P21 = self._observed_cube()
        self.ghm_moments_P21 = self.agama.ghMoments(degree=2, gridv=self.gridv,
                                                    matrix=self.datacube_P21, ghorder=6)

    def _observed_cube(self):
        target = fan.aperture_target(self, self.incl)
        xv = numpy.zeros((len(self.xy_a2vleP_P21), 6))
        beta = math.radians(self.incl)
        xv[:, 0] = self.xy_a2vleP_P21['x']
        xv[:, 1] = self.xy_a2vleP_P21['y_'] * math.cos(beta)
        xv[:, 2] = self.xy_a2vleP_P21['y_'] * math.sin(beta)
        xv[:, 4] = -self.xy_a2vleP_P21['vl'] * math.sin(beta)
        xv[:, 5] = self.xy_a2vleP_P21['vl'] * math.cos(beta)
        return target((xv, self.prob_P21)).reshape(len(self.sectAPP), -1)

    def solve_orbit_library(self, Upsilon, mats, datasets, rhs, pen_cons, pen_reg, mult):
        tag = float(numpy.asarray(mats[1])[0, 0])
        beta_deg = math.degrees(datasets[1].beta)
        axis = datasets[1].density.axisRatioZ
        self.solves.append(dict(tag=tag, beta_deg=beta_deg, axis_ratio_z=axis,
                                upsilon=float(Upsilon), density_tag=float(numpy.asarray(mats[0])[0, 0])))
        target_upsilon = 0.4 + 0.002 * beta_deg
        return (Upsilon - target_upsilon)**2 + 0.01 * tag, 0.01


def plan(tmp_path, grid=GRID, reference=None, mode='fan', **kwargs):
    argv = ['--incl-grid', ','.join(f'{x:g}' for x in grid), f'--{mode}']
    for key, value in MODEL.items():
        argv += [f'--{key}', repr(value)]
    if reference is not None:
        argv += ['--reference', repr(reference)]
    for key, value in kwargs.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return fan.build_plan(fan.parse_args(argv))


# --------------------------------------------------------------------------
# Geometry and plan
# --------------------------------------------------------------------------
def test_deprojection_is_inverted_by_the_projection_and_rejects_inadmissible_inclinations():
    for incl in (70.0, 75.5, 90.0):
        assert fan.projected_ratio(fan.axis_ratio_z(incl), incl) == pytest.approx(fan.Q_AP, abs=1e-12)
    with pytest.raises(fan.ConfigError):
        fan.axis_ratio_z(40.0)          # cos incl > q_ap
    table = fan.shape_table([70.0, 80.0, 90.0], 80.0)
    assert table[1]['ax_rz_rel_diff'] == pytest.approx(0.0, abs=1e-15)
    assert table[1]['q_proj_frozen'] == pytest.approx(fan.Q_AP, abs=1e-12)
    # The frozen shape misprojects the isophotes away from the reference, and more so further away.
    assert abs(table[0]['q_proj_rel_diff']) > abs(table[2]['q_proj_rel_diff']) > 0


def model_argv(**overrides):
    values = dict(MODEL, **overrides)
    return [x for key in fan.PARAM_KEYS for x in (f'--{key}', repr(values[key]))]


@pytest.mark.parametrize('argv, match', [
    (model_argv() + ['--incl-grid', '60,90'], 'within'),
    (model_argv() + ['--incl-grid', '90,70'], 'increasing'),
    (model_argv() + ['--incl-grid', '70,nan'], 'finite'),
    (model_argv() + ['--incl-grid', '70,90', '--reference', '80'], 'one of'),
    (model_argv(rh=100.0), 'bounds_original'),
    (model_argv(Q=0.5) + ['--Q1'], 'Q=1'),
    (['--Q', '1'], 'all of'),
    ([], 'no model'),
    (model_argv() + ['--target-counts', '9', '--benchmark'], 'within'),
])
def test_bad_plans_are_rejected(argv, match):
    with pytest.raises(fan.ConfigError, match=match):
        fan.build_plan(fan.parse_args(argv))


def test_reference_defaults_to_the_middle_and_only_its_member_is_exact(tmp_path):
    assert fan.choose_reference([70.0, 75.0, 90.0]) == 75.0
    assert fan.choose_reference([70.0, 90.0]) == 90.0          # tie resolved towards edge-on
    built = plan(tmp_path)
    assert built['reference'] == 80.0 and built['inclinations'] == GRID
    assert built['cost']['fan_s'] is None and built['cost']['speedup'] is None
    measured = fan.estimate_cost(5, orbit_s=400.0, solve_s=11.0, accum_frac=0.1)
    assert measured['separate_s'] == pytest.approx(2055.0)
    assert measured['fan_s'] == pytest.approx(400.0 * 1.4 + 55.0)
    assert measured['speedup'] > 3
    assert fan.matrix_bytes(100000, 25, 52) == 520000000


def test_dry_run_is_stdlib_only_and_creates_nothing(tmp_path):
    result = subprocess.run([sys.executable, '-S', str(DRIVER), '--dry-run', '--out-dir',
                             str(tmp_path / 'unused'), '--Q', '1', '--gh', '0', '--rh', '20',
                             '--rho0', '40'], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'reference 80 deg' in result.stdout and 'diagnostic (Q33)' in result.stdout
    assert not (tmp_path / 'unused').exists() and not list(tmp_path.iterdir())


# --------------------------------------------------------------------------
# Exactness of the per-inclination constraints
# --------------------------------------------------------------------------
def test_each_member_gets_the_exactly_deprojected_density_and_its_own_projection():
    mod = FakeModule()
    for incl in GRID:
        density_dataset, kinem_dataset = fan.build_datasets(mod, incl)
        assert kinem_dataset.beta == pytest.approx(math.radians(incl))
        for dataset in (density_dataset, kinem_dataset):
            assert dataset.density.axisRatioZ == pytest.approx(fan.axis_ratio_z(incl))
    # The harness inclination must not leak into a member built for another one.
    assert fan.kinem_params(mod, 70.0)['beta'] == pytest.approx(math.radians(70.0))
    assert mod.kinemParams2['beta'] == pytest.approx(math.radians(80.0))


def test_aperture_mass_control_is_zero_only_when_shape_and_projection_match():
    mod = FakeModule()
    report = fan.check_aperture_masses(mod, GRID, 80.0)
    control = [report['per_incl'][f'{i:g}']['exact_vs_reference_incl']['max'] for i in GRID]
    frozen = [report['per_incl'][f'{i:g}']['frozen_vs_exact']['max'] for i in GRID]
    assert max(control) < 1e-12          # exact deprojection => incl-invariant aperture masses
    assert frozen[GRID.index(80.0)] < 1e-12 and min(frozen[0], frozen[-1]) > 1e-3
    assert report['n_apertures'] == len(mod.sectAPP)


def test_observed_constraints_do_not_depend_on_inclination():
    mod = FakeModule()
    report = fan.check_observables(mod, GRID)
    assert all(entry['datacube']['max'] < 1e-12 and entry['gh_moments']['max'] < 1e-12
               for entry in report['per_incl'].values())
    boot = fan.check_observables(mod, [70.0], bootstrap=True)['per_incl']['70']
    assert 'gh_val' in boot and 'gh_err' in boot


def test_potential_check_measures_only_the_stellar_shape_difference():
    mod = FakeModule()
    report = fan.check_potential(mod, [MODEL], GRID, 80.0, mod.alphah, mod.betah)
    entry = report['models'][0]['per_incl']
    assert entry['80']['force']['apertures']['max'] < 1e-12
    assert entry['70']['force']['apertures']['max'] > entry['90']['force']['apertures']['max'] > 0
    assert set(entry['70']['force']) == set(fan.PROBE_REGIONS)
    assert entry['70']['v_circ']['max'] > 0


# --------------------------------------------------------------------------
# Fan bookkeeping
# --------------------------------------------------------------------------
def test_one_integration_feeds_every_inclination_with_its_own_matrix(tmp_path):
    mod, built = FakeModule(), plan(tmp_path, num_orbits=50)
    report = fan.run_fan(mod, built, mod.alphah, mod.betah, str(tmp_path))
    assert len(mod.agama.orbit_calls) == 1                      # one integration for three members
    call = mod.agama.orbit_calls[0]
    assert [t.kind for t in call['targets']] == ['density'] + ['losvd'] * len(GRID)
    assert [round(math.degrees(t.beta), 6) for t in call['targets'][1:]] == GRID
    assert call['potential'].stars.axisRatioZ == pytest.approx(fan.axis_ratio_z(80.0))
    assert mod.agama.seeds == [42] and call['n'] == 50
    # Every member solved with the density matrix and its OWN kinematic matrix.
    pairs = {(round(s['beta_deg'], 6), s['tag']) for s in mod.solves}
    assert pairs == {(incl, float(index)) for index, incl in enumerate(GRID, start=1)}
    assert {s['density_tag'] for s in mod.solves} == {0.0}
    assert all(s['axis_ratio_z'] == pytest.approx(fan.axis_ratio_z(s['beta_deg'])) for s in mod.solves)
    members = report[0]['members']
    assert {k: m['exact'] for k, m in members.items()} == {'70': False, '80': True, '90': False}
    # The fake minimum moves with inclination, so the members are genuinely different fits.
    assert len({round(m['upsilon'], 6) for m in members.values()}) == len(GRID)
    assert all(m['upsilon'] == pytest.approx(0.4 + 0.002 * float(k), abs=2e-3)
               for k, m in members.items())


def test_fan_rows_stay_out_of_the_search_pools_and_record_provenance(tmp_path):
    mod, built = FakeModule(), plan(tmp_path, num_orbits=20)
    fan.run_fan(mod, built, mod.alphah, mod.betah, str(tmp_path))
    files = list(tmp_path.glob('*.txt'))
    assert len(files) == 1 and files[0].name.startswith(fan.HISTORY_PREFIX)
    for glob_pattern in ('4UpsBoTorch_PCA_Sersic_*.txt', 'out_*.txt', '4result_*.txt'):
        assert not list(tmp_path.glob(glob_pattern))
    text = files[0].read_text()
    rows = [line.split() for line in text.splitlines() if not line.startswith('#')]
    assert [row[0] for row in rows] == [f'{incl:.3f}' for incl in GRID]
    assert all(len(row) >= 8 and float(row[1]) == MODEL['Q'] for row in rows)
    assert 'reference_incl=80.000' in text and 'not for pooling' in text
    assert text.count('exact=yes') == 1 and text.count('exact=no') == len(GRID) - 1
    # A second model appends; nothing is overwritten or deduplicated.
    fan.run_fan(mod, built, mod.alphah, mod.betah, str(tmp_path))
    assert len([line for line in files[0].read_text().splitlines()
                if not line.startswith('#')]) == 2 * len(GRID)


def test_benchmark_times_extra_targets_and_checks_bit_identity(tmp_path):
    mod = FakeModule()
    built = plan(tmp_path, mode='benchmark', num_orbits=10, target_counts='1,3')
    report = fan.benchmark(mod, built, mod.alphah, mod.betah)
    assert [run['kinem_targets'] for run in report['runs']] == [1, 3]
    assert report['runs'][0]['first_target_identical'] is None
    assert report['runs'][1]['first_target_identical'] is True
    assert report['runs'][0]['bytes_per_target'] == fan.matrix_bytes(
        10, len(mod.sectAPP), FakeTarget.NBINS, numpy.dtype(float).itemsize)
    assert len(mod.agama.orbit_calls) == 2
    assert report['accum_frac'] is not None and report['scaled']['count'] == len(GRID)


def test_summary_marks_approximate_members(tmp_path):
    mod, built = FakeModule(), plan(tmp_path, num_orbits=20)
    report = dict(plan=built, fan=fan.run_fan(mod, built, mod.alphah, mod.betah, str(tmp_path)))
    path = tmp_path / 'report.json'
    fan.write_report(report, str(path))
    rows = fan.summarize([str(path)])
    assert len(rows) == len(GRID) and sum(row['exact'] for row in rows) == 1
    text = fan.format_summary(rows)
    assert 'E1+E2' in text and text.count('yes') == 1
    assert json.loads(path.read_text())['plan']['reference'] == 80.0


def stub_harness_import(monkeypatch, mod):
    """load_harness imports the harness through importlib; hand it the fake module."""
    import importlib.util as util
    from types import SimpleNamespace
    recorded = {}

    def spec_from_file_location(name, path):
        recorded['path'] = path
        return SimpleNamespace(loader=SimpleNamespace(
            exec_module=lambda module: recorded.update(argv=list(sys.argv))))

    monkeypatch.setattr(util, 'spec_from_file_location', spec_from_file_location)
    monkeypatch.setattr(util, 'module_from_spec', lambda spec: mod)
    return recorded


def test_harness_is_imported_at_the_reference_inclination_without_shared_storage(monkeypatch):
    mod = FakeModule()
    recorded = stub_harness_import(monkeypatch, mod)
    argv = fan.parse_args(model_argv() + ['--fan', '--threads', '4'])
    before = list(sys.argv)
    assert fan.load_harness(argv, 80.0) is mod
    assert recorded['path'] == fan.rsm.EXP_SCRIPT and sys.argv == before
    assert recorded['argv'][recorded['argv'].index('--incl') + 1] == repr(80.0)
    assert '--no-resume' in recorded['argv'] and '--save-orblib' not in recorded['argv']
    assert mod.orblib_store is None and mod.completed_point(dict(MODEL)) is False
    with pytest.raises(fan.ConfigError, match='inclination'):
        fan.load_harness(argv, 90.0)       # harness built for another inclination
    mod.SAVE_ORBLIB = True
    with pytest.raises(fan.ConfigError, match='storage'):
        fan.load_harness(argv, 80.0)


def test_main_reports_configuration_errors_without_touching_agama(tmp_path, capsys):
    assert fan.main(['--fan', '--incl-grid', '50,90', '--Q', '1', '--gh', '0',
                     '--rh', '20', '--rho0', '40']) == 2
    assert 'incl fan:' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
