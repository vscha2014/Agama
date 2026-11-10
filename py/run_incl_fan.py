#!/usr/bin/env python3
"""Inclination fan: one orbit integration shared by several LOSVD projections (diagnostic).

A stored orbit library cannot be reused at another inclination: TargetLOSVD applies the
(alpha,beta,gamma) rotation inside addPoint during integration and finalizeDatacube folds the
cube into apertures, so the projection is baked into matrix_kinem (and no trajectories are
recorded). What can be shared is everything before the projection -- the potential, the orbit
initial conditions and the integration, i.e. ~97 % of the cost of a model (orbit_s ~ 400 s
against ~11 s for the whole Upsilon search). This driver therefore integrates once and passes
several LOSVD targets, one per inclination, to a single agama.orbit call.

The per-inclination observational side is rebuilt EXACTLY and for free:
  * ghm_val/ghm_err do not depend on incl at all (doc/ai/07_incl_penalty_analysis.md, section 2;
    checked numerically here by --diagnose, check "observables");
  * aperture masses in KinemDatasetGH and cons_val in DensityDataset are rebuilt from the exact
    deprojected Sersic density axRZst(i) = sqrt(q_ap^2 - cos^2 i)/sin i of each inclination.
Only two approximations remain, and both are measurable:
  E1  the stellar quadrupole in Multipole(rho_star + rho_halo) is that of the reference
      inclination (stars are ~10 % of the enclosed mass, and only their flattening changes);
  E2  the orbit initial conditions are sampled from the reference stellar density -- a choice of
      basis, not a constraint, since the orbit weights are free and the constraints are exact.
--diagnose bounds E1 without integrating a single orbit; --benchmark measures the marginal cost
and memory of an extra LOSVD target; the error of E1+E2 on the penalty needs the paired pilot
(exact vs fan at the same IC seeds) described in doc/ai/STATUS.md.

The fan is a DIAGNOSTIC accelerator: only its reference member is an exact model. Rows go to
incl_fan_*.txt, which no PCA/TuRBO pool glob matches, and are never pooled with search history.
Publishing fan numbers requires the PI decision asked in doc/ai/questions_for_pi.md (Q33).

Modes: --dry-run / --summarize (stdlib only), --diagnose (AGAMA, no orbits),
--benchmark (AGAMA, few orbits), --fan (AGAMA, one full integration per model).
"""
import argparse
import datetime
import glob
import json
import math
import os
import socket
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_single_model as rsm  # noqa: E402  (stdlib-only: EXP_SCRIPT, script_bounds, ...)

SCHEMA = 'incl-fan-v1'
Q_AP = 1 - 0.31                     # Wang et al. 2019 ellipticity, as in the harness
FAN_INCL_BOUNDS = (70.0, 90.0)      # user decision 2026-11-10: only 70-90 deg is in scope
PARAM_KEYS = rsm.PARAM_KEYS         # ('Q', 'gh', 'rh', 'rho0')
HISTORY_PREFIX = 'incl_fan'         # matched by no pool glob (4UpsBoTorch_* / out_*)
DEFAULT_GRID = (70.0, 75.0, 80.0, 85.0, 90.0)
# Probe regions for the potential comparison (kpc): the stellar scale radius, the aperture
# coverage (max_r = 2.1) and an outer margin.
PROBE_REGIONS = dict(inner=(0.0, 0.7), apertures=(0.0, 2.1), outer=(0.0, 5.0))
# Cost-model placeholders; --benchmark replaces accum_frac with a measured value.
COST_DEFAULTS = dict(orbit_s=400.0, solve_s=11.0)


class ConfigError(Exception):
    pass


# --------------------------------------------------------------------------
# Geometry of the deprojection (pure arithmetic, no AGAMA)
# --------------------------------------------------------------------------
def axis_ratio_z(incl, q_ap=Q_AP):
    """Intrinsic axis ratio of the exactly deprojected Sersic component (harness formula)."""
    beta = math.radians(incl)
    inner = q_ap**2 - math.cos(beta)**2
    if inner <= 0 or math.sin(beta) <= 0:
        raise ConfigError(f'incl={incl:g} is inadmissible: cos(incl) must stay below q_ap={q_ap:g}')
    return math.sqrt(inner) / math.sin(beta)


def projected_ratio(ax_rz, incl):
    """Projected axis ratio of a density stratified on ellipsoids with axisRatioZ=ax_rz."""
    beta = math.radians(incl)
    return math.sqrt(ax_rz**2 * math.sin(beta)**2 + math.cos(beta)**2)


def shape_table(inclinations, reference, q_ap=Q_AP):
    """Check D0: how far the frozen reference shape is from the exact one, per inclination."""
    ax_ref = axis_ratio_z(reference, q_ap)
    rows = []
    for incl in inclinations:
        ax_exact = axis_ratio_z(incl, q_ap)
        q_frozen = projected_ratio(ax_ref, incl)
        rows.append(dict(incl=incl, ax_rz_exact=ax_exact, ax_rz_reference=ax_ref,
                         ax_rz_rel_diff=(ax_ref - ax_exact) / ax_exact,
                         q_proj_exact=q_ap, q_proj_frozen=q_frozen,
                         q_proj_rel_diff=(q_frozen - q_ap) / q_ap))
    return rows


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------
def parse_grid(spec):
    try:
        values = [float(x) for x in str(spec).split(',') if x.strip()]
    except ValueError:
        raise ConfigError('--incl-grid: expected comma-separated numbers')
    lo, hi = FAN_INCL_BOUNDS
    if not values:
        raise ConfigError('--incl-grid: no values')
    if any(not math.isfinite(x) or not lo <= x <= hi for x in values):
        raise ConfigError(f'--incl-grid: finite values within [{lo:g},{hi:g}] required')
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ConfigError('--incl-grid: values must be strictly increasing')
    for value in values:
        axis_ratio_z(value)
    return values


def parse_counts(spec, available):
    try:
        counts = [int(x) for x in str(spec).split(',') if x.strip()]
    except ValueError:
        raise ConfigError('--target-counts: expected comma-separated integers')
    if not counts or any(c < 1 or c > available for c in counts):
        raise ConfigError(f'--target-counts: values within [1,{available}] required')
    if any(a >= b for a, b in zip(counts, counts[1:])):
        raise ConfigError('--target-counts: values must be strictly increasing')
    return counts


def choose_reference(inclinations, explicit=None):
    """The reference inclination defines the orbit basis; its own member stays exact."""
    if explicit is None:
        middle = 0.5 * (inclinations[0] + inclinations[-1])
        return min(inclinations, key=lambda x: (abs(x - middle), -x))
    if not any(abs(explicit - x) <= 1e-9 for x in inclinations):
        raise ConfigError('--reference must be one of the --incl-grid values')
    return next(x for x in inclinations if abs(explicit - x) <= 1e-9)


def read_models(args, bounds):
    rows = []
    if args.model_file:
        with open(args.model_file) as stream:
            for number, line in enumerate(stream, 1):
                line = line.split('#', 1)[0].strip()
                if not line:
                    continue
                values = line.split()
                if len(values) < len(PARAM_KEYS):
                    raise ConfigError(f'{args.model_file}:{number}: expected "Q gh rh rho0"')
                try:
                    rows.append({k: float(v) for k, v in zip(PARAM_KEYS, values)})
                except ValueError:
                    raise ConfigError(f'{args.model_file}:{number}: non-numeric values')
    explicit = {k: getattr(args, k) for k in PARAM_KEYS if getattr(args, k) is not None}
    if explicit:
        if len(explicit) != len(PARAM_KEYS):
            raise ConfigError('pass all of --Q/--gh/--rh/--rho0, or use --model-file')
        rows.append(explicit)
    if not rows:
        raise ConfigError('no model: pass --Q/--gh/--rh/--rho0 or --model-file '
                          '(no result numbers live in this repository)')
    for row in rows:
        outside = rsm.bounds_violations(row, bounds)
        if outside:
            raise ConfigError('outside bounds_original: ' + '; '.join(outside))
        if args.Q1 and row['Q'] != 1.0:
            raise ConfigError('--Q1 requires Q=1 in every model row')
    return rows


def matrix_bytes(num_orbits, num_aper, num_bsplines, itemsize=4):
    """In-memory size of one LOSVD target matrix (AGAMA returns StorageNumT = float32)."""
    return int(num_orbits) * int(num_aper) * int(num_bsplines) * int(itemsize)


def estimate_cost(count, orbit_s=None, solve_s=None, accum_frac=None):
    """Cost of a K-member fan against K independent models, given the accumulation fraction f
    (the share of integration time spent accumulating one LOSVD target; measured by --benchmark)."""
    orbit_s = COST_DEFAULTS['orbit_s'] if orbit_s is None else float(orbit_s)
    solve_s = COST_DEFAULTS['solve_s'] if solve_s is None else float(solve_s)
    separate = count * (orbit_s + solve_s)
    if accum_frac is None:
        return dict(count=count, orbit_s=orbit_s, solve_s=solve_s, accum_frac=None,
                    separate_s=separate, fan_s=None, speedup=None,
                    note='accum_frac unmeasured; run --benchmark before quoting a speedup')
    fan = orbit_s * (1.0 + (count - 1) * float(accum_frac)) + count * solve_s
    return dict(count=count, orbit_s=orbit_s, solve_s=solve_s, accum_frac=float(accum_frac),
                separate_s=separate, fan_s=fan, speedup=(separate / fan) if fan > 0 else None)


def build_plan(args, bounds=None):
    bounds = rsm.script_bounds() if bounds is None else bounds
    inclinations = parse_grid(args.incl_grid)
    reference = choose_reference(inclinations, args.reference)
    plan = dict(schema=SCHEMA, mode=args.mode, inclinations=inclinations, reference=reference,
                models=read_models(args, bounds), bounds=bounds, ic_seed=args.ic_seed,
                threads=args.threads, num_orbits=args.num_orbits, l24=bool(args.l24),
                Q1=bool(args.Q1), shape=shape_table(inclinations, reference),
                cost=estimate_cost(len(inclinations), accum_frac=args.accum_frac),
                memory=dict(per_target_bytes=matrix_bytes(args.num_orbits, 25, 52),
                            extra_targets=len(inclinations) - 1))
    if args.mode == 'benchmark':
        plan['target_counts'] = parse_counts(args.target_counts, len(inclinations))
    return plan


def format_plan(plan):
    lines = [f"incl fan plan ({plan['mode']}): {len(plan['inclinations'])} inclinations, "
             f"reference {plan['reference']:g} deg, {len(plan['models'])} model(s)",
             'shape (frozen reference vs exact deprojection):']
    for row in plan['shape']:
        mark = ' <- reference (exact)' if abs(row['incl'] - plan['reference']) <= 1e-9 else ''
        lines.append(f"  incl={row['incl']:5.1f}  axRZ_exact={row['ax_rz_exact']:.4f}  "
                     f"axRZ_ref={row['ax_rz_reference']:.4f} ({row['ax_rz_rel_diff']:+.2%})  "
                     f"q_proj_frozen={row['q_proj_frozen']:.4f} ({row['q_proj_rel_diff']:+.2%}){mark}")
    cost = plan['cost']
    if cost['fan_s'] is None:
        lines.append(f"cost: {cost['separate_s']:.0f} s for {cost['count']} separate models; "
                     f"fan unknown ({cost['note']})")
    else:
        lines.append(f"cost: separate {cost['separate_s']:.0f} s vs fan {cost['fan_s']:.0f} s "
                     f"(f={cost['accum_frac']:.3f}, speedup {cost['speedup']:.2f}x)")
    lines.append(f"memory: +{plan['memory']['per_target_bytes']/1e9:.2f} GB per extra LOSVD target "
                 f"(float32, {plan['num_orbits']} orbits, 25 apertures x 52 b-splines), "
                 f"{plan['memory']['extra_targets']} extra")
    lines.append('only the reference member is an exact model; fan rows are diagnostic (Q33)')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
# AGAMA-side helpers
# --------------------------------------------------------------------------
def kinem_params(mod, incl):
    return dict(mod.kinemParams2, beta=math.radians(incl))


def stellar_density(mod, incl):
    return mod.agama.Density(type='Sersic', sersicIndex=mod.Sersic_m, mass=mod.massSt,
                             scaleRadius=mod.scaleRst, axisRatioZ=axis_ratio_z(incl, mod.q_ap))


def aperture_target(mod, incl):
    return mod.agama.Target(apertures=mod.sectAPP, gridx=mod.gridx, gridy=mod.gridy,
                            **kinem_params(mod, incl))


def build_datasets(mod, incl, density=None):
    """Datasets of one inclination with EXACT constraints (observations do not depend on incl)."""
    density = stellar_density(mod, incl) if density is None else density
    return [mod.agama.schwarzlib.DensityDataset(density=density, tolerance=0.0, **mod.densityParams),
            mod.agama.schwarzlib.KinemDatasetGH(
                density=density, tolerance=0.01,
                ghm_val=mod.ghm_val_P21[:, mod.ind], ghm_err=mod.ghm_err_P21[:, mod.ind],
                apertures=mod.sectAPP, gridx=mod.gridx, gridy=mod.gridy, **kinem_params(mod, incl))]


def halo_density(mod, model, alphah, betah):
    return mod.agama.Density(type='spheroid', alpha=alphah, beta=betah, gamma=model['gh'],
                             axisratioz=model['Q'], densitynorm=model['rho0'],
                             scaleradius=model['rh'], outercutoffradius=55.0, cutoffstrength=2.5)


def build_potential(mod, density_stars, density_halo):
    """The harness recipe; the diagnostic callback only hands out the lmax=4 baseline."""
    return mod.agama.Potential(type='Multipole',
                               density=mod.agama.Density(density_stars, density_halo),
                               lmax=int(getattr(mod, 'POT_LMAX', 4)), mmax=0,
                               gridSizeR=int(getattr(mod, 'POT_NGRID', 23)))


def _deviation(numpy, measured, reference):
    measured = numpy.asarray(measured, dtype=float)
    reference = numpy.asarray(reference, dtype=float)
    if measured.shape != reference.shape or not (numpy.all(numpy.isfinite(measured))
                                                 and numpy.all(numpy.isfinite(reference))):
        raise ValueError('invalid arrays in a diagnostic comparison')
    scale = numpy.abs(reference)
    use = scale > 0
    if not numpy.any(use):
        raise ValueError('reference values are all zero')
    relative = numpy.abs(measured[use] - reference[use]) / scale[use]
    return dict(n=int(use.sum()), rms=float(numpy.sqrt(numpy.mean(relative**2))),
                max=float(relative.max()))


def check_observables(mod, inclinations, bootstrap=False):
    """Check D4: the observational constraints do not depend on incl (doc 07, section 2).

    The data are built as the inverse rotation by beta and the Target applies the forward one,
    so the two cancel. Recomputed here from the harness's own star table for every inclination
    and compared with the harness values -- the whole scheme rests on this invariance."""
    import numpy
    stars, prob = mod.xy_a2vleP_P21, mod.prob_P21
    report = dict(harness_incl=float(mod.incl), bootstrap=bool(bootstrap), per_incl={})
    for incl in inclinations:
        beta = math.radians(incl)
        xv = numpy.zeros((len(stars), 6), dtype='float64')
        xv[:, 0] = stars['x']
        xv[:, 1] = stars['y_'] * math.cos(beta)
        xv[:, 2] = stars['y_'] * math.sin(beta)
        xv[:, 4] = -stars['vl'] * math.sin(beta)
        xv[:, 5] = stars['vl'] * math.cos(beta)
        target = aperture_target(mod, incl)
        cube = target((xv, prob)).reshape(len(mod.sectAPP), -1)
        moments = mod.agama.ghMoments(degree=mod.degree, gridv=mod.gridv, matrix=cube,
                                      ghorder=mod.ghorder)
        entry = dict(datacube=_deviation(numpy, cube, mod.datacube_P21),
                     gh_moments=_deviation(numpy, moments[:, mod.ind],
                                           mod.ghm_moments_P21[:, mod.ind]))
        if bootstrap:
            boot = numpy.vstack([xv] * mod.n_boot)
            numpy.random.seed(42)   # the harness's fixed observation-error realisation
            diff = numpy.hstack([mod.err_P21] * mod.n_boot) * numpy.random.normal(size=len(boot))
            boot[:, 4] += -diff * math.sin(beta)
            boot[:, 5] += diff * math.cos(beta)
            cubes = numpy.zeros((len(mod.sectAPP) * mod.n_boot, cube.shape[1]), dtype='float64')
            for i in range(mod.n_boot):
                cubes[i*len(mod.sectAPP):(i+1)*len(mod.sectAPP), :] = target(
                    (boot[i*len(xv):(i+1)*len(xv), :], prob)).reshape(len(mod.sectAPP), -1)
            errors = numpy.std(cubes.reshape(mod.n_boot, -1), axis=0).reshape(cube.shape)
            val, err = mod.agama.schwarzlib.ghMomentsErrors(
                degree=mod.degree, gridv=mod.gridv, values=cube, errors=errors, ghorder=mod.ghorder)
            entry['gh_val'] = _deviation(numpy, val[:, mod.ind], mod.ghm_val_P21[:, mod.ind])
            entry['gh_err'] = _deviation(numpy, err[:, mod.ind], mod.ghm_err_P21[:, mod.ind])
        report['per_incl'][f'{incl:g}'] = entry
    return report


def check_aperture_masses(mod, inclinations, reference):
    """Check D1: the exact deprojection keeps aperture masses incl-invariant (q_proj == q_ap),
    so the exact-vs-exact column is a pipeline control and the frozen column is the shape error."""
    import numpy
    density_ref = stellar_density(mod, reference)
    exact, frozen = {}, {}
    for incl in inclinations:
        target = aperture_target(mod, incl)
        density = stellar_density(mod, incl)
        exact[incl] = numpy.asarray(target(density)) / density.totalMass()
        frozen[incl] = numpy.asarray(target(density_ref)) / density_ref.totalMass()
    base = exact[reference]
    return dict(n_apertures=len(mod.sectAPP), per_incl={
        f'{incl:g}': dict(exact_vs_reference_incl=_deviation(numpy, exact[incl], base),
                          frozen_vs_exact=_deviation(numpy, frozen[incl], exact[incl]))
        for incl in inclinations})


def check_intrinsic_density(mod, inclinations, reference):
    """Check D2: how different the intrinsic density constraints of the frozen shape would be.
    Informational -- the fan rebuilds this channel exactly for every inclination."""
    import numpy
    base = mod.agama.schwarzlib.DensityDataset(density=stellar_density(mod, reference),
                                               tolerance=0.0, **mod.densityParams).cons_val
    return dict(per_incl={f'{incl:g}': _deviation(
        numpy, mod.agama.schwarzlib.DensityDataset(density=stellar_density(mod, incl),
                                                   tolerance=0.0, **mod.densityParams).cons_val,
        base) for incl in inclinations})


def probe_points(numpy, radii=None, heights=None):
    radii = numpy.geomspace(0.02, PROBE_REGIONS['outer'][1], 24) if radii is None else radii
    heights = numpy.array([0.0, 0.05, 0.2, 0.5, 1.0, 2.0]) if heights is None else heights
    return numpy.array([[r, 0.0, z] for r in radii for z in heights])


def check_potential(mod, models, inclinations, reference, alphah, betah):
    """Bound E1 without orbits: forces and v_circ of Multipole(rho_star_ref + halo) against the
    potential built with the exactly deprojected stars of each inclination."""
    import numpy
    points = probe_points(numpy)
    radii = numpy.linalg.norm(points, axis=1)
    density_ref = stellar_density(mod, reference)
    report = dict(n_points=len(points), pot_lmax=int(getattr(mod, 'POT_LMAX', 4)),
                  pot_ngrid=int(getattr(mod, 'POT_NGRID', 23)),
                  regions={k: list(v) for k, v in PROBE_REGIONS.items()}, models=[])
    for model in models:
        halo = halo_density(mod, model, alphah, betah)
        force_frozen = numpy.asarray(build_potential(mod, density_ref, halo).force(points))
        entry = dict(model=dict(model), per_incl={})
        for incl in inclinations:
            exact = build_potential(mod, stellar_density(mod, incl), halo)
            force_exact = numpy.asarray(exact.force(points))
            norm = numpy.linalg.norm(force_exact, axis=1)
            if not numpy.all(norm > 0) or not numpy.all(numpy.isfinite(force_frozen)):
                raise ValueError('invalid force field in the potential check')
            relative = numpy.linalg.norm(force_frozen - force_exact, axis=1) / norm
            scores = {}
            for name, (lo, hi) in PROBE_REGIONS.items():
                index = numpy.flatnonzero((radii >= lo) & (radii <= hi))
                scores[name] = None if not len(index) else dict(
                    n=int(len(index)), rms=float(numpy.sqrt(numpy.mean(relative[index]**2))),
                    max=float(relative[index].max()))
            # v_circ^2 = |r * F_r| = |sum(F . x)|
            v_frozen = numpy.sqrt(numpy.abs(numpy.sum(force_frozen * points, axis=1)))
            v_exact = numpy.sqrt(numpy.abs(numpy.sum(force_exact * points, axis=1)))
            entry['per_incl'][f'{incl:g}'] = dict(force=scores,
                                                  v_circ=_deviation(numpy, v_frozen, v_exact))
        report['models'].append(entry)
    return report


def diagnose(mod, plan, alphah, betah, bootstrap=False):
    return dict(shape=plan['shape'],
                observables=check_observables(mod, plan['inclinations'], bootstrap),
                aperture_masses=check_aperture_masses(mod, plan['inclinations'], plan['reference']),
                intrinsic_density=check_intrinsic_density(mod, plan['inclinations'], plan['reference']),
                potential=check_potential(mod, plan['models'], plan['inclinations'],
                                          plan['reference'], alphah, betah))


# --------------------------------------------------------------------------
# Integration and fitting
# --------------------------------------------------------------------------
def sample_initial_conditions(mod, density_stars, potential, num_orbits, int_time, ic_seed):
    import numpy
    if ic_seed is not None:
        mod.agama.setRandomSeed(int(ic_seed))
    ic = numpy.vstack((density_stars.sample(int(num_orbits), potential=potential)[0]))
    times = potential.Tcirc(ic) * int_time
    if len(ic) != int(num_orbits) or not numpy.all(numpy.isfinite(ic)) or not numpy.all(times > 0):
        raise ValueError('invalid orbit initial conditions or integration times')
    return ic, times


def integrate(mod, potential, targets, ic, times, threads):
    """One agama.orbit call accumulating every target; trajectories are not requested."""
    import numpy
    start = time.perf_counter()
    with mod.agama.setNumThreads(int(threads)):
        matrices = mod.agama.orbit(potential=potential, ic=ic, time=times, Omega=0.0,
                                   targets=targets)
    seconds = time.perf_counter() - start
    if len(matrices) != len(targets) or any(
            numpy.shape(m) != (len(ic), len(t)) or not numpy.all(numpy.isfinite(m))
            for m, t in zip(matrices, targets)):
        raise ValueError('orbit integration returned invalid target matrices')
    return list(matrices), seconds


def fit(mod, datasets, matrices, regul=1.0, xatol=1e-3, bounds=(0.1, 1.6)):
    """Production-like Upsilon protocol on the full library (no sub-sampling)."""
    import numpy
    from scipy.optimize import minimize_scalar
    num_dof = sum(int(numpy.count_nonzero(d.cons_err > 0)) for d in datasets)
    if num_dof <= 0:
        raise ValueError('no positive-error constraints')
    mult = num_dof**0.5 * 10
    count = len(matrices[0])
    with numpy.errstate(divide='ignore'):
        context = dict(rhs=[d.cons_val / mult for d in datasets],
                       pen_cons=[2 * d.cons_err**-2 for d in datasets],
                       pen_reg=2.0 * regul * numpy.ones(count) * count, mult=mult)
    history, elapsed = [], 0.0

    def objective(upsilon):
        nonlocal elapsed
        penalty, seconds = mod.solve_orbit_library(float(upsilon), matrices, datasets, **context)
        penalty, elapsed = float(penalty), elapsed + seconds
        if not math.isfinite(penalty) or penalty < 0 or penalty >= rsm.FAILED_PENALTY:
            raise ValueError(f'failed penalty {penalty!r}; a sentinel is not a result')
        history.append([float(upsilon), penalty])
        return penalty

    result = minimize_scalar(objective, bounds=bounds, method='bounded',
                             options=dict(xatol=xatol, maxiter=50))
    if not result.success:
        raise ValueError(f'Upsilon search failed: {result.message}')
    return dict(penalty=float(result.fun), upsilon=float(result.x), probes=len(history),
                solve_s=elapsed, history=history,
                settings=dict(xatol=xatol, bounds=list(bounds), regul=regul, subsample=0.0))


def history_path(mod, out_dir, reference):
    return os.path.join(out_dir, f'{HISTORY_PREFIX}_{mod.hostname_proc}_i{reference:.1f}.txt')


def append_rows(path, mod, model, reference, results, note):
    """Harness row format, in a file no pool glob matches; members are marked exact or not."""
    with open(path, 'a') as stream:
        stream.write(f'# {HISTORY_PREFIX}: reference_incl={reference:.3f} exp={mod.EXP_ID} '
                     f'geom={mod.GEOM_HASH} {note}\n')
        stream.write('# fan members at incl != reference carry the E1+E2 approximation '
                     '(shared potential and orbit basis); not for pooling\n')
        stream.write(f'# Server: {mod.hostname_proc}\n')
        for incl, result in sorted(results.items()):
            exact = abs(incl - reference) <= 1e-9
            stream.write(f"# member incl={incl:.3f} exact={'yes' if exact else 'no'} "
                         f"probes={result['probes']} solve_s={result['solve_s']:.3f}\n")
            stream.write(f"{incl:0.3f} {model['Q']:0.15f} {model['gh']:0.15f} {model['rh']:0.15f} "
                         f"{model['rho0']:0.15f} {result['upsilon']:0.15f} "
                         f"{result['penalty']:0.15f} {datetime.datetime.now()}\n")


def run_fan(mod, plan, alphah, betah, out_dir):
    """One integration per model, shared by every inclination of the grid."""
    reference, inclinations = plan['reference'], plan['inclinations']
    datasets = {incl: build_datasets(mod, incl) for incl in inclinations}
    # One shared intrinsic density target (the constraints differ per inclination, the
    # orbit matrix does not) plus one LOSVD target per inclination.
    targets = [datasets[reference][0].target] + [datasets[incl][1].target for incl in inclinations]
    path = history_path(mod, out_dir, reference)
    density_ref = stellar_density(mod, reference)
    report = []
    for model in plan['models']:
        potential = build_potential(mod, density_ref, halo_density(mod, model, alphah, betah))
        ic, times = sample_initial_conditions(mod, density_ref, potential, plan['num_orbits'],
                                              mod.intTime, plan['ic_seed'])
        matrices, orbit_s = integrate(mod, potential, targets, ic, times, plan['threads'] or 1)
        results = {}
        for index, incl in enumerate(inclinations, start=1):
            results[incl] = fit(mod, datasets[incl], [matrices[0], matrices[index]])
            print(f"[fan] incl={incl:5.1f} penalty={results[incl]['penalty']:.6f} "
                  f"upsilon={results[incl]['upsilon']:.4f}"
                  f"{' (exact member)' if abs(incl - reference) <= 1e-9 else ''}", flush=True)
        append_rows(path, mod, model, reference, results,
                    note=f"ic_seed={plan['ic_seed']} orbits={plan['num_orbits']} "
                         f"orbit_s={orbit_s:.1f}")
        report.append(dict(model=dict(model), orbit_s=orbit_s, history=path,
                           members={f'{incl:g}': dict(
                               penalty=r['penalty'], upsilon=r['upsilon'], probes=r['probes'],
                               solve_s=r['solve_s'], exact=bool(abs(incl - reference) <= 1e-9))
                               for incl, r in results.items()}))
        del matrices
    return report


def benchmark(mod, plan, alphah, betah):
    """Marginal cost and memory of an extra LOSVD target; the first target must stay identical."""
    import numpy
    model = plan['models'][0]
    density_ref = stellar_density(mod, plan['reference'])
    potential = build_potential(mod, density_ref, halo_density(mod, model, alphah, betah))
    ic, times = sample_initial_conditions(mod, density_ref, potential, plan['num_orbits'],
                                          mod.intTime, plan['ic_seed'])
    datasets = {incl: build_datasets(mod, incl) for incl in plan['inclinations']}
    density_target = datasets[plan['reference']][0].target
    num_bsplines = datasets[plan['reference']][1].num_bsplines
    runs, baseline = [], None
    for count in plan['target_counts']:
        kinem = [datasets[incl][1].target for incl in plan['inclinations'][:count]]
        matrices, seconds = integrate(mod, potential, [density_target] + kinem, ic, times,
                                      plan['threads'] or 1)
        first = numpy.asarray(matrices[1])
        identical = None if baseline is None else bool(numpy.array_equal(first, baseline))
        if baseline is None:
            baseline = first.copy()
        runs.append(dict(kinem_targets=count, orbit_s=seconds, dtype=str(first.dtype),
                         bytes_per_target=matrix_bytes(len(ic), len(mod.sectAPP), num_bsplines,
                                                       first.dtype.itemsize),
                         first_target_identical=identical))
        del matrices, first
    marginal = None
    if len(runs) >= 2 and runs[-1]['kinem_targets'] > runs[0]['kinem_targets']:
        marginal = ((runs[-1]['orbit_s'] - runs[0]['orbit_s'])
                    / (runs[-1]['kinem_targets'] - runs[0]['kinem_targets']))
    accum = None if marginal is None or runs[0]['orbit_s'] <= 0 else marginal / runs[0]['orbit_s']
    return dict(num_orbits=plan['num_orbits'], runs=runs, marginal_s_per_target=marginal,
                accum_frac=accum,
                scaled=None if accum is None else estimate_cost(len(plan['inclinations']),
                                                                accum_frac=accum))


# --------------------------------------------------------------------------
# Harness loading and reports
# --------------------------------------------------------------------------
def load_harness(args, reference):
    import importlib.util
    argv = sys.argv[:]
    sys.argv = [rsm.EXP_SCRIPT, '--incl', repr(float(reference)), '--suffix', args.suffix,
                '--orblib-dir', args.orblib_dir, '--no-resume']
    if args.l24:
        sys.argv.append('--l24')
    if args.Q1:
        sys.argv.append('--Q1')
    if args.threads is not None:
        sys.argv += ['--n_threads', str(args.threads)]
    try:
        spec = importlib.util.spec_from_file_location('orblib_exp', rsm.EXP_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['orblib_exp'] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    rsm.prepare_module(mod, None)
    if mod.SAVE_ORBLIB or mod.REUSE_ORBLIB or mod.orblib_store is not None:
        raise ConfigError('the fan must run without shared orbit-library storage')
    if abs(float(mod.incl) - reference) > 1e-9:
        raise ConfigError('harness inclination does not match the fan reference')
    return mod


def write_report(report, path):
    with open(path, 'w') as stream:
        json.dump(report, stream, indent=1, sort_keys=True, default=str)
        stream.write('\n')


def format_diagnose(report):
    lines = ['observables (must be at round-off: the data do not depend on incl):']
    for incl, entry in sorted(report['observables']['per_incl'].items(), key=lambda kv: float(kv[0])):
        lines.append(f"  incl={float(incl):5.1f}  datacube max={entry['datacube']['max']:.2e}  "
                     f"GH max={entry['gh_moments']['max']:.2e}")
    lines.append('aperture masses (exact-vs-exact = pipeline control; frozen-vs-exact = shape error):')
    for incl, entry in sorted(report['aperture_masses']['per_incl'].items(), key=lambda kv: float(kv[0])):
        lines.append(f"  incl={float(incl):5.1f}  control max={entry['exact_vs_reference_incl']['max']:.2e}"
                     f"  frozen rms={entry['frozen_vs_exact']['rms']:.3%} "
                     f"max={entry['frozen_vs_exact']['max']:.3%}")
    lines.append('intrinsic density constraints, frozen vs exact (the fan rebuilds them exactly):')
    for incl, entry in sorted(report['intrinsic_density']['per_incl'].items(), key=lambda kv: float(kv[0])):
        lines.append(f"  incl={float(incl):5.1f}  rms={entry['rms']:.3%} max={entry['max']:.3%}")
    lines.append('E1: potential of the frozen stellar shape vs the exact one (aperture region):')
    for entry in report['potential']['models']:
        model = entry['model']
        lines.append(f"  model Q={model['Q']:g} gh={model['gh']:g} rh={model['rh']:g} "
                     f"rho0={model['rho0']:g}")
        for incl, scores in sorted(entry['per_incl'].items(), key=lambda kv: float(kv[0])):
            force = scores['force']['apertures']
            lines.append(f"    incl={float(incl):5.1f}  |dF|/|F| rms={force['rms']:.3%} "
                         f"max={force['max']:.3%}  v_circ max={scores['v_circ']['max']:.3%}")
    return '\n'.join(lines) + '\n'


def summarize(paths):
    rows = []
    for path in sorted(p for pattern in paths for p in glob.glob(pattern)):
        with open(path) as stream:
            report = json.load(stream)
        for entry in report.get('fan') or []:
            for incl, member in sorted(entry['members'].items(), key=lambda kv: float(kv[0])):
                rows.append(dict(report=path, reference=report['plan']['reference'],
                                 incl=float(incl), exact=member['exact'],
                                 penalty=member['penalty'], upsilon=member['upsilon'],
                                 **{k: entry['model'][k] for k in PARAM_KEYS}))
    return rows


def format_summary(rows):
    if not rows:
        return 'no fan members found\n'
    lines = ['incl   Q        gh       rh       rho0     Upsilon  penalty      exact  report']
    for row in rows:
        lines.append(f"{row['incl']:5.1f}  {row['Q']:7.4f}  {row['gh']:7.4f}  {row['rh']:7.3f}  "
                     f"{row['rho0']:7.3f}  {row['upsilon']:7.4f}  {row['penalty']:11.6f}  "
                     f"{'yes' if row['exact'] else 'no':5s}  {os.path.basename(row['report'])}")
    lines.append('penalties of non-exact members carry the unmeasured E1+E2 systematics')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', dest='mode', action='store_const', const='dry-run')
    mode.add_argument('--diagnose', dest='mode', action='store_const', const='diagnose',
                      help='checks D0-D4 and the E1 bound; no orbits are integrated')
    mode.add_argument('--benchmark', dest='mode', action='store_const', const='benchmark',
                      help='marginal cost and memory of an extra LOSVD target (few orbits)')
    mode.add_argument('--fan', dest='mode', action='store_const', const='fan',
                      help='one integration per model, shared by every inclination')
    mode.add_argument('--summarize', dest='mode', action='store_const', const='summarize')
    parser.set_defaults(mode='dry-run')
    parser.add_argument('--incl-grid', default=','.join(f'{x:g}' for x in DEFAULT_GRID))
    parser.add_argument('--reference', type=float, default=None,
                        help='inclination whose shape defines the orbit basis (default: grid middle)')
    for key in PARAM_KEYS:
        parser.add_argument(f'--{key}', type=float, default=None)
    parser.add_argument('--model-file', default=None, help='rows "Q gh rh rho0"')
    parser.add_argument('--num-orbits', type=int, default=None)
    parser.add_argument('--target-counts', default='1,2,4',
                        help='--benchmark: numbers of kinematic targets to time')
    parser.add_argument('--ic-seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--accum-frac', type=float, default=None,
                        help='measured accumulation fraction for the cost estimate')
    parser.add_argument('--bootstrap', action='store_true',
                        help='--diagnose: also recompute the 100-sample GH error bootstrap')
    parser.add_argument('--l24', action='store_true')
    parser.add_argument('--Q1', action='store_true')
    parser.add_argument('--suffix', default='incl_fan')
    parser.add_argument('--orblib-dir', default='./orblib')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--report', default=None)
    parser.add_argument('--reports', nargs='+', default=['incl_fan_*/report.json'],
                        help='--summarize: report globs')
    args = parser.parse_args(argv)
    if args.num_orbits is None:
        args.num_orbits = 2000 if args.mode == 'benchmark' else 100000
    if args.num_orbits < 1:
        raise ConfigError('--num-orbits must be positive')
    if args.threads is not None and args.threads < 1:
        raise ConfigError('--threads must be positive')
    if args.accum_frac is not None and not 0 <= args.accum_frac <= 1:
        raise ConfigError('--accum-frac must lie in [0,1]')
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.mode == 'summarize':
            print(format_summary(summarize(args.reports)), end='')
            return 0
        plan = build_plan(args)
        print(format_plan(plan), end='', flush=True)
        if args.mode == 'dry-run':
            return 0
    except ConfigError as error:
        print(f'incl fan: {error}', file=sys.stderr)
        return 2
    out_dir = args.out_dir or f'{HISTORY_PREFIX}_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    os.makedirs(out_dir, exist_ok=True)
    path = args.report or os.path.join(out_dir, 'report.json')
    report = dict(schema=SCHEMA, plan=plan, status='started', out_dir=out_dir,
                  host=os.environ.get('HOSTNAME_SUFFIX', socket.gethostname()),
                  started=datetime.datetime.now().isoformat(timespec='seconds'))
    code = 0
    try:
        mod = load_harness(args, plan['reference'])
        report['context'], report['grid'] = rsm.static_context(mod)
        if args.mode == 'diagnose':
            report['diagnose'] = diagnose(mod, plan, mod.alphah, mod.betah, args.bootstrap)
        elif args.mode == 'benchmark':
            report['benchmark'] = benchmark(mod, plan, mod.alphah, mod.betah)
        else:
            report['fan'] = run_fan(mod, plan, mod.alphah, mod.betah, out_dir)
        report['status'] = 'complete'
    except Exception as error:   # noqa: BLE001 - the report must record the failure
        traceback.print_exc()
        report['status'], report['error'], code = 'failed', repr(error), 1
    report['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
    write_report(report, path)
    if report['status'] == 'complete':
        if 'diagnose' in report:
            print(format_diagnose(report['diagnose']), end='')
        elif 'benchmark' in report:
            print(json.dumps(report['benchmark'], indent=1, sort_keys=True))
        else:
            print(format_summary(summarize([path])), end='')
    print(f'[incl fan] report: {path}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
