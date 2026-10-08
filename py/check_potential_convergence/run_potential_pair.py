"""Isolated Step 1B validation, field preflight and explicit seed-42 pilot.

Only --pilot samples ICs, integrates and fits. Free-Q runs five variants;
each Q1 control runs three, after explicit review of the free-Q report.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
SCHEMA = 'potential-pair-v1'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fields = load_module(HERE / 'check_potential_convergence.py', 'pair_fields')
single = load_module(HERE.parent / 'run_single_model.py', 'pair_single')
parse_seeds = single.parse_seed_list


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def array_identity(value):
    value = np.asarray(value)
    if value.dtype.hasobject:
        raise ValueError('Object arrays cannot identify diagnostic inputs')
    description = dict(shape=list(value.shape), dtype=value.dtype.str)
    digest = hashlib.sha256(canonical(description).encode())
    raw = memoryview(np.ascontiguousarray(value)).cast('B')
    for start in range(0, len(raw), 1024 * 1024):
        digest.update(raw[start:start + 1024 * 1024])
    return dict(description, sha256=digest.hexdigest())


def finite_array(value):
    value = np.asarray(value)
    return value.dtype.kind in 'fiu' and all(
        np.all(np.isfinite(value[start:start + 1024])) for start in range(0, len(value), 1024))


def validate_inputs(ic, times, count):
    if (np.shape(ic) != (count, 6) or np.shape(times) != (count,) or
            not finite_array(ic) or not finite_array(times) or np.any(np.asarray(times) <= 0)):
        raise ValueError('IC/time must have matching finite shapes and positive durations')


def check_bound(potentials, ic):
    kinetic = 0.5 * np.sum(np.asarray(ic)[:, 3:]**2, axis=1)
    energies = {}
    for name, potential in potentials.items():
        energy = kinetic + np.asarray(potential.potential(np.asarray(ic)[:, :3]))
        if energy.shape != kinetic.shape or not np.all(np.isfinite(energy)) or np.any(energy >= 0):
            raise ValueError(f'{name}: ICs are not finite bound orbits in the zero-at-infinity potential')
        energies[name] = dict(min=float(energy.min()), max=float(energy.max()))
    return energies


def save_array(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)


def save_library(directory, matrices, manifest):
    directory = Path(directory)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('Invalid diagnostic manifest schema')
    directory.mkdir()
    entries = []
    for index, matrix in enumerate(matrices):
        if np.ndim(matrix) != 2 or not finite_array(matrix):
            raise ValueError('Library matrices must be finite numeric matrices')
        path = directory / f'matrix_{index}.npy'
        save_array(path, matrix)
        entries.append(dict(file=path.name, sha256=fields.sha256_file(path), array=array_identity(matrix)))
    with (directory / 'library.json').open('x') as stream:
        json.dump(dict(schema=SCHEMA, manifest=manifest, manifest_id=identity(manifest), matrices=entries),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_library(directory, expected):
    directory = Path(directory)
    record = json.loads((directory / 'library.json').read_text())
    if (record.get('schema') != SCHEMA or canonical(record.get('manifest')) != canonical(expected) or
            record.get('manifest_id') != identity(expected)):
        raise ValueError('Diagnostic library manifest mismatch; no legacy or cross-variant reuse')
    matrices = []
    for index, entry in enumerate(record['matrices']):
        if entry['file'] != f'matrix_{index}.npy':
            raise ValueError('Invalid diagnostic matrix filename')
        path = directory / entry['file']
        if fields.sha256_file(path) != entry['sha256']:
            raise ValueError('Diagnostic library checksum mismatch')
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        if array_identity(matrix) != entry['array']:
            raise ValueError('Diagnostic matrix metadata mismatch')
        matrices.append(matrix)
    if len(matrices) != 2:
        raise ValueError('Expected the density and kinematic matrices')
    return matrices


def solver_context(datasets, matrices, regul=1.0):
    if len(datasets) != 2 or len(matrices) != 2:
        raise ValueError('Expected density and kinematic datasets')
    count = len(matrices[0])
    if count < 1 or any(np.ndim(m) != 2 or len(m) != count or not finite_array(m) for m in matrices):
        raise ValueError('Invalid full-library matrices')
    num_dof = sum(int(np.count_nonzero(d.cons_err > 0)) for d in datasets)
    if num_dof <= 0:
        raise ValueError('No positive-error constraints')
    mult = num_dof**0.5 * 10
    with np.errstate(divide='ignore'):
        pen_cons = [2 * d.cons_err**-2 for d in datasets]
    return dict(rhs=[d.cons_val / mult for d in datasets], pen_cons=pen_cons,
                pen_reg=2.0 * regul * np.ones(count) * count, mult=mult)


def solve_full(solver, datasets, matrices, upsilon, context, details=False):
    if not math.isfinite(upsilon) or not 0.1 <= upsilon <= 1.6:
        raise ValueError('Upsilon outside the diagnostic full-search bounds')
    result = solver(upsilon, matrices, datasets, **context, details=details)
    penalty = float(result['penalty'] if details else result[0])
    if not math.isfinite(penalty) or penalty < 0 or penalty >= single.FAILED_PENALTY:
        raise ValueError('Non-finite or failed penalty; a sentinel is not a result')
    if details and any(not finite_array(v) for v in [result['weights'], *result['residuals']]):
        raise ValueError('Non-finite weights or constraint residuals')
    return result if details else penalty


def fit_library(solver, datasets, matrices, input_upsilon, regul=1.0, xatol=1e-3,
                check_stop=lambda: None):
    context = solver_context(datasets, matrices, regul)
    history = []

    def objective(upsilon):
        check_stop()
        value = solve_full(solver, datasets, matrices, float(upsilon), context)
        history.append([float(upsilon), value])
        return value

    fixed = objective(input_upsilon)
    result = minimize_scalar(objective, bounds=(0.1, 1.6), method='bounded',
                             options=dict(xatol=xatol, maxiter=50))
    if not result.success:
        raise ValueError(f'Upsilon search failed: {result.message}')
    check_stop()
    final = solve_full(solver, datasets, matrices, float(result.x), context, details=True)
    return dict(penalty=float(final['penalty']), upsilon=float(result.x),
                fixed_upsilon=float(input_upsilon), fixed_penalty=fixed, history=history,
                settings=dict(bounds=[0.1, 1.6], xatol=xatol, maxiter=50, subsample=0.0, regul=regul),
                final=final)


def integrate_variant(agama, potential, datasets, ic, times, threads):
    validate_inputs(ic, times, len(ic))
    before = (array_identity(ic), array_identity(times))
    with agama.setNumThreads(threads):
        matrices = agama.orbit(potential=potential, ic=ic, time=times, Omega=0.0,
                               targets=[d.target for d in datasets])
    if before != (array_identity(ic), array_identity(times)):
        raise ValueError('Orbit integrator mutated the common IC/time arrays')
    if len(matrices) != len(datasets) or any(
            np.shape(m) != (len(ic), len(d.target)) or not finite_array(m)
            for m, d in zip(matrices, datasets)):
        raise ValueError('Orbit integration returned invalid target matrices')
    return matrices


def resources(directory):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = None
    if Path('/proc/self/status').exists():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1]) * 1024
    return dict(rss_bytes=rss, peak_rss_bytes=int(usage.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                free_disk_bytes=shutil.disk_usage(directory).free)


def pilot_variants(model):
    return ['A', 'B', 'B_native'] if model['Q'] == 1.0 else ['A', 'B', 'A_repeat', 'C', 'B_native']


def evaluate_pair(agama, potentials, stars, datasets, solver, directory, context, seed, input_upsilon,
                  num_orbits=100000, int_time=100.0, threads=1, native=False, check_stop=lambda: None,
                  pilot_checks=False, progress=lambda event: None):
    """Explicit computational kernel; non-orbital CLI modes never call it."""
    if parse_seeds(str(seed)) != [seed] or threads < 1 or num_orbits < 1 or int_time <= 0:
        raise ValueError('Invalid seed or integration settings')
    if not {'A', 'B'} <= potentials.keys() or any(k not in ('A', 'B', 'C', 'A_repeat') for k in potentials):
        raise ValueError('An explicit A/B pair is required')
    directory = Path(directory)
    directory.mkdir()
    report = dict(schema=SCHEMA, status='running', seed=seed, context=context, variants={},
                  orbit_calls=0, completed_integrations=0, ic_sampled=False, solver_checked=False,
                  penalty_checked=False, library_created=False, ic_admissibility_checked=False)
    started = time.perf_counter()
    manifests = {}
    variants = [name for name in ('A', 'B', 'A_repeat', 'C') if name in potentials] + (['B_native'] if native else [])
    estimated_library_bytes = num_orbits * sum(len(d.target) for d in datasets) * 8

    def record(stage, variant=None):
        report.update(stage=stage, active_variant=variant, elapsed_seconds=time.perf_counter()-started,
                      resources=resources(directory))
        fields.write_json(directory / 'report.json', report)
        event = {key: report[key] for key in ('stage', 'active_variant', 'elapsed_seconds', 'resources',
            'orbit_calls', 'completed_integrations', 'ic_sampled', 'solver_checked', 'penalty_checked',
            'library_created', 'ic_admissibility_checked')}
        progress(event)
        print(json.dumps(event), flush=True)

    def disk_gate(remaining):
        required = remaining * estimated_library_bytes + 2 * 1024**3
        if pilot_checks and shutil.disk_usage(directory).free < required:
            raise ValueError(f'Insufficient disk space: require {required} free bytes for remaining variants')

    record('prepare')
    try:
        check_stop()
        disk_gate(len(variants))
        potential_hashes = {}
        for name, potential in potentials.items():
            path = directory / f'potential_{name}.ini'
            potential.export(str(path))
            potential_hashes[name] = fields.sha256_file(path)
        record('sample_A')
        agama.setRandomSeed(seed)
        ic = np.asarray(stars.sample(num_orbits, potential=potentials['A'])[0])
        report['ic_sampled'] = True
        times = np.asarray(potentials['A'].Tcirc(ic)) * int_time
        validate_inputs(ic, times, num_orbits)
        if pilot_checks and (ic.dtype != np.float64 or times.dtype != np.float64):
            raise ValueError('Pilot IC/time must be native float64 arrays')
        save_array(directory / 'ic_A.npy', ic)
        save_array(directory / 'time_A.npy', times)
        radii = np.linalg.norm(ic[:, :3], axis=1)
        report['ic_radius'] = dict(min=float(radii.min()), max=float(radii.max()),
            count_outside_field_probes=int(np.count_nonzero((radii < 0.001) | (radii > 110))),
            scope='Initial positions only; not full orbit coverage')
        report['energies'] = check_bound(potentials, ic)
        report['ic_admissibility_checked'] = True
        periods = times / np.asarray(potentials['B'].Tcirc(ic))
        if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
            raise ValueError('Invalid B circular periods')
        save_array(directory / 'B_period_counts.npy', periods)
        report['B_period_counts'] = dict(min=float(periods.min()), median=float(np.median(periods)),
                                          max=float(periods.max()))
        observation_ids = [[array_identity(d.cons_val), array_identity(d.cons_err)] for d in datasets]
        for position, name in enumerate(variants):
            check_stop()
            disk_gate(len(variants)-position)
            potential_name = 'B' if name == 'B_native' else name
            potential = potentials[potential_name]
            if name == 'B_native':
                record('sample_native', name)
                agama.setRandomSeed(seed)
                branch_ic = np.asarray(stars.sample(num_orbits, potential=potential)[0])
                branch_times = np.asarray(potential.Tcirc(branch_ic)) * int_time
                validate_inputs(branch_ic, branch_times, num_orbits)
                if pilot_checks and (branch_ic.dtype != ic.dtype or branch_times.dtype != times.dtype):
                    raise ValueError('Native IC/time dtype differs from common inputs')
                save_array(directory / 'ic_B_native.npy', branch_ic)
                save_array(directory / 'time_B_native.npy', branch_times)
                report['native_energies'] = check_bound({'B_native': potential}, branch_ic)
                report['native_input_differences'] = dict(
                    ic_max_abs=float(np.max(np.abs(branch_ic-ic))),
                    time_max_abs=float(np.max(np.abs(branch_times-times))))
            else:
                branch_ic, branch_times = ic.copy(), times.copy()
            manifest = dict(schema=SCHEMA, context=context, variant=name, seed=seed,
                            potential_sha256=potential_hashes[potential_name], observations=observation_ids,
                            ic=array_identity(branch_ic), time=array_identity(branch_times),
                            num_orbits=num_orbits, int_time=int_time, threads=threads,
                            integrator=dict(Omega=0.0, accuracy='AGAMA default'),
                            protocol=dict(bounds=[0.1, 1.6], xatol=1e-3, maxiter=50, regul=1.0, subsample=0.0))
            report['orbit_calls'] += 1
            record('integrate', name)
            phase_start = time.perf_counter()
            matrices = integrate_variant(agama, potential, datasets, branch_ic, branch_times, threads)
            integration_seconds = time.perf_counter()-phase_start
            report['completed_integrations'] += 1
            check_stop()
            record('save_reload', name)
            phase_start = time.perf_counter()
            branch = directory / name
            manifest['matrix_dtypes'] = [np.asarray(m).dtype.str for m in matrices]
            if 'A' in manifests and manifest['matrix_dtypes'] != manifests['A']['matrix_dtypes']:
                raise ValueError('Matrix dtype changed between diagnostic variants')
            save_library(branch, matrices, manifest)
            report['library_created'] = True
            manifests[name] = manifest
            del matrices
            matrices = load_library(branch, manifest)
            storage_seconds = time.perf_counter()-phase_start
            record('full_solve', name)
            phase_start = time.perf_counter()
            stop = check_stop if pilot_checks else lambda: None
            fit = fit_library(solver, datasets, matrices, input_upsilon, check_stop=stop)
            report.update(solver_checked=True, penalty_checked=True)
            details = fit.pop('final')
            save_array(branch / 'weights.npy', details['weights'])
            for index, residual in enumerate(details['residuals']):
                save_array(branch / f'residual_{index}.npy', residual)
                save_array(branch / f'penalties_{index}.npy', details['penalties'][index])
                save_array(branch / f'superposition_{index}.npy', details['superpositions'][index])
            del details
            fit.update(manifest_id=identity(manifest), residual_kind='linear constraints, before error scaling',
                       seconds=dict(integration=integration_seconds, save_reload=storage_seconds,
                                    full_solve=time.perf_counter()-phase_start),
                       library_bytes=sum(p.stat().st_size for p in branch.iterdir()))
            fields.write_json(branch / 'fit.json', fit)
            report['variants'][name] = fit
            record('primary_fit_saved', name)
            if pilot_checks:
                record('strict_search', name)
                phase_start = time.perf_counter()
                strict = fit_library(solver, datasets, matrices, input_upsilon, xatol=1e-4, check_stop=stop)
                del strict['final']
                fit['strict_search'] = strict
                fit['strict_delta_penalty'] = strict['penalty']-fit['penalty']
                fit['seconds']['strict_search'] = time.perf_counter()-phase_start
                fields.write_json(branch / 'fit.json', fit)
            del matrices
            record('variant_complete', name)
        cross = {}
        for name in ('A', 'B'):
            check_stop()
            record('cross_solve', name)
            matrices = load_library(directory / name, manifests[name])
            solve_context = solver_context(datasets, matrices)
            cross[name] = {f'at_{other}': float(solve_full(solver, datasets, matrices,
                          report['variants'][other]['upsilon'], solve_context)) for other in ('A', 'B')}
            del matrices
        a, b = (report['variants'][name] for name in ('A', 'B'))
        report.update(status='ok', delta_penalty=b['penalty']-a['penalty'],
                      delta_fixed_penalty=b['fixed_penalty']-a['fixed_penalty'],
                      delta_upsilon=b['upsilon']-a['upsilon'], cross_penalties=cross)
        if pilot_checks:
            controls = {}
            checks = {f'{name}_strict_search': fit['strict_delta_penalty'] for name, fit in report['variants'].items()}
            checks.update({f'{name}_repeat_solve': cross[name][f'at_{name}']-report['variants'][name]['penalty']
                           for name in ('A', 'B')})
            for name, reference in (('A_repeat', 'A'), ('C', 'B'), ('B_native', 'B')):
                if name in report['variants']:
                    controls[f'{name}_minus_{reference}'] = report['variants'][name]['penalty']-report['variants'][reference]['penalty']
            if 'A_repeat' in report['variants']:
                checks['A_repeat'] = controls['A_repeat_minus_A']
                checks['A_repeat_fixed'] = report['variants']['A_repeat']['fixed_penalty']-a['fixed_penalty']
            report['control_deltas'] = controls
            report['technical_checks'] = dict(budget=0.001, deltas=checks,
                passed=all(math.isfinite(v) and abs(v) <= 0.001 for v in checks.values()))
            report['sensitivity_flags'] = [name for name, value in controls.items()
                                          if name != 'A_repeat_minus_A' and abs(value) > 0.01]
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status='stopped', error=str(error) or 'Interrupted')
    except Exception as error:
        report.update(status='failed', error=repr(error))
    record('complete' if report['status'] == 'ok' else report['status'])
    return report


def summarize_pairs(reports, seeds):
    records = list(reports)
    if len({identity(r.get('context', {}).get('model')) for r in records}) > 1:
        raise ValueError('Summarize one physical model at a time')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Repeated planned seeds')
    owners = {seed: [r for r in records if r.get('seed') == seed] for seed in seeds}
    duplicate = [seed for seed, rows in owners.items() if len(rows) > 1]
    complete = [rows[0] for rows in owners.values() if len(rows) == 1 and rows[0].get('status') == 'ok'
                and all(isinstance(rows[0].get(k), (int, float)) and math.isfinite(rows[0][k])
                        for k in ('delta_penalty', 'delta_upsilon'))]
    values = [r['delta_penalty'] for r in complete]
    mean = statistics.mean(values) if values else None
    std = statistics.stdev(values) if len(values) > 1 else None
    sem = std / math.sqrt(len(values)) if std is not None else None
    ci = None if sem is None else [mean - student_t.ppf(0.975, len(values)-1)*sem,
                                  mean + student_t.ppf(0.975, len(values)-1)*sem]
    return dict(records=records, requested=list(seeds), complete=len(complete),
                missing=[seed for seed, rows in owners.items() if not rows], duplicate_seeds=duplicate,
                failed=[seed for seed, rows in owners.items() if len(rows) == 1 and rows[0] not in complete],
                unexpected=[r.get('seed') for r in records if r.get('seed') not in seeds],
                mean_delta_penalty=mean, median_delta_penalty=statistics.median(values) if values else None,
                std_delta_penalty=std, sem_delta_penalty=sem, mc_mean_interval95=ci,
                interval_scope='Approximate IC Monte Carlo mean interval, not observational uncertainty')


def recipe_identity(recipe):
    return identity({k: v for k, v in recipe.items() if k not in ('source', 'sha256')})


def read_evidence(path, model, recipe, include_control=False):
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    top_path = path.parent.parent / 'report.json'
    top = json.loads(top_path.read_text())
    if not any(entry['directory'] == path.parent.name for entry in top['models']):
        raise ValueError('Model report is not listed in its top-level report')
    if any(report['model'].get(k) != model[k] for k in fields.PARAMS):
        raise ValueError('Field-check model does not match the requested model exactly')
    if recipe_identity(top['recipe']) != recipe_identity(recipe):
        raise ValueError('Field-check physical recipe differs from the current harness')
    if report['reference'].get('converged') is not True:
        raise ValueError('Field reference did not converge')
    if top['options']['radii'] < 192 or top['options']['angles'] < 17:
        raise ValueError('Dense field evidence is required')
    if report['thresholds']['rms'] != 1e-4 or report['thresholds']['maximum'] != 1e-3:
        raise ValueError('Field-check thresholds differ from the agreed checks')
    scores = report['comparisons']['angular_l24']['scores']
    if any(not fields.passes(scores.get(name), 1e-4, 1e-3) for name in fields.REGIONS):
        raise ValueError('Refined field does not pass every sampled region')
    coefficients = {}
    tags = {'A': 'baseline', 'B': 'angular_l24', 'stars': 'stars_baseline', 'halo': 'halo_baseline'}
    if include_control:
        control = report['comparisons'].get('common_fine', {}).get('scores', {})
        if any(not fields.passes(control.get(name), 1e-4, 1e-3) for name in fields.REGIONS):
            raise ValueError('Fine control C does not pass every sampled region')
        tags['C'] = 'common_fine'
    for name, tag in tags.items():
        record = next(c for c in report['coefficients'] if c['tag'] == tag)
        if record['coefficient_file'] != tag + '.ini':
            raise ValueError('Invalid field coefficient filename')
        source = path.parent / record['coefficient_file']
        grid = fields.read_grid(source)
        if not np.array_equal(grid, record['actual_grid']) or len(grid) != record['requested']['gridSizeR']:
            raise ValueError('Field coefficient grid does not match report metadata')
        coefficients[name] = dict(path=str(source), sha256=fields.sha256_file(source),
                                  requested=record['requested'], actual_grid=grid.tolist())
    a, b = (coefficients[name] for name in ('A', 'B'))
    if a['requested'] != recipe['baseline']:
        raise ValueError('Field baseline settings differ from the harness')
    expected = dict(recipe['baseline'], gridSizeR=184, lmax=24,
                    rmin=a['actual_grid'][0], rmax=a['actual_grid'][-1])
    if b['requested'] != expected:
        raise ValueError('B must use the Step 1A baseline-frozen radial grid')
    if include_control:
        c = coefficients['C']
        expected_c = dict(recipe['baseline'], gridSizeR=399, lmax=40,
                          rmin=c['actual_grid'][0], rmax=c['actual_grid'][-1])
        if c['requested'] != expected_c:
            raise ValueError('Pilot control C must be the recorded N399/l40 common_fine')
    return dict(report=str(path), report_sha256=fields.sha256_file(path),
                top_sha256=fields.sha256_file(top_path), agama=top['agama'],
                model=report['model'], recipe=top['recipe'],
                recorded_harness_sha256=top['recipe']['sha256'], coefficients=coefficients)


def validate_harness_interface(path):
    path = Path(path)
    if path.name != fields.DEFAULT_HARNESS.name:
        raise ValueError('Only the experimental harness may be loaded, never production')
    tree = fields.ast.parse(path.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, fields.ast.FunctionDef)}
    objective = functions.get('halo_IC_lib_weights_pca_fixed')
    if (objective is None or 'diagnostic' not in [arg.arg for arg in objective.args.args] or
            'solve_orbit_library' not in functions):
        raise ValueError('Harness lacks the Step 1B diagnostic interface; use the updated experimental copy')


def dependency_status():
    return {name: importlib.util.find_spec(name) is not None for name in
            ('numpy', 'scipy', 'agama', 'torch', 'botorch', 'gpytorch', 'sklearn', 'requests')}


def load_harness(path, model, threads, directory):
    path = Path(path).resolve()
    argv, search_path, cwd = sys.argv[:], sys.path[:], Path.cwd()
    previous_table = os.environ.get('AGAMA_TABLE3')
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, handlers)
    os.environ['AGAMA_TABLE3'] = str(path.parent / 'table3.dat')
    try:
        sys.argv = [str(path), '--incl', str(model['incl']), '--suffix', 'potential_preflight',
                    '--no-resume', '--n_threads', str(threads)]
        sys.path.insert(0, str(path.parent))
        os.chdir(directory)
        return load_module(path, 'potential_diagnostic_harness')
    finally:
        sys.argv, sys.path = argv, search_path
        os.chdir(cwd)
        if previous_table is None:
            os.environ.pop('AGAMA_TABLE3', None)
        else:
            os.environ['AGAMA_TABLE3'] = previous_table
        for number, handler in handlers.items():
            signal.signal(number, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def force_comparison(measured, reference, points):
    measured, reference = np.asarray(measured), np.asarray(reference)
    if (measured.shape != points.shape or reference.shape != points.shape or
            not np.all(np.isfinite(measured)) or not np.all(np.isfinite(reference))):
        raise ValueError('Invalid force API result')
    denominator = np.linalg.norm(reference, axis=1)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError('Reference force norm must be finite and positive')
    relative = np.linalg.norm(measured-reference, axis=1)/denominator
    if not np.all(np.isfinite(relative)):
        raise ValueError('Non-finite relative force difference')
    radii = np.linalg.norm(points, axis=1)
    scores = {}
    for name, (lo, hi) in fields.REGIONS.items():
        indices = np.flatnonzero((radii >= lo) & (radii <= hi))
        if not len(indices):
            scores[name] = None
            continue
        values = relative[indices]
        worst = indices[np.argmax(values)]
        scores[name] = dict(n=len(indices), rms=float(np.sqrt(np.mean(values**2))),
                            max=float(values.max()), worst_xyz=points[worst].tolist(),
                            worst_radius=float(radii[worst]))
    return dict(scores=scores, rms=scores['all']['rms'], max=scores['all']['max'],
                tolerance=1e-9, within_reproduction_tolerance=bool(relative.max() <= 1e-9))


def field_preflight(agama, baseline, stars, halo, evidence, directory, report=None, live=None):
    coefficients = evidence['coefficients']
    if any(fields.sha256_file(record['path']) != record['sha256'] for record in coefficients.values()):
        raise ValueError('Field evidence checksum changed after validation')
    report = {} if report is None else report
    names = ('A', 'B', 'C') if 'C' in coefficients else ('A', 'B')
    results = {name: dict(status='not_checked', requested=coefficients[name]['requested'],
                          source_sha256=coefficients[name]['sha256']) for name in names}
    report['potentials'] = results
    points = fields.probe_points([coefficients[k]['actual_grid'] for k in ('A', 'stars', 'halo')], 192, 17)
    star_params, halo_params = fields.density_parameters(evidence['model'], evidence['recipe'])
    report['field_reference'] = dict(kind='live reconstruction from archived Step 1A recipe',
                                     stars=star_params, halo=halo_params, probe_points=len(points))

    def persist():
        fields.write_json(directory / 'report.json', report)

    persist()
    reference_density = agama.Density(agama.Density(**star_params), agama.Density(**halo_params))
    combined = agama.Density(stars, halo)
    for name, record in results.items():
        record.update(status='running', stage='build')
        persist()
        try:
            potential = baseline if name == 'A' else agama.Potential(density=combined, **record['requested'])
            reference = agama.Potential(density=reference_density, **record['requested'])
            record['stage'] = 'export'
            persist()
            path = directory / f'{name}.ini'
            potential.export(str(path))
            grid = fields.read_grid(path)
            record.update(actual_grid=grid.tolist(), sha256=fields.sha256_file(path),
                          grid_matches_evidence=bool(np.array_equal(grid, coefficients[name]['actual_grid'])))
            record['export_matches_evidence'] = record['sha256'] == record['source_sha256']
            record['stage'] = 'live_reproduction'
            persist()
            live_force = np.asarray(potential.force(points))
            record['live_reproduction'] = dict(force_comparison(live_force, reference.force(points), points),
                                               gates_preflight=True, reference='reconstructed live Step 1A recipe')
            record['stage'] = 'export_roundtrip'
            persist()
            reloaded_force = np.asarray(agama.Potential(str(path)).force(points))
            record['export_roundtrip'] = dict(force_comparison(reloaded_force, live_force, points),
                                              gates_preflight=False, reference='live harness field',
                                              measured='reloaded current export')
            record['stage'] = 'serialized_reproduction'
            persist()
            source_force = agama.Potential(coefficients[name]['path']).force(points)
            record['serialized_reproduction'] = dict(force_comparison(reloaded_force, source_force, points),
                                                      gates_preflight=True, reference='reloaded archived export')
            failures = []
            if not record['grid_matches_evidence']:
                failures.append('radial grid differs from Step 1A')
            if not record['export_matches_evidence']:
                failures.append('coefficient export differs from Step 1A')
            for check in ('live_reproduction', 'serialized_reproduction'):
                if not record[check]['within_reproduction_tolerance']:
                    failures.append(f'{check} exceeds 1e-9')
            record.update(status='failed' if failures else 'pass', stage='complete')
            if failures:
                record['error'] = '; '.join(failures)
            elif live is not None:
                live[name] = potential
        except KeyboardInterrupt:
            record.update(status='interrupted', error='KeyboardInterrupt')
            raise
        except Exception as error:
            record.update(status='failed', error=repr(error))
        finally:
            persist()
    report['serialization_warnings'] = [name for name, record in results.items()
                                        if 'export_roundtrip' in record and
                                        not record['export_roundtrip']['within_reproduction_tolerance']]
    persist()
    failed = [name for name, record in results.items() if record['status'] != 'pass']
    if failed:
        raise ValueError(f'Field preflight failed for {", ".join(failed)}; see saved per-variant metrics')
    return results


def preflight(args, plan, directory, report=None, runtime_state=None):
    report = {} if report is None else report
    context = dict(python=sys.executable, python_version=sys.version,
                   numpy=np.__version__, scipy=fields.scipy.__version__,
                   container_image_id=os.environ.get('POTENTIAL_PAIR_IMAGE_ID'))
    report.update(context=context, stage='import_harness')
    fields.write_json(directory / 'report.json', report)
    mod = load_harness(args.harness, plan['model'], args.threads, directory)
    current_agama = fields.module_identity(mod.agama)
    context['agama'] = current_agama
    report['stage'] = 'runtime_identity'
    fields.write_json(directory / 'report.json', report)
    if mod.SAVE_ORBLIB or mod.REUSE_ORBLIB or mod.orblib_store is not None:
        raise ValueError('Shared storage must be disabled for the diagnostic')
    binary_hashes = lambda info: {v['sha256'] for v in info['files'] if '.so' in Path(v['path']).name or v['path'].endswith('.pyd')}
    if not binary_hashes(current_agama) or binary_hashes(current_agama) != binary_hashes(plan['evidence']['agama']):
        raise ValueError('AGAMA binary differs from field evidence; validate fields in this environment first')
    runtime, geometry = single.static_context(mod)
    observations = [[array_identity(d.cons_val), array_identity(d.cons_err)] for d in mod.datasets]
    geometry_arrays = [array_identity(value) for value in (mod.gridx, mod.gridy, mod.gridv, *mod.sectAPP)]
    context.update(runtime=runtime, geometry=geometry, observations=observations, geometry_arrays=geometry_arrays,
                   catalogue_sha256=fields.sha256_file(Path(args.harness).resolve().parent / 'table3.dat'))
    report['stage'] = 'field_checks'
    fields.write_json(directory / 'report.json', report)

    def capture(**inputs):
        if (inputs['num_orbits'] != 100000 or inputs['int_time'] != 100.0 or inputs['regul'] != 1.0 or
                inputs['upsilon_bounds'] != (0.1, 1.6)):
            raise ValueError('Harness integration/solve constants changed')
        live = {}
        result = field_preflight(mod.agama, inputs['baseline'], inputs['density_stars'], inputs['density_halo'],
                                 plan['evidence'], directory, report, live=live)
        if runtime_state is not None:
            runtime_state.update(module=mod, inputs=inputs, live=live)
        return result

    potentials = mod.halo_IC_lib_weights_pca_fixed(
        np.zeros(4), None, mod.bounds_original, mod.densityStars, mod.datasets, mod.alphah, mod.betah,
        direct_params={k: plan['model'][k] for k in single.PARAM_KEYS}, diagnostic=capture)
    return dict(context=context, potentials=potentials, stage='complete',
                manifest_id=identity(dict(plan=plan, context=context, potentials=potentials)))


def validate_pilot_models(models, index):
    if (len(models) != 3 or index not in (0, 1, 2) or
            any(models[i]['Q'] != 1.0 for i in (0, 1)) or models[2]['Q'] == 1.0 or
            models[0]['rh'] >= models[1]['rh'] or len({m['incl'] for m in models}) != 1):
        raise ValueError('Pilot requires three fixed rows: Q1 small, Q1 large, free-Q at the same inclination')


def reviewed_freeq(path, plan):
    if path is None:
        raise ValueError('Q1 pilot requires --reviewed-freeq REPORT after inspecting the five-variant pilot')
    report = json.loads(Path(path).read_text())
    if (report.get('status') != 'pilot_pass' or report['plan']['model']['Q'] == 1.0 or
            report['plan']['models_sha256'] != plan['models_sha256'] or
            report['evaluation']['seed'] != 42 or report['evaluation']['completed_integrations'] != 5 or
            set(report['evaluation']['variants']) != set(pilot_variants(report['plan']['model'])) or
            report['evaluation']['technical_checks'].get('passed') is not True):
        raise ValueError('Reviewed free-Q report is incomplete, incompatible or fails the technical budget')
    return dict(path=str(Path(path).resolve()), sha256=fields.sha256_file(path),
                environment_signature=report['environment_signature'])


def pilot_environment(plan, context):
    return identity(dict(implementation={k: plan[k] for k in (
        'implementation_sha256', 'fields_implementation_sha256', 'single_implementation_sha256')},
        harness=plan['recipe']['sha256'], recipe=recipe_identity(plan['recipe']),
        threads=plan['threads'], context={k: context[k] for k in (
            'python_version', 'numpy', 'scipy', 'agama', 'container_image_id',
            'catalogue_sha256', 'geometry_arrays', 'observations')}))


def run_pilot(args, plan, directory, report):
    state = {}
    requested_stop = []
    previous = {}

    def signal_stop(number, frame):
        requested_stop.append(signal.Signals(number).name)

    def check_stop():
        if requested_stop or (directory / 'STOP').exists():
            raise InterruptedError(','.join(requested_stop) if requested_stop else 'STOP file')

    def progress(event):
        report.update(event)
        fields.write_json(directory / 'report.json', report)

    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, signal_stop)
        check_stop()
        report.update(preflight(args, plan, directory, report, runtime_state=state))
        report['environment_signature'] = pilot_environment(plan, report['context'])
        review = plan.get('reviewed_freeq')
        if review and (fields.sha256_file(review['path']) != review['sha256'] or
                       review['environment_signature'] != report['environment_signature']):
            raise ValueError('Reviewed free-Q provenance differs from current pilot environment/code/threads')
        check_stop()
        live = state['live']
        if plan['model']['Q'] != 1.0:
            live['A_repeat'] = live['A']
        context = dict(report['context'], model=plan['model'], construction_mode='live_density',
            environment_signature=report['environment_signature'], recipe=plan['recipe'],
            potentials=report['potentials'], field_reference=report['field_reference'],
            potential_zero='finite-mass live Multipole, zero at infinity',
            planned_variants=plan['pilot_variants'], delta_P=0.01, technical_budget=0.001)
        result = evaluate_pair(state['module'].agama, live, state['inputs']['density_stars'],
            state['inputs']['datasets'], state['inputs']['solve_library'], directory / 'seed_42', context,
            42, plan['model']['upsilon'], threads=args.threads, native=True, check_stop=check_stop,
            pilot_checks=True, progress=progress)
        report['evaluation'] = result
        if result['status'] != 'ok':
            report.update(status=result['status'], error=result.get('error'))
        else:
            report['status'] = 'pilot_pass' if result['technical_checks']['passed'] else 'needs_review'
            if result['completed_integrations'] != len(plan['pilot_variants']):
                raise ValueError('Pilot completed an unexpected number of integrations')
        with (directory / 'pilot.tsv').open('x') as stream:
            stream.write('variant\tpenalty\tUpsilon\tfixed_penalty\tstrict_delta_penalty\tintegration_seconds\tlibrary_bytes\n')
            for name, fit in result['variants'].items():
                stream.write('\t'.join(str(v) for v in (name, fit['penalty'], fit['upsilon'],
                    fit['fixed_penalty'], fit.get('strict_delta_penalty'),
                    fit['seconds']['integration'], fit['library_bytes'])) + '\n')
        return 0 if report['status'] == 'pilot_pass' else 3 if report['status'] == 'needs_review' else 1
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status='stopped', error=str(error) or 'Interrupted')
        return 1
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        fields.write_json(directory / 'report.json', report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--harness', type=Path, default=fields.DEFAULT_HARNESS)
    parser.add_argument('--field-report', required=True, type=Path, help='Dense Step 1A model_NNN/report.json')
    parser.add_argument('--seeds', default='42', help='Preflight never seeds; the pilot accepts only 42')
    parser.add_argument('--reviewed-freeq', type=Path, help='Explicit review acknowledgement: successful free-Q pilot report')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-inputs', action='store_true', help='No AGAMA/harness import or output files')
    mode.add_argument('--preflight', action='store_true', help='Load datasets and verify fields; no IC/orbits/solve')
    mode.add_argument('--validate-pilot', action='store_true', help='Validate the three-row pilot plan and selected evidence, no AGAMA')
    mode.add_argument('--pilot-preflight', action='store_true', help='Pilot fields including free-Q C; no IC/orbits/solve')
    mode.add_argument('--pilot', action='store_true', help='Execute this model: five free-Q or three Q1 integrations, seed=42')
    args = parser.parse_args(argv)
    try:
        args.seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.model_index < 0 or args.threads < 1:
        parser.error('Require model-index >= 0 and threads >= 1')
    if (args.preflight or args.pilot_preflight or args.pilot) and args.output is None:
        parser.error('Runtime modes require a new --output directory')
    if (args.pilot or args.pilot_preflight or args.validate_pilot) and args.seeds != [42]:
        parser.error('The approved pilot uses seed 42 only')
    if args.reviewed_freeq is not None and not args.pilot:
        parser.error('--reviewed-freeq is only for a Q1 --pilot')
    return args


def main(args=None):
    args = parse_args() if args is None else args
    try:
        validate_harness_interface(args.harness)
        recipe = fields.read_recipe(args.harness)
        models = fields.read_models(args.models, recipe)
        if args.model_index >= len(models):
            raise ValueError('model-index outside input rows')
        model = models[args.model_index]
        pilot_mode = args.pilot or args.pilot_preflight or args.validate_pilot
        if pilot_mode:
            validate_pilot_models(models, args.model_index)
        evidence = read_evidence(args.field_report, model, recipe,
                                 include_control=pilot_mode and model['Q'] != 1.0)
        plan = dict(schema=SCHEMA, model=model, model_index=args.model_index, recipe=recipe,
                    models_sha256=fields.sha256_file(args.models), evidence=evidence,
                    implementation_sha256=fields.sha256_file(__file__),
                    fields_implementation_sha256=fields.sha256_file(HERE / 'check_potential_convergence.py'),
                    single_implementation_sha256=fields.sha256_file(HERE.parent / 'run_single_model.py'),
                    planned_seeds=args.seeds, threads=args.threads)
        if pilot_mode:
            plan.update(pilot_variants=pilot_variants(model), delta_P=0.01, technical_budget=0.001,
                        construction_mode='live_density', num_orbits=100000, int_time=100.0,
                        seed=42, shared_storage=False, network=False)
        if args.pilot:
            if model['Q'] == 1.0:
                plan['reviewed_freeq'] = reviewed_freeq(args.reviewed_freeq, plan)
            elif args.reviewed_freeq is not None:
                raise ValueError('Free-Q pilot must not use --reviewed-freeq')
        if args.validate_inputs or args.validate_pilot:
            print(json.dumps(dict(status='validated', plan=plan, dependencies=dependency_status(),
                                  orbit_calls=0, ic_sampled=False, penalty_checked=False), indent=2))
            return 0
        directory = fields.create_output(args.output.resolve())
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f'Input error: {error}', file=sys.stderr)
        return 2
    report = dict(status='running', plan=plan, orbit_calls=0, ic_sampled=False,
                  penalty_checked=False, solver_checked=False, library_created=False,
                  ic_admissibility_checked=False, free_disk_bytes=shutil.disk_usage(directory).free)
    fields.write_json(directory / 'report.json', report)
    code = 0
    try:
        with (directory / ('pilot.log' if args.pilot else 'preflight.log')).open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            if args.pilot:
                code = run_pilot(args, plan, directory, report)
            else:
                report.update(preflight(args, plan, directory, report))
                report['status'] = 'preflight_pass'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=repr(error))
        code = 1
    fields.write_json(directory / 'report.json', report)
    print(json.dumps(dict(status=report['status'], output=str(directory), error=report.get('error'),
                          serialization_warnings=report.get('serialization_warnings', []),
                          orbit_calls=report['orbit_calls'], penalty_checked=report['penalty_checked'])))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
