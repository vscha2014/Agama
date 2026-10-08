"""Step 1B preparation: validation/preflight CLI and a mock-tested paired kernel.

The CLI has no orbit-execution mode. Real pilot scheduling requires a separate
approval and entry point; preflight never samples ICs, integrates or fits.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

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
    if (record.get('schema') != SCHEMA or record.get('manifest') != expected or
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


def fit_library(solver, datasets, matrices, input_upsilon, regul=1.0, xatol=1e-3):
    context = solver_context(datasets, matrices, regul)
    history = []

    def objective(upsilon):
        value = solve_full(solver, datasets, matrices, float(upsilon), context)
        history.append([float(upsilon), value])
        return value

    fixed = objective(input_upsilon)
    result = minimize_scalar(objective, bounds=(0.1, 1.6), method='bounded',
                             options=dict(xatol=xatol, maxiter=50))
    if not result.success:
        raise ValueError(f'Upsilon search failed: {result.message}')
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


def evaluate_pair(agama, potentials, stars, datasets, solver, directory, context, seed, input_upsilon,
                  num_orbits=100000, int_time=100.0, threads=1, native=False, check_stop=lambda: None):
    """Explicit computational kernel; not reachable from the non-orbital CLI."""
    if parse_seeds(str(seed)) != [seed] or threads < 1 or num_orbits < 1 or int_time <= 0:
        raise ValueError('Invalid seed or integration settings')
    if not {'A', 'B'} <= potentials.keys() or any(k not in ('A', 'B', 'C', 'A_repeat') for k in potentials):
        raise ValueError('An explicit A/B pair is required')
    directory = Path(directory)
    directory.mkdir()
    report = dict(schema=SCHEMA, status='running', seed=seed, context=context, variants={})
    fields.write_json(directory / 'report.json', report)
    manifests = {}
    try:
        check_stop()
        potential_hashes = {}
        for name, potential in potentials.items():
            path = directory / f'potential_{name}.ini'
            potential.export(str(path))
            potential_hashes[name] = fields.sha256_file(path)
        agama.setRandomSeed(seed)
        ic = np.asarray(stars.sample(num_orbits, potential=potentials['A'])[0])
        times = np.asarray(potentials['A'].Tcirc(ic)) * int_time
        validate_inputs(ic, times, num_orbits)
        save_array(directory / 'ic_A.npy', ic)
        save_array(directory / 'time_A.npy', times)
        report['energies'] = check_bound(potentials, ic)
        periods = times / np.asarray(potentials['B'].Tcirc(ic))
        if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
            raise ValueError('Invalid B circular periods')
        report['B_period_counts'] = dict(min=float(periods.min()), median=float(np.median(periods)),
                                          max=float(periods.max()))
        observation_ids = [[array_identity(d.cons_val), array_identity(d.cons_err)] for d in datasets]
        variants = [name for name in ('A', 'B', 'A_repeat', 'C') if name in potentials] + (['B_native'] if native else [])
        for name in variants:
            check_stop()
            potential_name = 'B' if name == 'B_native' else name
            potential = potentials[potential_name]
            if name == 'B_native':
                agama.setRandomSeed(seed)
                branch_ic = np.asarray(stars.sample(num_orbits, potential=potential)[0])
                branch_times = np.asarray(potential.Tcirc(branch_ic)) * int_time
                validate_inputs(branch_ic, branch_times, num_orbits)
                save_array(directory / 'ic_B_native.npy', branch_ic)
                save_array(directory / 'time_B_native.npy', branch_times)
                check_bound({'B_native': potential}, branch_ic)
            else:
                branch_ic, branch_times = ic.copy(), times.copy()
            manifest = dict(schema=SCHEMA, context=context, variant=name, seed=seed,
                            potential_sha256=potential_hashes[potential_name], observations=observation_ids,
                            ic=array_identity(branch_ic), time=array_identity(branch_times),
                            num_orbits=num_orbits, int_time=int_time, threads=threads,
                            integrator=dict(Omega=0.0, accuracy='AGAMA default'),
                            protocol=dict(bounds=[0.1, 1.6], xatol=1e-3, maxiter=50, regul=1.0, subsample=0.0))
            matrices = integrate_variant(agama, potential, datasets, branch_ic, branch_times, threads)
            check_stop()
            branch = directory / name
            manifest['matrix_dtypes'] = [np.asarray(m).dtype.str for m in matrices]
            if 'A' in manifests and manifest['matrix_dtypes'] != manifests['A']['matrix_dtypes']:
                raise ValueError('Matrix dtype changed between diagnostic variants')
            save_library(branch, matrices, manifest)
            manifests[name] = manifest
            del matrices
            matrices = load_library(branch, manifest)
            fit = fit_library(solver, datasets, matrices, input_upsilon)
            details = fit.pop('final')
            save_array(branch / 'weights.npy', details['weights'])
            for index, residual in enumerate(details['residuals']):
                save_array(branch / f'residual_{index}.npy', residual)
                save_array(branch / f'penalties_{index}.npy', details['penalties'][index])
                save_array(branch / f'superposition_{index}.npy', details['superpositions'][index])
            del matrices, details
            fit.update(manifest_id=identity(manifest), residual_kind='linear constraints, before error scaling')
            fields.write_json(branch / 'fit.json', fit)
            report['variants'][name] = fit
            fields.write_json(directory / 'report.json', report)
        cross = {}
        for name in ('A', 'B'):
            check_stop()
            matrices = load_library(directory / name, manifests[name])
            solve_context = solver_context(datasets, matrices)
            cross[name] = {f'at_{other}': float(solve_full(solver, datasets, matrices,
                          report['variants'][other]['upsilon'], solve_context)) for other in ('A', 'B')}
            del matrices
        a, b = (report['variants'][name] for name in ('A', 'B'))
        report.update(status='ok', delta_penalty=b['penalty']-a['penalty'],
                      delta_fixed_penalty=b['fixed_penalty']-a['fixed_penalty'],
                      delta_upsilon=b['upsilon']-a['upsilon'], cross_penalties=cross)
    except (InterruptedError, KeyboardInterrupt) as error:
        report.update(status='stopped', error=str(error) or 'Interrupted')
    except Exception as error:
        report.update(status='failed', error=repr(error))
    fields.write_json(directory / 'report.json', report)
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


def read_evidence(path, model, recipe):
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


def field_preflight(agama, baseline, stars, halo, evidence, directory, report=None):
    coefficients = evidence['coefficients']
    if any(fields.sha256_file(record['path']) != record['sha256'] for record in coefficients.values()):
        raise ValueError('Field evidence checksum changed after validation')
    report = {} if report is None else report
    results = {name: dict(status='not_checked', requested=coefficients[name]['requested'],
                          source_sha256=coefficients[name]['sha256']) for name in ('A', 'B')}
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


def preflight(args, plan, directory, report=None):
    report = {} if report is None else report
    context = dict(python=sys.executable, python_version=sys.version,
                   numpy=np.__version__, scipy=fields.scipy.__version__)
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
        return field_preflight(mod.agama, inputs['baseline'], inputs['density_stars'], inputs['density_halo'],
                               plan['evidence'], directory, report)

    potentials = mod.halo_IC_lib_weights_pca_fixed(
        np.zeros(4), None, mod.bounds_original, mod.densityStars, mod.datasets, mod.alphah, mod.betah,
        direct_params={k: plan['model'][k] for k in single.PARAM_KEYS}, diagnostic=capture)
    return dict(context=context, potentials=potentials, stage='complete',
                manifest_id=identity(dict(plan=plan, context=context, potentials=potentials)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--model-index', type=int, default=0)
    parser.add_argument('--harness', type=Path, default=fields.DEFAULT_HARNESS)
    parser.add_argument('--field-report', required=True, type=Path, help='Dense Step 1A model_NNN/report.json')
    parser.add_argument('--seeds', default='42', help='Planned seeds only; preflight does not seed/sample')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-inputs', action='store_true', help='No AGAMA/harness import or output files')
    mode.add_argument('--preflight', action='store_true', help='Load datasets and verify fields; no IC/orbits/solve')
    args = parser.parse_args(argv)
    try:
        args.seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.model_index < 0 or args.threads < 1:
        parser.error('Require model-index >= 0 and threads >= 1')
    if args.preflight and args.output is None:
        parser.error('--preflight requires a new --output directory')
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
        evidence = read_evidence(args.field_report, model, recipe)
        plan = dict(schema=SCHEMA, model=model, model_index=args.model_index, recipe=recipe,
                    models_sha256=fields.sha256_file(args.models), evidence=evidence,
                    implementation_sha256=fields.sha256_file(__file__),
                    fields_implementation_sha256=fields.sha256_file(HERE / 'check_potential_convergence.py'),
                    single_implementation_sha256=fields.sha256_file(HERE.parent / 'run_single_model.py'),
                    planned_seeds=args.seeds, threads=args.threads)
        if args.validate_inputs:
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
        with (directory / 'preflight.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report.update(preflight(args, plan, directory, report))
        report['status'] = 'preflight_pass'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=repr(error))
        code = 1
    fields.write_json(directory / 'report.json', report)
    print(json.dumps(dict(status=report['status'], output=str(directory), error=report.get('error'),
                          serialization_warnings=report.get('serialization_warnings', []),
                          orbit_calls=0, penalty_checked=False)))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
