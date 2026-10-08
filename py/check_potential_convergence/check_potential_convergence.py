"""Deterministic field-only convergence check; no orbit libraries or fitting."""

import argparse
import ast
import csv
import datetime
import hashlib
import importlib
import json
import math
import operator
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad


HERE = Path(__file__).resolve().parent
DEFAULT_HARNESS = HERE.parent / 'Fornax_P21_PCA_w3Sersic_orblib_exp.py'
PARAMS = ('incl', 'Q', 'gh', 'rh', 'rho0', 'upsilon')
REGIONS = {'central': (0., 0.05), 'main': (0.05, 2.1),
           'outer_orbits': (2.1, 10.), 'halo_tail': (10., math.inf), 'all': (0., math.inf)}
PROBE_MIN = 0.001
PROBE_MAX = 110.0


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def numeric_expression(node, values):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Pow: operator.pow}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.Attribute) and ast.unparse(node) == 'numpy.pi':
        return math.pi
    if isinstance(node, ast.BinOp) and type(node.op) in operations:
        return operations[type(node.op)](numeric_expression(node.left, values),
                                          numeric_expression(node.right, values))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = numeric_expression(node.operand, values)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f'Unsupported harness expression: {ast.unparse(node)}')


def read_recipe(path):
    path = Path(path).resolve()
    tree = ast.parse(path.read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    values = {}
    for key in ('D_O22', 'D', 'q_ap', 'sc', 'massSt', 'scaleRst', 'Sersic_m',
                'alphah', 'betah', 'vscale', 'Upsilon_lower', 'Upsilon_upper'):
        values[key] = numeric_expression(assignments[key], values)
    expected_geometry = dict(beta='incl * numpy.pi / 180', sinbeta='numpy.sin(beta)',
                             cosbeta='numpy.cos(beta)', q_ap2='q_ap**2',
                             axRZst='(q_ap2 - cosbeta**2)**0.5 / sinbeta')
    for key, expression in expected_geometry.items():
        if ast.dump(assignments[key]) != ast.dump(ast.parse(expression, mode='eval').body):
            raise ValueError(f'Harness geometry changed: {key}; review this diagnostic')
    objective = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'halo_IC_lib_weights_pca_fixed')
    local = {target.id: node.value for node in ast.walk(objective) if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)}

    def parameters(call, function, dynamic):
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != function or call.args:
            raise ValueError(f'Unexpected harness constructor: {function}')
        keywords = {item.arg: item.value for item in call.keywords}
        for key, expression in dynamic.items():
            if key not in keywords or ast.dump(keywords.pop(key)) != ast.dump(
                    ast.parse(expression, mode='eval').body):
                raise ValueError(f'Harness constructor changed: {function}.{key}')
        return {key: numeric_expression(value, values) for key, value in keywords.items()}

    stars = parameters(assignments['densityStars'], 'agama.Density', dict(axisRatioZ='axRZst'))
    halo = parameters(local['densityHalo'], 'agama.Density',
                      dict(gamma='gh', axisratioz='Q', densitynorm='rho0', scaleradius='rh'))
    baseline = parameters(local['pot_gal'], 'agama.Potential',
                          dict(density='agama.Density(densityStars, densityHalo)'))
    if (stars.get('type') != 'Sersic' or halo.get('type', '').lower() != 'spheroid'
            or baseline.get('type') != 'Multipole' or baseline.get('mmax') != 0):
        raise ValueError('Unsupported harness density/potential family')
    if any(isinstance(node, ast.Call) and ast.unparse(node.func) == 'agama.setUnits'
           for node in ast.walk(tree)):
        raise ValueError('Harness now sets units; review the G=1 reference')
    return dict(stellar=stars, halo=halo, baseline=baseline, q_ap=values['q_ap'],
                distance=values['D'], vscale=values['vscale'],
                bounds=ast.literal_eval(assignments['bounds_original']),
                upsilon_bounds=[values['Upsilon_lower'], values['Upsilon_upper']],
                source=str(path), sha256=sha256_file(path))


def read_models(path, recipe):
    path = Path(path).resolve()
    models = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        text, _, label = line.partition('#')
        if not text.strip():
            continue
        tokens = text.split()
        source = f'{path}:{number}'
        try:
            if len(tokens) not in (6, 9, 10, 11):
                raise ValueError('expected 6, 9, 10 or 11 columns')
            model = dict(zip(PARAMS, map(float, tokens[:6])))
            penalty = None if len(tokens) == 6 else float(tokens[6 if len(tokens) == 9 else 7])
            if not all(math.isfinite(value) for value in model.values()):
                raise ValueError('non-finite model parameter')
            for name, bounds in recipe['bounds'].items():
                if name in model and not bounds[0] <= model[name] <= bounds[1]:
                    raise ValueError(f'{name} outside harness bounds {bounds}')
            if not recipe['upsilon_bounds'][0] <= model['upsilon'] <= recipe['upsilon_bounds'][1]:
                raise ValueError('Upsilon outside harness bounds')
            if not (0 < model['incl'] <= 90 and
                    math.cos(math.radians(model['incl'])) < recipe['q_ap']):
                raise ValueError('inclination has no admissible stellar deprojection')
            if penalty is not None and (not math.isfinite(penalty) or penalty <= 0):
                raise ValueError('invalid reference penalty')
            model.update(source=source, label=label.strip(), line=line, reference_penalty=penalty)
            models.append(model)
        except ValueError as error:
            raise ValueError(f'{source}: {error}') from error
    if not models:
        raise ValueError(f'{path}: no model rows')
    return models


def density_parameters(model, recipe):
    angle = math.radians(model['incl'])
    q_star = math.sqrt(recipe['q_ap']**2 - math.cos(angle)**2) / math.sin(angle)
    return (dict(recipe['stellar'], axisRatioZ=q_star),
            dict(recipe['halo'], gamma=model['gh'], axisratioz=model['Q'],
                 scaleradius=model['rh'], densitynorm=model['rho0']))


def halo_density(r, model, recipe):
    p = recipe['halo']
    x = np.asarray(r) / model['rh']
    return (model['rho0'] * x**(-model['gh']) * (1+x**p['alpha'])**(
        (model['gh']-p['beta'])/p['alpha']) *
        np.exp(-(np.asarray(r)/p['outercutoffradius'])**p['cutoffstrength']))


def integral(function, lo, hi, rtol):
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(function, lo, hi, epsabs=1e-13, epsrel=rtol, limit=300)
    if not math.isfinite(value) or not math.isfinite(error):
        raise ValueError('Non-finite quadrature result')
    return value


def enclosed_mass(density, radius, rtol=1e-10):
    if radius <= 0 or not math.isfinite(radius):
        raise ValueError('Mass integral requires a finite positive radius')
    return 4*math.pi*radius**3 * integral(lambda t: density(radius*t)*t*t, 0, 1, rtol)


def spherical_field(points, model, recipe, rtol=1e-10):
    points = np.asarray(points, dtype=float)
    radii = np.linalg.norm(points, axis=1)
    if model['Q'] != 1 or np.any(radii <= 0) or not np.all(np.isfinite(points)):
        raise ValueError('Spherical reference requires Q=1 and finite, nonzero positions')
    p = recipe['halo']
    rh, rho, gh = model['rh'], model['rho0'], model['gh']
    alpha, beta, rc, strength = (p[key] for key in ('alpha', 'beta', 'outercutoffradius', 'cutoffstrength'))
    unique, inverse = np.unique(np.append(radii, 1.0), return_inverse=True)
    masses, potentials = [], []
    for r in unique:
        x = r/rh
        mass = 4*math.pi*rho*rh**gh*r**(3-gh) * integral(
            lambda t: t**(2-gh)*(1+(x*t)**alpha)**((gh-beta)/alpha)*
            math.exp(-(r*t/rc)**strength), 0, 1, rtol)
        outer = rho*rh**2 * integral(
            lambda t: t**(1-gh)*(1+t**alpha)**((gh-beta)/alpha)*
            math.exp(-(rh*t/rc)**strength), x, math.inf, rtol)
        masses.append(mass)
        potentials.append(-mass/r - 4*math.pi*outer)
    mass = np.asarray(masses)[inverse[:-1]]
    return dict(acc=-points * (mass/radii**3)[:, None],
                pot=np.asarray(potentials)[inverse[:-1]], pivot=potentials[inverse[-1]])


def read_grid(path):
    lines = Path(path).read_text().splitlines()
    try:
        count = int(next(line.split('=', 1)[1] for line in lines
                         if line.strip().lower().startswith('gridsizer=')))
        start, stop = lines.index('#Phi') + 1, lines.index('#dPhi/dr')
        grid = np.array([float(line.split()[0]) for line in lines[start:stop]
                         if line.strip() and not line.lstrip().startswith('#')])
        derivatives = np.array([float(line.split()[0]) for line in lines[stop+1:]
                                if line.strip() and not line.lstrip().startswith('#')])
        if (len(grid) != count or count < 2 or not np.all(np.isfinite(grid))
                or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
                or not np.array_equal(grid, derivatives)):
            raise ValueError('invalid or inconsistent radial nodes')
    except (ValueError, StopIteration, IndexError) as error:
        raise ValueError(f'Cannot read Multipole radial grid from {path}: {error}') from error
    return grid


def expand_grid(settings, factor):
    step = math.log(settings['rmax']/settings['rmin']) / (settings['gridSizeR']-1)
    lo, hi = settings['rmin']/factor, settings['rmax']*factor
    return dict(settings, rmin=lo, rmax=hi, gridSizeR=math.ceil(math.log(hi/lo)/step)+1)


def probe_points(grids, radial_count, angle_count):
    parts = [np.geomspace(PROBE_MIN, PROBE_MAX, radial_count), np.array([0.05, 1., 2.1, 10.])]
    for grid in grids:
        grid = np.asarray(grid)
        for values in (grid, np.sqrt(grid[:-1]*grid[1:])):
            parts.append(values[(values >= PROBE_MIN) & (values <= PROBE_MAX)])
    radii = np.unique(np.concatenate(parts))
    angles = np.linspace(0, math.pi/2, angle_count)
    directions = np.column_stack((np.sin(angles), np.zeros(angle_count), np.cos(angles)))
    directions[0, 0] = directions[-1, 2] = 0.
    return (radii[:, None, None]*directions[None, :, :]).reshape(-1, 3)


def sample_field(potential, points):
    return dict(acc=np.asarray(potential.force(points), dtype=float),
                pot=np.asarray(potential.potential(points), dtype=float),
                pivot=float(potential.potential([1., 0., 0.])))


def add_fields(*fields):
    return {key: sum(field[key] for field in fields) for key in ('acc', 'pot', 'pivot')}


def compare_fields(test, reference, points):
    points = np.asarray(points)
    n = len(points)
    for field in (test, reference):
        if (np.shape(field['acc']) != (n, 3) or np.shape(field['pot']) != (n,)
                or not all(np.all(np.isfinite(field[key])) for key in ('acc', 'pot', 'pivot'))):
            raise ValueError('Fields must have finite values and matching shapes')
    denominator = np.linalg.norm(reference['acc'], axis=1)
    if np.any(denominator <= 0):
        raise ValueError('Reference force vanishes at a probe point')
    delta = test['acc'] - reference['acc']
    relative = np.linalg.norm(delta, axis=1)/denominator
    delta_phi = (test['pot'] - test['pivot']) - (reference['pot'] - reference['pivot'])
    radii = np.linalg.norm(points, axis=1)
    errors = dict(relative=relative, component_scaled=delta/denominator[:, None],
                  acceleration_difference=delta, potential_difference=delta_phi,
                  potential_scaled=delta_phi/np.maximum(denominator*radii,
                                                       np.abs(reference['pot']-reference['pivot'])))
    scores = {}
    for name, (lo, hi) in REGIONS.items():
        indices = np.flatnonzero((radii >= lo) & (radii <= hi))
        if not len(indices):
            scores[name] = None
            continue
        values = relative[indices]
        worst = indices[np.argmax(values)]
        scores[name] = dict(n=len(indices), rms=float(np.sqrt(np.mean(values**2))),
                            max=float(values.max()), p50=float(np.median(values)),
                            p95=float(np.quantile(values, 0.95)),
                            worst_xyz=points[worst].tolist(), worst_radius=float(radii[worst]),
                            max_component_scaled=np.max(np.abs(errors['component_scaled'][indices]), axis=0).tolist(),
                            max_potential_scaled=float(np.max(np.abs(errors['potential_scaled'][indices]))))
    return scores, errors


def passes(score, rms_tol, max_tol):
    return score is not None and score['rms'] <= rms_tol and score['max'] <= max_tol


def verdict(reference_converged, score, rms_tol, max_tol):
    if not reference_converged:
        return 'reference_unconverged'
    return 'pass' if passes(score, rms_tol, max_tol) else 'needs_refinement'


def create_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def module_identity(agama):
    names = [name for name in sys.modules if name == 'agama' or name.startswith('agama.')]
    files = sorted({str(Path(module.__file__).resolve()) for name in names
                    if (module := sys.modules[name]) is not None and getattr(module, '__file__', None)
                    and Path(module.__file__).is_file()})
    return dict(version=str(getattr(agama, '__version__', getattr(agama, 'version', 'unknown'))),
                files=[dict(path=path, sha256=sha256_file(path)) for path in files],
                binary_fingerprint_present=any('.so' in Path(path).name or path.endswith('.pyd') for path in files))


def agama_self_test(agama, directory):
    points = np.array([[0.01, 0, 0], [0, 0.1, 0], [0.3, 0.4, 0],
                       [0, 0, 1.], [2., 3., 4.], [100., 0, 0]])
    r2 = np.sum(points**2, axis=1)
    exact = dict(acc=-points/(1+r2)[:, None]**1.5, pot=-1/np.sqrt(1+r2), pivot=-1/math.sqrt(2))
    analytic = agama.Potential(type='Plummer', mass=1., scaleRadius=1.)
    check = compare_fields(sample_field(analytic, points), exact, points)[0]
    if not passes(check['all'], 1e-10, 1e-9):
        raise ValueError('AGAMA Plummer force does not match the G=1 analytic reference')
    approximate = agama.Potential(type='Multipole', density=analytic, gridSizeR=92,
                                  lmax=0, mmax=0, rmin=1e-4, rmax=1e3)
    path = directory / 'smoke_plummer.ini'
    approximate.export(str(path))
    nodes = read_grid(path)
    measured = sample_field(approximate, points)
    check = compare_fields(measured, exact, points)[0]
    loaded = agama.Potential(file=str(path))
    roundtrip = compare_fields(sample_field(loaded, points), measured, points)[0]
    if not passes(check['all'], 1e-5, 1e-4) or not passes(roundtrip['all'], 1e-8, 1e-7):
        raise ValueError('AGAMA Multipole/force/export smoke test failed')
    return dict(status='pass', analytic_units='G=1', multipole=check, roundtrip=roundtrip,
                actual_grid=nodes.tolist())


def check_model(agama, model, recipe, args, directory):
    stars_params, halo_params = density_parameters(model, recipe)
    stars, halo = agama.Density(**stars_params), agama.Density(**halo_params)
    combined = agama.Density(stars, halo)
    records = {}

    def build(tag, density, settings):
        print(f"  {tag}: N={settings['gridSizeR']}, lmax={settings['lmax']}", flush=True)
        started = time.monotonic()
        potential = agama.Potential(density=density, **settings)
        path = directory / f'{tag}.ini'
        potential.export(str(path))
        grid = read_grid(path)
        metadata = dict(tag=tag, requested=settings, actual_grid=grid.tolist(),
                        coefficient_file=path.name, build_seconds=time.monotonic()-started)
        records[tag] = dict(potential=potential, meta=metadata, grid=grid)
        return records[tag]

    baseline = build('baseline', combined, dict(recipe['baseline']))
    star_base = build('stars_baseline', stars, dict(recipe['baseline']))
    halo_base = build('halo_baseline', halo, dict(recipe['baseline']))
    points = probe_points([record['grid'] for record in records.values()], args.radii, args.angles)

    def field(record):
        if 'field' not in record:
            record['field'] = sample_field(record['potential'], points)
        return record['field']

    def reference(density, base, prefix, spherical=False):
        grid = base['grid']
        settings = dict(recipe['baseline'], gridSizeR=args.radial_nodes[-1],
                        lmax=0 if spherical else args.angular_orders[-1],
                        rmin=min(float(grid[0]), PROBE_MIN/10),
                        rmax=max(float(grid[-1]), PROBE_MAX*10))
        coarse = build(prefix + '_ref_coarse', density, settings)
        radial_settings = dict(settings, gridSizeR=2*settings['gridSizeR']-1)
        radial = build(prefix + '_ref_radial', density, radial_settings)
        angular = radial if spherical else build(prefix + '_ref_angular', density,
                                                  dict(radial_settings, lmax=settings['lmax']+8))
        wide = build(prefix + '_ref_boundary', density, expand_grid(angular['meta']['requested'], 2.))
        checks = {}
        for name, low, high in (('radial', coarse, radial), ('angular', radial, angular),
                                ('boundary', angular, wide)):
            scores, _ = compare_fields(field(low), field(high), points)
            checks[name] = dict(scores=scores, passed=passes(scores['all'], args.rms_tol/10, args.max_tol/10))
        return wide, dict(converged=all(item['passed'] for item in checks.values()), checks=checks)

    star_ref, star_check = reference(stars, star_base, 'stars')
    halo_ref, halo_check = reference(halo, halo_base, 'halo', model['Q'] == 1.)
    if model['Q'] == 1.:
        halo_exact = spherical_field(points, model, recipe)
        tight = spherical_field(points, model, recipe, rtol=1e-12)
        scores, _ = compare_fields(halo_exact, tight, points)
        halo_reference_ok = passes(scores['all'], args.rms_tol/10, args.max_tol/10)
        halo_check['quadrature'] = dict(scores=scores, passed=halo_reference_ok)
        halo_check['reference_kind'] = 'independent spherical quadrature (G=1)'
        halo_reference = tight
    else:
        halo_reference_ok = halo_check['converged']
        halo_check['reference_kind'] = 'numerically converged Multipole, not analytic'
        halo_reference = field(halo_ref)
    ref = add_fields(field(star_ref), halo_reference)
    reference_ok = star_check['converged'] and halo_reference_ok
    grid = baseline['grid']
    frozen = dict(recipe['baseline'], rmin=float(grid[0]), rmax=float(grid[-1]))
    candidates = [('baseline', baseline)]
    for nodes in args.radial_nodes:
        if nodes != recipe['baseline']['gridSizeR']:
            record = build(f'auto_n{nodes}', combined, dict(recipe['baseline'], gridSizeR=nodes))
            candidates.append((record['meta']['tag'], record))
        record = build(f'radial_n{nodes}', combined, dict(frozen, gridSizeR=nodes))
        candidates.append((record['meta']['tag'], record))
    fixed = dict(frozen, gridSizeR=args.radial_nodes[-1])
    for order in args.angular_orders:
        record = build(f'angular_l{order}', combined, dict(fixed, lmax=order))
        candidates.append((record['meta']['tag'], record))
    boundary = dict(fixed, lmax=args.angular_orders[-1])
    for factor in (2., 4.):
        record = build(f'range_x{int(factor)}', combined, expand_grid(boundary, factor))
        candidates.append((record['meta']['tag'], record))
    star_settings, halo_settings = (record['meta']['requested'] for record in (star_ref, halo_ref))
    lo, hi = min(star_settings['rmin'], halo_settings['rmin']), max(star_settings['rmax'], halo_settings['rmax'])
    spacing = min(math.log(p['rmax']/p['rmin'])/(p['gridSizeR']-1) for p in (star_settings, halo_settings))
    common = build('common_fine', combined, dict(recipe['baseline'], rmin=lo, rmax=hi,
                   gridSizeR=math.ceil(math.log(hi/lo)/spacing)+1,
                   lmax=max(star_settings['lmax'], halo_settings['lmax'])))
    candidates.append(('common_fine', common))
    comparisons = {}
    curves = {}
    path = directory / 'errors.tsv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['variant', 'x', 'y', 'z', 'radius', 'relative_force',
                         'delta_ax', 'delta_ay', 'delta_az', 'component_x_scaled',
                         'component_y_scaled', 'component_z_scaled', 'delta_potential_difference',
                         'potential_scaled'])

        def compare(tag, measured, target, converged):
            scores, errors = compare_fields(measured, target, points)
            comparisons[tag] = dict(scores=scores, status=verdict(converged, scores['all'], args.rms_tol, args.max_tol),
                                    main_status=verdict(converged, scores['main'], args.rms_tol, args.max_tol))
            curves[tag] = errors['relative']
            for index, xyz in enumerate(points):
                writer.writerow([tag, *xyz, np.linalg.norm(xyz), errors['relative'][index],
                                 *errors['acceleration_difference'][index], *errors['component_scaled'][index],
                                 errors['potential_difference'][index], errors['potential_scaled'][index]])

        for tag, record in candidates:
            compare(tag, field(record), ref, reference_ok)
        compare('split_baseline', add_fields(field(star_base), field(halo_base)), ref, reference_ok)
        compare('stars_baseline', field(star_base), field(star_ref), star_check['converged'])
        compare('halo_baseline', field(halo_base), halo_reference, halo_reference_ok)
        compare('halo_ref_vs_reference', field(halo_ref), halo_reference, halo_reference_ok)
    eligible = [record for tag, record in candidates if comparisons[tag]['status'] == 'pass']
    recommended = min(eligible, key=lambda record: record['meta']['requested']['gridSizeR'] *
                      (record['meta']['requested']['lmax']+1)**2)['meta']['tag'] if eligible else None
    report = dict(status=comparisons['baseline']['status'], model=model,
                  density_parameters=dict(stars=stars_params, halo=halo_params),
                  units=dict(G=1, upsilon_applied=False, physical_acceleration_factor=model['upsilon']*recipe['vscale']**2),
                  reference=dict(converged=reference_ok, stars=star_check, halo=halo_check),
                  comparisons=comparisons, coefficients=[record['meta'] for record in records.values()],
                  probe_points=len(points), thresholds=dict(rms=args.rms_tol, maximum=args.max_tol,
                                                           reference_rms=args.rms_tol/10, reference_maximum=args.max_tol/10),
                  field_only_candidate=recommended, penalty_checked=False,
                  summary_scope='All sampled radii/directions, not all possible stellar orbits; no statistical confidence.')
    write_json(directory / 'report.json', report)
    if not args.no_plots:
        plot_results(directory, points, curves, records)
    return report


def plot_results(directory, points, curves, records):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    radius = np.linalg.norm(points, axis=1)
    log_radii, inverse = np.unique(np.round(np.log10(radius), 12), return_inverse=True)
    radii = 10**log_radii
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, prefix in zip(axes, ('radial_', 'angular_')):
        for tag, values in curves.items():
            if tag == 'baseline' or tag.startswith(prefix) or tag == 'split_baseline':
                maxima = np.zeros(len(radii))
                np.maximum.at(maxima, inverse, values)
                ax.loglog(radii, np.maximum(maxima, 1e-16), label=tag)
        ax.axvspan(0.05, 2.1, color='grey', alpha=0.1)
        ax.set(xlabel='r [kpc]', ylabel='max over sampled angles: relative force error')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(directory / 'force_errors.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    dots = ax.scatter(points[:, 0], points[:, 2], c=np.log10(np.maximum(curves['baseline'], 1e-16)), s=9)
    ax.set(xscale='symlog', yscale='symlog', xlabel='R [kpc]', ylabel='z [kpc]')
    fig.colorbar(dots, ax=ax, label='log10 relative force error (baseline)')
    fig.tight_layout()
    fig.savefig(directory / 'error_map.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    tags = ['baseline', 'stars_baseline', 'halo_baseline', 'common_fine']
    for index, tag in enumerate(tags):
        grid = records[tag]['grid']
        ax.plot(grid, np.full(len(grid), index), '|', label=tag)
    ax.axvspan(0.05, 2.1, color='grey', alpha=0.1)
    ax.set(xscale='log', xlabel='actual radial nodes [kpc]', yticks=range(len(tags)), yticklabels=tags)
    fig.tight_layout()
    fig.savefig(directory / 'radial_grids.png', dpi=150)
    plt.close(fig)


def increasing_ints(text):
    try:
        values = tuple(int(part) for part in text.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from error
    if len(values) < 2 or any(a >= b for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError('need at least two strictly increasing integers')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--models', type=Path, help='Local rows: incl Q gh rh rho0 Upsilon, or history/J rows')
    parser.add_argument('--harness', type=Path, default=DEFAULT_HARNESS, help='Read via AST only, never imported')
    parser.add_argument('--output', type=Path, help='New directory only; default runs/<UTC timestamp>')
    parser.add_argument('--validate-inputs', action='store_true', help='Validate recipe/models without AGAMA or output files')
    parser.add_argument('--self-test', action='store_true', help='Only local AGAMA Plummer/Multipole/export smoke test')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--radial-nodes', type=increasing_ints, default=(23, 46, 92, 184))
    parser.add_argument('--angular-orders', type=increasing_ints, default=(4, 8, 12, 16, 24))
    parser.add_argument('--radii', type=int, default=96)
    parser.add_argument('--angles', type=int, default=9)
    parser.add_argument('--rms-tol', type=float, default=1e-4)
    parser.add_argument('--max-tol', type=float, default=1e-3)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test and (args.models or args.validate_inputs):
        parser.error('--self-test is separate from --models/--validate-inputs')
    if not args.self_test and args.models is None:
        parser.error('--models is required unless --self-test is selected')
    if args.threads < 1 or args.radii < 16 or args.angles < 3:
        parser.error('require threads >= 1, radii >= 16, angles >= 3')
    if not (math.isfinite(args.rms_tol) and math.isfinite(args.max_tol) and 0 < args.rms_tol <= args.max_tol):
        parser.error('require finite 0 < rms-tol <= max-tol')
    if not 2 <= args.radial_nodes[0] <= args.radial_nodes[-1] <= 2048:
        parser.error('radial nodes must be in [2, 2048]')
    if args.angular_orders[0] < 0 or args.angular_orders[-1] > 56 or any(l % 2 for l in args.angular_orders):
        parser.error('angular orders must be even integers in [0, 56]')
    return args


def main(argv=None):
    args = parse_args(argv)
    report = dict(status='running', started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  implementation_sha256=sha256_file(__file__), python=sys.executable,
                  platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                  options={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  models=[], penalty_checked=False)
    output = None
    try:
        recipe = read_recipe(args.harness) if not args.self_test else None
        models = read_models(args.models, recipe) if not args.self_test else []
        if args.validate_inputs:
            print(json.dumps(dict(status='validated', recipe=recipe, models=models), ensure_ascii=False, indent=2))
            return 0
        output = create_output(args.output or HERE / 'runs' / datetime.datetime.now(
            datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        report['recipe'] = recipe
        if args.models:
            report['models_sha256'] = sha256_file(args.models)
        write_json(output / 'report.json', report)
        os.environ['MPLCONFIGDIR'] = str(output / '.matplotlib')
        agama = importlib.import_module('agama')
        report['agama'] = module_identity(agama)
        with agama.setNumThreads(args.threads):
            report['smoke_test'] = agama_self_test(agama, output)
            write_json(output / 'report.json', report)
            for index, model in enumerate(models):
                directory = create_output(output / f'model_{index:03d}')
                print(f'Model {index}: {model["source"]}', flush=True)
                entry = dict(directory=directory.name, model=model, status='running')
                report['models'].append(entry)
                write_json(output / 'report.json', report)
                result = check_model(agama, model, recipe, args, directory)
                entry.update(status=result['status'], field_only_candidate=result['field_only_candidate'])
                write_json(output / 'report.json', report)
        statuses = {item['status'] for item in report['models']}
        report['status'] = ('reference_unconverged' if 'reference_unconverged' in statuses else
                            'needs_refinement' if 'needs_refinement' in statuses else 'pass')
        report['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(output / 'report.json', report)
        print(f'{report["status"]}: {output / "report.json"}')
        return 0 if report['status'] == 'pass' else 3
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if report['models'] and report['models'][-1]['status'] == 'running':
            report['models'][-1].update(status='failed', error=report['error'])
        if output is not None:
            write_json(output / 'report.json', report)
        print(report['error'], file=sys.stderr)
        if isinstance(error, ModuleNotFoundError) and error.name == 'agama':
            print('Use the Python environment with working AGAMA (e.g. user gala); no Docker is required.', file=sys.stderr)
        return 2 if isinstance(error, (ValueError, FileExistsError)) and output is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
