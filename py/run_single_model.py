#!/usr/bin/env python3
"""Single-model check of the experimental harness (not production).

Re-evaluates one halo model with Fornax_P21_PCA_w3Sersic_orblib_exp.py as a
library: one orbit integration, the orbit library is saved through the
memory-fixed path (claim -> save_slot -> block writer + HashingWriter ->
register), then several Upsilon protocols run on the same library:

  exp    module defaults (xatol 5e-3, 25 % sub-sample, final full solve);
         integrates and saves; its row goes to the normal history file
         (shared d1 pool), exactly as the search would write it;
  reuse  the same settings on the reloaded float64 library (storage round trip);
  prod   production-like search: xatol 1e-3, full library, min_pen = res.fun.

reuse/prod rows go to a side file that the pool globs do not match. A JSON +
text report is written next to them. `--summarize` aggregates reports and
needs only the standard library (runs on the host).

IC-seed scan (`--ic-seeds 1-100`, diagnostic only, DECISIONS 2026-10-04):
one module import, then for every seed K agama.setRandomSeed(K) and one `exp`
evaluation of the same model. No orbit library is saved or reused. Rows go to
a seed history file (`# ic_seed: K` before each block) that the pool globs do
not match, plus a per-seed .tsv; seeds already complete there are skipped.

Default parameters: the minimum-penalty row at --incl of the production
free-Q history 4UpsBoTorch_PCA_Sersic_*.txt in the working directory, resolved
at run time (no result numbers live in this public repository). See
doc/ai/harness/orblib_exp.md, section "Single-model check".
"""
import argparse
import datetime
import json
import math
import os
import socket
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_SCRIPT = os.path.join(HERE, 'Fornax_P21_PCA_w3Sersic_orblib_exp.py')

DEFAULT_PARAMS_FROM = ['4UpsBoTorch_PCA_Sersic_*.txt']
PARAM_KEYS = ('Q', 'gh', 'rh', 'rho0')

# Overrides relative to the module defaults (None = keep the module value).
PROTOCOLS = {
    'exp':   dict(xatol=None, subsample=None, target='pool'),
    'reuse': dict(xatol=None, subsample=None, target='side'),
    'prod':  dict(xatol=1e-3, subsample=0.0, target='side'),
}
FAILED_PENALTY = 1e5
EXPECTED_PEAK_MB = 2000.0
SEED_TAG = '# ic_seed:'
SEED_MAX = 2**31 - 1
SEED_TABLE_COLUMNS = ('seed', 'status', 'penalty', 'upsilon', 'probes', 'sample_s', 'orbit_s',
                      'solveopt_total_s', 'wall_s', 'peak_rss_mb', 'omp_threads', 'finished')


class ProtocolError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# IC seeds
# --------------------------------------------------------------------------
def parse_seed_list(spec):
    """'1-100' or '1,5,9-12' -> list of ints. AGAMA's setRandomSeed(0) takes the
    seed from the clock, so 0 (and negatives, repeats) are rejected."""
    seeds = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        low, dash, high = part.partition('-')
        try:
            low, high = int(low), int(high) if dash else int(low)
        except ValueError:
            raise ValueError(f'bad seed item {part!r} (integers >= 1, ranges A-B)')
        if low < 1 or high < low or high > SEED_MAX:
            raise ValueError(f'bad seed item {part!r} (integers in [1, {SEED_MAX}], ranges A-B with A <= B)')
        seeds.extend(range(low, high + 1))
    if not seeds:
        raise ValueError('empty seed list')
    if len(set(seeds)) != len(seeds):
        raise ValueError('repeated seeds')
    return seeds


def split_seeds(seeds, workers):
    """Round-robin: worker j gets seeds[j], seeds[j+W], ..."""
    if not 1 <= workers <= len(seeds):
        raise ValueError(f'{workers} workers for {len(seeds)} seeds')
    return [seeds[j::workers] for j in range(workers)]


def mark_seed(path, seed):
    """`# ic_seed: K` on its own line right before the block the module writes."""
    with open(path, 'a+b') as stream:
        stream.seek(0, os.SEEK_END)
        lead = b''
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            lead = b'' if stream.read(1) == b'\n' else b'\n'
        stream.write(lead + f'{SEED_TAG} {seed} (agama.setRandomSeed before the orbit-IC sampling)\n'.encode())
        stream.flush()
        os.fsync(stream.fileno())


def read_seed_history(path):
    """Seeds whose block is complete (data row + '# End of history') in `path`."""
    done, seed, row = {}, None, None
    try:
        with open(path, errors='replace') as stream:
            lines = stream.read().splitlines()
    except FileNotFoundError:
        return done
    for line in lines:
        if line.startswith(SEED_TAG):
            try:
                seed, row = int(line[len(SEED_TAG):].split()[0]), None
            except (ValueError, IndexError):
                seed, row = None, None
        elif line.startswith('# End of history'):
            if seed is not None and row is not None and 0 < row['penalty'] < FAILED_PENALTY:
                done[seed] = row
            seed, row = None, None
        elif seed is not None and row is None and line.strip() and not line.startswith('#'):
            try:
                values = [float(t) for t in line.split()[:7]]
            except ValueError:
                continue
            if len(values) == 7 and all(math.isfinite(v) for v in values):
                row = dict(upsilon=values[5], penalty=values[6])
    return done


def seed_table_path(seed_file):
    return os.path.splitext(seed_file)[0] + '.tsv'


def append_seed_table(path, result):
    new = not os.path.exists(path)
    with open(path, 'a') as stream:
        if new:
            stream.write('\t'.join(SEED_TABLE_COLUMNS) + '\n')
        stream.write('\t'.join('' if result.get(k) is None else str(result.get(k))
                               for k in SEED_TABLE_COLUMNS) + '\n')


# --------------------------------------------------------------------------
# Reference rows
# --------------------------------------------------------------------------
# Data layouts (token counts):
#   out_* / 4Ups*      incl Q gh rh rho0 Ups penalty DATE TIME            (9)
#   J_factor_*         incl Q gh rh rho0 Ups rho0xUps penalty J log10J   (10)
#   Jcomputed_from_*   the same + count                                  (11)
_PENALTY_COLUMN = {9: 6, 10: 7, 11: 7}


def parse_reference_line(line):
    if not line.strip() or line.lstrip().startswith('#'):
        return None
    tokens = line.split()
    column = _PENALTY_COLUMN.get(len(tokens))
    if column is None:
        return None
    try:
        values = [float(t) for t in tokens[:column + 1]]
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    row = dict(incl=values[0], Q=values[1], gh=values[2], rh=values[3],
               rho0=values[4], upsilon=values[5], penalty=values[column])
    if row['penalty'] <= 0 or row['upsilon'] <= 0:
        return None
    return row


def best_reference(paths, incl):
    best = None
    for path in sorted(paths):
        with open(path, errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                row = parse_reference_line(line)
                if row is None or abs(row['incl'] - incl) > 1e-6:
                    continue
                if best is None or row['penalty'] < best['penalty']:
                    best = dict(row, source=f'{os.path.basename(path)}:{number}',
                                line=line.rstrip('\n'))
    return best


def script_bounds(path=None):
    """Global bounds_original of the experimental script, read with ast
    (standard library only: --list-models runs on the host)."""
    import ast
    path = path or EXP_SCRIPT
    with open(path) as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'bounds_original' for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f'{path}: no global bounds_original')


def bounds_violations(params, bounds):
    """The evaluation clips parameters to bounds_original silently, so a point
    outside them would run (and be recorded) as a different model."""
    return [f'{key}={params[key]!r} outside [{bounds[key][0]!r}, {bounds[key][1]!r}]'
            for key in PARAM_KEYS if not bounds[key][0] <= params[key] <= bounds[key][1]]


def read_models(path, bounds=None):
    """Models file for launch_multi_model.sh: one history row per container
    (any layout above), optional trailing '# label' used as the source.
    Rows outside bounds_original are refused."""
    models = []
    with open(path, errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            text, _, label = line.partition('#')
            row = parse_reference_line(text)
            if row is not None:
                row['source'] = label.strip() or f'{os.path.basename(path)}:{number}'
                models.append(row)
    keys = [tuple(row[k] for k in ('incl',) + PARAM_KEYS) for row in models]
    if len(set(keys)) != len(keys):
        raise SystemExit(f'{path}: duplicate parameter sets (one library name, one AGAMA realisation)')
    bounds = script_bounds() if bounds is None else bounds
    outside = [f"{row['source']}: {', '.join(bounds_violations(row, bounds))}"
               for row in models if bounds_violations(row, bounds)]
    if outside:
        raise SystemExit(f'{path}: rows outside bounds_original of {os.path.basename(EXP_SCRIPT)} '
                         '(they would be clipped): ' + '; '.join(outside))
    return models


# --------------------------------------------------------------------------
# History block written by halo_IC_lib_weights_pca_fixed
# --------------------------------------------------------------------------
_COMMENT_KEYS = (('orbitlib times', 'orbitlib'), ('orblib saved:', 'saved'),
                 ('orblib reused:', 'reused'), ('solveOpt summary:', 'solveopt'),
                 ('orblib archive-first-wins:', 'archived_rebuild'))


def _key_values(text):
    result = {}
    for token in text.split():
        if '=' not in token:
            continue
        key, value = token.strip('()').split('=', 1)
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def parse_last_block(path, server):
    """Last history block of `server` in `path` (row + timing comments)."""
    try:
        with open(path, errors='replace') as stream:
            lines = stream.read().splitlines()
    except FileNotFoundError:
        return None
    starts = [i for i, line in enumerate(lines) if line.strip() == f'# Server: {server}']
    if not starts:
        return None
    block = {}
    for line in lines[starts[-1] + 1:]:
        if line.startswith('# End of history'):
            break
        if not line.startswith('#'):
            if 'row' not in block and line.strip():
                block['row'] = [float(t) for t in line.split()[:7]]
            continue
        text = line[1:].strip()
        for prefix, key in _COMMENT_KEYS:
            if text.startswith(prefix):
                block[key] = _key_values(text[len(prefix):])
                if key in ('saved', 'reused'):
                    block[key]['name'] = text[len(prefix):].split()[0]
    return block if 'row' in block else None


# --------------------------------------------------------------------------
# Module preparation and protocol driver (work on any module-like object)
# --------------------------------------------------------------------------
def module_argv(args):
    # Seed scan: neither save nor reuse — every seed is a new realisation of one
    # library name, and nothing of it is kept.
    storage = [] if args.ic_seeds else ['--save-orblib', '--reuse-orblib']
    argv = [EXP_SCRIPT, '--incl', repr(float(args.incl)), '--suffix', args.suffix,
            *storage, '--orblib-dir', args.orblib_dir, '--no-resume']
    if args.n_threads is not None:
        argv += ['--n_threads', str(args.n_threads)]
    return argv


def _no_checkpoint():
    """The runner keeps no search checkpoint; replacing save_initial_checkpoint
    also skips the per-evaluation checkpoint write at the end of the call."""


def prepare_module(mod, store):
    mod.orblib_store = store
    # Without this every point already in the pool (incl. our own exp row and
    # the other realisations) would raise OrblibBusyError('completed point').
    mod.completed_point = lambda params: False
    mod.checkpoint_for_stop = _no_checkpoint
    # Initialised only inside run_pca_optimization in the module.
    mod.best_overall_target = -float('inf')
    mod.best_overall_Upsilon = None
    mod.number_of_h_IC_lw = 0
    mod.number_of_find_w_U = 0


def _peak_rss_reset():
    try:
        with open('/proc/self/clear_refs', 'w') as stream:
            stream.write('5')
    except OSError:
        pass


def _peak_rss_mb():
    try:
        with open('/proc/self/status') as stream:
            for line in stream:
                if line.startswith('VmHWM:'):
                    return float(line.split()[1]) / 1e3
    except (OSError, ValueError, IndexError):
        pass
    return None


def run_protocol(mod, name, params, files, seed, rng_factory, ic_seed=None):
    settings = PROTOCOLS[name]
    saved = {key: getattr(mod, key) for key in ('UPS_XATOL', 'UPS_SUBSAMPLE_FRAC', 'UpsFile')}
    target = files[settings['target']]
    try:
        if settings['xatol'] is not None:
            mod.UPS_XATOL = settings['xatol']
        if settings['subsample'] is not None:
            mod.UPS_SUBSAMPLE_FRAC = settings['subsample']
        mod.UpsFile = target
        # Empty history => full Upsilon bracket, as for a worker's first model.
        del mod._ups_recent[:]
        mod.proc_rng = rng_factory(seed)
        effective = dict(xatol=mod.UPS_XATOL, subsample=mod.UPS_SUBSAMPLE_FRAC)
        pc = mod._params_to_dummy_pc(dict(params), None, mod.bounds_original)
        probes = mod.number_of_find_w_U
        _peak_rss_reset()
        if ic_seed is not None:
            # Resets every per-thread AGAMA stream; densityStars.sample() is the
            # first consumer inside the evaluation ⇒ the realisation depends on
            # the seed only (not on order or on earlier evaluations).
            mod.agama.setRandomSeed(int(ic_seed))
        start = time.perf_counter()
        value = mod.halo_IC_lib_weights_pca_fixed(
            pc, None, mod.bounds_original, mod.densityStars, mod.datasets,
            mod.alphah, mod.betah, direct_params=dict(params))
        wall = time.perf_counter() - start
    finally:
        for key, value_saved in saved.items():
            setattr(mod, key, value_saved)
    penalty = -float(value)
    block = parse_last_block(target, mod.hostname_proc)
    if not math.isfinite(penalty) or penalty >= FAILED_PENALTY or block is None:
        raise ProtocolError(f'{name}: evaluation failed (penalty={penalty})')
    orbitlib = block.get('orbitlib', {})
    result = dict(
        penalty=penalty, upsilon=block['row'][5], probes=mod.number_of_find_w_U - probes,
        wall_s=wall, peak_rss_mb=_peak_rss_mb(), file=target,
        settings=effective, reused='reused' in block, saved='saved' in block,
        archived_rebuild='archived_rebuild' in block,
        sample_s=orbitlib.get('sample_s'), orbit_s=orbitlib.get('orbit_s'),
        omp_threads=orbitlib.get('omp_threads'), ic_seed=ic_seed,
        save_s=block.get('saved', {}).get('save_s'),
        size_mb=block.get('saved', {}).get('size_MB'),
        load_s=block.get('reused', {}).get('load_s'),
        solveopt_total_s=block.get('solveopt', {}).get('total_s'))
    if name != 'exp' and not result['reused']:
        raise ProtocolError(f'{name}: library was not reused; the comparison would mix realisations')
    return result


def compute_deltas(report):
    protocols, ref = report['protocols'], report['reference']['penalty']
    penalty = {name: value['penalty'] for name, value in protocols.items()}
    deltas = {}
    for name in ('exp', 'prod', 'reuse'):
        if name in penalty and ref is not None:
            deltas[f'{name}_minus_ref'] = penalty[name] - ref
    for a, b in (('reuse', 'exp'), ('prod', 'exp')):
        if a in penalty and b in penalty:
            deltas[f'{a}_minus_{b}'] = penalty[a] - penalty[b]
    return deltas


def check_library(storage, store, name, path):
    info = dict(name=name, path=path, exists=os.path.exists(path))
    if not info['exists']:
        return info
    info['size'] = os.path.getsize(path)
    info['md5'] = storage.file_hash(path)
    try:
        info['metadata'] = storage.metadata(path, deep=True)
        info['metadata_ok'] = True
    except Exception as error:
        info['metadata_ok'], info['metadata_error'] = False, repr(error)
    row = store.row(name) if store is not None else None
    if row is not None:
        info['store_state'], info['store_size'], info['store_md5'] = row['state'], row['size'], row['md5']
        info['store_consistent'] = (row['size'] == info['size'] and row['md5'] == info['md5'])
    return info


def write_report(report, path):
    report['updated'] = datetime.datetime.now().isoformat(timespec='seconds')
    temporary = f'{path}.{os.getpid()}.tmp'
    with open(temporary, 'w') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    text = os.path.splitext(path)[0] + '.txt'
    with open(text, 'w') as stream:
        stream.write(format_report(report))


def format_report(report):
    ref = report['reference']
    lines = [f"single-model check {report.get('suffix')} status={report.get('status')}",
             f"  params: " + ' '.join(f"{k}={report['params'][k]:.10g}" for k in ('Q', 'gh', 'rh', 'rho0'))
             + f" incl={report['incl']}",
             f"  reference: penalty={ref['penalty']} Upsilon={ref['upsilon']} ({ref['source']})"]
    grid = report.get('grid')
    if grid:
        lines.append(f"  grid identical to production formula: {grid['identical']} "
                     f"(exp x/y max={grid['exp']['gridx_max']:.6f}/{grid['exp']['gridy_max']:.6f}, "
                     f"prod={grid['prod']['gridx_max']:.6f}/{grid['prod']['gridy_max']:.6f})")
    for name, value in report.get('protocols', {}).items():
        lines.append(f"  {name:6s} penalty={value['penalty']:.6f} Upsilon={value['upsilon']:.5f} "
                     f"probes={value['probes']} wall={value['wall_s']:.1f}s "
                     f"peak_rss={value['peak_rss_mb']} MB reused={value['reused']} saved={value['saved']}")
    for key, value in report.get('deltas', {}).items():
        lines.append(f"  {key} = {value:+.6f}")
    scan = report.get('ic_seeds')
    if scan:
        results = scan.get('results', {})
        ok = sum(1 for value in results.values() if value.get('status') == 'ok')
        lines.append(f"  ic_seeds: {ok}/{len(scan.get('requested', []))} ok "
                     f"(seed file {report.get('context', {}).get('seed_file')})")
        for key in sorted(results, key=int):
            value = results[key]
            penalty = value.get('penalty')
            lines.append(f"    seed {int(key):6d} {value.get('status'):6s} "
                         + (f"penalty={penalty:.6f} Upsilon={value.get('upsilon'):.5f} "
                            f"orbit_s={value.get('orbit_s')} wall_s={value.get('wall_s')}"
                            if penalty is not None else str(value.get('error')))
                         + (' (resumed)' if value.get('resumed') else ''))
    library = report.get('orblib', {})
    if library and not scan:
        lines.append(f"  orblib {library.get('name')} exists={library.get('exists')} "
                     f"size={library.get('size')} md5={library.get('md5')} "
                     f"metadata_ok={library.get('metadata_ok')} store_consistent={library.get('store_consistent')}")
    if report.get('error'):
        lines.append(f"  error: {report['error']}")
    return '\n'.join(lines) + '\n'


def execute(mod, args, report, store, rng_factory, storage):
    params = report['params']
    name, path = report['orblib']['name'], report['orblib']['path']
    files = dict(pool=mod.UpsFile, side=args.side_file or f'single_{mod.hostname_proc}.txt')
    report['context'].update(pool_file=files['pool'], side_file=files['side'])
    code = 0
    try:
        for protocol in args.protocols:
            if protocol != 'exp' and not os.path.exists(path):
                raise ProtocolError(f'{protocol}: library {name} is not local')
            print(f'[single] protocol {protocol}', flush=True)
            report['protocols'][protocol] = run_protocol(
                mod, protocol, params, files, args.subsample_seed, rng_factory)
            report['deltas'] = compute_deltas(report)
            write_report(report, args.report)
            if store.stopped():
                raise storage.StorageStop('stop requested during the single-model check')
        report['status'] = 'ok'
    except storage.StorageStop as error:
        report['status'], report['error'], code = 'stopped', error.reason, 75
    except Exception as error:
        traceback.print_exc()
        report['status'], report['error'], code = 'failed', repr(error), 1
    report['orblib'].update(check_library(storage, store, name, path))
    if report['status'] == 'ok' and not (report['orblib'].get('metadata_ok')
                                         and report['orblib'].get('store_consistent', True)):
        report['status'], code = 'failed', 1
        report['error'] = 'saved library failed the metadata/store check'
    report['deltas'] = compute_deltas(report)
    write_report(report, args.report)
    print(format_report(report), end='', flush=True)
    return code


def execute_seeds(mod, args, report, store, rng_factory, storage):
    """IC-seed scan: the `exp` evaluation of one model for every seed, in one process."""
    params, scan = report['params'], report['ic_seeds']
    seed_file = args.seed_file or f'seeds_{mod.hostname_proc}.txt'
    table = seed_table_path(seed_file)
    files = dict(pool=seed_file, side=seed_file)
    report['context'].update(seed_file=seed_file, seed_table=table)
    results = scan['results']
    for seed, row in read_seed_history(seed_file).items():
        if seed in scan['requested']:
            results[str(seed)] = dict(row, ic_seed=seed, status='ok', resumed=True)
    code = 0
    try:
        for seed in scan['requested']:
            if results.get(str(seed), {}).get('status') == 'ok':
                print(f'[seeds] ic_seed={seed} already in {seed_file} — skipped', flush=True)
                continue
            store.check_stop()
            print(f'[seeds] ic_seed={seed}', flush=True)
            mark_seed(seed_file, seed)
            try:
                result = dict(run_protocol(mod, 'exp', params, files, args.subsample_seed,
                                           rng_factory, ic_seed=seed), status='ok')
            except Exception as error:
                traceback.print_exc()
                result = dict(ic_seed=seed, status='failed', error=repr(error))
            result['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
            results[str(seed)] = result
            append_seed_table(table, dict(result, seed=seed))
            write_report(report, args.report)
            store.check_stop()
        failed = [s for s in scan['requested'] if results.get(str(s), {}).get('status') != 'ok']
        report['status'] = 'failed' if failed else 'ok'
        report['error'] = f'failed seeds: {failed}' if failed else None
        code = 1 if failed else 0
    except storage.StorageStop as error:
        report['status'], report['error'], code = 'stopped', error.reason, 75
    except Exception as error:
        traceback.print_exc()
        report['status'], report['error'], code = 'failed', repr(error), 1
    write_report(report, args.report)
    print(format_report(report), end='', flush=True)
    return code


def static_context(mod):
    import numpy
    exp = dict(gridx_max=float(mod.gridx_max), gridy_max=float(mod.gridy_max))
    prod = dict(gridx_max=float(mod.bound_circR[1][-1] + mod.gridx_min),
                gridy_max=float(mod.bound_circR[3][-1] * mod.q_ap + mod.gridy_min))
    identical = all(abs(exp[k] - prod[k]) <= 1e-12 for k in exp)
    context = dict(exp_id=mod.EXP_ID, hostname_proc=mod.hostname_proc, geom_hash=mod.GEOM_HASH,
                   evaluation_context=mod.EVALUATION_CONTEXT,
                   num_dof=int(sum(int(numpy.sum(d.cons_err > 0)) for d in mod.datasets)),
                   n_apertures=len(mod.sectAPP), double=bool(mod.DOUBLE), n_bin=int(mod.N_BIN))
    return context, dict(exp=exp, prod=prod, identical=identical)


def resolve_reference(args):
    explicit = [key for key in PARAM_KEYS if getattr(args, key) is not None]
    if len(explicit) == len(PARAM_KEYS) and not args.params_from:
        ref = dict(incl=args.incl, upsilon=None, penalty=None, source='command line', line=None)
    else:
        import glob
        patterns = args.params_from or DEFAULT_PARAMS_FROM
        paths = [p for pattern in patterns for p in glob.glob(pattern)]
        ref = best_reference(paths, args.incl)
        if ref is None:
            raise SystemExit(f'No valid row with incl={args.incl} in {patterns}; '
                             'pass --Q/--gh/--rh/--rho0 (and --ref-penalty) explicitly')
        print(f"[single] reference row {ref['source']}: {ref['line']}", flush=True)
        if explicit:
            ref['source'] += ' (overridden: ' + ','.join(explicit) + ')'
    for key in explicit:
        ref[key] = getattr(args, key)
    if args.ref_penalty is not None:
        ref['penalty'] = args.ref_penalty
    if args.ref_upsilon is not None:
        ref['upsilon'] = args.ref_upsilon
    if args.ref_source is not None:
        ref['source'] = args.ref_source
    return ref


def base_report(args, ref):
    return dict(status='started', error=None, host=os.environ.get('HOSTNAME_SUFFIX', socket.gethostname()),
                suffix=args.suffix, incl=args.incl,
                params={k: ref[k] for k in ('Q', 'gh', 'rh', 'rho0')},
                reference=dict(penalty=ref.get('penalty'), upsilon=ref.get('upsilon'),
                               source=ref.get('source'), line=ref.get('line')),
                subsample_seed=args.subsample_seed, protocols={}, deltas={},
                ic_seeds=dict(requested=list(args.ic_seeds), results={}) if args.ic_seeds else None,
                context={}, grid=None, orblib={},
                started=datetime.datetime.now().isoformat(timespec='seconds'))


def run(args):
    import importlib.util
    import numpy
    ref = resolve_reference(args)
    report = base_report(args, ref)
    sys.argv = module_argv(args)
    spec = importlib.util.spec_from_file_location('orblib_exp', EXP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['orblib_exp'] = mod
    spec.loader.exec_module(mod)
    outside = bounds_violations(report['params'], mod.bounds_original)
    if outside:
        report['status'] = 'failed'
        report['error'] = 'outside bounds_original (would be clipped): ' + '; '.join(outside)
        write_report(report, args.report)
        print(format_report(report), end='', flush=True)
        return 2
    import orblib_storage
    report['context'], report['grid'] = static_context(mod)
    name, path = mod.orblib_key(*(report['params'][k] for k in ('Q', 'gh', 'rh', 'rho0')))
    report['orblib'] = dict(name=name, path=path, exists=os.path.exists(path))
    if args.ic_seeds:
        report['orblib'].update(stored=False)
    if args.preflight:
        report['status'] = 'preflight'
        write_report(report, args.report)
        print(format_report(report), end='', flush=True)
        return 0
    store = orblib_storage.Store(mod.ORBLIB_DIR,
                                 int(os.environ.get('ORBLIB_RESERVE_BYTES', '2000000000')),
                                 save_slots=int(os.environ.get('ORBLIB_SAVE_SLOTS', '2')))
    prepare_module(mod, store)
    driver = execute_seeds if args.ic_seeds else execute
    return driver(mod, args, report, store, numpy.random.default_rng, orblib_storage)


# --------------------------------------------------------------------------
# Host-side summary (standard library only)
# --------------------------------------------------------------------------
def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values), max=max(values))


def summarize(paths):
    reports = []
    for path in sorted(paths):
        with open(path) as stream:
            reports.append(json.load(stream))
    ref = next((r['reference']['penalty'] for r in reports if r.get('reference')), None)
    summary = dict(reports=len(reports), reference_penalty=ref,
                   status={r.get('suffix'): r.get('status') for r in reports},
                   grid_identical=sorted({r['grid']['identical'] for r in reports if r.get('grid')}),
                   orblib_names=sorted({r['orblib'].get('name') for r in reports if r.get('orblib')} - {None}))
    for protocol in PROTOCOLS:
        summary[protocol] = _stats([r['protocols'].get(protocol, {}).get('penalty') for r in reports])
    for key in ('reuse_minus_exp', 'prod_minus_exp', 'prod_minus_ref', 'exp_minus_ref'):
        summary[key] = _stats([r.get('deltas', {}).get(key) for r in reports])
    peaks = [v.get('peak_rss_mb') for r in reports for v in r.get('protocols', {}).values()]
    peaks += [v.get('peak_rss_mb') for r in reports
              for v in ((r.get('ic_seeds') or {}).get('results') or {}).values()]
    summary['peak_rss_mb'] = _stats(peaks)
    summary['seed_scan'] = summarize_seeds(reports, ref)
    summary['save_s'] = _stats([r['protocols'].get('exp', {}).get('save_s') for r in reports])
    exp, prod = summary['exp'], summary['prod']
    scatter = (prod or {}).get('std') or (exp or {}).get('std')
    if prod and ref is not None and scatter:
        summary['prod_minus_ref_in_scatter'] = (prod['mean'] - ref) / scatter
    worst = (summary['peak_rss_mb'] or {}).get('max')
    summary['memory_verdict'] = (None if worst is None else
                                 'within expected' if worst <= EXPECTED_PEAK_MB else
                                 f'above expected {EXPECTED_PEAK_MB:.0f} MB (see per-protocol peaks)')
    return reports, summary


def _parse_time(text):
    try:
        return datetime.datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def summarize_seeds(reports, ref):
    scans = [r for r in reports if r.get('ic_seeds')]
    if not scans:
        return None
    rows, owners = [], {}
    for r in scans:
        for key, value in r['ic_seeds'].get('results', {}).items():
            row = dict(seed=int(key), suffix=r.get('suffix'),
                       **{k: value.get(k) for k in ('status', 'penalty', 'upsilon', 'orbit_s', 'wall_s',
                                                    'peak_rss_mb', 'omp_threads', 'resumed')})
            rows.append(row)
            if row['status'] == 'ok':
                owners.setdefault(row['seed'], []).append(row['suffix'])
    rows.sort(key=lambda row: (row['seed'], row['suffix'] or ''))
    requested = sorted(s for r in scans for s in r['ic_seeds'].get('requested', []))
    ok = [row for row in rows if row['status'] == 'ok' and row['penalty'] is not None]
    penalties = [row['penalty'] for row in ok]
    scan = dict(requested=len(requested), ok=len(ok),
                missing=sorted(set(requested) - {row['seed'] for row in ok}),
                seeds_in_several_reports=sorted(s for s, who in owners.items() if len(who) > 1),
                penalty=_stats(penalties), upsilon=_stats([row['upsilon'] for row in ok]),
                orbit_s=_stats([row['orbit_s'] for row in ok]), wall_s=_stats([row['wall_s'] for row in ok]),
                omp_threads=sorted({row['omp_threads'] for row in ok} - {None}),
                distinct_penalties=len({round(p, 12) for p in penalties}),
                seed42_penalty=next((row['penalty'] for row in ok if row['seed'] == 42), None),
                reference_penalty=ref, rows=rows)
    scan['identical_realisations'] = scan['distinct_penalties'] < len(penalties)
    stats = scan['penalty']
    if stats and stats['std']:
        scan['sem'] = stats['std'] / math.sqrt(stats['n'])
        cuts = statistics.quantiles(penalties, n=100, method='inclusive')
        scan['quantiles'] = dict(q16=cuts[15], median=statistics.median(penalties), q84=cuts[83])
        if ref is not None:
            scan['mean_minus_ref'] = stats['mean'] - ref
            scan['mean_minus_ref_in_std'] = (stats['mean'] - ref) / stats['std']
            scan['fraction_below_ref'] = sum(p <= ref for p in penalties) / len(penalties)
    # Throughput of this layout: seeds evaluated in this run (resumed ones excluded).
    fresh = [row for row in ok if not row['resumed']]
    starts = [t for t in (_parse_time(r.get('started')) for r in scans) if t]
    ends = [t for t in (_parse_time(r.get('updated')) for r in scans) if t]
    if fresh and starts and ends and max(ends) > min(starts):
        hours = (max(ends) - min(starts)).total_seconds() / 3600
        scan['wall_h'] = hours
        scan['models_per_hour'] = len(fresh) / hours
    return scan


def format_seed_scan(scan):
    lines = [f"  seed scan: {scan['ok']}/{scan['requested']} ok, missing={scan['missing']}, "
             f"distinct penalties={scan['distinct_penalties']}, identical_realisations={scan['identical_realisations']}"]
    for key in ('penalty', 'sem', 'quantiles', 'reference_penalty', 'mean_minus_ref', 'mean_minus_ref_in_std',
                'fraction_below_ref', 'seed42_penalty', 'upsilon', 'orbit_s', 'wall_s', 'omp_threads',
                'wall_h', 'models_per_hour', 'seeds_in_several_reports'):
        lines.append(f"    {key}: {scan.get(key)}")
    lines.append('    seed\tstatus\tpenalty\tUpsilon\torbit_s\twall_s\tsuffix')
    for row in scan['rows']:
        lines.append('    ' + '\t'.join(str(row.get(k)) for k in
                                         ('seed', 'status', 'penalty', 'upsilon', 'orbit_s', 'wall_s', 'suffix')))
    return lines


def format_summary(reports, summary):
    lines = [f"single-model summary: {summary['reports']} reports, reference penalty={summary['reference_penalty']}"]
    for r in reports:
        p = r.get('protocols', {})
        cells = ' '.join(f"{name}={p[name]['penalty']:.6f}" for name in PROTOCOLS if name in p)
        peak = max((v.get('peak_rss_mb') or 0) for v in p.values()) if p else None
        lines.append(f"  {r.get('suffix')}: status={r.get('status')} {cells} peak_rss={peak} MB")
    for key in ('exp', 'prod', 'reuse', 'reuse_minus_exp', 'prod_minus_exp', 'prod_minus_ref',
                'peak_rss_mb', 'save_s'):
        lines.append(f"  {key}: {summary.get(key)}")
    for key in ('prod_minus_ref_in_scatter', 'grid_identical', 'orblib_names', 'memory_verdict'):
        lines.append(f"  {key}: {summary.get(key)}")
    if summary.get('seed_scan'):
        lines += format_seed_scan(summary['seed_scan'])
    return '\n'.join(lines) + '\n'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], allow_abbrev=False)
    parser.add_argument('--incl', type=float, default=90.0)
    for key in ('Q', 'gh', 'rh', 'rho0'):
        parser.add_argument(f'--{key}', type=float, default=None)
    parser.add_argument('--ref-penalty', type=float, default=None)
    parser.add_argument('--ref-upsilon', type=float, default=None)
    parser.add_argument('--ref-source', default=None,
                        help='Label of the reference row in the report (e.g. FILE:LINE)')
    parser.add_argument('--params-from', nargs='+', default=None, metavar='GLOB',
                        help='Take the min-penalty row at --incl from history/J-factor files '
                             f'(default {DEFAULT_PARAMS_FROM[0]} unless all four parameters are given)')
    parser.add_argument('--suffix', default=None)
    parser.add_argument('--orblib-dir', default=None)
    parser.add_argument('--side-file', default=None)
    parser.add_argument('--report', default=None)
    parser.add_argument('--protocols', default=None,
                        help='Default exp,reuse,prod; only exp with --ic-seeds')
    parser.add_argument('--ic-seeds', default=None, metavar='SPEC',
                        help="IC-seed scan, e.g. '1-100' or '1,5,9-12' (integers >= 1): "
                             'agama.setRandomSeed(K) + one exp evaluation per seed, no orbit library kept')
    parser.add_argument('--seed-file', default=None,
                        help='History file of the seed scan (default seeds_<hostname_proc>.txt); '
                             'a .tsv with the same stem gets one line per seed')
    parser.add_argument('--split-seeds', default=None, metavar='SPEC',
                        help='Print the round-robin split of SPEC over --workers, one comma list '
                             'per line (standard library only) and exit')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--subsample-seed', type=int, default=20261001)
    parser.add_argument('--n_threads', type=int, default=None)
    parser.add_argument('--preflight', action='store_true',
                        help='Import, report geometry and library name; no integration')
    parser.add_argument('--summarize', nargs='+', default=None, metavar='REPORT',
                        help='Aggregate JSON reports (standard library only) and exit')
    parser.add_argument('--summary-json', default=None)
    parser.add_argument('--list-models', default=None, metavar='FILE',
                        help='Print the rows of a models file tab-separated (incl Q gh rh rho0 '
                             'penalty Upsilon source; standard library only) and exit')
    args = parser.parse_args(argv)
    if args.split_seeds is not None:
        try:
            args.split_seeds = split_seeds(parse_seed_list(args.split_seeds), args.workers or 1)
        except ValueError as error:
            parser.error(f'--split-seeds: {error}')
    elif args.summarize is None and args.list_models is None:
        if not args.suffix or not args.orblib_dir:
            parser.error('--suffix and --orblib-dir are required')
        if args.ic_seeds is not None:
            try:
                args.ic_seeds = parse_seed_list(args.ic_seeds)
            except ValueError as error:
                parser.error(f'--ic-seeds: {error}')
        if args.protocols is None:
            args.protocols = 'exp' if args.ic_seeds else 'exp,reuse,prod'
        args.protocols = [p for p in args.protocols.split(',') if p]
        unknown = set(args.protocols) - set(PROTOCOLS)
        if unknown or not args.protocols or args.protocols[0] != 'exp' or len(set(args.protocols)) != len(args.protocols):
            parser.error('--protocols: exp first, then any of reuse,prod (no repeats)')
        if args.ic_seeds and args.protocols != ['exp']:
            parser.error('--ic-seeds runs the exp protocol only (no library is kept for reuse/prod)')
        args.report = args.report or f'report_{args.suffix}.json'
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.split_seeds is not None:
        for chunk in args.split_seeds:
            print(','.join(map(str, chunk)))
        return 0
    if args.list_models is not None:
        models = read_models(args.list_models)
        for row in models:
            print('\t'.join([repr(row[k]) for k in ('incl',) + PARAM_KEYS + ('penalty', 'upsilon')]
                            + [row['source'].replace('\t', ' ')]))
        return 0 if models else 1
    if args.summarize is not None:
        reports, summary = summarize(args.summarize)
        print(format_summary(reports, summary), end='')
        if args.summary_json:
            with open(args.summary_json, 'w') as stream:
                json.dump(summary, stream, indent=2, sort_keys=True)
        return 0 if reports and all(r.get('status') == 'ok' for r in reports) else 1
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
