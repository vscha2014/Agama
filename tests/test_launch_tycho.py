"""Mocked tests for py/launch_tycho.sh (no systemd, AGAMA, rclone or network)."""
import fcntl
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'py/launch_tycho.sh'
SCRIPT = 'Fornax_P21_PCA_w3Sersic_orblib_exp.py'

STUBS = {
    'systemd-run': r'''#!/bin/bash
echo "systemd-run $*" >> "$HOME/events"
while [ $# -gt 0 ]; do
    case "$1" in
        -p) shift 2 ;;
        -*) shift ;;
        *) break ;;
    esac
done
exec "$@"
''',
    'python': r'''#!/bin/bash
if [ "$1" = "-c" ]; then exit "${PY_IMPORT_FAIL:-0}"; fi
echo "python $* OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS" >> "$HOME/events"
sfx=""; q1=""; pot=""; prev=""; rep=""; seedf=""; prots=""
for a in "$@"; do
    [ "$prev" = "--suffix" ] && sfx="$a"
    [ "$prev" = "--report" ] && rep="$a"
    [ "$prev" = "--seed-file" ] && seedf="$a"
    [ "$prev" = "--protocols" ] && prots="$a"
    [ "$a" = "--Q1" ] && q1="Q1"
    [ "$a" = "--l24" ] && pot="_l24n184"
    prev="$a"
done
exp="${q1}d1_nb250_gh0${pot}_ser0"
if [ -n "$seedf" ]; then
    echo "# ic_seed: 1" >> "$seedf"
    echo "90.0 0.27 0 2.4 90 0.66 1.25${sfx} t" >> "$seedf"
    echo -e "seed\tstatus" > "${seedf%.txt}.tsv"
else
    echo "90.0 0.27 0 2.4 90 0.66 1.25${sfx} t" >> "out_${HOSTNAME_SUFFIX}_${exp}_${sfx}.txt"
    if [ -n "$prots" ] && [ "$prots" != "exp" ]; then
        echo "90.0 0.27 0 2.4 90 0.66 1.25${sfx} t" >> "single_${HOSTNAME_SUFFIX}_${exp}_${sfx}.txt"
    fi
fi
echo "log" > "log_${HOSTNAME_SUFFIX}_${exp}_${sfx}.txt"
if [ -n "$rep" ]; then
    echo "{\"status\": \"ok\", \"suffix\": \"$sfx\"}" > "$rep"
    echo "single-model check $sfx status=ok" > "${rep%.json}.txt"
fi
echo "worker $sfx done"
if [ -n "${PY_HANG:-}" ]; then
    trap 'echo "checkpoint written $sfx"; exit 75' TERM INT
    touch "$HOME/started"
    while true; do sleep 0.1; done
fi
exit "${PY_EXIT:-0}"
''',
    'rclone': r'''#!/bin/bash
echo "rclone $*" >> "$HOME/events"
case "$1" in
    lsf) exit "${RCLONE_LSF:-0}" ;;
    copyto) case "$2" in yandex:*) exit 1 ;; esac ;;
esac
exit 0
''',
    'curl': '#!/bin/bash\necho "curl $*" >> "$HOME/events"\n',
    'loginctl': '#!/bin/bash\necho "${LINGER:-yes}"\n',
    'ionice': '#!/bin/bash\nshift 2\nexec "$@"\n',
}


def workdir(tmp_path, memory_max='8192000000', controllers='cpuset cpu io memory pids'):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in STUBS.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    work = tmp_path / 'py'
    work.mkdir()
    (work / LAUNCHER.name).write_text(LAUNCHER.read_text())
    (work / SCRIPT).write_text(
        "bounds_original = {'Q': (0.05, 2.5), 'gh': (0.0, 1.6), 'rh': (0.5, 30.0), "
        "'rho0': (10.0, 120.0), 'Upsilon': (0.1, 1.6)}\n")
    (work / 'run_single_model.py').write_text((ROOT / 'py/run_single_model.py').read_text())
    for name in ('orblib_storage.py', 'table3.dat'):
        (work / name).touch()
    (tmp_path / 'meminfo').write_text('MemTotal:       32000000 kB\nMemAvailable:   30000000 kB\n')
    (tmp_path / 'proc_cgroup').write_text('0::/user.slice/agama-test.scope\n')
    scope = tmp_path / 'cgfs/user.slice/agama-test.scope'
    scope.mkdir(parents=True)
    (scope / 'memory.max').write_text(memory_max + '\n')
    (scope / 'memory.high').write_text('7372800000\n')
    (scope / 'memory.swap.max').write_text('0\n')
    (scope / 'cpu.max').write_text('400000 100000\n')
    (scope / 'memory.peak').write_text('2500000000\n')
    user = tmp_path / 'usercg'
    user.mkdir()
    (user / 'cgroup.controllers').write_text(controllers + '\n')
    return work


def run(tmp_path, *args, setup=None, **env):
    work = workdir(tmp_path, **{k: env.pop(k) for k in ('memory_max', 'controllers') if k in env})
    if setup:
        setup(work)
    environment = dict(
        os.environ, HOME=str(tmp_path), PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        HOSTNAME_SUFFIX='testhost', CONDA_ENV='', PYTHON='python', TYCHO_TIMESTAMP='20261010_120000',
        TYCHO_MEMINFO=str(tmp_path / 'meminfo'), TYCHO_PROC_CGROUP=str(tmp_path / 'proc_cgroup'),
        TYCHO_CGROUP_FS=str(tmp_path / 'cgfs'), TYCHO_USER_CGROUP=str(tmp_path / 'usercg'),
        TYCHO_MIN_FREE_GB='0', TYCHO_MONITOR_INTERVAL='1', TYCHO_STOP_GRACE='1')
    environment.update(env)
    environment.pop('TYCHO_IN_SCOPE', None)
    environment.pop('ORBLIB_SAVE_SLOTS', None)
    result = subprocess.run(['bash', str(work / LAUNCHER.name), *args], cwd=work, text=True,
                            capture_output=True, timeout=60, env=environment)
    events_file = tmp_path / 'events'
    events = events_file.read_text().splitlines() if events_file.exists() else []
    return result, events, work


def pythons(events):
    return [e.split() for e in events if e.startswith('python ')]


def output(result):
    return result.stdout[-4000:] + result.stderr[-4000:]


def test_default_run_uses_quarter_budget_one_worker_and_local_libraries(tmp_path):
    result, events, work = run(tmp_path, '--threads=4')
    assert result.returncode == 0, output(result)
    scope = [e for e in events if e.startswith('systemd-run') and '--unit=' in e]
    assert len(scope) == 1
    for prop in ('CPUQuota=400%', 'MemoryMax=8192000000', 'MemoryHigh=7372800000',
                 'MemorySwapMax=0', 'IOWeight=10', 'OOMPolicy=continue', '--user', '--scope'):
        assert prop in scope[0].split()
    assert '--unit=agama-tycho-d1_nb250_gh0_ser0_i90p0-20261010_120000' in scope[0]
    (argv,) = pythons(events)
    assert argv[argv.index('--n_threads') + 1] == '4'
    assert argv[argv.index('--suffix') + 1] == 'p0'
    assert argv[argv.index('--orblib-dir') + 1] == str(work / 'orblib')
    for flag in ('--save-orblib', '--reuse-orblib', '--no-resume', '--delete-checkpoint', 'OMP=4', 'OPENBLAS=4'):
        assert flag in argv
    assert '--stream-orblib' not in argv and '--Q1' not in argv and '--no-double' not in argv
    host = work / 'out_testhost_d1_nb250_gh0_ser0.txt'
    assert 'HASH:' in host.read_text() and '1.25p0' in host.read_text()
    assert not (work / 'out_testhost_d1_nb250_gh0_ser0_p0.txt').exists()
    assert any(e.startswith('rclone copyto') and 'out_testhost_d1_nb250_gh0_ser0.txt' in e
               and 'yandex:galAgama/out_' in e for e in events)
    assert any(e.startswith('rclone deletefile') and 'out_testhost_d1_nb250_gh0_ser0_p0.txt' in e
               for e in events)
    assert any('tycho_logs/d1_nb250_gh0_ser0_i90.0_20261010_120000' in e for e in events)
    log = next(work.glob('launch_tycho_*.log')).read_text()
    assert 'memory.max=8192000000' in log and 'memory.peak=2500000000' in log
    assert (work / 'monitor_tycho_d1_nb250_gh0_ser0_i90.0_20261010_120000.log').exists()


def test_two_workers_split_threads(tmp_path):
    result, events, work = run(tmp_path, '--threads=4', '--nproc=2')
    assert result.returncode == 0, output(result)
    runs = pythons(events)
    assert sorted(r[r.index('--suffix') + 1] for r in runs) == ['p0', 'p1']
    for r in runs:
        assert r[r.index('--n_threads') + 1] == '2' and 'OMP=2' in r and 'OPENBLAS=2' in r
    assert 'CPUQuota=400%' in next(e for e in events if '--unit=' in e)
    host = (work / 'out_testhost_d1_nb250_gh0_ser0.txt').read_text()
    assert '1.25p0' in host and '1.25p1' in host


def test_q1_mode_uses_q1_pool(tmp_path):
    result, events, work = run(tmp_path, '--threads=4', '--Q1', '--incl=85.0')
    assert result.returncode == 0, output(result)
    (argv,) = pythons(events)
    assert '--Q1' in argv and argv[argv.index('--incl') + 1] == '85.0'
    assert (work / 'out_testhost_Q1d1_nb250_gh0_ser0.txt').exists()
    assert '--unit=agama-tycho-Q1d1_nb250_gh0_ser0_i85p0-' in next(e for e in events if '--unit=' in e)


def test_resume_merges_previous_rows_and_keeps_checkpoint(tmp_path):
    def setup(work):
        (work / 'out_testhost_d1_nb250_gh0_ser0_p0.txt').write_text('90.0 0.3 0 2 80 0.7 1.9 old t\n')
        (work / 'checkpoint_testhost_d1_nb250_gh0_ser0_p0.pkl').write_bytes(b'x')
    result, events, work = run(tmp_path, '--threads=4', '--resume', setup=setup)
    assert result.returncode == 0, output(result)
    (argv,) = pythons(events)
    assert '--no-resume' not in argv and '--delete-checkpoint' not in argv
    host = (work / 'out_testhost_d1_nb250_gh0_ser0.txt').read_text()
    assert host.count('HASH:') == 2 and 'RESUME-PRE' in host and '1.9 old' in host


def test_merge_is_idempotent_by_hash(tmp_path):
    def setup(work):
        (work / 'out_testhost_d1_nb250_gh0_ser0_p0.txt').write_text('1 2 3\n')
        md5 = subprocess.run(['md5sum'], input='1 2 3\n', text=True, capture_output=True).stdout.split()[0]
        (work / 'out_testhost_d1_nb250_gh0_ser0.txt').write_text(f'# HASH:{md5}\n1 2 3\n')
    result, events, work = run(tmp_path, '--threads=4', setup=setup)
    assert result.returncode == 0, output(result)
    host = (work / 'out_testhost_d1_nb250_gh0_ser0.txt').read_text()
    assert host.count('1 2 3') == 1 and host.count('HASH:') == 2


def test_preflight_probes_limits_without_running(tmp_path):
    result, events, work = run(tmp_path, '--threads=4', '--preflight')
    assert result.returncode == 0, output(result)
    assert not pythons(events)
    assert any(e.startswith('systemd-run') and '--_print-limits' in e and 'MemorySwapMax=0' in e
               for e in events)
    assert 'memory.swap.max=0' in result.stdout and 'cpu.max=400000 100000' in result.stdout
    assert not any(e.startswith('rclone copyto') for e in events)


@pytest.mark.parametrize('env, message', [
    (dict(controllers='memory pids'), "контроллер cgroup 'cpu'"),
    (dict(PY_IMPORT_FAIL='1'), 'окружение python неполное'),
    (dict(TYCHO_MIN_FREE_GB='1000000000'), 'мало места'),
    (dict(memory_max='max'), 'лимит памяти scope не применён'),
])
def test_refuses_to_start_on_bad_environment(tmp_path, env, message):
    result, events, _ = run(tmp_path, '--threads=4', **env)
    assert result.returncode != 0
    assert message in result.stdout + result.stderr
    assert not pythons(events)


def test_missing_linger_only_warns(tmp_path):
    result, events, _ = run(tmp_path, '--threads=4', '--preflight', LINGER='no', TMUX='', STY='')
    assert result.returncode == 0, output(result)
    assert 'Linger=no' in result.stdout


@pytest.mark.parametrize('code, message', [('75', 'остановлен штатно'), ('137', 'OOM по MemoryMax')])
def test_worker_stop_codes_still_merge(tmp_path, code, message):
    result, events, work = run(tmp_path, '--threads=4', PY_EXIT=code)
    assert result.returncode == int(code)
    assert message in result.stdout
    assert '1.25p0' in (work / 'out_testhost_d1_nb250_gh0_ser0.txt').read_text()


def test_sigterm_to_launcher_is_forwarded_and_tee_survives(tmp_path):
    work = workdir(tmp_path)
    environment = dict(
        os.environ, HOME=str(tmp_path), PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        HOSTNAME_SUFFIX='testhost', CONDA_ENV='', PYTHON='python', TYCHO_TIMESTAMP='20261010_120000',
        TYCHO_MEMINFO=str(tmp_path / 'meminfo'), TYCHO_PROC_CGROUP=str(tmp_path / 'proc_cgroup'),
        TYCHO_CGROUP_FS=str(tmp_path / 'cgfs'), TYCHO_USER_CGROUP=str(tmp_path / 'usercg'),
        TYCHO_MIN_FREE_GB='0', TYCHO_MONITOR_INTERVAL='1', TYCHO_STOP_GRACE='30', PY_HANG='1')
    environment.pop('TYCHO_IN_SCOPE', None)
    proc = subprocess.Popen(['bash', str(work / LAUNCHER.name), '--threads=4'], cwd=work, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    deadline = time.time() + 20
    while not (tmp_path / 'started').exists():
        assert time.time() < deadline and proc.poll() is None
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=40)
    assert proc.returncode == 75, out[-3000:]
    assert 'остановлен штатно' in out
    runlog = next(work.glob('runlog_p0_*.log')).read_text()
    assert 'checkpoint written p0' in runlog
    assert '1.25p0' in (work / 'out_testhost_d1_nb250_gh0_ser0.txt').read_text()


def test_no_sync_skips_launcher_uploads(tmp_path):
    result, events, _ = run(tmp_path, '--threads=4', '--no-sync')
    assert result.returncode == 0, output(result)
    assert not any(e.startswith(('rclone copyto', 'rclone deletefile')) for e in events)


def test_offline_never_calls_rclone_and_bundles_results(tmp_path):
    def setup(work):
        (work / 'out_other_d1_nb250_gh0_ser0.txt').write_text('90 0.3 0 2 80 0.7 1.9 t\n')
        (work / '4UpsBoTorch_PCA_Sersic_x.txt').write_text('90 0.3 0 2 80 0.7 1.9 t\n')
    result, events, work = run(tmp_path, '--threads=4', '--offline', '--Q1', setup=setup)
    assert result.returncode == 0, output(result)
    assert not any(e.startswith('rclone') for e in events)
    (argv,) = pythons(events)
    assert '--offline' in argv and '--Q1' in argv
    log = next(work.glob('launch_tycho_*.log')).read_text()
    assert 'out_*_d1_nb250_gh0_ser0*: 1' in log and '4UpsBoTorch_PCA_Sersic*: 1' in log
    bundle = work / 'tycho_results_Q1d1_nb250_gh0_ser0_i90.0_20261010_120000.tar.gz'
    names = set(subprocess.run(['tar', '-tzf', str(bundle)], capture_output=True, text=True,
                               check=True).stdout.split())
    assert {'out_testhost_Q1d1_nb250_gh0_ser0.txt', 'log_testhost_Q1d1_nb250_gh0_ser0_p0.txt',
            'runlog_p0_Q1d1_nb250_gh0_ser0_i90.0_20261010_120000.log',
            'monitor_tycho_Q1d1_nb250_gh0_ser0_i90.0_20261010_120000.log',
            'launch_tycho_Q1d1_nb250_gh0_ser0_i90.0_20261010_120000.log'} <= names
    assert not any(n.endswith(('.npz', '.pkl')) for n in names)


def test_offline_preflight_skips_rclone(tmp_path):
    def setup(work):
        (work / 'out_other_Q1d1_nb250_gh0_ser0.txt').write_text('90 1.0 0 25 33.57 0.6 1.75 t\n')
    result, events, _ = run(tmp_path, '--threads=4', '--offline', '--Q1', '--preflight', setup=setup)
    assert result.returncode == 0, output(result)
    assert not any(e.startswith('rclone') for e in events)
    assert 'out_*_d1_nb250_gh0_ser0*: 1' in result.stdout
    assert 'холодный старт' in result.stdout


def test_models_mode_runs_sequentially_within_8gb_scope_and_bundles_offline(tmp_path):
    models_text = (
        '90.0 1.0 0.0 25.0 36.0 0.6003 1.7581 2026-10-06 14:42:52  # m0_rh25_rho36.0\n'
        '90.0 1.0 0.0 30.0 38.5 0.6003 1.7581 2026-10-06 14:42:52  # m1_rh30_rho38.5\n'
    )

    def setup(work):
        (work / 'models.txt').write_text(models_text)
        stop = work / 'orblib/.storage'
        stop.mkdir(parents=True)
        (stop / 'STOP').write_text('stale stop')
    result, events, work = run(tmp_path, '--threads=4', '--offline', '--models=models.txt', setup=setup)
    assert result.returncode == 0, output(result)
    assert not any(e.startswith('rclone') for e in events)
    assert not (work / 'orblib/.storage/STOP').exists()
    scope = [e for e in events if e.startswith('systemd-run') and '--unit=' in e]
    assert len(scope) == 1
    for prop in ('CPUQuota=400%', 'MemoryMax=8589934592', 'MemoryHigh=7730941131',
                 'MemorySwapMax=0', 'IOWeight=10', 'OOMPolicy=continue'):
        assert prop in scope[0].split()
    runs = pythons(events)
    assert len(runs) == 2
    assert [r[r.index('--rh') + 1] for r in runs] == ['25.0', '30.0']
    assert [r[r.index('--rho0') + 1] for r in runs] == ['36.0', '38.5']
    assert [r[r.index('--suffix') + 1] for r in runs] == ['multi20261010_120000m0', 'multi20261010_120000m1']
    for r in runs:
        assert r[r.index('--protocols') + 1] == 'exp'
        assert r[r.index('--n_threads') + 1] == '4' and 'OMP=4' in r and 'OPENBLAS=4' in r
    summary = (work / 'summary_multi_20261010_120000.txt').read_text()
    assert 'multi-model summary: 2 models' in summary and 'status=ok' in summary
    bundle = work / 'tycho_results_multi_d1_nb250_gh0_ser0_n2_20261010_120000.tar.gz'
    names = set(subprocess.run(['tar', '-tzf', str(bundle)], capture_output=True, text=True,
                               check=True).stdout.split())
    assert {'out_testhost_d1_nb250_gh0_ser0_multi20261010_120000m0.txt',
            'out_testhost_d1_nb250_gh0_ser0_multi20261010_120000m1.txt',
            'summary_multi_20261010_120000.txt',
            'report_multi_20261010_120000_m0.json',
            'report_multi_20261010_120000_m0.txt'} <= names


def test_no_save_orblib_keeps_reuse(tmp_path):
    result, events, _ = run(tmp_path, '--threads=4', '--no-save-orblib')
    assert result.returncode == 0, output(result)
    (argv,) = pythons(events)
    assert '--save-orblib' not in argv and '--reuse-orblib' in argv


@pytest.mark.parametrize('flag', ['--no-double', '--stream-orblib', '--n-bin=500', '--n_threads=8', '--q1'])
def test_rejects_incompatible_flags(tmp_path, flag):
    result, events, _ = run(tmp_path, flag)
    assert result.returncode == 2 and not events


def test_rejects_more_workers_than_threads(tmp_path):
    result, events, _ = run(tmp_path, '--threads=2', '--nproc=3')
    assert result.returncode == 2 and not events


def test_refuses_when_lock_held(tmp_path):
    holder = {}

    def setup(work):
        handle = open(work / '.tycho_launcher.lock', 'w')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        holder['handle'] = handle
    result, events, _ = run(tmp_path, '--threads=4', setup=setup)
    holder['handle'].close()
    assert result.returncode != 0 and 'уже работает' in result.stdout
    assert not pythons(events)


def test_no_shutdown_docker_or_sudo_commands():
    code = [line for line in LAUNCHER.read_text().splitlines() if not line.lstrip().startswith('#')]
    for line in code:
        assert not re.search(r'(^|[\s;&|(])(sudo|docker|shutdown|swapon|mkswap)\s', line), line


def test_git_results_auto_detects_parent_results_repo_and_pushes(tmp_path):
    remote = tmp_path / 'remote.git'
    results = tmp_path / 'results'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', 'clone', str(remote), str(results)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.email', 'test@example.com'], check=True)
    (results / 'out_prev_Q1d1_nb250_gh0_ser0.txt').write_text('90.0 1.0 0.0 25.0 33.57 0.6 1.75 t\n')
    subprocess.run(['git', '-C', str(results), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(results), 'commit', '-m', 'init'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'push', '-u', 'origin', 'HEAD'], check=True, capture_output=True)

    result, events, work = run(tmp_path, '--threads=4', '--Q1')
    assert result.returncode == 0, output(result)
    assert not any(e.startswith('rclone ') for e in events)
    (argv,) = pythons(events)
    assert '--offline' in argv and '--Q1' in argv
    assert not (work / 'out_prev_Q1d1_nb250_gh0_ser0.txt').exists()
    merged = results / 'out_testhost_Q1d1_nb250_gh0_ser0.txt'
    assert merged.exists() and not merged.is_symlink() and '1.25p0' in merged.read_text()
    verify = tmp_path / 'verify'
    subprocess.run(['git', 'clone', str(remote), str(verify)], check=True, capture_output=True)
    assert '1.25p0' in (verify / 'out_testhost_Q1d1_nb250_gh0_ser0.txt').read_text()
    assert len(list(verify.glob('logs/Q1d1_nb250_gh0_ser0_i90.0_*/log_testhost_Q1d1_nb250_gh0_ser0_p0.txt'))) == 1


def test_l24_search_writes_isolated_history_in_git_results(tmp_path):
    remote = tmp_path / 'remote.git'
    results = tmp_path / 'results'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', 'clone', str(remote), str(results)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.email', 'test@example.com'], check=True)
    (results / '4UpsBoTorch_PCA_Sersic_tycho.txt').write_text('90.0 0.267 0.0 2.41 90.4 0.65 1.32 t\n')
    subprocess.run(['git', '-C', str(results), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(results), 'commit', '-m', 'init'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'push', '-u', 'origin', 'HEAD'], check=True, capture_output=True)

    result, events, _ = run(tmp_path, '--threads=4', '--l24', '--incl=90.0')
    assert result.returncode == 0, output(result)
    (argv,) = pythons(events)
    assert '--l24' in argv and '--offline' in argv
    merged = results / 'out_testhost_d1_nb250_gh0_l24n184_ser0.txt'
    assert merged.exists() and '1.25p0' in merged.read_text()


def test_models_ic_seeds_runs_paired_batch_and_routes_to_logs_seed(tmp_path):
    models_text = (
        '90.000 0.267579 0.0 2.415367 90.27929 0.66275 1.44362 2026-10-22 22:04:15  # m0_freeQ\n'
        '90.000 1.000000 0.0 28.03331 43.45350 0.46579 1.52990 2026-10-16 04:19:35  # m1_Q1\n'
    )
    remote = tmp_path / 'remote.git'
    results = tmp_path / 'results'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', 'clone', str(remote), str(results)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.email', 'test@example.com'], check=True)
    mdir = results / 'logs/Q1d1_rh_bounds'
    mdir.mkdir(parents=True)
    (mdir / 'models_finalists.txt').write_text(models_text)
    subprocess.run(['git', '-C', str(results), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(results), 'commit', '-m', 'init'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'push', '-u', 'origin', 'HEAD'], check=True, capture_output=True)

    result, events, work = run(tmp_path, '--threads=4', '--l24', '--models=models_finalists.txt',
                               '--ic-seeds=1-20', TYCHO_MIN_FREE_GB='1000000000')
    assert result.returncode == 0, output(result)
    runs = pythons(events)
    assert len(runs) == 2
    for i, r in enumerate(runs):
        assert '--l24' in r
        assert r[r.index('--ic-seeds') + 1] == '1-20'
        assert r[r.index('--protocols') + 1] == 'exp'
        assert r[r.index('--seed-file') + 1].endswith(f'seeds_testhost_d1_nb250_gh0_l24n184_ser0_multi20261010_120000m{i}.txt')
    assert not list(results.glob('out_testhost_*.txt'))
    assert not list(work.glob('seeds_*')) and not list(work.glob('out_*'))
    seed_dir = results / 'logs/seed/seeds_multi_d1_nb250_gh0_l24n184_ser0_n2_20261010_120000'
    assert seed_dir.is_dir()
    assert len(list(seed_dir.glob('seeds_testhost_*.txt'))) == 2
    assert len(list(seed_dir.glob('seeds_testhost_*.tsv'))) == 2
    assert (seed_dir / 'summary_multi_20261010_120000.json').exists()
    assert (seed_dir / 'models_finalists.txt').exists()


def test_models_protocols_check_keeps_pool_clean_and_archives_side_files(tmp_path):
    models_text = (
        '90.000 0.267579 0.0 2.415367 90.27929 0.66275 1.44362 2026-10-22 22:04:15  # m0_freeQ\n'
        '90.000 1.000000 0.0 28.03331 43.45350 0.46579 1.52990 2026-10-16 04:19:35  # m1_Q1\n'
    )
    remote = tmp_path / 'remote.git'
    results = tmp_path / 'results'
    subprocess.run(['git', 'init', '--bare', str(remote)], check=True, capture_output=True)
    subprocess.run(['git', 'clone', str(remote), str(results)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.name', 'Test'], check=True)
    subprocess.run(['git', '-C', str(results), 'config', 'user.email', 'test@example.com'], check=True)
    (results / 'models_finalists.txt').write_text(models_text)
    subprocess.run(['git', '-C', str(results), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(results), 'commit', '-m', 'init'], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(results), 'push', '-u', 'origin', 'HEAD'], check=True, capture_output=True)

    result, events, work = run(tmp_path, '--threads=4', '--l24', '--models=models_finalists.txt',
                               '--protocols=exp,reuse,prod')
    assert result.returncode == 0, output(result)
    runs = pythons(events)
    assert len(runs) == 2
    for r in runs:
        assert '--l24' in r and r[r.index('--protocols') + 1] == 'exp,reuse,prod'
    assert not list(results.glob('out_testhost_*.txt'))
    assert not list(work.glob('out_*')) and not list(work.glob('single_*'))
    log_dir = results / 'logs/multi_d1_nb250_gh0_l24n184_ser0_n2_20261010_120000'
    assert len(list(log_dir.glob('out_testhost_*_multi*.txt'))) == 2
    assert len(list(log_dir.glob('single_testhost_*_multi*.txt'))) == 2
    assert (log_dir / 'summary_multi_20261010_120000.json').exists()


@pytest.mark.parametrize('extra_args', [
    ('--ic-seeds=1-20',),
    ('--protocols=exp,prod',),
    ('--models=models.txt', '--ic-seeds=0'),
    ('--models=models.txt', '--ic-seeds=1-20', '--protocols=exp,prod'),
])
def test_ic_seeds_and_protocols_reject_invalid_combinations(tmp_path, extra_args):
    def setup(work):
        (work / 'models.txt').write_text(
            '90.0 1.0 0.0 25.0 36.0 0.6003 1.7581 2026-10-06 14:42:52  # m0\n')
    result, events, _ = run(tmp_path, '--threads=4', *extra_args, setup=setup)
    assert result.returncode == 2 and not events
