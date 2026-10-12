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
sfx=""; exp="d1_nb250_gh0_ser0"; prev=""
for a in "$@"; do
    [ "$prev" = "--suffix" ] && sfx="$a"
    [ "$a" = "--Q1" ] && exp="Q1d1_nb250_gh0_ser0"
    prev="$a"
done
echo "90.0 0.27 0 2.4 90 0.66 1.25${sfx} t" >> "out_${HOSTNAME_SUFFIX}_${exp}_${sfx}.txt"
echo "log" > "log_${HOSTNAME_SUFFIX}_${exp}_${sfx}.txt"
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
    for name in (SCRIPT, 'orblib_storage.py', 'table3.dat'):
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
