"""Mocked tests for launch_potential_pair_local.sh (no systemd, AGAMA, docker, rclone or network)."""
import fcntl
import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / 'launch_potential_pair_local.sh'

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
echo "python $* OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MPL=$MPLBACKEND" >> "$HOME/events"
out=""; prev=""
for a in "$@"; do [ "$prev" = "--output" ] && out="$a"; prev="$a"; done
mkdir -p "$out"
printf '{"status": "running", "stage": "integrate"}\n' > "$out/report.json"
if [ -n "${PY_HANG:-}" ]; then
    trap 'echo "runner got TERM" >> "$HOME/events"; exit 1' TERM
    touch "$HOME/started"
    while true; do sleep 0.1; done
fi
sleep "${PY_SLEEP:-0}"
printf '{"status": "%s", "penalty": 1.234}\n' "${PY_STATUS:-pilot_pass}" > "$out/report.json"
exit "${PY_EXIT:-0}"
''',
    'systemctl': '#!/bin/bash\necho "systemctl $*" >> "$HOME/events"\nprintf "%s" "${ACTIVE_SCOPES:-}"\n',
    'curl': '#!/bin/bash\necho "curl $*" >> "$HOME/events"\nexit "${CURL_EXIT:-0}"\n',
    'loginctl': '#!/bin/bash\necho yes\n',
    'ionice': '#!/bin/bash\nshift 2\nexec "$@"\n',
    'df': '#!/bin/bash\nprintf "F 1024-blocks Used Available Capacity M\\nm 9 1 %s 1%% /m\\n" "${FREE_KB:-90000000}"\n',
}
FORBIDDEN = ('rclone', 'docker', 'sudo', 'shutdown')


def setup(tmp_path, memory_max='8589934592'):
    bins = tmp_path / 'bin'
    bins.mkdir()
    for name, body in {**STUBS, **{n: f'#!/bin/bash\necho "{n} $*" >> "$HOME/events"\n'
                                    for n in FORBIDDEN}}.items():
        (bins / name).write_text(body)
        (bins / name).chmod((bins / name).stat().st_mode | stat.S_IEXEC)
    code = tmp_path / 'py'
    diag = code / 'check_potential_convergence'
    diag.mkdir(parents=True)
    (diag / LAUNCHER.name).write_text(LAUNCHER.read_text())
    for path in (code / 'Fornax_P21_PCA_w3Sersic_orblib_exp.py', code / 'run_single_model.py',
                 code / 'table3.dat', diag / 'run_potential_pair.py', diag / 'check_potential_convergence.py'):
        path.touch()
    inputs = tmp_path / 'input dir'
    inputs.mkdir()
    (inputs / 'models.txt').write_text('synthetic\n')
    (inputs / 'freeq').mkdir()
    (inputs / 'freeq/report.json').write_text('{}')
    (tmp_path / 'review.json').write_text('{}')
    (tmp_path / 'outputs').mkdir()
    (tmp_path / 'meminfo').write_text('MemTotal: 32000000 kB\nMemAvailable: 30000000 kB\n')
    (tmp_path / 'proc_cgroup').write_text('0::/user.slice/pair.scope\n')
    scope = tmp_path / 'cgfs/user.slice/pair.scope'
    scope.mkdir(parents=True)
    (scope / 'memory.max').write_text(memory_max + '\n')
    (scope / 'memory.high').write_text('7730941131\n')
    (scope / 'memory.swap.max').write_text('0\n')
    (scope / 'cpu.max').write_text('400000 100000\n')
    (scope / 'memory.peak').write_text('3400000000\n')
    (scope / 'memory.events').write_text('low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n')
    (scope / 'cpu.stat').write_text('usage_usec 123\n')
    (tmp_path / 'usercg').mkdir()
    (tmp_path / 'usercg/cgroup.controllers').write_text('cpuset cpu io memory pids\n')
    return code


def environment(tmp_path, **extra):
    env = dict(os.environ, HOME=str(tmp_path), PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}",
               CONDA_ENV='', PYTHON='python', PAIR_TIMESTAMP='20261013_120000',
               PAIR_MEMINFO=str(tmp_path / 'meminfo'), PAIR_PROC_CGROUP=str(tmp_path / 'proc_cgroup'),
               PAIR_CGROUP_FS=str(tmp_path / 'cgfs'), PAIR_USER_CGROUP=str(tmp_path / 'usercg'),
               PAIR_MONITOR_INTERVAL='1', PAIR_STOP_GRACE='5')
    for key in ('PAIR_IN_SCOPE', 'TMUX', 'NTFY_TOPIC', 'NTFY_SERVER'):
        env.pop(key, None)
    env.update(extra)
    return env


def command(tmp_path, *args, mode='pilot', index=2, output='outputs/run1'):
    base = ['bash', str(tmp_path / 'py/check_potential_convergence' / LAUNCHER.name), f'--mode={mode}',
            f'--input-root={tmp_path / "input dir"}', '--models=models.txt',
            '--field-report=freeq/report.json', f'--model-index={index}',
            f'--output={tmp_path / output}', '--threads=4']
    return base + list(args)


def run(tmp_path, *args, memory_max='8589934592', env=None, **kw):
    setup(tmp_path, memory_max)
    result = subprocess.run(command(tmp_path, *args, **kw), text=True, capture_output=True,
                            timeout=60, env=environment(tmp_path, **(env or {})))
    events_file = tmp_path / 'events'
    events = events_file.read_text().splitlines() if events_file.exists() else []
    return result, events


def pythons(events):
    return [e.split() for e in events if e.startswith('python ')]


def text(result):
    return result.stdout[-3000:] + result.stderr[-3000:]


def assert_no_forbidden(events):
    assert not [e for e in events if e.split()[0] in FORBIDDEN]


@pytest.mark.parametrize('mode,flag', [('validate', '--validate-pilot'), ('preflight', '--pilot-preflight'),
                                       ('pilot', '--pilot')])
def test_runner_gets_real_paths_threads_and_one_limited_scope(tmp_path, mode, flag):
    result, events = run(tmp_path, mode=mode)
    assert result.returncode == 0, text(result)
    (scope,) = [e.split() for e in events if e.startswith('systemd-run')]
    for prop in ('--user', '--scope', 'CPUQuota=400%', 'MemoryMax=8589934592', 'MemoryHigh=7730941131',
                 'MemorySwapMax=0', 'IOWeight=10', 'OOMPolicy=continue',
                 f'--unit=agama-pair-{mode}-2-20261013_120000'):
        assert prop in scope
    (argv,) = pythons(events)
    assert argv[1] == '-B' and argv[2] == str(tmp_path / 'py/check_potential_convergence/run_potential_pair.py')
    assert flag in argv and argv[argv.index('--threads') + 1] == '4'
    assert argv[argv.index('--harness') + 1] == str(tmp_path / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py')
    assert argv[argv.index('--output') + 1] == str(tmp_path / 'outputs/run1/run')
    assert f"--field-report {tmp_path / 'input dir/freeq/report.json'} " in next(e for e in events if e.startswith('python '))
    assert 'OMP=4' in argv and 'OPENBLAS=1' in argv and 'MPL=Agg' in argv
    assert '--reviewed-freeq' not in argv
    out = tmp_path / 'outputs/run1'
    log = (out / 'launcher.log').read_text()
    for line in ('memory.max=8589934592', 'memory.swap.max=0', 'exit_code=0', 'runner_exit=0', 'notify=sent'):
        assert line in log
    assert 'memory.peak=3400000000' in (out / 'scope_state.txt').read_text()
    assert (out / 'resources.jsonl').read_text().count('"memory.peak": "3400000000"') >= 1
    (note,) = [e for e in events if e.startswith('curl ')]
    assert f'mode={mode} index=2 exit=0 status=pilot_pass' in note
    assert 'penalty' not in note and '1.234' not in note
    assert_no_forbidden(events)


def test_q1_pilot_passes_the_review_and_fields_mode_uses_the_dense_series(tmp_path):
    setup(tmp_path)
    env = environment(tmp_path)
    result = subprocess.run(command(tmp_path, f'--reviewed-freeq={tmp_path / "review.json"}', index=0),
                            text=True, capture_output=True, env=env, timeout=60)
    assert result.returncode == 0, text(result)
    fields = subprocess.run(command(tmp_path, mode='fields', output='outputs/fields'),
                            text=True, capture_output=True, env=env, timeout=60)
    assert fields.returncode == 0, text(fields)
    q1, dense = pythons((tmp_path / 'events').read_text().splitlines())
    assert q1[q1.index('--reviewed-freeq') + 1] == str(tmp_path / 'review.json')
    assert q1[q1.index('--model-index') + 1] == '0'
    assert dense[2].endswith('check_potential_convergence.py') and '--field-report' not in dense
    for flag, value in (('--threads', '1'), ('--radii', '192'), ('--angles', '17'),
                        ('--angular-orders', '4,8,12,16,24,32'), ('--radial-nodes', '23,46,92,184')):
        assert dense[dense.index(flag) + 1] == value


@pytest.mark.parametrize('args,kw,message', [
    (('--shutdown',), {}, 'VM-only'),
    (('--timeout=10',), {}, 'VM-only'),
    (('--image=agama:latest',), {}, 'VM-only'),
    (('--threads=0',), {}, 'positive integers'),
    (('--models=/etc/passwd',), {}, 'relative'),
    (('--models=../review.json',), {}, 'escapes'),
    (('--field-report=../review.json',), {}, 'escapes'),
    (('--field-report=missing.json',), {}, 'does not exist'),
    ((), {'index': 0}, 'reviewed-freeq'),
    (('--reviewed-freeq=x',), {'mode': 'preflight'}, 'only valid for Q1'),
    (('--reviewed-freeq=x',), {'index': 2}, 'only valid for Q1'),
    ((), {'mode': 'run'}, 'explicit --mode'),
    ((), {'output': 'missing/run1'}, 'parent'),
])
def test_input_errors_start_nothing(tmp_path, args, kw, message):
    result, events = run(tmp_path, *args, **kw)
    assert result.returncode == 2 and message in result.stderr, text(result)
    assert not pythons(events) and not any(e.startswith('systemd-run') for e in events)
    assert not (tmp_path / 'outputs/run1').exists()


def test_existing_output_is_never_reused(tmp_path):
    result, events = run(tmp_path)
    assert result.returncode == 0, text(result)
    before = (tmp_path / 'outputs/run1/launcher.log').read_text()
    again = subprocess.run(command(tmp_path), text=True, capture_output=True,
                           env=environment(tmp_path), timeout=60)
    assert again.returncode == 2 and 'already exists' in again.stderr
    assert (tmp_path / 'outputs/run1/launcher.log').read_text() == before


@pytest.mark.parametrize('code,status', [(3, 'needs_review'), (1, 'failed')])
def test_runner_exit_code_is_propagated(tmp_path, code, status):
    result, events = run(tmp_path, env=dict(PY_EXIT=str(code), PY_STATUS=status))
    assert result.returncode == code, text(result)
    log = (tmp_path / 'outputs/run1/launcher.log').read_text()
    assert f'exit_code={code}' in log
    assert any(f'exit={code} status={status}' in e and 'Priority: high' in e for e in events)


def test_refuses_without_scope_memory_limit(tmp_path):
    result, events = run(tmp_path, memory_max='max')
    assert result.returncode == 2 and 'memory.max is not set' in result.stderr
    assert not pythons(events)
    assert 'exit_code=2' in (tmp_path / 'outputs/run1/launcher.log').read_text()


def test_refuses_next_to_an_active_search_or_pilot(tmp_path):
    result, events = run(tmp_path, env=dict(ACTIVE_SCOPES='agama-tycho-d1_i90p0-1.scope loaded active running\n'))
    assert result.returncode == 2 and 'run sequentially' in result.stderr
    assert not pythons(events) and not (tmp_path / 'outputs/run1').exists()


def test_refuses_while_launch_tycho_holds_its_lock(tmp_path):
    setup(tmp_path)
    lock = tmp_path / 'py/.tycho_launcher.lock'
    lock.touch()
    with lock.open('w') as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        result = subprocess.run(command(tmp_path), text=True, capture_output=True,
                                env=environment(tmp_path), timeout=60)
    assert result.returncode == 2 and 'launch_tycho.sh is running' in result.stderr
    free = subprocess.run(command(tmp_path), text=True, capture_output=True,
                          env=environment(tmp_path), timeout=60)
    assert free.returncode == 0, text(free)


@pytest.mark.parametrize('env,message', [
    (dict(FREE_KB='1000'), 'GiB free space'),
    (dict(PY_IMPORT_FAIL='1'), 'lacks numpy'),
])
def test_environment_gates(tmp_path, env, message):
    result, events = run(tmp_path, env=env)
    assert result.returncode == 2 and message in result.stderr
    assert not pythons(events)


def test_memory_headroom_and_controller_gates(tmp_path):
    setup(tmp_path)
    (tmp_path / 'meminfo').write_text('MemAvailable: 9000000 kB\n')
    low = subprocess.run(command(tmp_path), text=True, capture_output=True, env=environment(tmp_path))
    assert low.returncode == 2 and 'headroom' in low.stderr
    (tmp_path / 'meminfo').write_text('MemAvailable: 30000000 kB\n')
    (tmp_path / 'usercg/cgroup.controllers').write_text('cpu pids\n')
    bare = subprocess.run(command(tmp_path), text=True, capture_output=True, env=environment(tmp_path))
    assert bare.returncode == 2 and "'memory' is not delegated" in bare.stderr


def test_stall_and_runtime_alerts_notify_once_and_never_stop(tmp_path):
    result, events = run(tmp_path, '--stall-alert=1', '--runtime-alert=1', env=dict(PY_SLEEP='4'))
    assert result.returncode == 0, text(result)
    log = (tmp_path / 'outputs/run1/launcher.log').read_text()
    assert log.count('stall_alert=') == 1 and log.count('runtime_alert=') == 1
    assert 'stall_notify=sent' in log and 'runtime_notify=sent' in log and 'stop_reason' not in log
    notes = [e for e in events if e.startswith('curl ')]
    assert len(notes) == 3 and 'stage=integrate; still running, not stopped' in notes[0]
    assert 'exit=0 status=pilot_pass' in notes[-1]


def test_no_notify_and_failed_notify(tmp_path):
    result, events = run(tmp_path, '--no-notify')
    assert result.returncode == 0 and not any(e.startswith('curl ') for e in events)
    other = tmp_path / 'second'
    other.mkdir()
    failed, _ = run(other, env=dict(CURL_EXIT='7'))
    assert failed.returncode == 0 and 'ntfy notification failed' in failed.stderr
    assert 'notify=failed' in (other / 'outputs/run1/launcher.log').read_text()


def test_oom_kill_is_reported(tmp_path):
    setup(tmp_path)
    (tmp_path / 'cgfs/user.slice/pair.scope/memory.events').write_text('oom 1\noom_kill 1\n')
    result = subprocess.run(command(tmp_path), text=True, capture_output=True,
                            env=environment(tmp_path, PY_EXIT='137'), timeout=60)
    assert result.returncode == 137 and 'OOM kill' in result.stderr
    out = tmp_path / 'outputs/run1'
    assert 'oom_kill=1' in (out / 'launcher.log').read_text()
    assert 'memory.events.oom_kill=1' in (out / 'scope_state.txt').read_text()
    events = (tmp_path / 'events').read_text()
    assert 'exit=137' in events and 'oom_kill=1' in events


def test_term_is_forwarded_to_the_runner_and_reported(tmp_path):
    setup(tmp_path)
    process = subprocess.Popen(command(tmp_path), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=environment(tmp_path, PY_HANG='1'))
    deadline = time.time() + 30
    while not (tmp_path / 'started').exists() and time.time() < deadline:
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    out, err = process.communicate(timeout=30)
    assert process.returncode == 124, out + err
    log = (tmp_path / 'outputs/run1/launcher.log').read_text()
    assert 'stop_reason=signal' in log and 'runner_exit=1' in log and 'exit_code=124' in log
    events = (tmp_path / 'events').read_text()
    assert 'runner got TERM' in events and 'stopped=signal' in events
    assert_no_forbidden(events.splitlines())


def test_source_has_no_cloud_container_or_shutdown_calls():
    body = '\n'.join(line for line in LAUNCHER.read_text().splitlines()
                     if not line.lstrip().startswith(('#', "'", 'printf')))
    for word in ('rclone', 'docker ', 'sudo', 'shutdown -', 'rm -'):
        assert word not in body
