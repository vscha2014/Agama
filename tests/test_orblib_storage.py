import ast
import contextlib
import importlib.util
import io
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile
from types import SimpleNamespace

import numpy
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('orblib_storage', ROOT / 'py/orblib_storage.py')
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)
NAME = 'orblib_i90.0_d1_nb250_ser0_geomabcdef_1234567890.npz'
CALC_SCRIPT = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'


def script_namespace(names, **extra):
    """Функции расчётного скрипта по именам, без импорта модуля.

    Скрипт монолитный и выполняет код верхнего уровня (argparse, agama),
    поэтому тесты вытаскивают нужные определения через ast.
    """
    nodes = [node for node in ast.parse(CALC_SCRIPT.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names), sorted(names)
    ns = dict(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CALC_SCRIPT), 'exec'), ns)
    return ns


def writer_namespace():
    import zipfile as zipfile_module
    return script_namespace(['write_orblib_npz'], numpy=numpy,
                            zipfile=zipfile_module, _NPZ_BLOCK_BYTES=4096)


def library(path):
    numpy.savez_compressed(path, Q=1.0, gh=0.0, rh=7.0, rho0=10.0, incl=90.0,
                           double=1, n_bin=250, ser_id=0, numOrbits=2, trajsize=1000,
                           intTime=100.0, degree=2, ghorder=6, gridv=numpy.arange(5.0),
                           ic=numpy.zeros((2, 6)), inttime=numpy.ones(2),
                           matrix_dens=numpy.zeros((2, 3)), matrix_kinem=numpy.ones((2, 5)))


@pytest.mark.parametrize('hash_name', ['md5', 'MD5', 'Md5'])
def test_rclone_inventory_parses_real_lsjson_hashes(monkeypatch, hash_name):
    remote = storage.Rclone('yandex:galAgama/orblib', 'unused.conf')
    checksum = storage.digest(b'archive')
    response = [{'Path': 'sample.tar', 'Name': 'sample.tar', 'Size': 7,
                 'IsDir': False, 'Hashes': {hash_name: checksum.upper(), 'sha256': 'unused'}}]
    monkeypatch.setattr(remote, 'run', lambda *args: json.dumps(response).encode())
    assert remote.inventory() == {'sample.tar': dict(size=7, md5=checksum)}


@pytest.mark.parametrize('hashes', [None, {}, {'sha1': 'abc'}, {'md5': ''}, {'md5': 'bad'}])
def test_missing_or_invalid_md5_is_rejected_before_streaming(monkeypatch, hashes):
    remote = storage.Rclone('yandex:galAgama/orblib', 'unused.conf')
    calls = []

    def run(*args):
        calls.append(args)
        return json.dumps([dict(Path='sample.tar', Size=7, Hashes=hashes)]).encode()

    monkeypatch.setattr(remote, 'run', run)
    with pytest.raises(ValueError, match='sample.tar.*MD5'):
        remote.inventory()
    assert all(call[0] == 'lsjson' for call in calls)


def test_rclone_stat_targets_one_object(monkeypatch):
    remote = storage.Rclone('yandex:galAgama/orblib', 'unused.conf')
    calls = []
    row = dict(Path='part.tar', Size=7, Hashes={'md5': storage.digest(b'archive')})

    def run(*args):
        calls.append(args)
        return json.dumps(row).encode()

    monkeypatch.setattr(remote, 'run', run)
    assert remote.stat('part.tar')['size'] == 7
    assert calls == [('lsjson', 'yandex:galAgama/orblib/part.tar', '--stat', '--hash')]


@pytest.mark.parametrize('code', [1, 3, 4, 5])
def test_rclone_stat_does_not_hide_access_or_transfer_errors(monkeypatch, code):
    remote = storage.Rclone('yandex:galAgama/orblib', 'unused.conf')

    def run(*args):
        raise storage.RcloneError('lsjson', code, 'test failure')

    monkeypatch.setattr(remote, 'run', run)
    if code in (3, 4):
        assert remote.stat('missing.tar') is None
    else:
        with pytest.raises(storage.RcloneError):
            remote.stat('missing.tar')


def test_rclone_conflicting_hash_keys_are_rejected():
    with pytest.raises(ValueError, match='conflicting MD5'):
        storage.remote_info(dict(Path='a.tar', Size=1, Hashes={
            'MD5': storage.digest(b'a'), 'md5': storage.digest(b'b')}))


class Remote:
    def __init__(self):
        self.files = {}
        self.copies = 0
        self.fail = False
        self.bad_hash = False

    def inventory(self):
        return {name: self.stat(name) for name in self.files}

    def stat(self, name):
        if name not in self.files:
            return None
        data = self.files[name]
        return dict(size=len(data), md5='bad' if self.bad_hash else storage.digest(data))

    def read(self, name):
        return self.files[name]

    def copy(self, path, name):
        self.copies += 1
        if self.fail:
            raise OSError('offline')
        self.files[name] = Path(path).read_bytes()

    def move(self, source, destination):
        if destination in self.files:
            raise FileExistsError(destination)
        self.files[destination] = self.files.pop(source)


@pytest.fixture
def store(tmp_path):
    return storage.Store(tmp_path / 'orblib', reserve_bytes=0)


def ready(store, managed=True):
    library(store.root / NAME)
    store.register(NAME, managed=managed)


def test_publish_verify_manifest_then_remove(store):
    ready(store)
    remote = Remote()
    assert storage.deliver(store, remote, attempts=3, delay=0)
    assert not (store.root / NAME).exists()
    assert remote.stat('objects/' + NAME)
    receipt = json.loads(remote.read('catalog/' + NAME + '.json'))
    assert receipt['md5'] == remote.stat('objects/' + NAME)['md5']
    assert store.archived(NAME, receipt['metadata'])
    count = remote.copies
    assert storage.deliver(store, remote, attempts=3, delay=0)
    assert remote.copies == count


def test_delivery_exhaustion_persists_stop_and_file(store):
    ready(store)
    remote = Remote()
    remote.fail = True
    assert not storage.deliver(store, remote, attempts=3, delay=0)
    assert remote.copies == 3
    assert store.stopped()
    assert (store.root / NAME).exists()
    with pytest.raises(storage.StorageStop):
        store.check_stop()
    assert not storage.deliver(store, remote, attempts=3, delay=0)
    assert remote.copies == 3


def test_recover_pending_before_allowing_work(store):
    ready(store)
    remote = Remote()
    remote.fail = True
    assert not storage.deliver(store, remote, attempts=1, delay=0)
    remote.fail = False
    storage.prepare(store, remote, resume=True, attempts=3, delay=0)
    assert not store.stopped()
    assert not store.pending()
    assert not (store.root / NAME).exists()


def test_same_size_wrong_hash_never_removes_local(store):
    ready(store)
    remote = Remote()
    remote.files['objects/' + NAME] = b'x' * (store.root / NAME).stat().st_size
    assert not storage.deliver(store, remote, attempts=1, delay=0)
    assert (store.root / NAME).exists()
    assert store.stopped()


def test_manifest_failure_preserves_file_and_recovers_without_reupload(store):
    ready(store)
    remote = Remote()
    original = remote.copy

    def fail_manifest(path, name):
        if name.startswith('catalog/'):
            raise OSError('manifest offline')
        original(path, name)

    remote.copy = fail_manifest
    assert not storage.deliver(store, remote, attempts=1, delay=0)
    assert (store.root / NAME).exists()
    assert remote.stat('objects/' + NAME)
    copies = remote.copies
    remote.copy = original
    storage.prepare(store, remote, resume=True, attempts=1, delay=0)
    assert remote.copies == copies + 1
    assert not (store.root / NAME).exists()


def test_existing_unmanaged_file_is_uploaded_but_not_deleted(store):
    library(store.root / NAME)
    remote = Remote()
    storage.prepare(store, remote, resume=True, attempts=1, delay=0)
    assert (store.root / NAME).exists()
    assert store.archived(NAME, storage.metadata(store.root / NAME))


def test_old_local_copies_require_explicit_verified_prune(store):
    ready(store, managed=False)
    remote = Remote()
    assert storage.deliver(store, remote, attempts=1, delay=0)
    storage.prune_verified(store, remote)
    assert (store.root / NAME).exists()
    assert not store.row(NAME)['managed']
    storage.prune_verified(store, remote, apply=True)
    assert not (store.root / NAME).exists()
    assert remote.stat('objects/' + NAME)


def test_prune_never_deletes_without_verified_remote(store):
    ready(store, managed=False)
    remote = Remote()
    assert storage.deliver(store, remote, attempts=1, delay=0)
    remote.files['objects/' + NAME] = b'corrupt'
    with pytest.raises(ValueError):
        storage.prune_verified(store, remote, apply=True)
    assert (store.root / NAME).exists()
    assert not store.row(NAME)['managed']


def test_unknown_legacy_archive_blocks_start_without_download(store):
    remote = Remote()
    remote.files['orblib_i90.0_d1_nb250_ser0__host_20260101_000000.tar'] = b'archive'
    with pytest.raises(RuntimeError, match='index'):
        storage.prepare(store, remote, resume=True)
    assert remote.copies == 0


def test_unfinished_run_requires_explicit_resume(store):
    ready(store)
    store.stop('upload failure')
    with pytest.raises(RuntimeError, match='resume'):
        storage.prepare(store, Remote(), resume=False)
    assert (store.root / NAME).exists()


def test_orphan_managed_file_recovered(store):
    claim = store.claim(NAME, 1024)
    library(store.root / NAME)
    store.release(claim, NAME)
    remote = Remote()
    storage.prepare(store, remote, resume=True, attempts=1, delay=0)
    assert not (store.root / NAME).exists()
    assert remote.stat('objects/' + NAME)


def test_disk_wait_observes_stop(store, monkeypatch):
    store.reserve_bytes = 10**30
    monkeypatch.setattr(storage.time, 'sleep', lambda seconds: store.stop('offline'))
    with pytest.raises(storage.StorageStop):
        store.claim(NAME, 1024)


def test_active_writer_is_not_uploaded(store):
    claim = store.claim(NAME, 1024)
    library(store.root / NAME)
    store.register(NAME)
    remote = Remote()
    assert storage.deliver(store, remote, attempts=1, delay=0)
    assert not remote.files
    store.release(claim, NAME)
    assert storage.deliver(store, remote, attempts=1, delay=0)
    assert remote.stat('objects/' + NAME)


def test_metadata_collision_rejected(store):
    ready(store)
    remote = Remote()
    assert storage.deliver(store, remote, attempts=1, delay=0)
    expected = storage.metadata_bytes(remote.read('objects/' + NAME))
    expected['rh'] = 6.999999
    with pytest.raises(ValueError, match='metadata'):
        store.archived(NAME, expected)


def test_legacy_index_stream_has_members_and_hash(tmp_path, monkeypatch):
    path = tmp_path / NAME
    library(path)
    payload = path.read_bytes()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        member = tarfile.TarInfo(NAME)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    stream.seek(0)
    entries = storage.index_stream(stream, tmp_path,
                                   expected=dict(size=len(stream.getvalue()), md5=storage.digest(stream.getvalue())))
    assert entries[0]['name'] == NAME
    assert entries[0]['md5'] == storage.digest(payload)
    assert entries[0]['metadata']['rh'] == 7
    stream.seek(0)
    with pytest.raises(ValueError, match='stream size/MD5'):
        storage.index_stream(stream, tmp_path, expected=dict(size=len(stream.getvalue()), md5='wrong'))
    stream.seek(0)
    monkeypatch.setattr(storage.tempfile, 'TemporaryFile', lambda **kwargs: pytest.fail('unneeded staging'))
    assert storage.index_stream(stream, tmp_path, existing_dir=tmp_path) == entries


def test_metadata_accepts_only_serialization_roundoff():
    assert storage.compatible_metadata({'rh': 1.234567890123456}, {'rh': 1.23456789012346})
    assert not storage.compatible_metadata({'rh': 7.0}, {'rh': 6.999999})
    assert not storage.compatible_metadata({'numOrbits': 100000}, {'numOrbits': 99999})


def test_second_controller_does_not_stop_the_running_one(store):
    with open(store.control / 'uploader.lock', 'a') as lock:
        storage.fcntl.flock(lock, storage.fcntl.LOCK_EX)
        result = subprocess.run([sys.executable, str(ROOT / 'py/orblib_storage.py'), 'prepare',
                                 '--root', str(store.root)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not store.stopped()


@pytest.mark.parametrize('pending', [False, True])
def test_final_check_is_local_and_rejects_pending_queue(store, pending):
    if pending:
        ready(store)
    result = subprocess.run([sys.executable, str(ROOT / 'py/orblib_storage.py'), 'check',
                             '--root', str(store.root)], capture_output=True, text=True, timeout=10)
    assert result.returncode == (75 if pending else 0), result.stderr
    if pending:
        assert (store.root / NAME).exists()
        assert store.stopped()


@pytest.mark.parametrize('scenario,no_shutdown', [
    ('upload_failure', False), ('upload_failure', True),
    ('uploader_crash', False), ('recovery_failure', False),
])
def test_launcher_checkpoints_before_shutdown_without_network_wait(tmp_path, scenario, no_shutdown):
    launcher = tmp_path / 'launch_orblib_exp.sh'
    launcher.write_text((ROOT / 'py/launch_orblib_exp.sh').read_text())
    for name in ('Fornax_P21_PCA_w3Sersic_orblib_exp.py', 'table3.dat', 'orblib_storage.py'):
        (tmp_path / name).touch()
    config = tmp_path / '.config/rclone'
    config.mkdir(parents=True)
    (config / 'rclone.conf').touch()
    prefix = r'''
hostname() { printf 'testhost'; }
nproc() { printf '8'; }
curl() { return 1; }
rclone() { printf 'network %s\n' "$*" >> "$HOME/events"; return 0; }
sudo() { [ "$1" = shutdown ] && printf 'shutdown\n' >> "$HOME/events"; return 0; }
python3() {
    mkdir -p "$WORK_DIR/orblib/.storage"
    printf 'storage %s\n' "$2" >> "$HOME/events"
    case "$2" in
        prepare)
            if [ "$SCENARIO" = recovery_failure ]; then
                touch "$WORK_DIR/orblib/.storage/STOP"
                return 75
            fi ;;
        watch)
            while [ ! -f "$HOME/active_p0" ] || [ ! -f "$HOME/active_p1" ]; do sleep 0.02; done
            printf 'delivery_exhausted\n' >> "$HOME/events"
            if [ "$SCENARIO" != uploader_crash ]; then touch "$WORK_DIR/orblib/.storage/STOP"; fi
            return 75 ;;
        stop) touch "$WORK_DIR/orblib/.storage/STOP" ;;
        finish) printf 'UNEXPECTED retry\n' >> "$HOME/events" ;;
    esac
    return 0
}
docker() {
    case "$1" in
        image|info|stats|stop|kill) return 0 ;;
    esac
    local previous='' suffix=''
    for arg in "$@"; do
        if [ "$previous" = --suffix ]; then suffix="$arg"; fi
        previous="$arg"
    done
    printf 'started_%s\n' "$suffix" >> "$HOME/events"
    touch "$HOME/active_${suffix}"
    while [ ! -f "$WORK_DIR/orblib/.storage/STOP" ]; do sleep 0.02; done
    printf '90 1 0.4 7 10 0.6 5\n' > "$WORK_DIR/out_${HOSTNAME_ENV}_${EXP_ID}_${suffix}.txt"
    printf 'pending library\n' > "$WORK_DIR/orblib/pending_${suffix}.npz"
    printf 'checkpoint after model\n' > "$WORK_DIR/checkpoint_${HOSTNAME_ENV}_${EXP_ID}_${suffix}.pkl"
    printf 'checkpoint_%s\n' "$suffix" >> "$HOME/events"
    return 75
}
source "$0" "$@"
'''
    args = ['bash', '-c', prefix, str(launcher), '--Q1', '--nproc=2', '--resume']
    if no_shutdown:
        args.append('--no-shutdown')
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, timeout=30,
                            env=dict(os.environ, HOME=str(tmp_path), SCENARIO=scenario,
                                     ORBLIB_SWAPFILE='0'))
    assert result.returncode != 0, result.stdout + result.stderr
    events = (tmp_path / 'events').read_text().splitlines()
    assert ('shutdown' in events) != no_shutdown
    assert 'UNEXPECTED retry' not in events
    assert events.count('storage prepare') == 1
    assert not any(' cat ' in event or ' rcat ' in event for event in events)
    if scenario == 'recovery_failure':
        assert not any(event.startswith('started_') for event in events)
    else:
        exhausted = events.index('delivery_exhausted')
        # Единственная разрешённая сеть после STOP — аварийная выгрузка логов
        # (обычный upload_to_yadisk при STOP остаётся no-op).
        after = [event for event in events[exhausted:] if event.startswith('network ')]
        assert after and all('galaxy_results_emergency' in event for event in after)
        if not no_shutdown:
            last = max(i for i, event in enumerate(events) if 'galaxy_results_emergency' in event)
            assert last < events.index('shutdown')
        for suffix in ('p0', 'p1'):
            assert events.count('started_' + suffix) == 1
            assert events.index('checkpoint_' + suffix) > exhausted
            if not no_shutdown:
                assert events.index('checkpoint_' + suffix) < events.index('shutdown')
            assert (tmp_path / f'orblib/pending_{suffix}.npz').exists()
            assert (tmp_path / f'checkpoint_testhost_Q1d1_nb250_gh0_ser0_{suffix}.pkl').exists()


WATCHDOG_PREFIX = r'''
hostname() { printf 'testhost'; }
nproc() { printf '8'; }
curl() { printf 'notify %s\n' "$*" >> "$HOME/events"; return 0; }
rclone() { printf 'network %s\n' "$*" >> "$HOME/events"; return 0; }
sudo() {
    if [ "$1" = shutdown ]; then
        printf 'shutdown\n' >> "$HOME/events"
        return "${SHUTDOWN_RC:-0}"
    fi
    return 0
}
swapon() { return 0; }
python3() {
    mkdir -p "$WORK_DIR/orblib/.storage"
    printf 'storage %s\n' "$2" >> "$HOME/events"
    case "$2" in
        watch) while [ ! -f "$WORK_DIR/orblib/.storage/STOP" ]; do sleep 0.05; done; return 0 ;;
        stop) touch "$WORK_DIR/orblib/.storage/STOP" ;;
    esac
    return 0
}
docker() {
    case "$1" in
        image|info|stats) return 0 ;;
        stop) printf 'docker_stop\n' >> "$HOME/events"; return 0 ;;
        kill) return 0 ;;
    esac
    printf 'run %s\n' "$*" >> "$HOME/events"
    while [ ! -f "$WORK_DIR/orblib/.storage/STOP" ]; do sleep 0.05; done
    return 75
}
source "$0" "$@"
'''


def watchdog_workdir(tmp_path, avail_mb, psi):
    launcher = tmp_path / 'launch_orblib_exp.sh'
    launcher.write_text((ROOT / 'py/launch_orblib_exp.sh').read_text())
    for name in ('Fornax_P21_PCA_w3Sersic_orblib_exp.py', 'table3.dat', 'orblib_storage.py'):
        (tmp_path / name).touch()
    config = tmp_path / '.config/rclone'
    config.mkdir(parents=True)
    (config / 'rclone.conf').touch()
    (tmp_path / 'meminfo').write_text(
        f'MemTotal:       32000000 kB\nMemAvailable:   {avail_mb * 1024} kB\n')
    (tmp_path / 'psi').write_text(f'some avg10=0.00 avg60={psi} avg300=0.00 total=1\n'
                                  f'full avg10=0.00 avg60=0.00 avg300=0.00 total=1\n')
    return launcher


@pytest.mark.parametrize('trigger', ['pressure', 'memavail', 'stall'])
def test_launcher_watchdog_stops_workers_and_uploads_before_shutdown(tmp_path, trigger):
    """Оба триггера ватчдога: порядок — docker stop → аварийная выгрузка → shutdown."""
    avail, psi = 8000, '0.00'
    env = dict(os.environ, HOME=str(tmp_path), ORBLIB_SWAPFILE='0',
               ORBLIB_WATCH_INTERVAL='1', ORBLIB_STOP_GRACE='1',
               ORBLIB_MEMINFO=str(tmp_path / 'meminfo'), ORBLIB_PSI_PATH=str(tmp_path / 'psi'),
               ORBLIB_MIN_AVAIL_MB='100', ORBLIB_PSI_LIMIT='1000',
               ORBLIB_PSI_SAMPLES='1', ORBLIB_STALL_TIMEOUT='3600')
    if trigger == 'pressure':
        psi, env['ORBLIB_PSI_LIMIT'] = '55.00', '20'
    elif trigger == 'memavail':
        avail, env['ORBLIB_MIN_AVAIL_MB'] = 500, '2048'
    else:
        env['ORBLIB_STALL_TIMEOUT'] = '3'
    launcher = watchdog_workdir(tmp_path, avail, psi)
    result = subprocess.run(['bash', '-c', WATCHDOG_PREFIX, str(launcher), '--Q1', '--nproc=2'],
                            cwd=tmp_path, text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 75, result.stdout[-4000:] + result.stderr[-4000:]
    events = (tmp_path / 'events').read_text().splitlines()
    runs = [event for event in events if event.startswith('run ')]
    assert len(runs) == 2
    limit = next(part for part in runs[0].split() if part.startswith('--memory='))
    assert f'--memory-swap={limit.split("=")[1]}' in runs[0].split()
    assert 'docker_stop' in events
    assert 'storage stop' in events
    emergency = max(i for i, event in enumerate(events) if 'galaxy_results_emergency' in event)
    assert emergency < events.index('shutdown')
    log = next(tmp_path.glob('launch_orblib_*.log')).read_text()
    assert 'СРАБОТАЛ ВАТЧДОГ' in log
    assert list(tmp_path.glob('monitor_Q1d1_nb250_gh0_ser0_i90.0_*.log'))


def test_launcher_reports_a_failed_shutdown_instead_of_hiding_it(tmp_path):
    env = dict(os.environ, HOME=str(tmp_path), ORBLIB_SWAPFILE='0', SHUTDOWN_RC='1',
               ORBLIB_WATCH_INTERVAL='1', ORBLIB_STOP_GRACE='1',
               ORBLIB_MEMINFO=str(tmp_path / 'meminfo'), ORBLIB_PSI_PATH=str(tmp_path / 'psi'),
               ORBLIB_MIN_AVAIL_MB='2048', ORBLIB_PSI_LIMIT='1000', ORBLIB_PSI_SAMPLES='1',
               ORBLIB_STALL_TIMEOUT='3600')
    launcher = watchdog_workdir(tmp_path, 500, '0.00')
    result = subprocess.run(['bash', '-c', WATCHDOG_PREFIX, str(launcher), '--Q1', '--nproc=2'],
                            cwd=tmp_path, text=True, capture_output=True, timeout=60, env=env)
    assert result.returncode == 75, result.stdout[-4000:] + result.stderr[-4000:]
    log = next(tmp_path.glob('launch_orblib_*.log')).read_text()
    assert 'НЕ УДАЛОСЬ запланировать выключение' in log
    events = (tmp_path / 'events').read_text().splitlines()
    assert any(event.startswith('notify ') and 'Priority: urgent' in event
               and 'shutdown' in event for event in events)


def test_legacy_index_is_required_to_be_current(store):
    remote = Remote()
    archive = 'orblib_i90.0_d1_nb250_ser0__host_20260101_000000.tar'
    remote.files[archive] = b'new version'
    remote.files['catalog/legacy.json'] = json.dumps(
        dict(archive=archive, size=3, md5=storage.digest(b'old'), entries=[])).encode()
    with pytest.raises(RuntimeError, match='index'):
        storage.prepare(store, remote, resume=True)


@pytest.mark.parametrize('failures,stop_during_orbit', [(1, False), (2, False), (0, True)])
def test_evaluation_saves_same_arrays_or_stops_without_fake_penalty(store, monkeypatch, failures, stop_during_orbit):
    class BeforeSolve(BaseException):
        pass

    class Constraints:
        def __gt__(self, other):
            raise BeforeSolve

    calls = []
    orbits = []

    def orbit(**kwargs):
        orbits.append(kwargs)
        if stop_during_orbit:
            store.stop('delivery exhausted while model was running')
        return [numpy.ones((2, 3)), numpy.ones((2, 3))]

    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    ns = script_namespace(
        ['halo_IC_lib_weights_pca_fixed', 'orblib_key', 'write_orblib_npz', 'log_mem'],
        numpy=numpy, os=os, time=time, contextlib=contextlib, zipfile=zipfile,
        hashlib=storage.hashlib, _NPZ_BLOCK_BYTES=4096, TRAJSIZE_STORED=0,
        HashingWriter=storage.HashingWriter,
        orblib_store=store, StorageStop=storage.StorageStop, LibraryBusy=storage.LibraryBusy,
        OrblibBusyError=RuntimeError, completed_point=lambda params: False,
        Q1=True, DOUBLE=True, N_BIN=250, SER_ID=0, GEOM_HASH='abcdef', incl=90.0,
        ORBLIB_DIR=str(store.root), SAVE_ORBLIB=True, REUSE_ORBLIB=True,
        hostname_proc='test', _ORBLIB_BUILD_TTL_SEC=7200, _claim_file=lambda *a: True,
        release_reservation=lambda *a: None, orblib_counter=0,
        gridv=numpy.arange(5.0), degree=2, ghorder=6, AGAMA_ORBIT_THREADS=8,
        agama=SimpleNamespace(Density=lambda *a, **kw: None, orbit=orbit,
                              setNumThreads=lambda n: contextlib.nullcontext(),
                              Potential=lambda **kw: SimpleNamespace(Tcirc=lambda ic: numpy.ones(2))))
    original = ns['write_orblib_npz']

    def save(stream, matrices, *args, **kwargs):
        calls.append(matrices['matrix_kinem'])
        if len(calls) <= failures:
            raise OSError('temporary write error')
        return original(stream, matrices, *args, **kwargs)

    ns['write_orblib_npz'] = save
    stars = SimpleNamespace(sample=lambda *a, **kw: [numpy.zeros((2, 6))])
    datasets = [SimpleNamespace(target=[0, 1, 2], cons_err=Constraints())] * 2
    bounds = dict(Q=(0.05, 2.5), gh=(0, 1.6), rh=(0.5, 7), rho0=(10, 120))
    error = storage.StorageStop if failures == 2 else BeforeSolve
    with pytest.raises(error):
        ns['halo_IC_lib_weights_pca_fixed'](
            None, None, bounds, stars, datasets, 2, 3, numOrbits=2,
            direct_params=dict(Q=1.0, gh=0.4, rh=7.0, rho0=10.0))
    assert len(orbits) == 1
    assert 'trajsize' not in orbits[0]
    assert len(calls) == (1 if failures == 0 else 2)
    assert all(array is calls[0] for array in calls)
    if failures == 2:
        assert store.stopped()
        assert not list(store.root.glob('*.npz'))
    else:
        assert len(store.pending()) == 1
        assert len(list(store.root.glob('*.npz'))) == 1


def written_library(scalars=None, rows=40):
    """Библиотека, записанная новым блочным писателем; возвращает (bytes, writer)."""
    dens = numpy.arange(rows * 3, dtype=numpy.float32).reshape(rows, 3)
    kinem = numpy.arange(rows * 7, dtype=numpy.float32).reshape(rows, 7) * 0.5
    ic = numpy.arange(rows * 6, dtype=numpy.float64).reshape(rows, 6)
    inttime = numpy.arange(rows, dtype=numpy.float64) + 1.0
    meta = dict(Q=1.0, gh=0.0, rh=7.0, rho0=10.0, incl=90.0, numOrbits=rows,
                trajsize=0, intTime=100.0, gridv=numpy.arange(5.0), degree=2,
                ghorder=6, n_bin=250, double=1, ser_id=0)
    meta.update(scalars or {})
    stream = io.BytesIO()
    writer = storage.HashingWriter(stream)
    writer_namespace()['write_orblib_npz'](
        writer, dict(matrix_dens=dens, matrix_kinem=kinem), ic, inttime, meta)
    return stream.getvalue(), writer, dict(matrix_dens=dens, matrix_kinem=kinem,
                                           ic=ic, inttime=inttime)


def test_block_writer_reproduces_savez_content_and_hashes_in_one_pass():
    data, writer, arrays = written_library()
    assert writer.size == len(data)
    assert writer.checksum.hexdigest() == storage.digest(data)
    reference = io.BytesIO()
    numpy.savez_compressed(
        reference, matrix_dens=arrays['matrix_dens'].astype(numpy.float64),
        matrix_kinem=arrays['matrix_kinem'].astype(numpy.float64),
        ic=arrays['ic'], inttime=arrays['inttime'],
        Q=1.0, gh=0.0, rh=7.0, rho0=10.0, incl=90.0, numOrbits=40, trajsize=0,
        intTime=100.0, gridv=numpy.arange(5.0), degree=2, ghorder=6,
        n_bin=250, double=1, ser_id=0)
    reference.seek(0)
    with numpy.load(io.BytesIO(data)) as new, numpy.load(reference) as old:
        assert sorted(new.files) == sorted(old.files)
        for name in old.files:
            assert new[name].dtype == old[name].dtype, name
            assert numpy.array_equal(new[name], old[name]), name
        assert new['matrix_kinem'].dtype == numpy.dtype('<f8')
    assert storage.metadata_bytes(data)['trajsize'] == 0


def test_block_writer_never_materialises_a_float64_copy():
    import tracemalloc
    big = numpy.ones((200000, 7), dtype=numpy.float32)
    write = writer_namespace()['write_orblib_npz']
    sink = SimpleNamespace(write=lambda data: None, tell=lambda: 0, flush=lambda: None)
    tracemalloc.start()
    write(sink, dict(matrix_kinem=big), numpy.zeros((2, 6)), numpy.zeros(2), {})
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < big.size * 8 / 10, peak


@pytest.mark.parametrize('damage', ['truncate', 'flip'])
def test_written_identity_rejects_a_damaged_file(store, damage):
    data, writer, _ = written_library()
    path = store.root / NAME
    if damage == 'truncate':
        path.write_bytes(data[:-64])
    else:
        index = len(data) // 2
        path.write_bytes(data[:index] + bytes([data[index] ^ 0xFF]) + data[index + 1:])
    with pytest.raises((ValueError, zipfile.BadZipFile, EOFError)):
        store.register(NAME, size=writer.size, md5=writer.checksum.hexdigest())
    assert not store.pending()


def test_register_rejects_a_mismatching_identity(store):
    data, writer, _ = written_library()
    (store.root / NAME).write_bytes(data)
    with pytest.raises(ValueError, match='MD5 mismatch'):
        store.register(NAME, size=writer.size, md5=storage.digest(b'other'))
    with pytest.raises(ValueError, match='size mismatch'):
        store.register(NAME, size=writer.size + 1, md5=writer.checksum.hexdigest())
    store.register(NAME, size=writer.size, md5=writer.checksum.hexdigest())
    assert store.row(NAME)['md5'] == writer.checksum.hexdigest()


def test_shallow_metadata_skips_the_crc_pass_but_still_reads_headers(store, monkeypatch):
    data, _, _ = written_library()
    path = store.root / NAME
    path.write_bytes(data)
    monkeypatch.setattr(zipfile.ZipFile, 'testzip', lambda self: pytest.fail('deep CRC pass'))
    assert storage.metadata(path, deep=False)['numOrbits'] == 40


def test_trajsize_is_outside_the_compatibility_key(store):
    """Старая (trajsize=1000) и новая (0) библиотеки равно пригодны."""
    expected = dict(Q=1.0, gh=0.0, rh=7.0, rho0=10.0, incl=90.0, numOrbits=40,
                    intTime=100.0, degree=2, ghorder=6, n_bin=250, double=1, ser_id=0,
                    gridv_md5=storage.digest(numpy.arange(5.0).tobytes()),
                    gridv_dtype='<f8')
    for stored in (0, 1000):
        data, _, _ = written_library(dict(trajsize=stored))
        assert storage.compatible_metadata(storage.metadata_bytes(data), expected)
    data, _, _ = written_library()
    conflicting = dict(expected, gridv_md5=storage.digest(b'other grid'))
    assert not storage.compatible_metadata(storage.metadata_bytes(data), conflicting)


def test_save_slot_caps_concurrent_writers_and_observes_stop(store, monkeypatch):
    store.save_slots = 2
    with store.save_slot(), store.save_slot():
        monkeypatch.setattr(storage.time, 'sleep', lambda seconds: store.stop('no free slot'))
        with pytest.raises(storage.StorageStop):
            with store.save_slot():
                pytest.fail('a third writer must wait for a slot')
    store.stop_path.unlink()
    monkeypatch.undo()
    with pytest.raises(RuntimeError):
        with store.save_slot():
            raise RuntimeError('writer failed')
    with store.save_slot(), store.save_slot():
        pass


def test_sigterm_requests_a_stop_that_becomes_a_checkpoint(store, tmp_path):
    checkpoints = []
    ns = script_namespace(['_handle_stop_signal'], signal=signal, orblib_store=store,
                          checkpoint_for_stop=lambda: checkpoints.append('written'))
    ns['_handle_stop_signal'](int(signal.SIGTERM), None)
    assert store.stop_path.read_text() == 'signal SIGTERM from controller'
    # Расчётный цикл опрашивает STOP и превращает его в StorageStop(75),
    # который верхний обработчик скрипта конвертирует в checkpoint + exit 75.
    with pytest.raises(storage.StorageStop) as error:
        store.check_stop()
    assert error.value.code == 75
    assert not checkpoints
    # Без хранилища checkpoint пишется напрямую.
    bare = script_namespace(['_handle_stop_signal'], signal=signal, orblib_store=None,
                            checkpoint_for_stop=lambda: checkpoints.append('direct'))
    with pytest.raises(SystemExit) as exit_error:
        bare['_handle_stop_signal'](int(signal.SIGINT), None)
    assert exit_error.value.code == 75
    assert checkpoints == ['direct']


ARCHIVE = 'orblib_i90.0_d1_nb250_ser0__test_20260919_011920_part000.tar'


def local_archive(tmp_path, archive_name=ARCHIVE):
    archives = tmp_path / 'orblib'
    archives.mkdir(exist_ok=True)
    payload_file = tmp_path / 'payload.npz'
    library(payload_file)
    payload = payload_file.read_bytes()
    path = archives / archive_name
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo(NAME)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return path


def test_local_index_uses_seekable_members_without_staging_or_network(tmp_path, monkeypatch):
    archive = local_archive(tmp_path)
    original = archive.read_bytes()
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    monkeypatch.setattr(storage.tempfile, 'TemporaryFile', lambda **kw: pytest.fail('npz staging'))
    monkeypatch.setattr(storage.Rclone, 'run', lambda *a: pytest.fail('network during local index'))
    storage.index_local_archives(work, archive.parent)
    indexes = list((work.root / 'catalog').glob('*.json'))
    assert len(indexes) == 1
    record = json.loads(indexes[0].read_bytes())
    assert record['archive'] == ARCHIVE
    assert record['size'] == len(original)
    assert record['md5'] == storage.digest(original)
    assert record['entries'][0]['metadata']['rh'] == 7
    assert record['_local_stat']['size'] == len(original)
    assert archive.read_bytes() == original
    assert not list(work.root.rglob('*.npz'))
    assert not (archive.parent / 'catalog').exists()
    monkeypatch.setattr(storage, 'index_local_archive', lambda *a: pytest.fail('unchanged tar reindexed'))
    storage.index_local_archives(work, archive.parent)


def test_local_indexes_publish_only_after_cloud_verification(tmp_path):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    storage.index_local_archives(work, archive.parent)
    remote = Remote()
    original = archive.read_bytes()
    remote.files[ARCHIVE] = original
    storage.publish_local_indexes(work, remote, archive.parent)
    public = list((archive.parent / 'catalog').glob('*.json'))
    assert len(public) == 1
    record = json.loads(public[0].read_bytes())
    assert '_local_stat' not in record
    assert remote.read('catalog/' + public[0].name) == public[0].read_bytes()
    assert remote.files[ARCHIVE] == original
    assert all(name == ARCHIVE or name.startswith('catalog/') for name in remote.files)
    copies = remote.copies
    storage.publish_local_indexes(work, remote, archive.parent)
    assert remote.copies == copies
    reader = storage.Store(tmp_path / 'runtime', reserve_bytes=0)
    storage.import_catalog(reader, remote)
    assert reader.archived(NAME, record['entries'][0]['metadata'])


@pytest.mark.parametrize('change', ['cloud', 'local'])
def test_changed_tar_blocks_all_publication(tmp_path, change):
    first = local_archive(tmp_path)
    second = local_archive(tmp_path, ARCHIVE.replace('000.tar', '001.tar'))
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    storage.index_local_archives(work, first.parent)
    remote = Remote()
    remote.files = {p.name: p.read_bytes() for p in (first, second)}
    if change == 'cloud':
        data = remote.files[second.name]
        remote.files[second.name] = b'x' + data[1:]
    else:
        with second.open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises((ValueError, RuntimeError), match=second.name):
        storage.publish_local_indexes(work, remote, first.parent)
    assert remote.copies == 0
    assert not (first.parent / 'catalog').exists()


def test_local_change_during_publication_prevents_success(tmp_path):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    storage.index_local_archives(work, archive.parent)
    remote = Remote()
    remote.files[ARCHIVE] = archive.read_bytes()
    copy = remote.copy

    def mutate(path, name):
        copy(path, name)
        with archive.open('ab') as stream:
            stream.write(b'changed during publication')

    remote.copy = mutate
    with pytest.raises(ValueError, match='changed during publication'):
        storage.publish_local_indexes(work, remote, archive.parent)


def test_local_index_detects_source_change_during_read(tmp_path, monkeypatch):
    archive = local_archive(tmp_path)
    original_metadata = storage.metadata

    def change_source(member):
        result = original_metadata(member)
        with archive.open('ab') as stream:
            stream.write(b'changed')
        return result

    monkeypatch.setattr(storage, 'metadata', change_source)
    with pytest.raises(ValueError, match='changed'):
        storage.index_local_archive(archive)


def test_local_index_rejects_link_members_and_names_archive(tmp_path):
    archive = local_archive(tmp_path)
    with tarfile.open(archive, 'w') as stream:
        member = tarfile.TarInfo(NAME)
        member.type = tarfile.SYMTYPE
        member.linkname = '/etc/passwd'
        stream.addfile(member)
    with pytest.raises(ValueError, match=ARCHIVE):
        storage.index_local_archive(archive)


def test_local_index_work_must_be_outside_archive_directory(tmp_path):
    archive = local_archive(tmp_path)
    with pytest.raises(ValueError, match='overlap'):
        storage.index_locations(archive.parent / 'work', archive.parent)


def test_publish_does_not_overwrite_existing_local_catalog(tmp_path):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    storage.index_local_archives(work, archive.parent)
    staged = next((work.root / 'catalog').glob('*.json'))
    public = archive.parent / 'catalog'
    public.mkdir()
    existing = public / staged.name
    existing.write_bytes(b'previous content')
    remote = Remote()
    remote.files[ARCHIVE] = archive.read_bytes()
    with pytest.raises(FileExistsError):
        storage.publish_local_indexes(work, remote, archive.parent)
    assert existing.read_bytes() == b'previous content'
    assert remote.copies == 0


def test_local_index_cli_creates_default_workdir_without_rclone(tmp_path):
    archive = local_archive(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / 'py/orblib_storage.py'),
                             'index', '--local-archives', 'orblib'], cwd=tmp_path,
                            env=dict(os.environ, PATH=''), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((tmp_path / 'orblib-index-work/catalog').glob('*.json'))
    assert not (archive.parent / 'catalog').exists()


def test_stream_failure_reports_archive_hashes_and_rclone_stderr(tmp_path, monkeypatch):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    remote = SimpleNamespace(root='test:orblib', timeout=1,
                             inventory=lambda: {ARCHIVE: dict(size=archive.stat().st_size, md5='0' * 32)},
                             command=lambda *args: ['mock-rclone'])

    def popen(*args, stderr, **kwargs):
        stderr.write(b'network failure details')
        stderr.flush()
        return SimpleNamespace(stdout=io.BytesIO(archive.read_bytes()),
                               poll=lambda: 1, wait=lambda **kw: 1,
                               kill=lambda: pytest.fail('already exited'))

    monkeypatch.setattr(storage.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError) as error:
        storage.index_archives(work, remote)
    message = str(error.value)
    assert ARCHIVE in message
    assert 'expected=' in message and 'actual=' in message
    assert 'network failure details' in message and 'rclone cat exit=1' in message


def test_publish_retries_only_indexes_after_network_failure(tmp_path, monkeypatch):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    storage.index_local_archives(work, archive.parent)
    remote = Remote()
    remote.files[ARCHIVE] = archive.read_bytes()
    remote.fail = True
    with pytest.raises(OSError):
        storage.publish_local_indexes(work, remote, archive.parent)
    assert list((work.root / 'catalog').glob('*.json'))
    assert list((archive.parent / 'catalog').glob('*.json'))
    assert not any(name.startswith('catalog/') for name in remote.files)
    monkeypatch.setattr(storage, 'index_local_archive', lambda *args: pytest.fail('tar reread'))
    remote.fail = False
    storage.publish_local_indexes(work, remote, archive.parent)
    assert sum(name.endswith('.json') for name in remote.files) == 1


def test_staging_symlink_cannot_write_unverified_indexes_to_sync_folder(tmp_path):
    archive = local_archive(tmp_path)
    work = storage.Store(tmp_path / 'work', reserve_bytes=0)
    (work.root / 'catalog').symlink_to(archive.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        storage.index_local_archives(work, archive.parent)
    assert not list(archive.parent.glob('*.json'))


def test_local_index_cli_publish_with_realistic_mock_rclone(tmp_path):
    archive = local_archive(tmp_path)
    remote = tmp_path / 'remote/orblib'
    remote.mkdir(parents=True)
    (remote / ARCHIVE).write_bytes(archive.read_bytes())
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    mock = bindir / 'rclone'
    mock.write_text(f'#!{sys.executable}\n' + '''import hashlib
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')

def remote_path(value):
    return Path(os.environ['MOCK_REMOTE']) / value.removeprefix('mock:')

if args[0] == 'lsjson':
    path = remote_path(args[1])
    if not path.is_file():
        sys.exit(4)
    data = path.read_bytes()
    print(json.dumps(dict(Path=path.name, Name=path.name, IsDir=False,
                          Size=len(data), Hashes={'md5': hashlib.md5(data).hexdigest()})))
elif args[0] == 'copyto':
    path = remote_path(args[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(args[1]).read_bytes())
elif args[0] == 'moveto':
    remote_path(args[1]).rename(remote_path(args[2]))
else:
    raise RuntimeError('Unexpected rclone operation: ' + args[0])
''')
    mock.chmod(0o755)
    calls = tmp_path / 'calls.jsonl'
    args = [sys.executable, str(ROOT / 'py/orblib_storage.py'), 'index',
            '--local-archives', 'orblib', '--publish', '--remote', 'mock:orblib']
    env = dict(os.environ, PATH=str(bindir), MOCK_REMOTE=str(remote.parent), MOCK_CALLS=str(calls))
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    public = next((archive.parent / 'catalog').glob('*.json'))
    assert (remote / 'catalog' / public.name).read_bytes() == public.read_bytes()
    operations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert all(op[0] != 'cat' for op in operations)
    assert all('/catalog/' in op[2] for op in operations if op[0] == 'copyto')
    assert (remote / ARCHIVE).read_bytes() == archive.read_bytes()
    retry = subprocess.run([sys.executable, str(ROOT / 'py/orblib_storage.py'), 'publish-indexes',
                            '--local-archives', 'orblib', '--remote', 'mock:orblib'],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert retry.returncode == 0, retry.stdout + retry.stderr
    after = [json.loads(line) for line in calls.read_text().splitlines()]
    assert not any(op[0] == 'copyto' for op in after[len(operations):])


def test_cli_publish_flag_requires_local_archives(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'py/orblib_storage.py'), 'index', '--publish'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert '--publish requires' in result.stderr
    assert not (tmp_path / 'orblib-index-work').exists()


def test_manifest_cannot_reference_missing_library(store):
    remote = Remote()
    remote.files['catalog/' + NAME + '.json'] = json.dumps(dict(
        name=NAME, object='objects/' + NAME, size=1, md5=storage.digest(b'x'), metadata={})).encode()
    with pytest.raises(ValueError, match='Missing'):
        storage.prepare(store, remote, resume=True)
