import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import numpy
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('orblib_storage', ROOT / 'py/orblib_storage.py')
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)
NAME = 'orblib_i90.0_d1_nb250_ser0_geomabcdef_1234567890.npz'


def library(path):
    numpy.savez_compressed(path, Q=1.0, gh=0.0, rh=7.0, rho0=10.0, incl=90.0,
                           double=1, n_bin=250, ser_id=0, numOrbits=2, trajsize=1000,
                           intTime=100.0, degree=2, ghorder=6, gridv=numpy.arange(5.0),
                           ic=numpy.zeros((2, 6)), inttime=numpy.ones(2),
                           matrix_dens=numpy.zeros((2, 3)), matrix_kinem=numpy.ones((2, 5)))


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
sudo() { printf 'shutdown\n' >> "$HOME/events"; return 0; }
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
    if [ "$1" = image ]; then return 0; fi
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
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True, timeout=15,
                            env=dict(os.environ, HOME=str(tmp_path), SCENARIO=scenario))
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
        assert not any(event.startswith('network ') for event in events[exhausted:])
        for suffix in ('p0', 'p1'):
            assert events.count('started_' + suffix) == 1
            assert events.index('checkpoint_' + suffix) > exhausted
            if not no_shutdown:
                assert events.index('checkpoint_' + suffix) < events.index('shutdown')
            assert (tmp_path / f'orblib/pending_{suffix}.npz').exists()
            assert (tmp_path / f'checkpoint_testhost_Q1d1_nb250_gh0_ser0_{suffix}.pkl').exists()


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

    original = numpy.savez_compressed
    calls = []
    orbits = []

    def save(*args, **kwargs):
        calls.append(kwargs['matrix_kinem'])
        if len(calls) <= failures:
            raise OSError('temporary write error')
        return original(*args, **kwargs)

    def orbit(**kwargs):
        orbits.append(True)
        if stop_during_orbit:
            store.stop('delivery exhausted while model was running')
        return [numpy.ones((2, 3)), numpy.ones((2, 3)), None]

    monkeypatch.setattr(numpy, 'savez_compressed', save)
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    source = ROOT / 'py/Fornax_P21_PCA_w3Sersic_orblib_exp.py'
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in ('halo_IC_lib_weights_pca_fixed', 'orblib_key')]
    ns = dict(numpy=numpy, os=os, time=time, hashlib=storage.hashlib,
              orblib_store=store, StorageStop=storage.StorageStop, LibraryBusy=storage.LibraryBusy,
              OrblibBusyError=RuntimeError, completed_point=lambda params: False,
              Q1=True, DOUBLE=True, N_BIN=250, SER_ID=0, GEOM_HASH='abcdef', incl=90.0,
              ORBLIB_DIR=str(store.root), SAVE_ORBLIB=True, REUSE_ORBLIB=True,
              hostname_proc='test', _ORBLIB_BUILD_TTL_SEC=7200, _claim_file=lambda *a: True,
              release_reservation=lambda *a: None, orblib_counter=0,
              gridv=numpy.arange(5.0), degree=2, ghorder=6,
              agama=SimpleNamespace(Density=lambda *a, **kw: None, orbit=orbit,
                                    Potential=lambda **kw: SimpleNamespace(Tcirc=lambda ic: numpy.ones(2))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    stars = SimpleNamespace(sample=lambda *a, **kw: [numpy.zeros((2, 6))])
    datasets = [SimpleNamespace(target=[0, 1, 2], cons_err=Constraints())] * 2
    bounds = dict(Q=(0.05, 2.5), gh=(0, 1.6), rh=(0.5, 7), rho0=(10, 120))
    error = storage.StorageStop if failures == 2 else BeforeSolve
    with pytest.raises(error):
        ns['halo_IC_lib_weights_pca_fixed'](
            None, None, bounds, stars, datasets, 2, 3, numOrbits=2,
            direct_params=dict(Q=1.0, gh=0.4, rh=7.0, rho0=10.0))
    assert len(orbits) == 1
    assert len(calls) == (1 if failures == 0 else 2)
    assert all(array is calls[0] for array in calls)
    if failures == 2:
        assert store.stopped()
        assert not list(store.root.glob('*.npz'))
    else:
        assert len(store.pending()) == 1
        assert len(list(store.root.glob('*.npz'))) == 1


def test_manifest_cannot_reference_missing_library(store):
    remote = Remote()
    remote.files['catalog/' + NAME + '.json'] = json.dumps(dict(
        name=NAME, object='objects/' + NAME, size=1, md5=storage.digest(b'x'), metadata={})).encode()
    with pytest.raises(ValueError, match='Missing'):
        storage.prepare(store, remote, resume=True)
