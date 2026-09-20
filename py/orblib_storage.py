import argparse
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import subprocess
import tarfile
import tempfile
import time
import zipfile


NAME_RE = re.compile(r'orblib_i[-+0-9.]+_d[01]_nb[0-9]+_ser[0-9]+_geom[0-9a-fA-F]+_[0-9a-fA-F]{10}\.npz')
SCALARS = ('Q', 'gh', 'rh', 'rho0', 'incl', 'double', 'n_bin', 'ser_id',
           'numOrbits', 'trajsize', 'intTime', 'degree', 'ghorder')


class StorageStop(SystemExit):
    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__(75)


class LibraryBusy(RuntimeError):
    pass


def digest(data):
    return hashlib.md5(data).hexdigest()


def file_hash(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def npy_header(stream):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('Invalid npy magic')
    version = stream.read(2)
    size_format = '<H' if version == b'\x01\x00' else '<I'
    length = struct.unpack(size_format, stream.read(struct.calcsize(size_format)))[0]
    if length > 65536:
        raise ValueError('Oversized npy header')
    return ast.literal_eval(stream.read(length).decode('latin1').strip())


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        result = {}
        for name in SCALARS:
            with archive.open(name + '.npy') as stream:
                header = npy_header(stream)
                dtype = header['descr']
                formats = {'f8': 'd', 'f4': 'f', 'i8': 'q', 'i4': 'i', 'u8': 'Q', 'u4': 'I'}
                if header['shape'] != () or dtype[1:] not in formats:
                    raise ValueError('Invalid scalar metadata: ' + name)
                fmt = ('>' if dtype[0] == '>' else '<') + formats[dtype[1:]]
                value = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
                if not math.isfinite(value):
                    raise ValueError('Non-finite metadata: ' + name)
                result[name] = value
        with archive.open('gridv.npy') as stream:
            header = npy_header(stream)
            result['gridv_md5'] = digest(stream.read())
            result['gridv_dtype'] = header['descr']
        for name in ('matrix_dens', 'matrix_kinem', 'ic', 'inttime'):
            with archive.open(name + '.npy') as stream:
                header = npy_header(stream)
            if header['descr'] not in ('<f8', '>f8', '=f8'):
                raise ValueError('Expected float64: ' + name)
            if not header['shape'] or header['shape'][0] != result['numOrbits']:
                raise ValueError('Invalid array shape: ' + name)
        if archive.testzip() is not None:
            raise ValueError('Corrupt npz member')
    return result


def metadata_bytes(data):
    return metadata(io.BytesIO(data))


def compatible_metadata(actual, expected):
    for name, value in expected.items():
        other = actual.get(name)
        if isinstance(value, (int, float)) and isinstance(other, (int, float)):
            if not math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12):
                return False
        elif value != other:
            return False
    return True


def validate_name(name):
    if not NAME_RE.fullmatch(name):
        raise ValueError('Invalid library name: ' + name)


class Store:
    def __init__(self, root, reserve_bytes=2_000_000_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.control = self.root / '.storage'
        self.control.mkdir(exist_ok=True)
        self.reserve_bytes = reserve_bytes
        self.stop_path = self.control / 'STOP'
        self.finish_path = self.control / 'FINISH'
        self.db = self.control / 'queue.sqlite'
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS libraries (name TEXT PRIMARY KEY, state TEXT NOT NULL, '
                       'metadata TEXT, size INTEGER, md5 TEXT, receipt TEXT, managed INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS reservations (name TEXT PRIMARY KEY, bytes INTEGER NOT NULL)')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def stopped(self):
        return self.stop_path.exists()

    def stop(self, reason):
        if not self.stopped():
            atomic_bytes(self.stop_path, str(reason).encode())

    def check_stop(self):
        if self.stopped():
            raise StorageStop(self.stop_path.read_text())

    def connection_rows(self):
        with self.connection() as db:
            return db.execute('SELECT * FROM libraries').fetchall()

    def row(self, name):
        with self.connection() as db:
            return db.execute('SELECT * FROM libraries WHERE name=?', (name,)).fetchone()

    def archived(self, name, expected=None):
        row = self.row(name)
        if not row:
            return False
        actual = json.loads(row['metadata']) if row['metadata'] else None
        if actual is not None and expected is not None and not compatible_metadata(actual, expected):
            raise ValueError('Conflicting library metadata: ' + name)
        return bool(row['receipt'])

    def lock(self, name):
        validate_name(name)
        stream = open(self.control / (name + '.lock'), 'a')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise LibraryBusy(name)
        return stream

    def claim(self, name, size):
        self.check_stop()
        stream = self.lock(name)
        try:
            while True:
                self.check_stop()
                with self.connection() as db:
                    db.execute('BEGIN IMMEDIATE')
                    outstanding = db.execute('SELECT COALESCE(SUM(bytes),0) FROM reservations').fetchone()[0]
                    stats = os.statvfs(self.root)
                    if stats.f_bavail * stats.f_frsize >= self.reserve_bytes + outstanding + size and stats.f_favail > 32:
                        db.execute('INSERT OR REPLACE INTO reservations VALUES (?,?)', (name, size))
                        db.execute("INSERT OR IGNORE INTO libraries(name,state,managed) VALUES (?,'building',1)", (name,))
                        return stream
                if not outstanding and not self.pending():
                    self.stop(f'Insufficient disk space/inodes for {name}; no uploads can free space')
                    self.check_stop()
                time.sleep(1)
        except BaseException:
            stream.close()
            raise

    def release(self, stream, name):
        if stream is not None:
            try:
                with self.connection() as db:
                    db.execute('DELETE FROM reservations WHERE name=?', (name,))
            finally:
                stream.close()

    def register(self, name, managed=True):
        validate_name(name)
        path = self.root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Not a regular library: ' + name)
        meta = metadata(path)
        checksum = file_hash(path)
        with open(path, 'rb') as stream:
            os.fsync(stream.fileno())
        fsync_directory(self.root)
        with self.connection() as db:
            old = db.execute('SELECT * FROM libraries WHERE name=?', (name,)).fetchone()
            if old and old['metadata'] and not compatible_metadata(json.loads(old['metadata']), meta):
                raise ValueError('Conflicting library metadata: ' + name)
            receipt = old['receipt'] if old else None
            db.execute('INSERT OR REPLACE INTO libraries VALUES (?,?,?,?,?,?,?)',
                       (name, 'ready', json.dumps(meta, sort_keys=True), path.stat().st_size,
                        checksum, receipt, old['managed'] if old else int(managed)))
            db.execute('DELETE FROM reservations WHERE name=?', (name,))

    def pending(self):
        with self.connection() as db:
            return db.execute("SELECT * FROM libraries WHERE state IN ('ready','uploading','verified') ORDER BY name").fetchall()

    def receipt(self, receipt):
        name = receipt['name']
        validate_name(name)
        with self.connection() as db:
            old = db.execute('SELECT * FROM libraries WHERE name=?', (name,)).fetchone()
            meta = json.dumps(receipt['metadata'], sort_keys=True)
            if old and old['metadata'] and not compatible_metadata(json.loads(old['metadata']), receipt['metadata']):
                raise ValueError('Conflicting library metadata: ' + name)
            if old:
                db.execute('UPDATE libraries SET receipt=?, metadata=? WHERE name=?',
                           (json.dumps(receipt, sort_keys=True), meta, name))
            else:
                db.execute('INSERT INTO libraries VALUES (?,?,?,?,?,?,?)',
                           (name, 'remote', meta, receipt['size'], receipt['md5'],
                            json.dumps(receipt, sort_keys=True), 0))

    def remove_verified(self, name):
        row = self.row(name)
        if not row['receipt']:
            raise RuntimeError('Missing receipt: ' + name)
        path = self.root / name
        if row['managed'] and path.exists():
            if path.is_symlink() or path.stat().st_size != row['size'] or file_hash(path) != row['md5']:
                raise ValueError('Local file changed before cleanup: ' + name)
            path.unlink()
            fsync_directory(self.root)
        with self.connection() as db:
            db.execute("UPDATE libraries SET state='remote' WHERE name=?", (name,))


class Rclone:
    def __init__(self, root, config, timeout=900):
        self.root = root.rstrip('/')
        self.config = config
        self.timeout = timeout
        self.deadline = None

    def command(self, *args):
        remaining = self.timeout if self.deadline is None else self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Delivery deadline exceeded')
        return ['rclone', *args, '--config', self.config, '--retries', '1', '--low-level-retries', '1',
                '--timeout', '1m', '--contimeout', '30s', '--max-duration', f'{max(1, int(remaining))}s',
                '--cutoff-mode', 'HARD']

    def run(self, *args):
        remaining = self.timeout if self.deadline is None else max(0.01, self.deadline - time.monotonic())
        result = subprocess.run(self.command(*args), capture_output=True, timeout=remaining)
        if result.returncode:
            raise RuntimeError(f'rclone {args[0]} failed ({result.returncode}): '
                               + result.stderr.decode(errors='replace')[-1000:])
        return result.stdout

    def inventory(self):
        rows = json.loads(self.run('lsjson', self.root, '--recursive', '--files-only', '--hash'))
        return {r['Path']: dict(size=r['Size'], md5=r.get('Hashes', {}).get('MD5', '').lower()) for r in rows}

    def stat(self, name):
        return self.inventory().get(name)

    def read(self, name):
        return self.run('cat', self.root + '/' + name)

    def copy(self, path, name):
        self.run('copyto', str(path), self.root + '/' + name, '--checksum')

    def move(self, source, destination):
        if self.stat(destination) is not None:
            raise FileExistsError(destination)
        self.run('moveto', self.root + '/' + source, self.root + '/' + destination, '--immutable')


def verify(remote, name, size, checksum):
    actual = remote.stat(name)
    if actual != dict(size=size, md5=checksum):
        raise ValueError('Remote size/MD5 mismatch: ' + name)


def publish(remote, path, destination):
    size, checksum = Path(path).stat().st_size, file_hash(path)
    if remote.stat(destination) is not None:
        verify(remote, destination, size, checksum)
        return
    temporary = destination + '.' + checksum + '.partial'
    remote.copy(path, temporary)
    verify(remote, temporary, size, checksum)
    remote.move(temporary, destination)
    verify(remote, destination, size, checksum)


def verify_receipt(remote, receipt):
    if 'archive' in receipt:
        verify(remote, receipt['archive'], receipt['archive_size'], receipt['archive_md5'])
    else:
        verify(remote, receipt['object'], receipt['size'], receipt['md5'])
        if json.loads(remote.read('catalog/' + receipt['name'] + '.json')) != receipt:
            raise ValueError('Remote receipt changed: ' + receipt['name'])


def transfer(store, remote, row):
    name = row['name']
    receipt = json.loads(row['receipt']) if row['receipt'] else None
    if receipt:
        verify_receipt(remote, receipt)
    else:
        destination = 'objects/' + name
        publish(remote, store.root / name, destination)
        receipt = dict(name=name, object=destination, size=row['size'], md5=row['md5'],
                       metadata=json.loads(row['metadata']))
        path = store.control / (name + '.json')
        atomic_bytes(path, json.dumps(receipt, sort_keys=True).encode())
        publish(remote, path, 'catalog/' + name + '.json')
        store.receipt(receipt)
    with store.connection() as db:
        db.execute("UPDATE libraries SET state='verified' WHERE name=?", (name,))
    store.remove_verified(name)


def deliver(store, remote, attempts=3, delay=5, recovering=False):
    if store.stopped() and not recovering:
        return False
    for row in store.pending():
        try:
            lock = store.lock(row['name'])
        except LibraryBusy:
            continue
        try:
            row = store.row(row['name'])
            with store.connection() as db:
                db.execute("UPDATE libraries SET state='uploading' WHERE name=?", (row['name'],))
            for attempt in range(attempts):
                if store.stopped() and not recovering:
                    return False
                try:
                    if isinstance(remote, Rclone):
                        remote.deadline = time.monotonic() + remote.timeout
                    print(f"Delivery {row['name']}: attempt {attempt + 1}/{attempts}", flush=True)
                    transfer(store, remote, row)
                    print(f"Verified {row['name']}; local cleanup permitted={bool(row['managed'])}", flush=True)
                    break
                except Exception as error:
                    print(f"Delivery error for {row['name']}: {error}", flush=True)
                    if attempt + 1 == attempts:
                        store.stop(f"Delivery failed after {attempts} attempts: {row['name']}: {error}")
                        return False
                    time.sleep(delay)
                finally:
                    if isinstance(remote, Rclone):
                        remote.deadline = None
        finally:
            lock.close()
    return True


def import_catalog(store, remote):
    inventory = remote.inventory()
    covered_archives = set()
    receipts = []
    for name in sorted(inventory):
        if not name.startswith('catalog/') or not name.endswith('.json'):
            continue
        data = remote.read(name)
        if inventory[name] != dict(size=len(data), md5=digest(data)):
            raise ValueError('Catalog checksum mismatch: ' + name)
        record = json.loads(data)
        if 'entries' in record:
            archive = record['archive']
            if inventory.get(archive) != dict(size=record['size'], md5=record['md5']):
                continue
            covered_archives.add(archive)
            for entry in record['entries']:
                receipts.append(dict(entry, archive=archive, archive_size=record['size'], archive_md5=record['md5']))
        else:
            if record['object'] != 'objects/' + record['name']:
                raise ValueError('Invalid object reference')
            if inventory.get(record['object']) != dict(size=record['size'], md5=record['md5']):
                raise ValueError('Missing or changed object: ' + record['object'])
            receipts.append(record)
    unknown = [name for name in inventory if name.startswith('orblib_') and name.endswith('.tar')
               and name not in covered_archives]
    if unknown:
        raise RuntimeError('Legacy archives need an explicit index operation: ' + ', '.join(unknown))
    with store.connection() as db:
        db.execute('UPDATE libraries SET receipt=NULL')
    for receipt in receipts:
        if not store.archived(receipt['name'], receipt['metadata']):
            store.receipt(receipt)
    for row in store.connection_rows():
        if not row['receipt'] and row['state'] == 'remote' and not (store.root / row['name']).exists():
            raise RuntimeError('Archived library no longer verified: ' + row['name'])


def prepare(store, remote, resume=False, attempts=3, delay=5):
    if not resume and (store.stopped() or store.pending()):
        raise RuntimeError('Unfinished storage state; use --resume')
    import_catalog(store, remote)
    with store.connection() as db:
        db.execute('DELETE FROM reservations')
    for path in sorted(store.root.glob('*.npz')):
        store.register(path.name, managed=False)
    if not deliver(store, remote, attempts, delay, recovering=True):
        raise RuntimeError('Recovery delivery failed; local files retained')
    stats = os.statvfs(store.root)
    if stats.f_bavail * stats.f_frsize < store.reserve_bytes or stats.f_favail <= 32:
        raise OSError('Insufficient checkpoint/log reserve; review verified old local copies before starting')
    for path in (store.stop_path, store.finish_path):
        if path.exists():
            path.unlink()
    fsync_directory(store.control)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.checksum = hashlib.md5()
        self.size = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.checksum.update(data)
        self.size += len(data)
        return data


def index_stream(stream, directory, expected=None, existing_dir=None):
    entries = []
    stream = HashingReader(stream)
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            validate_name(member.name)
            if not member.isfile():
                raise ValueError('Non-regular tar member')
            local = Path(existing_dir) / member.name if existing_dir is not None else None
            source = archive.extractfile(member)
            checksum = hashlib.md5()
            if local is not None and local.is_file() and not local.is_symlink():
                while chunk := source.read(1024 * 1024):
                    checksum.update(chunk)
                if local.stat().st_size != member.size or file_hash(local) != checksum.hexdigest():
                    raise ValueError('Local copy differs from tar member; index in a separate directory: ' + member.name)
                meta = metadata(local)
            else:
                if shutil.disk_usage(directory).free < member.size + 2_000_000_000:
                    raise OSError('Insufficient space for one legacy member')
                with tempfile.TemporaryFile(dir=directory) as temporary:
                    while chunk := source.read(1024 * 1024):
                        checksum.update(chunk)
                        temporary.write(chunk)
                    temporary.seek(0)
                    meta = metadata(temporary)
            entries.append(dict(name=member.name, size=member.size, md5=checksum.hexdigest(), metadata=meta))
    while stream.read(1024 * 1024):
        pass
    actual = dict(size=stream.size, md5=stream.checksum.hexdigest())
    if expected is not None and expected != actual:
        raise ValueError('Legacy archive stream size/MD5 mismatch')
    return entries


def index_archives(store, remote):
    for name, info in remote.inventory().items():
        if not name.startswith('orblib_') or not name.endswith('.tar'):
            continue
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(remote.command('cat', remote.root + '/' + name),
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                entries = index_stream(process.stdout, store.control, expected=info, existing_dir=store.root)
                if process.wait(timeout=remote.timeout):
                    raise RuntimeError('Legacy archive streaming failed: ' + name)
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        verify(remote, name, info['size'], info['md5'])
        record = dict(archive=name, size=info['size'], md5=info['md5'], entries=entries)
        destination = 'catalog/legacy_' + digest(name.encode()) + '_' + info['md5'] + '.json'
        local = store.control / Path(destination).name
        atomic_bytes(local, json.dumps(record, sort_keys=True).encode())
        publish(remote, local, destination)


def prune_verified(store, remote, apply=False):
    import_catalog(store, remote)
    for row in store.connection_rows():
        name = row['name']
        path = store.root / name
        if row['managed'] or not row['receipt'] or not path.exists():
            continue
        lock = store.lock(name)
        try:
            receipt = json.loads(row['receipt'])
            if path.is_symlink() or not compatible_metadata(metadata(path), receipt['metadata']):
                raise ValueError('Local metadata conflict: ' + name)
            verify_receipt(remote, receipt)
            print(f"{'PRUNE' if apply else 'WOULD PRUNE'} {name}", flush=True)
            if apply:
                store.register(name, managed=False)
                with store.connection() as db:
                    db.execute("UPDATE libraries SET managed=1,state='verified' WHERE name=?", (name,))
                store.remove_verified(name)
        finally:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description='Durable single-VM orbit library delivery')
    parser.add_argument('action', choices=('prepare', 'watch', 'index', 'stop', 'finish', 'check', 'prune-verified'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--remote', default='yandex:galAgama/orblib')
    parser.add_argument('--config', default=os.path.expanduser('~/.config/rclone/rclone.conf'))
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--reserve-bytes', type=int, default=2_000_000_000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--apply', action='store_true', help='Delete verified legacy local copies (prune-verified only)')
    parser.add_argument('--reason', default='Controller requested stop')
    args = parser.parse_args()
    if args.attempts < 1 or args.timeout < 1 or args.reserve_bytes < 0:
        parser.error('Invalid storage limits')
    if args.apply and args.action != 'prune-verified':
        parser.error('--apply is only valid with prune-verified')
    store = Store(args.root, args.reserve_bytes)
    remote = Rclone(args.remote, args.config, args.timeout)
    if args.action == 'stop':
        store.stop(args.reason)
        return 0
    if args.action == 'finish':
        atomic_bytes(store.finish_path, b'finish')
        return 0
    try:
        with open(store.control / 'uploader.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print('Another storage controller is active', flush=True)
                return 2
            if args.action == 'prepare':
                prepare(store, remote, args.resume, args.attempts)
            elif args.action == 'index':
                index_archives(store, remote)
            elif args.action == 'prune-verified':
                prune_verified(store, remote, args.apply)
            elif args.action == 'check':
                if store.stopped() or store.pending():
                    raise RuntimeError('Undelivered libraries or a storage stop remain')
            else:
                while deliver(store, remote, args.attempts):
                    if store.finish_path.exists() and not store.pending():
                        return 0
                    time.sleep(1)
                return 75
    except Exception as error:
        store.stop(str(error))
        print(str(error), flush=True)
        return 75
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
