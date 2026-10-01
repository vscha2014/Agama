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


def stream_md5(stream):
    checksum = hashlib.md5()
    while chunk := stream.read(1024 * 1024):
        checksum.update(chunk)
    return checksum.hexdigest()


def file_hash(path):
    with open(path, 'rb') as stream:
        return stream_md5(stream)


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


def metadata(path, deep=True):
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
        if deep and archive.testzip() is not None:
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
    def __init__(self, root, reserve_bytes=2_000_000_000, save_slots=2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.control = self.root / '.storage'
        self.control.mkdir(exist_ok=True)
        self.reserve_bytes = reserve_bytes
        self.save_slots = max(1, int(save_slots))
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

    @contextmanager
    def save_slot(self):
        # Caps how many workers may convert/compress/hash a library at the same
        # time: the 2026-09-29 stall was eight simultaneous savers on one VM.
        # No check_stop() on entry: a model whose integration already finished
        # must still be saved after a delivery STOP, exactly as claim() allows.
        # While *waiting* for a slot the stop is honoured, so a jammed queue
        # cannot hold a worker forever.
        stream = None
        while stream is None:
            for index in range(self.save_slots):
                candidate = open(self.control / f'save{index}.slot', 'a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                stream = candidate
                break
            if stream is None:
                self.check_stop()
                time.sleep(1)
        try:
            yield
        finally:
            stream.close()

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

    def register(self, name, managed=True, size=None, md5=None):
        validate_name(name)
        path = self.root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Not a regular library: ' + name)
        # A writer that hashed the bytes as it produced them supplies its own
        # size/MD5: the fsynced file is then validated by one full read
        # (hash comparison) instead of a testzip() pass plus a hash pass.
        meta = metadata(path, deep=size is None or md5 is None)
        checksum = file_hash(path)
        if size is not None and path.stat().st_size != size:
            raise ValueError(f'Written size mismatch: {name}; '
                             f'expected={size}, actual={path.stat().st_size}')
        if md5 is not None and checksum != md5:
            raise ValueError(f'Written MD5 mismatch: {name}; expected={md5}, actual={checksum}')
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


class RcloneError(RuntimeError):
    def __init__(self, operation, returncode, stderr):
        self.returncode = returncode
        super().__init__(f'rclone {operation} failed ({returncode}): {stderr[-1000:]}')


def remote_info(row):
    name = row.get('Path', row.get('Name', '<unknown>'))
    hashes = row.get('Hashes') or {}
    checksums = {value.strip().lower() for key, value in hashes.items()
                 if key.lower() == 'md5' and isinstance(value, str)}
    if len(checksums) != 1 or not re.fullmatch(r'[0-9a-f]{32}', next(iter(checksums), '')):
        raise ValueError(f'{name}: missing, invalid or conflicting MD5 in rclone metadata')
    size = row.get('Size')
    if type(size) is not int or size < 0:
        raise ValueError(f'{name}: invalid size in rclone metadata: {size!r}')
    return dict(size=size, md5=checksums.pop())


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
            raise RcloneError(args[0], result.returncode, result.stderr.decode(errors='replace'))
        return result.stdout

    def inventory(self):
        rows = json.loads(self.run('lsjson', self.root, '--recursive', '--files-only', '--hash'))
        return {r['Path']: remote_info(r) for r in rows if not r.get('IsDir', False)}

    def stat(self, name):
        try:
            row = json.loads(self.run('lsjson', self.root + '/' + name, '--stat', '--hash'))
        except RcloneError as error:
            if error.returncode in (3, 4):
                return None
            raise
        if row is None or row.get('IsDir', False):
            return None
        return remote_info(row)

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
        raise ValueError(f'Remote size/MD5 mismatch: {name}; '
                         f'expected size={size}, md5={checksum}; actual={actual!r}')


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


class HashingWriter:
    """Size + MD5 of everything written, computed in the writing pass.

    Deliberately offers no seek(): zipfile then writes data descriptors instead
    of rewinding to patch local headers, so the bytes we hash are exactly the
    bytes that reach the file.
    """

    def __init__(self, stream):
        self.stream = stream
        self.checksum = hashlib.md5()
        self.size = 0

    def write(self, data):
        view = memoryview(data)
        self.stream.write(view)
        self.checksum.update(view)
        self.size += view.nbytes
        return view.nbytes

    def tell(self):
        return self.size

    def flush(self):
        self.stream.flush()


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
        raise ValueError(f'Legacy archive stream size/MD5 mismatch: expected={expected!r}; actual={actual!r}')
    return entries


def index_archives(store, remote):
    for name, info in remote.inventory().items():
        if not name.startswith('orblib_') or not name.endswith('.tar'):
            continue
        if (type(info.get('size')) is not int or info['size'] < 0
                or not re.fullmatch(r'[0-9a-f]{32}', info.get('md5') or '')):
            raise ValueError(f'{name}: invalid expected size/MD5; refusing to read archive')
        print(f'Indexing remote archive {name}: size={info["size"]}, md5={info["md5"]}', flush=True)
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(remote.command('cat', remote.root + '/' + name),
                                       stdout=subprocess.PIPE, stderr=errors)
            try:
                entries = index_stream(process.stdout, store.control, expected=info, existing_dir=store.root)
                if process.wait(timeout=remote.timeout):
                    raise RuntimeError('Legacy archive streaming failed')
            except Exception as error:
                killed = process.poll() is None
                if killed:
                    process.kill()
                code = process.wait()
                errors.seek(0, os.SEEK_END)
                errors.seek(max(0, errors.tell() - 4000))
                stderr = errors.read().decode(errors='replace')
                raise RuntimeError(f'{name}: {error}; rclone cat exit={code}, '
                                   f'killed_by_indexer={killed}; stderr={stderr!r}') from error
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


def index_locations(work, archives):
    work, archives = Path(work).resolve(), Path(archives).resolve(strict=True)
    if not archives.is_dir():
        raise ValueError(f'Archive directory not found: {archives}')
    if work == archives or archives in work.parents or work in archives.parents:
        raise ValueError('Index work directory and archive directory must not overlap')
    return archives


def source_stamp(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def index_filename(record):
    name, checksum = record['archive'], record['md5']
    if Path(name).name != name or not name.startswith('orblib_') or not name.endswith('.tar'):
        raise ValueError(f'Invalid archive name in index: {name}')
    if not re.fullmatch(r'[0-9a-f]{32}', checksum):
        raise ValueError(f'{name}: invalid index MD5')
    return 'legacy_' + digest(name.encode()) + '_' + checksum + '.json'


def local_archive_paths(archives):
    paths = sorted(archives.glob('orblib_*.tar'))
    if not paths:
        raise ValueError(f'No orblib_*.tar archives found in {archives}')
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('Expected regular tar files, not links or directories')
    return paths


def index_local_archive(path):
    path = Path(path)
    before = source_stamp(path)
    entries = []
    member_name = '<tar header>'
    print(f'Indexing local {path.name}: {before["size"]} bytes, no extraction', flush=True)
    try:
        with path.open('rb') as stream:
            with tarfile.open(fileobj=stream, mode='r:') as archive:
                for member in archive:
                    member_name = member.name
                    validate_name(member.name)
                    if not member.isfile():
                        raise ValueError('Non-regular tar member')
                    with archive.extractfile(member) as source:
                        meta = metadata(source)
                        source.seek(0)
                        checksum = stream_md5(source)
                    entries.append(dict(name=member.name, size=member.size,
                                        md5=checksum, metadata=meta))
                    print(f'  validated member {len(entries)}: {member.name}', flush=True)
            stream.seek(0)
            checksum = stream_md5(stream)
    except Exception as error:
        raise ValueError(f'{path.name}, member {member_name}: {error}') from error
    if source_stamp(path) != before:
        raise ValueError(f'{path.name}: archive changed during indexing; no index accepted')
    return dict(archive=path.name, size=before['size'], md5=checksum,
                entries=entries, _local_stat=before)


def staged_index(store, path):
    if (store.root / 'catalog').is_symlink():
        raise ValueError('Staging catalog must not be a symlink')
    stamp = source_stamp(path)
    pattern = 'legacy_' + digest(path.name.encode()) + '_*.json'
    matches = []
    for index in sorted((store.root / 'catalog').glob(pattern)):
        record = json.loads(index.read_bytes())
        if record.get('_local_stat') != stamp:
            continue
        if (record.get('archive') != path.name or record.get('size') != stamp['size']
                or index_filename(record) != index.name or not isinstance(record.get('entries'), list)):
            raise ValueError(f'Invalid staged index: {index}')
        matches.append(record)
    if len(matches) > 1:
        raise ValueError(f'{path.name}: multiple current staged indexes; review work directory')
    return matches[0] if matches else None


def index_local_archives(store, archives):
    archives = index_locations(store.root, archives)
    catalog = store.root / 'catalog'
    if catalog.is_symlink():
        raise ValueError(f'Staging catalog must not be a symlink: {catalog}')
    catalog.mkdir(exist_ok=True)
    for path in local_archive_paths(archives):
        record = staged_index(store, path)
        if record is not None:
            print(f'Reusing staged index for unchanged local archive: {path.name}', flush=True)
            continue
        record = index_local_archive(path)
        atomic_bytes(catalog / index_filename(record), json.dumps(record, sort_keys=True).encode())
        print(f'Staged {path.name}: size={record["size"]}, md5={record["md5"]}', flush=True)
    print(f'Local indexing complete; unpublished indexes: {catalog}', flush=True)


def publish_local_indexes(store, remote, archives):
    archives = index_locations(store.root, archives)
    records = []
    for path in local_archive_paths(archives):
        record = staged_index(store, path)
        if record is None:
            raise ValueError(f'{path.name}: no current local index; run index --local-archives again')
        records.append((path, record))
    for path, record in records:
        verify(remote, path.name, record['size'], record['md5'])
        print(f'Cloud tar verified: {path.name}', flush=True)
    catalog = archives / 'catalog'
    if catalog.is_symlink():
        raise ValueError(f'Publication catalog must not be a symlink: {catalog}')
    prepared = []
    for path, record in records:
        if source_stamp(path) != record['_local_stat']:
            raise ValueError(f'{path.name}: archive changed before publication')
        public_record = {key: value for key, value in record.items() if key != '_local_stat'}
        content = json.dumps(public_record, sort_keys=True).encode()
        destination = catalog / index_filename(record)
        if destination.is_symlink() or (destination.exists() and destination.read_bytes() != content):
            raise FileExistsError(f'Conflicting local catalog file, not overwritten: {destination}')
        prepared.append((path, record, destination, content))
    catalog.mkdir(exist_ok=True)
    for path, record, destination, content in prepared:
        if source_stamp(path) != record['_local_stat']:
            raise ValueError(f'{path.name}: archive changed before publication')
        verify(remote, path.name, record['size'], record['md5'])
        if destination.is_symlink():
            raise FileExistsError(f'Conflicting local catalog link: {destination}')
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != content:
                raise FileExistsError(f'Conflicting local catalog file: {destination}')
        else:
            atomic_bytes(destination, content)
        publish(remote, destination, 'catalog/' + destination.name)
        verify(remote, path.name, record['size'], record['md5'])
        print(f'Published and verified index: {destination.name}', flush=True)
    for path, record in records:
        if source_stamp(path) != record['_local_stat']:
            raise ValueError(f'{path.name}: archive changed during publication')
    print('All local indexes published; tar files were not uploaded or changed', flush=True)


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
    parser.add_argument('action', choices=('prepare', 'watch', 'index', 'publish-indexes', 'stop', 'finish', 'check', 'prune-verified'))
    parser.add_argument('--root', help='Work directory; defaults to orblib-index-work with --local-archives')
    parser.add_argument('--local-archives', type=Path, help='Read existing uncompressed tar files locally without extraction')
    parser.add_argument('--publish', action='store_true', help='After local indexing, verify cloud tar hashes and publish only indexes')
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
    if args.local_archives is not None and args.action not in ('index', 'publish-indexes'):
        parser.error('--local-archives is only valid with index or publish-indexes')
    if args.publish and (args.action != 'index' or args.local_archives is None):
        parser.error('--publish requires index --local-archives')
    if args.action == 'publish-indexes' and args.local_archives is None:
        parser.error('publish-indexes requires --local-archives')
    if args.root is None:
        if args.local_archives is None:
            parser.error('--root is required without --local-archives')
        args.root = 'orblib-index-work'
    if args.local_archives is not None:
        args.local_archives = index_locations(args.root, args.local_archives)
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
                if args.local_archives is None:
                    index_archives(store, remote)
                else:
                    index_local_archives(store, args.local_archives)
                    if args.publish:
                        publish_local_indexes(store, remote, args.local_archives)
            elif args.action == 'publish-indexes':
                publish_local_indexes(store, remote, args.local_archives)
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
