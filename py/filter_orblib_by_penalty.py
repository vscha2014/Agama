import argparse
import glob
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile
import tempfile


ARCHIVE_NAME_RE = re.compile(
    r"^orblib_i(?P<incl>[-+]?\d+(?:\.\d+)?)_d(?P<double>[01])_nb(?P<n_bin>\d+)_ser(?P<ser_id>\d+)__(?P<origin>.+?)(?:\.tar)?$"
)
TIMESTAMP_RE = re.compile(r"_\d{8}_\d{6}(?:_part\d{3})?$")


def parse_archive_name(path):
    match = ARCHIVE_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            "имя архива должно иметь вид "
            "orblib_i90.0_d0_nb250_ser0__HOST[_YYYYmmdd_HHMMSS][.tar]"
        )
    result = match.groupdict()
    result["incl"] = float(result["incl"])
    result["double"] = int(result["double"])
    result["n_bin"] = int(result["n_bin"])
    result["ser_id"] = int(result["ser_id"])
    result["origin"] = TIMESTAMP_RE.sub("", result["origin"])
    return result


def find_history_files(archive, config, gh_id):
    prefix = (
        f"out_{glob.escape(config['origin'])}_d{config['double']}_"
        f"nb{config['n_bin']}_gh{gh_id}_ser{config['ser_id']}"
    )
    names = glob.glob(str(archive.parent / f"{prefix}.txt"))
    names.extend(glob.glob(str(archive.parent / f"{prefix}_p*.txt")))
    return sorted({Path(name) for name in names})


def model_key(values):
    text = "_".join(f"{value:.6g}" for value in values)
    return hashlib.md5(text.encode()).hexdigest()[:10]


def load_penalties(history_files, incl):
    penalties = {}
    rows = 0
    for path in history_files:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                fields = line.split()
                if not fields or fields[0].startswith("#") or len(fields) < 7:
                    continue
                try:
                    values = [float(field) for field in fields[:7]]
                except ValueError:
                    continue
                if abs(values[0] - incl) > 0.01 or not math.isfinite(values[6]):
                    continue
                penalties.setdefault(model_key(values[1:5]), []).append(values[6])
                rows += 1
    return {key: min(found) for key, found in penalties.items()}, rows


def archive_model_re(config):
    incl = re.escape(f"{config['incl']:.1f}")
    return re.compile(
        rf"^orblib_i{incl}_d{config['double']}_nb{config['n_bin']}_"
        rf"ser{config['ser_id']}_geom[0-9a-fA-F]+_(?P<key>[0-9a-fA-F]{{10}})\.npz$"
    )


def classify_members(members, config, penalties, cutoff):
    pattern = archive_model_re(config)
    remove = []
    keep = []
    unmatched = []
    recognized = 0
    for member in members:
        match = pattern.match(Path(member.name).name)
        if match is None:
            keep.append(member)
            if member.name.endswith(".npz"):
                unmatched.append(member.name)
            continue
        recognized += 1
        key = match.group("key").lower()
        penalty = penalties.get(key)
        if penalty is None:
            keep.append(member)
            unmatched.append(member.name)
        elif penalty > cutoff:
            remove.append((member, penalty))
        else:
            keep.append(member)
    return keep, remove, unmatched, recognized


def required_space(members):
    return 10240 + sum(512 + ((member.size + 511) // 512) * 512 for member in members)


def rewrite_archive(path, source, keep):
    free = shutil.disk_usage(path.parent).free
    needed = required_space(keep)
    if free < needed:
        raise OSError(
            f"недостаточно свободного места: нужно примерно {needed / 1e9:.2f} GB, "
            f"доступно {free / 1e9:.2f} GB"
        )
    source_mode = stat.S_IMODE(path.stat().st_mode)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    expected = [(member.name, member.size, member.type) for member in keep]
    try:
        with tarfile.open(temp_path, "w:") as destination:
            for member in keep:
                payload = source.extractfile(member) if member.isreg() else None
                try:
                    destination.addfile(member, payload)
                finally:
                    if payload is not None:
                        payload.close()
        with tarfile.open(temp_path, "r:") as check:
            actual = [(member.name, member.size, member.type) for member in check.getmembers()]
        if actual != expected:
            raise RuntimeError("проверка нового архива не пройдена: состав или размеры изменились")
        os.chmod(temp_path, source_mode)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Удаляет из tar-шарда орбитные модели с penalty выше cutoff. "
            "История out_* ищется рядом с архивом; без --apply архив не изменяется."
        )
    )
    parser.add_argument("archive", type=Path, help="tar-шард orblib_i...__HOST[...][.tar]")
    parser.add_argument("--cutoff", type=float, default=27.3, help="порог penalty (по умолчанию 27.3)")
    parser.add_argument("--gh-id", type=int, default=0, help="GH-реализация истории (по умолчанию 0)")
    parser.add_argument("--apply", action="store_true", help="атомарно заменить исходный архив")
    return parser


def main():
    args = build_parser().parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"архив не найден: {archive}")
    if not math.isfinite(args.cutoff):
        raise ValueError("cutoff должен быть конечным числом")

    config = parse_archive_name(archive)
    history_files = find_history_files(archive, config, args.gh_id)
    if not history_files:
        raise FileNotFoundError(
            f"рядом с архивом не найдены out-файлы для host={config['origin']}, "
            f"d={config['double']}, nb={config['n_bin']}, gh={args.gh_id}, "
            f"ser={config['ser_id']}"
        )
    penalties, rows = load_penalties(history_files, config["incl"])
    if not penalties:
        raise ValueError(f"в найденных out-файлах нет данных для incl={config['incl']:.1f}")

    with tarfile.open(archive, "r:") as source:
        members = source.getmembers()
        keep, remove, unmatched, recognized = classify_members(
            members, config, penalties, args.cutoff
        )
        print(f"Архив: {archive}")
        print(f"История: {len(history_files)} файлов, {rows} строк, {len(penalties)} моделей")
        for path in history_files:
            print(f"  {path.name}")
        print(f"Распознано моделей в архиве: {recognized}")
        if recognized == 0:
            raise ValueError("в архиве не найдено ни одной модели ожидаемого формата")
        print(f"Удалить (min penalty > {args.cutoff:g}): {len(remove)}")
        print(f"Сохранить: {len(keep)} элементов")
        print(f"Без соответствующего penalty/неизвестный формат: {len(unmatched)}")
        for member, penalty in remove:
            print(f"  DELETE {member.name}  penalty={penalty:.15g}")
        for name in unmatched:
            print(f"  KEEP?  {name}  penalty не найден")

        if not args.apply:
            print("Проверочный режим: архив не изменён. Для применения добавьте --apply.")
            return 0
        if not remove:
            print("Нет моделей для удаления: архив не изменён.")
            return 0
        rewrite_archive(archive, source, keep)

    print(f"Исходный архив атомарно заменён; удалено моделей: {len(remove)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
