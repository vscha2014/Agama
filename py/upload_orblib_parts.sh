#!/bin/bash
set -euo pipefail

WORK_DIR="$(cd "$(dirname "$0")" && pwd)"
ORBLIB_DIR="${ORBLIB_DIR:-${WORK_DIR}/orblib}"
REMOTE_ROOT="${REMOTE_ROOT:-yandex:galAgama}"
RCLONE_CONFIG="${RCLONE_CONFIG:-${HOME}/.config/rclone/rclone.conf}"
PART_SIZE_GB="${PART_SIZE_GB:-40}"
UPLOAD_ATTEMPTS="${UPLOAD_ATTEMPTS:-3}"
SNAPSHOT=""
HOST_NAME="$(hostname)"
RUN_TIMESTAMP=""
APPLY=0

usage() {
    cat <<'EOF'
Usage:
  upload_orblib_parts.sh --snapshot=PATH [options]

Options:
  --snapshot=PATH       snapshot_before from the failed run (required)
  --orblib-dir=PATH     directory containing .npz files (default: ./orblib)
  --remote-root=REMOTE  rclone destination directory (default: yandex:galAgama)
  --rclone-config=PATH  rclone config path
  --host=NAME           host component of shard names (default: hostname)
  --timestamp=VALUE     YYYYmmdd_HHMMSS (default: inferred from snapshot)
  --part-size-gb=N      decimal GB per part (default: 40)
  --attempts=N          upload attempts per part (default: 3)
  --apply               upload; without this flag only print the plan
  -h, --help            show this help

The script never deletes orbit libraries or the snapshot and never creates a
local tar file. Successfully uploaded parts are skipped on a repeated run when
their remote size matches the planned size.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --snapshot=*)      SNAPSHOT="${arg#*=}" ;;
        --orblib-dir=*)    ORBLIB_DIR="${arg#*=}" ;;
        --remote-root=*)   REMOTE_ROOT="${arg#*=}" ;;
        --rclone-config=*) RCLONE_CONFIG="${arg#*=}" ;;
        --host=*)          HOST_NAME="${arg#*=}" ;;
        --timestamp=*)     RUN_TIMESTAMP="${arg#*=}" ;;
        --part-size-gb=*)  PART_SIZE_GB="${arg#*=}" ;;
        --attempts=*)      UPLOAD_ATTEMPTS="${arg#*=}" ;;
        --apply)           APPLY=1 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$SNAPSHOT" ] || { echo "ERROR: --snapshot is required" >&2; exit 2; }
SNAPSHOT="$(readlink -f "$SNAPSHOT")"
ORBLIB_DIR="$(readlink -f "$ORBLIB_DIR")"
[ -f "$SNAPSHOT" ] || { echo "ERROR: snapshot not found: $SNAPSHOT" >&2; exit 1; }
[ -d "$ORBLIB_DIR" ] || { echo "ERROR: orblib directory not found: $ORBLIB_DIR" >&2; exit 1; }
[[ "$PART_SIZE_GB" =~ ^[0-9]+$ ]] && [ "$PART_SIZE_GB" -ge 1 ] \
    || { echo "ERROR: --part-size-gb must be an integer >= 1" >&2; exit 2; }
[[ "$UPLOAD_ATTEMPTS" =~ ^[0-9]+$ ]] && [ "$UPLOAD_ATTEMPTS" -ge 1 ] \
    || { echo "ERROR: --attempts must be an integer >= 1" >&2; exit 2; }
[[ "$HOST_NAME" =~ ^[A-Za-z0-9._+-]+$ ]] \
    || { echo "ERROR: invalid --host: $HOST_NAME" >&2; exit 2; }

if [ -z "$RUN_TIMESTAMP" ]; then
    if [[ "$(basename "$SNAPSHOT")" =~ \.snapshot_before_([0-9]{8}_[0-9]{6})\.txt$ ]]; then
        RUN_TIMESTAMP="${BASH_REMATCH[1]}"
    else
        echo "ERROR: cannot infer timestamp; pass --timestamp=YYYYmmdd_HHMMSS" >&2
        exit 2
    fi
fi
[[ "$RUN_TIMESTAMP" =~ ^[0-9]{8}_[0-9]{6}$ ]] \
    || { echo "ERROR: invalid --timestamp: $RUN_TIMESTAMP" >&2; exit 2; }

if [ "$APPLY" -eq 1 ]; then
    command -v rclone >/dev/null || { echo "ERROR: rclone not found" >&2; exit 1; }
    [ -f "$RCLONE_CONFIG" ] || { echo "ERROR: rclone config not found: $RCLONE_CONFIG" >&2; exit 1; }
fi

TMP_DIR=$(mktemp -d "${WORK_DIR}/.orblib_parts_XXXXXX")
trap 'rm -rf "$TMP_DIR"' EXIT
BASELINE="${TMP_DIR}/baseline.txt"
CURRENT="${TMP_DIR}/current.txt"
NEW_LIST="${TMP_DIR}/new.txt"
sort -u "$SNAPSHOT" > "$BASELINE"
find "$ORBLIB_DIR" -maxdepth 1 -type f -name '*.npz' -printf '%f\n' | sort -u > "$CURRENT"
comm -13 "$BASELINE" "$CURRENT" > "$NEW_LIST"

N_NEW=$(wc -l < "$NEW_LIST")
[ "$N_NEW" -gt 0 ] || { echo "No new .npz files relative to $SNAPSHOT"; exit 0; }

KEY=""
while IFS= read -r name; do
    if ! [[ "$name" =~ ^orblib_(i[-+0-9.]+_d[01]_nb[0-9]+_ser[0-9]+)_geom[0-9A-Fa-f]+_[0-9A-Fa-f]{10}\.npz$ ]]; then
        echo "ERROR: unexpected orblib filename: $name" >&2
        exit 1
    fi
    file_key="${BASH_REMATCH[1]}"
    if [ -z "$KEY" ]; then
        KEY="$file_key"
    elif [ "$KEY" != "$file_key" ]; then
        echo "ERROR: mixed experiment keys: $KEY and $file_key" >&2
        exit 1
    fi
    path="${ORBLIB_DIR}/${name}"
    [ -f "$path" ] && [ ! -L "$path" ] && [ -s "$path" ] \
        || { echo "ERROR: invalid orbit library: $path" >&2; exit 1; }
done < "$NEW_LIST"

PART_LIMIT=$((PART_SIZE_GB * 1000 * 1000 * 1000))
declare -a PART_LISTS PART_SIZES PART_NAMES
part_index=0
part_size=1024
part_count=0
part_list=$(printf '%s/part_%03d.list' "$TMP_DIR" "$part_index")
: > "$part_list"

finish_part() {
    [ "$part_count" -gt 0 ] || return 0
    PART_LISTS+=("$part_list")
    PART_SIZES+=("$part_size")
    PART_NAMES+=("$(printf 'orblib_%s__%s_%s_part%03d.tar' \
        "$KEY" "$HOST_NAME" "$RUN_TIMESTAMP" "$part_index")")
}

while IFS= read -r name; do
    size=$(stat -c '%s' -- "${ORBLIB_DIR}/${name}")
    padded=$(( (size + 511) / 512 * 512 ))
    member_size=$((512 + padded))
    if [ $((1024 + member_size)) -gt "$PART_LIMIT" ]; then
        echo "ERROR: one member exceeds the part limit: $name" >&2
        exit 1
    fi
    if [ "$part_count" -gt 0 ] && [ $((part_size + member_size)) -gt "$PART_LIMIT" ]; then
        finish_part
        part_index=$((part_index + 1))
        part_size=1024
        part_count=0
        part_list=$(printf '%s/part_%03d.list' "$TMP_DIR" "$part_index")
        : > "$part_list"
    fi
    printf '%s\n' "$name" >> "$part_list"
    part_size=$((part_size + member_size))
    part_count=$((part_count + 1))
done < "$NEW_LIST"
finish_part

printf 'New libraries: %d\n' "$N_NEW"
printf 'Experiment key: %s\n' "$KEY"
printf 'Snapshot: %s\n' "$SNAPSHOT"
printf 'Part limit: %s bytes (%s GB)\n' "$PART_LIMIT" "$PART_SIZE_GB"
printf 'Parts: %d\n' "${#PART_LISTS[@]}"
for i in "${!PART_LISTS[@]}"; do
    printf '  %s  files=%d  size=%s bytes (%s)\n' \
        "${PART_NAMES[$i]}" \
        "$(wc -l < "${PART_LISTS[$i]}")" \
        "${PART_SIZES[$i]}" \
        "$(numfmt --to=iec "${PART_SIZES[$i]}")"
done

if [ "$APPLY" -ne 1 ]; then
    echo "Dry run: nothing uploaded. Add --apply to start upload."
    exit 0
fi

remote_size() {
    rclone lsf "$1" --format s --config "$RCLONE_CONFIG" 2>/dev/null | head -n 1
}

remote_md5() {
    rclone md5sum "$1" --config "$RCLONE_CONFIG" 2>/dev/null \
        | awk 'NR == 1 {print tolower($1)}'
}

remove_partial() {
    rclone deletefile "$1" --yandex-hard-delete --config "$RCLONE_CONFIG" \
        >/dev/null 2>&1 || true
}

upload_part() {
    local list_file="$1" expected_size="$2" part_name="$3"
    local final_remote="${REMOTE_ROOT}/${part_name}"
    local partial_remote="${REMOTE_ROOT}/.uploading_${part_name}.partial"
    local existing_size
    existing_size=$(remote_size "$final_remote" || true)
    if [ "$existing_size" = "$expected_size" ]; then
        echo "SKIP existing part with matching size: $part_name"
        return 0
    fi
    if [ -n "$existing_size" ]; then
        echo "ERROR: remote part exists with wrong size: $part_name" >&2
        return 1
    fi

    local attempt fifo md5_file md5_pid pipeline_rc md5_rc local_hash remote_hash uploaded_size
    fifo="${TMP_DIR}/${part_name}.fifo"
    md5_file="${TMP_DIR}/${part_name}.md5"
    for ((attempt = 1; attempt <= UPLOAD_ATTEMPTS; attempt++)); do
        echo "UPLOAD $part_name attempt ${attempt}/${UPLOAD_ATTEMPTS}"
        remove_partial "$partial_remote"
        rm -f "$fifo" "$md5_file"
        mkfifo "$fifo"
        set +e
        md5sum < "$fifo" > "$md5_file" &
        md5_pid=$!
        tar --format=ustar --blocking-factor=1 --verbatim-files-from \
                -cf - -C "$ORBLIB_DIR" -T "$list_file" \
            | tee "$fifo" \
            | rclone rcat "$partial_remote" --size "$expected_size" \
                --config "$RCLONE_CONFIG" --stats-one-line
        pipeline_rc=$?
        wait "$md5_pid"
        md5_rc=$?
        set -e
        rm -f "$fifo"
        if [ "$pipeline_rc" -ne 0 ] || [ "$md5_rc" -ne 0 ]; then
            echo "  tar/rcat failed: pipeline=$pipeline_rc md5=$md5_rc" >&2
            remove_partial "$partial_remote"
            continue
        fi
        local_hash=$(awk 'NR == 1 {print tolower($1)}' "$md5_file")
        uploaded_size=$(remote_size "$partial_remote" || true)
        remote_hash=$(remote_md5 "$partial_remote" || true)
        if [ "$uploaded_size" != "$expected_size" ] || [ "$remote_hash" != "$local_hash" ]; then
            echo "  partial verification failed: size=$uploaded_size/$expected_size md5=$remote_hash/$local_hash" >&2
            remove_partial "$partial_remote"
            continue
        fi
        if ! rclone moveto "$partial_remote" "$final_remote" \
                --config "$RCLONE_CONFIG"; then
            echo "  moveto failed" >&2
            remove_partial "$partial_remote"
            continue
        fi
        uploaded_size=$(remote_size "$final_remote" || true)
        remote_hash=$(remote_md5 "$final_remote" || true)
        if [ "$uploaded_size" = "$expected_size" ] && [ "$remote_hash" = "$local_hash" ]; then
            echo "OK $part_name size=$expected_size md5=$local_hash"
            return 0
        fi
        echo "ERROR: final verification failed for $part_name" >&2
        return 1
    done
    echo "ERROR: upload attempts exhausted for $part_name" >&2
    return 1
}

for i in "${!PART_LISTS[@]}"; do
    if ! upload_part "${PART_LISTS[$i]}" "${PART_SIZES[$i]}" "${PART_NAMES[$i]}"; then
        echo "ERROR: multipart upload stopped at ${PART_NAMES[$i]}" >&2
        exit 1
    fi
done

echo "All multipart shards uploaded successfully. Local .npz files were not removed."
