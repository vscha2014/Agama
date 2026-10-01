# Plan — fix the 2026-09-29 stall: peak RAM in the orblib save path + guard-rails

Status: **approved, not implemented**. Written 2026-10-01 as a hand-off brief
for the session that implements it. Canonical behaviour of the harness lives in
`orblib_exp.md`; frozen scientific definitions in `../CONTRACT.md`; decisions in
`../DECISIONS.md`. When the work is done, fold the new behaviour into
`orblib_exp.md` and delete or archive this file.

## 0. How to use this document

Read first: `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`,
`orblib_exp.md`, `../CONTRACT.md`, skill `/harness-dev`.

Editable: `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`, `py/orblib_storage.py`,
`py/launch_orblib_exp.sh`, `tests/test_orblib_*.py`.
Frozen: `*_yaVM.py`, `J_factor_*.py`, `launch_docker_parallel.sh`, `src/**`,
`schwarzlib.py`, `table3.dat`.

Work through §3 → §8 in order. Line numbers are anchors from commit `ecc7079`,
not exact addresses — re-grep before editing.

## 1. Diagnosis — why it hung although 32 vCPU / 32 GB used to be enough

Evidence: the 09-29 run directory plus the kernel journal the user collected
after the reboot (`stop_orblib_error`), which the first post-mortem did not have.

1. **Memory, and a livelock rather than an OOM.** `systemd-journald: Under
   memory pressure, flushing caches` first appears at **16:26:53** — the moment
   integration ended — then journald hits its watchdog at 16:29:14, is SIGKILLed
   at 16:35:35, fails to start at 16:35:56 and 16:52:03, and pressure messages
   continue until 21:33. No `oom-killer`, no `Killed process`, no `Errno 28`.
   With `swap 0` the kernel can reclaim only *file* pages, so allocations kept
   succeeding while the box spent hours refaulting text/page-cache pages —
   exactly the monitored signature of **84 MB/s reads, ~0 writes, SSH dead**.
2. **The baseline was already at the edge, before any new code.**
   `agama.orbit(..., trajsize=1000)` runs with `separateTime=False` (the
   FutureWarning is in every docker log) and allocates the deprecated object
   format: 100000 × (1000×6 float32 = 24 KB) + 100000 × (1000 float64 = 8 KB)
   ≈ **3.2 GB per worker ≈ 25 GB for eight** — thrown away one line later by
   `matrices = matrices[:-1]`.
3. **The 09-19 run that "fit" wrote no libraries.** Its block in
   `log_galaxyschwarzschildfornax_Q1d1_..._p0.txt` (`# Start: 2026-09-19`) has
   **zero** `orblib` lines: no float64 copy, no compression, no whole-file
   hashing. It is not evidence that the 09-29 workload fits.
4. **What 09-29 added, in lockstep on 8 workers.** `_arr64` holds full
   **float64 duplicates** of the target matrices while the float32 originals are
   alive: matrix_kinem 100000 × (25 apertures × 52 B-spline amplitudes = 1300),
   matrix_dens 100000 × ~127 ⇒ ≈1.14 GB per worker, **≈9 GB for eight**, all
   within the same minute (workers start together, integration takes the same
   ~22 min). `_arr64` is never released — it stays referenced through
   `register()` and the whole Upsilon search until the function returns.
5. **Each worker then re-read its own 590 MB file twice**, inside the worker:
   `register()` → `metadata()` → `testzip()` (decompresses every member) and
   `file_hash()` (MD5) ≈ 1.2 GB of I/O × 8 into an already thrashing page cache.
   Matches the final log state: `.npz` present, **no** `[orblib] saved` for
   seven workers, and p4's `.npz` mtime 1.5 h after the others.

Contributors, ranked: discarded trajectories ≈25 GB (pre-existing, consumed all
head-room); float64 duplicates ≈9 GB (new); eight concurrent deflate + double
full-file re-reads (new); zero swap (pressure becomes livelock instead of a
visible failure); no progress watchdog (the launcher waited forever on
live-but-thrashing PIDs).

Ruled out: ENOSPC (plenty of free space), rclone/cloud (network ≈0, nothing
reached state `ready`), aperture/PCA errors, BoTorch hang.

Salvage: the seven `.npz` pass `orblib_storage.metadata()` (ZIP CRC + headers +
parameters); the eighth model has only a `.building` marker.

## 2. Decisions already taken (do not re-litigate)

- Drop the trajectory request; `trajsize` leaves the library compatibility key
  so the seven surviving libraries stay reusable.
- Host swapfile **and** per-container `docker --memory` with swap disabled
  inside the container.
- Resource sampler + fast memory-pressure trigger + 60-minute no-progress
  backstop; SIGTERM must produce a checkpoint; emergency logs to
  `galaxy_results_emergency/`.
- Default **4 workers** × 8 threads for library-building runs.

## 3. `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`

**A1. Stop allocating discarded trajectories** (≈3.2 GB/worker).
`agama.orbit(...)` at ~1388–1395: drop the `trajsize=trajsize` argument; delete
`matrices = matrices[:-1]` (~1397) and assert
`len(matrices) == len(datasets)`; update the `[orbitlib]` log line (~1398) to
state that trajectories are not recorded.

**A2. `trajsize` compatibility rule.**
- Constants at ~928–936: keep `numOrbits = 100000`, replace `trajsize = 1000`
  with `TRAJSIZE_STORED = 0` (truthful: nothing recorded) and remove `trajsize`
  from the `EVALUATION_CONTEXT` tuple — verified that **no** history file
  contains a `# storage-context` line yet, so no existing row is invalidated
  (the reader filter is at ~240–242).
- `_expected` dict at ~1322–1327: remove the `trajsize` key, so
  `Store.archived()` accepts both a legacy library (`trajsize=1000`) and a new
  one (`0`). The field does not influence
  `matrix_dens`/`matrix_kinem`/`ic`/`inttime`.
- Write `trajsize=TRAJSIZE_STORED` into new `.npz` (~1441) — the field must stay
  present because `orblib_storage.SCALARS` requires it.

**A3. Write the `.npz` without duplicating anything** (≈1.14 GB/worker).
Replace the `_arr64` dict + `numpy.savez_compressed` block (~1426–1452) with a
helper (e.g. `write_orblib_npz(stream, matrices, ic, inttime, scalars)`):
`zipfile.ZipFile(stream, 'w', ZIP_DEFLATED, allowZip64=True)`; for the two big
matrices write a float64 `.npy` header via
`numpy.lib.format.write_array_header_1_0({'descr': '<f8', 'fortran_order': False,
'shape': arr.shape})` and then row blocks
`arr[i:i+n].astype(numpy.float64).tobytes()` with the block sized ≈8 MB; `ic`,
`inttime`, `gridv` and the scalars keep `numpy.lib.format.write_array`. Keep the
existing atomic flow (`.tmp` → flush → `fsync` → `os.replace` → `register`) and
the `StorageStop`-on-failure behaviour (~1463–1467).

**A4. Hash while writing.** Wrap the output file object in the new
`HashingWriter` so size + MD5 come out of the same pass, and call
`store.register(_ol_name, size=..., md5=...)` (~1453–1454).

**A5. Cap concurrent savers.** Wrap convert + write + register in
`with store.save_slot():`, default `ORBLIB_SAVE_SLOTS=2` (env, read next to
`ORBLIB_RESERVE_BYTES` at ~216), blocking with `check_stop()` polling. Removes
the "all workers compress at once" spike. Release in the existing `finally`
(~1479–1482), ordering: save slot → build claim → storage claim.

**A6. Graceful SIGTERM/SIGINT.** After the store is created (~216–218) install
handlers that write the storage `STOP` marker with reason
`signal <name> from controller`; the existing `check_storage_stop()` /
`store.check_stop()` polls then raise `StorageStop`, which the
`except StorageStop` branch (~4333–4336) already turns into
`checkpoint_for_stop()` + exit 75 — a path the launcher understands. With no
store, checkpoint directly and exit 75. Caveat to document: a signal arriving
inside a long C call (`agama.orbit`, `agama.solveOpt`) is only acted on when
that call returns — hence the launcher's stop grace period (§5 C5a). A container
killed by the cgroup OOM gets SIGKILL and writes **no** checkpoint; what
survives: history rows already written, delivered `.npz`, `flock` released by
process death, and `.building`/`.resv` claims expiring by TTL.

**A7. Phase/RSS instrumentation.** `log_mem(tag)` from `/proc/self/statm` plus
`MemAvailable`, printed before/after integration, before/after the npz write,
after `register`, and after the Upsilon search.

## 4. `py/orblib_storage.py`

- `metadata(path, deep=True)` (~87–115): `deep=False` skips `testzip()`; all
  current call sites keep the default.
- `register(name, managed=True, size=None, md5=None)` (~231–249): when a
  verified in-stream identity is supplied, compare it with the fsynced file's
  size/MD5 and use `metadata(..., deep=False)` — one pass instead of three.
  `prepare` (~480–481), `prune_verified` (~734–739) and indexing keep deep
  validation.
- `Store.save_slot()` contextmanager: N `flock` slot files under `.storage/`,
  reusing the `lock()` pattern (~191–199); honours `check_stop()`.
- `HashingWriter` next to `HashingReader` (~493–503).
- CLI and queue state machine unchanged.

## 5. `py/launch_orblib_exp.sh`

**C1.** Worker default for library-building runs: `_NPROC_AUTO = N_VCPU / 8`
(⇒ 4 on 32 vCPU, 8 threads each) at ~30–34; `--nproc=` still overrides.

**C2. New step 0a — host swap.** If `swapon --show` is empty and
`ORBLIB_SWAPFILE` (default `16G`, `0` disables): `/swapfile` via fallocate →
chmod 600 → mkswap → swapon, `vm.swappiness=10`, idempotent, sudo. Purpose is
explicitly **host** protection (sshd, journald, orchestrator, rclone), not
extending a bloated worker's life. Failure ⇒ warning, continue, never `die`.

**C3. Swapless per-container limit** in `docker run` (~469–502):
`--memory=${LIM} --memory-swap=${LIM}`. In docker `--memory-swap` is the
*combined* RAM+swap ceiling, so equality means no container swap and an
OOM-kill at the limit; leaving it unset would default to `2 × --memory` and
reproduce the slow crawl. `LIM = (MemTotal − 4 GB)/N_PROC`, overridable via
`ORBLIB_MEM_LIMIT`. A killed container exits 137 and lands in the existing
per-container error path (partial-history merge + `notify`, ~525–534).

**C3a. Verify swap limiting works** before trusting it: `docker info` for
`No swap limit support`, and on cgroup v2 the presence of `memory.swap.max`.
If swap accounting is unavailable (cgroup v1 without `swapaccount=1`), docker
ignores `--memory-swap` silently — then log a prominent warning and either skip
the swapfile or set `vm.swappiness=1`, and lower `ORBLIB_STALL_TIMEOUT`.

**C4. Resource sampler** (background, 60 s) →
`monitor_${RUN_TAG}_${TIMESTAMP}.log`: `free -m`, `/proc/pressure/memory`,
`docker stats --no-stream`, `df -h` for the orblib filesystem. Started next to
the uploader (~689–690), stopped in `on_exit` (~238–248), uploaded with the
other logs.

**C5. Two-level stall detection** in the step-3 wait loop (~710–720).
- *Fast trigger (minutes):* `MemAvailable` below `ORBLIB_MIN_AVAIL_MB`
  (default 2048) **or** `some avg60` in `/proc/pressure/memory` above
  `ORBLIB_PSI_LIMIT` (default 20 %) for `ORBLIB_PSI_SAMPLES` (default 3)
  consecutive samples. This is what the 09-29 signature (first pressure message
  16:26:53, SSH gone by 16:29) would have caught in minutes.
- *Slow backstop:* newest mtime across `dockerlog_*`, `log_*_p*.txt`,
  `out_*_p*.txt`, `orblib/*.npz*` not advancing for `ORBLIB_STALL_TIMEOUT`
  (default **3600 s**) — covers "alive but not progressing" without pressure.

**C5a. Graceful path, fixed order:** diagnostics dump (free, pressure, top-10 by
RSS, `docker stats`) → `storage_cli stop --reason "<trigger>"` →
`docker stop -t ${ORBLIB_STOP_GRACE:-1800}` per container (SIGTERM, then
SIGKILL) → bounded wait on worker PIDs with `docker kill` after grace + margin →
`emergency_upload` → `notify urgent` → `schedule_shutdown`. Upload must precede
shutdown; `shutdown -h +1` adds another minute of margin.

**C6. Emergency log delivery** `emergency_upload()`: rclone copy of `$LOGFILE`,
`dockerlog_*_${TIMESTAMP}.log`, `log_*_${EXP_ID}_p*.txt`, `monitor_*.log`,
`out_*` and `orblib/.storage/STOP` into
`${RCLONE_REMOTE}:${REMOTE_DIR}/galaxy_results_emergency/${RUN_TAG}_${TIMESTAMP}/`.
It deliberately **bypasses** the `storage_stopped && return 0` guard in
`upload_to_yadisk` (~337–350) that makes uploads a silent no-op during a STOP —
that guard is why the 09-29 logs had to be copied off the VM by hand. Call it
from both stall triggers, from the `storage_stopped` branch of step 3
(~742–747) and from `on_exit`.

**C7. Shutdown accountability.** The watchdog adds no shutdown logic of its own:
it calls the existing `schedule_shutdown`, which already runs
`sudo shutdown -h +N` (~234–235), so no new privilege is needed — passwordless
sudo is confirmed (the aborted 09-29 launch powered the VM off a minute after
`die`). Replace the `|| true` that hides a failed shutdown: check the exit
status, log an explicit line and send an `urgent` notification when shutdown
could not be scheduled, so the VM cannot silently keep burning quota.

## 6. Tests (`tests/test_orblib_storage.py`, `tests/test_orblib_q1.py`)

Both suites extract functions via `ast` — the calculation script must not be
imported. Keep arrays tiny; mock docker/rclone/shutdown.

- npz writer round-trip: `numpy.load` returns exactly
  `float32_input.astype(float64)`; member dtype `<f8`; `metadata()` accepts the
  file; in-stream MD5 equals `file_hash`; arrays and member names match what
  `savez_compressed` produced before.
- memory guard: `tracemalloc` peak for a 200k-element input stays within a few
  blocks (no whole-array float64 copy).
- `metadata(deep=False)` + size/MD5 still rejects a truncated or byte-flipped
  file; `register(size=…, md5=…)` rejects a mismatching identity.
- `save_slot()`: with 2 slots the third acquirer blocks, releases on exception,
  honours `STOP`.
- compatibility: an `expected` dict without `trajsize` matches both a legacy
  (`trajsize=1000`) and a new (`0`) library; a differing `gridv_md5` still
  conflicts.
- signal handler: SIGTERM writes `STOP` with the expected reason, the evaluation
  loop raises `StorageStop` and a checkpoint is written.
- launcher mocked dry run (`docker`/`rclone`/`sudo`/`curl` stubs on `PATH`,
  `--no-shutdown`): (a) `docker run` carries `--memory` and an equal
  `--memory-swap`; (b) the fast pressure trigger fires from a faked
  `MemAvailable`/PSI source; (c) the slow watchdog fires with
  `ORBLIB_STALL_TIMEOUT=5`; (d) `docker stop` is called and the emergency
  directory is populated **before** the shutdown call; (e) a failing
  `sudo shutdown` stub yields the explicit log line and the notification.

## 7. Verification

```bash
cd tests && ../.venv-ai/bin/python -m pytest -q test_orblib_q1.py \
    test_orblib_storage.py --rootdir=. --import-mode=importlib -p no:cacheprovider
cd .. && python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py py/orblib_storage.py
bash -n py/launch_orblib_exp.sh && git diff --check && git status --short
git diff --stat   # must not list *_yaVM.py, J_factor_*, launch_docker_parallel.sh, src/**
```

Optional full-scale local check (ask the user first — needs ~2 GB RAM and
~1.2 GB temp disk): old vs new writer on synthetic float32 arrays of the real
shapes (100000×1300 and 100000×127); expected peak ≈1.2 GB → <100 MB with
identical reloaded values.

Expected per-worker peak afterwards: ≈1.5–2 GB (float32 matrices + solver)
instead of ≈4–5 GB ⇒ ≈8 GB for four workers on 31 GiB, with a host swapfile, a
swapless container limit and two stall triggers as the safety net.

## 8. Documentation and bookkeeping (end of task)

- `orblib_exp.md`: new save path, save slots, signal handling, sampler, both
  stall triggers, container memory/swap policy, emergency upload, worker
  default, `trajsize` rule. Then retire this plan file.
- `../DECISIONS.md`: Q21 answered (trajectories dropped, `trajsize` out of the
  key) plus the operational decisions (host-only swap, swapless container, fast
  trigger + 60-minute backstop, 4 workers); delete Q21 from
  `../questions_for_pi.md`.
- `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`: append the journal
  evidence, turn §4 from "suspected" into "confirmed", record the seven verified
  libraries; update the run row in `results/REGISTRY.md`.
- Rewrite `../STATUS.md` (≤60 lines). Nothing new in `AGENTS.md`.

## 9. Recommended run order afterwards (no run without an explicit go-ahead)

1. `--reuse-orblib` pass over the seven surviving libraries with 4 workers to
   obtain their `Upsilon`/`penalty` without re-integration; this also validates
   that a stored library still reproduces a penalty after A1/A2.
2. Only then a fresh 4-worker `--Q1` search on the widened bounds.

## 10. Risks

- Dropping `trajsize` changes the `agama.orbit` call. Target matrices are
  accumulated by `RuntimeFncTarget` during integration and are independent of
  trajectory recording, and IC sampling is untouched — but this must be shown,
  not assumed: §9.1 recomputes a penalty from a stored library, and one control
  model is re-integrated and compared.
- Hand-written `.npy` headers must match numpy exactly → covered by round-trip
  tests.
- The swapless container limit converts thrashing into a killed container: a
  legitimate spike now loses one model (SIGKILL, no checkpoint) instead of
  finishing slowly. Mitigated by the 4-worker default and a generous limit
  (~6.7 GB vs ~2 GB expected peak).
- If the kernel lacks swap accounting, `--memory-swap` is ignored silently —
  handled explicitly by C3a rather than assumed away.
- The swapfile needs sudo and 16 GB of free disk; skipped with a warning if
  unavailable.
- `metadata(deep=False)` on the hot path trades a full CRC re-read for a
  size+MD5 comparison of the freshly fsynced file; deep validation stays on
  every path that inspects files this process did not write.
