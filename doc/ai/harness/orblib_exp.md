# Runbook — experimental harness `orblib_exp`

Files: `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`, `py/launch_orblib_exp.sh`,
`py/orblib_storage.py`, single-model check `py/run_single_model.py` +
`py/launch_single_model.sh` (§3b), tests in `tests/test_orblib_q1.py`,
`tests/test_orblib_storage.py`, `tests/test_single_model.py`. Production scripts are **not** part of this and
must not be touched (see `../CONTRACT.md`, `production.md`).

## 1. Experiment identity

- `EXP_ID = d{0|1}_nb{N}_gh{G}_ser{S}`; fixed-Q writes `Q1d1_nb250_gh0_ser0`.
  `EXP_ID` is embedded in `hostname_proc`, so every derived file is isolated per
  experiment. Penalties from different `d`/`nb` are never pooled.
- `out_{tag}.txt` — history (one line per model, read by parallel workers);
  `log_{tag}.txt` — human-readable log, read by nobody.
- Variations: `--no-double`, `--n-bin`, `--gh-id`, `--ser-id` (>0 raises
  `NotImplementedError` on purpose), `--save-orblib`, `--reuse-orblib`,
  `--stream-orblib`, `--Q1`.
- `gh` is the halo inner slope (a search parameter); `gh_id` is the observational
  GH perturbation id. Orbit matrices depend on `(Q, gh, rh, rho0, incl, double,
  n_bin, ser_id)` but **not** on `gh_id`.
- Datacube grid is built from the actual aperture vertices (not from sectors 1/3),
  and `GEOM_HASH` enters the orbit-library cache key.

## 2. Fixed-Q (`--Q1`) mode

- `./launch_orblib_exp.sh --Q1 --incl=90.0` (add `--nproc=8`, `--no-shutdown`).
  `--Q1 --no-double` is rejected. `incl` is fixed per launch, never optimised.
- Reads both Q modes' compatible history (all hosts, per-process and merged),
  then filters rows to `Q=1`; free-Q readers accept all Q. Writers, logs and
  checkpoints stay separate. Never deduplicate rows.
- External prior points are projected to `Q=1` and their penalty **recomputed**;
  old penalties are not training observations.
- PCA weights everywhere: `exp(-(penalty − penalty.min()) / 0.1)` — prevents
  all-zero underflow on cold-start penalties above 100. The weighted std is
  floored at `WeightedScaler.STD_FLOOR_FRAC` (0.05) × the bound width in the
  transformed space (log10 `rh`, `rho0`), so best rows sharing a bound value no
  longer blow up the PCA box (2026-10-05, harness only).
- Q1 seeding before the first PCA: per source (compatible `out`, canonical
  `4Ups`, opt-in PA46.8) at most 24 nearest-`Q` candidates, ranked by old penalty
  within their own source, ordered with rank-biased per-process RNG; at most 10
  evaluations per process; already-present `Q=1` points and repeated projections
  are skipped before computing. Shared history is refreshed every 4 candidates
  and at the end; a worker may stop early once shared history gained 10 valid
  points. Having 10 points with penalty < 10 skips the stage, but that threshold
  is **never required**. A deficit is filled by one bounded LHS batch; persistent
  failure is an explicit error. Resume from a checkpoint skips this stage.
- Q1 checkpoints store physical parameters and reproject into the PCA basis
  rebuilt at resume.
- TuRBO trust region (`_tr_bounds`): `[centre ± L/2·range]` clamped to the PCA
  box (upper-bound bug fixed 2026-10-05, also in production; DECISIONS).
  Histories before that fix were searched with a TR extending above the box.
- The J-factor post-processing does not yet parse `Q1d1_*` filenames.

## 3. Streaming orbit-library storage (single VM)

- `--stream-orblib` supersedes the legacy tar download/snapshot functions still
  present in the launcher; never run both at once. New libraries are individual
  objects under `galAgama/orblib/objects/` with receipts under `catalog/`.
- `py/orblib_storage.py` is stdlib-only; SQLite state, locks, stop/finish markers
  and receipts live in the gitignored `py/orblib/.storage/`. One controller per
  workspace; multi-VM publication is not implemented.
- Old tars need a one-time index. Prefer the local-archive procedure below when
  the tar files are already downloaded. The existing streaming command
  `python3 orblib_storage.py index --root ./orblib --timeout 7200` still reads
  tar contents from the cloud; never run it during development without an
  explicit user request.
- A local `.npz` is deleted only after the remote object and manifest pass
  size + MD5 checks. Pre-existing untracked libraries are uploaded but not
  auto-deleted; `prune-verified --root ./orblib` lists, `--apply` deletes only
  freshly re-verified copies (needs explicit permission).
- Defaults: `ORBLIB_UPLOAD_ATTEMPTS=3`, `ORBLIB_FILE_TIMEOUT=900` s per whole
  file attempt, `ORBLIB_RESERVE_BYTES=2e9` for checkpoints/logs/metadata; workers
  additionally reserve the estimated uncompressed size plus ZIP overhead from a
  shared budget.
- On exhausted delivery attempts: STOP — no new models, running models finish and
  checkpoint locally without mandatory cloud sync, the launcher waits and then
  shuts down even with undelivered files (`--no-shutdown` suppresses only the
  shutdown). Never delete a VM or disk that still holds undelivered state.
- `--resume` drains the queue first; persistent failure powers off without
  starting workers. Old free-Q checkpoints without physical parameters are
  rejected in streaming mode. A completed checkpoint has a checksum-bound
  completion marker, so an interrupted one cannot be silently overwritten.
- Completed-point prevention reads compatible history before the expensive
  evaluation and rechecks after claiming the library; no rows are deleted.
  New rows carry a storage-context marker; untagged legacy rows keep the old
  experiment/inclination compatibility assumption. Metadata comparisons allow
  only 1e-12 roundoff, not the six-digit filename quantisation.
- Archive-first-wins: if a library already has a receipt but no usable local
  copy (deleted after delivery, or unreadable), the model is re-integrated and
  its penalty recorded, but the new realisation is **not** stored. This is
  reported before integration as `[orblib] ВНИМАНИЕ: … archived but not local
  … result not stored (#N in this process)` and in the history comment
  `# orblib archive-first-wins … archived_rebuilds=N`. There is no cloud fetch.
- Indexing: stream-mode libraries need none (each is its own object plus
  receipt, fetchable with `rclone copyto`). Legacy tar indexes store member
  name/size/MD5/metadata but no tar offsets; fetching one member currently means
  reading the tar (see `DECISIONS.md`, 2026-10-01).

### Local tar indexing and verified publication

Run from the installed `py/` directory, where `orblib_storage.py` and `orblib/`
are present. Stop the Yandex.Disk sync client first and leave it stopped until
publication succeeds. Do not run this alongside a calculation or another
storage controller. The helper does not start/stop the sync client itself.

```bash
python3 orblib_storage.py index --local-archives orblib --publish
```

- `orblib-index-work/` is created automatically (override with `--root`). It
  must be outside the archive/sync directory, including after symlink resolution.
  The standard `py/orblib-index-work/` path is gitignored.
- The helper opens existing **uncompressed `.tar`** files read-only and reads
  embedded `.npz` ZIP members through seekable file views. It validates metadata,
  ZIP CRCs, member MD5s and whole-tar MD5s without extracting `.npz` files or
  creating tar copies. Extra disk space is needed only for JSON indexes and
  control files; the streaming reader's one-member-plus-2-GB requirement does
  not apply to this local mode. AGAMA/NumPy are not required for the helper.
- Unpublished indexes go to `orblib-index-work/catalog/`, not into the sync tree.
  They include a local source stamp (device, inode, size, mtime and ctime).
  Unchanged, successfully indexed archives are reused on retry. Source changes
  during indexing are rejected; changes since indexing require re-indexing.
- Without `--publish`, indexing is entirely local and does not call rclone.
  With `--publish`, every selected cloud tar must first match the computed size
  and MD5. A missing/mismatching tar aborts before any publication.
- Only then are public indexes written under `orblib/catalog/` (without the
  private local source stamp) and uploaded to `yandex:galAgama/orblib/catalog/`.
  Cloud tar metadata is rechecked around publication; final index size/MD5 is
  verified. No tar or `.npz` payloads are uploaded, deleted, or rewritten.
  Conflicting existing local/remote index files are not overwritten.
- `--remote` and `--config` override the standard rclone remote and the current
  user's config. `--timeout` limits rclone operations, not local tar scanning.
- All selected `orblib_*.tar` files are indexed (both d0 and d1, not Q1-only).
  On an archive/member error the helper reports the names and stops; a partial
  set of staging indexes is not accepted as a complete publication batch.

Separate offline indexing and retryable publication:

```bash
python3 orblib_storage.py index --local-archives orblib
python3 orblib_storage.py publish-indexes --local-archives orblib
```

If publication fails, keep the sync client stopped and repeat `publish-indexes`
after resolving the error. This validates source stamps and cloud checksums
without decompressing libraries again. After exit code 0 and the final
`All local indexes published` message, the sync client may be restarted.

The corrected helper must also be deployed on the VM: `rclone lsjson` hash
names are parsed case-insensitively, and missing/invalid MD5s fail before a
large archive is streamed. `stat()` requests one object's metadata instead of
rehashing an entire local tree. Stream errors include archive name, expected
and actual size/MD5, rclone exit status and stderr. Tests exercise lowercase
JSON, missing hashes, tiny local tar files, publication failure/retry and
mocked rclone CLI, with no real cloud operations.

## 3a. Memory budget and watchdogs (after the 2026-09-29 stall)

The stall was a memory-reclaim livelock, not an OOM: with `swap 0` the kernel
could only reclaim file pages, so eight workers saving libraries at once kept
allocating while the VM spent hours refaulting pages (84 MB/s reads, dead SSH).
Everything below exists to keep that from recurring.

- **No trajectories.** `agama.orbit()` is called without `trajsize`; target
  matrices are accumulated during integration and do not depend on trajectory
  recording. Previously `trajsize=1000` allocated ≈3.2 GB per worker that the
  next line discarded. The script asserts one matrix per dataset.
- **Explicit OpenMP thread count for `agama.orbit`.** torch ships its own
  `libgomp.so.1` with the system SONAME, so AGAMA binds to it, and
  `torch.set_num_threads(1)` at import sets the main thread's OpenMP limit to 1:
  until 2026-10-03 every integration ran on one core. The call is now wrapped in
  `with agama.setNumThreads(AGAMA_ORBIT_THREADS)`, where
  `AGAMA_ORBIT_THREADS = --n_threads or len(os.sched_getaffinity(0))` (the
  container cpuset). It must be explicit: `setNumThreads(0)` restores the value
  seen at its first call, i.e. 1. IC sampling stays outside, so libraries for the
  same point stay bit-identical; torch/BoTorch keep one thread. The count is
  logged in `[orbitlib]` and in the `# orbitlib times` history comment
  (`omp_threads=`). Verified 2026-10-03: same library MD5 as the single-thread
  run, integration ≈20× faster on 32 vCPU. Production has the same issue (Q23).
- **`trajsize` is outside the library compatibility key.** New files store
  `trajsize=TRAJSIZE_STORED=0` (the field stays present because
  `orblib_storage.SCALARS` requires it), and the `_expected` dict omits it, so a
  legacy library (`1000`) and a new one (`0`) are equally reusable. It is also
  out of `EVALUATION_CONTEXT`; no history row carried a `# storage-context`
  line when this changed, so nothing was invalidated.
- **Block writer.** `write_orblib_npz()` writes the `.npz` member by member:
  the two big matrices get a hand-written `'<f8'` `.npy` header plus ~8 MB row
  blocks converted on the fly, so no full float64 duplicate of
  `matrix_dens`/`matrix_kinem` ever exists (previously ≈1.14 GB per worker).
  Member names, dtypes and values are identical to the old
  `savez_compressed` output. Atomic `.tmp` → `fsync` → `os.replace` is kept.
- **Hash while writing.** `HashingWriter` yields size + MD5 in the writing pass;
  `store.register(name, size=…, md5=…)` then validates with one full read and
  `metadata(deep=False)` instead of a `testzip()` pass plus a hash pass. Deep
  validation stays on every path inspecting files this process did not write
  (`prepare`, `prune-verified`, indexing).
- **Save slots.** Convert + write + register run inside `store.save_slot()`,
  `ORBLIB_SAVE_SLOTS=2` flock slots under `.storage/`. There is deliberately no
  stop check on entry — a model whose integration finished must still be saved
  after a delivery STOP — but the stop is honoured while waiting for a slot.
- **Signals.** SIGTERM/SIGINT write the storage `STOP` marker with reason
  `signal <name> from controller`; the existing polls turn that into
  `StorageStop` → checkpoint → exit 75, which the launcher understands. Without
  a store the checkpoint is written directly. A signal arriving inside a long C
  call (`agama.orbit`, `agama.solveOpt`) is acted on only after that call
  returns — hence the launcher's `ORBLIB_STOP_GRACE`. A container killed by the
  cgroup OOM gets SIGKILL and writes **no** checkpoint; what survives: history
  rows already written, delivered `.npz`, released `flock`s, expiring claims.
- **`log_mem(tag)`** prints RSS and `MemAvailable` before/after integration,
  around the npz write, after `register` and after the Upsilon search.
- **Worker default** is `nproc/8` (4 on 32 vCPU, 8 threads each) because the
  binding constraint is memory, not CPU. `--nproc=` still overrides.
- **Host swapfile** (step 0a, `ORBLIB_SWAPFILE=16G`, `0` disables): protects
  host processes (sshd, journald, orchestrator, rclone) only. Failure is a
  warning, never fatal.
- **Swapless container limit:** `--memory=$LIM --memory-swap=$LIM`, where
  `LIM = (MemTotal − 4 GB)/N_PROC` (override with `ORBLIB_MEM_LIMIT`, bytes).
  In docker `--memory-swap` is the *combined* RAM+swap ceiling, so equality
  means no container swap and an OOM-kill at the limit; leaving it unset
  defaults to `2 × --memory`. `check_swap_limit_support` asks
  `docker info --format '{{.SwapLimit}}'` (and falls back to the
  `No swap limit support` warning); where accounting is missing docker ignores
  `--memory-swap` silently, so the launcher warns, sets `vm.swappiness=1` and
  caps `ORBLIB_STALL_TIMEOUT` at 900 s. Exit 137 lands in the existing
  per-container error path.
- **Resource sampler** every `ORBLIB_MONITOR_INTERVAL=60` s →
  `monitor_{RUN_TAG}_{TIMESTAMP}.log` (`free -m`, `/proc/pressure/memory`,
  `docker stats`, `df`), uploaded with the other logs.
- **Two stall triggers** in the step-3 wait loop, sampled every
  `ORBLIB_WATCH_INTERVAL=10` s:
  *fast* — `MemAvailable` below `ORBLIB_MIN_AVAIL_MB=2048`, or
  `/proc/pressure/memory` `some avg60` above `ORBLIB_PSI_LIMIT=20` % for
  `ORBLIB_PSI_SAMPLES=3` consecutive samples;
  *slow backstop* — newest mtime across `dockerlog_*`, `log_*_p*.txt`,
  `out_*_p*.txt`, `orblib/*.npz*` not advancing for
  `ORBLIB_STALL_TIMEOUT=3600` s. `ORBLIB_MEMINFO`/`ORBLIB_PSI_PATH` exist so
  tests can drive both triggers without real memory pressure.
- **Fixed graceful order** (`handle_stall`): diagnostics (free, pressure,
  top-10 by RSS, `docker stats`, `df`) → `storage_cli stop` →
  `docker stop -t ${ORBLIB_STOP_GRACE:-1800}` → bounded wait on worker PIDs,
  then `docker kill` → `emergency_upload` → urgent notify →
  `schedule_shutdown 1`. Upload always precedes shutdown.
- **`emergency_upload()`** copies `$LOGFILE`, `monitor_*.log`, `dockerlog_*`,
  `log_*_p*.txt`, `out_*` and `.storage/STOP` to
  `galaxy_results_emergency/{RUN_TAG}_{TIMESTAMP}/`, deliberately bypassing the
  `storage_stopped && return 0` guard in `upload_to_yadisk` — that guard is why
  the 09-29 logs had to be copied off the VM by hand. Called from both triggers,
  from the `storage_stopped` branch of step 3 and from `on_exit`.
- **Shutdown accountability:** a failed `sudo shutdown` is no longer hidden by
  `|| true`; the exit status is logged explicitly and an `urgent` notification
  is sent so the VM cannot silently keep burning quota.

Expected per-worker peak afterwards: ≈1.5–2 GB instead of ≈4–5 GB ⇒ ≈8 GB for
four workers on 31 GiB.

## 3b. Single-model check (`run_single_model.py`, `launch_single_model.sh`)

Purpose: verify the memory fix and the experimental pipeline on real data by
re-evaluating one known model, not by searching. Tests: `tests/test_single_model.py`.

- `bash launch_single_model.sh` (from the installed `py/`): default point is
  the minimum-penalty row at `--incl` (default 90) of the production free-Q
  history `4UpsBoTorch_PCA_Sersic_*.txt` in the working directory, resolved at
  run time and logged verbatim (no result numbers in tracked files); its
  penalty/Upsilon are the reference. Configuration `d1_nb250_gh0_ser0`.
  Without those files all four `--Q= --gh= --rh= --rho0=` (or
  `--params-from=`) are required; layouts of `out_*`/`4Ups*`, `J_factor_*` and
  `Jcomputed_*` are detected by token count.
  Flags: `--repeats=4`, `--no-shutdown`, `--no-upload`, `--preflight`
  (one container, geometry/library name only, no uploads, no shutdown),
  `--Q= --gh= --rh= --rho0= --ref-penalty= --ref-upsilon= --params-from=GLOB
  --protocols=`. Refuses to start while `orblib/.launcher.lock` is held or when
  free disk < `repeats × SINGLE_BYTES_PER_REPEAT + ORBLIB_RESERVE_BYTES`.
- The runner imports the experimental script as a module (`--save-orblib
  --reuse-orblib --orblib-dir`, no `--stream-orblib`/`--Q1`/`--no-double`),
  installs a **local** `Store` (claim → save slot → block writer + hashing →
  `register`), replaces `completed_point` (otherwise the point already in the
  pool would be refused) and `checkpoint_for_stop` (no search checkpoint), and
  initialises the globals that normally come from `run_pca_optimization`.
  `pc_coords` is a dummy vector: `None` would crash the row writer.
- Protocols on **one** integration, each with an empty `_ups_recent` (full
  bracket) and `proc_rng` reseeded by `--subsample-seed` (AGAMA RNG untouched):
  `exp` (module defaults; integrates and saves; the only row written to the
  shared d1 pool, `out_<host>_d1_nb250_gh0_ser0_single<TS>r<i>.txt`),
  `reuse` (same settings on the reloaded float64 library) and `prod`
  (`xatol=1e-3`, no sub-sample, `min_pen=res.fun`). `reuse`/`prod` rows go to
  `single_<hostname_proc>.txt`, which no pool glob matches (a reuse row is the
  same IC realisation, and a prod row would carry a wrong `storage-context`).
- Report `report_single_<TS>_r<i>.json/.txt`: penalties, Upsilon, probes,
  wall time, peak RSS (`VmHWM`, reset per protocol), orbit/save/load times,
  deltas vs reference, both datacube grid upper bounds (`grid_identical` vs the
  production `bound_circR` formula), `GEOM_HASH`, `EVALUATION_CONTEXT`,
  library size/MD5/deep metadata check and store consistency.
  `run_single_model.py --summarize report_*.json` (stdlib only) aggregates.
- Each realisation has its own suffix and `orblib_single/<TS>_r<i>/`: the
  library name depends only on the parameters, so all realisations share it.
  Consequently each container has its own two save slots (four concurrent
  savers — stricter than the search). After the containers: pool files are
  uploaded as-is to `galAgama/` (unique names, no merge into the host file),
  reports/side files/logs to `galAgama/single_model/<RUN_TAG>_<TS>/`, and **only
  r0's** library goes to the shared catalog via `orblib_storage.py prepare
  --resume --root orblib_single/<TS>_r0` (size + MD5, receipt; the managed local
  copy is deleted after verification). If `catalog/<name>.json` already exists
  the upload is skipped and the file kept. A name known only from a **tar
  index** (legacy shards) makes `prepare` fail with `Conflicting library
  metadata: <name>` (same name, another realisation); the launcher treats
  exactly that message as archive-first-wins too (file kept, isolated root left
  stopped, exit code unaffected). r1… stay on the VM.
- Parameters outside the script's `bounds_original` are refused (the objective
  would clip them silently and record another model): `run()` stops after the
  module import with `status=failed`, exit 2, before any integration.
- Memory protections, watchdogs, emergency upload and verified shutdown are
  copies of §3a (`ORBLIB_*` variables keep their meaning); SIGTERM from
  `docker stop` makes the runner write a partial report and exit 75.
- Reading the result: `grid_identical=false` ⇒ the comparison with the
  production penalty is biased by geometry; `prod − ref` against the scatter of
  the repeats ⇒ consistency with production; `prod − exp` ⇒ Upsilon speed-up
  effect (Q15); `reuse − exp` ⇒ storage round trip (≈0 expected).

## 3c. Several different models in parallel (`launch_multi_model.sh`)

Re-evaluates N **different** known models at once (one container each, cores
split evenly), without protocol comparison: `run_single_model.py --protocols exp`
= one integration + the standard Upsilon search + library save. Tests:
`tests/test_single_model.py` (`*multi*`, `*models_file*`).

- `bash launch_multi_model.sh --models=FILE [--no-shutdown] [--no-upload] [--preflight]`.
  FILE: one history row per container in any layout the runner parses
  (`out_*`/`4Ups*`, `J_factor_*`, `Jcomputed_*`), optional trailing `# label`;
  comments/garbage lines are skipped; duplicate parameter sets are refused
  (same library name and, with AGAMA's fixed start seed, the same realisation),
  so are rows outside `bounds_original` (read from the script with `ast`).
  `run_single_model.py --list-models FILE` (stdlib) shows what will run.
  `incl` and `Q gh rh rho0` come from the row, its penalty/Upsilon are the
  reference (`--ref-penalty/--ref-upsilon`, label → `--ref-source`). The file is
  kept outside the repository (no result numbers in tracked files).
- Names: suffix `multi<TS>m<i>`, library dir `orblib_single/<TS>_m<i>/`, report
  `report_multi_<TS>_m<i>.json/.txt`, pool row
  `out_<host>_d1_nb250_gh0_ser0_multi<TS>m<i>.txt` (uploaded as-is to `galAgama/`),
  results/logs/models file → `galAgama/single_model/multi_d1_nb250_gh0_ser0_n<N>_<TS>/`,
  summary `summary_multi_<TS>.txt` = the per-model text reports concatenated.
- **Every** model's library goes to the shared catalog (`prepare --resume` +
  `check` per directory, skipped when `catalog/<name>.json` exists or the name
  is in a tar index — see §3b).
- `--preflight` runs all N containers without integration, no history/`.npz`
  upload, no shutdown. Lock `orblib_single/.launcher.lock` is shared with
  `launch_single_model.sh`; memory protections/watchdogs/shutdown as §3b.

## 3d. IC-seed scan (`--ic-seeds`, Q22 option A, DECISIONS 2026-10-04)

Measures the orbit-IC realisation scatter of **one** model. Diagnostic only:
the search and production never call `agama.setRandomSeed` (CONTRACT §Randomness).
Tests: `tests/test_single_model.py` (`*seed*`).

- `bash launch_single_model.sh --ic-seeds=1-100 --repeats=4 [--preflight]`
  (point as in §3b, default = min-penalty row at `--incl`). Seeds (integers
  ≥ 1, ranges/lists; `0` = clock in AGAMA ⇒ rejected) are split round-robin by
  `run_single_model.py --split-seeds SPEC --workers N` (stdlib): container j
  gets seeds j+1, j+1+N, … Bad spec, more containers than seeds or
  `--protocols` ≠ `exp` ⇒ exit 2 before any side effect.
- Each container imports the module **once** (saves ~18 s/seed) and for every
  seed calls `agama.setRandomSeed(K)` right before `halo_IC_lib_weights_pca_fixed`
  (`exp` protocol, empty `_ups_recent`, same `--subsample-seed`). The reset
  covers all per-thread AGAMA streams and `densityStars.sample()` is the only
  RNG consumer of the script, run single-threaded ⇒ the realisation depends on
  the seed only (not on order, layout or earlier seeds). AGAMA's start seed is
  42 (`src/math_random.cpp`) ⇒ seed 42 should reproduce the default-stream
  result of a fresh process — a built-in check of the mechanism.
- No library: the module gets neither `--save-orblib` nor `--reuse-orblib`
  (every seed is a new realisation under the same library name). The local
  `Store` is kept for STOP/claim handling only.
- History: `seeds_<host>_d1_nb250_gh0_ser0_single<TS>r<i>.txt` (no pool glob
  matches `seeds_*`), the standard block preceded by `# ic_seed: K`; one line
  per seed in the `.tsv` with the same stem (seed, status, penalty, Upsilon,
  probes, sample/orbit/solveOpt/wall s, peak RSS, `omp_threads`, time).
  A rerun with the same `--seed-file` skips seeds whose block is complete;
  a failed seed is recorded and the scan continues (container exit 1);
  STOP/SIGTERM ⇒ exit 75 between or inside evaluations.
- Report `report_single_<TS>_r<i>.json` → `ic_seeds.requested/results`;
  `--summarize` adds `seed_scan`: n ok / missing, mean/std/SEM, q16/median/q84,
  `mean_minus_ref` (+ in std, fraction of seeds ≤ ref), `seed42_penalty`,
  `distinct_penalties`/`identical_realisations`, orbit/wall time,
  `omp_threads`, `models_per_hour` (layout throughput) and the per-seed table.
- Upload: seed files (failure ⇒ exit code 1), reports, summary, logs →
  `galAgama/seed_scan/seeds_d1_nb250_gh0_ser0_i<incl>_<TS>/`; nothing to `galAgama/`
  (pool) or to the catalog. Memory protections, watchdogs, shutdown as §3b.
- Layout: 4×8 recommended (overlaps the single-threaded Upsilon search of one
  container with the integration of others; peak ≈ 4 × 2.6 GB). 1×32 is ~10 %
  slower by estimate; 8×4 gains little and leaves ~1 GB margin per container.

## 3e. Field-only potential check (Step 1A)

Separate tool, tests and full local-run instructions:
[`py/check_potential_convergence/README.md`](../../../py/check_potential_convergence/README.md).
It reads the harness recipe via AST, compares gravitational fields and real radial
meshes, and never integrates orbits, fits weights, seeds RNG, accesses cloud storage
or writes history rows. The user runs `--self-test` and then `--models FILE` under
local `gala` Python with AGAMA; no Docker. Fresh output directories only.
Q=1 halo forces use independent quadrature; stars/free-Q references must demonstrate
numerical convergence. `pass` concerns sampled fields, not penalty. Scientific Step 1B runs are deferred.

## 3f. Step 1B non-orbital preparation

`py/check_potential_convergence/run_potential_pair.py` has only `--validate-inputs`
(AST/JSON/export validation, no AGAMA import) and `--preflight` (local full harness
runtime, observations and A/B fields, no IC/orbits/solve). Instructions: its directory
README §6. The public CLI intentionally has no pilot execution mode.

The objective accepts an optional programmatic `diagnostic` callback after baseline
construction, before sampling/library access/fitting/history. It requires explicit
in-bounds parameters and disabled save/reuse/store. Callback failures propagate;
normal evaluation still follows its original path. `solve_orbit_library` shares
the existing solve/penalty equations and optionally returns weights, predictions
and unscaled linear-constraint residuals. No new search or production flags.

The diagnostic kernel is covered with fake AGAMA: common IC/time, separate native
controls, full-library Upsilon protocol, exact context/checksum matching, mmap
matrix reload and preserved partial reports. Ordinary orblib keys/storage are
unchanged; diagnostic matrices live in independent directories with manifests.
Source Step 1A recipe is checked independently of the changed harness source hash;
preflight requires the same AGAMA binary, matching export bytes/grids, and live
field agreement with an independently reconstructed archived recipe (max 1e-9).
Live-to-export/reload differences are separate non-gating serialization diagnostics;
AGAMA export recomputes/prunes harmonics and is not lossless. Reloaded current/archived
exports are also compared. Runtime context and per-variant stages/metrics persist
before failures; a normal failure of A does not suppress B's diagnostics.
A preflight pass certifies reproduction of live construction, not accuracy of a
reloaded .ini as an orbit potential. Neither preflight nor mock tests establish
IC admissibility, solver readiness or pilot memory/runtime. Q29 execution/tolerance
approval remains open; no VM, network, shared pool/catalog or shutdown is implied.

## 4. Editing rules

- `bounds_original` is defined **twice** in the script — change both or neither.
- Keep diffs surgical; no refactoring of neighbouring code.
- Document new behaviour **here**, not in `AGENTS.md`.

## 5. Verification (safe, no AGAMA, no cloud)

```bash
cd tests && ../.venv-ai/bin/python -m pytest -q test_orblib_q1.py \
    test_orblib_storage.py test_single_model.py --rootdir=. --import-mode=importlib -p no:cacheprovider
cd .. && python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py \
    py/orblib_storage.py py/run_single_model.py tests/test_orblib_q1.py
bash -n py/launch_orblib_exp.sh && bash -n py/launch_single_model.sh \
    && bash -n py/launch_multi_model.sh && git diff --check
```

Do not run `python -m pytest` from the repo root: local `py/` shadows pytest's
`py` module. Launcher and helper `--help` are side-effect free.
