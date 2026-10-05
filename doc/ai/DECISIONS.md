# Decision log

Append-only. One entry per PI decision or per design choice made without the
PI (marked as such). Never rewrite past entries; add a new one that supersedes.
Format: `date | topic | decision | consequence`.

## 2026-06 — Stage-1 PI answers (Q1–Q14, originally in `questions_for_pi.md`)

- **Geometry.** `posang = 46.8` is wrong → `42.3`; `q_ap = 0.7` → `1 − 0.31`.
  All results computed with the old values renamed to `*_PA46.8_*`; new runs
  keep the canonical names. The archive may seed initial points only with
  penalties recomputed (`--init-from-pa468`, opt-in).
- **Q1 penalty.** Cannot be treated as χ². Ranking score only.
- **Q2 J weighting.** As implemented: `w = exp(-(penalty − pen_min)/pen_sigma)`,
  `pen_sigma = max(std, 1e-6)`, normalised, weighted KDE.
- **Q3 `incl` prior.** Physical constraint only:
  `axRZst = sqrt(q_ap² − cos²i)/sin i` real ⇒ `cos i < q_ap`.
- **Q4 result files.** Use all of them, including per-process `_pN` files
  mid-run.
- **Q5 legacy `4UpsBoTorch_Sersic.txt`.** Same column format, older code; keep
  reading it.
- **Q6 `4result_*`.** Monitoring only.
- **Q7 dedup.** Never deduplicate stored rows, not even across hosts — each row
  is a unique experiment result.
- **Q8 metadata sidecars.** Allowed, provided existing file formats stay intact.
- **Q9 bounds.** Changing `bounds_original` requires asking the PI first.
- **Q10 frozen inputs.** `massSt = 14.0` and `D = 143 kpc` fixed; Sersic-error
  propagation must be *separate new code*, not edits to the main script.
  Sersic index uncertainty is ±0.006 (per `Wang_2019_table1…txt`), not ±0.06.
- **Q11 RNG.** Give parallel workers different seeds; the GH observation-error
  bootstrap stays at fixed `seed = 42`.
- **Q12 J-factor script paths.** Analysis runs on the local machine; the
  hardcoded `YADISK_DIR` with the user name is intentional.
- **Q13 Dockerfile.** Added to the repo root together with `check_agama.py`,
  `entrypoint.sh`.
- **Q14 iterations.** PI asked for a recommendation → production
  `pca_update_interval = 12`, `n_iter = 40` per run.

## 2026-08/09 — Experimental harness (agent decisions, PI-approved scope)

- **2026-08 | experiment isolation** | `EXP_ID = d{0|1}_nb{N}_gh{G}_ser{S}` is
  embedded in every derived filename | penalties of different `double`/`n_bin`
  are never pooled.
- **2026-09 | fixed-Q mode** | CLI flag `--Q1` (uppercase Q; lowercase `q` is
  reserved for the stellar flattening in the article); writes
  `Q1d1_nb250_gh0_ser0`, reads both Q modes but filters rows to `Q=1` |
  see `harness/orblib_exp.md`.
- **2026-09 | bounds widened** | `rh 3.5 → 7.0`, `rho0 34 → 10` after the `Q=1`
  run pinned `rh` at the upper bound | diagnostic range, not a physical claim;
  if a new minimum again lands on the bound, build a profile in `rh` instead of
  widening further.
- **2026-09 | orbit-library storage** | streaming per-file upload with MD5
  receipts replaces the tar-snapshot flow; on exhausted delivery attempts the
  run stops cleanly and the VM powers off with data intact | old tars need a
  one-time `index`.
- **2026-09 | harness/paper separation** (this task) | experiment knowledge in
  `doc/ai/`, article drafts in a nested private `paper/` repo, run facts in
  `results/REGISTRY.md` | AGENTS.md stays ≤ 120 lines of pointers. `results/`
  and `paper/` are local-only (public repo): unpublished numbers are never
  committed, so the registry lives outside version control.

## 2026-09-28 — Local archive indexing (PI-requested)

- **MD5 interoperability** | accept case-insensitive hash names in rclone JSON;
  reject missing/invalid hashes before reading archive data | add real-format
  JSON tests and detailed size/hash/stream error reporting.
- **Local-first indexing** | read already downloaded uncompressed tar members
  directly, without extraction; stage JSON under `orblib-index-work/`, outside
  the sync directory | no temporary orbit-library copies or AGAMA execution.
- **Publication gate** | with the sync daemon stopped, verify cloud tar sizes
  and MD5s before writing ready indexes into `orblib/catalog/` and uploading
  only those indexes | failed verification must not publish indexes; existing
  archives are never rewritten. Publication is separately retryable without
  re-indexing unchanged local archives.
- **Execution ownership** | the user runs indexing/publication from their local
  installation with their configured rclone; the agent supplies code, commands
  and isolated tests | no archive cleanup, credentials changes, VM launch or
  shutdown is part of this task.

## 2026-10-01 — Memory fix for library-saving runs (approved plan, now implemented)

Diagnosis of the 2026-09-29 stall (kernel journal + docker logs): a
memory-reclaim livelock, not an OOM — with `swap 0` only file pages were
reclaimable, so eight concurrent library savers kept the VM refaulting for hours.
Details and the per-item behaviour: `harness/orblib_exp.md` §3a.

- **Q21 answered: trajectories dropped** | `agama.orbit()` is called without
  `trajsize` (the result was discarded one line later, ≈3.2 GB per worker) |
  **open verification**: §9.1 of the retired plan still has to be run — recompute
  a penalty from a stored library and re-integrate one control model to show
  `matrix_dens`/`matrix_kinem`/`penalty` are unchanged. Until that is done,
  treat the change as justified-but-unverified on real data.
- **`trajsize` out of the compatibility key** | new files store `0`, `_expected`
  omits the field, `EVALUATION_CONTEXT` no longer hashes it | the seven surviving
  2026-09-29 libraries (`trajsize=1000`) stay reusable; no history row was
  invalidated because none carried a `# storage-context` line yet.
- **Block npz writer + hash while writing** | no full float64 duplicate of the
  big matrices; `HashingWriter` + `register(size=…, md5=…)` + `metadata(deep=False)`
  replace a `testzip()` pass plus a separate hash pass | file format unchanged
  (same member names/dtypes/values); deep validation stays on every path that
  inspects files the process did not write.
- **Save slots** | `ORBLIB_SAVE_SLOTS=2` flock slots cap concurrent savers; no
  stop check on entry | a model whose integration finished is still saved after a
  delivery STOP, as `claim()` already allowed.
- **Host-only swap + swapless container** | `ORBLIB_SWAPFILE=16G` protects sshd/
  journald/orchestrator/rclone; `--memory=$LIM --memory-swap=$LIM` forbids
  container swap so a runaway worker is OOM-killed (137) instead of thrashing the
  VM | a legitimate spike now loses one model without a checkpoint; mitigated by
  the 4-worker default and a generous limit. If the kernel lacks swap accounting
  docker ignores `--memory-swap`, so that case is detected and logged.
- **Fast trigger + 60-minute backstop + fixed graceful order** | PSI/MemAvailable
  trigger in minutes, no-progress backstop at 3600 s; stop → docker stop →
  bounded wait → emergency upload → notify → shutdown | logs now reach
  `galaxy_results_emergency/` even during a STOP, and a failed `sudo shutdown` is
  logged and notified instead of swallowed by `|| true`.
- **Default 4 workers** (`nproc/8`) for library-building runs | the binding
  constraint is memory, not CPU | `--nproc=` still overrides.
- **Run order afterwards** (not started, needs explicit go-ahead): first a
  `--reuse-orblib` pass over the seven surviving libraries with 4 workers, only
  then a fresh 4-worker `--Q1` search on the widened bounds.

## 2026-10-01 — Archived-but-not-local libraries; tar offsets deferred (agent, user-approved plan)

- **Warning + counter, no behaviour change** | when `store.archived(name)` is true
  but no usable local copy loads, `note_archived_rebuild()` prints
  `[orblib] ВНИМАНИЕ: … archived but not local … result not stored (#N)` before
  `agama.orbit`; the per-process `orblib_archived_rebuilds` also goes into the
  `# orblib archive-first-wins … archived_rebuilds=N` history comment | the
  evaluation still runs and its penalty is recorded; the wasted integration
  (typical cause: changed `EVALUATION_CONTEXT` after local cleanup) is now visible.
- **Legacy tar indexes: no re-indexing** | the user already ran
  `index --local-archives orblib --publish`; the indexes hold member name, size,
  MD5, metadata and archive size/MD5 — enough for `import_catalog`/`archived()`.
  The next `prepare` fails on any cloud `orblib_*.tar` without an index, so
  coverage is checked automatically | stream-mode libraries need no indexing
  (individual `objects/<name>` + `catalog/<name>.json`).
- **Tar offsets deferred** | only a future-fetch speed-up; revisit when a
  re-score/fetch mode is designed and ranged `rclone cat --offset --count` is
  confirmed on Yandex. Then: header-only `tarfile` scan (`offset_data`), a
  separate sidecar file (never rewrite existing `legacy_*.json` — `publish()`
  refuses different content under the same name), MD5 check of the fetched range.

## 2026-10-01 — Single-model check of the memory fix (user-approved plan)

- **Separate runner + orchestrator** (`py/run_single_model.py`,
  `py/launch_single_model.sh`) instead of a new mode in the tested launcher |
  the experimental script is imported as a module and only module globals are
  patched inside the runner process; production, the exp script and
  `launch_orblib_exp.sh` stay unchanged.
- **Point** | the free-Q best row at `incl=90` of the production history,
  configuration `d1_nb250_gh0_ser0` (production always doubles). Resolved at run
  time from `4UpsBoTorch_PCA_Sersic_*.txt` rather than hard-coded, so no result
  numbers enter the public repository.
- **4 parallel realisations** | memory stress test like the launcher's 4
  workers plus the IC-realisation scatter needed to judge `prod − ref`. Each
  has its own orbit-library directory (same parameter-derived name).
- **No trajectory A/B re-integration** | not requested; the check rests on the
  penalty comparison and on peak-RSS measurements.
- **History** | only the `exp` row (exactly what the search writes) enters the
  shared d1 pool, under a unique file name; `reuse` (same realisation) and
  `prod` (would carry a wrong `storage-context`) go to a side file.
- **Libraries** | r0's library is delivered to the shared catalog
  (`prepare --resume`, size + MD5, receipt); skipped when the name is already
  archived; r1… stay on the VM.
- **Shutdown by default**, as the launcher; `--no-shutdown` disables.
- Not run yet: the VM run needs an explicit go-ahead.

## 2026-10-03 — Parallel re-evaluation of several known models (user request)

- **Selection** | local d1 free-Q history (production `4UpsBoTorch_PCA_Sersic_*`
  + `Jcomputed_from_raw_*`; d0 and `PA46.8` excluded). First the global best
  plus the next three by penalty; revised the same day on request to the global
  best plus three *distant* good regions: farthest-point selection among rows
  within Δpenalty < 0.06 of the best (≈ realisation scatter), parameters
  normalised by production bounds. The history has a single basin, so these are
  the three ends of its valley (large rh, small rh, gh > 0); genuinely different
  regions all have Δpenalty ≳ 0.6. Rows live in a local-only models file.
- **New orchestrator** `py/launch_multi_model.sh` (copy of the single-model
  memory protections) instead of a mode in the tested `launch_single_model.sh`;
  runner reused with `--protocols exp` plus two small additions
  (`--list-models`, `--ref-source`).
- **Upsilon is still searched** (standard `exp` protocol): the penalty is
  defined as the minimum over Upsilon (CONTRACT); no exp/reuse/prod comparison.
- **All libraries delivered** to the shared catalog (each model is distinct).
- Not run yet: the VM run needs an explicit go-ahead.

## 2026-10-03 — OpenMP threads for orbit integration in the harness (user request)

- **Cause confirmed** | torch's bundled `libgomp.so.1` is shared with AGAMA;
  `torch.set_num_threads(1)` ⇒ `omp_get_max_threads()` = 1 in the main thread
  (8 without torch, 8 inside `agama.setNumThreads(8)`), so `agama.orbit` ran on
  one core per container.
- **Harness fix** | only `agama.orbit` is wrapped in
  `agama.setNumThreads(AGAMA_ORBIT_THREADS)` (`--n_threads` or cpuset size);
  IC sampling, torch/BoTorch and solveOpt unchanged. Results must be
  bit-identical (orbits are independent rows); only wall time changes.
- **Production** untouched; porting is Q23.

## 2026-10-04 — Q22: distinct AGAMA IC seeds in the diagnostic runner (user acting for the PI)

- **Decision** | option A: `agama.setRandomSeed(K)` (K ≥ 1) is called **only**
  by `run_single_model.py --ic-seeds`, right before each `exp` evaluation.
  Search (`Fornax_P21_PCA_w3Sersic_orblib_exp.py`) and production still never
  seed AGAMA; CONTRACT §Randomness unchanged. Per-process seeding of the
  search would be a separate question (Q24, not asked yet).
- **Run design** | 100 seeds (1…100, includes AGAMA's start seed 42 as a
  check against the default-stream result) of the production free-Q best at
  `incl = 90` (d1); no orbit library saved or reused; rows only in separate
  `seeds_*` history files with `# ic_seed: K` per block plus a `.tsv`, nothing
  in the d1 pool or the catalog; layout 4 × 8 CPU, each container runs its 25
  seeds sequentially in one process; VM shutdown at the end as usual.
- **Layout reasoning** | with the OpenMP fix one model is ≈145 s of 32-thread
  integration + ≈20 s mostly serial Upsilon search (no save). ×20.5 on 32 vCPU
  suggests hyper-threading, so parallel containers cannot speed up integration
  itself; they only overlap the serial phases (~10 % by estimate). 4 × 8 keeps
  the peak at ≈10 GB; 8 × 4 gains little with ~1 GB margin per container.
  The run also measures the 4 × 8 throughput (`models_per_hour`).
- Not run yet: the VM run needs an explicit go-ahead.

## 2026-10-05 — TuRBO trust-region upper bound fixed in harness and production (user acting for the PI)

- **Bug** | `TuRBO_PCA_Fixed._tr_bounds` computed the upper TR bound as
  `hi_norm·range + pca_bounds_upper` instead of `+ pca_bounds_lower`, so in every
  PC the TR reached from `centre − L/2` to above the PCA box; proposals left the
  box and were clipped to `bounds_original` in `pca_to_params_fixed` (consistent
  with boundary pile-ups). Same line in all four scripts.
- **Decision** | fixed (one line, `+ self.pca_bounds_lower`) in the harness and,
  on the user's explicit instruction, in `Fornax_P21_symm_PCA_w3Sersic_yaVM.py`,
  `_yaVM_timed.py` and `Fornax_P21_symm_PCA_w3Sersic.py` (the handoff's Q24 is
  thereby answered as option A and not filed). Penalty, apertures, Upsilon search
  and bounds unchanged: only the search dynamics change. Histories produced
  before this commit (incl. the free-Q baseline and the 09-19 Q=1 run) were
  searched with the wide upper TR; their rows stay valid evaluations.
- Test: `test_trust_region_bounds_stay_inside_pca_box` (all four scripts).

## 2026-10-05 — Floor on the weighted std of the PCA scaler (harness only; user acting for the PI)

- **Problem** | with weights `exp(-(p − pmin)/0.1)` the best rows dominate; when they
  share a value (e.g. `rh` on a bound) the weighted std of that column is ≈0 but
  above the old `1e-10` guard, so the column was scaled by ~1e-7 and the PCA box
  blew up (09-19 log: PC1 `[-406, 773]`), making TR lengths meaningless.
- **Decision** | in all three PCA builders (`build_initial_pca_from_bootstrap`,
  `_update_pca_model`, `run_pca_optimization`) `weighted_std` is floored at
  `WeightedScaler.STD_FLOOR_FRAC = 0.05` × the width of `bounds_original` in the
  transformed space (log10 for `rh`, `rho0`). Weight temperature 0.1 unchanged.
  Penalty, bounds and the evaluation are unchanged; only the search geometry.
- **Production** (`_yaVM.py` etc.) unchanged — has the same scaler; porting would
  be a separate question. Test: `test_pca_box_stays_finite_when_best_rows_share_a_bound_value`.
