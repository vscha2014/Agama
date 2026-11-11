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

## 2026-10-06 — Q28: `rh` upper bound raised to 30 kpc (user acting for the PI)

- **Answer** | `rh` may be widened. Both copies of `bounds_original` in the harness
  now use `rh ∈ [0.5, 30]` kpc, for **both** modes (free Q and `--Q1`); the other
  bounds are unchanged; CONTRACT updated. Production untouched. 30 kpc covers the
  near-uniform-core limit of the Zhao `alpha=2, beta=3, gh=0` halo inside the data.
- **Why** | `Q=1` probes (local `results/Q1d1_rh_bounds/NOTES.md`): penalty falls
  monotonically up to the old bound 7 kpc, so the minimum was bound-limited.
- **Side effects** | cold-start LHS samples `rh` linearly ⇒ most fresh LHS points
  land at rh > 7 (only without history); the reservation distance is normalised
  by the bound width ⇒ the same `reserve_eps` is ≈4.5× wider in kpc along `rh`;
  the PCA std floor of `rh` grows accordingly (log10 width). `EVALUATION_CONTEXT`
  does not include bounds ⇒ history stays compatible.
- **Protection** | `run_single_model.py` refuses parameters outside the script's
  `bounds_original` (`--list-models` and `run()`), because the objective clips them
  silently and would record another model under the requested label.
- **Delivery** | `launch_single_model.sh`/`launch_multi_model.sh`: `prepare`
  failing with `Conflicting library metadata: <that name>` = the name is in a
  legacy tar index ⇒ archive-first-wins (file kept on the VM, no failure code).
  `compatible_metadata` keeps comparing `trajsize`: the conflict is genuine (another
  realisation under one name) and must not be published.

## 2026-10-06 — Field-only potential convergence diagnostic (user request)

- **Scope** | Step 1A only, isolated under `py/check_potential_convergence/` with
  tests and usage documentation. No changes to the harness, production, scientific
  definitions, launchers or storage; no orbit integration, IC sampling or fitting.
- **Execution** | user will run the installed AGAMA locally as `gala`, without
  Docker. A standalone `--self-test` checks Plummer forces and Multipole export;
  the agent ran only numeric/mocked tests. Real AGAMA validation remains pending.
- **Inputs/units** | read the harness recipe and bounds through restricted AST,
  never execute it; models are explicit local rows. Compare the unscaled G=1
  integration potential, not densities multiplied by Upsilon twice.
- **Reference** | independent spherical quadrature for Q=1 halo; separately tested
  radial/angular/domain convergence for stars and non-spherical halos. An
  unconverged reference cannot produce a pass. Changing grid extent does not
  change the physical halo cutoff.
- **Isolation** | fresh output directory only; actual grids, fingerprints and
  reports retained locally, nothing enters the shared history or library catalog.
  Step 1B (penalty sensitivity) and adoption of new production settings require
  separate approval. Behaviour/commands: `py/check_potential_convergence/README.md`.

## 2026-10-08 — Step 1B implementation and checks without orbits (user request)

- **Approved scope** | prepare the implementation and local mocked/numeric checks,
  not the scientific pilot. Q29 practical tolerance, real seeded/common-IC runs,
  seed-series expansion and production adoption are still pending.
- **Isolation** | a separate diagnostic module with only validation/preflight CLI;
  the paired computational kernel is tested with fake AGAMA, not exposed as a
  real-orbit CLI. No cloud operations, shared pool/catalog writes or data deletion.
- **Harness interface** | a guarded optional callback after baseline construction,
  and extraction of the existing solve equations into a shared helper. Ordinary
  defaults, physical recipe, RNG behaviour and library keys are unchanged.
  No production/upstream, launcher, storage or CONTRACT changes.
- **Verification boundary** | local development dependencies suffice for mock tests
  and static input/evidence validation. A real no-orbit preflight needs the full
  AGAMA/harness environment, available locally or on a separately approved VM;
  it does not verify the real solver, orbit coverage or pilot resource requirements.
  Commands and remaining work: diagnostic README §6 and harness §3f.

## 2026-10-08 — Separate live reproduction from potential serialization (user request)

- **Scope** | fix the non-orbital preflight, verify it, repeat the free-Q preflight
  locally and prepare the continuation handoff. No orbit pilot or production adoption.
- **Reason** | Multipole export reconstructs/prunes coefficients; comparing a live
  object directly to its text export reloaded conflates recipe reproduction with
  serialization error. A spherical smoke test does not establish lossless export.
- **Checks** | keep max 1e-9 for live-to-live reconstruction from the archived recipe
  and for comparison of two reloaded exports; require exact export/grid identity.
  Measure live-to-reloaded differences separately, with explicit warnings rather
  than treating them as recipe drift. This does not certify exported orbit potentials.
- **Reporting** | persist context and per-variant stages/metrics before failures,
  including all/main scores and worst coordinates. Test lossy export, actual field
  drift, nonfinite forces and partial results. Keep the scientific Step 1A thresholds,
  harness, upstream and CONTRACT unchanged by this correction.
- **Outcome location** | local results/registry/handoff hold the actual run details;
  Q29 pilot/tolerance remains open. No scientific penalty evaluation was authorised.

## 2026-10-08 — Two Q1 preflights only (explicit user approval)

- **Scope** | execute the two remaining local Q1 no-orbit preflights sequentially,
  using the existing exact inputs/dense evidence and fresh isolated output paths.
  No IC sampling, orbit integration, solveOpt, shared history/library writes or cloud.
- **Verification** | static input validation, live and serialized reproduction,
  exact grid/export checks, independent export-roundtrip reporting and no-orbit flags.
  Run details remain in the local registry, Q1 notes and potential-check handoff.
- **Boundary** | Q29 common/native seeded protocol, practical penalty tolerance,
  scientific pilot entry point/execution and later seed expansion are recommendations,
  not approved by this preflight-only answer. Production and CONTRACT stay unchanged.

## 2026-10-08 — Q29: prepare the seed-42 pilot for ya VM (user request)

- **Approval** | prepare the pilot according to the recommended design, not execute
  AGAMA/VM/Docker/cloud operations. This supersedes the previous implementation-only
  prohibition on a scientific entry point; real execution remains a separate gate.
- **Protocol** | option B: saved common float64 baseline IC/time for A/B, with native
  B controls. Three fixed input rows; seed 42 only. Free-Q has A/B/A-repeat/C/native
  (five integrations), then each Q1 has A/B/native (three each), eleven in total.
  A is the ordinary baseline; live B is N184/l24 with each baseline's radial limits;
  C is the exact N399/l40 common_fine from dense free-Q evidence. Never integrate a
  loaded coefficient export. No change to production RNG or global scientific recipe.
- **Solve/tolerance** | full-library bounded Upsilon [0.1,1.6], xatol=1e-3, maxiter=50,
  fixed-input and cross-Upsilon solves. An additional xatol=1e-4 check on the same
  matrices does not replace the primary fit. Practical delta_P=0.01; technical
  budget=0.001. These are engineering thresholds, not observational significance.
- **Gates** | validate the exact models/evidence; field-preflight the actual live
  A/B/C objects before sampling. Preserve the AGAMA binary guard on VM; a different
  image needs separately reviewed field evidence, not a bypass. Review the five
  free-Q integrations and resources before explicitly passing its report to Q1.
  Large A→B alone does not fail the pilot; C/B and native/common sensitivity remains
  a separate interpretation issue. Other seeds require another decision.
- **VM isolation** | dedicated single-container launcher, read-only code/input,
  no network/upload/deletion. Optional `--shutdown` and `--notify` (both off by
  default, user request) act only after the container exits and logs/state are
  synced; the delay is a cancellation window, not a compute estimate. Notifications
  come from the host, carry status/exit code only — never penalties or deltas — and
  a failed one never blocks shutdown; failures are logged and warned, never silenced.
  Results are not uploaded, so they stay on the VM disk. Equal RAM and total RAM+swap
  limits, host/disk headroom, timeout/no-progress stops, resource logs and retained
  container state. No changes to ordinary launchers, storage, production or CONTRACT.
- **Implementation** | pilot modes extend the diagnostic runner only; common kernel
  reports partial stages, timings, current/process-peak RSS, exact IC/time and matrix
  manifests. JSON-normalised manifest comparison handles tuple bounds without
  weakening checksums. Harness signal handlers cannot escape its isolated import.
- **Verification/scope** | mock/numeric tests and static validation of local input
  rows, syntax and frozen-file diffs; no new real field/orbit/solver/VM run. Commands
  and artifacts are documented in the existing diagnostic README §7 and local handoff.

## 2026-10-10 — Searches on the workstation tycho via the experimental harness (user request)

- **Strategy** | instead of porting the harness features (OpenMP for
  `agama.orbit`, no trajectories, orbit libraries, timing/RSS logs, signals,
  reservations, PCA fixes, bounds) into the old fork
  `Fornax_P21_symm_PCA_w3Sersic.py`, tycho runs
  `Fornax_P21_PCA_w3Sersic_orblib_exp.py` unchanged through the new
  `py/launch_tycho.sh`. The fork and `run_single.sh` stay untouched (no second
  copy to keep in sync).
- **Budget** | at most a quarter of the host: conda env + one `systemd-run
  --user --scope` for the whole launcher, `CPUQuota = nproc/4 × 100 %`,
  `MemoryMax = MemTotal/4`, `MemorySwapMax=0`, `OOMPolicy=continue`,
  nice/ionice; explicit `--n_threads`. Default 1 worker × 4 threads, option 2×2.
- **Libraries** | saved and reused locally (`--save-orblib --reuse-orblib`), no
  `--stream-orblib`, no cloud catalog; preflight requires 40 GB free.
- **History** | experimental d1 pool `out_tycho_d1_nb250_gh0_ser0*`
  (`Q1d1_*` with `--Q1`), not the canonical `4Ups*` pool.
- Mocked tests only (`tests/test_launch_tycho.py`); a property probe of the
  scope (`--_print-limits`) was checked on the dev machine. No real tycho run.

## 2026-10-12 — tycho runs without network sync (user decision)

- rclone on tycho cannot list Yandex.Disk (`410 GoneError`, also with a fresh
  binary, while `about` works); it will **not** be fixed. Code and result files
  travel by hand (Yandex.Disk daemon on another machine, or scp).
- Consequence: tycho runs are fully offline. User choices: an offline switch
  in the harness (not a stub `rclone`), ntfy kept, disk space checked by the user.
  Implemented as `--offline` in the harness (`load_from_yadisk`/`sync_to_yadisk`
  return before any rclone call, `finalize` skips the sync alarm; default off, VM
  unchanged) and in `launch_tycho.sh` (no rclone, results bundle for manual
  transfer). Details: `harness/orblib_exp.md` §3g.
- Next on tycho (user plan): first `incl=90`; then the Q29 potential-convergence
  pilot that never ran on the ya VM, adapted to tycho; then Q=1 studies.

## 2026-10-13 — Q29 pilot on tycho: native wrapper (user decisions)

- New `py/check_potential_convergence/launch_potential_pair_local.sh`; the runner and
  the docker VM wrapper are unchanged. One systemd user scope, 4 threads (`nproc/4`).
- **Memory 8 GiB** (as on the VM, slightly above MemTotal/4): user choice;
  `--memory-gb` overrides. Swap forbidden in the scope (OOM-kill instead).
- **Sequential only**: refuse when a `agama-tycho-*`/`agama-pair-*` scope is active
  or the `launch_tycho.sh` lock is held.
- **Watchdogs notify only**: on tycho there is no billing and no swapless host
  livelock (scope `MemorySwapMax=0` → OOM-kill); a fixed timeout risks stopping a
  legitimately long refined-potential integration. Stall (`report.json` unchanged
  6 h) and runtime (48 h) send one ntfy each; the host-memory stop is dropped.
- `--threads` must be the same for free-Q and Q1 (environment signature).
- ntfy on by default (`--no-notify`), status only. No real tycho run by the agent.

## 2026-10-14 — Q29 pilot executed on tycho (closes Q29)

- User ran the eleven-integration seed-42 pilot on tycho with
  `launch_potential_pair_local.sh` instead of the ya VM: validate → preflight → free-Q →
  agent review → both Q1 controls. All `pilot_pass`, technical budget met, no OOM.
- The tycho conda env resolved to a different AGAMA build (same 1.0.156, other `agama.so`)
  than the 2026-10-08 evidence; the binary guard stopped the preflight. User chose fresh
  Step 1A evidence in that env (`--mode=fields`) rather than the old env; field metrics and
  statuses matched the old evidence to ≤ 2e-6 relative. The guard was not bypassed.
- Outcome (numbers local only): N184/l24 is converged at penalty level (C agrees);
  free-Q changes by ≫ δP between N23/l4 and N184/l24, Q1 controls by ≲ δP. Follow-up: Q30.

## 2026-10-20 — Git-backed `results/` sync for VM and tycho launchers (user decision)

- **Context** | Yandex.Disk (`rclone`) is unavailable across protocols; history and run
  artifacts now live in the private GitHub repository `results/` (locally excluded from
  the public `Agama` repository), while code and `py/table3.dat` are updated on VMs via
  `git pull` of the public `Agama` repository.
- **Decision** | All experimental launchers (`launch_multi_model.sh`,
  `launch_single_model.sh`, `launch_orblib_exp.sh`, `launch_tycho.sh`) auto-detect
  `${WORK_DIR}/../results/.git` (or accept `--results-dir=DIR`, `--git-results`,
  `--no-git-results`). At start they `git pull --ff-only` and the Python scripts read
  historical `4Ups*`/`out_*` directly from `../results/` (`-v "${RESULTS_DIR}:/results"` in
  Docker) without copying them into `py/`. At finish and on emergency, per-worker
  `out_<host>_<EXP_ID>_<sfx>.txt` files are merged via `append_and_remove` (`HASH:`) into
  the end of the regular file `../results/out_<host>_<EXP_ID>.txt` (no symlinks), while
  logs, reports and `MODELS_FILE` go into `../results/logs/<RUN_TAG>_<TS>/` (`../results/logs/seed/<RUN_TAG>_<TS>/`
  for `--ic-seeds`), followed by `git commit` + `git push` with `pull --rebase` retries.
  `.npz` orbit libraries (~618 MB) stay local on the VM (`ORBLIB_LOCAL_STORE=1` in
  `launch_orblib_exp.sh` preserves save slots and `STOP` handling without `rclone`).

## 2026-10-22 — Q30: refined multipole (`--l24`) for free-Q searches and `Q1d1` consolidation (user decision)

- **Skip fixed-seed valley check** | Do not run a separate `seed=42` pass on `models_top4.txt`;
  proceed directly to the `TuRBO-PCA` `free-Q` search with the converged multipole
  `N=184, lmax=24` (`--l24`, equivalent to `--pot-lmax=24 --pot-ngrid=184`).
- **Zero-overhead `pot_orbit`** | Keep the literal `pot_gal = agama.Potential(..., lmax=4, mmax=0, gridSizeR=23)`
  assignment inside `if diagnostic is not None or (_pot_lmax, _pot_ngrid) == (4, 23):` so the
  Step 1A/1B AST parser (`read_recipe`) and `diagnostic` callback remain intact, while during
  ordinary `--l24` runs (`diagnostic is None`) only `pot_orbit` (`lmax=24, gridSizeR=184`) is
  constructed at runtime (zero extra CPU or RAM overhead).
- **Consolidate `Q=1` history into `Q1d1_*`** | All `Q=1` blocks from single/multi-model probe
  runs previously written into `out_*_d1_nb250_gh0_ser0.txt` are moved into
  `out_*_Q1d1_nb250_gh0_ser0.txt` (`results/`), leaving only old `free-Q` (`lmax=4`) rows in
  `out_*_d1_nb250_gh0_ser0.txt`. Future `Q=1` runs via `run_single_model.py` and launchers use `--Q1`.
- **History and library isolation vs `Q=1` reuse** | Under `--l24`, `HISTORY_ID = d1_nb250_gh0_l24n184_ser0`;
  the `free-Q` PCA/TuRBO pool reads only `out_*_d1_nb250_gh0_l24n184_ser0*.txt` (not mixing in
  the `Q=1` pool). All previous `free-Q` (`lmax=4`) files (`4UpsBoTorch_PCA_Sersic*.txt` and
  `out_*_d1_nb250_gh0_ser0*.txt`) are included in `prior_seed_patterns` to seed initial `free-Q`
  points (`Q != 1`) with orbits and penalties recomputed in `N=184, lmax=24`. For `Q == 1.0`,
  `orblib_key` omits `_l24n184` and `--Q1` reads `Q1d1` history, so existing `Q=1` penalties
  and `.npz` libraries remain reusable without recomputation.

## 2026-10-26 — Finalist batch: library check fix and seed-scan isolation (agent, user request)

- **Bug** (run `…_n2_20261025_225604`) | the runner's final store check compared a deeply
  valid, pre-existing but never registered library with the `claim()` placeholder
  (`state=building`, size/MD5 NULL) and failed both reports after all protocols completed.
- **Fix** | a placeholder of a library only read in this run is reported as
  `store_registered=false`, `store_consistent=null` and is not a failure. Still fatal:
  deep metadata failure, a non-null size/MD5 mismatch, and a library saved by this run
  without registration. The Store itself and its records are unchanged (no implicit
  registration of old files).
- **Control mode** | `launch_tycho.sh --models … --protocols≠exp` passes the new runner flag
  `--require-library` (stop before work if the file is missing; fail if a protocol
  integrated instead of reusing). Reports carry `requested_potential` /
  `potential_provenance`; Q=1 names omit the potential tag, so a reused Q=1 library under
  `--l24` is labelled `unknown_shared_q1_key`, not L24.
- **Seed mode** | `--ic-seeds` uses an isolated `orblib_seed/<TS>_m<i>/` store directory
  (fresh integrations, no save/reuse), so the working `orblib/.storage` gets no records.
- **Summary** | `distinct_models` is reported; the std across different models is not used
  as realisation scatter (`prod_minus_ref_in_scatter` omitted).
- Mocked tests only; the seed scan itself is run by the user on tycho.

## 2026-10-28 — Q profile with common random numbers (user acting for the PI; plan item 2)

- **Goal** | test whether the flattened (free-Q) and spherical (Q=1) solutions are connected:
  profile of the penalty over Q with (gh, rh, rho0) re-optimised in the whole `bounds_original`
  box (the narrow PCA box of the free-Q search is not used). New driver `py/run_q_profile.py`,
  `launch_tycho.sh --q-profile`; harness, storage, production and CONTRACT unchanged.
- **CRN** | Q22 extended to this diagnostic driver only: `agama.setRandomSeed(42)` before every
  evaluation (42 = AGAMA's start seed = the default stream of a fresh process, so earlier
  single/multi-model rows are seed-42 realisations). Justification: Q29 showed `A_repeat ≡ A` and
  only small B_native−B shifts at a fixed seed; the seed-42 rho0 sweeps of the Q=1 probe rounds are
  smooth far below the realisation scatter. The smoothness cut at the lowest Q is only logged.
- **Grid** | lowest Q = Q of the Q29 seed-42 L24 free-Q point (taken from the priors), then
  0.30, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0 (0.30 added: the compact branch is narrow in Q in the old
  L4 rows). Whole grid in one run.
- **Starts / budget (variant A)** | two continuation branches over the whole grid (`compact` up
  from the lowest Q, `extended` down from Q=1, each starting from the neighbour's best), 10
  evaluations per branch per Q, 6 Sobol points per Q, a `global` branch (+10) only for a far good
  Sobol point ⇒ 208 evaluations (max 288), ~31 h on tycho. Not the search's "40 iterations": that
  budget would leave 2–3 steps per branch per Q.
- **Priors** | known seed-42 rows enter the GP without recomputation: the Q29 B_native L24
  points and the Q=1 probe rounds 1–4 (N23/l4; at Q=1 the l4→l24 shift is small); duplicates keep
  l24. File: private `results/logs/qprofile/qprofile_prior_points.txt`.
- **Storage** | rows only in `qprofile_*` files (no pool glob) — they do not enter the PCA. Orbit
  libraries of evaluations with penalty ≤ 3.5 and Q ≠ 1 are saved after the evaluation into the
  shared `orblib/` under the standard name (never overwriting), so a later `--reuse-orblib` search
  hitting such a point does not integrate it again (its row then is the seed-42 realisation).
  Q=1 libraries are not saved (untagged names would mix L24 with N23/l4).
- **Re-evaluation with other seeds** | not planned now; decided after the run from the size of
  any barrier/branch crossing relative to the known two-realisation scatter.
- Mocked tests only (`tests/test_q_profile.py`, `test_launch_tycho.py`); the run is done by the user.

## 2026-10-31 — Q profile: dense grid beyond Q=1, rejected points in the GP, budget extension (user)

- **Grid** | next profile at Q = 0.28, 0.29 (compact wall), 0.95 (spherical wall), 1.1, 1.2
  (prolate side) with the same starts. `--q-grid` no longer has to end at 1: the extended branch
  starts at the grid Q closest to 1 (here 0.95) and is continued down, then up; compact goes up
  from the lowest Q as before (at 0.95 it starts from the best at 0.29 — accepted; a warm start
  from the previous profile was declined). Default grid behaviour is unchanged.
- **Smoothness check** | off for this run (`--no-smooth-check`): CRN smoothness is already
  measured; the 4 points would take 40 % of the compact budget at the wall.
- **Rejected points** | a completed evaluation with a finite penalty ≥ `FAILED_PENALTY` (1e5;
  not the 1e6 sentinel) is `rejected`: it enters the GP at its Q capped at 1e5 so cusp regions
  are not proposed again (the 2026-10-29 run repeated gh = 0.64 at Q=1), is never a branch best
  or baseline, and does not count towards `--max-failures`. Legacy `failed` records with such a
  penalty in the error text are treated alike. The penalty threshold itself is unchanged.
- **Budget extension** | on resume larger `--local-budget/--global-budget` continue every branch
  stopped by budget (trust region kept); budgets omitted ⇒ stored; smaller ⇒ refused. A
  per-Q filter was declined.
- Mocked tests only; the run is started by the user on tycho.

## 2026-11-03 — Resolve local parameters before varying the IC seed (user)

- Keep AGAMA IC seed 42 for the next parameter investigation; defer additional seed scans
  until the local parameter regions have been explored at that seed.
- Two independent local searches around the compact and near-spherical candidates are under
  discussion. Their proposed windows and budgets are in the private run notes; this is not
  approval to launch them or alter the frozen physical bounds.
- Any such search must reset the IC seed before each evaluation and use observations with
  matching seed/protocol provenance. The ordinary PCA search advances AGAMA's RNG and pools
  historical realisations, so its existing launch command does not meet this requirement.

## 2026-11-03 — Local 4D-PCA search at seed 42, near-spherical box (user)

- **Driver** | new `py/run_local_pca.py` + `launch_tycho.sh --local-pca` (harness §3j), not flags in
  the harness's `run_pca_optimization` (pool/reservation/prior logic, no per-evaluation seed).
  Evaluation path shared with the Q profile (`build_evaluator(prefix=…)`): `setRandomSeed(42)`
  before every evaluation, `exp` protocol, `.npz` ≤ 3.5 into the shared `orblib/`.
- **PCA 4D** | all four components of (Q, gh, log rh, log rho0), built as in the harness (best 30 %,
  weights exp(−ΔP/0.1), std floor 0.05 × box width), refit every 6 evaluations; the local box is
  imposed by linear constraints in PC space. GP on −log P with learned noise (as the Q profile,
  robust to outlier priors), not the harness's fixed noise on −P.
- **First run** | box Q 0.90–1.05, gh 0–0.20, rh 10–30, rho0 15–60; 60 new evaluations = 8 Sobol +
  52 TuRBO (one trust region, restart on collapse); the 30 in-box seed-42 points of the two Q
  profiles enter without recomputation (same harness SHA256 and evaluation context, checked at
  dry-run and start). Preflight runs a BoTorch self-test of the constrained proposal.
- Mocked tests only; the run on tycho is started by the user. The compact-region search will use
  the same driver with another `--box` (not decided yet).

## 2026-11-04 — Compact local search with explicit boundary starts (user)

- Prepare the compact-region analogue with the same 4D PCA, per-evaluation AGAMA seed 42,
  L24/exp evaluator and internal Upsilon search. Keep physical bounds and scientific definitions.
- **Budget:** 60 new evaluations. User selected four explicit gh=0 starts, four dispersed Sobol
  points and 52 TuRBO steps rather than eight Sobol points. The starts probe the lower-Q side;
  their other halo coordinates come from the compatible compact candidate. A fixed-coordinate
  slice is not a minimised Q profile.
- **Inputs:** only provenance-checked Q-profile observations are reused without recalculation.
  Exact compact box, coordinates and source results are private:
  `results/logs/localpca/HANDOFF_LOCALPCA_COMPACT.md` and `compact_initial_points.txt`.
- **Implementation:** optional `--initial-points` in `run_local_pca.py` and `launch_tycho.sh`;
  coordinates are evaluated before Sobol and count towards the budget. Ordered starts and file
  SHA256 are checkpointed; recorded starts are not repeated on resume. Old states without
  these fields remain compatible when no file is supplied.
- Preparation/tests only, no agent-run optimisation or tycho operations. The user launches after
  delivery and successful preflight, separately from the near-spherical run. Other IC seeds and
  J-factor work remain deferred until local parameter searches are assessed.

## 2026-11-05 — Weak-cusp local search (user)

- Same driver/evaluator as the two core searches (4D PCA, AGAMA seed 42 per evaluation,
  L24/exp, internal Upsilon); no code change. Box limited in Q to the sampled weak-cusp side
  (user chose the narrower Q range); physical bounds and scientific definitions unchanged.
- **Budget:** first stage of 24 new evaluations = four explicit fixed-shape Q starts at the
  best weak-cusp coordinates, four Sobol points, 16 TuRBO steps; extension by `--budget` on
  resume only after inspecting the result (edge hits, late improvement).
- Only provenance-checked Q-profile observations are reused as priors. Exact box, starts and
  numbers are private: `results/logs/localpca/HANDOFF_LOCALPCA_WEAKCUSP.md`,
  `weakcusp_initial_points.txt`. Run on tycho is started by the user after preflight.

## 2026-11-06 — Paired IC-seed scan of the two seed-42 optima on gray (user)

- **Models** | both: the near-spherical (m0) and the compact (m1) local-PCA optima, in one run. The
  compact point is kept because the decisive quantity is the paired gap, and gray may be a
  different AGAMA build, so only points computed in the same run are directly comparable.
  Parameters are taken at full double precision from the search states (private models file).
- **Seeds** | `42,1-20` in this order. Seed 42 first = cross-host control: it must reproduce the
  tycho seed-42 values (bit-identical or within the Q29 technical budget); otherwise gray results
  are interpreted only within this run. Seeds 1–20 share labels with the 2026-10-26 scan.
- **Run** | `launch_tycho.sh --l24 --git-results --threads=4 --models=… --ic-seeds=42,1-20` on gray
  with `TYCHO_MEM_MAX` = 3.5 GiB: ≤ 1/3 of the CPUs and ≤ 1/2 of the RAM (the `--models` default
  of 8 GiB exceeds gray's RAM). `exp`, L24, fresh orbits, no libraries; no code change.
- **Git** | launchers push only `results/` with rebase retries; concurrent tycho/gray runs write
  disjoint timestamped directories. Hand-off with commands and checks (private):
  `results/logs/seed/HANDOFF_SEED_NEARSPH_COMPACT_GRAY.md`. The run is started by the user.

## 2026-11-07 — gray launch without a delegated `cpu` controller (user; run completed)

- gray delegates only `memory pids` to the user manager and its systemd rejects `OOMPolicy` for
  scopes. Instead of an admin change, the seed scan ran inside an outer `systemd-run --user --scope`
  with MemoryMax / MemoryHigh / MemorySwapMax=0 and `taskset` to four distinct physical cores, with
  `launch_tycho.sh --no-scope --threads=4 --git-results`. Memory limit and swap ban were verified in the
  scope (`memory.oom.group=0`); CPU stays ≤ 1/3 by affinity. No code change.
- Seed-42 control failed the bit-level criterion (different build, different `GEOM_HASH`), but the IC
  stream is the same; gray results combine with tycho at the 0.01 level only. Numbers: private run NOTES.

## 2026-11-06 — Fixed-gh=1 (r^-1 inner cusp) search at seed 42 (user)

- **Why** | the cusp audit found no optimised gh≈1 model (only Sobol probes), so an exclusion
  statement needs an optimised fixed-gh=1 family. Frozen alpha=2: this is an r^-1 inner cusp in our
  family, not literal NFW (alpha=1 would need PI approval and a harness change).
- **Code** | user chose exact gh=1 over a narrow gh slab: `run_local_pca.py` accepts `NAME=v:v` in
  `--box` and searches the remaining free parameters (≥ 2) with a k-dimensional PCA/GP/TuRBO
  (§3j). All-free boxes are bit-identical; harness, evaluator, bounds and CONTRACT unchanged.
- **Run** | one run Q 0.05–1.3, gh=1, physical rh 0.5–30 and rho0 10–120; L24/exp, seed 42, internal
  Upsilon; 40 new evaluations = 6 explicit starts across Q (rho0 matched to the inner density of good fits) + 6 Sobol +
  28 TuRBO; no prior lies on gh=1 (both Q-profile states pinned for the context check). Private
  hand-off and starts: `results/logs/localpca/HANDOFF_LOCALPCA_GH1.md`, `gh1_initial_points.txt`.
  Started by the user on tycho after preflight; extension or a second-basin run decided afterwards.

## 2026-11-07 — AGAMA on fornax-dwarf: native install in ~gala, no Docker (user)

- **Host** | fornax-dwarf: Debian 13, AMD Ryzen 9 7900X (Zen 4, AVX-512), 24 threads, 61 GiB, NVMe RAID1;
  sudo needs a password, user manager delegates `cpu memory pids`, linger enabled without sudo.
- **Why no Docker** | `agama_ready.tar.gz` is a `docker save` of `agama:latest` (Ubuntu 24.04, Python 3.12,
  upstream master, `-march=native` on an unknown Intel CPU). Bit-level parity with the VM is not required
  (0.01 level suffices, as gray vs tycho); a fresh `-march=native` build on Zen 4 is safer and faster; the VM
  launchers end with `sudo shutdown` and rely on the dead rclone path. Native `launch_tycho.sh` covers the host.
- **Why not system Python** | Debian 3.13 lacks pip/ensurepip/`Python.h` without sudo; user chose Miniforge
  `~/miniforge3`, env `agamaBoTorch`, Python 3.12 (as VM/tycho). Only python+pip from conda, the rest via pip.
- **Order** | Python stack first so numpy is final (torch 2.12.0+cpu, botorch 0.18.1, gpytorch 1.15.2,
  linear_operator 0.6.1, scikit-learn 1.9.0, numpy 2.5.4) → cvxopt 1.3.2 wheel + `cvxopt.h`/`blas_redefines.h`
  from the 1.3.2 sdist → GSL 2.7, GLPK 5.0 static `-fPIC` → Eigen 3.4.0 headers → UNSIO fork (`-fPIC`), all in
  `~/Agama/extras` → `pip install --no-build-isolation --build-option=--yes .`; setup.py generated
  `Makefile.local` with `-march=native -fopenmp` and HAVE_PYTHON/EIGEN/CVXOPT/GLPK/UNSIO. AGAMA 1.0.159.
  Four transitive deps released < 7 days ago were pinned back (astropy-iers-data, python-dateutil 2.9.0.post0,
  threadpoolctl 3.7.0, mypy_extensions 1.1.0); full pins in `~/agama-install/requirements.lock`.
- **Checks** | `ldd` clean (static GSL/GLPK/UNSIO), `check_agama.py`/`check_imports.py` OK (run from outside
  the repo root, else the bare `./agama.so` shadows the package), `py/alltest` 31 OK / 1 SKIPPED (amuse),
  mocked harness tests 213 passed. Protocol, lock files and logs: `~/agama-install/` on the host.
- **Open** | preflight and a seed-42 cross-host control are run by the user; no GitHub key for private
  `results/` on the host yet (deploy key or `--offline`).

## 2026-11-07 — Free halo alpha in the near-spherical local search, PCA 5 → 4 (user)

- **Parameter** | Our frozen `alphah = 2` is the Zhao/Spheroid transition sharpness — Hayashi's alpha
  (Hayashi, Chiba & Ishiyama 2020, ApJ 904, 45; Hayashi et al. 2023, ApJ 953, 185; Hayashi et al. 2026,
  PASJ 78, 745), `beta_trans` in Wardana, Chiba & Hayashi 2025 (ApJ 982, 167). User chose Hayashi's
  flat-prior range **0.5 ≤ alpha ≤ 3** (not 0–3: AGAMA needs alpha > 0).
- **Scope** | Experimental opt-in only (`run_local_pca.py --fit-alpha`, `launch_tycho.sh`), harness §3k.
  CONTRACT, production, `alphah = 2` default and `bounds_original` unchanged; proposing alpha for
  production would need the PI.
- **Search** | Previous near-spherical box Q 0.90–1.05, gh 0–0.20, rh 10–30, rho0 15–60 plus alpha 0.5–3;
  incl=90, L24, seed 42 per evaluation, internal Upsilon. PCA of five parameters cut to **four**
  components (user's choice over 4 PCA + separate alpha); dropped component pinned at the best point
  of each refit; stop if alpha's loading on the kept components < 0.05.
- **Budget** | 80 new evaluations = 4 starts (best near-spherical point, alpha 2/0.5/1/3; alpha=2 is the
  control) + 16 Sobol + 60 TuRBO. Old alpha=2 penalties are not imported. Private starts:
  `results/logs/localpca/alpha_initial_points.txt`, hand-off `HANDOFF_LOCALPCA_ALPHA.md`.
- **Libraries** | alpha enters the hashed key with a schema marker and the `.npz` metadata; legacy keys
  and files unchanged. Mocked tests only; the run is started by the user after preflight.

## 2026-11-07 — fornax-dwarf: project times in MSK, run limits (user)

- **Time zone** | fornax-dwarf runs in system UTC (NTP-synced); other hosts use MSK. Git is unaffected
  (commits store epoch + offset), but run-directory names (`date +%Y%m%d_%H%M%S`), log lines, history
  `# Time:`/`# Start` headers and `datetime.now()` columns are zone-less local time. User chose
  `export TZ=Europe/Moscow` for user `gala` only (`~/.bashrc`, `~/.profile`); system zone and other users
  unchanged, no sudo. Convention: all project times are MSK.
- **Limits** | the free-alpha run uses at most 8 of 12 physical cores and ≤ 48 GB RAM:
  `taskset -c 0-7,12-19` (cores 0–7 with SMT siblings) + `--threads=16` (CPUQuota 1600 %) and
  `--memory-gb=44` (44 GiB ≈ 47.2 GB, swap 0). conda is not on PATH there: activate via
  `source ~/miniforge3/etc/profile.d/conda.sh && conda activate agamaBoTorch` before the launcher.

## 2026-11-08 — Q31: prepare diagnostic rh profiles at beta=3,4,7,10 (user)

- **Request** | "Подготовь диагностические профили по rh для β=3,4,7,10".
  Authorises experimental beta support and preparation, not a long AGAMA run,
  a production change, an aperture change or a new J-factor definition.
- **Implementation** | `run_rh_profile.py`, `launch_tycho.sh --rh-profile`;
  harness §3l. Explicit beta in density evaluation, library key/metadata and
  history. Even beta=3 is isolated from legacy libraries; defaults stay frozen.
- **Initial diagnostic plan** | rh=0.5,1,2,5,10,20,30 within existing bounds;
  nuisance Q/gh/rho0 use the full physical box and alpha [0.5,3].
  Four free dimensions per fixed (beta,rh), full PCA 4→4; one TuRBO/GP per cell.
  Incl=90, L24, seed-42 CRN and the existing Upsilon protocol.
- **Budget/continuation** | Up to 24 evaluations per cell including 8 Sobol,
  supplied initial coordinates and one neighbour continuation; max 672.
  Starts supply coordinates only. Per-cell checkpoints; increasing the budget
  extends every cell without replacing its original continuation coordinates.
  Finite-budget minima remain upper estimates; boundary hits are explicit.
- **Delivery** | Private starts and instructions in
  `results/logs/rhprofile/{rh_beta_initial_points.txt,HANDOFF_RH_BETA.md}`.
  Existing fornax-dwarf limits: 8 physical cores/16 SMT threads, 44 GiB, swap 0.
  A synthetic full-4D BoTorch self-test passed on that host from a temporary
  code copy, without importing/evaluating AGAMA; host repository untouched.
- **Remaining** | Radial J decomposition and inclusion of outer tracers were
  not requested in this preparation task; tracked separately in Q32.

## 2026-11-10 — Local refinement of the extended family at fixed rh, stage A (user)

- **Request** | "создай файл стартов и инструкцию … Этап B пока не надо" after the completed
  Q31 rh/beta profiles left most large-rh cell winners at their inherited starts.
- **Design** | Existing `run_rh_profile.py` / `launch_tycho.sh --rh-profile`, no code change.
  beta=3,10; rh=30,20,10,5 (continuation from larger rh); local search box
  Q 0.93–1.05, gh 0–0.05, rho0 25–70, alpha 1.5–3 (optimisation aid, physical bounds unchanged).
  20 evaluations/cell: 2 starts + ≤1 continuation + 4 Sobol + TuRBO; ≤160 total; `.npz` only
  at penalty ≤ 1.25. Seed-42 CRN, L24, incl=90, apertures and Upsilon protocol unchanged.
- **rh stays fixed per cell** | A free local rh drifts to the upper edge of its window
  (the free-rh [10,30] alpha search ended at 29.93), and PCA 5→4 would pin a data-dependent
  direction at the anchor. Full 4D PCA per fixed (beta, rh) keeps the profile well defined.
- **Deferred** | Stage B (compact family at rh=2,5); beta=4,7 only if beta=3 and 10 disagree;
  paired IC-seed check of refined optima if the profile is flat at the 0.001–0.01 level.
- **Delivery** | Private `results/logs/rhprofile/{rh_local_extended_starts.txt,HANDOFF_RH_LOCAL_A.md}`.

## 2026-11-10 — Inclination fan: shared orbit basis, scope and driver (user)

- **Question asked** | Can the same orbit libraries be reused for other inclinations if the
  change of the stellar shape turns out to be unimportant?
- **Finding (code, not opinion)** | Stored `.npz` cannot be reused at another inclination at
  all: `TargetLOSVD::addPoint` applies the `(alpha,beta,gamma)` rotation during integration
  and `finalizeDatacube` folds the cube into apertures, so the projection lives inside
  `matrix_kinem`, and trajectories are not recorded. Reuse can only mean sharing everything
  before the projection (potential, ICs, integration: `orbit_s ≈ 400 s` vs `solveOpt ≈ 11 s`),
  i.e. one `agama.orbit` call with one LOSVD target per inclination.
- **Refinement of the premise** | The shape question is narrower than it looked: the
  observations do not depend on `incl` (doc 07 §2), and the aperture masses and
  `DensityDataset` constraints are rebuilt from the exact `axRZst(i)` of each member for
  free. Only E1 (stellar quadrupole in the potential) and E2 (IC sampling density, a basis
  choice) stay approximate; the reference member remains exact.
- **User answers** | Range 70–90° only; the purpose is numbers for the article; nothing is to
  be run yet beyond the free diagnostics; the code goes into a new driver `py/run_incl_fan.py`.
- **Delivered** | `py/run_incl_fan.py` (`--dry-run`/`--summarize` stdlib-only, `--diagnose`
  without orbits, `--benchmark`, `--fan`) and `tests/test_incl_fan.py` (22 tests, no AGAMA).
  Production, `CONTRACT` and the harness script are untouched; fan rows go to
  `incl_fan_*.txt`, matched by no pool glob. Harness runbook §3m.
- **Blocked** | Any fan run meant to produce article numbers waits for Q33 (A: screening only,
  final numbers recomputed exactly; B: published with a measured E1+E2 systematic; C: no fan).

## 2026-11-11 — Devin sessions continue on fornax-dwarf; plans and summaries in results/doc (user)

- **Request** | Move the non-git Devin hand-off artefacts into the private `results` repo
  (new folder `doc`) and continue the same dialogues on fornax-dwarf.
- **In git (private results/doc)** | `plans/` (project plans from `~/.devin/plans`), `summaries/` and
  `summaries_compaction/` (from `~/.local/share/devin/{,cli/}summaries`), `SESSIONS.md` index and the
  read-only generator `export_sessions.py`. Moved, not copied: originals removed after a byte comparison.
  Personal sessions (non-project working directory or an explicit id list) are excluded.
- **Not in git** | `sessions.db` (2.5 GB): consistent `sqlite3` backup, `working_directory`
  `/home/wind/Agama` → `/home/gala/Agama` in the copy only, zstd (154 MB) + sha256-checked transfer to
  `~gala/.local/share/devin/cli/` (mode 600, dirs 700). Integrity and row counts equal to the source.
  Server permissions `config.json` with `/home/wind` → `/home/gala`. Devin versions equal (3.12.43).
- **Rule** | One authoritative host per dialogue: from now on fornax-dwarf. The local DB is an
  archive; two copies diverge and cannot be merged. The copy also contains the user's personal
  sessions (dir mode 700 on a shared host); delete them with `/rm-session` if undesired.

## 2026-11-11 — Stage-A rh refinement analysed (user)

- **Outcome** | Run `rhprofile_…_20261110_131919` (fornax-dwarf, 160 evaluations, exit 0): shared
  coordinates reproduce penalty/Upsilon bit-for-bit; the transplanted free-alpha finalist stays best at
  rh≥10 for beta=3 and 10; both branches still decrease to the rh=30 edge. Only beta=10, rh=5 moved
  (alpha at its upper bound, not converged). No bound, aperture or production change follows.
- **Next** | Stage B stays deferred. Recommended follow-ups are recorded in STATUS (Q32-A J along the
  profile; optional paired seeds and a Q scan near Q≈0.99); none is started without a user request.
