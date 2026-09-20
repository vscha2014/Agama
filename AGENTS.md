# AGENTS.md — rules for AI agents working in this repository

## 1. Project mission

Schwarzschild orbit-superposition modelling of the Fornax dSph with AGAMA/forstand.
We search for dark-matter halo parameters (`axRZ`/`Q`, `gammah`/`gh`, `rhalo`/`rh`,
`rho0`) and visible-component parameters (`incl`, `Upsilon`) that best match
observed stellar kinematics (minimal `penalty` over GH moments), then derive the
J-factor and its uncertainty. Production runs happen on a Yandex Cloud VM via
Docker; this repo holds the code, input data, and result-file examples.

## 2. Read this first

- `README` — upstream AGAMA overview.
- `Devin_instructions.txt` — the PI's task description (Russian).
- `doc/ai/00_project_context.md` — goals, compute scheme, scientific contract.
- `doc/ai/01_repository_map.md` — what lives where; what is off-limits.
- `doc/ai/02_agama_reference_digest.md` — AGAMA concepts used here (digest of `doc/reference.pdf`).
- `doc/ai/03_current_pipeline.md` — end-to-end pipeline description.
- `doc/ai/04_result_file_formats.md` — result-file formats.
- `doc/ai/05_development_environment.md` — the `.venv-ai` dev environment.
- `doc/ai/questions_for_pi.md` — open questions; add new ones here, never guess.
- `doc/ai/skills/` — short skill notes.

## 3. Think before coding

- No hidden assumptions. If you assume something, write it down explicitly
  (in the task notes or `doc/ai/`).
- If several interpretations of a requirement or file format exist, list them
  explicitly; pick one only with justification, or ask.
- If something is unclear and the answer changes the design — **stop and ask**
  (add to `doc/ai/questions_for_pi.md`).
- Before starting a task, write a short plan (steps + expected outcome).
- Every step must have a verification criterion: how will you know it worked?

## 4. Simplicity first

- Minimum code that solves the task.
- No speculative features ("might be useful later").
- No abstractions for one-off tasks.
- No configurability that nobody requested.
- If a solution fits in 50 lines, do not write 200.

## 5. Surgical changes

- Change only what the task requires.
- Do not refactor neighboring code "while you're there".
- Do not reformat large files (no mass whitespace/style diffs).
- Do not delete unrelated dead code.
- Clean up only after your own changes (temp files, debug prints you added).

## 6. Goal-driven execution

- Task → explicit success criteria, written before implementation.
- Bug → reproduce with a test or minimal repro first, then fix.
- Change → verification (test, dry-run, diff inspection) before declaring done.
- Multi-step tasks: execute with a verification loop — verify each step before
  building the next on top of it.

## 7. Scientific contracts (do not change without PI approval)

- `penalty` definition and computation path (orbit-weight solve, GH dataset term).
- Aperture construction (sectors, binning, symmetrization, `max_r`).
- Gauss–Hermite conventions (`ghorder`, `degree`, moment indices, bootstrap errors).
- Brent (bounded scalar) search over `Upsilon` and its bounds/tolerances.
- J-factor formula and integration settings.
- Units and scaling conventions (`vscale`, distance, position angle).
  Confirmed geometry (Wang et al. 2019, Sersic; PI-approved correction):
  `posang = 42.3`, `q_ap = 1 − 0.31 = 0.69`, `D = 143 kpc`. (Old, wrong:
  `posang = 46.8`, `q_ap = 0.7`.)
- Input observational data (`py/table3.dat`) — frozen.
- Sersic assumptions (`Sersic_m = 0.80 ± 0.006`, `massSt = 14.0` frozen,
  scale radius 16.4′ ± 0.2′, flattening deprojection `axRZst`). Sersic-error
  propagation (Goal 3) is a **separate** future code, not edits to the main script.
- Production parameter bounds (`bounds_original` in the main script).
- RNG policy: torch (BoTorch) gets a per-process seed (logged); a per-process
  `proc_rng` decorrelates optimizer seeding (initial-point selection, Goal 0);
  the GH observation-error bootstrap is **fixed at `seed=42`** (identical across
  workers). None of these affect AGAMA orbit-IC sampling (`Density.sample` uses
  AGAMA's own RNG, `agama.setRandomSeed` not called → orbit libraries identical
  across processes). Do not re-pin a global numpy seed.
- Result-file naming: canonical = `4UpsBoTorch_PCA_Sersic_*` /
  `4result_BoTorch_PCA_Sersic_*`; wrong-posang archive = `*_PA46.8_*`. The
  canonical glob does not match the archive. Use **all** result files; **never
  dedup across files/hosts** (unique experiment results).

An agent **may recommend** extending parameter bounds (or other contract changes)
in documentation/analysis reports, but must **not** change them in production
code without explicit approval.

## 8. Expensive execution policy

Forbidden without an explicit user request:

- production optimization runs (`Fornax_P21_symm_PCA_w3Sersic_yaVM.py` full run);
- full Docker/Yandex VM launches (`launch_docker_parallel.sh`);
- long BoTorch/TuRBO runs;
- orbit-library regeneration;
- mass rewriting of result files.

Allowed without asking:

- static inspection (grep, AST, reading code/data);
- reading any repo files;
- small dry-runs **if the script already supports them** (do not add dry-run
  modes just to run something);
- `--help` invocations;
- `python -m py_compile`;
- unit/simulation tests that do not trigger production computation.

## 9. Generated data policy

- Never commit `.venv-ai/` (or any venv).
- Never commit extracted full text of `doc/reference.pdf` (temporary extraction
  only under gitignored `doc/ai/_tmp/`).
- Never commit logs, results, checkpoints (`*.pkl`), sqlite files, run artifacts.
- Never delete the committed result-file examples (`py/4Ups*`, `py/4result*`).
- New generated artifacts go to a gitignored path (extend `.gitignore`
  minimally and with justification).

## 10. Yandex shared-run constraint

Hard constraint for any future harness/orchestration design:

- Parallel processes **must see each other's results as they appear** (live,
  during the run) — today this works via shared workspace files + Yandex.Disk
  sync read at PCA-update time.
- Do not design a future harness where results become visible only after the
  whole run finishes (no buffering results until completion).
- Future duplicate-prevention must be **preventive** (e.g., reserving candidate
  points before evaluation), not post-hoc cleanup of duplicated computations.

## 11. Experimental fixed-Q launch and lightweight verification

- `py/launch_orblib_exp.sh --Q1 --incl=90.0 --no-shutdown` selects fixed `Q=1`
  with ordinary doubling (`d1`); `--Q1 --no-double` is rejected. `incl` remains
  fixed per launch, not optimized. Defaults remain `n_bin=250`, GH/Sersic IDs 0.
- Use uppercase `Q` in halo-shape tags; the PI reserves lowercase `q` for the
  intrinsic flattening of the stellar profile in the article. The CLI flag is `--Q1`.
- Fixed-Q writes use `Q1d1_nb250_gh0_ser0`; free-Q writes retain `d1_nb250_gh0_ser0`.
  Both modes read both histories with matching d/nb/gh/ser settings, including
  per-process and merged files on all hosts. Fixed-Q readers additionally filter
  rows to Q=1; free-Q readers accept all Q. Do not deduplicate results across files.
  Cloud history is refreshed in both modes; local host files of both modes are
  protected from overwrites. Logs/checkpoints and output writers remain separate.
  External prior parameters are projected to Q=1 and their penalty is recomputed;
  their old penalties are not training observations. Orbit cache keys retain Q
  in the parameter hash, so compatible Q=1 libraries remain reusable.
- PCA exponential weights use `exp(-(penalty - min(penalty))/0.1)` in all three
  construction/update paths. This preserves normalized weights while preventing
  all-zero underflow on cold-start penalties above 100.
- Before initial PCA, Q1 seeding searches current-inclination history, falling
  back to the nearest inclination. Each source (compatible out, canonical 4Ups,
  opt-in PA46.8) contributes at most 24 nearest-Q candidates; old penalties rank
  candidates only within their source, with rank-biased per-process RNG ordering.
  At most 10 evaluations per process are attempted; existing target-Q1 points
  and repeated projected candidates are skipped before calculation, not deleted
  from history. Reservations use the target Q1 parameters and are always released.
  Shared history is refreshed every 4 candidate visits and at the end; workers
  can stop early once the shared history has 10 more valid points than at entry.
  Having 10 points with penalty<10 skips this optional stage, but achieving that
  threshold is NEVER required. If fewer than 10 valid evaluations remain, one
  bounded LHS batch fills the deficit; persistent failures produce an explicit
  insufficient-data error. Existing checkpoints skip nearest-Q seeding on resume;
  the old post-PCA prior stage is disabled for Q1, unchanged for free Q.
- Q1 checkpoints include physical parameters for reprojection into the PCA basis
  rebuilt at resume. History is shared live, including merged files from other
  hosts. Existing J-factor filename parsing does not yet accept `Q1d1_*` tags;
  extending that post-processing is a separate task.
- Safe tests (AST-extracted functions, NumPy PCA substitute, mocked Docker/rclone;
  no AGAMA/BoTorch runs): from `tests/`, run
  `../.venv-ai/bin/python -m pytest -q test_orblib_q1.py --rootdir=. --import-mode=importlib -p no:cacheprovider`.
  Do not use `python -m pytest` from the repo root: local `py/` shadows pytest's
  `py` compatibility module. Syntax checks from the root:
  `python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py tests/test_orblib_q1.py`
  and `bash -n py/launch_orblib_exp.sh`. Launcher `--help` is side-effect-free.

## 12. Streaming orbit-library storage (single VM)

- The experimental launcher now passes `--stream-orblib`. This supersedes the
  legacy tar-download/final-snapshot flow still present as unused functions in
  the launcher. New libraries are individual objects under `galAgama/orblib/objects/`,
  with verified receipts under `catalog/`. Do not run the legacy tar uploader
  concurrently with the streaming controller.
- `py/orblib_storage.py` uses only the Python standard library on the host.
  SQLite state, file locks, stop/finish markers and receipts live under the
  already ignored `py/orblib/.storage/`. One launcher/controller per workspace;
  multi-VM publication coordination is not implemented.
- Existing tar archives require an explicit one-time `index` operation before
  `prepare`. The indexer streams each tar, reuses matching local members or
  stages at most one missing member, validates ZIP contents and the whole tar
  MD5/size, and publishes a small index.
  This is a real cloud operation and must NOT be run during development.
  Production invocation from the workspace: `python3 orblib_storage.py index
  --root ./orblib --timeout 7200` (supply `--remote`/`--config` if non-default).
  Subsequent launches download metadata only, not old matrices.
- Each new managed .npz is removed locally only after the final remote object
  and its manifest pass size/MD5 checks. Existing untracked local libraries
  are uploaded/verified but NOT automatically deleted. No penalty filtering.
  A failed transfer keeps the sole local copy and durable queue.
  For a separately authorized transition cleanup, `prune-verified --root ./orblib`
  lists verified old local copies without deletion; adding `--apply` deletes only
  those copies after fresh remote verification. Never run this against real data
  without explicit permission for that deletion.
- Defaults: `ORBLIB_UPLOAD_ATTEMPTS=3`, `ORBLIB_FILE_TIMEOUT=900` seconds for the
  entire per-file attempt, `ORBLIB_RESERVE_BYTES=2000000000` for checkpoints/logs
  and metadata. Workers additionally reserve the estimated uncompressed library
  size plus ZIP overhead, sharing the budget across all processes.
- Exhausted delivery attempts set STOP: no new models; already running models
  finish and write local checkpoints without mandatory cloud sync. The launcher
  waits for workers, then shuts down even with undelivered files. No unbounded
  final-upload retry loop. `--no-shutdown` suppresses shutdown only. The VM disk
  must survive power-off; never delete a VM/disk containing undelivered state.
- `--resume` first drains the saved queue; persistent delivery failure powers off
  without starting new workers. Initial-stage checkpoints rebuild seeding from
  saved history; TuRBO checkpoints contain physical observations in both Q modes
  and reproject into the rebuilt PCA basis. Old free-Q checkpoints without
  physical parameters are rejected in streaming mode rather than interpreting
  stale PCA coordinates. Interrupted checkpoints cannot silently be overwritten
  by a clean launch; a completed checkpoint has a checksum-bound completion marker.
- Completed-point prevention reads compatible out-history before expensive
  evaluation and rechecks after claiming the library. No history rows are deleted
  or deduplicated. New rows carry a storage-context marker for observation/numerical
  settings. Untagged legacy out-history retains the existing experiment/inclination
  compatibility assumption; changed legacy science settings require review.
  Metadata/point comparisons allow only 1e-12 relative/absolute serialization
  roundoff, not the much wider six-significant-digit filename quantization.
- Safe checks from `tests/`: `../.venv-ai/bin/python -m pytest -q test_orblib_q1.py
  test_orblib_storage.py --rootdir=. --import-mode=importlib -p no:cacheprovider`.
  Tests use small arrays and mocked rclone/Docker/shutdown, never AGAMA or cloud.
  Also compile the modified Python files, run `bash -n py/launch_orblib_exp.sh`,
  and `git diff --check`. Helper and launcher `--help` are side-effect-free.
