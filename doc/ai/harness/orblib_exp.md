# Runbook — experimental harness `orblib_exp`

Files: `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`, `py/launch_orblib_exp.sh`,
`py/orblib_storage.py`, tests in `tests/test_orblib_q1.py`,
`tests/test_orblib_storage.py`. Production scripts are **not** part of this and
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
  all-zero underflow on cold-start penalties above 100.
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
- The J-factor post-processing does not yet parse `Q1d1_*` filenames.

## 3. Streaming orbit-library storage (single VM)

- `--stream-orblib` supersedes the legacy tar download/snapshot functions still
  present in the launcher; never run both at once. New libraries are individual
  objects under `galAgama/orblib/objects/` with receipts under `catalog/`.
- `py/orblib_storage.py` is stdlib-only; SQLite state, locks, stop/finish markers
  and receipts live in the gitignored `py/orblib/.storage/`. One controller per
  workspace; multi-VM publication is not implemented.
- Old tars need a one-time `python3 orblib_storage.py index --root ./orblib
  --timeout 7200` — a real cloud operation, never during development.
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

## 4. Editing rules

- `bounds_original` is defined **twice** in the script — change both or neither.
- Keep diffs surgical; no refactoring of neighbouring code.
- Document new behaviour **here**, not in `AGENTS.md`.

## 5. Verification (safe, no AGAMA, no cloud)

```bash
cd tests && ../.venv-ai/bin/python -m pytest -q test_orblib_q1.py \
    test_orblib_storage.py --rootdir=. --import-mode=importlib -p no:cacheprovider
cd .. && python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py \
    py/orblib_storage.py tests/test_orblib_q1.py
bash -n py/launch_orblib_exp.sh && git diff --check
```

Do not run `python -m pytest` from the repo root: local `py/` shadows pytest's
`py` module. Launcher and helper `--help` are side-effect free.
