# Runbook — production pipeline (Yandex VM)

Files (frozen, PI approval required for any edit):
`py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py`, `py/J_factor_Sersic_Fornax_P21_symm.py`,
`py/launch_docker_parallel.sh`. Scientific definitions: `../CONTRACT.md`.

## Flow

```
table3.dat + Sersic params
  → apertures / GH moments (+ bootstrap errors, seed 42)
  → halo Density + Multipole potential + 100k-orbit library
  → Brent over Upsilon → penalty
  → 4Ups (history) / 4result (log) files, synced to Yandex.Disk
  → J_factor_Sersic_Fornax_P21_symm.py (J tables, histograms, corner plots)
```

## Orchestration

- Yandex Cloud VM (32 vCPU / 32 GB). `launch_docker_parallel.sh` starts parallel
  Docker containers (`p0…pN`, pinned CPU ranges; 8 workers in recent runs) that
  share one `/workspace` bind mount = the VM's `py/` directory.
- Each worker appends to `4UpsBoTorch_PCA_Sersic_<host>_pN.txt` and
  `4result_BoTorch_PCA_Sersic_<host>_pN.txt`.
- Outer loop: TuRBO/BoTorch (`SingleTaskGP`, `qLogNoisyExpectedImprovement`) in a
  3-component PCA space of the 4 halo parameters (`rh`, `rho0` log-scaled),
  `n_iter = 40` per run, PCA rebuilt every `pca_update_interval = 12` iterations
  when ≥5 new points are buffered. Checkpoint every 3 iterations; `--resume`.
- Initial points for a fresh `incl`: optional PA46.8 seeding (penalties
  recomputed) → bootstrap from the nearest other `incl` → LHS; all decorrelated
  per worker through `proc_rng`.

## Hard constraint — live result sharing

`load_fresh_data_from_files()` runs at every PCA rebuild: it pulls fresh files
from Yandex.Disk via rclone and globs all local canonical files including other
workers' `_pN` files (excluding its own). **Workers must keep seeing each
other's results mid-run**; never design a harness that only publishes at the
end. Duplicate prevention must be preventive (reserve a candidate before
evaluating), never post-hoc deletion of computed rows.

Concurrency primitives in use: `flock` on `.upload_lock`, `.done_pN` markers,
per-process `checkpoint_<host>_pN.pkl`. After a container exits, its `_pN` files
are merged into host-level files with an MD5 `HASH:` of the data block
(idempotent per destination file only), uploaded, and the `_pN` files removed.

## Known latent bug (not fixed here)

`DONE_COUNT=$(ls .../.done_* | wc -l)` under `set -euo pipefail` aborts the whole
launcher when **no** container succeeded, skipping merging, uploads and
notifications. Fixed in `launch_orblib_exp.sh`, still present in
`launch_docker_parallel.sh` — Q20, awaiting PI approval to touch production.

## Never run from a dev machine

The launcher mutates shared cloud state and ends with `sudo shutdown`; rclone
syncs overwrite shared history. Production optimisation, orbit-library
regeneration and long BoTorch runs are likewise forbidden without an explicit
user request.
