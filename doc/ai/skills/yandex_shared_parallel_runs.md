# Yandex shared parallel runs — what to remember

- 4 Docker workers share one `/workspace` (the `py/` dir on the VM) and write
  per-process files `*_p0..p3`; cross-VM sharing goes through Yandex.Disk
  (rclone) directory `galAgama`.
- **Live visibility is a hard requirement**: workers read each other's fresh
  results (local glob + Yandex.Disk download) at every PCA update (production
  `pca_update_interval=12`, `load_fresh_data_from_files`, excluding their own
  file). Any future harness must preserve mid-run visibility — no buffering of
  results until run completion.
- Merging into per-host files is done by `append_and_remove` in
  `launch_docker_parallel.sh` with an MD5 `HASH:` of the data block — idempotent
  per destination file only. **Never dedup across files/hosts** (PI: each row is
  a unique experiment result). Use all canonical files.
- **RNG policy** (see the big comment near line ~429 of the main script):
  - torch (BoTorch) gets a per-process seed (logged) → decorrelated acquisition
    proposals between workers.
  - `proc_rng = default_rng(process_seed)` is a per-process/per-host RNG used for
    OPTIMIZER SEEDING only (initial-point selection) — see Goal 0 below.
  - the GH-moment observation-error bootstrap is **fixed at `seed=42`** so the
    observed-data error realization (and thus penalties) is identical across
    workers.
  - none of these touch AGAMA orbit-IC sampling (`Density.sample` uses AGAMA's
    own RNG via `agama.setRandomSeed`, never called) → orbit libraries are
    identical across processes. Do not re-pin a global numpy seed.
- **PA46.8 seeding** (opt-in `--init-from-pa468`): new correct-posang runs may
  draw candidate parameter points from the wrong-posang archive at the same incl,
  but penalties are **recomputed** with the corrected geometry; archive penalties
  never enter the PCA model (separate `seed_patterns`, not `storage/host`).
- **Goal 0 (duplicate parallel calculations) — investigated + partly fixed:**
  - *Finding:* for a not-yet-explored incl (empty history), the initial-point
    phase was the main duplication source — `select_bootstrap_candidates` is
    fully deterministic and the LHS fallback used `default_rng(seed=42)`, so all
    4 workers (and all hosts) recomputed **identical** initial points.
  - *Fix (implemented):* both bootstrap candidate selection and the LHS fallback
    now use the per-process `proc_rng`. `select_bootstrap_candidates(rng=...)`
    draws a different penalty-weighted subset per worker (overlap dropped from
    100% to ~40% in a 40→12 test), still biased toward the best points. Small
    pools (`len(pool) <= n`) fall back to deterministic selection (unavoidable).
  - *Main loop:* already decorrelated by the per-process torch seed (different
    acquisition proposals); live visibility every 12 iters reduces later overlap.
  - *Still open (recommendation, not built):* there is **no hard reservation** of
    candidate points. For a strict guarantee (incl. cross-host, no shared FS),
    a preventive shared reservation file (reserve-before-evaluate) would be
    needed — must stay live and preventive, not post-hoc cleanup.
- Concurrency primitives in use: `flock` on `.upload_lock` for uploads,
  `.done_pN` marker files, per-process checkpoints `checkpoint_<host>_pN.pkl`.
  Respect them when adding orchestration.
- Do not run the launcher or rclone sync from the dev machine; it shuts down
  the VM (`sudo shutdown`) and mutates shared cloud state.
