# Project context

## Scientific goal

Schwarzschild orbit-superposition modelling of the Fornax dSph galaxy using AGAMA
(forstand framework by E. Vasiliev). The code searches for the dark-matter halo
parameters and visible-component parameters that minimize the discrepancy (`penalty`)
between Gauss–Hermite (GH) coefficients of the model line-of-sight velocity
distribution and GH coefficients estimated from observed stellar radial velocities
(`py/table3.dat`, Pace et al. 2021 sample).

For a fixed inclination `incl`, the sky plane is split into apertures (sectors);
GH moments are computed per aperture for data and model. The optimizer searches over
halo parameters:

- `Q` (a.k.a. `axRZ`) — halo z-axis ratio;
- `gh` (a.k.a. `gammah`) — inner slope of the halo Spheroid profile;
- `rh` (a.k.a. `rhalo`) — halo scale radius;
- `rho0` — halo density normalization.

For each parameter set, a Brent (bounded scalar) search finds the best `Upsilon`
(mass scale factor: for a fixed orbit library the gravitational field and velocities
can be rescaled). The end goal: halo parameters (`axRZ`, `gammah`, `rhalo`,
`rho0*Upsilon`) and visible-component parameters (`incl`, `Upsilon`) giving the best
agreement with the data, plus the J-factor (dark-matter annihilation observable) and
its uncertainty.

## Current computational scheme

- Executor: Yandex Cloud VM (32 vCPU / 32 GB RAM), launched via
  `py/launch_docker_parallel.sh`.
- The launcher runs **4 parallel Docker containers** (suffixes `p0..p3`, pinned to
  CPU ranges 0-7/8-15/16-23/24-31), each running
  `py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py --incl <incl> --suffix pN`.
- Each process appends its evaluations to its own files
  `4UpsBoTorch_PCA_Sersic_<host>_pN.txt` (history) and
  `4result_BoTorch_PCA_Sersic_<host>_pN.txt` (log).
- Results are shared **live**: every PCA rebuild calls
  `load_fresh_data_from_files()`, which downloads fresh files from Yandex.Disk
  (rclone) and globs all local `4UpsBoTorch_PCA_Sersic_*.txt` files — including the
  per-process files of the other parallel workers (excluding its own, already in
  memory). So workers see each other's results as they appear.
- The optimization is TuRBO/BoTorch over a PCA-reduced parameter space
  (3 components of the 4 scaled parameters; `rh` and `rho0` log-scaled).
  The PCA model is **rebuilt every 12 iterations** (production
  `pca_update_interval=12`; function default 15) if at least 5 new points are
  buffered, using all fresh data from disk/cloud.
- After each container finishes, its per-process files are merged (with MD5
  dedup of data blocks) into the host-level files
  `4UpsBoTorch_PCA_Sersic_<host>.txt` / `4result_BoTorch_PCA_Sersic_<host>.txt`
  and uploaded to Yandex.Disk; per-process files are then deleted.
- Checkpoint/resume: `checkpoint_<host>_pN.pkl`, `--resume` flag in the launcher.
- Post-processing: `py/J_factor_Sersic_Fornax_P21_symm.py` reads the merged history
  files (from a mounted Yandex.Disk dir), filters by `incl` and penalty percentile,
  computes J-factors for several aperture angles and builds weighted histograms /
  corner plots.

## Current user goals (from `Devin_instructions.txt`)

- Goal 0: check the launch script for duplicate computations between parallel
  processes (same parameters computed twice), especially on a fresh inclination.
- Goal 1: optimization over `incl`; add `incl` as an analysis parameter in the
  statistical/J-factor analysis instead of a per-inclination task.
- Goal 2: assess whether the explored parameter space is sufficient or should be
  extended.
- Goal 3: propagate input-data errors (velocity errors in `table3.dat` → GH-moment
  errors via Monte Carlo; Sersic profile parameter errors from literature) into the
  good-model boundaries and the resulting J-factor bounds.

None of these goals are implemented in Stage-0; Stage-0 only builds documentation
and the dev environment.

## Scientific contract (must not change without explicit PI approval)

- `penalty` definition and computation (`KinemDatasetGH.getPenalty`, the weighting
  in `find_weights_Ups`).
- Aperture construction (sector splitting of the sky plane, `n_bin=250`,
  `max_r=2.1`, the symmetrization of the data sample).
- Gauss–Hermite conventions (`ghorder=6`, `degree=2`, used moment indices
  `ind=(1,2,6,7,8,9)`, Monte-Carlo GH error estimation with `n_boot=100`).
- Brent (bounded `minimize_scalar`) search over `Upsilon` in `[0.1, 1.6]`,
  `xatol=1e-3`, `maxiter=50`.
- J-factor formula and integration settings in `J_factor_Sersic_Fornax_P21_symm.py`.
- Potential/density definitions: stellar Sersic (`Sersic_m=0.80 ± 0.006`,
  `massSt=14.0` frozen, `scaleRst` from 16.4′ ± 0.2′), halo Spheroid
  (`alphah=2.0`, `betah=3`, `outercutoffradius=55.0`, `cutoffstrength=2.5`),
  Multipole potential settings.
- Production parameter bounds (`bounds_original`):
  `Q∈(0.05,2.5)`, `gh∈(0.0,1.6)`, `rh∈(0.5,3.5)`, `rho0∈(34,120)`,
  `Upsilon∈(0.1,1.6)`. PI: ask before proposing changes.
- Units convention (velocity scale `vscale=(2*6.67/3.086)**0.5`, distance
  `D=143 kpc`, **position angle 42.3°**, **`q_ap = 1 − 0.31 = 0.69`**; Wang
  et al. 2019, Sersic). Old wrong values were `posang=46.8`, `q_ap=0.7`.
- Input observational data (`py/table3.dat`).

## Confirmed facts (PI answers, see `questions_for_pi.md`)

- Data line format of `4UpsBoTorch_PCA_Sersic_*.txt`:
  `incl Q gh rh rho0 Upsilon penalty timestamp` (col 0 carries `incl`).
- `penalty` is the GH kinematic term `numpy.sum(penalties[1])`. **It is NOT a
  χ²** — treat it only as a relative ranking score.
- J-factor good-model weighting (in `J_factor_Sersic_Fornax_P21_symm.py`):
  weighted KDE with `w = exp(-(penalty − pen_min) / pen_sigma)`,
  `pen_sigma = max(penalties.std(), 1e-6)`, normalized.
- `incl` admissibility: stellar deprojection
  `axRZst = sqrt(q_ap² − cos² incl)/sin incl` requires **cos(incl) < q_ap = 0.69**
  (⇒ incl ≳ 46.5°). This is the physical prior on `incl` for Goal 1.
- Result files: **use all** files (per-process `_pN` + merged per-host);
  `_pN` are valid mid-run, merged at the end. **Never dedup** across files or
  hosts — unique experiment results. Legacy `4UpsBoTorch_Sersic.txt` has the same
  column format (older code). `4result_*` logs are monitoring-only.
- Wrong-posang results were renamed to `*_PA46.8_*` (archive). New correct-posang
  runs keep canonical names; the canonical glob does not match the archive. The
  archive may seed initial points **only with penalties recomputed**
  (`--init-from-pa468`, opt-in).
- RNG policy: torch per-process seed (logged) + per-process `proc_rng` for
  optimizer seeding (Goal 0); GH observation-error bootstrap fixed at `seed=42`
  (identical across workers). AGAMA orbit-IC sampling uses AGAMA's own RNG
  (`agama.setRandomSeed` not called) → orbit libraries identical across processes.
- PCA-model rebuilt every 12 iterations (production `pca_update_interval=12`;
  function default 15) inside `run_pca_optimization` (`n_iter=40` per run).

## Known unknowns

See `questions_for_pi.md`. Resolved: PA46.8 seeding is opt-in (`--init-from-pa468`);
production `pca_update_interval=12`. Remaining: whether to extend
`bounds_original` (ask PI first).
