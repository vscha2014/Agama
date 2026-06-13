# Current pipeline

All line numbers refer to `py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py` ("the script")
unless noted. This is documentation only; no behavior was changed in Stage-0.

```text
input data (py/table3.dat, Sersic profile params)
  → aperture/GH preprocessing (sectors, datacube, GH moments + MC errors)
  → model/orbit evaluation (halo Density + Multipole potential + orbit library)
  → Brent search over Upsilon (bounded minimize_scalar)
  → penalty (GH kinematic term of the weighted LSQ objective)
  → 4Ups / 4result files (per-process, merged per-host, synced to Yandex.Disk)
  → J-factor analysis (J_factor_Sersic_Fornax_P21_symm.py)
```

## 1. Input data and preprocessing (script lines ~150–500)

- `incl` comes from `--incl` CLI arg (validated in (0, 90]), default 90.0
  (lines ~149–162). The Docker launcher passes `--incl="$INCL"` (default 90.0,
  overridable via `--incl=` argument of `launch_docker_parallel.sh`).
- `table3.dat` loaded (path overridable via env `AGAMA_TABLE3`); velocities
  rescaled by `vscale`; rows filtered by membership probability (col 9 > 0);
  sample point-symmetrized (each star duplicated with x,y,v negated).
- Sky plane split into a central ellipse plus 8 angular sectors with radial bins
  of ~`n_bin=250` stars (probability-weighted), out to `max_r=2.1` → polygon
  apertures `sectAPP`.
- Geometry (Wang et al. 2019, Sersic; corrected): `posang=42.3`, `q_ap=1-0.31=0.69`,
  `D=143 kpc`. `gamma2=(posang-90)*pi/180`; stellar deprojection
  `axRZst=sqrt(q_ap^2-cos^2 incl)/sin incl` (requires `cos incl < q_ap`).
- `agama.Target(type='LOSVD', apertures=sectAPP, gridx, gridy, gridv,
  degree=2, symmetry='t', alpha=0, beta=incl, gamma=f(posang), psf=0.01)`
  produces the observed datacube; `agama.ghMoments(ghorder=6)` gives GH moments.
- RNG policy: torch gets a per-process seed (logged, from OS entropy mixed with
  `hostname_proc`); a per-process `proc_rng` decorrelates initial-point selection
  (Goal 0); the GH observation-error bootstrap below is fixed at `seed=42`
  (identical across workers). AGAMA orbit-IC sampling uses AGAMA's own RNG
  (`agama.setRandomSeed` not called) → orbit libraries identical across processes.
- GH-moment errors via bootstrap: `n_boot=100` resamplings of velocities with
  their reported errors → `schwarzlib.ghMomentsErrors`; moments used:
  `ind=(1,2,6,7,8,9)` (v, sigma, h3..h6 ordering per GHmoments conventions).
- Two datasets are built: `DensityDataset` (Sersic stellar density, 3D density
  constraints, tolerance 0) and `KinemDatasetGH` (GH constraints, tolerance 0.01).

## 2. Model evaluation (`halo_IC_lib_weights_pca_fixed`, lines ~764–908)

For a candidate `(Q, gh, rh, rho0)` (from PCA coords or direct):

1. Build `densityHalo = Density('Spheroid', alpha=2, beta=3, gamma=gh,
   axisratioz=Q, densitynorm=rho0, scaleradius=rh, outercutoffradius=55,
   cutoffstrength=2.5)`.
2. `pot_gal = Potential('Multipole', density=stars+halo, lmax=4, mmax=0)`.
3. Sample `numOrbits=100000` ICs from the stellar density; integrate orbits for
   `100*Tcirc` recording target matrices (`trajsize=1000`).
4. `find_weights_Ups(Upsilon)`: rescale matrices by Υ, solve orbit weights with
   `agama.solveOpt` (regularization `regul=1.0`), compute
   `pen = sum(penalties[1])` — the **kinematic (GH) penalty only**.
5. **Brent over Upsilon**: `minimize_scalar(bounded, [0.1, 1.6], xatol=1e-3,
   maxiter=50)` wrapped in `FunctionLogger` (history written to the 4Ups file).
6. Append result line + Upsilon history to `4UpsBoTorch_PCA_Sersic_<hostproc>.txt`;
   return `-min_pen` as the BoTorch target.

Failures return `-1e6` (logged as penalty 1e6).

## 3. Outer optimization (`run_pca_optimization`, lines ~1930–2503)

- TuRBO-style trust-region BoTorch optimization (`SingleTaskGP`,
  `qLogNoisyExpectedImprovement`) in a PCA space of the 4 halo parameters
  (`n_components=3`, `rh`/`rho0` in log10; bounds expanded ×2.5 in PC space).
- Called from `__main__` with `n_iter=40`, `cutoff_start=2.0`, `resume`
  per CLI flags. Before it, `diagnose_pca_space()` prints data diagnostics.
- **Initial PCA model**: built from all available history points for this `incl`
  (adaptive penalty cutoff keeping the best `target_fraction=0.3`). If too few
  points exist for this inclination, seeding order is:
  1. (opt-in `--init-from-pa468`) `seed_points_from_patterns`: read candidate
     params from PA46.8 archive **at the same incl** and **recompute** penalty
     with the corrected geometry (stale penalties never enter the PCA model);
  2. `bootstrap_initial_points_from_nearest_incl` + `select_bootstrap_candidates`:
     re-evaluate best points from the **nearest other** inclination
     (`find_nearest_incl_data` deliberately excludes the target incl);
  3. LHS random fallback.
  All three sources are decorrelated per worker via the per-process `proc_rng`
  (`select_bootstrap_candidates(rng=...)` draws a penalty-weighted subset; LHS
  uses `proc_rng` instead of a fixed seed) — Goal 0, so parallel workers don't
  recompute identical initial points for an unexplored incl.
- **PCA update every 12 iterations** (production `pca_update_interval=12` passed
  from `__main__`; function default 15; needs ≥5 buffered new points):
  `_update_pca_model` re-reads ALL fresh files (see §4), refits scaler+PCA,
  recomputes PC bounds and re-projects the observation history; TuRBO state is
  updated in place.
- Checkpoint (`checkpoint_<hostproc>.pkl`) saved every 3 iterations; `--resume`
  restores X_obs/Y_obs/TuRBO state.

## 4. How parallel processes see each other's results

- Each of the 4 containers shares the same `/workspace` bind mount (= `py/` dir
  on the VM), writing to files suffixed `_p0.._p3`.
- `load_fresh_data_from_files(storage_patterns, host_patterns, incl_filter,
  exclude_suffix=hostname_proc)` (lines ~1508–1598):
  1. calls `load_from_yadisk(...)` to download fresh result files from
     Yandex.Disk via rclone (results from other VMs/hosts);
  2. globs `4UpsBoTorch_PCA_Sersic_<host>.txt`, `..._<host>_p*.txt` and storage
     patterns `4UpsBoTorch_Sersic.txt`, `4UpsBoTorch_PCA_Sersic_*.txt`;
  3. skips the calling process's own file (`exclude_suffix`), parses data lines
     (7 columns), filters by `incl` (±0.01), validity, penalty < 1e5.
- This happens at PCA-model build/update time — so **cross-process result
  visibility is live during the run** (every 12 iterations), not only at the end.
- After a container exits, `launch_docker_parallel.sh::run_container` merges its
  `_pN` files into the host-level files with an MD5 hash of the data block to
  prevent duplicate appends, uploads merged files to Yandex.Disk (under `flock`),
  and deletes `_pN` files locally and remotely.

## 5. J-factor analysis (`py/J_factor_Sersic_Fornax_P21_symm.py`)

- Reads merged history files from a mounted Yandex.Disk directory
  (`YADISK_DIR=/home/gala/Yandex.Disk/galAgama`, hardcoded; patterns
  `4UpsBoTorch_Sersic.txt`, `4UpsBoTorch_PCA_Sersic_*.txt`).
- Filters by `incl_target` (a single inclination — this is what Goal 1 wants to
  generalize), applies adaptive penalty cutoff (best `target_fraction=0.30`,
  hard ceiling `cutoff_start=0.60`).
- For each good model rebuilds the halo density and integrates ρ² along the line
  of sight over cones θ ∈ {0.1, 0.2, 0.5, 1.0}° (D=143 kpc) → J(θ); writes
  result tables and weighted histograms / corner plots back to YADISK_DIR.

## 6. Parts that are still unclear

- Exact statistical normalization of `penalty` (the `mult = sqrt(num_dof)*10`
  scaling and tolerance weighting make it non-trivially related to χ²) — asked
  in `questions_for_pi.md`.
- Whether duplicate evaluations between parallel processes are effectively
  prevented: workers share data only at PCA-update boundaries, and TuRBO
  acquisition is stochastic; there is no explicit reservation of candidate
  points (this is the PI's Goal 0, to be analyzed in a later stage).
- Weighting used for the "weighted histogram" of J (likelihood weights from
  penalty? uniform over good models?) — needs PI confirmation.
- The role of legacy file `4UpsBoTorch_Sersic.txt` (referenced in patterns but
  not present in the repo).
