# Iteration / PCA-update tuning (Q14 — APPLIED)

Answer to PI question Q14. **Applied (PI-approved):** `__main__` now calls
`run_pca_optimization(n_iter=40, pca_update_interval=12, ...)`. The function
default remains `pca_update_interval=15`, but production passes **12**. `n_iter`
stays 40.

## Behavior

- A PCA rebuild fires when `iter % pca_update_interval == 0` **and** ≥5 new points
  are buffered (`_update_pca_model`). With `n_iter=40` and interval **12**,
  rebuilds occur at iter 12, 24, 36 (3 refreshes). Previously (interval 15): at
  most 2 rebuilds (iter 15, 30).
- Each rebuild refits `StandardScaler`+`PCA` from ALL fresh cross-process data,
  recomputes PC bounds, and re-projects the observation history; the TuRBO trust
  region is updated in place.

## Trade-off

- **Too rare**: as workers push into new regions of (Q, gh, rh, rho0), the PCA
  basis (fit on early data) becomes stale → acquisition explores in a subspace
  that no longer matches the active region.
- **Too frequent**: the PC coordinate frame keeps shifting, so the GP is refit on
  re-projected history and the trust region is repeatedly perturbed → slower
  convergence and noisier acquisition. Refits also cost time and read all files.

## Recommendation

- **Rule of thumb**: `pca_update_interval ≈ n_iter / 3 … n_iter / 4`, with the
  existing ≥5-new-points guard kept. For `n_iter=40` that means **12** (rebuilds
  at iter 12, 24, 36 → 3 refreshes, ~12 evals to let the region settle between
  them). Scalable form: `pca_update_interval = max(8, n_iter // 4)`.
- **Fresh-posang campaign start**: data is sparsest early, so the first rebuilds
  add the most value. Optionally make the first interval shorter (e.g. rebuild at
  iter 6 then every 12) — a small, optional refinement, not essential.
- **`n_iter`**: 40 per process is reasonable given resume + 4 parallel workers +
  multiple hosts. Effective sampling ≈ `n_iter × n_parallel × n_relaunches`, all
  sharing data live. Keep `n_iter=40` unless a single launch needs deeper local
  refinement; increasing it mainly helps if PCA updates are also more frequent.

## As applied

- `pca_update_interval=12` is passed from the `__main__` call site (function
  default left at 15). A `--pca-update-interval` CLI flag could be added later if
  per-run tuning is needed. No other contract item is affected.
