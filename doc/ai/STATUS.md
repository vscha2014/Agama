# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-11-07: free-alpha local search prepared
(not run). Numerical results private; no calculations executed by the agent.

## Where the science stands

- Completed free-Q L24 and spherical Q1 searches at incl=90 provide two finalists; their
  historical minima do not calibrate IC uncertainty. Numerical results remain in results/.
- Paired seed scan favours the old free-Q finalist over the strict-spherical finalist.
  It measures IC-noise robustness of two fixed points, not observational significance.
  Details: private `results/logs/seed/seeds_multi_…_20261026_081001/NOTES.md`.
- Q profiles use seed-42 CRN, re-optimise (gh, rh, rho0) at fixed Q in the full physical box,
  and keep their rows separate from ordinary search pools. Both runs are analysed:
  private `results/logs/qprofile/qprofile_…_20261029_111955/NOTES.md` and
  `results/logs/qprofile/qprofile_…_20261102_145506/NOTES.md`.
- Budget-limited profile searches give upper bounds on the true minima, not a proven barrier;
  sparse Q samples do not establish lost CRN correlation or a numerical error.
- Both local seed-42 searches are done (private NOTES in `results/logs/localpca/localpca_…_20261104_134856/`
  and `…_20261105_094229/`). Compact minimum is closed except gh=0, lower Q is worse; the near-spherical
  one ranks clearly better at seed 42 but sits on gh=0/rh=30 and seed 42 is known to favour that side.
- Paired 20-seed scan of these two optima (gray): the near-spherical point stays better at every seed;
  the seed-42 selection bias is small for both. Fixed points only; gray differs from tycho at the
  bit level (other build, other `GEOM_HASH`) but shares the IC stream. Private
  `results/logs/seed/seeds_multi_…_20261106_132732/NOTES.md`.
- Cusp audit: optimised fixed-gh=1 search done (seed 42) — strongly disfavoured, rho0 bound-conditional.
  Weak cusps: free-Q local search confirms a shallow plateau worse than the cores (not a proven barrier).
  The frozen alpha=2 family differs from literal NFW. See `results/logs/CUSP_ASSESSMENT_20261105.md`.

## Code state

- Production and CONTRACT unchanged. Runner library check, `--require-library`,
  `--ic-seeds` paired batches, git-backed `results/` (`harness/orblib_exp.md` §3d, §3g–§3h) as before.
- Q profile (§3i, DECISIONS 2026-10-28): seed-42 CRN, trust-region BO per Q in the full box, continuation
  + Sobol/global starts, optional priors, isolated `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1), resume; `195cba8`.
- Local PCA search (DECISIONS 2026-11-03, §3j): `run_local_pca.py` + `launch_tycho.sh --local-pca` —
  TuRBO in a full 4D PCA basis inside a local box, seed 42 before every evaluation, Q-profile
  seed-42 points as priors (context-checked), BoTorch self-test in preflight.
- Explicit starts (DECISIONS 2026-11-04): `--initial-points=FILE` before Sobol, within the budget, resumable.
- Fixed box parameters (DECISIONS 2026-11-06): `--box NAME=v:v` → k-dim PCA/GP/TuRBO over the free
  ones (gh=1:1 search); all-free boxes bit-identical.
- Free alpha (DECISIONS 2026-11-07, §3k): `--local-pca --fit-alpha`, alpha ∈ [0.5, 3] (Hayashi), PCA 5→4
  with the dropped PC pinned at the best point; alpha in library keys/metadata; legacy unchanged. 437 tests pass.

## Next steps (started by the user)

1. Gray seed scan done (DECISIONS 2026-11-06/07). Optional: Q-valley transect across seeds.
2. Cusps (tycho, seed 42) analysed: weak cusp worse than cores at every Q (`…_094514/NOTES.md`); optimised
   gh=1 far worse still, only at extreme flattening and rho0=10 (`…_20261106_170728/NOTES.md`). Stop runs
   with `systemctl … kill --signal=TERM`: Ctrl-C inside `agama.orbit` exits 130 (state intact).
3. J-factor/mass comparison of the near-spherical vs compact families; rh=30/gh=0 bound question; other incl.
   Free-alpha run (80 evals) ready: private `results/logs/localpca/HANDOFF_LOCALPCA_ALPHA.md`; user preflight.
4. New host fornax-dwarf (Ryzen 7900X): native AGAMA in `~gala` (env `agamaBoTorch`), all checks passed
   (DECISIONS 2026-11-07, §3g); repos cloned with write access; `TZ=Europe/Moscow` for `gala` (system UTC);
   free-alpha run there with ≤ 8 cores / 44 GiB. User: preflight, seed-42 control.

## Open questions

Q15 (Upsilon Brent → production), Q16 (J weighting), Q17–Q19 (shards, `ic`/`inttime`, naming),
Q20 (production `pipefail`), Q23 (OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28–Q30, TR bound, PCA std floor, git-results sync, Q profile, local PCA, gray seed scan, free alpha.
