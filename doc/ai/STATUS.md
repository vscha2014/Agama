# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-11-08: diagnostic rh/beta profiles
prepared (Q31), synthetic BoTorch preflight passed. No new AGAMA model calculations.

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
- Free-alpha search completed: modest gain vs same-host alpha=2 control; high-alpha plateau reaches
  the prior edge, gh/rh boundaries persist. PCA retains alpha, drops mostly gh. Private
  `results/logs/localpca/localpca_alpha_…_20261107_232859/NOTES.md`; no precise alpha determination.

## Code state

- Production and CONTRACT unchanged; library/seed/git-results protocols remain as in §3d, §3g–§3h.
- Q profile (§3i, DECISIONS 2026-10-28): seed-42 CRN, trust-region BO per Q in the full box, continuation
  + Sobol/global starts, optional priors, isolated `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1), resume; `195cba8`.
- Local PCA search (DECISIONS 2026-11-03, §3j): `run_local_pca.py` + `launch_tycho.sh --local-pca` —
  full 4D TuRBO, local box, seed 42, context-checked Q-profile priors, BoTorch self-test in preflight.
- Explicit starts before Sobol (2026-11-04), fixed box parameters via NAME=v:v (2026-11-06), resumable.
- Free alpha (DECISIONS 2026-11-07, §3k): `--local-pca --fit-alpha`, alpha ∈ [0.5, 3] (Hayashi), PCA 5→4
  with the dropped PC pinned at the best point; alpha in library keys/metadata; legacy unchanged.
- rh profiles (§3l, Q31): beta=3/4/7/10, 7 radii in existing bounds, full 4D nuisance PCA, no old penalties.
  Beta in keys/metadata/history; per-cell resume, budget extension; prepared only, no dynamical run.

## Next steps (started by the user)

1. Deliver prepared rh/beta code/starts, then run after preflight when requested; up to 672 evaluations.
   Private `results/logs/rhprofile/HANDOFF_RH_BETA.md`. Radial J/outer-tracer follow-ups: Q32.
2. J-factor/mass comparison of near-spherical vs compact families; rh=30/gh=0 bounds; other incl.
   A variable-alpha J calculation needs separate treatment; production remains frozen.
3. fornax-dwarf native AGAMA (`~gala`, `agamaBoTorch`) has completed its first search and seed-42 control.
   Repos have write access; `TZ=Europe/Moscow` for gala (system UTC); taskset ≤8 cores / scope ≤44 GiB.
   Stop via `systemctl … kill --signal=TERM`: Ctrl-C inside `agama.orbit` exits 130 (state intact).

## Open questions

Q15 (Upsilon Brent → production), Q16 (J weighting), Q17–Q19 (shards, `ic`/`inttime`, naming),
Q20 (production `pipefail`), Q23 (OpenMP wrap), Q32 (radial J/outer tracers) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28–Q31, TR bound, PCA std floor, git sync, Q profile, local PCA, gray seeds, free alpha.
