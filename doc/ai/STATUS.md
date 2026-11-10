# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-11-10: inclination-fan driver written
(code only, nothing run); the rh/beta upper-radius direction remains open.

## Where the science stands

- Completed free-Q L24 and spherical Q1 searches at incl=90 provide two finalists (numbers in
  results/); their historical minima do not calibrate IC uncertainty, and the paired seed scan
  favours the free-Q one — fixed-point robustness only (`results/logs/seed/…_20261026_081001/`).
- Q profiles use seed-42 CRN and optimise (gh,rh,rho0) in the full box; isolated from search pools.
  Both analysed (`results/logs/qprofile/qprofile_…_{20261029_111955,20261102_145506}/NOTES.md`);
  budget-limited profiles give upper bounds on the true minima, not a proven barrier.
- Both local seed-42 searches are done (`results/logs/localpca/localpca_…_{20261104_134856,20261105_094229}/`):
  the compact minimum is closed except gh=0 and lower Q is worse, while the near-spherical one ranks
  clearly better at seed 42 but sits on gh=0/rh=30, a side seed 42 is known to favour.
- Paired 20-seed scan of these two optima (gray): the near-spherical point stays better at every seed
  and the seed-42 selection bias is small for both; gray differs from tycho at the bit level but shares
  the IC stream (`results/logs/seed/seeds_multi_…_20261106_132732/NOTES.md`).
- Cusp audit: fixed-gh=1 strongly disfavoured (rho0 bound-conditional), weak cusps form a shallow
  plateau worse than the cores, frozen alpha=2 ≠ literal NFW (`results/logs/CUSP_ASSESSMENT_20261105.md`).
- Free-alpha search completed: modest gain vs same-host alpha=2 control; high-alpha plateau reaches
  the prior edge, gh/rh boundaries persist, no precise alpha determination; PCA retains alpha and
  drops mostly gh (`results/logs/localpca/localpca_alpha_…_20261107_232859/NOTES.md`).
- rh/beta profiles completed: every beta branch still favours the upper radius edge, extended-cell
  winners remain inherited starts, the full-box budget did not establish nuisance convergence
  (`results/logs/rhprofile/rhprofile_…_20261108_122200/NOTES.md`).

## Code state

- Production and CONTRACT unchanged; library/seed/git-results protocols remain as in §3d, §3g–§3h.
- Q profile (§3i, DECISIONS 2026-10-28): seed-42 CRN, trust-region BO per Q in the full box, continuation
  + Sobol/global starts, optional priors, isolated `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1), resume; `195cba8`.
- Local PCA (§3j): full 4D TuRBO, local box, seed 42, checked priors, BoTorch preflight, resumable;
  explicit starts before Sobol, fixed parameters via NAME=v:v. Free alpha (§3k): `--fit-alpha`,
  alpha ∈ [0.5, 3], PCA 5→4, alpha in keys/metadata.
- rh profiles (§3l, Q31): beta=3/4/7/10, 7 radii in existing bounds, full 4D nuisance PCA, no old
  penalties; beta in keys/metadata/history, per-cell resume, budget extension; `5408074`.
- Inclination fan (§3m, Q33, DECISIONS 2026-11-10): `py/run_incl_fan.py` + `tests/test_incl_fan.py`.
  Stored `.npz` are unusable at another incl (projection sits inside `matrix_kinem`); the fan shares
  potential/ICs/integration, gives every member the EXACT constraints of its own `axRZst(i)`, leaving
  only E1 (stellar quadrupole in the potential) and E2 (IC basis). Reference member exact, grid
  70–90°, rows isolated in `incl_fan_*.txt`; nothing has been run.

## Next steps (started by the user)

1. Stage A local refinement prepared (DECISIONS 2026-11-10): beta=3,10 × rh=30,20,10,5, fixed rh,
   local box, ≤160 evals; private `results/logs/rhprofile/HANDOFF_RH_LOCAL_A.md`. Stage B deferred; Q32.
2. J-factor/mass comparison of near-spherical vs compact families; rh=30/gh=0 bounds; variable-alpha
   J needs separate treatment; production frozen. Other incl: `run_incl_fan.py --diagnose` and
   `--benchmark` on an AGAMA host (none locally), then Q33 before any `--fan` run for the paper.
3. fornax-dwarf (`~gala`, `agamaBoTorch`): `TZ=Europe/Moscow`; taskset ≤8 cores / scope ≤44 GiB.
   Stop via `systemctl … kill --signal=TERM`: Ctrl-C inside `agama.orbit` exits 130 (state intact).

## Open questions

Q15 (Upsilon Brent → production), Q16 (J weighting), Q17–Q19 (shards, `ic`/`inttime`, naming),
Q20 (production `pipefail`), Q23 (OpenMP wrap), Q32 (radial J/outer tracers), Q33 (shared orbit
basis across inclinations) — see `questions_for_pi.md`. Decided (`DECISIONS.md`): Q21, Q22, Q28–Q31,
TR bound, PCA std floor, git sync, Q profile, local PCA, gray seeds, free alpha, incl fan scope.
