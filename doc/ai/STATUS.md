# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-10-31: the Q-profile run on
tycho (2026-10-29/31) completed and was analysed; numbers in the private results.

## Where the science stands

- Completed free-Q L24 and spherical Q1 searches at incl=90 provide two finalists.
  Their historical minima and adaptive clusters do not calibrate IC uncertainty
  or establish connectivity of the branches. Numerical results remain in results/.
- The Upsilon control (2026-10-25) showed a small protocol effect on the finalists' gap;
  its reports were marked failed only by the runner's store check (fixed). Notes under
  `results/logs/multi_d1_nb250_gh0_l24n184_ser0_n2_20261025_225604/`.
- Paired seed scan (seeds 1–20, fresh L24 orbits for both finalists, 40/40 ok):
  the free-Q finalist is better in almost every seed; the mean paired gap is many
  MC standard errors from zero and agrees with the historical gap, because both
  historical records carry a similar winner's-curse bias. This is IC-noise robustness
  of two fixed points, not observational significance; connectivity of the branches
  is untested. Numbers: private `results/logs/seed/seeds_multi_…_20261026_081001/NOTES.md`.
  Root search pools are unchanged.
- Q profile (seed-42 CRN, (gh, rh, rho0) re-optimised in the full box at 8 values of Q,
  incl=90): completed on tycho, all branches budget-limited, exit 0. The flattened and the
  near-spherical minima are separated by a penalty hump over the intermediate Q that is far
  larger than the realisation scatter and the optimisation slack; both minima are narrow in
  Q; compact and extended halo families coexist at every Q and become degenerate in penalty
  near Q=0.9. Q=1 is the grid end, prolate halos were not probed. Numbers and follow-ups:
  private `results/logs/qprofile/qprofile_…_20261029_111955/NOTES.md`.

## Code state

- Production, CONTRACT, harness and storage are unchanged. Runner library check
  (placeholder of a read-only library is not a failure), `--require-library`,
  `potential_provenance`, `--ic-seeds` paired batches with isolated `orblib_seed/` store and
  git-backed `results/` (`harness/orblib_exp.md` §3d, §3g–§3h) as before.
- Public code HEAD `db4d80e`: Q profile (§3i, DECISIONS 2026-10-28): seed-42 CRN, trust-region
  BO per Q in the full box, `compact`/`extended` continuation + Sobol/global starts, optional
  priors, isolated `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1) into `orblib/`, resume.
- Uncommitted (2026-10-31, DECISIONS): Q-profile grid without Q=1 / beyond 1 (extended starts
  at the grid Q closest to 1, goes down then up); finite penalty ≥ 1e5 ⇒ `rejected`, fed to the
  GP capped at 1e5, not a consecutive failure; budget extension on resume; `--no-smooth-check`;
  launcher options `--local-budget/--global-budget/--no-smooth-check`.
- Verification: 302 harness/launcher/profile tests + 167 potential-check tests pass
  (mocked/numeric); py_compile, bash -n, git diff --check clean.

## Next steps (need user go-ahead on tycho)

1. Plan item 3: commit/push, `git pull` on tycho, then
   `launch_tycho.sh --q-profile --q-grid=0.28,0.29,0.95,1.1,1.2 --no-smooth-check` (preflight
   first; 130 evaluations, max 180, ~22 h; check free disk for the saved libraries).
   Other-seed re-evaluation is not needed for the hump.
2. Optional: extend the finalist scan by ~8 seeds if a tighter paired SEM is needed.
3. J-factor/mass comparison of both branches incl. outer continuation; other inclinations.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync,
finalist batch library check (2026-10-26), Q profile with CRN (2026-10-28).
