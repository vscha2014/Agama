# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-11-04: compact local search prepared;
explicit boundary starts at the existing IC seed. Numerical results private; no new run executed.

## Where the science stands

- Completed free-Q L24 and spherical Q1 searches at incl=90 provide two finalists; their
  historical minima do not calibrate IC uncertainty. Numerical results remain in results/.
- The Upsilon control (2026-10-25) showed a small protocol effect on the finalists' gap;
  its reports were marked failed only by the runner's store check (fixed). Notes under
  `results/logs/multi_d1_nb250_gh0_l24n184_ser0_n2_20261025_225604/`.
- Paired seed scan (seeds 1–20, fresh L24 orbits for both finalists, 40/40 ok):
  the free-Q finalist is better in almost every seed; the mean paired gap is many
  MC standard errors from zero and agrees with the historical gap, because both
  historical records carry a similar winner's-curse bias. This is IC-noise robustness
  of two fixed points, not observational significance or a connectivity test.
  Numbers: private `results/logs/seed/seeds_multi_…_20261026_081001/NOTES.md`.
  Root search pools are unchanged.
- Q profiles use seed-42 CRN, re-optimise (gh, rh, rho0) at fixed Q in the full physical box,
  and keep their rows separate from ordinary search pools. Both runs are analysed:
  private `results/logs/qprofile/qprofile_…_20261029_111955/NOTES.md` and
  `results/logs/qprofile/qprofile_…_20261102_145506/NOTES.md`.
- The private notes distinguish the candidate penalty gap from the intermediate profile
  height. Budget-limited searches give upper bounds on the true minima, not a proof of
  a minimum barrier. Sparse Q samples do not establish a loss of CRN correlation or a
  numerical error. Earlier stronger claims in the second run's notes were corrected.

## Code state

- Production, CONTRACT, harness and storage are unchanged. Runner library check
  (placeholder of a read-only library is not a failure), `--require-library`,
  `potential_provenance`, `--ic-seeds` paired batches with isolated `orblib_seed/` store and
  git-backed `results/` (`harness/orblib_exp.md` §3d, §3g–§3h) as before.
- Q profile (§3i, DECISIONS 2026-10-28): seed-42 CRN, trust-region BO per Q in the full box,
  `compact`/`extended` continuation + Sobol/global starts, optional priors, isolated
  `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1) into `orblib/`, resume.
- `195cba8`: grids beyond Q=1, rejected points capped at 1e5 in GP, budget extension, no-smooth-check.
- Local PCA search (DECISIONS 2026-11-03, §3j): `run_local_pca.py` + `launch_tycho.sh --local-pca` —
  TuRBO in a full 4D PCA basis inside a local box, seed 42 before every evaluation, Q-profile
  seed-42 points as priors (context-checked), BoTorch self-test in preflight.
- Explicit starts (DECISIONS 2026-11-04): `--initial-points=FILE` before Sobol, within the new-eval
  budget; checkpointed indices/input hash, old states remain resume-compatible. 392 mocked tests pass.

## Next steps (need user go-ahead on tycho)

1. Deliver checked code and private compact inputs; confirm near-spherical run status on tycho.
   Follow `results/logs/localpca/HANDOFF_LOCALPCA_COMPACT.md`: preflight, then 4 starts + 4 Sobol
   + 52 TuRBO. One launcher at a time; do not update code under a running search.
2. Analyse both local regions at seed 42 before additional IC seeds (DECISIONS 2026-11-03).
3. Optional field-only potential validation beyond Q=1.
4. J-factor/mass comparison of both branches incl. outer continuation; other inclinations.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync,
finalist batch library check (2026-10-26), Q profile with CRN (2026-10-28).
