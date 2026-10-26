# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-10-26: library-check fix
for the finalist batch prepared; finalist seed scan ready to run on tycho.

## Where the science stands

- Completed free-Q L24 and spherical Q1 searches at incl=90 provide two finalists.
  Their historical minima and adaptive clusters do not calibrate IC uncertainty
  or establish connectivity of the branches. Numerical results remain in results/.
- The Upsilon control (2026-10-25) completed exp/reuse/prod on each existing library;
  the protocol effect on the finalists' gap is small. Both reports were marked failed
  only by the runner's store check (placeholder SQLite records, files valid). See the
  private run's NOTES and postmortem under
  `results/logs/multi_d1_nb250_gh0_l24n184_ser0_n2_20261025_225604/`.
- The spherical control library is historical N23/l4 despite requested `--l24`;
  reuse does not recompute orbits. A fresh common-L24 paired seed comparison is
  still outstanding. Root search pools are unchanged.

## Code state

- Public code HEAD `61ad7d0` plus uncommitted fix (runner, tycho launcher, tests,
  docs). Production, CONTRACT, harness and storage are unchanged.
- Runner library check: placeholder of a read-only library is not a failure; real
  size/MD5 mismatch, metadata failure and saved-but-unregistered stay fatal.
  `--require-library` for control runs; reports carry `potential_provenance`
  (Q=1 cache keys omit the potential tag).
- `launch_tycho.sh --models … --ic-seeds=SPEC`: fresh integrations, isolated
  `orblib_seed/<TS>_m<i>/` store, seed files/reports and paired `dP` summary in
  `results/logs/seed/`; `--summarize` reports `distinct_models` and no cross-model scatter.
- Git-backed `results/` mode (`harness/orblib_exp.md` §3g–§3h) unchanged.
- Verification: 247 harness/launcher tests + 167 potential-check tests pass
  (mocked/numeric); py_compile, bash -n, git diff --check clean.

## Next steps (need user go-ahead on tycho)

1. Commit/push the fix, update tycho, run
   `launch_tycho.sh --l24 --models=models_finalists_q1_freeq.txt --ic-seeds=1-20`
   (commands in private `results/logs/Q1d1_rh_bounds/HANDOFF_FREEQ_L24.md` §5).
2. Analyse paired `ΔP(seed)` (mean, SEM, sign fraction) — not new records.
3. Then: profile across Q, J-factor/outer-continuation analysis, other inclinations.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync,
finalist batch library check (2026-10-26).
