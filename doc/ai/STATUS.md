# Project status

Rewritten at the end of every task; ≤60 lines. Update 2026-10-28: Q-profile driver
(`run_q_profile.py`, `launch_tycho.sh --q-profile`) implemented; run pending on tycho.

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
  reuse does not recompute orbits.
- Paired seed scan (seeds 1–20, fresh L24 orbits for both finalists, 40/40 ok):
  the free-Q finalist is better in almost every seed; the mean paired gap is many
  MC standard errors from zero and agrees with the historical gap, because both
  historical records carry a similar winner's-curse bias. This is IC-noise robustness
  of two fixed points, not observational significance; connectivity of the branches
  is untested. Numbers: private `results/logs/seed/seeds_multi_…_20261026_081001/NOTES.md`.
  Root search pools are unchanged.

## Code state

- Public code HEAD `9dc7557` (library-check fix, isolated seed store; used by the seed
  scan). Production, CONTRACT, harness and storage are unchanged.
- Runner library check: placeholder of a read-only library is not a failure; real
  size/MD5 mismatch, metadata failure and saved-but-unregistered stay fatal.
  `--require-library` for control runs; reports carry `potential_provenance`
  (Q=1 cache keys omit the potential tag).
- `launch_tycho.sh --models … --ic-seeds=SPEC`: fresh integrations, isolated
  `orblib_seed/<TS>_m<i>/` store, seed files/reports and paired `dP` summary in
  `results/logs/seed/`; `--summarize` reports `distinct_models` and no cross-model scatter.
- Git-backed `results/` mode (`harness/orblib_exp.md` §3g–§3h) unchanged.
- Q profile (§3i, DECISIONS 2026-10-28, uncommitted): seed-42 CRN, trust-region BO per Q
  in the full box, `compact`/`extended` continuation + Sobol/global starts, priors from
  earlier seed-42 rows, isolated `qprofile_*` rows, `.npz` ≤ 3.5 (Q≠1) into `orblib/`, resume.
- Verification: 287 harness/launcher/profile tests + 167 potential-check tests pass
  (mocked/numeric); py_compile, bash -n, git diff --check clean.

## Next steps (need user go-ahead on tycho)

1. Q profile on tycho: `bash launch_tycho.sh --q-profile --preflight`, then
   `bash launch_tycho.sh --q-profile` (~31 h; stop/continue with `--resume-profile=<dir>`).
   Hand-off: private `results/logs/qprofile/HANDOFF_QPROFILE.md`.
2. After it: decide on other-seed re-evaluation of decisive points; densify Q near a
   branch switch (plan item 3).
3. Optional: extend the finalist scan by ~8 seeds if a tighter paired SEM is needed.
4. J-factor/mass comparison of both branches incl. outer continuation; other inclinations.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync,
finalist batch library check (2026-10-26), Q profile with CRN (2026-10-28).
