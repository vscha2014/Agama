# Project status

Rewritten (not appended) at the end of every task. Keep ≤ 60 lines.
Last update: 2026-09-30.

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` (run `legacy_d0_nb250_gh0_ser0`).
- Fixed spherical halo (`Q=1`) best: `penalty = 3.5395`, but `rh` sat on the old
  upper bound 3.5 kpc (run `Q1d1_nb250_gh0_ser0_i90.0_20260919_011920`).
  The 2.30 penalty gap does **not** yet exclude a spherical halo.
- Bounds were widened afterwards to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`. The
  first launch that used them (2026-09-29) produced **no** new evaluations: the
  VM became unresponsive after the first integrations, leading suspect RAM
  pressure from discarded trajectories (Q21). Seven `.npz` libraries survive on
  the VM; see `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`.
- Article headline currently: `log10 J(0.5°) ≈ 18.59` (M0 estimator,
  penalty-weighted spread). Comparable published value: Hayashi et al. 2016,
  `17.90 (+0.28/−0.16)` at D = 147 kpc, axisymmetric Jeans — methods differ.

## Code state

- Experimental harness `py/Fornax_P21_PCA_w3Sersic_orblib_exp.py` +
  `py/launch_orblib_exp.sh` + `py/orblib_storage.py`: `--Q1` mode, widened
  bounds, aperture-vertex datacube grid, streaming orbit-library delivery.
  Optimisation/delivery verified by light mocked tests; no new real calculation.
- Local indexing support: rclone MD5 case handling fixed, seekable tar indexing
  without extraction, staged indexes outside sync, cloud verification before
  index-only publication. Tiny-archive/CLI tests pass; the user will perform the
  full indexing and cloud publication (`harness/orblib_exp.md`).
- Production scripts (`*_yaVM.py`, `J_factor_*.py`, `launch_docker_parallel.sh`)
  untouched; the `pipefail`/`ls .done_*` latent bug found in the experimental
  launcher still exists there (Q20, unanswered).
- Harness work is committed through `cda222c`; the local branch is ahead of
  `origin/master` (pushes have been failing for lack of git credentials).
- 2026-09-22: agent harness reorganised — `AGENTS.md` trimmed to pointers,
  `CONTRACT`/`STATUS`/`DECISIONS`/`harness/*` split out, `results/` registry and
  private `paper/` introduced (`DECISIONS.md`).

## Next steps (proposed, need PI go-ahead for anything expensive)

0. Verify the seven surviving 09-29 libraries offline, then get their
   `Upsilon`/`penalty` via the reuse path; cut peak memory (Q21) and re-launch
   small before any 8-worker run. Do not run `prepare`/`prune-verified --apply`
   on that VM first — see the 09-29 post-mortem.
1. Profile runs at fixed `rh = 3.5 / 5 / 7 kpc` with `Q=1`, re-optimising
   `gh, rho0, Upsilon` — cheaper than a wide free rerun.
2. Re-check `Upsilon` with a full-library solve on a few stored best libraries
   before any new large launch.
3. User-run `orblib_storage.py index --local-archives orblib --publish` with
   the sync daemon stopped; restart sync only after verified publication.
4. Decide the J-factor weighting method (Q16) before finalising article numbers.
5. Recompute the 77 lost orbit libraries targeted, not by rerunning the search.

## Open questions

Q15 (Upsilon Brent speed-ups into production), Q16 (J weighting / sampling
density), Q17–Q19 (shard consolidation, `ic`/`inttime` storage, shard naming),
Q20 (same `pipefail` bug in the production launcher), Q21 (drop the discarded
`trajsize` request to cut peak RAM) — see `questions_for_pi.md`.
