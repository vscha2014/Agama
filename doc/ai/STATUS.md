# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-01.

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` (run `legacy_d0_nb250_gh0_ser0`).
- Fixed spherical halo (`Q=1`) best: `penalty = 3.5395`, but `rh` sat on the old
  upper bound 3.5 kpc (run `Q1d1_nb250_gh0_ser0_i90.0_20260919_011920`).
  The 2.30 penalty gap does **not** yet exclude a spherical halo.
- Bounds were widened afterwards to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`. The
  first launch that used them (2026-09-29) produced **no** new evaluations: the
  VM fell into a memory-reclaim livelock (confirmed from the kernel journal —
  `swap 0`, pressure from 16:26:53, no OOM-kill). Seven `.npz` survive, verified
  reusable; see `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`.
- Article headline: `log10 J(0.5°) ≈ 18.59` (M0, penalty-weighted spread). Nearest
  published value: Hayashi et al. 2016, `17.90 (+0.28/−0.16)`, D = 147 kpc,
  axisymmetric Jeans — methods differ.

## Code state

- Harness (`py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`, `py/launch_orblib_exp.sh`,
  `py/orblib_storage.py`): `--Q1`, widened bounds, aperture-vertex datacube grid,
  streaming orbit-library delivery.
- **2026-10-01 — memory fix implemented** (details: `harness/orblib_exp.md` §3a,
  `DECISIONS.md`): no trajectory request (Q21 answered), `trajsize` out of the
  compatibility key, block `.npz` writer with no float64 duplicate, size+MD5 from
  the writing pass, `ORBLIB_SAVE_SLOTS=2`, SIGTERM → STOP → checkpoint → exit 75,
  per-phase RSS logging. Launcher: 4 workers by default, host swapfile, swapless
  container memory limit, resource sampler, fast MemAvailable/PSI trigger + 3600 s
  no-progress backstop, emergency log upload before shutdown, failed `sudo
  shutdown` reported. 127 mocked tests pass; no AGAMA/docker/cloud run. Expected
  per-worker peak ≈1.5–2 GB vs ≈4–5 GB — **not yet measured on the VM**.
- Production scripts (`*_yaVM.py`, `J_factor_*.py`, `launch_docker_parallel.sh`)
  untouched; the `pipefail`/`ls .done_*` latent bug found in the experimental
  launcher still exists there (Q20, unanswered).
- Committed through `76fbd4f`; the memory fix is **uncommitted** in the working
  tree. The branch is ahead of `origin/master` (pushes fail: no git credentials).

## Next steps (proposed, need PI go-ahead for anything expensive)

0. `--reuse-orblib` pass over the seven surviving 09-29 libraries with 4 workers
   to get their `Upsilon`/`penalty` without re-integration; do not run
   `prepare`/`prune-verified --apply` on that VM first.
1. Re-verify the trajectory change on real data (one re-integrated control model
   vs a stored library) — the only open item of the memory fix.
2. Only then a fresh 4-worker `--Q1` search on the widened bounds; or profile
   runs at fixed `rh = 3.5 / 5 / 7 kpc`, re-optimising `gh, rho0, Upsilon`.
3. User-run `orblib_storage.py index --local-archives orblib --publish` with the
   sync daemon stopped; restart sync only after verified publication.
4. Decide the J-factor weighting method (Q16) before finalising article numbers.
5. Recompute the 77 lost orbit libraries targeted, not by rerunning the search.

## Open questions

Q15 (Upsilon Brent speed-ups into production), Q16 (J weighting / sampling
density), Q17–Q19 (shard consolidation, `ic`/`inttime` storage, shard naming),
Q20 (same `pipefail` bug in the production launcher) — see
`questions_for_pi.md`. Q21 is answered (see `DECISIONS.md`).
