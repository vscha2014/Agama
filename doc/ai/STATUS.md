# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-03 (multi-model launcher).

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**). Earlier notes wrongly attributed it to
  `legacy_d0_nb250_gh0_ser0`; that d0 run reached only ≈1.60 (fixed in the registry).
- Fixed spherical halo (`Q=1`) best: `penalty = 3.5395`, but `rh` sat on the old
  upper bound 3.5 kpc (run `Q1d1_nb250_gh0_ser0_i90.0_20260919_011920`).
  The 2.30 penalty gap does **not** yet exclude a spherical halo.
- Bounds were widened afterwards to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`. The
  first launch that used them (2026-09-29) produced **no** new evaluations
  (memory-reclaim livelock, `swap 0`, no OOM-kill). Seven `.npz` survive, verified
  reusable; see `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`.
- Article headline: `log10 J(0.5°) ≈ 18.59` (M0, penalty-weighted spread). Nearest
  published value: Hayashi et al. 2016, `17.90 (+0.28/−0.16)`, D = 147 kpc,
  axisymmetric Jeans — methods differ.

## Code state

- Harness (`py/Fornax_P21_PCA_w3Sersic_orblib_exp.py`, `py/launch_orblib_exp.sh`,
  `py/orblib_storage.py`): `--Q1`, widened bounds, aperture-vertex datacube grid,
  streaming orbit-library delivery.
- Memory fix (`92f009e`, §3a) and archived-not-local warning (`13f1659`)
  committed: no trajectories, block `.npz` writer, 2 save slots, SIGTERM → STOP → checkpoint, RSS logging;
  launcher with 4 workers, host swap, swapless container limit, watchdogs,
  emergency upload. **Measured on the VM 2026-10-01**: integration+save ≈1 GB,
  peaks only in the solve phases (higher for float64 reuse), no host pressure.
- Single-model check (uncommitted `py/run_single_model.py`,
  `py/launch_single_model.sh`) **ran OK** 2026-10-01 (local
  `results/single_d1_nb250_gh0_ser0_i90.0/NOTES.md`): storage round trip exact,
  Upsilon speed-ups negligible, grid identical; penalty above the reference but
  inside production's scatter there. AGAMA's RNG starts from one seed per process
  ⇒ the 4 repeats were one realisation (Q22). Single-thread `agama.orbit` (torch shares libgomp) fixed in the harness 10-03
  and verified (same MD5, ≈20× faster integration); production port = Q23.
- New (uncommitted, not run): `py/launch_multi_model.sh --models=FILE` — N
  different models in parallel, `exp` protocol only, all libraries delivered
  (§3c). Local `results/multi_d1_nb250_gh0_ser0_top4/`: global best + 3 valley ends.
- Branch ahead of origin (no push credentials). Production scripts untouched.

## Next steps (proposed, need PI go-ahead for anything expensive)

0. On go-ahead: multi-model run of `models_top4.txt` (`--preflight` first).
1. Distinct IC seeds for repeats (Q22) — needed to judge harness vs production.
2. `--reuse-orblib` pass over the seven 09-29 libraries, 4 workers (no `prepare`/`prune` first).
3. Only then a fresh 4-worker `--Q1` search on the widened bounds; or profile
   runs at fixed `rh = 3.5 / 5 / 7 kpc`, re-optimising `gh, rho0, Upsilon`.
4. Later: batch re-score/fetch mode for changed data (tar offsets then).
5. Decide the J-factor weighting method (Q16) before finalising article numbers.
6. Recompute the 77 lost orbit libraries targeted, not by rerunning the search.

## Open questions

Q15 (Upsilon Brent speed-ups into production), Q16 (J weighting / sampling
density), Q17–Q19 (shard consolidation, `ic`/`inttime` storage, shard naming),
Q20 (`pipefail` bug in the production launcher), Q22 (IC seeds for diagnostic
repeats) — see `questions_for_pi.md`. Q21 is answered (see `DECISIONS.md`).
