# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-05 (TuRBO TR bound fix).

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
- Single-model check (`run_single_model.py`, `launch_single_model.sh`) and
  multi-model launcher (§3b–§3c) committed in `70375a3`; single check ran OK
  2026-10-01 (local `results/single_d1_nb250_gh0_ser0_i90.0/NOTES.md`), but its
  repeats were one AGAMA realisation (same start seed per process). OpenMP fix
  for `agama.orbit` verified 10-03 (same MD5, ≈20× faster); production port = Q23.
- IC-seed scan (Q22 option A, §3d, committed `64afb3e`): `launch_single_model.sh
  --ic-seeds=1-100 --repeats=4` **ran OK 2026-10-04** (local `results/seed/NOTES.md`,
  REGISTRY): seeding works (seed 42 bit-identical to the default stream); the
  production reference is a winner's-curse minimum, harness − production offset
  consistent with zero ⇒ memory-fix/harness check closed. Penalty noise of one
  evaluation measured (numbers in NOTES) — matters for Q16 and best-model ranking.
  4×8 layout faster than 1×32; peak memory well below limits.

## Next steps (proposed, need PI go-ahead for anything expensive)

1. Optional: one 8×4 throughput measurement (memory allows it).
2. Q=1 on widened bounds: TuRBO TR upper-bound bug **fixed 2026-10-05** (harness + 3
   production copies, committed). Next: PI questions (PCA std floor, GP noise, per-process
   seed), `rh` probes via `launch_multi_model.sh`, then decide on the full `--Q1` search.
   Hand-off: `results/Q1d1_nb250_gh0_ser0_i90.0_widened/HANDOFF.md` (local-only).
3. Later: batch re-score/fetch mode (tar offsets then); Q16 J weighting before
   final article numbers; targeted recompute of the 77 lost libraries.

## Open questions

Q15 (Upsilon Brent speed-ups into production), Q16 (J weighting / sampling
density), Q17–Q19 (shard consolidation, `ic`/`inttime` storage, shard naming),
Q20 (`pipefail` bug in the production launcher), Q23 (OpenMP wrap in production)
— see `questions_for_pi.md`. Q21, Q22, TR-bound fix: answered (`DECISIONS.md`).
