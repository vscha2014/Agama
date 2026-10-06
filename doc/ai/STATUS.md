# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-06 (Q=1 probe round 2 analysed).

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**). Earlier notes wrongly attributed it to
  `legacy_d0_nb250_gh0_ser0`; that d0 run reached only ≈1.60 (fixed in the registry).
- Fixed spherical halo (`Q=1`): two probe rounds 2026-10-05/06 (local
  `results/Q1d1_rh_bounds/NOTES.md`): penalty falls with `rh` and flattens towards the
  uniform-core limit (rh 25 of bound 30); a lower stellar M/L helps further. The gap to
  free-Q shrank to a fraction of its 09-19 size but remains clearly positive.
- Bounds were widened 2026-09 to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`. The
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
2. Q=1 (rh ≤ 30, bounds guard, tar-index delivery committed): either a third probe round
   (rh 25/30 × rho0 sweep, gh 0.1/0.2) or the `--Q1` search (expected to drift to the
   rh bound; `1/rh` parametrisation = PI question).
   Hand-off: `results/Q1d1_rh_bounds/HANDOFF.md` (local-only).
3. Later: batch re-score/fetch (tar offsets); Q16 J weighting; recompute 77 lost libraries.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, TR bound, PCA std floor.
