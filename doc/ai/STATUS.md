# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-12 (tycho `--offline` implemented; no real run).

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**). Earlier notes wrongly attributed it to
  `legacy_d0_nb250_gh0_ser0`; that d0 run reached only ≈1.60 (fixed in the registry).
- Fixed spherical halo (`Q=1`): two probe rounds 2026-10-05/06 favour a nearly
  uniform inner halo along the evaluated branch; no nuisance-profiled Q=1 minimum
  or measured core size yet. Global halo-mass extrapolation needs a plausibility check.
  Literature, analytic diagnostics and interpretation caveats: local `results/Q1d1_rh_bounds/NOTES.md`.
- Bounds were widened 2026-09 to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`. The
  first launch that used them (2026-09-29) produced **no** new evaluations
  (memory-reclaim livelock, `swap 0`, no OOM-kill). Seven `.npz` survive, verified
  reusable; see `results/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`.
- Existing article J baseline belongs to production free-Q, not the new Q=1 probes.
  Before publication, test J sensitivity to outer continuation and broaden comparison
  beyond Hayashi et al. (2016), including Pascale et al. (2018) action-based models.

## Code state

- Harness: `--Q1`, widened bounds, aperture-vertex grid, streaming orbit-library delivery.
- `py/launch_tycho.sh` (`470c1f9` + offline): harness on tycho in ¼ of the host
  (systemd user scope, 4 threads, MemTotal/4, no swap, local libraries, `out_tycho_d1_*`);
  replaces the old fork `Fornax_P21_symm_PCA_w3Sersic.py`; `--offline` (no rclone) in harness+launcher. §3g.
- `py/check_potential_convergence/`: seed-42 pilot CLI/offline VM wrapper prepared after `e650f76`.
  324 mocked/numeric tests pass; local model preflights complete; no real pilot/VM run.
- Memory fix (`92f009e`, §3a), archived-not-local warning (`13f1659`): no trajectories,
  block `.npz` writer, save slots, SIGTERM → checkpoint, RSS logs, launcher watchdogs.
  **VM 2026-10-01**: integration+save ≈1 GB, peaks only in solve phases, no host pressure.
- Single-model check / multi-model launcher (§3b–§3c, `70375a3`) ran OK 2026-10-01
  (local `results/single_d1_nb250_gh0_ser0_i90.0/NOTES.md`). OpenMP fix for
  `agama.orbit` verified 10-03 (same MD5, ≈20× faster); production port = Q23.
- IC-seed scan (Q22 A, §3d, `64afb3e`) **ran OK 2026-10-04** (local `results/seed/NOTES.md`):
  seeding works; production reference is a winner's-curse minimum, harness − production
  offset consistent with zero; single-evaluation penalty noise measured (Q16, ranking).
  4×8 layout faster than 1×32; peak memory well below limits.

## Next steps (proposed, need PI go-ahead for anything expensive)

1. tycho (offline, DECISIONS 2026-10-12): user preflight + incl=90 run with `--offline`; adapt the
   Q29 pilot launcher to tycho, then Q=1. Handoff: local `results/tycho_d1_nb250_gh0_ser0/HANDOFF.md`.
2. Q29: authorise VM evidence/preflight, then free-Q pilot; review before the two Q1 controls.
   Handoff: `results/Q1d1_rh_bounds/HANDOFF_POTENTIAL_CHECK.md` §8.14; commands: diagnostic README §7.
   Keep exact local inputs/evidence; a different VM binary needs field validation, not a bypass.
   Use live potentials, not .ini; no automatic seed expansion, cloud delivery or production adoption.
3. Later: batch re-score/fetch (tar offsets); Q16 J weighting; recompute 77 lost libraries.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production), Q29 (pilot execution gates) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, TR bound, PCA std floor.
