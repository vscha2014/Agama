# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-22 (`results/` consolidated; Round 4 right-edge `rho0` probe completed on `tycho`, closing the `Q=1, incl=90°` search).

## Where the science stands

- Free-Q baseline best model: `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**). Earlier notes wrongly attributed it to
  `legacy_d0_nb250_gh0_ser0`; that d0 run reached only ≈1.60 (fixed in the registry).
- Fixed spherical halo (`Q=1`): four targeted probe rounds (2026-10-05/06/15/22) and an
  automated 40-iteration TuRBO-PCA search on tycho (2026-10-16, `results/logs/spherical_free/`)
  completely bracket the `Q=1, incl=90°` minimum inside the physical box (`gh ≈ 0`, flat
  uniform-core plateau `rh ≈ 25–30` kpc, interior `rho0` optimum with sharp rise at `rho0 ≥ 47`,
  invariant `rho0·Upsilon`, 53 models with `penalty ≤ 2.0`). Q29 pilot on tycho (2026-10-14)
  confirmed baseline `N=23, lmax=4` is sufficient for `Q=1` (`|ΔP| ≲ δP`).
  Details: local `results/logs/spherical_free/NOTES.md` and `results/logs/Q1d1_rh_bounds/NOTES.md`.
- Bounds were widened 2026-09 to `rh ∈ [0.5, 7]`, `rho0 ∈ [10, 120]`, and 2026-10-06
  to `rh ∈ [0.5, 30]`. The 2026-09-29 launch produced **no** new evaluations
  (memory-reclaim livelock, `swap 0`, no OOM-kill). Seven `.npz` survive, verified
  reusable; see `results/logs/Q1d1_nb250_gh0_ser0_i90.0_20260929/postmortem.md`.
- Existing article J baseline belongs to production free-Q, not the new Q=1 runs.
  Before publication, test J sensitivity to outer continuation and broaden comparison
  beyond Hayashi et al. (2016), including Pascale et al. (2018) action-based models.

## Code state

- Harness: `--Q1`, widened bounds, aperture-vertex grid, streaming orbit-library delivery;
  `std_floor` applied before `< 1e-10 → 1.0`; `SIGTERM` in TuRBO phase saves `phase='turbo'`
  checkpoint with `params_obs` even without `--stream-orblib`; `ORBLIB_LOCAL_STORE=1`
  supports local save slots and `STOP` markers when running `--offline`.
- Git-backed `results/` mode (`harness/orblib_exp.md` §3h): `launch_multi_model.sh`,
  `launch_single_model.sh`, `launch_orblib_exp.sh` and `launch_tycho.sh` auto-detect
  `../results/.git`, read history directly from `../results/` without copying to `py/`,
  merge new `out_*` rows into `../results/out_<host>_<EXP_ID>.txt` (no symlinks), save
  logs/reports to `../results/logs/<RUN_TAG>_<TS>/`, keep `.npz` locally, and `git commit/push`.
- `py/launch_tycho.sh` (§3g) & `py/check_potential_convergence/` (README §8): 392 mocked
  and numeric tests pass; Q29 pilot and Round 4 probe completed on tycho.
- Memory fix (`92f009e`, §3a), single/multi-model launchers (`70375a3`, §3b–§3c), OpenMP
  fix for `agama.orbit`, and IC-seed scan (`64afb3e`, §3d) verified on VM.

## Next steps (proposed, need PI go-ahead for anything expensive)

1. Compute the `Q=1` J-factor and outer-continuation sensitivity on the completed `Q=1` pool
   (`results/logs/Q1d1_rh_bounds/HANDOFF_TYCHO_REORG.md`, skill `/jfactor-report`).
2. Q30 (PI): refined multipole for free-Q? Recommended: A/B pairs at 3–4 more free-Q points,
   then switch harness if the shift varies. Notes: local `results/logs/Q1d1_rh_bounds/NOTES.md`.
3. Later: batch re-score/fetch (tar offsets); Q16 J weighting; recompute 77 lost libraries.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production), Q30 (refined multipole for free-Q) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, TR bound, PCA std floor, git-results sync.
