# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-22 (`--l24` refined multipole mode implemented; all `Q=1` rows consolidated into `out_*_Q1d1_nb250_gh0_ser0.txt`; Q30 decided).

## Where the science stands

- Free-Q baseline (`N=23, lmax=4`): best `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**). In the Q29 seed-42 pilot on tycho (2026-10-14), switching a
  flattened halo from `N=23, lmax=4` to the converged `N=184, lmax=24` shifted `penalty`
  by `≫ δP`, so `free-Q` searches now move to `--l24` (`d1_nb250_gh0_l24n184_ser0`).
- Fixed spherical halo (`Q=1`): four targeted probe rounds (2026-10-05/06/15/22) and an
  automated 40-iteration TuRBO-PCA search on tycho (2026-10-16, `results/logs/spherical_free/`)
  completely bracket the `Q=1, incl=90°` minimum inside the physical box (`gh ≈ 0`, flat
  uniform-core plateau `rh ≈ 25–30` kpc, interior `rho0` optimum with sharp rise at `rho0 ≥ 47`,
  invariant `rho0·Upsilon`). Q29 confirmed baseline `N=23, lmax=4` is sufficient for `Q=1`
  (`|ΔP| ≲ δP`). All `Q=1` history rows are consolidated in `results/out_*_Q1d1_nb250_gh0_ser0.txt`.
- Existing article J baseline belongs to production free-Q (`lmax=4`), not the new `Q=1` or
  `--l24` runs. Before publication, compute `Q=1` J-factor, test outer-continuation sensitivity,
  and compare with literature including Hayashi et al. (2016) and Pascale et al. (2018).

## Code state

- Harness: `--Q1`, `--l24` (`--pot-lmax=24 --pot-ngrid=184`, isolated
  `HISTORY_ID = d1_nb250_gh0_l24n184_ser0`, zero-overhead `pot_orbit` while preserving
  baseline `pot_gal` for the Step 1A/1B AST parser and `diagnostic` callback, `Q=1`-preserving
  `orblib_key`, `free-Q` seeding from `prior_seed_patterns` = `4Ups*` + old `out_*_d1_*` with
  orbit/penalty recomputation), widened bounds, aperture-vertex grid, `std_floor` before
  `< 1e-10 → 1.0`, `SIGTERM` TuRBO checkpointing, and `ORBLIB_LOCAL_STORE=1` for `--offline`.
- Git-backed `results/` mode (`harness/orblib_exp.md` §3h): `launch_multi_model.sh`,
  `launch_single_model.sh`, `launch_orblib_exp.sh` and `launch_tycho.sh` auto-detect
  `../results/.git`, read history directly from `../results/`, merge new `out_*` rows into
  `../results/out_<host>_<EXP_ID>.txt`, save logs/reports to `../results/logs/<RUN_TAG>_<TS>/`,
  keep `.npz` locally, and `git commit/push`.
- Verification: 400 mocked and numeric tests pass (`tests/` + `py/check_potential_convergence/`).

## Next steps (proposed, need PI go-ahead for anything expensive)

1. Run the `free-Q` search with refined multipole `N=184, lmax=24` (`bash launch_tycho.sh --l24 --incl=90.0`;
   handoff: `results/logs/Q1d1_rh_bounds/HANDOFF_FREEQ_L24.md`).
2. Compute the `Q=1` J-factor and outer-continuation sensitivity on the consolidated `Q=1` pool
   (`results/out_*_Q1d1_nb250_gh0_ser0.txt`, skill `/jfactor-report`).
3. Later: batch re-score/fetch (tar offsets); Q16 J weighting; recompute 77 lost libraries.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync.
