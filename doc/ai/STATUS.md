# Project status

Rewritten (not appended) at the end of every task; ≤ 60 lines. Update: 2026-10-25 (paired `--models=FILE --ic-seeds=1-20` and `--protocols=exp,reuse,prod` batch modes added to `launch_tycho.sh` / `run_single_model.py` for comparing the `free-Q` and `Q=1` finalists on `tycho`).

## Where the science stands

- Free-Q baseline (`N=23, lmax=4`): best `penalty ≈ 1.24` at `incl = 89.5°`,
  `log10 J(0.5°) ≈ 18.58` — production free-Q history (`4UpsBoTorch_PCA_Sersic_*`,
  always doubled ⇒ **d1**).
- Free-Q on converged multipole (`N=184, lmax=24`, `--l24` at `incl = 90°` on tycho,
  `results/out_tycho_d1_nb250_gh0_l24n184_ser0.txt`): 10 prior-seed evaluations plus
  40 TuRBO-PCA iterations completed with zero boundary hits on `(Q, rh, rho0, Upsilon)`
  and `gh ≈ 0` (core). The optimum parameters `(Q, gh, rh, rho0, Upsilon)` and
  `log10 J(0.5°)` are invariant relative to the `lmax=4` valley, while the converged
  `free-Q` objective remains systematically better than spherical `Q=1` both in best
  realisation and in cluster mean, with a compact physical scale radius `rh`.
  Details: local `results/logs/d1_nb250_gh0_l24n184_ser0_i90.0_20261022_171603/NOTES.md`.
- Fixed spherical halo (`Q=1`): four targeted probe rounds (2026-10-05/06/15/22) and a
  40-iteration TuRBO-PCA search on tycho (2026-10-16, `results/logs/spherical_free/`)
  bracket the `Q=1, incl=90°` minimum (`gh ≈ 0`, uniform-core plateau `rh ≈ 25–30` kpc,
  interior `rho0` optimum, invariant `rho0·Upsilon`). All `Q=1` rows live in
  `results/out_*_Q1d1_nb250_gh0_ser0.txt`.

## Code state

- Harness: `--Q1`, `--l24` (`--pot-lmax=24 --pot-ngrid=184`, isolated
  `HISTORY_ID = d1_nb250_gh0_l24n184_ser0`, zero-overhead `pot_orbit` while preserving
  baseline `pot_gal` for the Step 1A/1B AST parser and `diagnostic` callback, `Q=1`-preserving
  `orblib_key`, `free-Q` seeding from `prior_seed_patterns` = `4Ups*` + old `out_*_d1_*` with
  orbit/penalty recomputation), widened bounds, aperture-vertex grid, `std_floor` before
  `< 1e-10 → 1.0`, `SIGTERM` TuRBO checkpointing, and `ORBLIB_LOCAL_STORE=1` for `--offline`.
- Git-backed `results/` mode (`harness/orblib_exp.md` §3g–§3h): launchers auto-detect
  `../results/.git`, read history from `../results/`, merge new `out_*` rows into
  `../results/out_<host>_<EXP_ID>.txt`, save logs to `../results/logs/[seed/]<RUN_TAG>_<TS>/`,
  support paired `--models=FILE --ic-seeds=SPEC` and `--protocols=exp,reuse,prod` batches on
  `tycho` with `by_model`/`paired` `dP` summary, keep `.npz` locally, and `git commit/push`.
- Verification: 404 mocked and numeric tests pass (`tests/` + `py/check_potential_convergence/`).

## Next steps (proposed, need PI go-ahead for anything expensive)

1. Run the prepared finalist batch on `tycho` (`--l24 --models=models_finalists_q1_freeq.txt`
   with `--protocols=exp,reuse,prod` and `--ic-seeds=1-20`) to measure paired `ΔP(seed)`.
2. Compute and compare `J`-factors and outer-continuation sensitivity for the completed
   `Q=1` and `--l24` `free-Q` pools (`/jfactor-report`), and update article text (`/paper-writing`).
3. Later: check other `incl` under `--l24`; batch re-score/fetch; Q16 J weighting; 77 lost `.npz`.

## Open questions

Q15 (Upsilon Brent into production), Q16 (J weighting / sampling density), Q17–Q19
(shard consolidation, `ic`/`inttime`, shard naming), Q20 (production `pipefail`), Q23
(OpenMP wrap in production) — see `questions_for_pi.md`.
Decided (`DECISIONS.md`): Q21, Q22, Q28, Q29, Q30, TR bound, PCA std floor, git-results sync.
