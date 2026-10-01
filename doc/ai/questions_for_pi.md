# Open questions for the PI

Only **unanswered** questions live here. When the PI answers, move the decision
to `DECISIONS.md` (with its consequence) and delete the entry from this file.
Do not guess answers. Format of a new entry:

```
## Qn. Short title  [affects: contract section / code / analysis]
Context (2–5 lines). Options with trade-offs. Recommendation. What is blocked.
```

Answered so far: Q1–Q14 and the posang/`q_ap` correction → see `DECISIONS.md`.

---

## Q15. Porting the Upsilon Brent speed-ups into production  [affects: CONTRACT §Upsilon]

Profiling (`incl=90`, 6 evaluations): `solveOpt` costs 3.6–5.1 s per probe, and a
full Brent search takes 10–13 probes (~44–52 s per model). The optimum is very
stable across the halo parameter space (`Upsilon* ≈ 0.617 ± 0.011`), while the
fixed golden-section probes at `Upsilon ≈ 1.03` and `0.45` are always useless
and the last 2–4 probes refine the 3rd–4th digit at `Δpenalty < 1e-3`.

Already implemented in `..._yaVM_timed.py` and inherited by `orblib_exp.py`:
adaptive bracket around the median of recent `Upsilon*` (`UPS_BRACKET_DELTA=0.1`,
`UPS_BRACKET_NMED=8`, with fallback to the full range if the minimum hits the
narrow edge); `UPS_XATOL=5e-3` (max penalty error ~0.015, < 2 % of the KDE
`pen_sigma`); orbit sub-sampling `UPS_SUBSAMPLE_FRAC=0.25` for the inner search
plus one full-library solve at the found point (the recorded penalty stays exact,
but the final solve bypasses the logger and is absent from the Upsilon history).

**Question:** may these be ported into `Fornax_P21_symm_PCA_w3Sersic_yaVM.py`?
Validation plan before porting: N control points with and without the options,
comparing `Upsilon*` (must agree within `xatol`), `penalty` and the final
J-factor.

## Q16. J-factor weighting: account for the optimizer's sampling density?  [affects: CONTRACT §J-factor]

Current weights use quality only: `w = exp(-(penalty − pen_min)/pen_sigma)`.
But the sample was produced by BoTorch/TuRBO, so the search density `q(θ)`
itself concentrates near the optimum; points there dominate both by quality and
by count, biasing J towards the best fit and shrinking its spread. J-factor
tables contain exactly duplicated parameter rows. (This concerns *statistical
processing*, not storage — the "never dedup" rule in `CONTRACT.md` is about
keeping files intact; please confirm there is no conflict.)

Candidates: (A) profile over penalty — centre at `min penalty`, interval over
`{θ: penalty ≤ pen_min + Δ}`, independent of sampling density but needing a
calibration of Δ (hard, since `penalty` is not a χ²); (B) importance weights
`ω_i ∝ L(θ_i)/q̂(θ_i)` with `q̂` estimated by kNN in `(Q, gh, rh, rho0)`;
(C) full posterior (MCMC/nested) over a penalty surrogate — rigorous, expensive.

Only the diagnostic `py/diagnose_J_weighting.py` exists; it compares the current
method with A and B and changes nothing in production. **Blocked:** the final
J-factor number and its uncertainty for the article.

## Q17. Consolidation policy for legacy orbit-library tar shards

Each old `launch_orblib_exp.sh` run uploaded its own
`orblib_{KEY}__{host}_{timestamp}.tar`. With streaming storage in place, when and
by whom should the remaining shards be merged or retired (threshold by
count/size? manual offline step? delete sources after verification)?

## Q18. Keep `ic` and `inttime` inside the `.npz` libraries?

They are only consumed by `agama.orbit`, which reuse skips; nothing in
solve/penalty/Upsilon reads them afterwards. Cost ≈ 4.8 MB (float64, 100k
orbits) plus `inttime`, i.e. < 1 % of a ~500 MB library. The only argument for
keeping them: orbit ICs are stochastic, so they are the only way to reproduce
identical orbits for reintegration at a different `trajsize` or for
phase-space diagnostics. **Question:** is any such analysis planned?

## Q19. Shard file-name format

Currently `{HOSTNAME_ENV}_{TIMESTAMP}` = `$(hostname)_YYYYmmdd_HHMMSS`, as in
`launch_docker_parallel.sh`. Confirm the hostname must equal `HOSTNAME_ENV` and
that the timestamp format is acceptable.

## Q20. Same `pipefail` bug in the production launcher

`DONE_COUNT=$(ls .../.done_* 2>/dev/null | wc -l)` under `set -euo pipefail`:
if no container produced a marker (all crashed), the failed glob aborts the
whole launcher, skipping merging, the final upload, log sync and notifications.
Reproduced with mocked docker/rclone/curl. Fixed in `launch_orblib_exp.sh`
(count derived from the tracked `FAILED`, per-container notifications, master
log uploaded right after step 3, `curl -f` checked). The identical construct is
still in `py/launch_docker_parallel.sh`. **Question:** approve a separate task
to fix production?
