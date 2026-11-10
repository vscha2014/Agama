# Open questions for the PI

Only **unanswered** questions live here. When the PI answers, move the decision
to `DECISIONS.md` (with its consequence) and delete the entry from this file.
Do not guess answers. Format of a new entry:

```
## Qn. Short title  [affects: contract section / code / analysis]
Context (2–5 lines). Options with trade-offs. Recommendation. What is blocked.
```

Answered so far: Q1–Q14, Q21, Q22, Q28–Q30 and the posang/`q_ap` correction → see `DECISIONS.md`.

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

## Q23. Orbit integration runs on one thread (torch shares AGAMA's libgomp)  [affects: production speed, `_yaVM.py`]

Live check 2026-10-03: during `agama.orbit` one thread at ~100 %, no OpenMP team,
although `OMP_NUM_THREADS=8`. The process maps only
`torch/lib/libgomp.so.1` (same SONAME as the system libgomp ⇒ AGAMA binds to it),
so `torch.set_num_threads(1)` at import calls `omp_set_num_threads(1)` for the
main thread and AGAMA's parallel loops get a team of 1 — confirmed in `agama:latest`:
`omp_get_max_threads()` = 8 without torch, 1 after `torch.set_num_threads(1)`,
8 inside `agama.setNumThreads(8)`. Production has the same
lines (`_yaVM.py:21–22`, before `import agama`). Import order does not help.
Proposed minimal fix (harness first, production only with approval):
`with agama.setNumThreads(N): matrices = agama.orbit(...)` with explicit
`N = args.n_threads or len(os.sched_getaffinity(0))` (`setNumThreads(0)` would
restore 1, the value seen at its first call). IC sampling left outside, so the
libraries stay bit-identical (each orbit row is computed independently).
Validation: rerun the single-model point ⇒ same MD5 and penalty, ~N× faster
integration. The harness fix was applied on 2026-10-03 (user request, see
`DECISIONS.md`) and verified on the VM: bit-identical library (same MD5),
penalty equal to round-off, orbit integration ≈20× faster on 32 vCPU.
**Question:** port the same wrap to `_yaVM.py`?

## Q32. Radial support for J and outer tracers  [affects: CONTRACT / apertures / J]

Q31's beta/rh diagnostic preparation was selected by the user on 2026-11-08
(see DECISIONS); its other proposed steps remain open. The literature and
read-only catalogue audit are private:
`results/logs/BETA_RH_J_LITERATURE_20261108.md`.
An inner Zhao profile mainly constrains `(beta-gh)/rh**alpha`; a cored J
can depend on unobserved halo extent and the assumed 55 kpc cutoff.

Options: (A) separately decompose J by radius using actual alpha/beta, preserving
the full J convention and showing an inner contribution plus outer extrapolation;
(B) prepare an experiment using outer tracers currently excluded by max_r,
with membership/bootstrap validation and new apertures/constraints; old/new
raw penalties would not be directly comparable; (C) introduce an explicitly
justified tidal/cosmological prior, or report a separately defined restricted J.
The last option adds assumptions or changes the quantity being reported.

Recommendation: inspect the prepared beta/rh profiles, then A and B before
tightening outer-halo priors. **Question:** which of these follow-ups should be
implemented? Apertures, outer cutoff, J definitions and production remain frozen.

## Q33. Shared orbit basis across inclinations (incl fan)  [affects: CONTRACT §Stellar component / paper numbers]

A stored `.npz` cannot be reused at another inclination: `TargetLOSVD::addPoint`
applies the `(alpha,beta,gamma)` rotation during integration, so the projection is
baked into `matrix_kinem` and no trajectories are kept. What can be shared is the
part before the projection — potential, orbit ICs and the integration itself,
~97 % of a model (`orbit_s ≈ 400 s` vs `solveOpt ≈ 11 s`). `py/run_incl_fan.py`
(diagnostic, production untouched) integrates once and passes one LOSVD target per
inclination. The observational side is unaffected by `incl` (doc 07 §2, re-checked
numerically), and the aperture masses and `DensityDataset` constraints are rebuilt
from the **exact** `axRZst(i)` of each member, so only two approximations remain:
E1 the stellar quadrupole in `Multipole(rho_star+rho_halo)` is that of the reference
inclination (stars ≈ 10 % of the enclosed mass, only their flattening changes);
E2 the ICs are sampled from the reference stellar density — a basis choice, not a
constraint, since the weights are free. The reference member stays exact.
For 70–90° with reference 80° the frozen shape mis-projects the isophotes by
≤ 4.8 % in `q` (axRZst 0.638…0.690). Nothing has been run yet.

Options: (A) fan only as a screening accelerator for `penalty(incl)` / the shift of
the DM optimum, with the final published numbers recomputed exactly per inclination;
(B) fan numbers published directly, with E1+E2 measured once (paired exact-vs-fan
pilot, ≥ 10 IC seeds against the IC scatter sd ≈ 0.019–0.044) and quoted as a
systematic; (C) no fan at all — every inclination keeps its own full integration.

Recommendation: A, and run the stage-0 diagnostics (`--diagnose`, minutes of CPU,
no orbits) plus the `--benchmark` before committing to anything; B only if the
pilot shows the systematic to be below the IC scatter. **Question:** is a shared
orbit basis acceptable for the article at all, and under which of A/B? Blocked
until answered: any fan run intended to produce numbers for the paper.
