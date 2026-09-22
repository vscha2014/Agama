# Project context and goals

Schwarzschild orbit-superposition modelling of the Fornax dSph with AGAMA
(forstand). For a fixed inclination the sky plane is split into apertures,
Gauss–Hermite moments are compared between model and data (`py/table3.dat`), and
the optimizer searches the halo parameters `Q, gh, rh, rho0` (plus `Upsilon` by
an inner Brent search) that minimise `penalty`. The end products are the halo
parameters, the inclination, and the J-factor with its uncertainty.

Frozen definitions: `CONTRACT.md`. How runs are executed: `harness/production.md`
(production) and `harness/orblib_exp.md` (experiments). Current state and next
steps: `STATUS.md`.

## Goals (PI, `Devin_instructions.txt`)

- **Goal 0** — prevent duplicate evaluations between parallel workers.
  *Partly done*: per-process `proc_rng` decorrelates initial-point selection and
  the LHS fallback (overlap 100 % → ~40 % in a 40→12 test); the main loop is
  decorrelated by the per-process torch seed. Preventive reservations exist in
  the experimental harness; production has no hard reservation.
- **Goal 1** — treat `incl` as an analysis parameter instead of running one
  inclination at a time. Penalties are comparable across `incl` (the
  observational side does not depend on it, see `07_incl_penalty_analysis.md`);
  the physical prior is `cos incl < q_ap`.
- **Goal 2** — decide whether the explored parameter space is sufficient.
  *In progress*: `rh` and `rho0` bounds were widened after the `Q=1` run pinned
  `rh` at its upper bound; no run has used the new bounds yet.
- **Goal 3** — propagate input-data errors (velocity errors → GH moments via
  Monte Carlo; Sersic parameter errors from Wang et al. 2019) into the
  good-model region and the J-factor bounds. *Not started*; per PI this must be
  **separate new code**, not edits to the main script. The `gh_id` / `ser_id`
  slots in the experimental script are the placeholders for it (currently
  `NotImplementedError` for any value > 0).
