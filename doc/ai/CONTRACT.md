# Scientific contract

Single source of truth for the frozen scientific definitions of this project.
**Nothing here may be changed without explicit PI approval.** An agent may
*recommend* a change in a report or in `questions_for_pi.md`; it must not edit
production code to implement one. Answers that created these entries are logged
in `DECISIONS.md`.

## Frozen inputs

- `py/table3.dat` — observed Fornax radial velocities (Pace et al. 2021).
  Never edited. Membership filter: column 9 > 0.
- Distance `D = 143 kpc`; velocity scale `vscale = sqrt(2*6.67/3.086)`
  (code masses are therefore in 1e6 M☉).
- Geometry (Wang et al. 2019, Sersic): `posang = 42.3`, `q_ap = 1 − 0.31 = 0.69`.
  Superseded wrong values: `posang = 46.8`, `q_ap = 0.7` (their results live in
  the `*_PA46.8_*` archive).
- Stellar component: Sersic `m = 0.80 ± 0.006`, `massSt = 14.0` (frozen),
  scale radius `16.4′ ± 0.2′`, deprojection
  `axRZst = sqrt(q_ap² − cos² incl) / sin incl` ⇒ admissible `cos incl < q_ap`
  (`incl ≳ 46.5°`).
- Halo Spheroid: `alphah = 2.0`, `betah = 3`, `outercutoffradius = 55.0`,
  `cutoffstrength = 2.5`; potential via `Multipole` (`lmax=4`, `mmax=0`).

## Observables and penalty

- Apertures: central ellipse + 8 angular sectors, radial bins of ~`n_bin` stars
  (production 250), `max_r = 2.1`, point-symmetrised sample (doubling).
- Gauss–Hermite: `ghorder = 6`, `degree = 2`, moment indices `ind=(1,2,6,7,8,9)`;
  observation-error bootstrap `n_boot = 100` at fixed `seed = 42`.
- `penalty` = `sum(penalties[1])`, the GH-kinematic term of
  `KinemDatasetGH.getPenalty` after `agama.solveOpt` (`regul=1.0`,
  `mult = sqrt(num_dof)*10`). **It is not a χ²** — a relative ranking score only.
  Penalties are comparable across `incl` (the observational side is independent
  of `incl`) but **not** across different `double` / `n_bin` settings.
- `Upsilon`: bounded Brent (`minimize_scalar`) on `[0.1, 1.6]`, `xatol = 1e-3`,
  `maxiter = 50`. The `_timed`/`orblib_exp` variants additionally use an
  adaptive bracket, `UPS_XATOL = 5e-3` and orbit sub-sampling with a final
  full solve; porting that into production needs PI approval (Q15, open).

## Search space

Production bounds (`bounds_original`, defined in **two** places in each script —
keep them in sync):

| parameter | bounds | note |
|---|---|---|
| `Q` (`axRZ`) | 0.05 – 2.5 | fixed to 1 in `--Q1` mode |
| `gh` (`gammah`) | 0.0 – 1.6 | negative values are not admissible |
| `rh` (`rhalo`) | 0.5 – 7.0 kpc | widened 2026-09 from 3.5 (diagnostic, see DECISIONS) |
| `rho0` | 10 – 120 | widened 2026-09 from 34 |
| `Upsilon` | 0.1 – 1.6 | |

## J-factor

- Rebuild the halo density of each selected model with `densitynorm = rho0*Upsilon`
  (stars: `mass = massSt*Upsilon`), integrate ρ² along the line of sight over
  cones θ ∈ {0.1, 0.2, 0.5, 1.0}°, D = 143 kpc.
- Unit conversion: `rho_conv = 1e6 * 1.989e33 / 1.783e-24 / kpc_to_cm**3`,
  `J = J_code * rho_conv**2 * kpc_to_cm`.
- Good-model weighting: `w = exp(-(penalty − pen_min) / pen_sigma)`,
  `pen_sigma = max(penalties.std(), 1e-6)`, normalised; adaptive cutoff keeps
  the best `target_fraction = 0.30` (ceiling `cutoff_start = 0.60`).
- Whether to correct for the optimizer's sampling density (importance weights)
  is **open** — Q16 in `questions_for_pi.md`.

## Randomness

- torch/BoTorch: per-process seed (logged). `proc_rng`: per-process RNG for
  optimizer seeding / initial-point selection only.
- GH observation-error bootstrap: fixed `seed = 42`, identical across workers.
- AGAMA orbit-IC sampling uses AGAMA's own RNG (`agama.setRandomSeed` is never
  called). Do not re-pin a global numpy seed.

## Result data

- Data line of `4UpsBoTorch_PCA_Sersic_*.txt` / `out_*.txt`:
  `incl Q gh rh rho0 Upsilon penalty timestamp`.
- Use **all** files (per-process `_pN` and merged per-host, all hosts).
  **Never deduplicate** stored rows — each row is a unique experiment result.
  (Collapsing duplicates inside a *statistical* estimate is a separate question,
  Q16; it does not license deleting rows.)
- Canonical names `4UpsBoTorch_PCA_Sersic_*` / `4result_BoTorch_PCA_Sersic_*`;
  the wrong-posang archive `*_PA46.8_*` is not matched by canonical globs and
  may only seed initial points with penalties recomputed (`--init-from-pa468`).
- `4result_*` logs are monitoring-only.
