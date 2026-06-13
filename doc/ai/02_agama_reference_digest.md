# AGAMA reference digest (targeted)

Project-oriented digest of `doc/reference.pdf`. The LaTeX source
`doc/reference.tex` is committed and readable as plain text, so no PDF extraction
is needed; section labels below (`\label{sec:...}`) refer to it and can be grepped
directly (`grep -n 'label{sec:Units}' doc/reference.tex`).

## 1. Units and coordinate conventions (`sec:Units`, `sec:Coords`, `sec:CoordinateDetails`)

- AGAMA uses dimensionless internal units with the single convention **G = 1**;
  results must be independent of the choice of base units.
- The Python interface can install a global unit converter (`agama.setUnits`),
  otherwise all quantities are assumed to already satisfy G=1.
- **This project does not call `setUnits`**: it works in G=1 units with length in
  kpc-scaled sky units (`sc = pi*D/180`, D=143 kpc) and velocities divided by
  `vscale = sqrt(2*6.67/3.086)` (see `Fornax_P21_symm_PCA_w3Sersic_yaVM.py`,
  lines ~193–196). Treat this scaling as part of the scientific contract.
- Coordinate systems: Cartesian/Cylindrical/Spherical/ProlSph. Observed system
  XYZ vs intrinsic xyz related by **three Euler angles (alpha, beta, gamma)**
  (`sec:CoordinateDetails`, `doc/EulerAngles.pdf`, `doc/ForstandAngles.pdf`).
  In the Fornax script: `alpha=0`, `beta = incl` (radians), `gamma` from the
  position angle 46.8°.

## 2. Density / potential objects (`sec:Potential`, `sec:PotentialDetails`)

- `agama.Density` / `agama.Potential` accept `type=...` plus named parameters.
- Used in this project:
  - `Density(type='Spheroid', alpha, beta, gamma, axisratioz, densitynorm,
    scaleradius, outercutoffradius, cutoffstrength)` — the DM halo (generalized
    αβγ profile with exponential cutoff). The optimized params map: `gamma=gh`,
    `axisratioz=Q`, `densitynorm=rho0`, `scaleradius=rh`; fixed `alpha=2, beta=3`.
  - `Density(type='Sersic', sersicIndex, mass, scaleRadius, axisRatioZ)` — the
    stellar component (Sersic m=0.80; flattening `axRZst` deprojected from the
    apparent axis ratio `q_ap` and `incl`).
  - `Potential(type='Multipole', density=..., lmax=4, mmax=0, gridSizeR=23)` —
    spherical-harmonic potential expansion of stars+halo (axisymmetric, mmax=0).
- Densities can be combined: `agama.Density(d1, d2)`.

## 3. Visible component / Sersic-related concepts

- Sersic is one of AGAMA's built-in density profiles (potential section of the
  reference). Deprojection of an observed flattening to intrinsic `axisRatioZ`
  given inclination is done in the project script itself:
  `axRZst = sqrt(q_ap^2 - cos^2 i)/sin i`. Confirmed inputs (Wang et al. 2019,
  Sersic): apparent axis ratio `q_ap = 1 - Ellipticity = 1 - 0.31 = 0.69`,
  position angle `posang = 42.3`, `Sersic_m = 0.80 ± 0.006`, `r_s = 16.4′ ± 0.2′`.
  **Physical admissibility of `incl`**: the square root requires
  `cos(incl) < q_ap = 0.69` ⇒ `incl ≳ 46.5°` (the prior on `incl` for Goal 1).
- J-factor good-model weighting (analysis script) is a weighted KDE with
  `w = exp(-(penalty - pen_min)/pen_sigma)`, `pen_sigma = max(std, 1e-6)`,
  normalized; `penalty` is a ranking score, **not χ²**.
- Mass scaling: the stellar mass-to-light/mass normalization enters as
  `Upsilon`; AGAMA's Schwarzschild machinery supports rescaling one orbit
  library to different mass normalizations Υ by rescaling velocities by √Υ
  (`sec:Schwarzschild`, end; details in `sec:SchwarzschildDetails`).

## 4. Orbit integration / orbit library (`sec:Orbits`, `sec:OrbitDetails`, `sec:Schwarzschild`)

- `agama.orbit(potential, ic, time, targets=[...], trajsize, Omega)` integrates
  many orbits (8th-order Runge-Kutta `DOP853`-like; Cartesian in practice) and
  simultaneously records each orbit's contribution matrix for each `Target`.
- Schwarzschild workflow (`sec:Schwarzschild`):
  1. build potential Φ and Target objects;
  2. sample initial conditions (here: `densityStars.sample(numOrbits=100000,
     potential=pot_gal)`);
  3. integrate orbits recording arrays u_{i,n};
  4. solve for non-negative orbit weights minimizing a weighted L2 residual plus
     a quadratic regularization term (`agama.solveOpt(matrix, rhs, rpenq, xpenq)`);
  5. (optional) N-body realization.
- Integration time per orbit is set in dynamical times: `time = pot.Tcirc(ic) *
  intTime` with `intTime=100` in this project.

## 5. Projection and line-of-sight observables (`sec:Schwarzschild`, app. `sec:SchwarzschildDetails`)

- `agama.Target(type='LOSVD', apertures=..., gridx, gridy, gridv, degree,
  symmetry, alpha, beta, gamma, psf, velpsf)` records PSF-convolved LOSVDs in
  B-spline representation over sky apertures (arbitrary polygons).
- Applying the target to an N-body-like point set `target((xv, masses))` yields a
  datacube (apertures × velocity basis), from which Gauss–Hermite moments are
  obtained via `agama.ghMoments(degree, gridv, matrix, ghorder)`.
- GH conventions: see `doc/GHmoments.pdf`. Errors of GH moments via the helper
  `agama.schwarzlib.ghMomentsErrors` (project uses bootstrap of velocity errors,
  n_boot=100).
- Euler-angle conventions for projection: `doc/EulerAnglesProjection.pdf`,
  `doc/ForstandAngles.pdf`.

## 6. Python wrapper usage patterns (`sec:Python`)

- Everything is exposed as `agama.*` classes/functions; docstrings via
  `help(agama.X)`. Most routines are vectorized over points/orbits and use OpenMP
  internally (`OMP_NUM_THREADS` controls threading — the Docker launcher pins it
  to 8 per process).
- Higher-level Schwarzschild helpers used here are not in the C extension but in
  `py/schwarzlib.py` (forstand): `DensityDataset`, `KinemDatasetGH` (with
  `getOrbitMatrix(matrix, Upsilon)`, `getPenalty(superposition, Upsilon)`),
  dataset `cons_val`/`cons_err` arrays.
- `agama.solveOpt` wraps the quadratic optimization (CVXOPT) for orbit weights.
- `agama.nonuniformGrid(nnodes, xmin, xmax)` builds the stretched grids used for
  the LOSVD spatial grids.

## 7. Numerical caveats relevant to the optimization

- Orbit-weight solution is a noisy function of model parameters: finite orbit
  library (1e5 orbits), random IC sampling (`numpy.random.seed(42)` is set, but
  sampling depends on the potential), regularization `regul=1.0`. Penalty is
  therefore stochastic at some level — relevant for GP/BoTorch modelling and for
  judging convergence of the Brent search (xatol=1e-3).
- Multipole expansion with lmax=4 limits how flattened/cuspy models are
  represented; strong cusps (`gh` near upper bound) and small `Q` are the risky
  corners of parameter space.
- Failed evaluations return -1e6 target (penalty 1e6); readers filter
  `penalty >= 1e5` and non-positive `rh`/`rho0`.
- Unit-roundoff-level differences ~1e-4..1e-6 are expected (reference, Units
  section) — don't over-interpret penalty differences below the stochastic floor.

## 8. Mapping: AGAMA concepts ↔ Fornax scripts

| AGAMA concept | Where used in `Fornax_P21_symm_PCA_w3Sersic_yaVM.py` |
|---|---|
| `Density('Sersic')` | `densityStars` (stellar component) |
| `Density('Spheroid')` | `densityHalo` in `halo_IC_lib_weights_pca_fixed` |
| `Potential('Multipole')` | `pot_gal` (stars+halo) |
| `Target('LOSVD', apertures=sectAPP, ...)` | observed/model datacubes, GH moments |
| `agama.ghMoments`, `schwarzlib.ghMomentsErrors` | data GH moments + MC errors |
| `density.sample` | orbit initial conditions |
| `agama.orbit(..., targets=...)` | orbit library + target matrices |
| `agama.solveOpt` | orbit weights inside `find_weights_Ups` |
| Υ velocity rescaling | `d.getOrbitMatrix(m, Upsilon)`, Brent over Upsilon |
| Euler angles (alpha,beta,gamma) | `beta=incl`, `gamma` from posang 46.8° |

## 9. Pointers into the reference

Page numbers vary between builds; use LaTeX labels in `doc/reference.tex`:

- Units: `sec:Units`; Coordinates: `sec:Coords`, details `sec:CoordinateDetails`.
- Potentials/densities: `sec:Potential`, details `sec:PotentialDetails`.
- Orbit integration: `sec:Orbits`, details `sec:OrbitDetails`.
- Schwarzschild modelling: `sec:Schwarzschild`, details appendix
  `sec:SchwarzschildDetails`.
- Python interface: `sec:Python`.
- Supplementary PDFs in `doc/`: `GHmoments.pdf`, `ForstandAngles.pdf`,
  `EulerAngles.pdf`, `EulerAnglesProjection.pdf`.
