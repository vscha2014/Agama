# AGENTS.md — rules for AI agents working in this repository

## 1. Project mission

Schwarzschild orbit-superposition modelling of the Fornax dSph with AGAMA/forstand.
We search for dark-matter halo parameters (`axRZ`/`Q`, `gammah`/`gh`, `rhalo`/`rh`,
`rho0`) and visible-component parameters (`incl`, `Upsilon`) that best match
observed stellar kinematics (minimal `penalty` over GH moments), then derive the
J-factor and its uncertainty. Production runs happen on a Yandex Cloud VM via
Docker; this repo holds the code, input data, and result-file examples.

## 2. Read this first

- `README` — upstream AGAMA overview.
- `Devin_instructions.txt` — the PI's task description (Russian).
- `doc/ai/00_project_context.md` — goals, compute scheme, scientific contract.
- `doc/ai/01_repository_map.md` — what lives where; what is off-limits.
- `doc/ai/02_agama_reference_digest.md` — AGAMA concepts used here (digest of `doc/reference.pdf`).
- `doc/ai/03_current_pipeline.md` — end-to-end pipeline description.
- `doc/ai/04_result_file_formats.md` — result-file formats.
- `doc/ai/05_development_environment.md` — the `.venv-ai` dev environment.
- `doc/ai/questions_for_pi.md` — open questions; add new ones here, never guess.
- `doc/ai/skills/` — short skill notes.

## 3. Think before coding

- No hidden assumptions. If you assume something, write it down explicitly
  (in the task notes or `doc/ai/`).
- If several interpretations of a requirement or file format exist, list them
  explicitly; pick one only with justification, or ask.
- If something is unclear and the answer changes the design — **stop and ask**
  (add to `doc/ai/questions_for_pi.md`).
- Before starting a task, write a short plan (steps + expected outcome).
- Every step must have a verification criterion: how will you know it worked?

## 4. Simplicity first

- Minimum code that solves the task.
- No speculative features ("might be useful later").
- No abstractions for one-off tasks.
- No configurability that nobody requested.
- If a solution fits in 50 lines, do not write 200.

## 5. Surgical changes

- Change only what the task requires.
- Do not refactor neighboring code "while you're there".
- Do not reformat large files (no mass whitespace/style diffs).
- Do not delete unrelated dead code.
- Clean up only after your own changes (temp files, debug prints you added).

## 6. Goal-driven execution

- Task → explicit success criteria, written before implementation.
- Bug → reproduce with a test or minimal repro first, then fix.
- Change → verification (test, dry-run, diff inspection) before declaring done.
- Multi-step tasks: execute with a verification loop — verify each step before
  building the next on top of it.

## 7. Scientific contracts (do not change without PI approval)

- `penalty` definition and computation path (orbit-weight solve, GH dataset term).
- Aperture construction (sectors, binning, symmetrization, `max_r`).
- Gauss–Hermite conventions (`ghorder`, `degree`, moment indices, bootstrap errors).
- Brent (bounded scalar) search over `Upsilon` and its bounds/tolerances.
- J-factor formula and integration settings.
- Units and scaling conventions (`vscale`, distance, position angle).
  Confirmed geometry (Wang et al. 2019, Sersic; PI-approved correction):
  `posang = 42.3`, `q_ap = 1 − 0.31 = 0.69`, `D = 143 kpc`. (Old, wrong:
  `posang = 46.8`, `q_ap = 0.7`.)
- Input observational data (`py/table3.dat`) — frozen.
- Sersic assumptions (`Sersic_m = 0.80 ± 0.006`, `massSt = 14.0` frozen,
  scale radius 16.4′ ± 0.2′, flattening deprojection `axRZst`). Sersic-error
  propagation (Goal 3) is a **separate** future code, not edits to the main script.
- Production parameter bounds (`bounds_original` in the main script).
- RNG policy: torch (BoTorch) gets a per-process seed (logged); a per-process
  `proc_rng` decorrelates optimizer seeding (initial-point selection, Goal 0);
  the GH observation-error bootstrap is **fixed at `seed=42`** (identical across
  workers). None of these affect AGAMA orbit-IC sampling (`Density.sample` uses
  AGAMA's own RNG, `agama.setRandomSeed` not called → orbit libraries identical
  across processes). Do not re-pin a global numpy seed.
- Result-file naming: canonical = `4UpsBoTorch_PCA_Sersic_*` /
  `4result_BoTorch_PCA_Sersic_*`; wrong-posang archive = `*_PA46.8_*`. The
  canonical glob does not match the archive. Use **all** result files; **never
  dedup across files/hosts** (unique experiment results).

An agent **may recommend** extending parameter bounds (or other contract changes)
in documentation/analysis reports, but must **not** change them in production
code without explicit approval.

## 8. Expensive execution policy

Forbidden without an explicit user request:

- production optimization runs (`Fornax_P21_symm_PCA_w3Sersic_yaVM.py` full run);
- full Docker/Yandex VM launches (`launch_docker_parallel.sh`);
- long BoTorch/TuRBO runs;
- orbit-library regeneration;
- mass rewriting of result files.

Allowed without asking:

- static inspection (grep, AST, reading code/data);
- reading any repo files;
- small dry-runs **if the script already supports them** (do not add dry-run
  modes just to run something);
- `--help` invocations;
- `python -m py_compile`;
- unit/simulation tests that do not trigger production computation.

## 9. Generated data policy

- Never commit `.venv-ai/` (or any venv).
- Never commit extracted full text of `doc/reference.pdf` (temporary extraction
  only under gitignored `doc/ai/_tmp/`).
- Never commit logs, results, checkpoints (`*.pkl`), sqlite files, run artifacts.
- Never delete the committed result-file examples (`py/4Ups*`, `py/4result*`).
- New generated artifacts go to a gitignored path (extend `.gitignore`
  minimally and with justification).

## 10. Yandex shared-run constraint

Hard constraint for any future harness/orchestration design:

- Parallel processes **must see each other's results as they appear** (live,
  during the run) — today this works via shared workspace files + Yandex.Disk
  sync read at PCA-update time.
- Do not design a future harness where results become visible only after the
  whole run finishes (no buffering results until completion).
- Future duplicate-prevention must be **preventive** (e.g., reserving candidate
  points before evaluation), not post-hoc cleanup of duplicated computations.
