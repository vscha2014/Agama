# Repository map

This repo is a clone of AGAMA (https://github.com/GalacticDynamics-Oxford/Agama)
plus project-specific files for the Fornax Schwarzschild-modelling project.
Only the parts relevant to this project are listed.

## Top level

| Path | Role |
|---|---|
| `Devin_instructions.txt` | PI's task description (Russian). Project goals 0–3. |
| `AGENTS.md` | Development rules for AI agents (read first). |
| `README`, `INSTALL`, `NEWS` | Upstream AGAMA docs. |
| `Makefile`, `Makefile.*`, `setup.py`, `Doxyfile` | Upstream AGAMA build system (C++ library + Python extension). |
| `Dockerfile` | Production image (`agama:latest`): builds AGAMA + GLPK/GSL/UNSIO + torch/botorch/sklearn/cvxopt + rclone. Used by `py/launch_docker_parallel.sh`. |
| `check_agama.py`, `check_imports.py` | Dockerfile build-time sanity checks (AGAMA C++/Python, all runtime imports). |
| `entrypoint.sh` | Docker entrypoint: maps host UID/GID, sets `RCLONE_CONFIG`, runs as host user. |
| `Wang_2019_table1_apjab31a9t1_ascii.txt` | Fornax structural params (Wang et al. 2019). Source of `posang=42.3±0.2`, Ellipticity `0.31±0.002` (⇒ `q_ap=0.69`), Sersic `m=0.80±0.006`, `r_s=16.4′±0.2′`. |
| `requirements-ai.txt` | Lightweight AI-dev venv deps (`.venv-ai`); NOT production deps. |
| `src/` | Upstream AGAMA C++ source (~163 files). **Do not modify.** |
| `tests/` | Upstream AGAMA C++ tests. |
| `data/` | Upstream AGAMA sample data. |
| `doc/` | AGAMA documentation: `reference.pdf` (147 pp.), `reference.tex` (LaTeX source — readable as text), topical PDFs (`GHmoments.pdf`, `ForstandAngles.pdf`, ...). |
| `doc/ai/` | AI-agent knowledge base (this directory). |
| `.gitignore` | Ignores build artifacts, `.venv-ai/`, `doc/ai/_tmp/`. |

## `py/` — Python layer (project lives here)

### Project source code (this project)

| File | Role |
|---|---|
| `py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py` | **Main production script.** Data loading, apertures, GH moments, TuRBO/BoTorch-PCA optimization over halo params, Brent over Upsilon, result-file writing, checkpointing, Yandex.Disk sync, ntfy notifications. **Scientific contract — do not modify without approval.** |
| `py/J_factor_Sersic_Fornax_P21_symm.py` | Analysis script: reads `4UpsBoTorch_*` history files, percentile filtering, J-factor computation and plots. **Scientific contract — do not modify without approval.** Note: hardcoded `YADISK_DIR=/home/gala/Yandex.Disk/galAgama`. |
| `py/launch_docker_parallel.sh` | VM orchestration: 4 Docker containers, file merging (MD5 dedup), rclone upload/download, resume, shutdown. **Do not modify without approval; never run in dev.** |

### Input data (frozen)

| File | Role |
|---|---|
| `py/table3.dat` | Observed stellar radial velocities of Fornax members (id, RA, Dec, flags, v_los, err_v, ..., membership prob in col 9). **Frozen — never edit.** |

### Computation results (examples; do not delete or regenerate)

| File | Role |
|---|---|
| `py/4UpsBoTorch_PCA_Sersic_<host>[_pN].txt` | **Canonical** evaluation history (correct-posang runs). Data lines: `incl Q gh rh rho0 Upsilon penalty timestamp`. Use all files; never dedup. |
| `py/4result_BoTorch_PCA_Sersic_<host>[_pN].txt` | Canonical optimization logs (monitoring only). |
| `py/4UpsBoTorch_PCA_PA46.8_Sersic_<host>.txt` | **Archive** of wrong-posang (46.8°, `q_ap=0.7`) results. NOT read by canonical globs. Used only as opt-in seed source (`--init-from-pa468`) with penalties recomputed. Do not delete. |
| `py/4result_BoTorch_PCA_PA46.8_Sersic_<host>.txt` | Archive of wrong-posang logs. Do not delete. |

### Upstream AGAMA Python (used as library; do not modify)

| File | Role |
|---|---|
| `py/schwarzlib.py` | forstand helper library: `DensityDataset`, `KinemDatasetGH`, `ghMomentsErrors`, penalty machinery. |
| `py/pygama.py`, `py/schwarzschild.py` | AGAMA Python utilities / older Schwarzschild driver. |
| `py/example_forstand.py`, `py/example_*.py`, `py/test_*.py`, `py/tutorial_*.ipynb` | Upstream examples/tests/tutorials (useful reading, not part of the project pipeline). |

## What must not change without approval

- `src/` (all C++), `py/schwarzlib.py`, `py/pygama.py` and other upstream AGAMA code;
- the three project files above (`Fornax_*`, `J_factor_*`, `launch_docker_parallel.sh`);
- `py/table3.dat` and other input data;
- committed `4Ups*/4result*` example files (do not delete or rewrite).

## What is OK to change in future engineering tasks

- `doc/ai/**` (this knowledge base), `AGENTS.md`;
- `requirements-ai.txt`, `.gitignore` (minimal additions);
- new analysis/harness code in **new files**, after agreeing the design with the PI;
- nothing that alters the scientific contract (see `00_project_context.md`).

## Files that were checked and not found

- No `Dockerfile` in the repo (the `agama:latest` image used by
  `launch_docker_parallel.sh` is built/maintained outside this repo).
- No `requirements*.txt` / `pyproject.toml` (build is via `setup.py`/`Makefile`).
  `requirements-ai.txt` (repo root) was added in Stage-0 for the lightweight AI venv only.
