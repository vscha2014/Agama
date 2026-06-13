# Development environment (AI tasks)

## 1. OS assumptions

Linux dev-machine (this repo at `/home/wind/Agama`). Production runs happen on a
separate Yandex Cloud VM in Docker (image `agama:latest`, built outside this
repo) — never on the dev machine.

## 2. Python version

System `python3` is **Python 3.12.3** (`/usr/bin/python3`). The AI venv is built
on it.

## 3. Existing dependency mechanisms (do not duplicate)

- The AGAMA C++ library + Python extension are built via `setup.py` / `Makefile`
  (see `INSTALL`). The compiled `agama` module is **not required** for Stage-0
  documentation/static-analysis work and is not installed in the AI venv.
- Production runtime deps live in the `agama:latest` Docker image, built from the
  repo-root `Dockerfile` (Ubuntu 24.04 + GLPK/GSL/UNSIO + AGAMA + torch/botorch/
  gpytorch/scikit-learn/cvxopt/mgefit/powerbin/requests + rclone). Build-time
  sanity checks: `check_agama.py` (AGAMA C++/Python) and `check_imports.py`
  (all runtime imports). There is no committed `requirements.txt`/`pyproject.toml`.
- `requirements-ai.txt` (repo root) is the minimal AI-dev layer only.

## 4. Creating the venv

```bash
cd /home/wind/Agama
python3 -m venv .venv-ai
source .venv-ai/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ai.txt
```

## 5. What the AI-venv dependencies are for

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | reading/analyzing tabular result files, small numeric checks |
| `pandas` | tabular analysis of `4UpsBoTorch_*` histories |
| `matplotlib` | small diagnostic plots |
| `pyyaml` | reading/writing small config/metadata files |
| `pypdf` | reading `doc/reference.pdf` and other PDFs when the `.tex` source is insufficient |
| `pytest` | lightweight unit/format tests |

Note: `doc/reference.tex` (LaTeX source of the reference) is committed, so most
documentation lookups need only grep — prefer it over PDF extraction.

## 6. Production/runtime-only dependencies (NOT in the AI venv)

Required by `Fornax_P21_symm_PCA_w3Sersic_yaVM.py` / `J_factor_*.py` at runtime,
provided by the production Docker image; do not install in `.venv-ai` unless a
specific task needs them:

- `torch`, `botorch`, `gpytorch` (BoTorch/TuRBO optimization);
- `scikit-learn` (PCA/StandardScaler);
- `requests` (ntfy notifications);
- compiled `agama` (C++ extension) and its CVXOPT dependency;
- `rclone` (system binary, Yandex.Disk sync).

Consequence: the production scripts cannot be imported in the AI venv (imports
fail at `torch`/`agama`). Static analysis (`py_compile`, grep, AST) works fine.

## 7. Activating / deactivating

```bash
source /home/wind/Agama/.venv-ai/bin/activate
# ... work ...
deactivate
```

## 8. Verifying the environment

```bash
source .venv-ai/bin/activate
python - <<'PY'
import numpy, scipy, pandas, matplotlib, yaml, pypdf
print("ai dev environment ok")
PY
python -m py_compile py/J_factor_Sersic_Fornax_P21_symm.py py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py
```

## 9. What must not be committed

- `.venv-ai/` (gitignored);
- extracted full text of `doc/reference.pdf` (use `doc/ai/_tmp/`, gitignored,
  for any temporary extraction);
- logs, checkpoints (`*.pkl`), computation results, sqlite/run artifacts;
- `__pycache__`/`*.pyc` (already gitignored).
