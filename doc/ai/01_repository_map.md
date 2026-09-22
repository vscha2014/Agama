# Repository map

A clone of AGAMA (GalacticDynamics-Oxford/Agama) plus this project's files.
Only project-relevant parts are listed. The repository is **public**: nothing
from `paper/` or `results/<run_id>/` may end up in tracked files.

## Project code (`py/`)

| File | Role |
|---|---|
| `Fornax_P21_symm_PCA_w3Sersic_yaVM.py` | **Production** optimizer. Frozen (`CONTRACT.md`). |
| `Fornax_P21_symm_PCA_w3Sersic_yaVM_timed.py` | Production copy with timing **and** Upsilon speed-ups (Q15). |
| `J_factor_Sersic_Fornax_P21_symm.py` | **Production** J-factor / plots. Frozen. Hardcoded `YADISK_DIR` is intentional. |
| `launch_docker_parallel.sh` | **Production** VM orchestrator. Frozen. Never run in dev. |
| `Fornax_P21_PCA_w3Sersic_orblib_exp.py` | Experimental harness (doubling / `n_bin` / `--Q1` / orbit libraries). Editable. |
| `launch_orblib_exp.sh`, `orblib_storage.py` | Experimental orchestrator and orbit-library delivery. Editable. |
| `diagnose_J_weighting.py`, `mass_histogram_all_incl.py`, `profile_likelihood_corner.py`, `filter_orblib_by_penalty.py` | Analysis helpers. |
| `table3.dat` | Observed velocities. **Frozen — never edit.** |
| `schwarzlib.py`, `pygama.py`, `schwarzschild.py`, `example_*`, `test_*`, `tutorial_*` | Upstream AGAMA. Do not modify. |

Run outputs (`4Ups*`, `4result*`, `out_*`, `log_*`, `orblib/`) are gitignored and
therefore **unreadable by the `read` tool** — use `exec` (`cat`, `python`) for
them, or keep curated copies under `results/`.

## Project directories

| Path | Role |
|---|---|
| `AGENTS.md` | Always-on rules (≤120 lines). |
| `doc/ai/` | Agent knowledge base (this directory). |
| `results/` | Run registry (`REGISTRY.md`) and per-run notes. Local-only, never committed. |
| `paper/` | Article drafts: nested **private** git repo, excluded locally. |
| `scripts/setup_local_excludes.sh` | Registers `results/` and `paper/` in `.git/info/exclude` (run once per clone). |
| `tests/test_orblib_*.py` | Light mocked tests for the experimental harness. |
| `.devin/skills/`, `.devin/rules/` | Task skills and the frozen-file rule. |
| `Dockerfile`, `entrypoint.sh`, `check_agama.py`, `check_imports.py` | Production image `agama:latest`. |
| `Wang_2019_table1_apjab31a9t1_ascii.txt` | Source of `posang`, ellipticity, Sersic parameters. |
| `requirements-ai.txt`, `.venv-ai/` | Lightweight dev venv (not production deps). |
| `src/`, `data/`, `doc/*.pdf`, `Makefile*`, `setup.py` | Upstream AGAMA. Do not modify. |

## Change policy

Editable: `doc/ai/**`, `AGENTS.md`, `.devin/**`, `results/**` (local),
`scripts/**`, the experimental harness and its tests, `.gitignore` (minimal
additions), new analysis code in **new** files.
Everything else needs PI approval — see `CONTRACT.md`.
