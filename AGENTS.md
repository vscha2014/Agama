# AGENTS.md

Schwarzschild orbit-superposition modelling of the Fornax dSph with AGAMA/forstand:
search dark-matter halo parameters that best reproduce the observed stellar
kinematics (minimal `penalty`), then derive the J-factor and its uncertainty.
Production runs happen on a Yandex Cloud VM via Docker; this repo holds code,
input data and curated results. PI's task description: `Devin_instructions.txt` (RU).

This repository is a **public** clone of upstream AGAMA. `results/` (run registry
and run notes) and `paper/` (article drafts) are local-only — never commit them
or quote their numbers into tracked files.

## Read first

1. `doc/ai/STATUS.md` — where the project stands, next steps (≤60 lines).
2. `doc/ai/CONTRACT.md` — frozen scientific definitions.
3. `doc/ai/01_repository_map.md` — what lives where; `doc/ai/README.md` — index.

New clone: run `bash scripts/setup_local_excludes.sh` once.

## Hard rules

- **Frozen — never edit without explicit PI approval:** `py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py`,
  `py/J_factor_Sersic_Fornax_P21_symm.py`, `py/launch_docker_parallel.sh`,
  `py/table3.dat`, `src/**`, `py/schwarzlib.py`, `py/pygama.py`, and every item
  in `doc/ai/CONTRACT.md` (penalty, apertures, GH conventions, Upsilon search,
  J-factor, units, Sersic assumptions, parameter bounds). You may *recommend*
  a change; propose it as a diff in your report, do not apply it.
- **Never run without an explicit user request:** production optimisation,
  `launch_docker_parallel.sh` / `launch_orblib_exp.sh`, long BoTorch/TuRBO runs,
  orbit-library regeneration, any real rclone/Yandex.Disk or VM operation
  (`orblib_storage.py index`, `prune-verified --apply`, `sudo shutdown`).
  Always allowed: reading, grep, `--help`, `python -m py_compile`, `bash -n`,
  the mocked tests below.
- **Never delete or rewrite experiment data.** Result rows are unique
  experiment results — no deduplication across files or hosts. Never delete a
  VM or disk holding undelivered orbit libraries.
- **Never commit** venvs, logs, results, checkpoints, `.npz`/`.pkl`, extracted
  PDF text. New artefacts go to a gitignored or locally excluded path.
- Gitignored files (`py/out_*`, `py/4Ups*`, `py/orblib/`, …) are invisible to
  the `read` tool — use `exec` (`cat`, `python`) for them.

## Workflow

1. Plan first: steps, explicit success criteria, and how each step is verified.
   Write assumptions down; if an ambiguity changes the design, ask instead.
2. Surgical changes only: no refactoring, reformatting or speculative features
   in neighbouring code. Minimum code that solves the task.
3. Ask the PI by appending a numbered entry to `doc/ai/questions_for_pi.md`
   (context, options with trade-offs, recommendation, what is blocked). When the
   answer arrives: move it to `doc/ai/DECISIONS.md` and delete the open entry.
4. Verify before declaring done (see commands below); production scripts must
   show an empty diff unless the task explicitly required changes.
5. End of task: rewrite `doc/ai/STATUS.md`, append to `doc/ai/DECISIONS.md` /
   `results/REGISTRY.md` if anything was decided or run. Report what you did.

## Where to write what

| Information | Destination |
|---|---|
| Current state, next steps | `doc/ai/STATUS.md` (rewrite, ≤60 lines) |
| PI answer or design decision | `doc/ai/DECISIONS.md` (append) |
| Open question | `doc/ai/questions_for_pi.md` |
| Frozen scientific definition | `doc/ai/CONTRACT.md` (PI approval only) |
| How the experimental harness behaves | `doc/ai/harness/orblib_exp.md` |
| How production runs | `doc/ai/harness/production.md` |
| Facts about a run (best model, N, status) | `results/REGISTRY.md` |
| Analysis or crash notes of one run | `results/<run_id>/NOTES.md`, `postmortem.md` |
| Article text, literature, drafts | `paper/` (private nested repo) |

Do **not** append learned project information to this file. `AGENTS.md` changes
only when a *universal* rule changes, and it stays ≤120 lines.

## Skills

| Situation | Skill |
|---|---|
| Launch/docker logs, crash, missing `.npz`, disk or upload failure | `/run-postmortem` |
| Inspect a run's history: best model, bounds, penalty statistics | `/analyze-results` |
| J-factor numbers, estimators, units, mass/corner plots | `/jfactor-report` |
| Editing `orblib_exp.py`, `launch_orblib_exp.sh`, `orblib_storage.py`, tests | `/harness-dev` |
| Abstract, article text, literature, presentation | `/paper-writing` |

## Verify

```bash
cd tests && ../.venv-ai/bin/python -m pytest -q test_orblib_q1.py \
    test_orblib_storage.py --rootdir=. --import-mode=importlib -p no:cacheprovider
cd .. && python3 -m py_compile py/Fornax_P21_PCA_w3Sersic_orblib_exp.py py/orblib_storage.py
bash -n py/launch_orblib_exp.sh && git diff --check && git status --short
```

Do not run `python -m pytest` from the repo root: the local `py/` directory
shadows pytest's `py` module.
