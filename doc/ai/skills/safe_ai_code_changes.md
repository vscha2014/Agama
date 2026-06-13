# Safe AI code changes (digest of AGENTS.md)

- Plan first: steps + success criteria + verification per step.
- Ambiguity → list interpretations; if it affects design, ask
  (`doc/ai/questions_for_pi.md`).
- Minimal, surgical diffs: no refactoring/reformatting of neighboring code, no
  speculative features or unrequested configurability.
- Never modify without PI approval: scientific contract items (penalty,
  apertures, GH conventions, Brent/Upsilon, J-factor, units, `table3.dat`,
  Sersic assumptions, `bounds_original`), upstream AGAMA code (`src/`,
  `py/schwarzlib.py`, ...), the three production files
  (`Fornax_*`, `J_factor_*`, `launch_docker_parallel.sh`).
- Never run: production optimization, `launch_docker_parallel.sh`, long
  BoTorch runs, orbit-library regeneration. Allowed: grep/read, `--help`,
  `py_compile`, light unit tests.
- Never commit: `.venv-ai/`, extracted PDF text, logs, results, `*.pkl`.
  New artifacts → gitignored paths only.
- Verify before declaring done: tests / `py_compile` / `git diff` inspection;
  production scripts must show an empty diff unless the task required changes.
