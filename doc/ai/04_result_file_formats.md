# Result file formats

Based on the committed examples in `py/` and the writing code in
`py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py` and `py/launch_docker_parallel.sh`.

## 1. `4UpsBoTorch_PCA_Sersic_*.txt` — evaluation history (machine-readable)

Written by `halo_IC_lib_weights_pca_fixed` (script lines ~890–903), one block per
objective-function evaluation:

```text
# Server: <host>_<pN>
<incl> <Q> <gh> <rh> <rho0> <Upsilon> <penalty> <YYYY-MM-DD HH:MM:SS.ffffff>
# PCA: <pc1> <pc2> <pc3>
# Optimization history (Upsilon values -> function values):
#    0: [<Upsilon>] -> <penalty>
#    ...
# End of history
<blank line>
```

- **No global header.** Comment lines start with `#`; readers
  (`load_fresh_data_from_files`, J-factor script) keep only non-comment lines
  with ≥7 whitespace-separated fields and parse `parts[:7]` as floats.
- **Data line columns** (0-based, as used by `load_fresh_data_from_files` and
  `compare_good_vs_acceptable`):

| Col | Name | Notes |
|---|---|---|
| 0 | `incl` | inclination in degrees; fixed per run; readers filter by `abs(incl - target) <= 0.01`. **This is how incl is "defined" per row** — every data line carries it. |
| 1 | `Q` = `axRZ` | halo z-axis ratio |
| 2 | `gh` = `gammah` | halo inner slope |
| 3 | `rh` = `rhalo` | halo scale radius |
| 4 | `rho0` | halo density normalization |
| 5 | `Upsilon` | best Υ from the Brent search for this parameter set |
| 6 | `penalty` | minimal penalty at that Υ (1e6 ⇒ failed evaluation; readers drop penalty ≥ 1e5) |
| 7 | timestamp | `datetime.datetime.now()`, two whitespace-separated tokens (date + time). **Not parsed** by readers (they take only `parts[:7]`). |

- **J-factor is NOT in these files** — it is computed downstream by
  `J_factor_Sersic_Fornax_P21_symm.py` into separate `J_factor_*` outputs (on
  Yandex.Disk, not committed).
- **Seed/worker id**: the worker is identified by the `# Server: <host>_<pN>`
  comment preceding each block and by the filename suffix; it is not a column.
- The `# PCA:` line gives the candidate's coordinates in the then-current PCA
  space (space changes after each PCA update, so these are not comparable across
  the whole file).
- Merged host-level files (`4UpsBoTorch_PCA_Sersic_<host>.txt`) additionally
  contain merge separators written by `append_and_remove` in the launcher:

```text
# ============================================================
# Добавлено из: <src file>
# Label:  RESULT-OK: incl=..., suffix=pN, host=...
# Time:   ...
# Lines:  <n>
# HASH:<md5 of data lines>
# ============================================================
```

  The `HASH:` line is used for idempotent merging (skip if already present).

### Fields needed for future analysis with `incl` as a parameter

Already present: `incl` (col 0), all 4 halo params, `Upsilon`, `penalty`.
Available only from comments: worker id, timestamp (col 7 exists but is unparsed).
**Missing / ambiguous**: random seed, software version, PCA-space id, run id —
candidates for a future metadata sidecar (needs PI approval, see questions).

## 2. `4result_BoTorch_PCA_Sersic_*.txt` — optimization log (human-readable)

Written via the `_write` helper inside `run_pca_optimization` (and the launcher
merge machinery). **Not a tabular format**; it is a free-form log containing:

- header block: `# TuRBO-PCA Optimization Log`, `# Server: <host>_<pN>`,
  `# Start: <timestamp>`, storage/host patterns, `# Iterations planned: 40`,
  `# PCA components: 3`, `# Target fraction: 0.3`, `# Expand bounds by: 2.5`;
- data-loading and adaptive-cutoff diagnostics (file line counts, penalty range);
- PCA-model construction reports (explained variance per PC, bounds);
- per-iteration progress (`--- Итерация k/40 ... ---`, best target, TR size);
- PCA-update reports (production interval 12 iterations);
- final best-parameter summary (`РЕЗУЛЬТАТ TuRBO-PCA:` with incl, Q, gh, rh,
  rho0, Upsilon, penalty) and `# End:` timestamp.

`incl` appears in logged messages (e.g. `Загружено строк (incl=87.5)`) and in the
final summary; there is no per-row column structure. **No script in the repo
parses these files** — they appear to be for monitoring/debugging only (to be
confirmed by the PI, see `questions_for_pi.md`).

## 3. Naming convention: canonical vs PA46.8 archive

- **Canonical** (correct geometry `posang=42.3`, `q_ap=0.69`):
  `4UpsBoTorch_PCA_Sersic_<host>[_pN].txt` and
  `4result_BoTorch_PCA_Sersic_<host>[_pN].txt`. New runs write these.
- **Archive** (wrong geometry `posang=46.8`, `q_ap=0.7`):
  `4UpsBoTorch_PCA_PA46.8_Sersic_*` / `4result_BoTorch_PCA_PA46.8_Sersic_*`.
  The canonical glob `4UpsBoTorch_PCA_Sersic_*` does **not** match these (the
  `PA46.8_` segment breaks the pattern), so archive penalties never reach a new
  run's PCA model. The archive is read **only** by the opt-in seeding path
  (`--init-from-pa468`, `seed_points_from_patterns`), which **recomputes** penalty.

## 4. Resolved semantics (PI answers)

1. **Penalty**: **not χ²**; treat col 6 only as a relative ranking score.
2. **Files**: use **all** files (per-process `_pN` + merged per-host). `_pN` are
   valid mid-run; merged at the end of a VM run. Both are kept.
3. **No dedup**: do **not** deduplicate across files or hosts — each row is a
   unique experiment result (overrides the earlier Goal-0 dedup note).
4. **`4result_*` logs**: monitoring only; no analysis parses them.
5. **Legacy `4UpsBoTorch_Sersic.txt`** (no `_PCA_`): same column format, made by
   older code; still read via `storage_patterns`.
6. **Timestamp column**: written but never parsed (readers take `parts[:7]`);
   VM local time.
