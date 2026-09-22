# Decision log

Append-only. One entry per PI decision or per design choice made without the
PI (marked as such). Never rewrite past entries; add a new one that supersedes.
Format: `date | topic | decision | consequence`.

## 2026-06 — Stage-1 PI answers (Q1–Q14, originally in `questions_for_pi.md`)

- **Geometry.** `posang = 46.8` is wrong → `42.3`; `q_ap = 0.7` → `1 − 0.31`.
  All results computed with the old values renamed to `*_PA46.8_*`; new runs
  keep the canonical names. The archive may seed initial points only with
  penalties recomputed (`--init-from-pa468`, opt-in).
- **Q1 penalty.** Cannot be treated as χ². Ranking score only.
- **Q2 J weighting.** As implemented: `w = exp(-(penalty − pen_min)/pen_sigma)`,
  `pen_sigma = max(std, 1e-6)`, normalised, weighted KDE.
- **Q3 `incl` prior.** Physical constraint only:
  `axRZst = sqrt(q_ap² − cos²i)/sin i` real ⇒ `cos i < q_ap`.
- **Q4 result files.** Use all of them, including per-process `_pN` files
  mid-run.
- **Q5 legacy `4UpsBoTorch_Sersic.txt`.** Same column format, older code; keep
  reading it.
- **Q6 `4result_*`.** Monitoring only.
- **Q7 dedup.** Never deduplicate stored rows, not even across hosts — each row
  is a unique experiment result.
- **Q8 metadata sidecars.** Allowed, provided existing file formats stay intact.
- **Q9 bounds.** Changing `bounds_original` requires asking the PI first.
- **Q10 frozen inputs.** `massSt = 14.0` and `D = 143 kpc` fixed; Sersic-error
  propagation must be *separate new code*, not edits to the main script.
  Sersic index uncertainty is ±0.006 (per `Wang_2019_table1…txt`), not ±0.06.
- **Q11 RNG.** Give parallel workers different seeds; the GH observation-error
  bootstrap stays at fixed `seed = 42`.
- **Q12 J-factor script paths.** Analysis runs on the local machine; the
  hardcoded `YADISK_DIR` with the user name is intentional.
- **Q13 Dockerfile.** Added to the repo root together with `check_agama.py`,
  `entrypoint.sh`.
- **Q14 iterations.** PI asked for a recommendation → production
  `pca_update_interval = 12`, `n_iter = 40` per run.

## 2026-08/09 — Experimental harness (agent decisions, PI-approved scope)

- **2026-08 | experiment isolation** | `EXP_ID = d{0|1}_nb{N}_gh{G}_ser{S}` is
  embedded in every derived filename | penalties of different `double`/`n_bin`
  are never pooled.
- **2026-09 | fixed-Q mode** | CLI flag `--Q1` (uppercase Q; lowercase `q` is
  reserved for the stellar flattening in the article); writes
  `Q1d1_nb250_gh0_ser0`, reads both Q modes but filters rows to `Q=1` |
  see `harness/orblib_exp.md`.
- **2026-09 | bounds widened** | `rh 3.5 → 7.0`, `rho0 34 → 10` after the `Q=1`
  run pinned `rh` at the upper bound | diagnostic range, not a physical claim;
  if a new minimum again lands on the bound, build a profile in `rh` instead of
  widening further.
- **2026-09 | orbit-library storage** | streaming per-file upload with MD5
  receipts replaces the tar-snapshot flow; on exhausted delivery attempts the
  run stops cleanly and the VM powers off with data intact | old tars need a
  one-time `index`.
- **2026-09 | harness/paper separation** (this task) | experiment knowledge in
  `doc/ai/`, article drafts in a nested private `paper/` repo, run facts in
  `results/REGISTRY.md` | AGENTS.md stays ≤ 120 lines of pointers. `results/`
  and `paper/` are local-only (public repo): unpublished numbers are never
  committed, so the registry lives outside version control.
