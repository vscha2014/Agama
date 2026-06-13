# Reading the scientific pipeline

- Start from `doc/ai/03_current_pipeline.md`, then read
  `py/Fornax_P21_symm_PCA_w3Sersic_yaVM.py` top-down: CLI/args → data prep &
  apertures → datasets → `halo_IC_lib_weights_pca_fixed` (one evaluation) →
  `run_pca_optimization` (outer loop) → `__main__`.
- Treat all numeric constants in the data-prep and evaluation sections as
  scientific contract (see `AGENTS.md` §7), even unexplained ones
  (`mult = sqrt(num_dof)*10`, `n_bin=250`, `max_r=2.1`, `psf2=0.01`, ...).
  Document them; never "fix" or "clean" them.
- Confirmed geometry (Wang et al. 2019, Sersic): `posang=42.3`,
  `q_ap=1-0.31=0.69`, `D=143 kpc`; admissible `incl` requires `cos incl < q_ap`.
  `penalty` is a ranking score, **not χ²**. J-factor good-model weighting is a
  weighted KDE `w=exp(-(penalty-pen_min)/pen_sigma)`. These are PI-confirmed.
- For AGAMA semantics, grep `doc/reference.tex` by labels (see
  `doc/ai/02_agama_reference_digest.md` §9) instead of extracting the PDF;
  forstand helpers are in `py/schwarzlib.py`.
- Result-file column meanings: verify against the writer code
  (`f.write(f"{incl} {Q} {gh} {rh} {rho0} {Ups} {pen} ...")`) and the readers
  (`load_fresh_data_from_files`), not against assumptions. Formats:
  `doc/ai/04_result_file_formats.md`.
- The production scripts import `torch`/`agama` at module level — they cannot be
  imported in `.venv-ai`. Use static analysis (`py_compile`, AST, grep); if you
  need to test a pure-Python helper, replicate it in a test file rather than
  importing the production module.
- Anything unclear that affects analysis (penalty semantics, weights, canonical
  files) → `doc/ai/questions_for_pi.md`, do not guess.
