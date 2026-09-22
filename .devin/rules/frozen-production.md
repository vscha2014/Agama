---
trigger: glob
globs: ["py/Fornax_P21_symm_PCA_w3Sersic_yaVM*.py", "py/J_factor_Sersic_Fornax_P21_symm.py", "py/launch_docker_parallel.sh", "py/schwarzlib.py", "py/pygama.py", "py/table3.dat", "src/**"]
description: "Production and upstream files are read-only"
---

This file is frozen (`doc/ai/CONTRACT.md`). Read it, do not edit it and never
run it. If a change is needed, put the proposed diff in your report and a
question in `doc/ai/questions_for_pi.md`; implement experiments in
`py/Fornax_P21_PCA_w3Sersic_orblib_exp.py` instead.
