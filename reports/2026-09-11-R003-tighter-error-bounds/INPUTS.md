# R003 — inputs

All inputs are small result files of report R002 (`reports/2026-09-11-R002-multishooting-figure-atlas/analysis/results/`),
copied byte-for-byte into `external_data/` at repo commit `cafad1ac6c4894334c51ebe52291d3f114abeefb`.
Producing commands: R002 `analysis/01_data.jl` (fhn_*), `analysis/02_concept.jl` (concept_meta, lemma1/2). No file is larger than 200 KB.

| file | sha256 | produced by |
|---|---|---|
| `concept_meta.json` | `9fa95a77af51e600f652d31f283c6a8ccc267b5acb267b85b70afc5195b59d91` | R002 `02_concept.jl` — $L$, $\tilde L$, $\mu$ along the orbit and on the box |
| `fhn_data.csv` | `b9172358bb50071dd3b898c75c21ded61d0ca76f415fab40e7ea41d3d558e5dd` | R002 `01_data.jl` — noisy and clean FHN samples, $\Delta t = 1$ |
| `fhn_data_meta.json` | `f3fd5e3c12d587b9dd48b5dd11ee9584f71269f07fa9765a7b67c9c75d56d183` | R002 `01_data.jl` — noise descriptors |
| `fhn_fine.csv` | `5a7926101cbaa056ca1389dac3b05276cf787ccabdc48f5f6d25c0268a984adb` | R002 `01_data.jl` — true orbit at 0.05 spacing |
| `fhn_library.csv` | `40a6b2bf428ecc8e6deb6d41889fd88d84e0b984b94b613e8308ac379f6b82ae` | R002 `01_data.jl` — monomial library and true coefficients (the Jacobian is built from this file) |
| `lemma1_flow_sensitivity.csv` | `5e795b19c5655c77247cae31774460f908de7f7a980141a4fcb899441f73f4be` | R002 `02_concept.jl` — 24 initial-state probes |
| `lemma2_param_sensitivity.csv` | `31171a843bb73e8ff3ebedb93dfe7c8494757af7b3afb66ee14f8a8712506ba2` | R002 `02_concept.jl` — 24 parameter probes |

`deps/` is intentionally empty: the analysis imports no shared module (numpy/pandas only) and reads no Julia source.
