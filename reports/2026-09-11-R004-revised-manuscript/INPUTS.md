# R004 — inputs

R004 computes nothing new. Every figure, table and number is inherited from R002 and R003 (both in
`reports/`, repo HEAD `5c53e960a5a984e471a09546f72e53d6ba7d086c` at the time of writing) and copied **byte for byte** into this folder;
gates G1, G2 and G4 assert the byte identity. `deps/` is intentionally empty: the two Python scripts
import only numpy/pandas and read no Julia source. No file is larger than 1 MB.

| file in this folder | sha256 | source (relative to `reports/`) | produced by |
|---|---|---|---|
| `figures/R002_fig01_problem.pdf` | `75a3ab9c81907f83d525f518fc6af742aac04c6ef8f3589d9bbbd94eae288482` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig01_problem.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig02_single_shooting.pdf` | `0ed7658ea5a09f0869a83f3bb664866c33c7ac1b4dea148c81f81854e5ac2bd8` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig02_single_shooting.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig03_partition.pdf` | `e780778fdb039a33ee27796ffdb176d2694d40f02ef4bc5a5a87f5613c4efb0f` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig03_partition.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig04_multiple_shooting.pdf` | `b54b951acb8e6f78aa5eae8f07850343da2f523b6933710028ee7d3493961476` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig04_multiple_shooting.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig05_node_removal.pdf` | `8e7cf1a9855612a1ebc75ae2e0d6596d8163df2c1ca2fdd74d7f57b6c1ff04b2` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig05_node_removal.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig06_guess_propagation.pdf` | `17c3388d006b12bdf26b71358b3020f83eaa2652b5f69e5468d55ce6ff483ac5` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig06_guess_propagation.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig10_prop1_node_removal.pdf` | `b2dc29cef1e5adc5d8415d7a98be4a10b530158399bc21775ba314195561168e` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig10_prop1_node_removal.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig11_landscape_2d_v_v3.pdf` | `741a23002ca732ed0c54b37f39e80e194ce7218b051d6b34526d330a38fdec79` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig11_landscape_2d_v_v3.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig13_landscape_1d_slices.pdf` | `01bdd477a6babb4a00eb6974b97caaf7789935a18a7eb49816fe1f2f061d6645` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig13_landscape_1d_slices.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig14_hessian_conditioning.pdf` | `da1f2814bae42e99aa051f537d2f5f6b778736ccdb2a0bff6fc108a394e52510` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig14_hessian_conditioning.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig15_sweep_headline.pdf` | `46ac27dfcd1da02217d603d81b48306ec26343c8bd310632e665bb903e2520d0` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig15_sweep_headline.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig16_sweep_paired.pdf` | `d098591908377341cac8b3e0aaed67e9afffa8da459b96701ea3139544a47d61` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig16_sweep_paired.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig18_sweep_filmstrip.pdf` | `82afa50c53c1e77cf488c747ee28135bda3fc9f5b145e425ac78ee73e821aad2` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig18_sweep_filmstrip.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig20_final_coefficients.pdf` | `c2711674a8ad839ac2e261aa51df9dd022a8324f8fd82d6505d8ba8dc542e852` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig20_final_coefficients.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig21_schedule.pdf` | `5e14b6672f7be5e1635e1422ba08018c79c33aa6027b2cdc3b854909404971df` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig21_schedule.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig25_optimizers.pdf` | `fb4bee44574a76199b2add28f06a4c23dbf97f71cb8671b9b07ba0ef4ad855fc` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig25_optimizers.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig35_lotka_volterra.pdf` | `01c0b484c04091bdd08e21cfd33b045999a183b529b5dc47f647d036228a7ec8` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig35_lotka_volterra.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig36_lorenz.pdf` | `420d1d3b6ac971fbdbf48c3fbd08dafdd15fda49a35be9709b0c351f8a469892` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig36_lorenz.pdf` | R002 figures/make_figures.py |
| `figures/R002_fig37_post_hessian_path.pdf` | `176f5c52c4b3a77a3d53b6480170ec5993fc951ecf64b85c0fff5b1f422f8a83` | `2026-09-11-R002-multishooting-figure-atlas/figures/fig37_post_hessian_path.pdf` | R002 figures/make_figures.py |
| `figures/R003_fig01_bounds.pdf` | `dfd4a412e06c3af827f804c51b9eab1948724a007b0fc4f405d800a62a06a7c3` | `2026-09-11-R003-tighter-error-bounds/figures/fig01_bounds.pdf` | R003 figures/make_figures.py |
| `external_data/concept_costs.csv` | `8f39c2fe885674c9b6078e1ce59b5e5f99b673560be0898150cf407943f0ed35` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/concept_costs.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/concept_meta.json` | `9fa95a77af51e600f652d31f283c6a8ccc267b5acb267b85b70afc5195b59d91` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/concept_meta.json` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/fhn_data_meta.json` | `f3fd5e3c12d587b9dd48b5dd11ee9584f71269f07fa9765a7b67c9c75d56d183` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/fhn_data_meta.json` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/hessian_at_ptrue.csv` | `5be3063490cd4429622d271772b8b47e7d4f847e2c31b2cfe62e1344a4b839b6` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/hessian_at_ptrue.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/landscape_1d_minima.csv` | `48db40478c35413739be4631cabd82c675e52d61dfec8f8d7994c34b25b1c3dc` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/landscape_1d_minima.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/lemma1_flow_sensitivity.csv` | `5e795b19c5655c77247cae31774460f908de7f7a980141a4fcb899441f73f4be` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/lemma1_flow_sensitivity.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/lemma2_param_sensitivity.csv` | `31171a843bb73e8ff3ebedb93dfe7c8494757af7b3afb66ee14f8a8712506ba2` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/lemma2_param_sensitivity.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/post_hessian_at_minimizers.csv` | `42d73be031e318569d39da90aa7e51b7c5247ef3e074a16ab757d40d9d4f7468` | `2026-09-11-R002-multishooting-figure-atlas/analysis/results/post_hessian_at_minimizers.csv` | R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl |
| `external_data/tables/R003_summary.json` | `b324635fcd484479251187b9467a5c161fc34b014ddb5c891852e0adfec33dec` | `2026-09-11-R003-tighter-error-bounds/analysis/results/summary.json` | R003 analysis/compute_R003.py |
| `external_data/tables/T1_headline.csv` | `42e595abb29c2d5dd48a311e4b6b9457bc0514bcdb7a0632b2aa85a82972ad74` | `2026-09-11-R002-multishooting-figure-atlas/tables/T1_headline.csv` | R002 tables/make_tables.py |
| `external_data/tables/T1_headline.md` | `9ebbedeea4d27a2ffe95c0c46c49e620ccbe0a5df3e76de165da2291f031ad3d` | `2026-09-11-R002-multishooting-figure-atlas/tables/T1_headline.md` | R002 tables/make_tables.py |
| `external_data/tables/T1_headline.tex` | `4d6f4730d3552c197e6fab2022151ede6a97d0039182966152d417fb83f4e292` | `2026-09-11-R002-multishooting-figure-atlas/tables/T1_headline.tex` | R002 tables/make_tables.py |
| `external_data/tables/T2_variations.csv` | `93c4e8da6b9a8b80674ced82ab2a9bd2f8a8f424ba3f161e610d74b5f1ab3b8e` | `2026-09-11-R002-multishooting-figure-atlas/tables/T2_variations.csv` | R002 tables/make_tables.py |
| `external_data/tables/T5_other_systems.md` | `84c76f788c644b7555b1313bfaa6978d69034222cd16dd78fed1d0f8b3faa671` | `2026-09-11-R002-multishooting-figure-atlas/tables/T5_other_systems.md` | R002 tables/make_tables.py |
| `external_data/tables/T5_other_systems.tex` | `31899e8fa9308b29c55669be30609bc436454c63132c8686227a7b047dd5033e` | `2026-09-11-R002-multishooting-figure-atlas/tables/T5_other_systems.tex` | R002 tables/make_tables.py |
| `external_data/tables/constants.tex` | `445701ad8a5509131a7a06865353bf46cea150082abaeaa44c99df05f2c0a1ad` | `2026-09-11-R003-tighter-error-bounds/tables/constants.tex` | R003 analysis/make_tables.py |
| `external_data/tables/lemmas.tex` | `a0be719790f76bd23eb59a1386bb6ea5b44e57aa8e79d2e04af3168148a6a073` | `2026-09-11-R003-tighter-error-bounds/tables/lemmas.tex` | R003 analysis/make_tables.py |
| `external_data/tables/table_numbers.json` | `88f03388929aced86ae576406973b4f97b7c02d6c72496d0d3dcea5dbbd2a2db` | `2026-09-11-R002-multishooting-figure-atlas/tables/table_numbers.json` | R002 tables/make_tables.py |
| `external_data/tables/windows.tex` | `a9ef5c284550f2fad1464e2fc7f6d49ddff93a1cba040aa3790b6b4495a89046` | `2026-09-11-R003-tighter-error-bounds/tables/windows.tex` | R003 analysis/make_tables.py |
