# R002 number manifest (analysis/print_prose_facts.py)

Each entry: the value as computed now from analysis/results/, its type, source file and selector, and whether the same value appears in report.tex (searched as a rounded string).

| key | value (now) | type | source / selector | found in report.tex |
|---|---|---|---|---|
| T1_final_perr_GP | 0.321567 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_final_perr_control | 1.64903 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_min_perr_GP | 0.221816 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_argmin_kappa | 5 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_final_paired_diff | -1.13765 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_final_ci[0] | -1.4264 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| T1_final_ci[1] | -0.77189 | result_value | result_value: tables/T1_headline.csv (sweeps.csv, exp=main) | yes |
| seed1_plateau | True | derived_statistic | derived_statistic: sweeps.csv | no |
| n_seeds_main | 8 | count_from_inventory | count_from_inventory | yes |
| seed_p_err_range[0] | 1.49811 | derived_statistic | derived_statistic: sweeps.csv | yes |
| seed_p_err_range[1] | 1.84076 | derived_statistic | derived_statistic: sweeps.csv | yes |
| control_final_blowup_frac | 0.25 | derived_statistic | derived_statistic: sweeps.csv | yes |
| T2_schedule.coarse[0] | 0.523933 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_schedule.coarse[1] | 0.55 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_schedule.dense[0] | 0.301528 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_schedule.dense[1] | 0.8 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_schedule.jump[0] | 0.76354 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_schedule.jump[1] | 0.425 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_penalty.propagate_graded[0] | 0.273969 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_penalty.propagate_graded[1] | 0.825 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_penalty.reset_graded[0] | 1.63134 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_penalty.reset_graded[1] | 0.3 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.0[0] | 0.231306 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.0[1] | 0.8 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.01[0] | 0.36173 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.01[1] | 0.7 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.02[0] | 0.248177 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.02[1] | 0.875 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.05[0] | 0.221681 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.05[1] | 0.8 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.1[0] | 0.313454 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.1[1] | 0.625 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.2[0] | 1.88087 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.propagate σ=0.2[1] | 0.225 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.0[0] | 1.6329 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.0[1] | 0.25 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.01[0] | 1.60217 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.01[1] | 0.2 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.02[0] | 1.66943 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.02[1] | 0.225 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.05[0] | 1.64903 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.05[1] | 0.25 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.1[0] | 1.67891 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.1[1] | 0.225 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.2[0] | 1.70207 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_noise.reset σ=0.2[1] | 0.2 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.0[0] | 0.814155 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.0[1] | 0.375 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.001[0] | 0.843311 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.001[1] | 0.475 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.01[0] | 0.887459 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.01[1] | 0.525 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.05[0] | 0.221681 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.05[1] | 0.8 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.2[0] | 0.474128 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.0.2[1] | 0.65 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.1.0[0] | 1.00955 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_sparsity.1.0[1] | 0.6 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.bfgs_propagate[0] | 0.205338 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.bfgs_propagate[1] | 0.95 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.bfgs_reset[0] | 1.64806 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.bfgs_reset[1] | 0.225 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.lbfgs_propagate[0] | 0.205349 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.lbfgs_propagate[1] | 0.95 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.lbfgs_reset[0] | 1.64806 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_optimizer.lbfgs_reset[1] | 0.225 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.250[0] | 1.72187 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.250[1] | 0.325 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.500[0] | 1.42619 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.500[1] | 0.2 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.1000[0] | 0.608747 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.1000[1] | 0.575 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.2500[0] | 0.265828 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.2500[1] | 0.775 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.5000[0] | 0.198599 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_iters.5000[1] | 0.9 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.1[0] | 0.498847 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.1[1] | 0.75 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.2[0] | 0.337739 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.2[1] | 0.675 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.5[0] | 0.285297 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.5[1] | 0.825 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.10[0] | 0.265828 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.10[1] | 0.775 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.20[0] | 0.265828 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_substeps.20[1] | 0.775 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_propagate_flat[0] | 0.196653 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_propagate_flat[1] | 0.85 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_propagate_graded[0] | 0.196653 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_propagate_graded[1] | 0.85 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_reset_flat[0] | 1.65038 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_reset_flat[1] | 0.35 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_reset_graded[0] | 1.58863 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.01_reset_graded[1] | 0.375 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_propagate_flat[0] | 0.285715 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_propagate_flat[1] | 0.775 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_propagate_graded[0] | 0.285715 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_propagate_graded[1] | 0.775 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_reset_flat[0] | 1.58633 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_reset_flat[1] | 0.475 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_reset_graded[0] | 1.57292 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.03_reset_graded[1] | 0.35 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_propagate_flat[0] | 0.208623 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_propagate_flat[1] | 0.85 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_propagate_graded[0] | 0.208623 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_propagate_graded[1] | 0.85 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_reset_flat[0] | 1.64903 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_reset_flat[1] | 0.25 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_reset_graded[0] | 1.63134 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.1_reset_graded[1] | 0.3 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_propagate_flat[0] | 0.42367 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_propagate_flat[1] | 0.7 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_propagate_graded[0] | 0.349534 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_propagate_graded[1] | 0.825 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_reset_flat[0] | 2.44313 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | no |
| T2_seedscale.sc0.3_reset_flat[1] | 0.1 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_reset_graded[0] | 2.7018 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T2_seedscale.sc0.3_reset_graded[1] | 0.1 | result_value | result_value: tables/T2_variations.csv (sweeps.csv, κ=100) | yes |
| T3_inhouse_S10_err | 1.35938e-13 | result_value | result_value: solvers_eval.csv (κ=5, p*) | yes |
| T3_inhouse_S10_ms | 0.542875 | result_value | result_value: solvers_eval.csv (κ=5, p*) | yes |
| T4[0].finding | F2 (R001): mod 7 stiff branch drops the w-residual | result_value | result_value: bug_*.csv|json | no |
| T4[0].measure | share of J(p*) missing, κ=1/10/25/100 | result_value | result_value: bug_*.csv|json | no |
| T4[0].value | 11.5 % / 9.3 % / 17.0 % / 11.2 % / 11.9 % | result_value | result_value: bug_*.csv|json | no |
| T4[1].finding | F1 (R001): mod 6 odefun_new = linear truncation | result_value | result_value: bug_*.csv|json | no |
| T4[1].measure | ∂v̇/∂v at u=(1.5,0.4): pre-fix vs fixed | result_value | result_value: bug_*.csv|json | no |
| T4[1].value | 1 vs -1.25 | result_value | result_value: bug_*.csv|json | no |
| T4[2].finding | F4 (R001): sweep gradient = single-shooting gradient | result_value | result_value: bug_*.csv|json | no |
| T4[2].measure | cos∠(gκ, g100) at κ=1 / 5 | result_value | result_value: bug_*.csv|json | no |
| T4[2].value | 0.180 / 0.989 | result_value | result_value: bug_*.csv|json | no |
| T4[3].finding | NEW: mod 8 driver mixed monomial orders | result_value | result_value: bug_*.csv|json | no |
| T4[3].measure | spurious ‖p−p*‖ between orderings | result_value | result_value: bug_*.csv|json | no |
| T4[3].value | 2.501 | result_value | result_value: bug_*.csv|json | no |
| T4[4].finding | NEW: mod 8 forward_simulation_loss ignores data_ | result_value | result_value: bug_*.csv|json | no |
| T4[4].measure | J identical for 3 datasets pre-fix / differs post-fix | result_value | result_value: bug_*.csv|json | no |
| T4[4].value | True / True | result_value | result_value: bug_*.csv|json | no |
| T5.lorenz/bfgs/propagate[0] | 3.85167 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/bfgs/propagate[1] | 0.833333 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/bfgs/reset[0] | 31.9494 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/bfgs/reset[1] | 0.0333333 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/nm/propagate[0] | 18.1138 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/nm/propagate[1] | 0.233333 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/nm/reset[0] | 31.9494 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lorenz/nm/reset[1] | 0.0333333 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lv/nm/propagate[0] | 0.289962 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lv/nm/propagate[1] | 0.833333 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lv/nm/reset[0] | 1.40819 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| T5.lv/nm/reset[1] | 0.291667 | result_value | result_value: other_sweeps.csv (κ=100) | yes |
| N | 101 | protocol_constant | protocol_constant: ATLAS_CONTEXT.md | yes |
| Delta_t | 1 | protocol_constant | protocol_constant | yes |
| noise_rel | 0.05 | protocol_constant | protocol_constant | yes |
| eta_max | 0.158678 | result_value | result_value: fhn_data_meta.json | yes |
| noise_floor_J | 0.00515241 | result_value | result_value: fhn_data_meta.json | yes |
| L_traj | 3.05153 | result_value | result_value: concept_meta.json | yes |
| L_box | 5.3448 | result_value | result_value: concept_meta.json | yes |
| Ltilde | 10.5645 | result_value | result_value: concept_meta.json | yes |
| lognorm | 1.17129 | result_value | result_value: concept_meta.json | yes |
| core_loss_reldiff | 1.91672e-13 | result_value | result_value: core_check.json | yes |
| core_data_absdiff | 6.43929e-15 | result_value | result_value: core_check.json | yes |
| ms_per_eval | 0.918741 | result_value | result_value: core_check.json (timing, not gated) | yes |
| hess_lambda_max_k1 | 36.4507 | result_value | result_value: hessian_at_ptrue.csv | yes |
| hess_lambda_max_k100 | 1.20141e+06 | result_value | result_value: hessian_at_ptrue.csv | no |
| hess_nneg_k100 | 5 | result_value | result_value: hessian_at_ptrue.csv | yes |
| minima_k1 | 1.25 | result_value | result_value: landscape_1d_minima.csv | yes |
| minima_k100 | 7.91667 | result_value | result_value: landscape_1d_minima.csv | yes |

Result-class keys not found verbatim in report.tex (rounded differently or not quoted): 16

- T4[0].finding
- T4[0].measure
- T4[0].value
- T4[1].finding
- T4[1].measure
- T4[1].value
- T4[2].finding
- T4[2].measure
- T4[2].value
- T4[3].finding
- T4[3].measure
- T4[3].value
- T4[4].finding
- T4[4].measure
- T4[4].value
- hess_lambda_max_k100
