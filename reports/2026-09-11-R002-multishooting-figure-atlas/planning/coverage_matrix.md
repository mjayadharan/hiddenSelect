# Realised coverage matrix (computed by analysis/make_inventory.py)

Atomic unit per file = one row of the listed axes; `cells` = distinct axis tuples, `rows` = file rows.

## sweeps.csv  (rows 3634, cells 3094, sha256 98f73d9fc1a6…)

- `exp` (10 values): basin, iters, main, noise, optimizer, penalty, schedule, seedscale, sparsity, substeps
- `arm` (47 values): S1, S10, S2, S20, S5, best, bfgs_propagate, bfgs_reset, coarse, dense, gamma0.0, gamma0.001, gamma0.01, gamma0.05, gamma0.2, gamma1.0, it1000, it250, it2500, it500, it5000, jump, k1, k100, k5, lbfgs_propagate, lbfgs_reset, propagate, propagate_graded, reset, reset_graded, sc0.01_propagate_flat, sc0.01_propagate_graded, sc0.01_reset_flat, sc0.01_reset_graded, sc0.03_propagate_flat, sc0.03_propagate_graded, sc0.03_reset_flat, sc0.03_reset_graded, sc0.1_propagate_flat …
- `seed` (10 values): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- `window_size` (45 values): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 40, 45, 50, 55, 60, 65, 70, 75 …

| exp | arm | seeds | window sizes | rows |
|---|---|---|---|---|
| basin | k1 | 10 | 1 | 50 |
| basin | k100 | 10 | 1 | 50 |
| basin | k5 | 10 | 1 | 50 |
| iters | it1000 | 4 | 7 | 28 |
| iters | it250 | 4 | 7 | 28 |
| iters | it2500 | 4 | 7 | 28 |
| iters | it500 | 4 | 7 | 28 |
| iters | it5000 | 4 | 7 | 28 |
| main | best | 8 | 16 | 128 |
| main | propagate | 8 | 16 | 128 |
| main | reset | 8 | 16 | 128 |
| noise | propagate | 6 | 7 | 252 |
| noise | reset | 6 | 7 | 252 |
| optimizer | bfgs_propagate | 8 | 16 | 128 |
| optimizer | bfgs_reset | 8 | 16 | 128 |
| optimizer | lbfgs_propagate | 8 | 16 | 128 |
| optimizer | lbfgs_reset | 8 | 16 | 128 |
| penalty | propagate_graded | 8 | 16 | 128 |
| penalty | reset_graded | 8 | 16 | 128 |
| schedule | coarse | 8 | 4 | 32 |
| schedule | dense | 8 | 44 | 352 |
| schedule | jump | 8 | 2 | 16 |
| seedscale | sc0.01_propagate_flat | 8 | 7 | 56 |
| seedscale | sc0.01_propagate_graded | 8 | 7 | 56 |
| seedscale | sc0.01_reset_flat | 8 | 7 | 56 |
| seedscale | sc0.01_reset_graded | 8 | 7 | 56 |
| seedscale | sc0.03_propagate_flat | 8 | 7 | 56 |
| seedscale | sc0.03_propagate_graded | 8 | 7 | 56 |
| seedscale | sc0.03_reset_flat | 8 | 7 | 56 |
| seedscale | sc0.03_reset_graded | 8 | 7 | 56 |
| seedscale | sc0.1_propagate_flat | 8 | 7 | 56 |
| seedscale | sc0.1_propagate_graded | 8 | 7 | 56 |
| seedscale | sc0.1_reset_flat | 8 | 7 | 56 |
| seedscale | sc0.1_reset_graded | 8 | 7 | 56 |
| seedscale | sc0.3_propagate_flat | 8 | 7 | 56 |
| seedscale | sc0.3_propagate_graded | 8 | 7 | 56 |
| seedscale | sc0.3_reset_flat | 8 | 7 | 56 |
| seedscale | sc0.3_reset_graded | 8 | 7 | 56 |
| sparsity | gamma0.0 | 6 | 7 | 42 |
| sparsity | gamma0.001 | 6 | 7 | 42 |
| sparsity | gamma0.01 | 6 | 7 | 42 |
| sparsity | gamma0.05 | 6 | 7 | 42 |
| sparsity | gamma0.2 | 6 | 7 | 42 |
| sparsity | gamma1.0 | 6 | 7 | 42 |
| substeps | S1 | 4 | 7 | 28 |
| substeps | S10 | 4 | 7 | 28 |
| substeps | S2 | 4 | 7 | 28 |
| substeps | S20 | 4 | 7 | 28 |
| substeps | S5 | 4 | 7 | 28 |

## other_sweeps.csv  (rows 196, cells 196, sha256 dfaa74ee7d18…)

- `system` (2 values): lorenz, lv
- `arm` (2 values): propagate, reset
- `optimizer` (2 values): bfgs, nm
- `seed` (6 values): 1, 2, 3, 4, 5, 6
- `window_size` (7 values): 1, 2, 5, 10, 25, 50, 100

| system | optimizer | arm | seeds | window sizes | rows |
|---|---|---|---|---|---|
| lorenz | bfgs | propagate | 4 | 7 | 28 |
| lorenz | bfgs | reset | 4 | 7 | 28 |
| lorenz | nm | propagate | 4 | 7 | 28 |
| lorenz | nm | reset | 4 | 7 | 28 |
| lv | nm | propagate | 6 | 7 | 42 |
| lv | nm | reset | 6 | 7 | 42 |

## solvers_eval.csv  (rows 168, cells 168, sha256 e394bd65e3db…)

- `param` (3 values): ptrue, ptrue_plus_0.02randn_seed1, ptrue_plus_0.02randn_seed2
- `window_size` (4 values): 1, 5, 25, 100
- `method` (8 values): diffeq_ImplicitEuler_fixed, diffeq_RadauIIA5_adaptive, diffeq_Rodas5_adaptive, diffeq_Rosenbrock23_adaptive, diffeq_Tsit5_adaptive, diffeq_Tsit5_fixed, inhouse_Tsit5_fixed, reference_diffeq_Tsit5_adaptive
- `setting` (10 values): S=1, S=10, S=2, S=20, S=5, S=50, abstol=reltol=1e-12, abstol=reltol=1e-4, abstol=reltol=1e-8, dt=0.1

## solvers_sweep.csv  (rows 42, cells 42, sha256 b5e37a7462a6…)

- `method` (3 values): diffeq_Rosenbrock23_adaptive, diffeq_Tsit5_adaptive, inhouse_Tsit5_fixed_S10
- `seed` (2 values): 1, 2
- `window_size` (7 values): 1, 2, 5, 10, 25, 50, 100

## landscape_2d_v_v3.csv  (rows 26244, cells 4, sha256 45498688b740…)

- `window_size` (4 values): 1, 3, 10, 100

## landscape_2d_w_wv.csv  (rows 26244, cells 4, sha256 7582252b19b5…)

- `window_size` (4 values): 1, 3, 10, 100

## landscape_1d_minima.csv  (rows 84, cells 84, sha256 48db40478c35…)

- `direction` (12 values): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
- `window_size` (7 values): 1, 2, 5, 10, 25, 50, 100

## hessian_at_ptrue.csv  (rows 9, cells 9, sha256 5be3063490cd…)

- `window_size` (9 values): 1, 2, 3, 5, 10, 20, 25, 50, 100

## prop1_node_removal.csv  (rows 42, cells 42, sha256 e92e04d63af4…)

- `param` (3 values): near, true, wrong
- `experiment` (2 values): block, stride
- `n_removed` (14 values): 1, 2, 3, 5, 8, 12, 20, 50, 66, 75, 80, 90, 95, 98

## lemma1_flow_sensitivity.csv  (rows 2424, cells 24, sha256 5e795b19c565…)

- `start_index` (4 values): 1, 21, 41, 61
- `direction` (6 values): 1, 2, 3, 4, 5, 6

## lemma2_param_sensitivity.csv  (rows 2424, cells 24, sha256 31171a843bb7…)

- `start_index` (4 values): 1, 21, 41, 61
- `direction` (6 values): 1, 2, 3, 4, 5, 6

## bug_F2_w_residual.csv  (rows 20, cells 20, sha256 75537f3a43f5…)

- `window_size` (5 values): 1, 5, 10, 25, 100
- `param` (4 values): pert1, pert2, pert3, ptrue

## bug_F4_gradient.csv  (rows 7, cells 7, sha256 3316a54b46e7…)

- `window_size` (7 values): 1, 2, 5, 10, 25, 50, 100

## post_hessian_at_minimizers.csv  (rows 128, cells 128, sha256 42d73be031e3…)

- `seed` (8 values): 1, 2, 3, 4, 5, 6, 7, 8
- `window_size` (16 values): 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100
