| method | setting | J | |J−J_ref| | ms per eval |
|---|---|---|---|---|
| reference_diffeq_Tsit5_adaptive | abstol=reltol=1e-12 | 0.0132220 | 0 | 2.296 |
| inhouse_Tsit5_fixed | S=1 | 0.2818782 | 0.2687 | 0.05596 |
| inhouse_Tsit5_fixed | S=2 | 0.0132233 | 1.262e-06 | 0.1089 |
| inhouse_Tsit5_fixed | S=5 | 0.0132220 | 1.681e-10 | 0.2711 |
| inhouse_Tsit5_fixed | S=10 | 0.0132220 | 1.359e-13 | 0.5429 |
| inhouse_Tsit5_fixed | S=20 | 0.0132220 | 9.17e-15 | 1.085 |
| inhouse_Tsit5_fixed | S=50 | 0.0132220 | 2.382e-15 | 2.826 |
| diffeq_Tsit5_adaptive | abstol=reltol=1e-4 | 0.0132225 | 5.223e-07 | 0.2142 |
| diffeq_Tsit5_adaptive | abstol=reltol=1e-8 | 0.0132220 | 2.201e-10 | 0.5102 |
| diffeq_Rosenbrock23_adaptive | abstol=reltol=1e-8 | 0.0132218 | 1.747e-07 | 5.689 |
| diffeq_Rodas5_adaptive | abstol=reltol=1e-8 | 0.0132220 | 3.545e-10 | 1.976 |
| diffeq_RadauIIA5_adaptive | abstol=reltol=1e-8 | 0.0132220 | 7.083e-09 | 1.327 |
| diffeq_ImplicitEuler_fixed | dt=0.1 | 0.0120178 | 0.001204 | 1.537 |
| diffeq_Tsit5_fixed | dt=0.1 | 0.0132220 | 1.364e-13 | 0.7591 |
