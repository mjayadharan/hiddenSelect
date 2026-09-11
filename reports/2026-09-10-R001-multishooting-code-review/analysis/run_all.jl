# run_all.jl — ordered regeneration of every R001 result file.
for s in ("01_data_and_loss.jl", "02_rhs_and_bugs.jl", "03_stiff_branch.jl", "04_sweep.jl")
    println("\n=== ", s, " ==="); flush(stdout)
    run(`julia --project=$(joinpath(@__DIR__,"..","..","..")) --startup-file=no $(joinpath(@__DIR__, s))`)
end
