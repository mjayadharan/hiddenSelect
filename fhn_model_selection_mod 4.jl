include("integrator.jl")
# Packages for plotting and default theming
using CairoMakie
using LinearAlgebra, Statistics
using Optim, ForwardDiff
using Random
using Base.Threads, DataFrames
using DifferentialEquations
using BenchmarkTools
# using RiverseDiff
# using Plots
plotGray = Makie.Colors.colorant"#4D4D4D"
set_theme!(Theme(fontsize = 16, figure_padding = 10, textcolor = plotGray, fonts = Attributes(:regular => "Helvetica Neue Light"), size = (480, 480), Axis = (xgridvisible = false, ygridvisible = false, ytickalign = 1, xtickalign = 1, yticksmirrored = true, xticksmirrored = true, bottomspinecolor = plotGray, topspinecolor = plotGray, leftspinecolor = plotGray, rightsplinecolor = plotGray, xtickcolor = plotGray, ytickcolor = plotGray, spinewidth = 1.0)))

## Define a general 2d cubic ODE function inplace that satisfies the integration function structure
function odefun(dy, y, p, t)
    dy[1] = p[1 ] + p[2 ]*y[1] + p[3 ]*y[2] + p[4 ]*y[1]^2 + p[5 ]*y[1]*y[2] + p[6 ]*y[2]^2 + p[7 ]*y[1]^3 + p[8 ]*y[1]^2*y[2] + p[9 ]*y[1]*y[2]^2 + p[10]*y[2]^3
    dy[2] = p[11] + p[12]*y[1] + p[13]*y[2] + p[14]*y[1]^2 + p[15]*y[1]*y[2] + p[16]*y[2]^2 + p[17]*y[1]^3 + p[18]*y[1]^2*y[2] + p[19]*y[1]*y[2]^2 + p[20]*y[2]^3
    return dy
end
# Number of parameters in odefun
const Np = 20

# Default parameters for FHN
fhn_p = [0.5, 1.0, -1.0, 0.0, 0.0, 0.0, -1/3, 0.0, 0.0, 0.0, 0.7/12.5, 1.0/12.5, -0.8/12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fhn_y0 = [1.0, 1.0]

## Integrate FHN data
cache = Tsit5Cache(fhn_y0) # Build cache
δt = 0.01                  # Find time step
T = 156                    # Maxium integration time
N = Int(T/δt)              # Number of integration steps
t0 = 0.0                   # Initial time 
# Integrate 
_, ys = integrate(odefun, fhn_y0, fhn_p, t0, N, δt, cache)
tsall = 0.0:δt:T
# Downsample in time to generate sparser data
downsample = 100
alldata = [getindex.(ys, 1)'; getindex.(ys, 2)']
data = alldata[:, 1:downsample:end]
ts = tsall[1:downsample:end]
# The odd index of alldata now contains the v(t) data and the even index contains the w(t) data
alldata = alldata[:, argmin(abs.(tsall .- 50)):argmin(abs.(tsall .- 150))][:]
tsall = tsall[argmin(abs.(tsall .- 50)):argmin(abs.(tsall .- 150))]

# Crop in time
ind1 = argmin(abs.(ts .- 50))
ind2 = argmin(abs.(ts .- 150))
ts = ts[ind1:ind2]
t1 = ts[1]
ts =  ts .- t1
tsall = tsall .- t1
Nd = size(data, 2)
# Concatenate into long vector
# The odd index of data now contains the v(t) data and the even index contains the w(t) data
data = data[:, ind1:ind2][:]
# Add noise
Random.seed!(1287436679)
noise_sigma = 0.25
data[1:2:end] .+= noise_sigma*std(data[1:2:end])*randn(length(data[1:2:end]))
data[2:2:end] .+= noise_sigma*std(data[2:2:end])*randn(length(data[2:2:end]))
# Data time step
Δt = ts[2] - ts[1]

# Plot the result 
fig = Figure() 
ax1 = Axis(fig[1, 1])
lines!(ax1, tsall, alldata[1:2:end], color = Makie.ColorSchemes.Signac[12])
scatter!(ax1, ts, data[1:2:end], color = Makie.ColorSchemes.Signac[12], markersize = 8)
ax2 = Axis(fig[2, 1])
lines!(ax2, tsall, alldata[2:2:end], color = Makie.ColorSchemes.Signac[11])
scatter!(ax2, ts, data[2:2:end], color = Makie.ColorSchemes.Signac[11], markersize = 8)
hidexdecorations!(ax1, ticks = false)
ax1.ylabel = "v(t)"
ax2.ylabel = "w(t)"
ax2.xlabel = "t"
fig

## Data fitting loss functions
# A smooth version of l1 for gradient descent
function smoothl1(x, alpha = 500)
    ax = alpha * x 
    if abs(ax) > 40
        return abs(x)
    else
        return inv(alpha) * (log(1 + exp(-alpha*x)) + log(1+exp(alpha * x)))
    end
end

function stiff_integration_step!(cache, odefun, x, t0, p, dt, calcError=false)
    # Create a (temporary) copy of x since the ODEProblem takes an initial condition.
    x_0 = copy(x)
    tspan = (t0, t0 + dt)
    prob = ODEProblem(odefun, x_0, tspan, p)
    sol = solve(prob, Rodas5(), abstol=1e-6, reltol=1e-6)
    # sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-8)

    # Update x in place with the solution at t+dt
    copyto!(x, sol(t0 + dt))
    return x
end

"""
    forward_simulation_loss(x0, p, odefun D, t0, δt, S, γ = 0.0)

Loss function for forward simulation from x0 using parameters p in the odefun from initial time t0 with time step δt. 
The data are given in the concatenated vector data and the data points are every S timesteps. 

Optionally add a smooth L1 regularization term with weight γ.

TODO: At this point assumes the state is two dimension!!!
"""
# function forward_simulation_loss(x0, p, odefun, D, t0, δt, S, γ = 0.0, stiff_solver=false)
#     # Integration cache
#     cache = Tsit5Cache(x0)
#     # Current state
#     x = cache.ycur
#     copyto!(x, x0)

#     # Initial condition loss
#     data_loss = abs2(x0[1] - data[1]) + abs2(x0[2] - data[2])
#     integration_method = stiff_solver ? stiff_integration_step! : integration_step!
#     # Loop over all data points
#     for i = 1:div(length(D), 2) - 1
#         # Internal time steps
#         for j = 1:S
#             integration_method(cache, odefun, x, t0, p, δt, false)
#         end

#         # Detect if we are going unstable
#         any(abs.(x) .> 1e3) && (println("Breaking early: $(i), $(data_loss)"); data_loss += 1e2; return data_loss)#(println("Breaking early: $(data_loss)"); data_loss += 1e8; return data_loss)
#         any(isnan.(x)) && (println("Breaking early: $(i), $(data_loss)"); data_loss += 1e2; return data_loss) #(println("Breaking early: $(data_loss)"); data_loss += 1e8; return data_loss)

#         # Data loss for current time point
#         data_loss += abs2(x[1] - data[1 + 2i]) + abs2(x[2] - data[2 + 2i])
#     end

#     # Sparsity term 
#     sparse_loss = 0.0
#     for p_i in p
#         sparse_loss += γ*smoothl1(p_i)
#     end
#     return data_loss / length(D) + sparse_loss / length(p)
# end

function forward_simulation_loss(x0, p, odefun, D, t0, δt, S, 
    γ = 0.0, stiff_solver=false)
# S: internal steps (only relevant when stiff_solver=false in your code)
# D_1, D_2: the data columns for v(t) and w(t)

    D_1 = @view D[1:2:end]
    D_2 = @view D[2:2:end]
    if stiff_solver
        # -----------------------------------------------------
        # STIFF SOLVER BRANCH
        # -----------------------------------------------------
        final_time = t0 + δt*(length(D_1) - 1)
        t_eval = t0:δt:final_time

        # Set up the ODE problem
        prob = ODEProblem(odefun, x0, (t0, final_time), p)

        # Solve with a stiff solver, e.g., Rodas3
        # sol = solve(prob, Rodas3(); saveat=t_eval, abstol=1e-6, reltol=1e-6)
        sol = solve(prob, Tsit5(); saveat=t_eval, abstol=1e-8, reltol=1e-8)



        # # Check for blow-ups or NaNs in the solution
        if length(D_1) != length([getindex(u_ind,1) for u_ind in sol.u]) ||
            any([getindex(u_ind,1) for u_ind in sol.u] .> 1e3) ||
            any([getindex(u_ind,2) for u_ind in sol.u] .> 1e3) ||
            any(isnan.([getindex(u_ind,1) for u_ind in sol.u])) ||
            any(isnan.([getindex(u_ind,2) for u_ind in sol.u]))
            @info "Breaking early at the p value: $(p)"
                return 1e3
        end


        # Sum-of-squares difference to data
        # D_1[i] is v-data, D_2[i] is w-data, and sol.u[i][1], sol.u[i][2] the solution states
        # data_loss = 0.0
        # for (i, t) in enumerate(t_eval)
        #     data_loss += abs2(sol.u[i][1] - D_1[i]) + abs2(sol.u[i][2] - D_2[i])
        # end

        data_loss = norm(D_1- [getindex(u_ind,1) for u_ind in sol.u], 2)^2 + norm(D_2- [getindex(u_ind,2) for u_ind in sol.u], 2)^2
        # Add smooth L1 regularization if desired
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        # Normalize by number of data points (optional) and return
        return data_loss/length(t_eval) + sparse_loss/length(p)

    else
        # -----------------------------------------------------
        # NON-STIFF BRANCH (YOUR ORIGINAL LOOP-BASED APPROACH)
        # -----------------------------------------------------
        # Integration cache
        cache = Tsit5Cache(x0)
        x = cache.ycur
        copyto!(x, x0)

        # Initial condition loss
        data_loss = abs2(x0[1] - D_1[1]) + abs2(x0[2] - D_2[1])
        integration_method = integration_step!

        # Loop over the data points
        for i = 1:length(D_1) - 1
            # Internal time steps
            for j = 1:S
                integration_method(cache, odefun, x, t0, p, δt, false)
            end

            # Detect blow-ups
            if any(abs.(x) .> 1e3) || any(isnan.(x))
                @info "Breaking early at p value $(p), solution blew up or NaN."
                return 1e3
            end
            # Data loss for current time point
            data_loss += abs2(x[1] - D_1[i+1]) + abs2(x[2] - D_2[i+1])
        end

        # Sparsity term
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        return data_loss / (2*length(D_1)) + sparse_loss / length(p)
    end
end

#Function for weighting data loss
weight_func_linear(x;α=1,n=10) = α*(x-1)/(n-1)
weight_func_exp(x; α=1, n=10, k=0.05) = α * (exp(k*(x - 1)) - 1) / (exp(k*(n - 1)) - 1)
weight_func_quad(x; α=1, n=10) = α * ((x - 1)^2) / ((n - 1)^2)

# plot(1:length(D_1),dum_fun.(1:length(D_1);α=0.5,n=length(D_1)))
function forward_simulation_loss_windows(x0, p, odefun, D, t0, δt, S, 
    γ = 0.0, num_windows=0, stiff_solver=false)
# S: internal steps (only relevant when stiff_solver=false in your code)
# D_1, D_2: the data columns for v(t) and w(t)

    D_1 = @view D[1:2:end]
    D_2 = @view D[2:2:end]
    num_windows = num_windows>0 ? num_windows : length(D_1)-1
    window_size = div(length(D_1), num_windows)
    if stiff_solver
        # -----------------------------------------------------
        # STIFF SOLVER BRANCH
        # -----------------------------------------------------
        Δt = δt * S
        final_time = t0 + Δt*(length(D_1) - 1)
        t_eval = t0:Δt:final_time

        # Set up the ODE problem
        prob = ODEProblem(odefun, x0, (t0, final_time), p)

        # Solve with a stiff solver, e.g., Rodas3
        # sol = solve(prob, Rodas3(); saveat=t_eval, abstol=1e-5, reltol=1e-5)
        sol = solve(prob, Tsit5(); saveat=t_eval, abstol=1e-8, reltol=1e-8)



        # # Check for blow-ups or NaNs in the solution
        if length(D_1) != length([getindex(u_ind,1) for u_ind in sol.u]) ||
            any([getindex(u_ind,1) for u_ind in sol.u] .> 1e3) ||
            any([getindex(u_ind,2) for u_ind in sol.u] .> 1e3) ||
            any(isnan.([getindex(u_ind,1) for u_ind in sol.u])) ||
            any(isnan.([getindex(u_ind,2) for u_ind in sol.u]))
            @info "Breaking early at the p value: $(p)"
                return 1e3
        end


        # Sum-of-squares difference to data
        # D_1[i] is v-data, D_2[i] is w-data, and sol.u[i][1], sol.u[i][2] the solution states
        # data_loss = 0.0
        # for (i, t) in enumerate(t_eval)
        #     data_loss += abs2(sol.u[i][1] - D_1[i]) + abs2(sol.u[i][2] - D_2[i])
        # end

        data_loss = norm(D_1- [getindex(u_ind,1) for u_ind in sol.u], 2)^2 + norm(D_2- [getindex(u_ind,2) for u_ind in sol.u], 2)^2
        # Add smooth L1 regularization if desired
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        # Normalize by number of data points (optional) and return
        return data_loss/length(t_eval) + sparse_loss/length(p)

    else
        # -----------------------------------------------------
        # NON-STIFF BRANCH (YOUR ORIGINAL LOOP-BASED APPROACH)
        # -----------------------------------------------------
        # Integration cache
        cache = Tsit5Cache(x0)
        x = cache.ycur
        copyto!(x, x0)

        weight_vector = 0.01 .+weight_func_linear.(1:length(D_1);α=1,n=length(D_1))
        # weight_vector = ones(length(D_1))

        # Initial condition loss
        data_loss = weight_vector[1]*(abs2(x0[1] - D_1[1]) + abs2(x0[2] - D_2[1]))
        integration_method = integration_step!
        # Loop over the data points
        for i = 1:length(D_1) - 1
            #Resetting the initial condition for each window using data
            if i % window_size == 0
                copyto!(x, [D_1[i], D_2[i]])
            end
            # copyto!(x, [D_1[i], D_2[i]])
            # Internal time steps
            for j = 1:S
                integration_method(cache, odefun, x, t0, p, δt, false)
            end

            # Detect blow-ups
            if any(abs.(x) .> 1e3) || any(isnan.(x))
                @info "Breaking early at p value $(p), solution blew up or NaN."
                return 1e3
            end
            # Data loss for current time point
            data_loss += weight_vector[i+1]*(abs2(x[1] - D_1[i+1]) + abs2(x[2] - D_2[i+1]))
        end

        # Sparsity term
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        return data_loss / (2*length(D_1)) + sparse_loss / length(p)
    end
end


"""
    data_assimilation_loss(X, p, odefun, data, α, β, δt, S, γ = 0.0)

Loss function for data assimilation with weak model loss using parameters p in odefun with time step δt. 
The data are given in the concatenated vector data and the data points are every S timesteps. 

α weights the data loss and β weights the model loss.

Optionally add a smooth L1 regularization term with weight γ.

TODO: At this point assumes the state is two dimension!!!
"""
function data_assimilation_loss(X, p, odefun, data, α, β, δt, S, γ = 0.0)
    # Model cache
    cache = Tsit5Cache(X[1:2])
    x = cache.ycur
    # Calculate the data loss
    data_loss = 0.0
    for i = 0:div(length(data), 2) - 1
        data_loss += abs2(X[2*S*i + 1] - data[2*i + 1])
        data_loss += abs2(X[2*S*i + 2] - data[2*i + 2])
    end

    # Calculate the model loss
    model_loss = 0.0
    offset = 2
    # Initial condition
    x[1] = X[1]; x[2] = X[2]
    # Loop over all state variables 
    for _ = 1:div(length(X), 2) - 1
        # Next time point 
        x3 = X[1 + offset]
        x4 = X[2 + offset]
        # Perform integration step 
        x = integration_step(cache, odefun, x, t0, p, δt, false)
        model_loss += abs2(x3 - x[1])
        model_loss += abs2(x4 - x[2])
        # Update the current state 
        x[1] = x3
        x[2] = x4
        offset += 2
    end

    # Sparsity term
    sparse_loss = 0.0
    for p_i in p
        sparse_loss += γ*smoothl1(p_i)
    end

    # Combine loss terms
    return (α * data_loss + β * model_loss) / length(X) + sparse_loss / length(p)
end

## Set up the loss functions and automatic differentiation
# Parameters 
S = 10       # Number of internal time steps
δt = Δt / S     # Time step 
α = 1.0         # Data loss weight 
β = 100.0       # Model loss weight 
γ1 = 5e-3        # Sparsity weight 
γ2 = 5e-2        # Sparsity weight
# Length of the DA state vector
Nx = 2*(div(length(data), 2) - 1) * S + 2 

# Two term loss functions 
# da_loss(X, p) = data_assimilation_loss(X, p, odefun, data, α, β, δt, S, γ1)
# fs_loss(x0, p) = forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2, true)
# # Compile the reverse differentiation tapes
# fs_loss_tape = ReverseDiff.GradientTape(fs_loss, (randn(2), similar(fhn_p)))
# compiled_fs_loss_tape = ReverseDiff.compile(fs_loss_tape)
# da_loss_tape = ReverseDiff.GradientTape(da_loss, (randn(Nx), similar(fhn_p)))
# compiled_da_loss_tape = ReverseDiff.compile(da_loss_tape)

function fs_loss(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    # Note: t0, dt, S, and γ2 are as defined in your code.
    return forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2, false)
    # return forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2, false)

end
# Compute the gradient of fs_loss_wrapper with respect to x0_full.
function grad_fs_loss!(g,x)
     g.= ForwardDiff.gradient(fs_loss, x)
end

# function fs_loss_window(x)
#     x0 = view(x, 1:2)
#     p  = view(x, 3:length(x))
#     # Note: t0, dt, S, and γ2 are as defined in your code.
#     return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2,30, false)
#     # return forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2, false)

# end
# # Compute the gradient of fs_loss_wrapper with respect to x0_full.
# function grad_fs_loss_window!(g,x)
#      g.= ForwardDiff.gradient(fs_loss, x)
# end



#= Check gradient calculations 
fs_inputs = (data[1:2], fhn_p)
fs_results = similar.(fs_inputs)
ReverseDiff.gradient!(fs_results, compiled_fs_loss_tape, fs_inputs)
da_loss(X, fhn_p)
da_inputs = (X, fhn_p)
da_results = similar.(da_inputs)
ReverseDiff.gradient!(da_results, compiled_da_loss_tape, da_inputs)
=# 

# One term input loss functions and two term gradients for optim 
# da_loss(x) = da_loss(view(x, 1:length(x) - Np), view(x, length(x) - Np + 1:length(x)))
# grad_da_loss!(g, x) = ReverseDiff.gradient!((view(g, 1:length(g) - Np), view(g, length(g) - Np + 1:length(g))), compiled_da_loss_tape, (view(x, 1:length(x) - Np), view(x, length(x)- Np + 1:length(x))))
# fs_loss(x) = fs_loss(view(x, 1:2), view(x, 3:length(x)))
# grad_fs_loss!(g, x) = ReverseDiff.gradient!((view(g, 1:2), view(g, 3:length(g))), compiled_fs_loss_tape, (x[1:2], x[3:length(x)]))

# # # Optimizations 
# # Data assimilation optimization 
# options = Optim.Options(show_trace = true, iterations = 5000, show_every = 10)
# x0 = [zeros(Nx); 0.01*zeros(length(fhn_p))]
# optres = Optim.optimize(da_loss, grad_da_loss!, x0, BFGS(), options)

# # Gradient Free optimizatino for better initial guess
# Define the number of seeds
num_runs = 1

# Pre-allocate storage for results
results_outer = Vector{Tuple{Float64, Vector{Float64}}}(undef, num_runs)
results_inner = Vector{Tuple{Float64, Vector{Float64}}}(undef, div(length(data), 2) - 1)

x0 = [data[1:2]; 0.1*randn(length(fhn_p))]
x0_ = [data[1:2]; 0.1*randn(length(fhn_p))]
copyto!(x0_, x0)
# Run optimization in parallel
@threads for i in 1:num_runs
# for i in 1:num_runs

    # seed = rand(1:10_000_000)  # Generate a large random seed
    Random.seed!(i)  # Set seed for reproducibility

    # Define initial starting point
    # x0 = [data[1:2]; 0.01 * randn(length(fhn_p))]  # Small random perturbations
    # Define initial starting point with uniform sampling within a hypercube
    # lower_bound = -.001
    # upper_bound = .001
    # initial_guess = lower_bound .+ (upper_bound - lower_bound) .* rand(length(fhn_p))

    options = Optim.Options(show_trace = false, iterations = 2500)
    # num_windows_range = div(length(data), 2) - 1:-5:1
    num_windows_range = 100:-1:1

    for num_windows in num_windows_range
        # Define the loss function
        function fs_loss_window(x)
            x0 = view(x, 1:2)
            p  = view(x, 3:length(x))
            # Note: t0, dt, S, and γ2 are as defined in your code.
            return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2,num_windows, false)
            # return forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2, false)
        
        end
        # Compute the gradient of fs_loss_wrapper with respect to x0_full.
        function grad_fs_loss_window!(g,x)
            g.= ForwardDiff.gradient(fs_loss, x)
        end
        # Optimization options

        # # Run optimization
        optres = Optim.optimize(fs_loss_window, grad_fs_loss_window!, x0, NelderMead(), options)
        # optres = Optim.optimize(fs_loss_window, grad_fs_loss_window!, x0, BFGS(), options)

        copyto!(x0, optres.minimizer)
        # copyto!(x0, x0_)
        println("GF, window size: $num_windows, cost: $(Optim.minimum(optres))")
        results_inner[num_windows] = (Optim.minimum(optres), Optim.minimizer(optres))
        
    end

    # # Store seed, cost function value, and minimizer
    # results_outer[i] = (results_inner[1][1], results_inner[1][2])
    # results_outer[i] = (results_inner[5][1], results_inner[5][2])
    results_outer[i] = argmin(r -> r[1], results_inner)



    println("finished GFree optimization run $i")
end

# Save results to a file (optional)
# using DataFrames
df = DataFrame(Cost = [r[1] for r in results_outer], 
               Minimizer = [r[2] for r in results_outer])

##
# Forward simulation optimization
options = Optim.Options(show_trace = true, iterations = 2500, show_every = 1)
# # Use parameters from DA optimization for the initial conditions of the FS optimization
# # x0 = [data[1:2]; optres.minimizer[end - Np + 1:end]]
# x0 = [data[1:2]; 0.01*zeros(length(fhn_p))]
# # x0 = [data[1:2]; fhn_p + 0.05*randn(length(fhn_p))]


# # optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0, BFGS(), options)
# optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0, NelderMead(), options)
# x0_2 = [data[1:2]; optres2.minimizer[end-Np + 1:end]]

x0_2 = [data[1:2]; df.Minimizer[argmin(df.Cost)][end-Np + 1:end]]
# x0_2 = [data[1:2]; 0.00*randn(length(fhn_p))]




optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0_2, BFGS(), options)

## Plot the results 
trmstr = repeat(["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"], 2)
cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]

fig = Figure(size = (980, 480))

# Plotting the GradFree simulation results
T = ts[end] 
N = Int(T / δt)
_, ys = integrate(odefun, df.Minimizer[argmin(df.Cost)][1:2], df.Minimizer[argmin(df.Cost)][3:end], t0, N, δt, cache)
ax11 = Axis(fig[1, 1])
scatter!(ax11, ts, data[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
lines!(ax11, 0:δt:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
lines!(ax11, tsall, alldata[1:2:end], color = plotGray, linestyle = :dash)
ylims!(ax11, -3, 3)
xlims!(ax11, 0.0, ts[end])
hidexdecorations!(ax11, ticks = false)
ax21 = Axis(fig[2, 1])
scatter!(ax21, ts, data[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
lines!(ax21, 0:δt:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
lines!(ax21, tsall, alldata[2:2:end], color = plotGray, linestyle = :dash)
ylims!(ax21, -0.6, 1.65)
xlims!(ax21, 0.0, ts[end])
ax21.xlabel = "t"
ax21.ylabel = "w(t)"
ax11.ylabel = "v(t)"
ax21.xticks = 0:25:75

# plot the GradFree coefficient results
ax31 = Axis(fig[3, 1])
barplot!(ax31, 1:Np, df.Minimizer[argmin(df.Cost)][end-Np + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
scatter!(ax31, 1:Np, fhn_p, color = cvec)
ax31.xticks = (1:Np, trmstr)
ax31.xticklabelrotation = π/4
hlines!(ax31, 0.0, color = plotGray)
ylims!(ax31, -1.1, 1.1)
xlims!(ax31, 0.25, Np + 0.75)

# # Plot the DA simulation results
# T = ts[end] 
# N = Int(T / δt)
# _, ys = integrate(odefun, optres.minimizer[1:2], optres.minimizer[end-Np + 1:end], t0, N, δt, cache)
# ax11 = Axis(fig[1, 1])
# scatter!(ax11, ts, data[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
# lines!(ax11, 0:δt:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
# lines!(ax11, tsall, alldata[1:2:end], color = plotGray, linestyle = :dash)
# ylims!(ax11, -3, 3)
# xlims!(ax11, 0.0, ts[end])
# hidexdecorations!(ax11, ticks = false)
# ax21 = Axis(fig[2, 1])
# scatter!(ax21, ts, data[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
# lines!(ax21, 0:δt:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
# lines!(ax21, tsall, alldata[2:2:end], color = plotGray, linestyle = :dash)
# ylims!(ax21, -0.6, 1.65)
# xlims!(ax21, 0.0, ts[end])
# ax21.xlabel = "t"
# ax21.ylabel = "w(t)"
# ax11.ylabel = "v(t)"
# ax21.xticks = 0:25:75

# # plot the DA coefficient results
# ax31 = Axis(fig[3, 1])
# barplot!(ax31, 1:Np, optres.minimizer[end-Np + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
# scatter!(ax31, 1:Np, fhn_p, color = cvec)
# ax31.xticks = (1:Np, trmstr)
# ax31.xticklabelrotation = π/4
# hlines!(ax31, 0.0, color = plotGray)
# ylims!(ax31, -1.1, 1.1)
# xlims!(ax31, 0.25, Np + 0.75)

# Plot the FS simulation results
T = ts[end] 
N = Int(T / δt)
_, ys = integrate(odefun, optres2.minimizer[1:2], optres2.minimizer[end-Np + 1:end], t0, N, δt, cache)
ax12 = Axis(fig[1, 2])
scatter!(ax12, ts, data[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
lines!(ax12, 0:δt:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
lines!(ax12, tsall, alldata[1:2:end], color = plotGray, linestyle = :dash)
ylims!(ax12, -3, 3)
xlims!(ax12, 0.0, ts[end])
hidexdecorations!(ax12, ticks = false)
ax22 = Axis(fig[2, 2])
scatter!(ax22, ts, data[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
lines!(ax22, 0:δt:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
lines!(ax22, tsall, alldata[2:2:end], color = plotGray, linestyle = :dash)
ylims!(ax22, -0.6, 1.65)
xlims!(ax22, 0.0, ts[end])
hideydecorations!.([ax12, ax22], ticks = false)
ax22.xlabel = "t"
ax22.xticks = 0:25:75

# plot the FS coefficient results
ax32 = Axis(fig[3, 2])
barplot!(ax32, 1:Np, optres2.minimizer[end-Np + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
scatter!(ax32, 1:Np, fhn_p, color = cvec)
ax32.xticks = (1:Np, trmstr)
ax32.xticklabelrotation = π/4
hlines!(ax32, 0.0, color = plotGray)
ylims!(ax32, -1.1, 1.1)
xlims!(ax32, 0.25, Np + 0.75)
hideydecorations!(ax32, ticks = false)
ax31.ylabel = "Parameter value"

ax11.title = "GradFree multi-shoot optimization, with guess propogation"

ax12.title = "Forward Simulation optimization"
ax11.titlefont = "Helvetica Neue Light"
ax12.titlefont = "Helvetica Neue Light"
fig
# save("figs/init_seed_0.1_with_guess_propogation.png", fig)





#================================================================================
# Plotting the contour plots of the loss function
================================================================================#
function plot_contours!(fig, loss_function, center_param; ref_point=nothing, delta_range_=0.5, n_grids,
        index_pairs=[(3, 4), (5, 6), (7, 8), (9, 10)], figure_title="Loss function landscape")


    ref_point = ref_point === nothing ? center_param : ref_point
    # Grid size and variation range
    Ngrid       = n_grids
    delta_range = delta_range_  # How far left/right to scan from center

    # Create a grid layout for proper spacing
    grid = fig[1, 1] = GridLayout()
    # Label(fig[0, 1], figure_title, fontsize=24, tellheight=false) 

    # Store subplot axes and colorbars separately
    axs = Matrix{Axis}(undef, 2, 2)
    cbs = Matrix{Colorbar}(undef, 2, 2)  # Store separate colorbars for each plot

    # Iterate over the four parameter pairs
    for (idx, (p1_index, p2_index)) in enumerate(index_pairs)
        row, col = Tuple(CartesianIndices((2, 2))[idx])  # Convert index to 2x2 grid positions

        # Create subplot axis
        ax = Axis(grid[row, 2col-1],  # Place axes in odd columns
            xlabel = "p$(p1_index)",
            ylabel = "p$(p2_index)",
            title  = "loss_function (p$(p1_index), p$(p2_index)): #Windows: $figure_title")

        axs[row, col] = ax  # Store axis reference

        # Extract central values for the parameters
        p1_center = center_param[p1_index]
        p2_center = center_param[p2_index]

        # Define parameter ranges
        p1_values = LinRange(p1_center - delta_range, p1_center + delta_range, Ngrid)
        p2_values = LinRange(p2_center - delta_range, p2_center + delta_range, Ngrid)

        # Preallocate a matrix to hold cost values
        costvals = Matrix{Float64}(undef, Ngrid, Ngrid)

        # Evaluate loss_function at each (p1, p2) pair
        for i in 1:Ngrid
            for j in 1:Ngrid
                # Copy the best-fit parameter vector, then replace p1 and p2
                p_trial = copy(center_param)
                p_trial[p1_index] = p1_values[i]
                p_trial[p2_index] = p2_values[j]

                # Evaluate the loss function
                costvals[i, j] = loss_function(p_trial)
            end
        end
          # Apply clamping if limits are set
        costvals .= clamp.(costvals, -10, 100)
        # Plot filled contour
        hm = contourf!(ax, p1_values, p2_values, costvals; levels=40, colormap=:viridis)

        # Highlight the optimal value with a **solid red circle**
        scatter!(ax, [center_param[p1_index]], [center_param[p2_index]], 
                color=:red, markersize=12, strokewidth=3, strokecolor=:black)

        # Highlight the ref param value with a **solid orange circle**
        scatter!(ax, [ref_point[p1_index]], [ref_point[p2_index]], 
                color=:orange, markersize=20, strokewidth=3, strokecolor=:black)
        # text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="num_windows: $num_windows", fontsize=15, color=:red)
        

        # Define unique label for each colorbar
        cb_label = "loss function (p$(p1_index), p$(p2_index))"

        # Place colorbars **in separate adjacent columns** (even-numbered columns)
        cbs[row, col] = Colorbar(grid[row, 2col]; colorrange=extrema(costvals), label=cb_label)
    end

    # return fig
end

#================================================================================#
S=20
fig = Figure(size = (1800, 1800))  # Define the figure once
num_windows_ = 20
param_center = [data[1:2]; fhn_p]
function fs_loss_window(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, num_windows_, false)
end

plot_contours!(fig, fs_loss_window, param_center;
ref_point=results_inner[num_windows_][2], delta_range_=1, n_grids=50,
index_pairs=[(3, 4), (5, 6), (7, 8), (10, 15)],
figure_title="$num_windows_"
)
fig
save("figs/contour_plots/S=$(S)_N_windows=$num_windows_.png", fig)

#================================================================================#






#===============================================================================
# Animating contour plots of the loss function
===============================================================================#
S = 20
num_windows = 100
fig = Figure(size = (1800, 1800))  # Define the figure once

num_windows_range = vcat(length(results_inner):-10:10,10:-1:1)

# Start recording
record(fig, "figs/num_windows_plots/contour_plots/testing_S=$S.mp4", num_windows_range;framerate=1) do num_windows
    empty!(fig)  # Clear the figure before each frame

    function fs_loss_window(x)
        x0 = view(x, 1:2)
        p  = view(x, 3:length(x))
        return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, num_windows, false)
    end

    best_sol = optres2.minimizer
    param_center = [data[1:2]; fhn_p]

    plot_contours!(fig, fs_loss_window, param_center;
        ref_point=results_inner[num_windows][2], delta_range_=1, n_grids=50,
        index_pairs=[(3, 4), (5, 6), (7, 8), (10, 15)],
        figure_title="$num_windows"
    )
    # text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="num_windows: $num_windows", fontsize=15, color=:red)


    println("done with num_windows = $num_windows")
end


#=================================================================================
=================================================================================#





#=================================================================================
# Plotting the cost surface 
=================================================================================#
S=10
num_windows_ = 1
function fs_loss_window(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, num_windows_, false)
end


fig = Figure(size=(1300, 1300))
p1_index, p2_index = (3, 4)
delta_range = 1
n_grids = 100
title = "Cost Landscape"
center_param = [data[1:2]; fhn_p]
ax = LScene(fig[1, 1])  # Create a 3D scene

# Define parameter center values
p1_center = center_param[p1_index]
p2_center = center_param[p2_index]

# Create parameter ranges
p1_values = LinRange(p1_center - delta_range, p1_center + delta_range, n_grids)
p2_values = LinRange(p2_center - delta_range, p2_center + delta_range, n_grids)

# Compute loss function values
Z = [fs_loss_window(vcat(center_param[1:p1_index-1], [p1], 
                   center_param[p1_index+1:p2_index-1], [p2], 
                   center_param[p2_index+1:end])) for p1 in p1_values, p2 in p2_values]

# Find the min and max of Z for scaling
# z_min, z_max = extrema(Z)
Z_clipped = clamp.(Z, 0, 10)
# Plot surface and store the plot object
plt = surface!(ax, p1_values, p2_values, Z_clipped; colormap=:viridis)
text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="num_windows: $num_windows_", fontsize=15, color=:red)

# Reduce the title height so it doesn't push the plot down too much
# Label(fig[0, 1], "Cost Landscape", fontsize=30, tellheight=false)  


# Reduce the height of the title row to bring it closer to the plot
# rowsize!(fig.layout, 0, Auto(0.1))  # Adjust row heig

# Adjust the camera/view angle
# cam3d!(ax.scene, azimuth=100, elevation=100)  # Adjust azimuth & elevation angles

fig
save("figs/surface_plots/S=$(S)_N_windows=$num_windows_.png", fig)

#=================================================================================
=================================================================================#





#=================================================================================
# Animating the cost surface
=================================================================================#
S=10
p1_index, p2_index = (3, 4)
delta_range = 1
n_grids = 50
title = "Cost Landscape"
center_param = [data[1:2]; fhn_p]
num_windows_range = vcat(length(results_inner):-5:5,4:-1:1)
# Start recording
record(fig, "figs/surface_plots/S=$S.mp4", num_windows_range;framerate=1) do num_windows
    empty!(fig)  # Clear the figure before each frame

    function fs_loss_window(x)
        x0 = view(x, 1:2)
        p  = view(x, 3:length(x))
        return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, num_windows, false)
    end

    ax = LScene(fig[1, 1])  # Create a 3D scene

    # Define parameter center values
    p1_center = center_param[p1_index]
    p2_center = center_param[p2_index]

    # Create parameter ranges
    p1_values = LinRange(p1_center - delta_range, p1_center + delta_range, n_grids)
    p2_values = LinRange(p2_center - delta_range, p2_center + delta_range, n_grids)

    # Compute loss function values
    Z = [fs_loss_window(vcat(center_param[1:p1_index-1], [p1], 
                    center_param[p1_index+1:p2_index-1], [p2], 
                    center_param[p2_index+1:end])) for p1 in p1_values, p2 in p2_values]

    # Find the min and max of Z for scaling
    # z_min, z_max = extrema(Z)
    Z_clipped = clamp.(Z, 0, 200)
    # Plot surface and store the plot object
    plt = surface!(ax, p1_values, p2_values, Z_clipped; colormap=:viridis)

    # Label(fig[0, 1], "num windows: $num_windows", fontsize=23, tellheight=false)  
    # Add X-axis label BELOW the plot
    # Label(fig[2, 1], "num windows: $num_windows", fontsize=18)
    text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="num_windows: $num_windows", fontsize=15, color=:red)

    # Reduce the height of the title row to bring it closer to the plot
    # rowsize!(fig.layout, 0, Auto(0.1))  # Adjust row heig

    # Adjust the camera/view angle
    cam3d!(ax.scene, azimuth=100, elevation=100)  # Adjust azimuth & elevation angles




    println("done with num_windows = $num_windows")
end


#=================================================================================
=================================================================================#




