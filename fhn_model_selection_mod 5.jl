include("integrator.jl")
include("plotting_window_size.jl")
# Packages for plotting and default theming
using CairoMakie
using LinearAlgebra, Statistics
using Optim, ForwardDiff
using Random
using Base.Threads, DataFrames
using DifferentialEquations
using BenchmarkTools
using Logging
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
function forward_simulation_loss_windows(x0_, p_, odefun_, D_, t0_, δt_, S_, 
    γ_ = 0.0, window_size_=1, stiff_solver_=false; solver_=nothing,
    abs_tol_=abstol=1e-8, rel_tol_=abstol=1e-8, adaptive_=false, silent_=false)
# S: internal steps (only relevant when stiff_solver=false in your code)
# D_1, D_2: the data columns for v(t) and w(t)

    D_1 = @view D_[1:2:end]
    D_2 = @view D_[2:2:end]
    #window_size is k, k+1 data points are used to form a window
    #IF window size is 1, then a window is formed between every consecutive data points.
    # If window size is length(data)-1, the forward_simulation_loss function. 
    #Defaul behaviour is to take the window size as 1
    if window_size_ < 1 || window_size_ > length(D_1) - 1
        @warn "Window size out of bounds. Clamping to valid range [1, $(length(D_1) - 1)]."
        window_size_ = clamp(window_size_, 1, length(D_1) - 1)
    end

    if stiff_solver_
        # -----------------------------------------------------
        # STIFF SOLVER BRANCH
        # -----------------------------------------------------
        x = deepcopy(x0_)
        solver_ = solver_ === nothing ? Tsit5() : solver_
        prob = ODEProblem(odefun_, x, (t0_, t0_+δt_))
        t_init = t0_
        t_final = t0_
        Δt = δt_*S_
        data_loss = 0.0
        for shoot_ind in 1:window_size_:length(D_1)-1
            t_init = t_final
            t_final = clamp(t_init + window_size_*Δt, 0, t0_+Δt*(length(D_1) - 1))
            copyto!(x, [D_1[shoot_ind], D_2[shoot_ind]])
            prob = remake(prob, u0=x, tspan=(t_init, t_final), p=p_)
            # prob = ODEProblem(odefun_, x, (t_init, t_final), p_)

            # sol = solve(prob, solver_; saveat=t_init:Δt:t_final, abstol=abs_tol_, reltol=rel_tol_)
            if !silent_
                # sol = solve(prob, solver_; saveat=t_init:Δt:t_final, abstol=abs_tol_, reltol=rel_tol_)
                sol = solve(prob, solver_;dt=δt_, adaptive=adaptive_, saveat=t_init:Δt:t_final, abstol=abs_tol_, reltol=rel_tol_)

            else
                sol = with_logger(NullLogger()) do
                    solve(prob, solver_;dt=δt_, adaptive=adaptive_, saveat=t_init:Δt:t_final, abstol=abs_tol_, reltol=rel_tol_)
                end
            end

            # println("shoot_index:", shoot_ind, ",  data considered: ", shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1)), ",  time_interval ",t_init:Δt:t_final, ",  t_Final: $t_final" )


            # # Check for blow-ups or NaNs in the solution
            if length(shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1))) != length([getindex(u_ind,1) for u_ind in sol.u]) ||
                any([getindex(u_ind,1) for u_ind in sol.u] .> 1e3) ||
                any([getindex(u_ind,2) for u_ind in sol.u] .> 1e3) ||
                any(isnan.([getindex(u_ind,1) for u_ind in sol.u])) ||
                any(isnan.([getindex(u_ind,2) for u_ind in sol.u]))
                if !silent_
                    @info "Solution blew up or broken at shooting node: $shoot_ind"
                    # print(window_size_+1) != length([getindex(u_ind,1) for u_ind in sol.u]))
                end
                    return 1e3
            end
            data_loss += dot(D_1[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1))]- [getindex(u_ind,1) for u_ind in sol.u], 
                           D_1[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1))]- [getindex(u_ind,1) for u_ind in sol.u]) 
            + dot(D_2[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_2))]- [getindex(u_ind,2) for u_ind in sol.u], 
                           D_2[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_2))]- [getindex(u_ind,2) for u_ind in sol.u])

            #optionally setting the initial condition for the next window to the final condition of the previous window
            # x = sol.u[end]
        end

        # Add smooth L1 regularization if desired
        sparse_loss = sum(γ_ * smoothl1(pi) for pi in p_)

        # Normalize by number of data points (optional) and return
        return data_loss/(2*length(D_1)) + sparse_loss/length(p_)

    else
        # -----------------------------------------------------
        # NON-STIFF BRANCH (YOUR ORIGINAL LOOP-BASED APPROACH)
        # -----------------------------------------------------
        # Integration cache
        cache = Tsit5Cache(x0_)
        x = cache.ycur
        copyto!(x, x0_)

        # weight_vector = 0.01 .+weight_func_linear.(1:length(D_1);α=1,n=length(D_1))
        weight_vector = ones(length(D_1))

        # Initial condition loss
        data_loss = weight_vector[1]*(abs2(x0_[1] - D_1[1]) + abs2(x0_[2] - D_2[1]))
        integration_method = integration_step!
        # Loop over the data points
        for i = 1:length(D_1) - 1
            #Resetting the initial condition for each window using data
            if (i-1) % window_size_ == 0
                copyto!(x, [D_1[i], D_2[i]])
            end
            # copyto!(x, [D_1[i], D_2[i]])
            # Internal time steps
            for j = 1:S_
                # println("S_: $S_, j: $j")
                integration_method(cache, odefun_, x, t0_, p_, δt_, false)
            end

            # Detect blow-ups
            if any(abs.(x) .> 1e3) || any(isnan.(x))
                if !silent_
                    @info "Breaking at window: $((i-1)/window_size_) with window_size: $window_size_, solution blew up or NaN."
                end
                return 1e3
            end
            # Data loss for current time point
            data_loss += weight_vector[i+1]*(abs2(x[1] - D_1[i+1]) + abs2(x[2] - D_2[i+1]))
        end

        # Sparsity term
        sparse_loss = sum(γ_ * smoothl1(pi) for pi in p_)

        return data_loss / (2*length(D_1)) + sparse_loss / length(p_)
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
        x = integration_step(cache, odefun, x, t0_, p, δt, false)
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



x0 = [data[1:2]; 0.1*randn(length(fhn_p))]
x0_ = [data[1:2]; 0.1*randn(length(fhn_p))]
copyto!(x0_, x0)
# Run optimization in parallel

# window_size_range = 1:div(length(data),2)-1
window_size_range = 1:(div(div(length(data),2)-1, 2)+5)

# Pre-allocate storage for results
results_outer = Vector{Tuple{Float64, Vector{Float64}}}(undef, num_runs)
# results_inner = Vector{Tuple{Float64, Vector{Float64}}}(undef, div(length(data), 2) - 1)
results_inner = Dict{Int, Tuple{Float64, Vector{Float64}}}()

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
    best_min = 10000
    for window_size in window_size_range
        # Define the loss function
        function fs_loss_window(x)
            x0 = view(x, 1:2)
            p  = view(x, 3:length(x))
            # Note: t0, dt, S, and γ2 are as defined in your code.
            return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2,window_size, false)
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

        #updating guess only if we get a better minimum
        if Optim.minimum(optres) < best_min
            best_min = Optim.minimum(optres)
            copyto!(x0, optres.minimizer)
        end
        #Always updating guess with the current minimum
        # copyto!(x0, optres.minimizer)
        #Resetting the initial guess for the each new window size
        # copyto!(x0, x0_)

        println("GF, window size: $window_size, cost: $(Optim.minimum(optres))")
        results_inner[window_size] = (Optim.minimum(optres), Optim.minimizer(optres))
        
    end

    # # Store seed, cost function value, and minimizer
    # results_outer[i] = (results_inner[1][1], results_inner[1][2])
    # results_outer[i] = (results_inner[5][1], results_inner[5][2])
    # results_outer[i] = argmin(r -> r[1], results_inner)
    results_outer[i] = results_inner[argmin(r -> results_inner[r][1], keys(results_inner))]




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



#Plotting solutions
guess_prop_minimizer = df.Minimizer[argmin(df.Cost)]
FS_minimizer = optres2.minimizer

fig = Figure(size = (980, 480))
plotSolution!(fig, guess_prop_minimizer, FS_minimizer, ts, data, tsall, alldata, δt, t0, fhn_p, Np, odefun, cache;
    DA=false, GFreeFS=true, FS=true, integration_method = nothing)
fig
save("figs/window_size_plots/init_seed_0.1_with_best_guess_propogation.png", fig)


#Animating expanding window solutions
fig = Figure(size = (480, 480))
save_solution_animation!(fig, results_inner, ts, data, tsall, alldata, δt, t0, fhn_p, Np, odefun, cache;
    path_="figs/window_size_plots/best_guess_evolve.mp4", integration_method = nothing)       




#Saving cost_value vs window_size
fig = Figure()
save_cost_vs_windows!(fig, results_inner; title_= "best_guess_propogation")
save("figs/window_size_plots/cost_vs_windows_with_best_guess_prop.png", fig)



#Plotting contour_plots
S=100
δt = Δt / S     # Time step 
fig = Figure(size = (1800, 1800)) 
window_size_range = sort(collect(keys(results_inner)))
window_size = window_size_range[1]
window_size = 1
num_grids=50
delta_range=20
# solver = Rosenbrock23()
solver = nothing
function fs_loss_window(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size, true;
    solver_=solver, silent_=true, adaptive_=false)
end
center_param = [data[1:2]; fhn_p]
# center_param = [data[1:2]; -1.0.*ones(length(fhn_p))]

ref_point = results_inner[window_size_range[end]][2]
plot_contours!(fig, fs_loss_window, center_param; ref_point_=ref_point, delta_range_=delta_range, n_grids_=num_grids,
    index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], figure_title_="Window size: $window_size")
fig
save("figs/window_size_plots/contour_plots/S=$(S)_window_size=$window_size.png", fig)

#Animating contour plots
S=10
δt = Δt / S     # Time step 
fig = Figure(size = (1200, 1200)) 
center_param = [data[1:2]; fhn_p]
# results_inner_trimmed = Dict(k => results_inner[k] for k in 1:10:55)
# window_size_range = sort(collect(keys(results_inner_trimmed)))
n_grids = 50
window_size_range = 1:5:div(length(data),2)-1
# solver = Rosenbrock23()
solver = nothing
record(fig, "figs/window_size_plots/contour_plots/Rosenbrock23_S=$S.mp4", window_size_range;framerate=1) do window_size_
    empty!(fig)  # Clear the figure before each frame

    function fs_loss_window(x)
        x0 = view(x, 1:2)
        p  = view(x, 3:length(x))
        return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size_, true; silent_=true,
        solver_=solver, adaptive_=false)
        # return loss_function_window(x0, p, odefun, data, 0.0, δt, S, 10, window_size_, false; silent=true)
    end

    ref_point = haskey(results_inner, window_size_) ? results_inner[window_size_][2] : results_inner[argmin(r -> results_inner[r][1], keys(results_inner))][2]
    loss_function = fs_loss_window

    plot_contours!(fig, loss_function, center_param; ref_point_=ref_point, delta_range_=1, n_grids_=n_grids,
        index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], figure_title_="window_size: $window_size_")

    println("done with window_size = $window_size_")
end
# animate_contourplots!(fig, center_param, data, δt, γ2, odefun; loss_function_ = nothing, ref_point_=nothing, delta_range_=1, n_grids_=50,
#     loss_function_window_=forward_simulation_loss_windows,
#     index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], result_dict_=results_inner_trimmed, implicit_scheme_=false, figure_title="Contour ", S_=S_,
#     path_="figs/window_size_plots/contour_plots/S=$S_.mp4")


include("plotting_window_size.jl")

#Plotting cost function surface
S=10
δt = Δt / S     # Time step 
fig = Figure(size=(1300, 1300))
# center_param = [data[1:2]; fhn_p]
center_param = [data[1:2]; -1.0.*ones(length(fhn_p))]
window_size_range = 1:5:div(length(data),2)-1
window_size_ = 2
# window_size_ = window_size_range[end]
delta_range = 1
# n_grids = 50
solver = Rosenbrock23()
# solver = nothing
function fs_loss_window(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size_, true; silent_=true,
    solver_=solver, adaptive_=true)
    # return loss_function_window(x0, p, odefun, data, 0.0, δt, S, 10, window_size_, false; silent=true)
end

p1_index, p2_index = 3,4
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
surface!(ax, p1_values, p2_values, Z_clipped; colormap=:viridis)
text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="window_size: $window_size_", fontsize=15, color=:red)

fig
save("figs/window_size_plots/surface_plots/S=$(S)window_size=$window_size_.png", fig)

#Animating cost function surface
fig = Figure(size=(1300, 1300))
center_param = [data[1:2]; fhn_p]
window_size_range = 1:5:div(length(data),2)-1
n_grids = 50
delta_range = 1
p1_index, p2_index = 5,6

# Start recording
record(fig, "figs/window_size_plots/surface_plots/S=$S_.mp4", window_size_range;framerate=1) do window_size_
    empty!(fig)  # Clear the figure before each frame

    function fs_loss_window(x)
        x0 = view(x, 1:2)
        p  = view(x, 3:length(x))
        return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size_, false;silent_=true)
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

    # Add X-axis label BELOW the plot
    text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="window_size: $window_size_", fontsize=15, color=:red)

    # Reduce the height of the title row to bring it closer to the plot
    # rowsize!(fig.layout, 0, Auto(0.1))  # Adjust row heig

    # Adjust the camera/view angle
    cam3d!(ax.scene, azimuth=100, elevation=100)  # Adjust azimuth & elevation angles




    println("done with window_size = $window_size_")
end
fig



#Looking at Hessians
#Plotting contour_plots
H_vector = [Matrix{Float64}(undef, 22, 22) for _ in 1:10]
eig_values_vector = [Vector{Float64}(undef, 22) for _ in 1:10]
eig_vectors_vector = [Matrix{Float64}(undef, 22, 22) for _ in 1:10]

for (ind,S) in enumerate(10:10:100)
    # S=100
    println(S)
    δt = Δt / S     # Time step 
    fig = Figure(size = (1800, 1800)) 
    window_size_range = sort(collect(keys(results_inner)))
    window_size = window_size_range[1]
    window_size = 5
    num_grids=50
    delta_range=1
    # solver = Rosenbrock23()
    solver = Rodas5()

    # solver = nothing
    function fs_loss_window(x)
        x0 = view(x, 1:2)
        p  = view(x, 3:length(x))
        return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size, true;
        solver_=solver, silent_=true, adaptive_=false)
    end
    function hessian_fs_loss_window(x::AbstractVector{<:Real})
        ForwardDiff.hessian(fs_loss_window, x)
    end

    center_param = [data[1:2]; fhn_p]
    # center_param = [data[1:2]; -1.0.*ones(length(fhn_p))]

    ref_point = results_inner[window_size_range[end]][2]
    # plot_contours!(fig, fs_loss_window, center_param; ref_point_=ref_point, delta_range_=delta_range, n_grids_=num_grids,
    #     index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], figure_title_="Window size: $window_size")
    # fig

    # H_vector[ind]= hessian_fs_loss_window(vcat([data[1],data[2]],zeros(length(fhn_p))))
    H_vector[ind]= hessian_fs_loss_window(ref_point)


    eig_values_vector[ind], eig_vectors_vector[ind] = eigen(H_vector[ind])
end

for eig_values in eig_values_vector
    sort(eig_values)
end
fig = Figure(size = (900, 900))
ax = Axis(fig[1, 1], title="Visualization of 10 Eigenvalue Vectors", xlabel="Eigenvalue Index", ylabel="Eigenvalue")
colors = Makie.ColorSchemes.tab20
# colors_ = distinguishable_colors(num_vectors)
# Create a scatter plot
# plot(title="Visualization of 10 Eigenvalue Vectors", xlabel="Eigenvalue Index", ylabel="Eigenvalue")

for (i,eigen_values) in enumerate(eig_values_vector)
    scatter!(ax,1:22, eigen_values, color=colors[i],label="Vector $i", markersize=5)
end
fig


# Create a figure
fig = Figure(resolution=(800, 1000))
ax = Axis(fig[1, 1], title="Visualization of 10 Eigenvalue Vectors", xlabel="Eigenvalue Index", ylabel="Eigenvalue")

# Store legend entries
scatter_plots = []
colors = Makie.ColorSchemes.tab20
# Scatter plot for each vector with a different color
for (i,eigen_values) in enumerate(eigen_values_vector_explicit)
    plt = scatter!(ax,1:22, eigen_values, color=colors[i],label="Vector $i", markersize=5)
    push!(scatter_plots, (plt, "Vector $i"))  # Store plot & label for legend
end
fig

sum_=0
for (i,eigen_values) in enumerate(eig_values_vector)
    sum_local= dot(eigen_values-eig_values_vector[1], eigen_values-eig_values_vector[1])
    println("Eigenvalue Vector $i: ", sum_local)
    sum_+=sum_local
end
sum_

eigen_values_vector_explicit = deepcopy(eig_values_vector)

sum_=0
for (i,eigen_values) in enumerate(eig_values_vector)
    sum_local= dot(eigen_values_vector_explicit[i]-eig_values_vector[i], eigen_values_vector_explicit[i]-eig_values_vector[i])
    println("Eigenvalue Vector $i: ", sum_local)
    sum_+=sum_local
end
sum_


S=100
function fs_loss_window(x)
    x0 = view(x, 1:2)
    p  = view(x, 3:length(x))
    return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size, true;
    solver_=solver, silent_=true, adaptive_=false)
end
function hessian_fs_loss_window(x::AbstractVector{<:Real})
    ForwardDiff.hessian(fs_loss_window, x)
end

center_param = [data[1:2]; fhn_p]
# center_param = [data[1:2]; -1.0.*ones(length(fhn_p))]

ref_point = results_inner[window_size_range[end]][2]
# hessian_fs_loss_window(ref_point)
