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

prob_fhn = ODEProblem(odefun, fhn_y0, (0.0, 100.0), fhn_p)
sol_fhn = solve(prob_fhn, Tsit5(), abstol=1e-8, reltol=1e-8)

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
function custom_integrator(odefun_, x, p_, t0_, N, δt_, cache = Tsit5Cache(y0);
    solver_=nothing, calcerr = false, adaptive_=true, rel_tol_=1e-8, abs_tol_=1e-8) 
    solver_ = solver_ === nothing ? Tsit5() : solver_
    t_final = t0_+N*δt_
    prob = ODEProblem(odefun_, x, (t0_, t_final) , p_)
    sol = solve(prob, solver_;dt=δt_, adaptive=adaptive_,
     saveat=t0_:δt_:t_final, abstol=abs_tol_, reltol=rel_tol_)
    #  println(sol.t,",  ",sol.u)
    #  println(typeof(sol.t),",  ", typeof(sol.u))
     return (sol.t,sol.u)
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
function forward_simulation_loss_windows(x0_, unstable_param, p_, odefun_, D_, t0_, δt_, S_, 
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
        prob = ODEProblem(odefun_, x, (t0_, t0_+δt_),p_)
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
                push!(unstable_param, vcat(x,p_))
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
        x_temp = deepcopy(x0_)

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
            copyto!(x_temp, x)
            # Internal time steps
            for j = 1:S_
                # println("S_: $S_, j: $j")
                # copyto!(x_temp, x)
                integration_method(cache, odefun_, x, t0_, p_, δt_, false)
            end

            # Detect blow-ups
            if any(abs.(x) .> 1e3) || any(isnan.(x))
                if !silent_
                    @info "Breaking at window: $((i-1)/window_size_) with window_size: $window_size_, solution blew up or NaN."
                end
                push!(unstable_param, vcat(x_temp,p_))
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


function dummy_fun()
end
## Set up the loss functions and automatic differentiation
# Parameters 
S = 10    # Number of internal time steps
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
# window_size_range = 1:(div(div(length(data),2)-1, 2)+5)
# window_size_range = vcat(1:(div(div(length(data),2)-1, 2)+5), [100])
window_size_range = [1,50,100]


# Pre-allocate storage for results
results_outer = Vector{Tuple{Float64, Vector{Float64}}}(undef, num_runs)
# results_inner = Vector{Tuple{Float64, Vector{Float64}}}(undef, div(length(data), 2) - 1)
results_inner = Dict{Int, Tuple{Float64, Vector{Float64}}}()
unstable_points_dict = Dict()
# solver = Tsit5()
solver = Rosenbrock23()
# solver = ImplicitEuler()
# solver = RadauIIA5()

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
        unstable_points = []
        function fs_loss_window(x)
            x0 = view(x, 1:2)
            p  = view(x, 3:length(x))
            return forward_simulation_loss_windows(x0, unstable_points, p, odefun, data, 0.0, δt, S, γ2, window_size, false;
            solver_=solver, silent_=false, adaptive_=true, abs_tol_=1e-8, rel_tol_=1e-8)
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
        # if Optim.minimum(optres) < best_min
        #     best_min = Optim.minimum(optres)
        #     copyto!(x0, optres.minimizer)
        # end
        #Always updating guess with the current minimum
        # copyto!(x0, optres.minimizer)
        #Resetting the initial guess for the each new window size
        copyto!(x0, x0_)

        println("GF, window size: $window_size, cost: $(Optim.minimum(optres))")
        results_inner[window_size] = (Optim.minimum(optres), Optim.minimizer(optres))
        unstable_points_dict[window_size] = deepcopy(unstable_points)
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
# δt=0.01
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







