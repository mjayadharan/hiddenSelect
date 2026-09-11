include("integrator.jl")
include("plotting_window_size.jl")
include("helper_functions.jl")
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

"""
    odefun_poly!(du, u, p, t)

Defines the ordinary differential equation (ODE) function for a polynomial system.

# Arguments
- `du::AbstractVector`: The derivative of the state vector `u` with respect to time.
- `u::AbstractVector`: The state vector representing the current values of the system variables.
- `p::AbstractVector`: The parameter vector containing coefficients or constants for the polynomial system.
- `t::Real`: The current time.

# Description
This function computes the time derivative `du` of the state vector `u` based on a polynomial system defined by the parameters `p`. It is designed to be used with ODE solvers in Julia, such as those provided by the DifferentialEquations.jl package.

# Notes
- The function modifies `du` in place to improve performance and reduce memory allocations.
- Ensure that the dimensions of `u` and `p` are consistent with the polynomial system being modeled.
- Make sure that the parameters in `p` are ordered correctly to match the polynomial terms in the ODE system.
 For example, for dim=2, deg=3, the parameters in p are ordered as follows: 
 "1"  "x_2"  "x_2^2"  "x_2^3"  "x_1"  "x_1 * x_2"  "x_1 * x_2^2"  "x_1^2"  "x_1^2 * x_2"  "x_1^3"
 For other dimensions, and degrees, use the helper function multiindex_mapping to find teh correct ordering.
"""
@inline function odefun_poly!(du, u, p, t; indices=nothing, deg=3)
    num_monomials = length(indices)         # number of monomials
    dim = length(u)                         # dimension
    T = eltype(u)

    # Precompute u[j]^d for d=0:deg and for each j=1:dim
    # This avoids the inner loop having to compute exponentiation repeatedly.
    pow_table = Matrix{T}(undef, dim, deg+1)
    @inbounds for j in 1:dim
        pow_table[j, 1] = one(T)            # u[j]^0 = 1
        for d in 1:deg
            pow_table[j, d+1] = u[j]^d      # u[j]^d
        end
    end

    @inbounds for i in 1:dim
        offset = (i - 1) * num_monomials
        s = zero(T)
        @simd for k in 1:num_monomials
            prod_val = one(T)
            α = indices[k]  # α should be a tuple, SVector, or similar of integers.
            @inbounds for j in 1:dim
                # Use precomputed power: note α[j] is assumed to be in 0:deg.
                prod_val *= pow_table[j, α[j] + 1]
            end
            s += p[offset + k] * prod_val
        end
        du[i] = s
    end
    return du
end

# Default parameters for FHN
#Note that the orderin which the monomials appear in the parameters in is differn from odefun
#parameter arrangement  "1"  "x_2"  "x_2^2"  "x_2^3"  "x_1"  "x_1 * x_2"  "x_1 * x_2^2"  "x_1^2"  "x_1^2 * x_2"  "x_1^3"
fhn_p = [0.5, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1/3,
0.7/12.5,  -0.8/12.5, 0.0, 0.0, 1.0/12.5, 0.0, 0.0, 0.0, 0.0, 0.0]


# Number of parameters in odefun
const Np = 20

# # Default parameters for FHN (old version)
# fhn_p = [0.5, 1.0, -1.0, 0.0, 0.0, 0.0, -1/3, 0.0, 0.0, 0.0,
#  0.7/12.5, 1.0/12.5, -0.8/12.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
fhn_y0 = [1.0, 1.0]

# prob_fhn = ODEProblem(odefun, fhn_y0, (0.0, 100.0), fhn_p)
multi_index_set = multiindices(2, 3) #Multindex for identifying degree of state variables in monomials for dim=2 upto degree 3
prob_fhn = ODEProblem((du,u,p,t) -> odefun_poly!(du,u,p,t; indices=multi_index_set, deg=3),
fhn_y0, (0.0, 100.0), fhn_p)

sol_fhn = solve(prob_fhn, Tsit5(), abstol=1e-8, reltol=1e-8)



## Integrate FHN data
cache = Tsit5Cache(fhn_y0) # Build cache
δt = 0.01                  # Find time step
T = 156                    # Maxium integration time
N = Int(T/δt)              # Number of integration steps
t0 = 0.0                   # Initial time 
# Integrate 
_, ys = integrate((du, u, p, t) -> odefun_poly!(du, u, p, t; indices=multi_index_set, deg=3),
                  fhn_y0, fhn_p, t0, N, δt, cache)

tsall = 0.0:δt:T

# Downsample in time to generate sparser data
downsample = 100

# Instead of interlacing odd and even indices, construct a matrix where:
#   - Column 1 holds the v(t) data from getindex.(ys, 1)
#   - Column 2 holds the w(t) data from getindex.(ys, 2)
# Each row corresponds to one time point.
alldata = hcat(getindex.(ys, 1), getindex.(ys, 2))

# Downsample the rows of alldata and corresponding time points
data = alldata[1:downsample:end, :]  # data is now (num_points × 2)
ts = tsall[1:downsample:end]

# Crop in time using tsall indices (here, from t = 50 to t = 150)
ind_tall1 = argmin(abs.(tsall .- 50))
ind_tall2 = argmin(abs.(tsall .- 150))
alldata = alldata[ind_tall1:ind_tall2, :]
tsall = tsall[ind_tall1:ind_tall2]

# Crop the downsampled ts similarly using argmin on ts
ind1 = argmin(abs.(ts .- 50))
ind2 = argmin(abs.(ts .- 150))
ts = ts[ind1:ind2]
t1 = ts[1]
ts  = ts .- t1
tsall = tsall .- t1

Nd = size(data, 1)

# Crop data accordingly (using the same indices from ts)
data = data[ind1:ind2, :]

# Add noise separately to each state variable column.
# Column 1 is v(t) and column 2 is w(t).
Random.seed!(1287436679)
noise_sigma = 0.05
data[:, 1] .+= noise_sigma * std(data[:, 1]) * randn(size(data, 1))
data[:, 2] .+= noise_sigma * std(data[:, 2]) * randn(size(data, 1))
#appending time stamps to data
data = hcat(ts, data)

# Data time step
Δt = ts[2] - ts[1]

# Plot the result
fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, tsall, alldata[:, 1], color = Makie.ColorSchemes.Signac[12])
scatter!(ax1, data[:,1], data[:, 2], color = Makie.ColorSchemes.Signac[12], markersize = 8)
ax2 = Axis(fig[2, 1])
lines!(ax2, tsall, alldata[:, 2], color = Makie.ColorSchemes.Signac[11])
scatter!(ax2, data[:,1], data[:, 3], color = Makie.ColorSchemes.Signac[11], markersize = 8)
hidexdecorations!(ax1, ticks = false)
ax1.ylabel = "v(t)"
ax2.ylabel = "w(t)"
ax2.xlabel = "t"
fig
##################################################################################
# End of data generation
##################################################################################

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


"""
    forward_simulation_loss(x0, p, odefun, data, t0, δt, S, γ=0.0, stiff_solver=false)

Compute the loss by forward-simulating the ODE starting at initial condition `x0` 
using parameters `p` and comparing the solution to `data`.

Arguments:
- `x0` : initial state vector of length d.
- `p`  : parameter vector.
- `odefun` : ODE function of signature `odefun(du, u, p, t)` (works for any dimension d).
- `data` : a Nxd matrix where first column corresponds to the time time points,other columns represent a
           distinct state and row represent different times of measurement.
- `t0`   : initial time.
- `δt`   : integration time step.
- `S`    : number of internal steps per data point.
- `γ`    : sparsity regularization weight.
- `stiff_solver` : flag to select the stiff solver branch.

Returns the loss (data misfit plus a regularization term).
"""
function forward_simulation_loss(x0, p, odefun, data_, t0, δt, S, 
    γ = 0.0, stiff_solver=false)
    # S: internal steps (only relevant when stiff_solver=false in your code)
    N = size(data_,1 )    # N: number of data points
    d = size(data_,2) - 1 # d: state dimension (Note that first column of data is time pionts)
    # D_1 = @view data_[1:2:end]
    # D_2 = @view data_[2:2:end]
    if stiff_solver
        # -----------------------------------------------------
        # STIFF SOLVER BRANCH
        # -----------------------------------------------------
        final_time = t0 + δt*(N - 1)
        # t_eval = t0:δt:final_time
        t_eval = data_[:,1]

        # Set up the ODE problem
        prob = ODEProblem(odefun, x0, (t0, final_time), p)

        # Solve with a stiff solver, e.g., Rodas3
        # sol = solve(prob, Rodas3(); saveat=t_eval, abstol=1e-6, reltol=1e-6)
        sol = solve(prob, Tsit5(); saveat=t_eval, abstol=1e-8, reltol=1e-8)
        #Refactor: Avoid this allocation and use the  solution vector directly. 
        sol_mat = hcat(sol.u...)'


        # # Check for blow-ups or NaNs in the solution
        if N != size(sol_mat, 1) || any(sol_mat .> 1e3) || any(isnan.(sol_mat))
            @info "Breaking early at the p value: $(p)"
                return 1e3
        end

        # data_loss = norm(D_1- [getindex(u_ind,1) for u_ind in sol.u], 2)^2 + norm(D_2- [getindex(u_ind,2) for u_ind in sol.u], 2)^2
        data_loss = norm(data_[:,2:end]- sol_mat)^2
        # Add smooth L1 regularization if desired
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        # Normalize by number of data points (optional) and return
        return data_loss/N + sparse_loss/length(p)

    else
        # -----------------------------------------------------
        # NON-STIFF BRANCH
        # -----------------------------------------------------
        # Integration cache
        cache = Tsit5Cache(x0)
        x = cache.ycur
        copyto!(x, x0)

        # Initial condition loss
        # data_loss = abs2(x0[1] - D_1[1]) + abs2(x0[2] - D_2[1])
        data_loss = norm(x0 - data[1, 2:end])^2
        integration_method = integration_step!

        # Loop over the data points
        for i = 1:N - 1
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
            # data_loss += abs2(x[1] - D_1[i+1]) + abs2(x[2] - D_2[i+1])
            data_loss += norm(x - data[i+1, 2:end])^2
        end

        # Sparsity term
        sparse_loss = sum(γ * smoothl1(pi) for pi in p)

        return data_loss/N + sparse_loss/length(p)
    end
end

#Function for weighting data loss
weight_func_linear(x;α=1,n=10) = α*(x-1)/(n-1)
weight_func_exp(x; α=1, n=10, k=0.05) = α * (exp(k*(x - 1)) - 1) / (exp(k*(n - 1)) - 1)
weight_func_quad(x; α=1, n=10) = α * ((x - 1)^2) / ((n - 1)^2)


"""
    forward_simulation_loss_windows(x0_, unstable_param, p_, odefun_, data_, t0_, δt_, S_, 
        γ_ = 0.0, window_size_=1, stiff_solver_=false; solver_=nothing,
        abs_tol_=1e-8, rel_tol_=1e-8, adaptive_=false, silent_=false)

Compute the loss for an ODE forward simulation using a shooting windows approach, where the
data is provided as a matrix. In the data matrix `data_`, the first column contains time 
stamps and the remaining columns contain state variables. The function computes the discrepancy 
between the simulated solution and the provided data over each window, plus an optional 
regularization term.

# Parameters
- `x0_`: Initial state vector.
- `unstable_param`: Collection to which unstable parameters are pushed if the solution 
  blows up or produces NaNs.
- `p_`: Parameter vector for the ODE.
- `odefun_`: ODE function with signature `(du, u, p, t) -> ...`.
- `data_`: Data matrix of size (N × (1+d)), where N is the number of data points. The first 
  column holds time stamps and the remaining d columns hold the state variables.
- `t0_`: Initial time.
- `δt_`: Base time step for integration.
- `S_`: Number of internal integration steps (used in the non-stiff branch).
- `γ_`: Regularization weight for the smooth L1 penalty (default: 0.0).
- `window_size_`: Number of steps per shooting window (default: 1). A window comprises 
  `window_size_ + 1` data points.
- `stiff_solver_`: Boolean flag to choose the stiff solver branch (default: false).

# Keyword Arguments
- `solver_`: (Optional) ODE solver to use in the stiff solver branch. Defaults to `Tsit5()` 
  if not provided.
- `abs_tol_`: Absolute tolerance for the ODE solver (default: 1e-8).
- `rel_tol_`: Relative tolerance for the ODE solver (default: 1e-8).
- `adaptive_`: Boolean flag for using an adaptive time-stepping scheme (default: false).
- `silent_`: Boolean flag to suppress logging messages (default: false).

# Returns
- The computed loss, which is the sum of the data misfit (squared error between the simulated 
  solution and the corresponding rows of the data matrix) and the smooth L1 regularization term. 
  The loss is normalized appropriately. If the simulation becomes unstable (blows up or produces 
  NaN values), the function returns `1e3`.

# Details
The function operates in two modes:
1. **Stiff Solver Branch (`stiff_solver_ == true`):**
   - For each shooting window, the ODE problem is remade with the initial condition taken 
     from the corresponding row of `state_data` (i.e., `D_[:, 2:end]`).
   - The ODE is solved over the window using the specified stiff solver.
   - The solution is collected into a matrix (`sol_mat`), and the squared error is computed 
     against the corresponding segment of `state_data`.

2. **Non-Stiff Branch (`stiff_solver_ == false`):**
   - A loop-based integration is performed, resetting the initial condition at the beginning 
     of each window from the data.
   - The integration is performed using a fixed number of internal steps (`S_`), and the loss 
     is accumulated at each data point.

Ensure that the data matrix `D_` is structured correctly, with the first column containing 
time values and subsequent columns representing the state variables.
"""


function forward_simulation_loss_windows(x0_, unstable_param, p_, odefun_, data_, t0_, δt_, S_, 
    γ_ = 0.0, window_size_=1, stiff_solver_=false; solver_=nothing,
    abs_tol_=abstol=1e-8, rel_tol_=abstol=1e-8, adaptive_=false, silent_=false,
    track_stability=false)

     # Extract time and state data.
    time_data = @view data_[:, 1]              # time stamps
    state_data = @view data_[:, 2:end]          # each row is one data point; columns are state variables
    N = size(state_data, 1)                   # number of data points

    #window_size is k, k+1 data points are used to form a window
    #IF window size is 1, then a window is formed between every consecutive data points.
    # If window size is length(data)-1, the forward_simulation_loss function. 
    #Defaul behaviour is to take the window size as 1
    if window_size_ < 1 || window_size_ > N - 1
        @warn "Window size out of bounds. Clamping to valid range [1, $(length(state_data) - 1)]."
        window_size_ = clamp(window_size_, 1, N - 1)
    end

    if stiff_solver_
        # -----------------------------------------------------
        # STIFF SOLVER BRANCH
        # -----------------------------------------------------
        x = deepcopy(x0_)
        solver_ = solver_ === nothing ? Tsit5() : solver_
        # Create an initial ODE problem (we will remake it in the loop)
        prob = ODEProblem(odefun_, x, (t0_, t0_+δt_),p_)
        t_init = t0_
        t_final = t0_
        Δt = δt_*S_
        data_loss = 0.0
        #looping over shooting windows
        for shoot_ind in 1:window_size_:N-1
            t_init = t_final
            t_final = clamp(t_init + window_size_*Δt, t0_, t0_+Δt*(N - 1))
            # Use the state at the shooting node as initial condition.
            # copyto!(x, [D_1[shoot_ind], D_2[shoot_ind]])
            copyto!(x, state_data[shoot_ind, :])
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

            # Define the indices in the data for this shooting window.
            window_inds = shoot_ind:clamp(shoot_ind + window_size_, 1, N)
            # Build a solution matrix: each row corresponds to a saved time point.
            sol_mat = hcat(sol.u...)'

            # # Check for blow-ups or NaNs in the solution
            # Check for mismatches in the number of time points or if the solution blows up.
            if length(window_inds) != size(sol_mat, 1) || any(sol_mat .> 1e3) || any(isnan.(sol_mat))
                if !silent_
                    @info "Solution blew up or broke at shooting node: $shoot_ind"
                end
                if track_stability
                    # Store the unstable parameters for later analysis.
                    push!(unstable_param, vcat(x, p_))
                end
                return 1e3
            end
            # data_loss += dot(D_1[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1))]- [getindex(u_ind,1) for u_ind in sol.u], 
            #                D_1[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_1))]- [getindex(u_ind,1) for u_ind in sol.u]) 
            # + dot(D_2[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_2))]- [getindex(u_ind,2) for u_ind in sol.u], 
            #                D_2[shoot_ind:clamp(shoot_ind+window_size_,1,length(D_2))]- [getindex(u_ind,2) for u_ind in sol.u])
                    
             # Accumulate the squared error over the window.
             data_loss += norm(state_data[window_inds, :] - sol_mat)^2

            #optionally setting the initial condition for the next window to the final condition of the previous window
            # x = sol.u[end]
        end

        # Add smooth L1 regularization if desired
        sparse_loss = sum(γ_ * smoothl1(pi) for pi in p_)

        # Normalize by number of data points (optional) and return
        return data_loss/N + sparse_loss/length(p_)

    else
        # -----------------------------------------------------
        # NON-STIFF BRANCH (YOUR ORIGINAL LOOP-BASED APPROACH)
        # -----------------------------------------------------
        # Integration cache
        cache = Tsit5Cache(x0_)
        x = cache.ycur
        copyto!(x, x0_)
        x_temp = deepcopy(x0_)
        weight_vector = ones(N)

        # Initial condition loss
        # data_loss = weight_vector[1]*(abs2(x0_[1] - D_1[1]) + abs2(x0_[2] - D_2[1]))
        data_loss = weight_vector[1] * norm(x0_ - state_data[1, :])^2
        integration_method = integration_step!
        # Loop over the data points
        for i = 1:(N-1)
            #Resetting the initial condition for each window using data
            if (i-1) % window_size_ == 0
                # copyto!(x, [D_1[i], D_2[i]])
                copyto!(x, state_data[i, :])
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
                if track_stability
                    # Store the unstable parameters for later analysis.
                    push!(unstable_param, vcat(x_temp, p_))
                end
                return 1e3
            end
            # Data loss for current time point
            # data_loss += weight_vector[i+1]*(abs2(x[1] - D_1[i+1]) + abs2(x[2] - D_2[i+1]))
            # Accumulate loss for the (i+1)th data point.
            data_loss += weight_vector[i + 1] * norm(x - state_data[i + 1, :])^2
        end

        # Sparsity term
        sparse_loss = sum(γ_ * smoothl1(pi) for pi in p_)

        return data_loss/N + sparse_loss/length(p_)
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
dim = 2
t0 = 0.0
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

multi_index_set = multiindices(2, 3) #Multindex for identifying degree of state variables in monomials for dim=2 upto degree 3
odefun_generic = (du,u,p,t) -> odefun_poly!(du,u,p,t; indices=multi_index_set, deg=3)

function fs_loss(x)
    x0 = @view x[1:dim]
    p  = @view x[dim+1:end]
    # Note: t0, dt, S, and γ2 are as defined in your code.
    return forward_simulation_loss(x0, p, odefun_generic, data, t0, δt, S, γ2, false)
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



x0 = [data[1, 2:end]; 0.01*randn(length(fhn_p))]
x0_ = [data[1, 2:end]; 0.01*randn(length(fhn_p))]
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
            x0 = @view x[1:dim]
            p  = @view x[dim+1:end]
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
trmstr = ["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"]
cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]



#Plotting solutions
guess_prop_minimizer = df.Minimizer[argmin(df.Cost)]
FS_minimizer = optres2.minimizer
# δt=0.01
fig = Figure(size = (980, 480))
plotSolution!(fig, guess_prop_minimizer, FS_minimizer, ts, data, tsall, hcat(tsall, alldata), δt, t0, fhn_p, Np, odefun, cache, trm;
    DA=false, GFreeFS=true, FS=true, integration_method = nothing)
fig
# save("figs/window_size_plots/init_seed_0.1_with_best_guess_propogation.png", fig)


plot_extra = false
if plot_extra
    #Animating expanding window solutions
    fig = Figure(size = (480, 480))
    save_solution_animation!(fig, results_inner, ts, data, tsall, alldata, δt, t0, fhn_p, Np, odefun, cache;
        path_="figs/window_size_plots/best_guess_evolve.mp4", integration_method = nothing)       





    #Saving cost_value vs window_size
    fig = Figure()
    save_cost_vs_windows!(fig, results_inner; title_= "best_guess_propogation")
    save("figs/window_size_plots/cost_vs_windows_with_best_guess_prop.png", fig)
end






