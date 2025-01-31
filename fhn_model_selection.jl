include("integrator.jl")
# Packages for plotting and default theming
using CairoMakie
using LinearAlgebra, Statistics
using ReverseDiff, Optim
using Random
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

"""
    forward_simulation_loss(x0, p, odefun D, t0, δt, S, γ = 0.0)

Loss function for forward simulation from x0 using parameters p in the odefun from initial time t0 with time step δt. 
The data are given in the concatenated vector data and the data points are every S timesteps. 

Optionally add a smooth L1 regularization term with weight γ.

TODO: At this point assumes the state is two dimension!!!
"""
function forward_simulation_loss(x0, p, odefun, D, t0, δt, S, γ = 0.0)
    # Integration cache
    cache = Tsit5Cache(x0)
    # Current state
    x = cache.ycur
    copyto!(x, x0)

    # Initial condition loss
    data_loss = abs2(x0[1] - data[1]) + abs2(x0[2] - data[2])

    # Loop over all data points
    for i = 1:div(length(D), 2) - 1
        # Internal time steps
        for j = 1:S
            integration_step!(cache, odefun, x, t0, p, δt, false)
        end

        # Detect if we are going unstable
        any(abs.(x) .> 1e3) && (println("Breaking early: $(i), $(data_loss)"); data_loss += 1e5; return data_loss)#(println("Breaking early: $(data_loss)"); data_loss += 1e8; return data_loss)
        any(isnan.(x)) && (println("Breaking early: $(i), $(data_loss)"); data_loss += 1e5; return data_loss) #(println("Breaking early: $(data_loss)"); data_loss += 1e8; return data_loss)

        # Data loss for current time point
        data_loss += abs2(x[1] - data[1 + 2i]) + abs2(x[2] - data[2 + 2i])
    end

    # Sparsity term 
    sparse_loss = 0.0
    for p_i in p
        sparse_loss += γ*smoothl1(p_i)
    end
    return data_loss / length(D) + sparse_loss / length(p)
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
S = 20          # Number of internal time steps
δt = Δt / S     # Time step 
α = 1.0         # Data loss weight 
β = 100.0       # Model loss weight 
γ1 = 5e-3        # Sparsity weight 
γ2 = 5e-2        # Sparsity weight
# Length of the DA state vector
Nx = 2*(div(length(data), 2) - 1) * S + 2 

# Two term loss functions 
da_loss(X, p) = data_assimilation_loss(X, p, odefun, data, α, β, δt, S, γ1)
fs_loss(x0, p) = forward_simulation_loss(x0, p, odefun, data, 0.0, δt, S, γ2)
# Compile the reverse differentiation tapes
fs_loss_tape = ReverseDiff.GradientTape(fs_loss, (randn(2), similar(fhn_p)))
compiled_fs_loss_tape = ReverseDiff.compile(fs_loss_tape)
da_loss_tape = ReverseDiff.GradientTape(da_loss, (randn(Nx), similar(fhn_p)))
compiled_da_loss_tape = ReverseDiff.compile(da_loss_tape)

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
da_loss(x) = da_loss(view(x, 1:length(x) - Np), view(x, length(x) - Np + 1:length(x)))
grad_da_loss!(g, x) = ReverseDiff.gradient!((view(g, 1:length(g) - Np), view(g, length(g) - Np + 1:length(g))), compiled_da_loss_tape, (view(x, 1:length(x) - Np), view(x, length(x)- Np + 1:length(x))))
fs_loss(x) = fs_loss(view(x, 1:2), view(x, 3:length(x)))
grad_fs_loss!(g, x) = ReverseDiff.gradient!((view(g, 1:2), view(g, 3:length(g))), compiled_fs_loss_tape, (x[1:2], x[3:length(x)]))

# # Optimizations 
# Data assimilation optimization 
options = Optim.Options(show_trace = true, iterations = 5000, show_every = 10)
# x0 = [zeros(Nx); 0.01*zeros(length(fhn_p))]
x0 = [zeros(Nx); 0.1*randn(length(fhn_p))]

optres = Optim.optimize(da_loss, grad_da_loss!, x0, BFGS(), options)

##
# Forward simulation optimization
options = Optim.Options(show_trace = true, iterations = 2500, show_every = 1)
# Use parameters from DA optimization for the initial conditions of the FS optimization
x0 = [data[1:2]; optres.minimizer[end - Np + 1:end]]
# x0 = [data[1:2]; 0.01*zeros(length(fhn_p))]
# x0 = [data[1:2]; fhn_p + 0.05*randn(length(fhn_p))]


optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0, BFGS(), options)
# optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0, NelderMead(), options)
# x0_2 = [data[1:2]; optres2.minimizer[end-Np + 1:end]]

# optres2 = Optim.optimize(fs_loss, grad_fs_loss!, x0_2, BFGS(), options)


## Plot the results 
trmstr = repeat(["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"], 2)
cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]

fig = Figure(size = (980, 480))
# Plot the DA simulation results
T = ts[end] 
N = Int(T / δt)
_, ys = integrate(odefun, optres.minimizer[1:2], optres.minimizer[end-Np + 1:end], t0, N, δt, cache)
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

# plot the DA coefficient results
ax31 = Axis(fig[3, 1])
barplot!(ax31, 1:Np, optres.minimizer[end-Np + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
scatter!(ax31, 1:Np, fhn_p, color = cvec)
ax31.xticks = (1:Np, trmstr)
ax31.xticklabelrotation = π/4
hlines!(ax31, 0.0, color = plotGray)
ylims!(ax31, -1.1, 1.1)
xlims!(ax31, 0.25, Np + 0.75)

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

ax11.title = "Data Assimilation optimization"
ax12.title = "Forward Simulation optimization"
ax11.titlefont = "Helvetica Neue Light"
ax12.titlefont = "Helvetica Neue Light"
fig