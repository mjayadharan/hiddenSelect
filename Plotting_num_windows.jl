        #===================================================================================================
        # Saving animation
        ====================================================================================================#
        fig = Figure(size = (480, 480))
        T = ts[end] 
        N = Int(T / δt)
        ## Plot the results 
        trmstr = repeat(["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"], 2)
        cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]
                # scatter1
        _, ys = integrate(odefun, results_inner[1][2][1:2], results_inner[1][2][3:end], t0, N, δt, cache)

        # Create axes and plots, storing objects for updates
        ax11 = Axis(fig[1, 1])
        scatter1 = scatter!(ax11, ts, data[1:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
        lines1 = lines!(ax11, 0:δt:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
        linesdash1 = lines!(ax11, tsall, alldata[1:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax11, -3, 3)
        xlims!(ax11, 0.0, ts[end])
        hidexdecorations!(ax11, ticks=false)

        ax21 = Axis(fig[2, 1])
        scatter2 = scatter!(ax21, ts, data[2:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
        lines2 = lines!(ax21, 0:δt:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
        linesdash2 = lines!(ax21, tsall, alldata[2:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax21, -0.6, 1.65)
        xlims!(ax21, 0.0, ts[end])
        ax21.xlabel = "t"
        ax21.ylabel = "w(t)"
        ax11.ylabel = "v(t)"
        ax21.xticks = 0:25:75

        ax31 = Axis(fig[3, 1])
        bar_heights = results_inner[1][2][3:end]
        bar1 = barplot!(ax31, 1:Np, bar_heights, color = Makie.RGBA.(cvec, 0.25), strokecolor=plotGray, strokewidth=1)
        scatter3 = scatter!(ax31, 1:Np, fhn_p, color=cvec)
        ax31.xticks = (1:Np, trmstr)
        ax31.xticklabelrotation = π/4
        hlines!(ax31, 0.0, color=plotGray)
        ylims!(ax31, -1.1, 1.1)
        xlims!(ax31, 0.25, Np+0.75)
        ax31.ylabel = "Parameter value"
        ax11.title = "GradFree opt; guess_propogation; #windows: $length(results_inner)"
        ax11.titlefont = "Helvetica Neue Light"
        # Plotting the GradFree simulation results
    

        # Record animation
        record(fig, "./figs/animation.mp4", length(results_inner):-1:1; framerate=10) do num_wind
            empty!(fig)
            _, ys = integrate(odefun, results_inner[num_wind][2][1:2], results_inner[num_wind][2][3:end], t0, N, δt, cache)
            # Create axes and plots, storing objects for updates
            ax11 = Axis(fig[1, 1])
            scatter1 = scatter!(ax11, ts, data[1:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
            lines1 = lines!(ax11, 0:δt:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
            linesdash1 = lines!(ax11, tsall, alldata[1:2:end], color = plotGray, linestyle = :dash)
            ylims!(ax11, -3, 3)
            xlims!(ax11, 0.0, ts[end])
            hidexdecorations!(ax11, ticks=false)
    
            ax21 = Axis(fig[2, 1])
            scatter2 = scatter!(ax21, ts, data[2:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
            lines2 = lines!(ax21, 0:δt:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
            linesdash2 = lines!(ax21, tsall, alldata[2:2:end], color = plotGray, linestyle = :dash)
            ylims!(ax21, -0.6, 1.65)
            xlims!(ax21, 0.0, ts[end])
            ax21.xlabel = "t"
            ax21.ylabel = "w(t)"
            ax11.ylabel = "v(t)"
            ax21.xticks = 0:25:75
    
            ax31 = Axis(fig[3, 1])
            bar_heights = results_inner[num_wind][2][3:end]
            bar1 = barplot!(ax31, 1:Np, bar_heights, color = Makie.RGBA.(cvec, 0.25), strokecolor=plotGray, strokewidth=1)
            scatter3 = scatter!(ax31, 1:Np, fhn_p, color=cvec)
            ax31.xticks = (1:Np, trmstr)
            ax31.xticklabelrotation = π/4
            hlines!(ax31, 0.0, color=plotGray)
            ylims!(ax31, -1.1, 1.1)
            xlims!(ax31, 0.25, Np+0.75)
            ax31.ylabel = "Parameter value"
            ax11.title = "GradFree opt; guess_propogation; #windows: $(num_wind)"
            ax11.titlefont = "Helvetica Neue Light"
            # Plotting the GradFree simulation results
            println("finished the plotting of $num_wind")
        end

        #===================================================================================================
        # End of saving animation
        ====================================================================================================#

#===================================================================================================
# Plotting the cost function value vs number of windows guess_propogation
===================================================================================================#

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, [results_inner[i][1] for i in 1:1:length(results_inner)])
ax.title = "Cost function value vs number of windows guess_propogation"
# plot([result[1] for result in results_inner])
fig
save("figs/cost_vs_windows_with_guess_prop.png", fig)
#===================================================================================================#





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
num_windows_ = 1
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
record(fig, "figs/contour_plots/S=$S.mp4", num_windows_range;framerate=1) do num_windows
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
text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="num_windows: $num_windows", fontsize=15, color=:red)


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




