include("integrator.jl")


function plotSolution!(fig, guess_prop_minimizer_, FS_minimizer_, ts_, data_, tsall_, alldata_, δt_, t0_, true_param_, Np_, odefun_, cache_;
    DA=false, GFreeFS=true, FS=true, integration_method = nothing)

    # Plotting the GradFree simulation results
    T = ts_[end] 
    N_ = Int(T / δt_)

    trmstr = repeat(["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"], 2)
    cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]
    integration_method = integration_method === nothing ? integrate : integration_method
    if GFreeFS
        _, ys = integration_method(odefun_, guess_prop_minimizer_[1:2], guess_prop_minimizer_[end-Np_ + 1:end], t0_, N_, δt_, cache_)
        ax11 = Axis(fig[1, 1])
        scatter!(ax11, ts_, data_[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
        lines!(ax11, 0:δt_:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
        lines!(ax11, tsall_, alldata_[1:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax11, -3, 3)
        xlims!(ax11, 0.0, ts_[end])
        hidexdecorations!(ax11, ticks = false)
        ax21 = Axis(fig[2, 1])
        scatter!(ax21, ts_, data_[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
        lines!(ax21, 0:δt_:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
        lines!(ax21, tsall_, alldata_[2:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax21, -0.6, 1.65)
        xlims!(ax21, 0.0, ts_[end])
        ax21.xlabel = "t"
        ax21.ylabel = "w(t)"
        ax11.ylabel = "v(t)"
        ax21.xticks = 0:25:75

        # plot the GradFree coefficient results
        ax31 = Axis(fig[3, 1])
        barplot!(ax31, 1:Np_, guess_prop_minimizer_[end-Np_ + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
        scatter!(ax31, 1:Np_, true_param_, color = cvec)
        ax31.xticks = (1:Np_, trmstr)
        ax31.xticklabelrotation = π/4
        hlines!(ax31, 0.0, color = plotGray)
        ylims!(ax31, -1.1, 1.1)
        xlims!(ax31, 0.25, Np_ + 0.75)
    end
    if DA
        # Plot the DA simulation results
        T = ts_[end] 
        N_ = Int(T / δt_)
        _, ys = integration_method(odefun_, optres.minimizer[1:2], optres.minimizer[end-Np_ + 1:end], t0_, N_, δt_, cache_)
        ax11 = Axis(fig[1, 1])
        scatter!(ax11, ts_, data_[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
        lines!(ax11, 0:δt_:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
        lines!(ax11, tsall_, alldata_[1:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax11, -3, 3)
        xlims!(ax11, 0.0, ts_[end])
        hidexdecorations!(ax11, ticks = false)
        ax21 = Axis(fig[2, 1])
        scatter!(ax21, ts_, data_[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
        lines!(ax21, 0:δt_:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
        lines!(ax21, tsall_, alldata_[2:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax21, -0.6, 1.65)
        xlims!(ax21, 0.0, ts_[end])
        ax21.xlabel = "t"
        ax21.ylabel = "w(t)"
        ax11.ylabel = "v(t)"
        ax21.xticks = 0:25:75

        # plot the DA coefficient results
        ax31 = Axis(fig[3, 1])
        barplot!(ax31, 1:Np_, optres.minimizer[end-Np_ + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
        scatter!(ax31, 1:Np_, true_param_, color = cvec)
        ax31.xticks = (1:Np_, trmstr)
        ax31.xticklabelrotation = π/4
        hlines!(ax31, 0.0, color = plotGray)
        ylims!(ax31, -1.1, 1.1)
        xlims!(ax31, 0.25, Np_ + 0.75)
    end

    if FS

        # Plot the FS simulation results
        T = ts_[end] 
        N_ = Int(T / δt_)
        _, ys = integration_method(odefun_, FS_minimizer_[1:2], FS_minimizer_[end-Np_ + 1:end], t0_, N_, δt_, cache_)
        ax12 = Axis(fig[1, 2])
        scatter!(ax12, ts_, data_[1:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
        lines!(ax12, 0:δt_:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
        lines!(ax12, tsall_, alldata_[1:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax12, -3, 3)
        xlims!(ax12, 0.0, ts_[end])
        hidexdecorations!(ax12, ticks = false)
        ax22 = Axis(fig[2, 2])
        scatter!(ax22, ts_, data_[2:2:end], color = Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
        lines!(ax22, 0:δt_:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
        lines!(ax22, tsall_, alldata_[2:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax22, -0.6, 1.65)
        xlims!(ax22, 0.0, ts_[end])
        hideydecorations!.([ax12, ax22], ticks = false)
        ax22.xlabel = "t"
        ax22.xticks = 0:25:75

        # plot the FS coefficient results
        ax32 = Axis(fig[3, 2])
        barplot!(ax32, 1:Np_, FS_minimizer_[end-Np_ + 1:end], color = Makie.Colors.RGBA.(cvec, 0.25), strokecolor = plotGray, strokewidth = 1)
        scatter!(ax32, 1:Np_, true_param_, color = cvec)
        ax32.xticks = (1:Np_, trmstr)
        ax32.xticklabelrotation = π/4
        hlines!(ax32, 0.0, color = plotGray)
        ylims!(ax32, -1.1, 1.1)
        xlims!(ax32, 0.25, Np_ + 0.75)
        hideydecorations!(ax32, ticks = false)
        ax31.ylabel = "Parameter value"

        ax11.title = "GradFree multi-shoot optimization, with guess propogation"

        ax12.title = "Forward Simulation optimization"
        ax11.titlefont = "Helvetica Neue Light"
        ax12.titlefont = "Helvetica Neue Light"
    end
end

#===================================================================================================
===================================================================================================#



       
function save_solution_animation!(fig, result_dict_, ts_, data_, tsall_, alldata_, δt_, t0_, true_param_, Np_, odefun_, cache_;
    path_="./figs/animation.mp4", title_="window_size", integration_method = nothing)       
    #===================================================================================================
    # Saving animation
    ====================================================================================================#
    T = ts_[end] 
    N = Int(T / δt_)
    integration_method = integration_method === nothing ? integrate : integration_method

    ## Plot the results 
    trmstr = repeat(["1", "v", "w", "v²", "vw", "w²", "v³", "v²w", "vw²", "w³"], 2)
    cvec = [repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[12], 1.0), ], 10); repeat([Makie.Colors.RGBA(Makie.ColorSchemes.Signac[11], 1.0), ], 10)]
            # scatter1
    _, ys = integration_method(odefun_, result_dict_[1][2][1:2], result_dict_[1][2][end-Np_ + 1:end], t0_, N, δt_, cache_)

    # Create axes and plots, storing objects for updates
    ax11 = Axis(fig[1, 1])
    scatter1 = scatter!(ax11, ts_, data_[1:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
    lines1 = lines!(ax11, 0:δt_:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
    linesdash1 = lines!(ax11, tsall_, alldata_[1:2:end], color = plotGray, linestyle = :dash)
    ylims!(ax11, -3, 3)
    xlims!(ax11, 0.0, ts_[end])
    hidexdecorations!(ax11, ticks=false)

    ax21 = Axis(fig[2, 1])
    scatter2 = scatter!(ax21, ts_, data_[2:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
    lines2 = lines!(ax21, 0:δt_:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
    linesdash2 = lines!(ax21, tsall_, alldata_[2:2:end], color = plotGray, linestyle = :dash)
    ylims!(ax21, -0.6, 1.65)
    xlims!(ax21, 0.0, ts_[end])
    ax21.xlabel = "t"
    ax21.ylabel = "w(t)"
    ax11.ylabel = "v(t)"
    ax21.xticks = 0:25:75

    ax31 = Axis(fig[3, 1])
    bar_heights = result_dict_[1][2][end-Np_ + 1:end]
    bar1 = barplot!(ax31, 1:Np, bar_heights, color = Makie.RGBA.(cvec, 0.25), strokecolor=plotGray, strokewidth=1)
    scatter3 = scatter!(ax31, 1:Np, true_param_, color=cvec)
    ax31.xticks = (1:Np, trmstr)
    ax31.xticklabelrotation = π/4
    hlines!(ax31, 0.0, color=plotGray)
    ylims!(ax31, -1.1, 1.1)
    xlims!(ax31, 0.25, Np+0.75)
    ax31.ylabel = "Parameter value"
    ax11.title = "GradFree opt; guess_propogation; $title_: 1"
    ax11.titlefont = "Helvetica Neue Light"
    # Plotting the GradFree simulation results

    window_size_range = sort(collect(keys(result_dict_)))

    # Record animation
    record(fig, path_, window_size_range; framerate=1) do window_size_
        empty!(fig)
        _, ys = integration_method(odefun_, result_dict_[window_size_][2][1:2], result_dict_[window_size_][2][end-Np_ + 1:end], t0_, N, δt_, cache_)
        # Create axes and plots, storing objects for updates
        ax11 = Axis(fig[1, 1])
        scatter1 = scatter!(ax11, ts_, data_[1:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[12], 0.25))
        lines1 = lines!(ax11, 0:δt_:T, getindex.(ys, 1), color = Makie.ColorSchemes.Signac[12])
        linesdash1 = lines!(ax11, tsall_, alldata_[1:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax11, -3, 3)
        xlims!(ax11, 0.0, ts_[end])
        hidexdecorations!(ax11, ticks=false)

        ax21 = Axis(fig[2, 1])
        scatter2 = scatter!(ax21, ts_, data_[2:2:end], color = Makie.RGBA(Makie.ColorSchemes.Signac[11], 0.25))
        lines2 = lines!(ax21, 0:δt_:T, getindex.(ys, 2), color = Makie.ColorSchemes.Signac[11])
        linesdash2 = lines!(ax21, tsall_, alldata_[2:2:end], color = plotGray, linestyle = :dash)
        ylims!(ax21, -0.6, 1.65)
        xlims!(ax21, 0.0, ts_[end])
        ax21.xlabel = "t"
        ax21.ylabel = "w(t)"
        ax11.ylabel = "v(t)"
        ax21.xticks = 0:25:75

        ax31 = Axis(fig[3, 1])
        bar_heights = result_dict_[window_size_][2][end-Np_ + 1:end]
        bar1 = barplot!(ax31, 1:Np, bar_heights, color = Makie.RGBA.(cvec, 0.25), strokecolor=plotGray, strokewidth=1)
        scatter3 = scatter!(ax31, 1:Np, true_param_, color=cvec)
        ax31.xticks = (1:Np, trmstr)
        ax31.xticklabelrotation = π/4
        hlines!(ax31, 0.0, color=plotGray)
        ylims!(ax31, -1.1, 1.1)
        xlims!(ax31, 0.25, Np+0.75)
        ax31.ylabel = "Parameter value"
        ax11.title = "GradFree opt; guess_propogation; $title_: $window_size_"
        ax11.titlefont = "Helvetica Neue Light"
        # Plotting the GradFree simulation results
        println("finished the plotting of $window_size_")
    end
end

#===================================================================================================
# End of saving animation
====================================================================================================#

#===================================================================================================
# Plotting the cost function value vs number of windows guess_propogation
===================================================================================================#
function save_cost_vs_windows!(fig, results_dict_; title_="guess_propogation")
    ax = Axis(fig[1, 1])
    result_keys = collect(keys(results_dict_))
    scatter!(ax, result_keys, map(r -> results_dict_[r][1],result_keys))
    ax.title = "Cost function value vs number of windows $title_"
end
#===================================================================================================#







#===================================================================================================
# Cost function visualization============================================================================================
===================================================================================================#





#================================================================================
# Plotting the contour plots of the loss function
================================================================================#
function plot_contours!(fig, loss_function_, center_param_; ref_point_=nothing, delta_range_=1, n_grids_,
    index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], figure_title_="Loss function landscape")


    ref_point_ = ref_point_ === nothing ? center_param_ : ref_point_
    # Grid size and variation range

    # Create a grid layout for proper spacing
    grid = fig[1, 1] = GridLayout()
    # Label(fig[0, 1], figure_title, fontsize=24, tellheight=false) 

    # Store subplot axes and colorbars separately
    axs = Matrix{Axis}(undef, 2, 2)
    cbs = Matrix{Colorbar}(undef, 2, 2)  # Store separate colorbars for each plot

    # Iterate over the four parameter pairs
    for (idx, (p1_index, p2_index)) in enumerate(index_pairs_)
        row, col = Tuple(CartesianIndices((2, 2))[idx])  # Convert index to 2x2 grid positions

        # Create subplot axis
        ax = Axis(grid[row, 2col-1],  # Place axes in odd columns
            xlabel = "p$(p1_index)",
            ylabel = "p$(p2_index)",
            title  = "loss_function (p$(p1_index), p$(p2_index)): $figure_title_")

        axs[row, col] = ax  # Store axis reference

        # Extract central values for the parameters
        p1_center = center_param_[p1_index]
        p2_center = center_param_[p2_index]

        # Define parameter ranges
        p1_values = LinRange(p1_center - delta_range_, p1_center + delta_range_, n_grids_)
        p2_values = LinRange(p2_center - delta_range_, p2_center + delta_range_, n_grids_)

        # Preallocate a matrix to hold cost values
        costvals = Matrix{Float64}(undef, n_grids_, n_grids_)

        # Evaluate loss_function at each (p1, p2) pair
        for i in 1:n_grids_
            for j in 1:n_grids_
                # Copy the best-fit parameter vector, then replace p1 and p2
                p_trial = copy(center_param_)
                p_trial[p1_index] = p1_values[i]
                p_trial[p2_index] = p2_values[j]

                # Evaluate the loss function
                costvals[i, j] = loss_function_(p_trial)
            end
        end
        # Apply clamping if limits are set
        costvals .= clamp.(costvals, -10, 100)
        # Plot filled contour
        hm = contourf!(ax, p1_values, p2_values, costvals; levels=40, colormap=:viridis)

        # Highlight the optimal value with a **solid red circle**
        scatter!(ax, [center_param_[p1_index]], [center_param_[p2_index]], 
                color=:red, markersize=12, strokewidth=3, strokecolor=:black)

        # Highlight the ref param value with a **solid orange circle**
        scatter!(ax, [ref_point_[p1_index]], [ref_point_[p2_index]], 
                color=:orange, markersize=20, strokewidth=3, strokecolor=:black)
        # text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="window_size: $window_size_", fontsize=15, color=:red)
        

        # Define unique label for each colorbar
        cb_label = "loss function (p$(p1_index), p$(p2_index))"

        try
            # Place colorbars **in separate adjacent columns** (even-numbered columns)
            cost_extrema = extrema(costvals)
            if cost_extrema[1] == cost_extrema[2]
                cost_extrema = (cost_extrema[1] - 1, cost_extrema[2] + 1)
            end
            cbs[row, col] = Colorbar(grid[row, 2col]; colorrange=cost_extrema, label=cb_label)
        catch e
            println("Error occurred while creating colorbar: ", e)
            println("--------cost_function----valeus:$(cost_extrema))") 
        end
    end

    # return fig
end

# #================================================================================#
# S=20
# fig = Figure(size = (1800, 1800))  # Define the figure once
# window_size_ = 1
# param_center = [data[1:2]; fhn_p]
# function fs_loss_window(x)
# x0 = view(x, 1:2)
# p  = view(x, 3:length(x))
# return forward_simulation_loss_windows(x0, p, odefun, data, 0.0, δt, S, γ2, window_size_, false)
# end

# plot_contours!(fig, fs_loss_window, param_center;
# ref_point=results_inner[window_size_][2], delta_range_=1, n_grids=50,
# index_pairs=[(3, 4), (5, 6), (7, 8), (10, 15)],
# figure_title="window_size: $window_size_"
# )
# fig
# save("figs/contour_plots/S=$(S)_window_size=$window_size_.png", fig)

# #================================================================================#






#===============================================================================
# Animating contour plots of the loss function
===============================================================================#

function animate_contourplots!(fig, true_params_, data_, δt_, γ2_, odefun_; loss_function_ = nothing, loss_function_window_=nothing, delta_range_=1, n_grids_=50,
    index_pairs_=[(3, 4), (5, 6), (7, 8), (9, 10)], result_dict_, implicit_scheme_=false, figure_title="Cost Landscape", S_=10,
    path_="./figs/window_size_plots/contour_plots.mp4")


    window_size_range = sort(collect(keys(result_dict_)))

    center_param_ = [data_[1:2]; true_params_]

    # Start recording
    record(fig, path_, window_size_range;framerate=1) do window_size_
        empty!(fig)  # Clear the figure before each frame

        function fs_loss_window(x)
            x0 = view(x, 1:2)
            p  = view(x, 3:length(x))
            return loss_function_window_(x0, p, odefun_, data_, 0.0, δt_, S_, γ2_, window_size_, implicit_scheme_; silent=true)
        end

        loss_function_ = loss_function_ === nothing ? fs_loss_window : loss_function_
        ref_point = result_dict_[window_size_][2] 



        plot_contours!(fig, loss_function_, center_param_; ref_point_=ref_point, delta_range_=delta_range_, n_grids_=n_grids_,
            index_pairs_=index_pairs_, figure_title_="Loss function landscape")

        # text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="window_size: $window_size_", fontsize=15, color=:red)


        println("done with window_size = $window_size_")
    end
end


# num_windows_range = vcat(length(results_inner):-10:10,10:-1:1)






#=================================================================================
=================================================================================#





#=================================================================================
# Plotting the cost surface 
=================================================================================#
function plot_cost_surface!(fig, true_params_, data_, δt_, γ2_, odefun_; loss_function_ = 1, delta_range_=1, n_grids_=100,
    index_pair_=(3, 4), z_clamps_= (0,10), implicit_scheme_=false, figure_title="Cost Landscape", S_=10, window_size_=1)
    
    # function fs_loss_window(x)
    #     x0 = view(x, 1:2)
    #     p  = view(x, 3:length(x))
    #     return forward_simulation_loss_windows(x0, p, odefun_, data_, 0.0, δt_, S_, γ2_, window_size_, implicit_scheme_)
    # end
    # loss_function_ = loss_function_ === nothing ? fs_loss_window : loss_function_

    p1_index, p2_index = index_pair_

    center_param_ = [data_[1:2]; true_params_]
    ax = LScene(fig[1, 1])  # Create a 3D scene

    # Define parameter center values
    p1_center = center_param_[p1_index]
    p2_center = center_param_[p2_index]

    # Create parameter ranges
    p1_values = LinRange(p1_center - delta_range_, p1_center + delta_range_, n_grids_)
    p2_values = LinRange(p2_center - delta_range_, p2_center + delta_range_, n_grids_)
    # Compute loss function values
    Z = [loss_function_(vcat(center_param_[1:p1_index-1], [p1], 
                center_param_[p1_index+1:p2_index-1], [p2], 
                center_param_[p2_index+1:end])) for p1 in p1_values, p2 in p2_values]

    # Find the min and max of Z for scaling
    # z_min, z_max = extrema(Z)
    Z_clipped = clamp.(Z, z_clamps_[1], z_clamps_[2])
    # Plot surface and store the plot object
    surface!(ax, p1_values, p2_values, Z_clipped; colormap=:viridis)
    text!(ax.scene, Point3f(0.0, 0.0, 0.5), text="window_size: $window_size_", fontsize=15, color=:red)

    # Reduce the title height so it doesn't push the plot down too much
    # Label(fig[0, 1], "Cost Landscape", fontsize=30, tellheight=false)  


    # Reduce the height of the title row to bring it closer to the plot
    # rowsize!(fig.layout, 0, Auto(0.1))  # Adjust row heig

    # Adjust the camera/view angle
    # cam3d!(ax.scene, azimuth=100, elevation=100)  # Adjust azimuth & elevation angles

    # save("figs/surface_plots/S=$(S)_window_size=$window_size_.png", fig)

end 

#=================================================================================
=================================================================================#





#=================================================================================
# Animating the cost surface
=================================================================================#
function animate_surface_plots!(fig, true_params_, data_, δt_, γ2_, odefun_; loss_function_ = nothing, delta_range_=1, n_grids_=100,
    index_pair_=(3, 4), z_clamps_= (0,10), result_dict_, implicit_scheme_=false, figure_title="Cost Landscape", S_=10)
    

    p1_index, p2_index = index_pair_
    center_param_ = [data_[1:2]; true_params_]
    # num_windows_range = vcat(length(results_inner):-5:5,4:-1:1)

    window_size_range = sort(collect(keys(result_dict_)))

    # Start recording
    record(fig, "figs/surface_plots/S=$S_.mp4", window_size_range;framerate=1) do window_size_
        empty!(fig)  # Clear the figure before each frame

        function fs_loss_window(x)
            x0 = view(x, 1:2)
            p  = view(x, 3:length(x))
            return forward_simulation_loss_windows(x0, p, odefun_, data_, 0.0, δt_, S_, γ2_, window_size_, implicit_scheme_)
        end
        loss_function = loss_function_ === nothing ? fs_loss_window : loss_function_

        ax = LScene(fig[1, 1])  # Create a 3D scene

        # Define parameter center values
        p1_center = center_param_[p1_index]
        p2_center = center_param_[p2_index]

        # Create parameter ranges
        p1_values = LinRange(p1_center - delta_range_, p1_center + delta_range_, n_grids_)
        p2_values = LinRange(p2_center - delta_range_, p2_center + delta_range_, n_grids_)

        # Compute loss function values
        Z = [loss_function(vcat(center_param_[1:p1_index-1], [p1], 
                        center_param_[p1_index+1:p2_index-1], [p2], 
                        center_param_[p2_index+1:end])) for p1 in p1_values, p2 in p2_values]

        # Find the min and max of Z for scaling
        # z_min, z_max = extrema(Z)
        Z_clipped = clamp.(Z, z_clamps_[1], z_clamps_[2])

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
end


#=================================================================================
=================================================================================#