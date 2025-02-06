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