## Codes to integrate ODE with fixed time step using an optimized 5th order RK scheme Tsit5
"""
    Tsit5Tableau(T) 

    Generate structure containing tableau coefficients for Tsit5 with type T
"""
struct Tsit5Tableau{T}
    c::NTuple{7, T}
    b::NTuple{7, T} 
    b̂::NTuple{7, T}

    a2::NTuple{1 , T} 
    a3::NTuple{2 , T} 
    a4::NTuple{3 , T} 
    a5::NTuple{4 , T} 
    a6::NTuple{5 , T} 
    a7::NTuple{6 , T} 
end

# Print more readably 
Base.show(io::IO, tab::Tsit5Tableau{T}) where T  = print(io, "Tsit5Tableau{$T}")
function Base.show(io::IO, ::MIME"text/plain", tab::Tsit5Tableau{T}) where T
    println(io, "Tsit5Tableau{$T}:")
    println(io, "c : $(round.(tab.c, digits = 3))")
    println(io, "b : $(round.(tab.b, digits = 3))")
    println(io, "b̂ : $(round.(tab.b̂, digits = 3))")
    println(io, "a₂: $(round.(tab.a2, digits = 3))")
    println(io, "a₃: $(round.(tab.a3, digits = 3))")
    println(io, "a₄: $(round.(tab.a4, digits = 3))")
    println(io, "a₅: $(round.(tab.a5, digits = 3))")
    println(io, "a₆: $(round.(tab.a6, digits = 3))")
    println(io, "a₇: $(round.(tab.a7, digits = 3))")
end

# Coefficients taken from Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption
# Computers and Mathematics with Applications
# Ch. Tsitouras 2011
function Tsit5Tableau(T)
    c1 =  0.0
    c2 =  0.161 
    c3 =  0.327
    c4 =  0.9 
    c5 =  0.9800255409045097
    c6 =  1.0
    c7 =  1.0 

    b1 =  0.09646076681806523
    b2 =  0.01 
    b3 =  0.4798896504144996
    b4 =  1.379008574103742 
    b5 = -3.290069515436081
    b6 =  2.324710524099774
    b7 =  0.0

    b̂1 =  0.001780011052226
    b̂2 =  0.000816434459657
    b̂3 = -0.007880878010262
    b̂4 =  0.144711007173263
    b̂5 = -0.582357165452555
    b̂6 =  0.458082105929187
    b̂7 =  0.015151515151515152 #1 / 66

    a32 =  0.3354806554923570 
    a42 = -6.359448489975075
    a52 = -11.74888356406283 
    a62 = -12.92096931784711
    a43 =  4.362295432869581
    a53 =  7.495539342889836 
    a63 =  8.159367898576159
    a54 = -0.09249506636175525
    a64 = -0.07158497328140100 
    a65 = -0.02826905039406838

    a21 = c2
    a31 = c3 - a32
    a41 = c4 - a42 - a43
    a51 = c5 - a52 - a53 - a54
    a61 = c6 - a62 - a63 - a64 - a65

    a71 = b1 
    a72 = b2
    a73 = b3 
    a74 = b4 
    a75 = b5 
    a76 = b6

    ctup = T.((c1, c2, c3, c4, c5, c6, c7))
    btup = T.((b1, b2, b3, b4, b5, b6, b7))
    b̂tup = T.((b̂1, b̂2, b̂3, b̂4, b̂5, b̂6, b̂7))
    a2tup = T.((a21, ))
    a3tup = T.((a31, a32))
    a4tup = T.((a41, a42, a43))
    a5tup = T.((a51, a52, a53, a54))
    a6tup = T.((a61, a62, a63, a64, a65))
    a7tup = T.((a71, a72, a73, a74, a75, a76))
    return Tsit5Tableau{T}(ctup, btup, b̂tup, a2tup, a3tup, a4tup, a5tup, a6tup, a7tup)
end

"""
    Tsit5Cache{V, T}

Stores temporary vectors needed for the computation of the Tsit5 step

    Tsit5Cache(y::AbstractVector{T})

Generates the cache for initial condition type of y
"""
struct Tsit5Cache{V, T}
    tableau::Tsit5Tableau{T}
    # Storage for intermediate stages
    k1::V
    k2::V
    k3::V
    k4::V
    k5::V
    k6::V
    k7::V

    tmp::V
    # Previous time step
    ycur::V
    yprev::V
    yerr::V
end
# Print more readably 
Base.show(io::IO, cache::Tsit5Cache{V, T}) where {V, T}  = print(io, "Tsit5Cache{$V, $T}")
function Tsit5Cache(y::AbstractVector{T}) where T
    tab = Tsit5Tableau(T)
    k1 = similar(y)
    k2 = similar(y)
    k3 = similar(y)
    k4 = similar(y)
    k5 = similar(y)
    k6 = similar(y)
    k7 = similar(y)
    tmp = similar(y)
    ycur = similar(y)
    yprev = similar(y)
    yerr = similar(y)
    return Tsit5Cache{typeof(k1), T}(tab, k1, k2, k3, k4, k5, k6, k7, tmp, ycur, yprev, yerr)
end

# Inplace integration step
"""
    integration_step!(cache, odefun, y, t, p, δt, calcerr = true)
Perform an integration step of length δt using the memory allocated in cache updating y inplace. 

Assmes that odefun is an inplace function with inputs odefun(dy, y, p, t) that writes the result in dy 
    y - current state
    t - current time 
    p - ODE paramters
    δt - time step

If calcerr is true then a fourth order step is also calculated to estimate the error of the step
"""
function integration_step!(cache::Tsit5Cache, odefun, y, t, p, δt, calcerr = true)
    # Access from cache
    yprev = cache.yprev
    tmp  = cache.tmp

    # Save y as the previous time point
    copyto!(yprev, y)

    # Evaluate the stages
    # k1
    odefun(cache.k1, yprev, p, t)
    # k2
    tmp .= yprev .+ δt .* cache.tableau.a2[1] .* cache.k1 
    odefun(cache.k2, tmp, p, t + cache.tableau.c[2] * δt)
    # k3
    tmp .= yprev .+ δt .* cache.tableau.a3[1] .* cache.k1 .+ δt .* cache.tableau.a3[2] .* cache.k2
    odefun(cache.k3, tmp, p, t + cache.tableau.c[3] * δt)
    # k4
    tmp .= yprev .+ δt .* cache.tableau.a4[1] .* cache.k1 .+ δt .* cache.tableau.a4[2] .* cache.k2 .+ δt .* cache.tableau.a4[3] .* cache.k3
    odefun(cache.k4, tmp, p, t + cache.tableau.c[4] * δt)
    # k5
    tmp .= yprev .+ δt .* cache.tableau.a5[1] .* cache.k1 .+ δt .* cache.tableau.a5[2] .* cache.k2 .+ δt .* cache.tableau.a5[3] .* cache.k3 .+ δt .* cache.tableau.a5[4] .* cache.k4
    odefun(cache.k5, tmp, p, t + cache.tableau.c[5] * δt)
    # k5
    tmp .= yprev .+ δt .* cache.tableau.a6[1] .* cache.k1 .+ δt .* cache.tableau.a6[2] .* cache.k2 .+ δt .* cache.tableau.a6[3] .* cache.k3 .+ δt .* cache.tableau.a6[4] .* cache.k4 .+ δt .* cache.tableau.a6[5] .* cache.k5
    odefun(cache.k6, tmp, p, t + cache.tableau.c[6] * δt)
    # k6
    tmp .= yprev .+ δt .* cache.tableau.a7[1] .* cache.k1 .+ δt .* cache.tableau.a7[2] .* cache.k2 .+ δt .* cache.tableau.a7[3] .* cache.k3 .+ δt .* cache.tableau.a7[4] .* cache.k4 .+ δt .* cache.tableau.a7[5] .* cache.k5 .+ δt .* cache.tableau.a7[6] .* cache.k6
    odefun(cache.k7, tmp, p, t + cache.tableau.c[7] * δt)

    # Update the current state overwriting y
    y .= cache.tableau.b[1] .* cache.k1 .+ cache.tableau.b[2] .* cache.k2 .+ cache.tableau.b[3] .* cache.k3 .+ cache.tableau.b[4] .* cache.k4 .+ cache.tableau.b[5] .* cache.k5 .+ cache.tableau.b[6] .* cache.k6 .+ cache.tableau.b[7] .* cache.k7
    y .*= δt
    y .+= cache.yprev

    # Error calculation if requested
    if calcerr 
        cache.yerr .= cache.tableau.b̂[1] .* cache.k1 .+ cache.tableau.b̂[2] .* cache.k2 .+ cache.tableau.b̂[3] .* cache.k3 .+ cache.tableau.b̂[4] .* cache.k4 .+ cache.tableau.b̂[5] .* cache.k5 .+ cache.tableau.b̂[6] .* cache.k6 .+ cache.tableau.b̂[7] .* cache.k7
        cache.yerr .*= δt
        cache.yerr .+= cache.yprev
    end
    return nothing
end

"""
    integration_step(cache, odefun, y, t, p, δt, calcerr = true)
Perform an integration step of length δt using the memory allocated in cache out-of-place returning a new vector y. 

Assmes that odefun is an inplace function with inputs odefun(dy, y, p, t) that writes the result in dy 
    y - current state
    t - current time 
    p - ODE paramters
    δt - time step

If calcerr is true then a fourth order step is also calculated to estimate the error of the step
"""
function integration_step(cache::Tsit5Cache, odefun, y, t, p, δt, calcerr = true)
    yprev = cache.yprev
    tmp  = cache.tmp
    # Save y as the previous time point
    copyto!(yprev, y)

    # Evaluate the stages
    # k1
    odefun(cache.k1, yprev, p, t)
    # k2
    tmp .= yprev .+ δt .* cache.tableau.a2[1] .* cache.k1 
    odefun(cache.k2, tmp, p, t + cache.tableau.c[2] * δt)
    # k3
    tmp .= yprev .+ δt .* cache.tableau.a3[1] .* cache.k1 .+ δt .* cache.tableau.a3[2] .* cache.k2
    odefun(cache.k3, tmp, p, t + cache.tableau.c[3] * δt)
    # k4
    tmp .= yprev .+ δt .* cache.tableau.a4[1] .* cache.k1 .+ δt .* cache.tableau.a4[2] .* cache.k2 .+ δt .* cache.tableau.a4[3] .* cache.k3
    odefun(cache.k4, tmp, p, t + cache.tableau.c[4] * δt)
    # k5
    tmp .= yprev .+ δt .* cache.tableau.a5[1] .* cache.k1 .+ δt .* cache.tableau.a5[2] .* cache.k2 .+ δt .* cache.tableau.a5[3] .* cache.k3 .+ δt .* cache.tableau.a5[4] .* cache.k4
    odefun(cache.k5, tmp, p, t + cache.tableau.c[5] * δt)
    # k5
    tmp .= yprev .+ δt .* cache.tableau.a6[1] .* cache.k1 .+ δt .* cache.tableau.a6[2] .* cache.k2 .+ δt .* cache.tableau.a6[3] .* cache.k3 .+ δt .* cache.tableau.a6[4] .* cache.k4 .+ δt .* cache.tableau.a6[5] .* cache.k5
    odefun(cache.k6, tmp, p, t + cache.tableau.c[6] * δt)
    # k6
    tmp .= yprev .+ δt .* cache.tableau.a7[1] .* cache.k1 .+ δt .* cache.tableau.a7[2] .* cache.k2 .+ δt .* cache.tableau.a7[3] .* cache.k3 .+ δt .* cache.tableau.a7[4] .* cache.k4 .+ δt .* cache.tableau.a7[5] .* cache.k5 .+ δt .* cache.tableau.a7[6] .* cache.k6
    odefun(cache.k7, tmp, p, t + cache.tableau.c[7] * δt)

    y = cache.tableau.b[1] .* cache.k1 .+ cache.tableau.b[2] .* cache.k2 .+ cache.tableau.b[3] .* cache.k3 .+ cache.tableau.b[4] .* cache.k4 .+ cache.tableau.b[5] .* cache.k5 .+ cache.tableau.b[6] .* cache.k6 .+ cache.tableau.b[7] .* cache.k7
    y .*= δt
    y .+= cache.yprev

    # Error calculation if requested
    if calcerr 
        cache.yerr .= cache.tableau.b̂[1] .* cache.k1 .+ cache.tableau.b̂[2] .* cache.k2 .+ cache.tableau.b̂[3] .* cache.k3 .+ cache.tableau.b̂[4] .* cache.k4 .+ cache.tableau.b̂[5] .* cache.k5 .+ cache.tableau.b̂[6] .* cache.k6 .+ cache.tableau.b̂[7] .* cache.k7
        cache.yerr .*= δt
        cache.yerr .+= cache.yprev
    end
    return y
end

"""
    integrate(odefun, y0, p, t0, N, δt, cache = Tsit5Cache(y0); calcerr = false)

Integrate the ODE defined by odefun from initial condition y0 with ODE parameters p, initial time t0
Takes N integration steps resulting in a N+1 length vector using fixed timestep δt. 

Optionally a preallocated cache can be provided. If calcerr is specified that step error calculations are performed.
"""
function integrate(odefun, y0::Vector{T}, p, t0, N, δt, cache = Tsit5Cache(y0); calcerr = false) where {T}
    # Intialize the integration
    y = cache.ycur
    t = t0
    copyto!(y, y0)

    # Save initial state
    ys = [copy(y), ]
    ts = [t, ]
    # Loop
    for _ = 1:N
        # Perform integration step
        integration_step!(cache, odefun, y, t, p, δt, calcerr)
        # Increment time
        t += δt
        # save step
        push!(ys, copy(y))
        push!(ts, t)
        # Error calculations 
        #if calcerr
        #    cache.yerr .-= y
        #    err = sqrt(sum(abs2, cache.yerr))
        #    if err > errtol
        #        @warn "Error calculated to be $(err) at time $(t)"
        #    end
        #end
    end
    return ts, ys
end
