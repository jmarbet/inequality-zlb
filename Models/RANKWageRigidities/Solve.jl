"""
    solveModel(P, S)

Solves the model using the algorithm described in the notes.

"""
function solveModel(P, DSS;
    πPolicy = Array{Float64,2}(undef, 0, 0),
    HPolicy = Array{Float64,2}(undef, 0, 0),
    RIntercept = DSS.R)

    # Initialize inflation policy function
    if length(πPolicy) == 0
        πPolicy = P.π̃ .+ zeros(P.RGridSize, P.ξGridSize)
    end

    # Initialize labor policy function
    if length(HPolicy) == 0
        HPolicy = ones(P.RGridSize, P.ξGridSize) * DSS.H
    end

    # Initialize matrices
    πPolicyUpdate = similar(πPolicy)
    HPolicyUpdate = similar(HPolicy)
    πPolicyError = similar(πPolicy)
    HPolicyError = similar(HPolicy)
    errorCodes = zeros(Int64, size(πPolicy))
    ZLBNodes = zeros(Int64, size(πPolicy))

    # Interpolate πPolicy and HPolicy
    πPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), πPolicy, extrapolation_bc = Line())
    HPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), HPolicy, extrapolation_bc = Line())

    # Initialize inflation policy function iteration
    iter = 1
    dist = 10.0

    # Do inflation and labor policy funtion interation
    while(dist > P.tol)

        # Update the interpolation
        πPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), πPolicy, extrapolation_bc = Line())
        HPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), HPolicy, extrapolation_bc = Line())

        # Compute the proposal for new inflation policy function
        @threads for idx in CartesianIndices(πPolicy)

            # Get indices in the grid
            i_R = idx[1]
            i_ξ = idx[2]

            # Solve for policy function using non-linear solver
            f!(res, τ) = residFOC(res, τ, P, DSS, πPolicyInterpol, HPolicyInterpol, idx; RIntercept)
            res = nlsolve(f!, [πPolicy[idx], HPolicy[idx]])#, show_trace = false, ftol = 1e-4)
            πPolicyUpdate[idx] = res.zero[1]
            HPolicyUpdate[idx] = res.zero[2]
            errorCodes[idx] = converged(res) ? 0 : 1 # Note 0 means the solver converged. 1 means that there was some issue


            # Check if ZLB is binding
            ZLBNodes[idx] = checkZLB(P, DSS, P.RGrid[i_R], πPolicyUpdate[idx], HPolicyUpdate[idx]; RIntercept)

        end

        # Check the distance between current iteration and the previous one
        @. πPolicyError = abs.(πPolicyUpdate - πPolicy)
        @. HPolicyError = abs.(HPolicyUpdate - HPolicy)
        dist1 = maximum(πPolicyError)
        dist2 = maximum(HPolicyError)
        dist = max(dist1, dist2)

        # Display current iteration
        if P.showPolicyIterations
            println("PF Iteration: ", iter, " (Distance H policy: ", @sprintf("%2.8f", dist2),
                    ", Dist. π policy: ", @sprintf("%2.8f", dist1), ", Max π: ", @sprintf("%2.8f", maximum(πPolicy)),
                    " Min π: ", @sprintf("%2.8f", minimum(πPolicy)), ", ", sum(errorCodes .!= 0), " Nodes with error, ",
                    sum(ZLBNodes), " Nodes with binding ZLB (", @sprintf("%2.2f", sum(ZLBNodes)/length(ZLBNodes)*100), "% of all nodes, ",
                    @sprintf("%2.2f", sum(ZLBNodes .* (errorCodes .!= 0))/sum(errorCodes .!= 0)*100), "% of nodes with errors)")
        end

        # Update the policy functions
        @. πPolicy = P.λ * πPolicyUpdate + (1-P.λ) * πPolicy
        @. HPolicy = P.λ * HPolicyUpdate + (1-P.λ) * HPolicy
        iter = iter+1

    end

    if !P.showPolicyIterations
        println("PF Iteration: ", iter, " (Distance max(H policy, π policy): ", @sprintf("%2.8f", dist),
                ", Max π: ", @sprintf("%2.8f", maximum(πPolicy)), ", Min π: ", @sprintf("%2.8f", minimum(πPolicy)), ", ", sum(errorCodes .!= 0), " Nodes with error, ",
                sum(ZLBNodes), " Nodes with binding ZLB (", @sprintf("%2.2f", sum(ZLBNodes)/length(ZLBNodes)*100), "% of all nodes, ",
                @sprintf("%2.2f", sum(ZLBNodes .* (errorCodes .!= 0))/sum(errorCodes .!= 0)*100), "% of nodes with errors)")
    end

    return πPolicy, HPolicy


end


"""
    residFOC(policy, P, DSS, πPolicyInterpol, HPolicyInterpol, idx)

Auxiliary function that evaluates the FOC of the model and returns its error for
given states and policies.

"""
function residFOC(policy, P, DSS, πPolicyInterpol, HPolicyInterpol, idx; RIntercept = DSS.R)
    res = zeros(2)
    residFOC(res, policy, P, DSS, πPolicyInterpol, HPolicyInterpol, idx; RIntercept)
    return res
end


"""
    residFOC(policy, P, DSS, πPolicyInterpol, HPolicyInterpol, idx; RIntercept = DSS.R)

Auxiliary function that evaluates the FOC of the model and returns its error for
given states and policies.

"""
function residFOC(res, policy, P, DSS, πPolicyInterpol, HPolicyInterpol, idx; RIntercept = DSS.R)

    # Policies
    π = policy[1]
    H = policy[2]

    # Make sure that the solver doesn't try values for policies that lead to errors
    if π < 0.0 || H < 0
         res .= fill(NaN, 2)
         return
    end

    # Get indices in the grid
    i_R = idx[1]
    i_ξ = idx[2]

    # Wage inflation
    πw = π

    # Determine values of state variables
    RStarPrev = P.RGrid[i_R]
    ξ = P.ξGrid[i_ξ]

    # Check if ZLB was binding in previous period
    if P.bindingZLB && RStarPrev < P.ZLBLevel
        Rprev = P.ZLBLevel
    else
        Rprev = RStarPrev
    end

    # Set inflation indexation parameter
    if P.indexRotemberg
        πwInd = DSS.πw
    else
        πwInd = 1.0
    end

    # Output
    Y = H

    # Consumption (from resource constraint)
    C = Y
    
    # Real wage
    w = 1.0

    # Compute nominal interest rate set by the central bank
    R, RStar = monetaryPolicyRule(P, DSS, π, Y, RStarPrev; RIntercept)

    # Initialize the expectation terms
    bondExpec = 0.0
    unionExpec = 0.0

    for jj in 1:length(P.eNodes)

        # Compute TFP in the next period
        ξp = P.ξ̄ * (ξ/P.ξ̄)^P.ρ * exp(P.eNodes[jj])

        # Interpolate policy functions for next period
        πp = πPolicyInterpol(RStar, ξp)
        Hp = HPolicyInterpol(RStar, ξp)

        # Output
        Yp = Hp

        # Consumption (from resource constraint)
        Cp = Yp

        # Wage inflation
        πwp = πp

        # Stochastic discount factor
        q = ξp/ξ * P.β * Cp^(-P.σ) / C^(-P.σ)

        # Compute Euler equation expectation term
        bondExpec += q * R/πp * P.eWeights[jj]

        # Compute expectation firm in wage inflation equation
        unionExpec += P.β̃ * log(πwp/πwInd) * (Hp/H) * P.eWeights[jj]

    end

    # Residuals of optimality conditions
    res[1] = bondExpec - 1
    res[2] = P.θ/P.ε * (log(πw/πwInd) - unionExpec) - P.χ * H^(P.σ+P.ν) + (P.ε-1) / P.ε * (1 - P.τ) * w  # Wage inflation equation

end


"""
    solveSteadyState(P)

Computes the deterministic steady state (DSS) of the model.

"""
function solveSteadyState(P)

    @unpack π̃, β, ε, θ, β̃, σ, ν, χ, τ, B, indexRotemberg = P

    return solveSteadyState(π̃, β, ε, θ, β̃, σ, ν, χ, τ, B, indexRotemberg)

end


"""
    solveSteadyState(π̃, β, ε, θ, β̃, σ, ν, χ, τ, B, indexRotemberg)

Computes the deterministic steady state (DSS) of the model.

"""
function solveSteadyState(π̃, β, ε, θ, β̃, σ, ν, χ, τ, B, indexRotemberg)

    # Preference shock, inflation, wage inflation, and the real wage in the DSS
    ξ = 1.0
    π = π̃
    πw = π̃
    w = 1.0

    # From the Euler equation
    R = π / β
    r = 1 / β

    # Aggregate labor supply
    if indexRotemberg
        H = (1/χ * (ε-1)/ε * (1 - τ) * w)^(1/(σ+ν))
    else
        H = (1/χ * ((ε-1)/ε * (1 - τ) * w + θ * (1-β̃) / ε * log(πw)))^(1/(σ+ν))
    end

    # From the production function and market clearing
    Y = H
    C = Y

    # Transfers
    T = τ * w * H - (r-1) * B

    return (R = R,
            r = r,
            w = w,
            H = H,
            Y = Y,
            C = C,
            π = π,
            πw = πw,
            ξ = ξ,
            B = B,
            T = T)

end


"""
    simulateRemainingVariables(P, DSS, S, πPolicy, HPolicy)

Simulates the remaining variables of the model for a given sequence of states and
policy functions.

"""
function simulateRemainingVariables(P, DSS, S, πPolicy, HPolicy; ξ = S.ξ, RIntercept = DSS.R, RStarInit = RIntercept, burnIn = P.burnIn)

        # Check if ZLB was binding for initialization of RStar
        if P.bindingZLB && RStarInit < P.ZLBLevel
            RInit = P.ZLBLevel
        else
            RInit = RStarInit
        end

        # Create the interpolation functions
        πPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), πPolicy, extrapolation_bc = Line())
        HPolicyInterpol = linear_interpolation((P.RGrid, P.ξGrid), HPolicy, extrapolation_bc = Line())

        # Initalize states and main policies
        π = zeros(length(ξ))
        H = zeros(length(ξ))
        Y = zeros(length(ξ))
        R = zeros(length(ξ))
        RStar = zeros(length(ξ))

        # First period
        π[1] = πPolicyInterpol(RStarInit, ξ[1])
        H[1] = HPolicyInterpol(RStarInit, ξ[1])
        Y[1] = H[1]
        R[1], RStar[1] = monetaryPolicyRule(P, DSS, π[1], Y[1], RStarInit; RIntercept)

        # Remaining periods
        for tt in 2:length(ξ)
            π[tt] = πPolicyInterpol(RStar[tt-1], ξ[tt])
            H[tt] = HPolicyInterpol(RStar[tt-1], ξ[tt])
            Y[tt] = H[tt]
            R[tt], RStar[tt] = monetaryPolicyRule(P, DSS, π[tt], Y[tt], RStar[tt-1]; RIntercept)
        end

        # Consumption (from resource constraint)
        C = @. Y

        # Wage inflation
        πw = copy(π)

        # Compute the implied real rate
        r = [RInit/π[1]; @. R[1:end-1] / π[2:end]]

        # Real wage
        w = ones(length(ξ))

        # Check in which periods the ZLB binds
        ZLBBinds = (R .<= P.ZLBLevel)

        # Compute bond net supply and transfers
        B = ones(length(ξ)) * DSS.B
        T = @. P.τ * w * H - (r-1) * B

        # Remove burnIn periods
        R = R[burnIn+1:end]
        r = r[burnIn+1:end]
        ξ = ξ[burnIn+1:end]
        w = w[burnIn+1:end]
        H = H[burnIn+1:end]
        Y = Y[burnIn+1:end]
        C = C[burnIn+1:end]
        π = π[burnIn+1:end]
        πw = πw[burnIn+1:end]
        B = B[burnIn+1:end]
        T = T[burnIn+1:end]
        ZLBBinds = ZLBBinds[burnIn+1:end]

        # Fraction of periods where the ZLB binds
        ZLBBindsFrac = sum(ZLBBinds)/length(ZLBBinds)

        return (R = R,
                RStar = RStar,
                r = r,
                ξ = ξ,
                w = w,
                H = H,
                Y = Y,
                C = C,
                π = π,
                πw = πw,
                B = B,
                T = T,
                ZLBBinds = ZLBBinds,
                ZLBBindsFrac = ZLBBindsFrac)

end


"""
    monetaryPolicyRule(P, DSS, π, Y, RStarPrev; RIntercept = DSS.R)

Taylor rule of the model.

"""
function monetaryPolicyRule(P, DSS, π, Y, RStarPrev; RIntercept = DSS.R)

    # Compute nominal interest rate implied by asymmetric Taylor rule
    # Note ϕₕ = ϕₗ implies a standard Taylor rule
    RInf = (π < DSS.π) * (π/DSS.π)^(P.ϕₗ) + (π >= DSS.π) * (π/DSS.π)^(P.ϕₕ)
    RStar = RIntercept * (RStarPrev/RIntercept)^P.ρ_R * (RInf * (Y/DSS.Y)^P.ϕʸ)^(1-P.ρ_R)

    # Check whether ZLB is binding
    if P.bindingZLB && RStar < P.ZLBLevel
        R = P.ZLBLevel
    else
        R = RStar
    end

    # Set RStar = R to use R for inertia in the Taylor rule
    if !P.useRStarForTaylorInertia
        RStar = R
    end

    return R, RStar

end



"""
    checkZLB(P, DSS, RStarPrev, π, H; RIntercept = DSS.R)

Checks whether the ZLB is binding for a particular node in the state space.

"""
function checkZLB(P, DSS, RStarPrev, π, H; RIntercept = DSS.R)

    # Output
    Y = H

    # Compute nominal interest rate
    R, _ = monetaryPolicyRule(P, DSS, π, Y, RStarPrev; RIntercept)

    # Check whether the ZLB binds
    if R <= P.ZLBLevel
        ZLBBinds = 1
    else
        ZLBBinds = 0
    end

    return ZLBBinds

end


"""
    computeShocks(P)

Simulates aggregate shocks based on the settings defined in P.

"""
function computeShocks(P)

    # Simulate preference shock
    ξ = P.ξ̄ * ones(P.T)

    for tt in 2:P.T
        ξ[tt] = P.ξ̄ * (ξ[tt-1]/P.ξ̄)^P.ρ * exp(P.σ̃ * randn())
    end

    # Make sure ξ is within the grid bounds
    ξ = max.(P.ξGrid[1], ξ)
    ξ = min.(P.ξGrid[end], ξ)

    return (ξ = ξ,)

end


"""
    plotComparison(P, DSS, SSS, S, simSeries; RIntercept = DSS.R)

Plots several of the simulated series and deterministic and stochastic steady states.

"""
function plotComparison(P, DSS, SSS, S, simSeries; RIntercept = DSS.R)

    # Labor
    p1 = plot(simSeries.H, ylabel = "Labor",  legend = :none)
    hline!([DSS.H], label = "DSS", linestyle = :dash)
    hline!([SSS.H], label = "SSS", linestyle = :dot)

    # Consumption
    p2 = plot(simSeries.C, ylabel = "Consumption",  legend = :none)
    hline!([DSS.C], label = "DSS", linestyle = :dash)
    hline!([SSS.C], label = "SSS", linestyle = :dot)

    # Wage
    p3 = plot(simSeries.w, ylabel = "Wage", legend = :none)
    hline!([DSS.w], label = "DSS", linestyle = :dash)
    hline!([SSS.w], label = "SSS", linestyle = :dot)

    # Nominal interest rate
    p4 = plot(simSeries.R, ylabel = "Nominal Rate", legend = :none)
    hline!([DSS.R], label = "DSS", linestyle = :dash)
    hline!([SSS.R], label = "SSS", linestyle = :dot)
    if RIntercept != DSS.R
        hline!([RIntercept], label = "Intercept", linestyle = :dashdot)
    end

    # Inflation
    p5 = plot(simSeries.π, ylabel = "Inflation", legend = :none)
    hline!([DSS.π], label = "DSS", linestyle = :dash)
    hline!([SSS.π], label = "SSS", linestyle = :dot)

    # Preference shock
    p6 = plot(S.ξ, ylabel = "Preference Shock", legend = :none)
    hline!([DSS.ξ], label = "DSS", linestyle = :dash)
    hline!([SSS.ξ], label = "SSS", linestyle = :dot)

    # Inflation
    p9 = plot(simSeries.ZLBBinds, ylabel = "ZLB Binds", legend = :none, linetype=:steppost)

    # Combine the plots
    l = @layout [a; b; c; d; e; f]
    p = plot(p1, p2, p3, p4, p5, p6,
        layout = l, legendfontsize = 5, guidefontsize = 6, tickfontsize = 6, size = (600, 900))

    # Print some statistics
    println("Binding ZLB in ", 100*simSeries.ZLBBindsFrac, "% of simulated periods")
    println(sum(simSeries.ZLBBinds))
    println(length(simSeries.ZLBBinds))
    return p

end


"""
    computeStochasticSteadyState(P, DSS, S, πPolicy, HPolicy; RIntercept = DSS.R)

Computes the stochatic steady state of the model.

"""
function computeStochasticSteadyState(P, DSS, S, πPolicy, HPolicy; RIntercept = DSS.R)

    # Zero out aggregate shocks after burnIn Period
    ξTmp = copy(S.ξ)
    ξTmp[P.burnIn+1:end] .= P.ξ̄
    SZero = (ξ = ξTmp,)

    # Simulate other variables to find their stochasicsteady state
    simSeries = simulateRemainingVariables(P, DSS, SZero, πPolicy, HPolicy; RIntercept)

    return NamedTuple{keys(simSeries)}([x[end] for x in simSeries])

end
