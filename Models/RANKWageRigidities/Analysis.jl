"""
    computeIRFs(P, DSS, SSS, πPolicy, HPolicy; std = 1, length = 41)

Compute impulse response functions.

"""
function computeIRFs(P, DSS, SSS, πPolicy, HPolicy; std = 1, length = 41)

    # Simulate preference shock
    ξ = P.ξ̄ * ones(length+1)
    ξ[2] = exp(P.σ̃ * std)

    for tt in 3:length+1
        ξ[tt] = P.ξ̄ * (ξ[tt-1]/P.ξ̄)^P.ρ
    end

    IRFs = simulateRemainingVariables(P, DSS, (0,), πPolicy, HPolicy; ξ = ξ, RStarInit = SSS.R, burnIn = 1)

    return IRFs

end


"""
    plotIRFs(P, IRFs, SSS)

Plot set of impulse response functions.

"""
function plotIRFs(P, IRFs, SSS)

    # IRFs in levels
    p1 = plotIRF(IRFs.π, SSS.π, "Inflation (%)"; isInterestRate = true, inDeviations = false)
    p2 = plotIRF(IRFs.r, SSS.r, "Real Interest Rate (%)"; isInterestRate = true, inDeviations = false)
    p3 = plotIRF(IRFs.R, SSS.R, "Nominal Interest Rate (%)"; isInterestRate = true, inDeviations = false)
    p4 = plotIRF(IRFs.Y, SSS.Y, "Output"; isInterestRate = false, inDeviations = false)
    pa = plot(p1, p2, p3, p4, layout = 4, size = (720, 480))

    # IRFs in deviations from SSS
    p1 = plotIRF(IRFs.π, SSS.π, "Inflation (pp-dev.)"; isInterestRate = true, inDeviations = true)
    p2 = plotIRF(IRFs.r, SSS.r, "Real Rate (pp-dev.)"; isInterestRate = true, inDeviations = true)
    p3 = plotIRF(IRFs.R, SSS.R, "Nominal Rate (pp-dev.)"; isInterestRate = true, inDeviations = true)
    p4 = plotIRF(IRFs.Y, SSS.Y, "Output (%-dev.)"; isInterestRate = false, inDeviations = true)
    pb = plot(p1, p2, p3, p4, layout = 4, size = (720, 480))

    return pa, pb
end


"""
    plotIRFs(P, IRFs, SSS)

Plot particular impulse response function.

"""
function plotIRF(IRF, SSS, label; isInterestRate = false, inDeviations = true, color = :auto, linestyle = :auto, linewidth = 2, xlabel = "Quarter")

    # Transform the series
    if isInterestRate && inDeviations
        IRF = log.(IRF)*400 .- log(SSS)*400
        hlineLevel = 0
    elseif isInterestRate
        IRF = log.(IRF)*400
        hlineLevel = log.(SSS)*400
    elseif inDeviations
        IRF = (IRF / SSS .- 1) * 100
        hlineLevel = 0
    else
        hlineLevel = SSS
    end

    # Plot the IRFs and horizontal line at SSS or at zero
    p = plot(0:length(IRF)-1, IRF; color, linestyle, xlabel, linewidth, legend =:none, ylabel = label)
    hline!([hlineLevel], linestyle = :dash, color = :black)

    return p

end


"""
    plotIRF!(P, IRFs, SSS)

Plot particular impulse response function.

"""
function plotIRF!(p, IRF, SSS, label; isInterestRate = false, inDeviations = true, color = :auto, linestyle = :auto, linewidth = 2)

    # Transform the series
    if isInterestRate && inDeviations
        IRF = log.(IRF)*400 .- log(SSS)*400
    elseif isInterestRate
        IRF = log.(IRF)*400
    elseif inDeviations
        IRF = (IRF / SSS .- 1) * 100
    else
        hlineLevel = SSS
    end

    # Plot the IRFs and horizontal line at SSS or at zero
    plot!(p, 0:length(IRF)-1, IRF; color, linestyle, linewidth)

    nothing

end


"""
    computeValueFunction()

Example of how to compute and plot the value function.

"""
function computeValueFunction()

    tstart = time()

    # Load settings
    P = settings()

    # Draw the individual and aggregate shocks
    S = computeShocks(P)

    # Compute deterministic steady state
    DSS = solveSteadyState(P)

    # Solve representative agent model
    πPolicy, HPolicy = solveModel(P, DSS)

    # Compute value function
    V = computeValueFunction(P, S, DSS, πPolicy, HPolicy; showSteps = true)

    # Save the computed value function
    @save "Results/$(P.filenameFolder)/ValueFunction_$(P.filenameExt).bson" P V

    # Plot the value function
    VInterpol = linear_interpolation((P.RGrid, P.ξGrid), V, extrapolation_bc = Line())
    p = surface(P.RGrid, P.ξGrid, (x, y) -> VInterpol(x, y), xlabel = "R", ylabel = "ξ", zlabel = "Welfare", legend = :none)
    display(p)

    # Display runtime
    displayTimeElapsed(tstart)

end


"""
    computeValueFunction(P, S, DSS, πPolicy, HPolicy; showSteps = false)

Compute value function for given policies and settings.

"""
function computeValueFunction(P, S, DSS, πPolicy, HPolicy; showSteps = false)

    # Initalize progress indicator
    if showSteps
        p = ProgressUnknown(desc = "Value Function Iteration:",  color = :grey)
    else
        print("Value Function Iteration")
    end

    # Compute output and consumption policy
    Y = @. HPolicy
    CPolicy = @. HPolicy

    # Make an initial guess for the value function
    V = zeros(P.RGridSize, P.ξGridSize)
    V0 = copy(V)
    VError = similar(V)

    # Interpolate the value function (used for computing the expectation of the value function)
    V0Interpol = linear_interpolation((P.RGrid, P.ξGrid), V0, extrapolation_bc = Line())

    #
    dist = 10.0
    iter = 1

    while dist > P.tol

        @threads for idx in CartesianIndices(V)

            # Determne indices of 1D grids
            i_R = idx[1]
            i_ξ = idx[2]

            #
            Rprev = P.RGrid[i_R]
            ξ = P.ξGrid[i_ξ]

            # Compute nominal interest rate (which is a state in the next period)
            R, RStar = monetaryPolicyRule(P, DSS, πPolicy[idx], Y[idx], Rprev)

            # Compute the value function expectation
            EV = 0.0
            for ii in 1:P.nGHNodes

                # Preference shock in the next period
                ξPrime = P.ξ̄ * (ξ/P.ξ̄)^P.ρ * exp(P.eNodes[ii])

                EV += V0Interpol(RStar, ξPrime) * P.eWeights[ii]

            end

            # Update the value function
            if P.σ == 1.0
                V[idx] = ξ * (log(CPolicy[idx]) - P.χ * HPolicy[idx]^(1+P.ν)/(1+P.ν)) + P.β * EV
            else
                V[idx] = ξ * (1/(1-P.σ) * CPolicy[idx]^(1-P.σ) - P.χ * HPolicy[idx]^(1+P.ν)/(1+P.ν)) + P.β * EV
            end

        end

        # Compute the distance between the current and previous iteration
        @. VError = abs(V - V0)
        dist = maximum(VError)
        if showSteps
            ProgressMeter.next!(p; showvalues = [(:iter, iter), (:dist, dist), (:distAlt, mean(VError.^2))], valuecolor = :grey)
        end

        # Prepare the next iteration
        V0 .= V
        V0Interpol = linear_interpolation((P.RGrid, P.ξGrid), V0, extrapolation_bc = Line())
        iter += 1

    end

    # Display final iteration
    if !showSteps
        println(iter, " (Distance: ", dist, ")")
    else
        ProgressMeter.finish!(p)
    end

    return V

end


"""
    taylorRuleIntercept()

Evaluates how the changes in the intercept in the Taylor rule affect the steady state real rate.

"""
function taylorRuleIntercept(; config = tuple())

    # Load settings
    P = settings(; config ...)

    # Draw the individual and aggregate shocks
    S = computeShocks(P)

    # Compute deterministic steady state
    DSS = solveSteadyState(P)

    # Compute the stochasttic steady state with the DSS nominal rate as the intercept
    πPolicy, HPolicy = solveModel(P, DSS)
    SSSBaseline = computeStochasticSteadyState(P, DSS, S, πPolicy, HPolicy)

    # Intercepts to be evaluated
    RMinus = 0.005/4
    RPlus = 0.0025/4
    RIntercepts = range(DSS.R - RMinus, stop = DSS.R + RPlus, length = 11)

    # Initialize vector for results
    SSSs = Any[]

    for RIntercept in RIntercepts

        # Solve representative agent model
        πPolicy, HPolicy = solveModel(P, DSS; πPolicy, HPolicy, RIntercept)

        # Compute stochastic steady state
        SSS = computeStochasticSteadyState(P, DSS, S, πPolicy, HPolicy; RIntercept)

        # Add steady state to list
        push!(SSSs, SSS)

    end

    # Plot comparison
    p1 = plot(log.(RIntercepts)*400, [log(x.π)*400 for x in SSSs], xlabel = "Taylor Rule Intercept (%)", ylabel = "%", title = "Inflation", label = "SSS", legend = :none)
    scatter!([log(DSS.R)*400], [log(SSSBaseline.π)*400], color = :red, marker = :star, label = "SSS (Baseline)")
    scatter!([log(DSS.R)*400], [log(DSS.π)*400], color = :red, marker = :circle, label = "DSS (Baseline)")
    
    p2 = plot(log.(RIntercepts)*400, [log(x.r)*400 for x in SSSs], xlabel = "Taylor Rule Intercept (%)", ylabel = "%", title = "Real Rate", label = "SSS", legend = :none)
    scatter!([log(DSS.R)*400], [log(SSSBaseline.r)*400], color = :red, marker = :star, label = "SSS (Baseline)")
    scatter!([log(DSS.R)*400], [log(DSS.r)*400], color = :red, marker = :circle, label = "DSS (Baseline)")
    
    p3 = plot(log.(RIntercepts)*400, [log(x.R)*400 for x in SSSs], xlabel = "Taylor Rule Intercept (%)", ylabel = "%", title = "Nominal Rate", label = "SSS", legend = :none)
    scatter!([log(DSS.R)*400], [log(SSSBaseline.R)*400], color = :red, marker = :star, label = "SSS (Baseline)")
    scatter!([log(DSS.R)*400], [log(DSS.R)*400], color = :red, marker = :circle, label = "DSS (Baseline)")
    plot!(log.(RIntercepts)*400, log.(RIntercepts)*400, linestyle = :dashdot, color = :grey, label = "")
    
    p4 = plot(1:1, xlabel = "Taylor Rule Intercept (%)", ylabel = "%", label = "SSS", legend = :topleft, framestyle = :none)
    scatter!([NaN], [NaN], color = :red, marker = :star, label = "SSS (Baseline)")
    scatter!([NaN], [NaN], color = :red, marker = :circle, label = "DSS (Baseline)")
    
    pp = plot(p1, p2, p3, p4)
    display(pp)
    savefig(pp, "Figures/RANKWageRigidities/TaylorRuleIntercept.pdf")

    nothing

end
