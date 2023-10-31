"""
    prepAllResults()

Generates the all results needed for analysis and plotting, which take a lot of
time to compute.

"""
function prepAllResults()

    tstart = time()

    # Files containing solution
    filenames = ["Results/HANKWageRigidities/HANKWageRigidities_ZLB_pitilde_0_02_ELB_0_0.bson",
                 "Results/HANKWageRigidities/HANKWageRigidities_ZLB_pitilde_0_02_ELB_-1_0.bson"]

    # Settings
    outputFolder = "Results/HANKWageRigidities/Results"

    # Create the output folder
    if !isdir(outputFolder)
        mkpath(outputFolder)
    end

    #
    prepareResults(filenames, outputFolder)
    prepareAdditonalResults(filenames, outputFolder)

    # Display runtime
    displayTimeElapsed(tstart)

    nothing

end


"""
    prepareResults(filenames, outputFolder)

Generates the main results needed for analysis and plotting, which take a lot of
time to compute.

"""
function prepareResults(filenames, outputFolder)

    for filename in filenames

        println("----------------------------------------")
        println(filename)
        println("")

        # Load the settings
        @load filename P DSS bPolicy πwALM EπwCondALM

        # Compute stochastic steady state
        println("Computing SSS...")
        SSS, simSeriesSSS = computeStochasticSteadyState(P, DSS, bPolicy, πwALM, EπwCondALM; T = 500, burnIn = 0, avgPeriods = 1)
        @save "$(outputFolder)/SSS_$(P.filenameExt).bson" P SSS simSeriesSSS

        # Simulate series
        println("Simulating Series...")
        simSeries, _ = simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM; T = 101000, burnIn = 1000)
        @save "$(outputFolder)/Simulation_$(P.filenameExt).bson" P simSeries

        # Compute value function
        println("Computing Value Function...")
        V = computeValueFunction(P, DSS, bPolicy, πwALM, EπwCondALM)
        @save "$(outputFolder)/ValueFunction_$(P.filenameExt).bson" P V

        # Compute value function in DSS
        println("Computing Value Function in DSS...")
        VDSS = computeValueFunctionDSS(P, DSS)
        @save "$(outputFolder)/ValueFunctionDSS_$(P.filenameExt).bson" P VDSS

        # Compute impulse response functions starting from SSS
        println("Computing IRFs...")
        IRFs1 = computeIRFs(P, DSS, SSS, bPolicy, πwALM, EπwCondALM; std = -1, T = 41, savebCrossSeries = true)
        IRFs2 = computeIRFs(P, DSS, SSS, bPolicy, πwALM, EπwCondALM; std = -2, T = 41, savebCrossSeries = true)
        IRFs3 = computeIRFs(P, DSS, SSS, bPolicy, πwALM, EπwCondALM; std = -3, T = 41, savebCrossSeries = true)
        @save "$(outputFolder)/IRFs_$(P.filenameExt).bson" P IRFs1 IRFs2 IRFs3

    end

    nothing

end


"""
    prepareAdditonalResults(filenames, outputFolder)

Generates the additional results needed for analysis and plotting, which take
even more time to compute or use a lot of storage space.

"""
function prepareAdditonalResults(filenames, outputFolder)

    for filename in filenames

        println("----------------------------------------")
        println(filename)
        println("")

        # Load the settings
        @load filename P DSS bPolicy πwALM EπwCondALM

        println("Simulating Series (incl. bCross series)...")
        simSeries, _ = simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM; T = 11000, burnIn = 1000, savebCrossSeries = true)
        @save "$(outputFolder)/SimulationPlus_$(P.filenameExt).bson" P simSeries

    end

    nothing

end


"""
    simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM;
        bCrossInit = DSS.bCross,
        RStarInit = DSS.R,
        HInit = DSS.H,
        πwInit = DSS.πw,
        savebCrossSeries = true,
        removeFirstPeriod = false,
        burnIn = 0,
        T = P.T,
        ζ = P.ζ̄ * ones(T))

Simulates all variables of the model.

"""
function simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM;
        bCrossInit = DSS.bCross,
        RStarInit = DSS.RStar,
        HInit = DSS.H,
        πwInit = DSS.πw,
        remainingInit = DSS,
        savebCrossSeries = false,
        removeFirstPeriod = false,
        burnIn = 0,
        T = P.T,
        ζ = P.ζ̄ * ones(T),
        simulateAggregateShock = true)

    # Update settings
    P = settings(P; T = T)

    # Simulate the aggregate shock
    if simulateAggregateShock
        S = computeShocks(P)
    else
        S = (ζ = ζ, )
    end

    # Interpolate the ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    # Initialize the dictionary that holds all simulated variables
    simSeries = Dict()
    simSeries[:ζ] = S.ζ

    # Simulate main variables
    simSeries[:πw], simSeries[:EπwCond], simSeries[:H], simSeries[:RStar], bCross, bCrossSeries =
        simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCrossInit, RStarInit, HInit, πwInit; savebCrossSeries = true) # Need the whole bCrossSeries

    # Define which of the remaining variables should be saved
    remainingVariables = [:π, :Eπw, :Y, :w, :R, :r, :B, :T, :C, :ξ, :q]

    # Initialize each series
    for var in remainingVariables
        simSeries[var] = similar(simSeries[:πw])
        simSeries[var][1] = getfield(remainingInit, var)
    end

    # Compute period 2 to T for the remaining variables
    for tt in 2:P.T

        # Compute variables for period tt
        res = computeAllAggregateVariables(P, DSS, simSeries[:RStar][tt-1], simSeries[:ζ][tt],  bCrossSeries[:, :, tt], πwALMInterpol, EπwCondALMInterpol; πw = simSeries[:πw][tt])

        # Assign the computed values to period tt for each variable
        for var in remainingVariables
            simSeries[var][tt] = getfield(res, var)
        end

    end

    # Remove burn-in periods
    for var in keys(simSeries)
        simSeries[var] = simSeries[var][burnIn+1:end]
    end

    if savebCrossSeries
        simSeries[:bCross] = bCrossSeries[:, :, burnIn+1:end]
    end

    return simSeries, bCross

end


"""
    computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))

Computes "aggregate" variables for given states, the bond distribution, and
perceived laws of motion. This function also computes additional variables that
are not required for the solution algorithm.


"""
function computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))

    # Compute the main variables
    πw, Eπw, π, w, H, Y, T, RStarPrime = computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = πw)

    # Determine nominal interest rate set by central bank in previous period
    R = applyLowerBound(P, RStar)

    # Real interest rate
    r = R/π

    # Determine nominal interest rate set by central bank in current period
    RPrime = applyLowerBound(P, RStarPrime)

    # From aggregate resource constraint
    C = Y

    # Net supply of bonds
    B = DSS.B

    # Assign TFP and preference shock
    aggState = convertAggregateState(P, ζ)
    ξ = aggState.ξ
    q = aggState.q

    return (π = π,
            πw = πw,
            Eπw = Eπw,
            H = H,
            w = w,
            Y = Y,
            R = RPrime,
            RStar = RStarPrime,
            r = r,
            C = C,
            B = B,
            T = T,
            ξ = ξ,
            q = q)

end


"""
    computeStochasticSteadyState(P, DSS, bPolicy, πwALM, EπwCondALM; T = 500, burnIn = 0, avgPeriods = 1)

Computes the stochastic steady state of the model.

"""
function computeStochasticSteadyState(P, DSS, bPolicy, πwALM, EπwCondALM; T = 500, burnIn = 0, avgPeriods = 1)

    # Simulate aggregate shock (Note: shocks only hit during the burn-in period, afterwards ζ[t] = P.ζ̄)
    ζ = P.ζ̄ * ones(T + burnIn)

    for tt in 2:burnIn
        ζ[tt] = P.ζ̄ * (ζ[tt-1]/P.ζ̄)^P.ρ_ζ * exp(P.σ̃_ζ * randn())
    end

    # Simulate the model (starting from the DSS)
    simSeries, bCross = simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM; T = T+burnIn, ζ = ζ, simulateAggregateShock = false)

    # Take the average of the last "avgPeriods" number of periods as the SSS
    # (Note: if avgPeriods == 1, this just takes the last period of the simulation)
    vars = collect(keys(simSeries))
    SSSVec = Any[mean(simSeries[x][end-avgPeriods+1:end]) for x in vars if x != :bCross]

    # Add the bond distribution
    push!(vars, :bCross)
    push!(SSSVec, bCross)

    # Save the SSS as a named tuple
    SSS = NamedTuple{tuple(vars...)}(SSSVec)

    return SSS, simSeries

end


"""
    computeIRFs(P, DSS, SSS, bPolicy, πwALM, EπwCondALM)

Computes the impulse response functions of the model.

"""
function computeIRFs(P, DSS, SSS, bPolicy, πwALM, EπwCondALM;
    std = 1,
    T = 41,
    bCrossInit = SSS.bCross,
    RStarInit = SSS.RStar,
    HInit = SSS.H,
    πwInit = SSS.πw,
    remainingInit = SSS,
    savebCrossSeries = false)

    # Simulate aggregate shock (t=1: SSS, t=2: shock hits)
    ζ = P.ζ̄ * ones(T+1)

    for tt in 2:T+1
        if tt == 2
            ζ[tt] = P.ζ̄ * (ζ[tt-1]/P.ζ̄)^P.ρ_ζ * exp(P.σ̃_ζ * std)
        else
            ζ[tt] = P.ζ̄ * (ζ[tt-1]/P.ζ̄)^P.ρ_ζ
        end
    end

    # Simulate the model (starting from the SSS or the chosen initial point)
    simSeries, bCross = simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM;
        bCrossInit = bCrossInit,
        RStarInit = RStarInit,
        HInit = HInit,
        πwInit = πwInit,
        remainingInit = remainingInit,
        savebCrossSeries = savebCrossSeries,
        burnIn = 1,
        T = T+1,
        ζ = ζ,
        simulateAggregateShock = false)

    return simSeries

end


"""
    computePercentile(hist, x, perc)

Computes the percentile for given distribution hist and associated grid points x.
Uses linear interpolation of the CDF to compute percentiles that do not correspond
to a particular grid point.

"""
function computePercentile(hist, x, perc)

    # Check if the percentile can be computed
    if perc < hist[1]
        @warn "Percentile smaller than mass associated with smallest grid point"
        return NaN
    end

    # Approximate the CDF based on the histogram
    cdf1 = cumsum(hist, dims = 1)
    cdfInterpol = linear_interpolation(x, cdf1)

    # Compute the point in the grid that corresponds to percentile perc
    f(x) = cdfInterpol(x) - perc
    res = bisect(f, x[1], x[end])

    return res.x

end


"""
    computeConsumptionDistributionDSS(P, DSS; cGridSize = 100)

Computes the consumption distribution in the DSS.

"""
function computeConsumptionDistributionDSS(P, DSS; cGridSize = 100)

    function getAdjacentConsumptionGridPoints(c)

        c = min(cMax - 1e-6, c)
        c = max(cMin + 1e-6, c)
        i_cLow = floor(Int64, (c - cMin) / Δc) + 1
        i_cUp = ceil(Int64, (c - cMin) / Δc) + 1

        return i_cLow, i_cUp

    end

    function getReassignmentWeightConsumption(c, i_cLow, i_cUp)

        if i_cLow == i_cUp
            ω = 1.0
        else
            ω = 1 - (c - cGrid[i_cLow]) / (cGrid[i_cUp] - cGrid[i_cLow])
        end

        return ω

    end

    # Interpolate bPolicy
    bPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), DSS.bPolicy)

    # Compute consumption for each point in the bond distribution
    cTmp = similar(DSS.bCross)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        # States and policies
        b = P.bDenseGrid[i_b]
        s = P.sGrid[i_s]
        bp = bPolicyInterpol(b, i_s)

        # From the budget constraint
        cTmp[i_b, i_s] = computeIndividualCashOnHand(P, DSS, b, s) - bp

    end

    # Compute the grid for consumption
    cMax = maximum(cTmp)
    cMin = minimum(cTmp)
    cGrid = range(cMin, stop = cMax, length = cGridSize)
    Δc = cGrid[2] - cGrid[1]

    # Redistribute the probabilities in bCross on cGrid
    cCross = zeros(cGridSize, P.sGridSize)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        c = cTmp[i_b, i_s]
        i_cLow, i_cUp = getAdjacentConsumptionGridPoints(c)
        ω = getReassignmentWeightConsumption(c, i_cLow, i_cUp)

        cCross[i_cLow, i_s] +=  ω * DSS.bCross[i_b, i_s]
        cCross[i_cUp, i_s] += (1-ω) * DSS.bCross[i_b, i_s]

    end

    return cCross, cGrid

end


"""
    computeConsumptionDistribution(P, DSS, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol;
    aggVars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol),
    cGridSize = 100,
    cGrid = fill(NaN, cGridSize))

Computes the consumption distribution for given states, wealth distribution, and policies.

"""
function computeConsumptionDistribution(P, DSS, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol;
    aggVars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol),
    cGridSize = 100,
    cGrid = fill(NaN, cGridSize))

    function getAdjacentConsumptionGridPoints(c)

        c = min(cMax - 1e-6, c)
        c = max(cMin + 1e-6, c)
        i_cLow = floor(Int64, (c - cMin) / Δc) + 1
        i_cUp = ceil(Int64, (c - cMin) / Δc) + 1

        return i_cLow, i_cUp

    end

    function getReassignmentWeightConsumption(c, i_cLow, i_cUp)

        if i_cLow == i_cUp
            ω = 1.0
        else
            ω = 1 - (c - cGrid[i_cLow]) / (cGrid[i_cUp] - cGrid[i_cLow])
        end

        return ω

    end

    # Determine nominal interest rate set by central bank in previous period
    R = applyLowerBound(P, RStar)

    # Compute consumption for each point in the bond distribution
    cTmp = similar(bCross)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        # States and policies
        b = P.bDenseGrid[i_b]
        s = P.sGrid[i_s]
        bp = bPolicyInterpol(b, RStar, i_s, ζ)

        # From the budget constraint
        cTmp[i_b, i_s] = computeIndividualCashOnHand(P, b, s, R, aggVars.π, aggVars.w, aggVars.H, aggVars.T) - bp

    end

    # Compute the grid for consumption
    if isnan(cGrid[1]) # Create cGrid based on the consumption computed for each bond grid point
        cMax = maximum(cTmp)
        cMin = minimum(cTmp)
        cGrid = range(cMin, stop = cMax, length = cGridSize)
    else # Use the supplied cGrid
        cMax = cGrid[end]
        cMin = cGrid[1]
        cGridSize = length(cGrid)
    end
    Δc = cGrid[2] - cGrid[1]

    # Redistribute the probabilities in bCross on cGrid
    cCross = zeros(cGridSize, P.sGridSize)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        c = cTmp[i_b, i_s]
        i_cLow, i_cUp = getAdjacentConsumptionGridPoints(c)
        ω = getReassignmentWeightConsumption(c, i_cLow, i_cUp)

        cCross[i_cLow, i_s] +=  ω * bCross[i_b, i_s]
        cCross[i_cUp, i_s] += (1-ω) * bCross[i_b, i_s]

    end

    return cCross, cGrid

end


"""
    computeIncomeDistribution(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol;
    aggVars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol),
    incGridSize = 100,
    incGrid = fill(NaN, cGridSize))

Computes the income distribution for given states, wealth distribution, and policies.

"""
function computeIncomeDistribution(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol;
    aggVars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCross, πwALMInterpol, EπwCondALMInterpol),
    incGridSize = 100,
    incGrid = fill(NaN, incGridSize))

    function getAdjacentIncomeGridPoints(inc)

        inc = min(incMax - 1e-6, inc)
        inc = max(incMin + 1e-6, inc)
        i_incLow = floor(Int64, (inc - incMin) / Δinc) + 1
        i_incUp = ceil(Int64, (inc - incMin) / Δinc) + 1

        return i_incLow, i_incUp

    end

    function getReassignmentWeightIncome(inc, i_incLow, i_incUp)

        if i_incLow == i_incUp
            ω = 1.0
        else
            ω = 1 - (inc - incGrid[i_incLow]) / (incGrid[i_incUp] - incGrid[i_incLow])
        end

        return ω

    end

    # Determine nominal interest rate set by central bank in previous period
    R = applyLowerBound(P, RStar)

    # Compute income for each point in the bond distribution
    incTmp = similar(bCross)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        # States and policies
        b = P.bDenseGrid[i_b]
        s = P.sGrid[i_s]

        # From the budget constraint
        incTmp[i_b, i_s] = computeIndividualCashOnHand(P, b, s, R, aggVars.π, aggVars.w, aggVars.H, aggVars.T) - b

    end

    # Compute the grid for income
    if isnan(incGrid[1]) # Create incGrid based on the income computed for each bond grid point
        incMax = maximum(incTmp)
        incMin = minimum(incTmp)
        incGrid = range(incMin, stop = incMax, length = incGridSize)
    else # Use the supplied incGrid
        incMax = incGrid[end]
        incMin = incGrid[1]
        incGridSize = length(incGrid)
    end
    Δinc = incGrid[2] - incGrid[1]

    # Redistribute the probabilities in bCross on incGrid
    incCross = zeros(incGridSize, P.sGridSize)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        inc = incTmp[i_b, i_s]
        i_incLow, i_incUp = getAdjacentIncomeGridPoints(inc)
        ω = getReassignmentWeightIncome(inc, i_incLow, i_incUp)

        incCross[i_incLow, i_s] += ω * bCross[i_b, i_s]
        incCross[i_incUp, i_s] += (1-ω) * bCross[i_b, i_s]

    end

    return incCross, incGrid

end


"""
    computeIncomeDistributionDSS(P, DSS, incGridSize = 100, incGrid = fill(NaN, incGridSize))

Computes the income distribution in the DSS.

"""
function computeIncomeDistributionDSS(P, DSS, incGridSize = 100, incGrid = fill(NaN, incGridSize))

    function getAdjacentIncomeGridPoints(inc)

        inc = min(incMax - 1e-6, inc)
        inc = max(incMin + 1e-6, inc)
        i_incLow = floor(Int64, (inc - incMin) / Δinc) + 1
        i_incUp = ceil(Int64, (inc - incMin) / Δinc) + 1

        return i_incLow, i_incUp

    end

    function getReassignmentWeightIncome(inc, i_incLow, i_incUp)

        if i_incLow == i_incUp
            ω = 1.0
        else
            ω = 1 - (inc - incGrid[i_incLow]) / (incGrid[i_incUp] - incGrid[i_incLow])
        end

        return ω

    end

    # Compute income for each point in the bond distribution
    incTmp = similar(DSS.bCross)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        # States and policies
        b = P.bDenseGrid[i_b]
        s = P.sGrid[i_s]

        # From the budget constraint
        incTmp[i_b, i_s] = computeIndividualCashOnHand(P, DSS, b, s) - b

    end

    # Compute the grid for income
    if isnan(incGrid[1]) # Create incGrid based on the income computed for each bond grid point
        incMax = maximum(incTmp)
        incMin = minimum(incTmp)
        incGrid = range(incMin, stop = incMax, length = incGridSize)
    else # Use the supplied incGrid
        incMax = incGrid[end]
        incMin = incGrid[1]
        incGridSize = length(incGrid)
    end
    Δinc = incGrid[2] - incGrid[1]

    # Redistribute the probabilities in bCross on incGrid
    incCross = zeros(incGridSize, P.sGridSize)

    for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize

        inc = incTmp[i_b, i_s]
        i_incLow, i_incUp = getAdjacentIncomeGridPoints(inc)
        ω = getReassignmentWeightIncome(inc, i_incLow, i_incUp)

        incCross[i_incLow, i_s] += ω * DSS.bCross[i_b, i_s]
        incCross[i_incUp, i_s] += (1-ω) * DSS.bCross[i_b, i_s]

    end

    return incCross, incGrid

end


"""
    computeLaborIncomeDistribution(P, DSS, RStar, ζ, πw, πwALMInterpol, EπwCondALMInterpol)

Computes the labor income distribution for given states, wealth distribution, and policies.

"""
function computeLaborIncomeDistribution(P, DSS, RStar, ζ, πw, πwALMInterpol, EπwCondALMInterpol)

    # Initialize dummy for the bond distribution
    # (reuqired by computeAllAggregateVariables but not used for computing
    # variables required for this function)
    bCrossDummy = zeros(P.bDenseGridSize, P.sGridSize)

    # Compute aggregate variables
    vars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCrossDummy, πwALMInterpol, EπwCondALMInterpol; πw = πw)

    # Compute labor income for each productivity type
    laborIncomeGrid = similar(P.sGrid)
    for i_s in 1:P.sGridSize
        laborIncomeGrid[i_s] = (1 - P.τ) * vars.w * P.sGrid * vars.H
    end

    # Get the stationay distribution of the markov chain
    prob = P.P0

    return prob, laborIncomeGrid

end


"""
    computeLaborIncomeDistributionDSS(P, DSS)

Computes the labor income distribution in the DSS.

"""
function computeLaborIncomeDistributionDSS(P, DSS)

    # Compute labor income for each productivity type
    laborIncomeGrid = similar(P.sGrid)
    for i_s in 1:P.sGridSize
        laborIncomeGrid[i_s] = (1 - P.τ) * DSS.w * P.sGrid * DSS.H
    end

    # Get the stationay distribution of the markov chain
    laborIncomeCross = P.P0

    return laborIncomeCross, laborIncomeGrid

end


"""
    computeValueFunction(P, DSS, bPolicy, πwALM, EπwCondALM)

Computes the value function for given policy functions.

"""
function computeValueFunction(P, DSS, bPolicy, πwALM, EπwCondALM)

    # Initalize progress indicator
    if P.showPolicyIterations
        p = ProgressUnknown(desc = "Value Function Iteration:",  color = :grey)
    else
        print("Value Function Iteration: ")
    end

    # Interpolate ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    # Precompute variables that are not affected by the value function iteration
    RStarPrime, wealth, H, _, _, _, _, ζPrime =
        precomputeStaticPartsIndividualProblem(P, DSS, πwALMInterpol, EπwCondALMInterpol)
    cons = @. wealth - bPolicy

    # Make an initial guess for the value function
    V = zeros(P.bGridSize, P.RGridSize, P.sGridSize, P.ζGridSize)
    HMat = permutedims(H[:,:,:,:], (3, 1, 4, 2)) # H only varies over aggregate states
    if P.σ == 1.0
        @. V = (log(cons) - P.χ * HMat^(1+P.ν)/(1+P.ν)) / (1-P.β)
    else
        @. V = (1/(1-P.σ) * (cons)^(1-P.σ) - P.χ * HMat^(1+P.ν)/(1+P.ν)) / (1-P.β)
    end
    V0 = copy(V)
    VError = similar(V)

    # Interpolate the value function (used for computing the expectation of the value function)
    V0Interpol = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), V0, extrapolation_bc = Line())

    #
    dist = 10.0
    iter = 1

    while dist > P.tol

        @threads for idx in CartesianIndices(V)

            # Determne indices of 1D grids
            i_b = idx[1]
            i_R = idx[2]
            i_s = idx[3]
            i_ζ = idx[4]

            #
            b = P.bGrid[i_b]
            RStar = P.RGrid[i_R]
            s = P.sGrid[i_s]
            ζ = P.ζGrid[i_ζ]

            # Compute the value function expectation
            EV = 0.0
            for ii in 1:P.nGHNodes, jj in 1:P.sGridSize
                EV += V0Interpol(bPolicy[idx], RStarPrime[i_R, i_ζ], jj, ζPrime[i_ζ, ii]) * P.Ω[i_s, jj] * P.eWeights[ii]
            end

            # Determine preference and risk premium / discount factor shock
            aggState = convertAggregateState(P, ζ)
            ξ = aggState.ξ
            q = aggState.q

            # Update the value function
            if P.σ == 1.0
                V[idx] = ξ * (log(cons[idx]) - P.χ * H[i_R, i_ζ]^(1+P.ν)/(1+P.ν)) + P.β * q * EV
            else
                V[idx] = ξ * (1/(1-P.σ) * (cons[idx])^(1-P.σ) - P.χ * H[i_R, i_ζ]^(1+P.ν)/(1+P.ν)) + P.β * q * EV
            end

        end

        # Compute the distance between the current and previous iteration
        @. VError = abs(V - V0)
        dist = maximum(VError)

        # Update progress indicator
        if P.showPolicyIterations
            ProgressMeter.next!(p; showvalues = [(:iter, iter), (:dist, dist), (:distAlt, mean(VError.^2))], valuecolor = :grey)
        end

        # Prepare the next iteration
        V0 .= V
        V0Interpol = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), V0, extrapolation_bc = Line())
        iter += 1

    end

    # Display final iteration
    if !P.showPolicyIterations
        println(iter, " (Distance: ", dist, ")")
    else
        ProgressMeter.finish!(p)
    end

    return V

end


"""
    computeValueFunctionDSS(P, DSS)

Computes value function in the DSS.

"""
function computeValueFunctionDSS(P, DSS)

    # Initalize progress indicator
    if P.showPolicyIterations
        p = ProgressUnknown(desc = "Value Function Iteration (DSS):",  color = :grey)
    else
        print("Value Function Iteration (DSS): ")
    end

    # Compute wealth, consumption and labor for different grid points in the DSS
    wealth = zeros(P.bGridSize, P.sGridSize)
    cons = similar(wealth)

    for idx in CartesianIndices(wealth)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_s = idx[2]

        #
        b = P.bGrid[i_b]
        s = P.sGrid[i_s]

        # Compute consumption
        wealth[idx] = computeIndividualCashOnHand(P, DSS, b, s)
        cons[idx] = wealth[idx] - DSS.bPolicy[idx]

    end

    # Make an initial guess for the value function
    V = zeros(P.bGridSize, P.sGridSize)
    V0 = copy(V)
    VError = similar(V)

    # Interpolate the value function (used for computing the expectation of the value function)
    V0Interpol = linear_interpolation((P.bGrid, 1:P.sGridSize), V0, extrapolation_bc = Line())

    #
    dist = 10.0
    iter = 1

    while dist > P.tol

        @threads for idx in CartesianIndices(V)

            # Determne indices of 1D grids
            i_b = idx[1]
            i_s = idx[2]

            #
            b = P.bGrid[i_b]
            s = P.sGrid[i_s]
            ξ = P.ξ̄
            q = P.q̄

            # Compute the value function expectation
            EV = 0.0
            for jj in 1:P.sGridSize
                EV += V0Interpol(DSS.bPolicy[idx], jj) * P.Ω[i_s, jj]
            end

            # Update the value function
            if P.σ == 1.0
                V[idx] = ξ * (log(cons[idx]) - P.χ * DSS.H^(1+P.ν)/(1+P.ν)) + P.β * q * EV
            else
                V[idx] = ξ * (1/(1-P.σ) * (cons[idx])^(1-P.σ) - P.χ * DSS.H^(1+P.ν)/(1+P.ν)) + P.β * q * EV
            end

        end

        # Compute the distance between the current and previous iteration
        @. VError = abs(V - V0)
        dist = maximum(VError)

        # Update progress indicator
        if P.showPolicyIterations
            ProgressMeter.next!(p; showvalues = [(:iter, iter), (:dist, dist), (:distAlt, mean(VError.^2))], valuecolor = :grey)
        end

        # Prepare the next iteration
        V0 .= V
        V0Interpol = linear_interpolation((P.bGrid, 1:P.sGridSize), V0, extrapolation_bc = Line())
        iter += 1

    end

    # Display final iteration
    if !P.showPolicyIterations
        println(iter, " (Distance: ", dist, ")")
    else
        ProgressMeter.finish!(p)
    end

    return V

end


"""
    computeConsumptionPolicy(P, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol)

Computes the consumption policy for given bond policy and aggregate policies.

"""
function computeConsumptionPolicy(P, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol)

    _, wealth, _, _, _, _, _, _ = precomputeStaticPartsIndividualProblem(P, DSS, πwALMInterpol, EπwCondALMInterpol)

    cPolicy = @. wealth - bPolicy

    return cPolicy

end


"""
    computeConsumptionPolicyDSS(P, DSS)

Computes the consumption policy function for the DSS.

"""
function computeConsumptionPolicyDSS(P, DSS)

    # Compute wealth, consumption and labor for different grid points in the DSS
    wealth = zeros(P.bGridSize, P.sGridSize)
    cPolicy = similar(wealth)

    for idx in CartesianIndices(wealth)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_s = idx[2]

        #
        b = P.bGrid[i_b]
        s = P.sGrid[i_s]

        # From the budget constraint
        wealth[idx] = computeIndividualCashOnHand(P, DSS, b, s)
        cPolicy[idx] = wealth[idx] - DSS.bPolicy[idx]

    end

    return cPolicy

end


"""
    computeMPCs(P, DSS, cPolicyInterpol, πwALMInterpol, EπwCondALMInterpol; Δ = 0.0001, MPCOutOfBonds = false)

Computes the MPCs for given individual and aggregate policies.

"""
function computeMPCs(P, DSS, cPolicyInterpol, πwALMInterpol, EπwCondALMInterpol; Δ = 0.0001, MPCOutOfBonds = false)

    # Initialize dummy for the bond distribution
    # (reuqired by computeAllAggregateVariables but not used for computing
    # variables required for this function)
    bCrossDummy = zeros(P.bDenseGridSize, P.sGridSize)

    # Initialize the MPC matrix
    MPCs = zeros(P.bGridSize, P.RGridSize, P.sGridSize, P.ζGridSize)

    @threads for idx in CartesianIndices(MPCs)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_R = idx[2]
        i_s = idx[3]
        i_ζ = idx[4]

        # Get states
        b = P.bGrid[i_b]
        RStar = P.RGrid[i_R]
        s = P.sGrid[i_s]
        ζ = P.ζGrid[i_ζ]

        # Compute the real rate implied by the states
        aggVars = computeAllAggregateVariables(P, DSS, RStar, ζ, bCrossDummy, πwALMInterpol, EπwCondALMInterpol)
        r = aggVars.R / aggVars.π

        # Compute the marginal propensity to consume (MPC)
        if MPCOutOfBonds # MPC when given additional units of wealth b (which yields interest): c + b' = r * (b+Δ) + w * s * h + Pi - T
            MPCs[idx] = (cPolicyInterpol(b + Δ, RStar, i_s, ζ) - cPolicyInterpol(b, RStar, i_s, ζ)) / Δ
        else # MPC when given additional units of income: c + b' = r * b + Δ + w * s * h + Pi - T
            MPCs[idx] = (cPolicyInterpol(b + Δ/r, RStar, i_s, ζ) - cPolicyInterpol(b, RStar, i_s, ζ)) / Δ
        end


    end

    return MPCs

end


"""
    computeDSSMPCs(P, DSS; Δ = 0.0001, MPCOutOfBonds = false)

Computes the MPCs in the DSS.

"""
function computeDSSMPCs(P, DSS; Δ = 0.0001, MPCOutOfBonds = false)

    # Get consumption policy function
    cPolicy = computeConsumptionPolicyDSS(P, DSS)
    cPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), cPolicy, extrapolation_bc = Line())

    # Initialize the MPC matrix
    MPCs = zeros(P.bGridSize, P.sGridSize)

    @threads for idx in CartesianIndices(MPCs)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_s = idx[2]

        # Get states
        b = P.bGrid[i_b]
        s = P.sGrid[i_s]

        # Compute the real rate
        r = DSS.R / DSS.π

        # Compute the marginal propensity to consume (MPC)
        if MPCOutOfBonds # MPC when given additional units of wealth b (which yields interest): c + b' = r * (b+Δ) + w * s * h + Pi - T
            MPCs[idx] = (cPolicyInterpol(b + Δ, i_s) - cPolicyInterpol(b, i_s)) / Δ
        else # MPC when given additional units of income: c + b' = r * b + Δ + w * s * h + Pi - T
            MPCs[idx] = (cPolicyInterpol(b + Δ/r, i_s) - cPolicyInterpol(b, i_s)) / Δ
        end

    end

    return MPCs

end


"""
    getWealthShare(wealthShareType, perc, lorenzCurve, percentiles)

Compute wealth share.

"""
function getWealthShare(wealthShareType, perc, lorenzCurve, percentiles)

    if (1.0-perc) in percentiles && wealthShareType == :top

        wealthShare = 1 - lorenzCurve[percentiles .== 1.0-perc][1]

    elseif perc in percentiles && wealthShareType == :bottom

        wealthShare = lorenzCurve[percentiles .== perc][1]

    else

        wealthShare = NaN

    end

    return wealthShare

end


"""
    loadAllResults(filenames, inputFolder; loadSimulationResults = false, loadSimulationPlusResults = false)

Loads results for all models defined by "filenames".

"""
function loadAllResults(filenames, inputFolder; loadSimulationResults = false, loadSimulationPlusResults = false)

    res = Dict()

    for filename in filenames

        # Parse main solution file
        resMainParsed = BSON.parse(filename)

        # Check whether this is a RANK solution file
        if haskey(resMainParsed, :SSS)
            filenameExtRANK = resMainParsed[:P][:data][end-3] # Note: if P changes this might retrieve the wrong string
            filenamePrefixRANK = resMainParsed[:P][:data][end-1]
            delete!(resMainParsed, :P) # The RANK settings are not compatible with HANK settings
            delete!(resMainParsed, :VInterpol) # Old files with VInterpol are not loaded correctly
        end

        # Save the parsed file to the IO buffer
        io = IOBuffer()
        BSON.bson(io, resMainParsed)
        seek(io, 0)

        # Load the solution file normally
        resMain = BSON.load(io, @__MODULE__)

        # Check if these are the results for RANK
        if haskey(resMain, :SSS)
            res["$(filenamePrefixRANK)_$(filenameExtRANK)"] = resMain
            continue
        end

        # Load prepared results
        resSSS = BSON.load("$(inputFolder)/SSS_$(resMain[:P].filenameExt).bson", @__MODULE__)
        resV = BSON.load("$(inputFolder)/ValueFunction_$(resMain[:P].filenameExt).bson", @__MODULE__)
        resVDSS = BSON.load("$(inputFolder)/ValueFunctionDSS_$(resMain[:P].filenameExt).bson", @__MODULE__)
        resIRFs = BSON.load("$(inputFolder)/IRFs_$(resMain[:P].filenameExt).bson", @__MODULE__)

        if loadSimulationResults
            resSim = BSON.load("$(inputFolder)/Simulation_$(resMain[:P].filenameExt).bson", @__MODULE__)
        end

        if loadSimulationPlusResults
            if isfile("$(inputFolder)/SimulationPlus_$(resMain[:P].filenameExt).bson")
                resSimPlus = BSON.load("$(inputFolder)/SimulationPlus_$(resMain[:P].filenameExt).bson", @__MODULE__)
            else
                @warn "$(inputFolder)/SimulationPlus_$(resMain[:P].filenameExt).bson not found"
            end
        end

        # Interpolate the value functions
        P = resMain[:P]
        resV[:VInterpol] = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), resV[:V], extrapolation_bc = Line())
        resVDSS[:VDSSInterpol] = linear_interpolation((P.bGrid, 1:P.sGridSize), resVDSS[:VDSS], extrapolation_bc = Line())

        # Interpolate bond policy functions
        resMain[:bPolicyInterpol] = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), resMain[:bPolicy], extrapolation_bc = Line())
        resMain[:bPolicyDSSInterpol] = linear_interpolation((P.bGrid, 1:P.sGridSize), resMain[:DSS].bPolicy, extrapolation_bc = Line())

        # Interpolate ALMs
        resMain[:πwALMInterpol] = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), resMain[:πwALM], extrapolation_bc = Line())
        resMain[:EπwCondALMInterpol] = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), resMain[:EπwCondALM], extrapolation_bc = Line())

        # Compute consumption policy
        resMain[:cPolicy] = computeConsumptionPolicy(P, resMain[:DSS], resMain[:bPolicy], resMain[:πwALMInterpol], resMain[:EπwCondALMInterpol])
        resMain[:cPolicyInterpol] = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), resMain[:cPolicy], extrapolation_bc = Line())
        resMain[:cPolicyDSS] = computeConsumptionPolicyDSS(P, resMain[:DSS])
        resMain[:cPolicyDSSInterpol] = linear_interpolation((P.bGrid, 1:P.sGridSize), resMain[:cPolicyDSS], extrapolation_bc = Line())

        if loadSimulationResults
            resMain[:simSeries] = resSim[:simSeries]
        end

        if loadSimulationPlusResults && isfile("$(inputFolder)/SimulationPlus_$(resMain[:P].filenameExt).bson")
            resMain[:simSeriesPlus] = resSimPlus[:simSeries]
        end

        # Combine the disctionaries
        res["$(resMain[:P].filenameExt)"] = merge(resSSS, resV, resVDSS, resIRFs, resMain)

    end

    return res

end


"""
    updateResults()

Function to be used to update old bson-files whenever the settings struct changes format. 
Otherwise, old results cannot be loaded by loadAllResults().

"""
function updateResults()

    # Settings
    inputFolder = "Results/HANKWageRigidities/Results/"

    for file in readdir(inputFolder)

        if file in [".DS_Store", "tmp", "Calibration.txt", "DSSEvaluator.csv", "Results"]
            continue
        end

        #=if !occursin("FixedNNInitializationSeedRetry", file) && !occursin("FixedNNInitializationSeedNewShocks", file)
            continue
        end=#

        res = BSON.parse("$(inputFolder)/$(file)")
        writeToFile = false

        if haskey(res, :P)

            println(file)

            # Insert the additional parameter
            insert!(res[:P][:data], 12, false) # :useRStarForTaylorInertia
            insert!(res[:P][:data], 13, true) # :useLegacyRStar
            insert!(res[:P][:data], 29, 1.0) # :q̄
            insert!(res[:P][:data], 30, :Preference) # :aggShockType
            insert!(res[:P][:data], 31, 1.0) # :ζ̄
            deleteat!(res[:P][:data], 34) # remove :s̄
            insert!(res[:P][:data], 91, res[:P][:data][89]) # :profitShares
   

            #for ii in 87:length(res[:P][:data])
            #    println(res[:P][:data][ii])
            #end

            # Set write flag
            writeToFile = true

        end

        #=if haskey(res, :NNπw) && res[:NNπw][:tag] != "array"

            println(file)

            # Insert the additional parameter
            insert!(res[:NNπw][:data], 10, :softplus) # :activationFunction
            insert!(res[:NNEπwCond][:data], 10, :softplus) # :activationFunction

            # Set write flag
            writeToFile = true

        end=#

        # Save the parsed file
        #=if writeToFile
            println("Updating BSON file")
            BSON.bson("$(inputFolder)/$(file)", res)
        end=#

    end

end


"""
    computeLorenzCurve(P, B, bCross; stepSize = 0.01, giniType = :standard)

Computes the Lorenz curve and the Gini coefficient.

"""
function computeLorenzCurve(P, B, bCross; stepSize = 0.01, giniType = :standard)

    # Lorenz curve
    percentiles = floor(sum(bCross, dims = 2)[1] + stepSize, digits = 2):stepSize:1.0
    wealthPoorestIndicator = cumsum(vec(sum(bCross, dims = 2))) .<= percentiles'
    weightedWealth = sum(bCross .* P.bDenseGrid, dims = 2)
    lorenzCurve = [sum(weightedWealth[wealthPoorestIndicator[:,ii]]) / B for ii in 1:size(wealthPoorestIndicator,2)]

    # Gini coefficient
    if giniType == :standardLorenz

        A = sum((percentiles - lorenzCurve) * stepSize)
        B = sum(lorenzCurve * stepSize)
        giniCoeff = A / (A + B)

    elseif giniType == :standard

        giniCoeff = computeGiniCoefficient(vec(sum(bCross, dims = 2)), P.bDenseGrid)

    end

    return lorenzCurve, giniCoeff, percentiles

end


"""
    computeGiniCoefficient(dist, y)

Provides an alternative way to compute the Gini coefficient which is not dependent on the Lorenz curve.

"""
function computeGiniCoefficient(dist, y)

    # Compute mean
    μ = dot(dist, y)

    # Compute Gini coefficient
    G = 0.0

    for ii in 1:length(y), jj in 1:length(y)
        G += dist[ii] * dist[jj] *  abs(y[ii] - y[jj])
    end

    G = G / (2 * μ)

    return G

end


"""
    showDistributionStatistics(hist, grid)

Computes different statistics for given distribution.

"""
function showDistributionStatistics(hist, grid)

    # Compute desired statistics of the distribution of the distribution
    distMean = dot(hist, grid)
    distMedian = computePercentile(hist, grid, 0.5)
    dist99Perc = computePercentile(hist, grid, 0.99)
    dist90Perc = computePercentile(hist, grid, 0.90)
    dist30Perc = computePercentile(hist, grid, 0.30)
    distVariance = dot(hist, (grid .- distMean).^2)
    distGini = computeGiniCoefficient(hist, grid)

    if all(grid .>= 0.0)
        distMeanLog = dot(hist, log.(grid))
        distVarianceLog = dot(hist, (log.(grid) .- distMeanLog).^2)
    else
        distMeanLog = NaN
        distVarianceLog = NaN
    end

    # Print all statistics
    println("Mean: ", distMean)
    println("Median: ", distMedian)
    println("Mean to Median: ", distMean / distMedian)
    println("Coefficient of Variation: ", distVariance / distMean)
    println("Variance of logs: ", distVarianceLog)
    println("Gini index: ", distGini)
    println("99-50 Ratio: ", dist99Perc / distMedian)
    println("90-50 Ratio: ", dist90Perc / distMedian)
    println("50-30 Ratio: ", distMedian / dist30Perc)

    nothing

end
