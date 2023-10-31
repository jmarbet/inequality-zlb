"""
    updateπwALM!(P, coeff::Array{Float64,1}, πwALMUpate, πwALM,  πwSim, RStarSim, ζSim, DSS, EπwCondALM)

Updates the ALM for inflation using linear regressions for ZLB and non-ZLB periods.

"""
function updateπwALM!(P, coeff::Array{Float64,2}, πwALMUpate, πwALM, πwSim, RStarSim, ζSim, DSS, EπwCondALM)

    return _updateπwALM!(P, coeff, πwALMUpate, πwALM, πwSim, RStarSim, ζSim, DSS, EπwCondALM, Val(P.approximationTypeALM))

end


"""
    updateEπwCondALM!(P, coeff::Array{Float64,1}, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM)

Updates the ALM for the term related to inflation expectations using linear regressions for ZLB and non-ZLB periods.

"""
function updateEπwCondALM!(P, coeff::Array{Float64,2}, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM)

    return _updateEπwCondALM!(P, coeff, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM, Val(P.approximationTypeALM))

end


"""
    _updateπwALM!(P, coeff::Array{Float64,1}, πwALMUpate, πwALM,  πwSim, RStarSim, ζSim, DSS, EπwCondALM, ::Val{:DualRegression})

Updates the ALM for inflation using linear regressions for ZLB and non-ZLB periods.

"""
function _updateπwALM!(P, coeff::Array{Float64,2}, πwALMUpate, πwALM, πwSim, RStarSim, ζSim, DSS, EπwCondALM, ::Val{:DualRegression})

    # Determine ZLB periods
    ZLBIndSim = RStarSim .<= P.ZLBLevel
    ZLBIndSim = [ZLBIndSim[2:end]; false] # Shift by one period to corectly align with πSim

    # Remove burn-in periods
    πwSim = πwSim[P.burnIn+1:end-1]
    RStarSim = RStarSim[P.burnIn+1:end-1]
    ζSim = ζSim[P.burnIn+1:end-1]
    ZLBIndSim = ZLBIndSim[P.burnIn+1:end-1]

    # Run regressions and save statistics and coefficients
    MSEs = zeros(2)
    R2s = zeros(2)
    nObs = zeros(2)

    for ii in 1:2

        if ii == 1
            sel = .!ZLBIndSim # Non-ZLB periods
        else
            sel = ZLBIndSim # ZLB periods
        end

        # Prepare dependent and explanatory variables
        y = log.(πwSim[sel])
        x1 = log.(RStarSim[sel])
        x2 = log.(ζSim[sel])
        x3 = [log(RStarSim[sel][t]) * log(ζSim[sel][t]) for t in 1:length(πwSim[sel])]
        X = [ones(size(y,1),1) x1 x2 x3]

        # Estimate new coefficents of the ALM
        newCoeff = (X'*X) \ X' * y
        coeff[:, ii] .= newCoeff

        # Compute R^2 and MSE
        ȳ = mean(y)
        yFit = X * coeff[:, ii]
        R2s[ii] = sum((yFit .- ȳ).^2) / sum((y .- ȳ).^2)
        MSEs[ii] = mean((yFit .- ȳ).^2)
        nObs[ii] = length(y)

    end

    # Interpolate old ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    # Evaluate the regression model on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]

        # Get the state variables for the current node in the dense state space
        RStar = P.RDenseGrid[i_R]
        ζ = P.ζDenseGrid[i_ζ]

        # Use different ALM depending on whether the economy is in a ZLB period or not
        _, _, _, _, _, _, _, RStarPrime = computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol)

        if RStarPrime <= P.ZLBLevel
            ALM[idx] = exp(dot(coeff[:, 2], [1.0, log(RStar), log(ζ), log(RStar)*log(ζ)]))
        else
            ALM[idx] = exp(dot(coeff[:, 1], [1.0, log(RStar), log(ζ), log(RStar)*log(ζ)]))
        end
        
    end

    # Update the ALM
    @. πwALMUpate = P.λALM * ALM + (1-P.λALM) * πwALM

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim)
    dist = computeALMDistance(πwALM, πwALMUpate, visitedNodes)

    # Generate named tuple with all statistics
    trainingStats = (R2 = NamedTuple{(:NoZLB, :ZLB)}(R2s), MSE = NamedTuple{(:NoZLB, :ZLB)}(MSEs), nObs = NamedTuple{(:NoZLB, :ZLB)}(nObs), loss = NaN)
    validationStats = trainingStats
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    # Required to be consistent with the NN function used for updating the PLM
    dummyDataset = (inputs = [], outputs = [])

    return stats, ALM, dummyDataset, dummyDataset

end


"""
    _updateEπwCondALM!(P, coeff::Array{Float64,1}, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM, ::Val{:DualRegression})

Updates the ALM for the term related to inflation expectations using linear regressions for ZLB and non-ZLB periods.

"""
function _updateEπwCondALM!(P, coeff::Array{Float64,2}, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM, ::Val{:DualRegression})

    # Determine ZLB periods
    ZLBIndSim = RStarSim .<= P.ZLBLevel
    ZLBIndSim = [ZLBIndSim[3:end]; false; false] # Shift by two period to indicate when ZLB is binding at t+1

    # Remove burn-in periods
    EπwCondSim = EπwCondSim[P.burnIn+1:end-2]
    RStarSim = RStarSim[P.burnIn+1:end-2]
    ζSim = ζSim[P.burnIn+1:end-2]
    ζPrimeSim = ζPrimeSim[P.burnIn+1:end-2]
    ZLBIndSim = ZLBIndSim[P.burnIn+1:end-2]

    # Run regressions and save statistics and coefficients 
    MSEs = zeros(2)
    R2s = zeros(2)
    nObs = zeros(2)

    for ii in 1:2

        if ii == 1
            sel = .!ZLBIndSim # Non-ZLB periods
        else
            sel = ZLBIndSim # ZLB periods
        end

        # Prepare dependent and explanatory variables
        y = EπwCondSim[sel]
        x1 = log.(RStarSim[sel])
        x2 = log.(ζSim[sel])
        x3 = log.(ζPrimeSim[sel])
        x4 = [log(RStarSim[sel][t]) * log(ζSim[sel][t]) for t in 1:length(EπwCondSim[sel])]
        x5 = [log(RStarSim[sel][t]) * log(ζPrimeSim[sel][t]) for t in 1:length(EπwCondSim[sel])]
        x6 = [log(ζSim[sel][t]) * log(ζPrimeSim[sel][t]) for t in 1:length(EπwCondSim[sel])]
        X = [ones(size(y,1),1) x1 x2 x3 x4 x5 x6]

        # Estimate new coefficents of the ALM
        newCoeff = (X'*X) \ X' * y
        coeff[:, ii] .= newCoeff

        # Compute R^2 and MSE
        ȳ = mean(y)
        yFit = X * coeff[:, ii]
        R2s[ii] = sum((yFit .- ȳ).^2) / sum((y .- ȳ).^2)
        MSEs[ii] = mean((yFit .- ȳ).^2)
        nObs[ii] = length(y)

    end

    # Interpolate old ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    # Evaluate the regression model on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]
        i_ζp = idx[3]

        # Get the state variables for the current node in the dense state space
        RStar = P.RDenseGrid[i_R]
        ζ = P.ζDenseGrid[i_ζ]
        ζPrime = P.ζDenseGrid[i_ζp]

        # Use different ALM depending on whether the economy is in a ZLB period (at t+1) or not
        _, _, _, _, _, _, _, RStarPrime = computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol)
        _, _, _, _, _, _, _, RStarPrimePrime = computeAggregateVariables(P, DSS, RStarPrime, ζPrime, πwALMInterpol, EπwCondALMInterpol)

        if RStarPrimePrime <= P.ZLBLevel
            ALM[idx] = dot(coeff[:, 2], [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])
        else
            ALM[idx] = dot(coeff[:, 1], [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])
        end

    end

    # Update the ALM
    @. EπwCondALMUpdate = P.λALM * ALM + (1-P.λALM) * EπwCondALM

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim, ζPrimeSim)
    dist = computeALMDistance(EπwCondALM, EπwCondALMUpdate, visitedNodes)

    # Generate named tuple with all statistics
    trainingStats = (R2 = NamedTuple{(:NoZLB, :ZLB)}(R2s), MSE = NamedTuple{(:NoZLB, :ZLB)}(MSEs), nObs = NamedTuple{(:NoZLB, :ZLB)}(nObs), loss = NaN)
    validationStats = trainingStats
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    # Required to be consistent with the NN function used for updating the PLM
    dummyDataset = (inputs = [], outputs = [])

    return stats, ALM, dummyDataset, dummyDataset

end
