"""
    _updateπwALM!(P, coeff::Array{Float64,2}, πwALMUpate, πwALM,  πwSim, RStarSim, ζSim, DSS, EπwCondALM, ::Val{:QuadRegression})

Updates the ALM for inflation using linear regressions for ZLB and non-ZLB periods.

"""
function _updateπwALM!(P, coeff::Array{Float64,2}, πwALMUpate, πwALM, πwSim, RStarSim, ζSim, DSS, EπwCondALM, ::Val{:QuadRegression})

    return _updateπwALM!(P, coeff, πwALMUpate, πwALM, πwSim, RStarSim, ζSim, DSS, EπwCondALM, Val(:DualRegression))

end


"""
    _updateEπwCondALM!(P, coeff::Array{Float64,2}, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM, ::Val{:QuadRegression})

Updates the ALM for the term related to inflation expectations using linear regressions for ZLB and non-ZLB periods.

"""
function _updateEπwCondALM!(P, coeff::Array{Float64,2}, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim, DSS, πwALM, ::Val{:QuadRegression})

    # Determine ZLB periods
    ZLBIndSim = RStarSim .<= P.ZLBLevel
    ZLBIndSim = [ZLBIndSim[2:end]; false] # Shift by one period to indicate when ZLB is binding at t

    # Remove burn-in periods
    EπwCondSim = EπwCondSim[P.burnIn+1:end-1]
    RStarSim = RStarSim[P.burnIn+1:end-1]
    ζSim = ζSim[P.burnIn+1:end-1]
    ζPrimeSim = ζPrimeSim[P.burnIn+1:end-1]
    ZLBIndSim = ZLBIndSim[P.burnIn+1:end-1]

    # Run regressions and save statistics and coefficients 
    MSEs = zeros(4)
    R2s = zeros(4)
    nObs = zeros(4)

    for ii in 1:4

        # Initialize dependent and explanatory variables
        y = Float64[]
        x1 = Float64[]
        x2 = Float64[]
        x3 = Float64[]
        x4 = Float64[]
        x5 = Float64[]
        x6 = Float64[]

        for tt in 1:length(EπwCondSim)-1

            if ii == 1 # ZLB binding in neither period
                inludeInRegression = !ZLBIndSim[tt] && !ZLBIndSim[tt+1]
            elseif ii == 2 # ZLB binding in period t
                inludeInRegression = ZLBIndSim[tt] && !ZLBIndSim[tt+1]
            elseif ii == 3 # ZLB binding in period t+1
                inludeInRegression = !ZLBIndSim[tt] && ZLBIndSim[tt+1]
            elseif ii == 4 # ZLB binding in both periods
                inludeInRegression = ZLBIndSim[tt] && ZLBIndSim[tt+1]
            end

            if inludeInRegression
                push!(y, EπwCondSim[tt])
                push!(x1, log(RStarSim[tt]))
                push!(x2, log(ζSim[tt]))
                push!(x3, log(ζPrimeSim[tt]))
                push!(x4, log(RStarSim[tt]) * log(ζSim[tt]))
                push!(x5, log(RStarSim[tt]) * log(ζPrimeSim[tt]))
                push!(x6, log(ζSim[tt]) * log(ζPrimeSim[tt]))
            end

        end

        # Combine explanatory variables
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

        if RStarPrime <= P.ZLBLevel && RStarPrimePrime <= P.ZLBLevel
            ALM[idx] = dot(coeff[:, 4], [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])
        elseif RStarPrime > P.ZLBLevel && RStarPrimePrime <= P.ZLBLevel
            ALM[idx] = dot(coeff[:, 3], [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])
        elseif RStarPrime <= P.ZLBLevel && RStarPrimePrime > P.ZLBLevel
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
    trainingStats = (R2 = NamedTuple{(:NoZLBBoth, :ZLBToday, :ZLBTomorrow, :ZLBBoth)}(R2s), MSE = NamedTuple{(:NoZLBBoth, :ZLBToday, :ZLBTomorrow, :ZLBBoth)}(MSEs), nObs = NamedTuple{(:NoZLBBoth, :ZLBToday, :ZLBTomorrow, :ZLBBoth)}(nObs), loss = NaN)
    validationStats = trainingStats
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    # Required to be consistent with the NN function used for updating the PLM
    dummyDataset = (inputs = [], outputs = [])

    return stats, ALM, dummyDataset, dummyDataset

end
