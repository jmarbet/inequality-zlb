"""
    updateπwALM!(P, coeff::Array{Float64,1}, πwALMUpate, πwALM,  πwSim, RStarSim, ζSim)

Updates the ALM for inflation using a linear regression.

"""
function updateπwALM!(P, coeff::Array{Float64,1}, πwALMUpate, πwALM, πwSim, RStarSim, ζSim)

    # Remove burn-in periods
    πwSim = πwSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]

    # Prepare dependent and exlanatory variables
    y = log.(πwSim)
    x1 = log.(RStarSim)
    x2 = log.(ζSim)
    x3 = [log(RStarSim[t]) * log(ζSim[t]) for t in 1:length(πwSim)]
    X = [ones(size(y,1),1) x1 x2 x3]

    # Estimate new coefficents of the ALM
    newCoeff = (X'*X) \ X' * y
    coeff .= newCoeff

    # Evaluate the regression model on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]

        # Get the state variables for the current node in the dense state space
        RStar = P.RDenseGrid[i_R]
        ζ = P.ζDenseGrid[i_ζ]

        ALM[idx] = exp(dot(newCoeff, [1.0, log(RStar), log(ζ), log(RStar)*log(ζ)]))

    end

    # Update the ALM
    @. πwALMUpate = P.λALM * ALM + (1-P.λALM) * πwALM

    # Compute R^2 and MSE
    ȳ = mean(y)
    yFit = X * newCoeff
    R2 = sum((yFit .- ȳ).^2) / sum((y .- ȳ).^2)
    MSE = mean((yFit .- ȳ).^2)

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim)
    dist = computeALMDistance(πwALM, πwALMUpate, visitedNodes)

    # Get size of training sample
    nObs = length(y)

    # Generate named tuple with all statistics
    trainingStats = (R2 = R2, MSE = MSE, nObs = nObs, loss = NaN)
    validationStats = trainingStats
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    # Required to be consistent with the NN function used for updating the PLM
    dummyDataset = (inputs = [], outputs = [])

    return stats, ALM, dummyDataset, dummyDataset

end


"""
    updateEπwCondALM!(P, coeff::Array{Float64,1}, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

Updates the ALM for the term related to inflation expectations using a linear regression.

"""
function updateEπwCondALM!(P, coeff::Array{Float64,1}, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

    # Remove burn-in periods
    EπwCondSim = EπwCondSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]
    ζPrimeSim = ζPrimeSim[P.burnIn+1:end]

    # Prepare dependent and exlanatory variables
    y = EπwCondSim
    x1 = log.(RStarSim)
    x2 = log.(ζSim)
    x3 = log.(ζPrimeSim)
    x4 = [log(RStarSim[t]) * log(ζSim[t]) for t in 1:length(EπwCondSim)]
    x5 = [log(RStarSim[t]) * log(ζPrimeSim[t]) for t in 1:length(EπwCondSim)]
    x6 = [log(ζSim[t]) * log(ζPrimeSim[t]) for t in 1:length(EπwCondSim)]
    X = [ones(size(y,1),1) x1 x2 x3 x4 x5 x6]

    # Estimate new coefficents of the ALM
    newCoeff = (X'*X) \ X' * y
    coeff .= newCoeff

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

        ALM[idx] = dot(newCoeff, [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])

    end

    # Update the ALM
    @. EπwCondALMUpdate = P.λALM * ALM + (1-P.λALM) * EπwCondALM

    # Compute R^2 and MSE
    ȳ = mean(y)
    yFit = X * newCoeff
    R2 = sum((yFit .- ȳ).^2) / sum((y .- ȳ).^2)
    MSE = mean((yFit .- ȳ).^2)

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim, ζPrimeSim)
    dist = computeALMDistance(EπwCondALM, EπwCondALMUpdate, visitedNodes)

    # Get size of training sample
    nObs = length(y)

    # Generate named tuple with all statistics
    trainingStats = (R2 = R2, MSE = MSE, nObs = nObs, loss = NaN)
    validationStats = trainingStats
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    # Required to be consistent with the NN function used for updating the PLM
    dummyDataset = (inputs = [], outputs = [])

    return stats, ALM, dummyDataset, dummyDataset

end
