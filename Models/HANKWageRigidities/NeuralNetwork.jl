"""
    NeuralNetwork()

Basic settings of the neural network.

"""
@with_kw mutable struct NeuralNetwork{R}

    # Number of nodes of input, output and hidden layer
    nInputs::Int64 = 2
    nHidden::Int64 = 10
    nOutputs::Int64 = 1

    # Weights
    w1::Array{Float64,2} = randn(nHidden, nInputs)
    w2::Array{Float64,2} = randn(nOutputs, nHidden)

    # Biases
    b1::Array{Float64,1} = zeros(nHidden)
    b2::Array{Float64,1} = zeros(nOutputs)

    # Regularization parameter
    λ::Float64 = 0.1

    # Base learning speed during gradient descent
    learningSpeed::Float64 = 0.01

    # Activation function
    activationFunction::Symbol = :softplus

    # Normalization of inputs and outputs
    normFactors::R

end


"""
    reinitializeNeuralNetwork!(P, NN::NeuralNetwork, NNType)

Reinitializes a given Neural Network.

"""
function reinitializeNeuralNetwork!(P, NN::NeuralNetwork, NNType)

    # Initialize new neural networks
    NNπw, NNEπwCond = initializeALMApproximation(P)

    # Update the weights of the given network
    if NNType == :πw
        NN.w1 .= NNπw.w1
        NN.w2 .= NNπw.w2
        NN.b1 .= NNπw.b1
        NN.b2 .= NNπw.b2
    elseif NNType == :EπwCond
        NN.w1 .= NNEπwCond.w1
        NN.w2 .=  NNEπwCond.w2
        NN.b1 .= NNEπwCond.b1
        NN.b2 .=  NNEπwCond.b2
    else
        error("Unknown NNType = $(NNType)")
    end

    nothing

end


"""
    prepareData(P, πwSim, RStarSim, ζSim)

Normalizes the data, projects it onto grid knots and puts it in the form
required by the neural network.

"""
function prepareData(P, πwSim, RStarSim, ζSim; normFactors = tuple(), disableProjection = false)

    # Initialize inputs and outputs
    outputs = Array{Float64,1}[]
    inputs = Array{Float64,1}[]

    if P.projectDataOntoGridKnots && !disableProjection

        for i_R in 1:P.RDataPrepGridSize, i_ζ in 1:P.ζDataPrepGridSize

            # Get the state variables for the current node in the dense state space
            RStar = P.RDataPrepGrid[i_R]
            ζ = P.ζDataPrepGrid[i_ζ]

            # Initialize matrices for regression
            Y = Float64[]
            X1 = Float64[]
            X2 = Float64[]

            # Step sizes
            RStarHalfStep = convert(Float64, P.RDataPrepGrid.step) / 2
            ζHalfStep = convert(Float64, P.ζDataPrepGrid.step) / 2

            # Assign simulated data to grid points
            for ii in 1:length(πwSim)
                if abs(RStarSim[ii] - RStar) < RStarHalfStep && abs(ζSim[ii] - ζ) < ζHalfStep
                    push!(Y, πwSim[ii])
                    push!(X1, RStarSim[ii])
                    push!(X2, ζSim[ii])
                end
            end

            # Estimate midpoint if enough data is available
            if length(Y) > 5

                # In some cases, X1 only contains 1s (i.e. binding ZLB in each case).
                # To avoid issues due to multicolinearity, we only regress Y on X2
                # (i.e. πw on ζ) in these cases.
                if length(unique(X1)) != 1

                    X =[ones(length(Y)) X1 X2]
                    β = (X'*X)\X'*Y
                    push!(outputs, [dot([1, RStar, ζ], β)])
                    push!(inputs, [RStar, ζ])

                else

                    X =[ones(length(Y)) X2]
                    β = (X'*X)\X'*Y
                    push!(outputs, [dot([1, ζ], β)])
                    push!(inputs, [RStar, ζ])

                end

            end

        end

        # Compute normalization if it is not supplied
        if isempty(normFactors)
            πw = getNormalization(P, [x[1] for x in outputs])
            RStar = getNormalization(P, [x[1] for x in inputs])
            ζ = getNormalization(P, [x[2] for x in inputs])
            normFactors = (πw = πw, RStar = RStar, ζ = ζ)
        end

        for ii in 1:length(outputs)

            outputs[ii][1] =  (outputs[ii][1] - normFactors.πw.location) / normFactors.πw.scale
            inputs[ii][1] =  (inputs[ii][1] - normFactors.RStar.location) / normFactors.RStar.scale
            inputs[ii][2] =  (inputs[ii][2] - normFactors.ζ.location) / normFactors.ζ.scale

        end

    else

        # Compute normalization if it is not supplied
        if isempty(normFactors)
            πw = getNormalization(P, πwSim)
            RStar = getNormalization(P, RStarSim)
            ζ = getNormalization(P, ζSim)
            normFactors = (πw = πw, RStar = RStar, ζ = ζ)
        end

        # Normalize data
        πwSim = (πwSim .- normFactors.πw.location) / normFactors.πw.scale
        RStarSim = (RStarSim .- normFactors.RStar.location) / normFactors.RStar.scale
        ζSim = (ζSim .- normFactors.ζ.location) / normFactors.ζ.scale

        # Construct input/output matrices
        for ii in 1:length(πwSim)
            push!(outputs, [πwSim[ii]])
            push!(inputs, [RStarSim[ii], ζSim[ii]])
        end

    end

    return outputs, inputs, normFactors

end


"""
    prepareData(P, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

Normalizes the data, projects it onto grid knots and puts it in the form
required by the neural network.

"""
function prepareData(P, EπwCondSim, RStarSim, ζSim, ζPrimeSim; normFactors = tuple(), disableProjection = false)

    # Initialize inputs and outputs
    outputs = Array{Float64,1}[]
    inputs = Array{Float64,1}[]

    if P.projectDataOntoGridKnots && !disableProjection

        for i_R in 1:P.RDataPrepGridSize, i_ζ in 1:P.ζDataPrepGridSize, i_ζp in 1:P.ζDataPrepGridSize

            # Get the state variables for the current node in the dense state space
            RStar = P.RDataPrepGrid[i_R]
            ζ = P.ζDataPrepGrid[i_ζ]
            ζp = P.ζDataPrepGrid[i_ζp]

            # Initialize matrices for regression
            Y = Float64[]
            X1 = Float64[]
            X2 = Float64[]
            X3 = Float64[]

            # Step sizes
            RStarHalfStep = convert(Float64, P.RDataPrepGrid.step) / 2
            ζHalfStep = convert(Float64, P.ζDataPrepGrid.step) / 2

            # Assign simulated data to grid points
            for ii in 1:length(EπwCondSim)
                if abs(RStarSim[ii] - RStar) < RStarHalfStep && abs(ζSim[ii] - ζ) < ζHalfStep &&
                   abs(ζPrimeSim[ii] - ζp) < ζHalfStep
                    push!(Y, EπwCondSim[ii])
                    push!(X1, RStarSim[ii])
                    push!(X2, ζSim[ii])
                    push!(X3, ζPrimeSim[ii])
                end
            end

            # Estimate midpoint if enough data is available
            if length(Y) > 5

                # In some cases, X1 only contains 1s (i.e. binding ZLB in each case).
                # To avoid issues due to multicolinearity, we only regress Y on X2 and X3
                # (i.e. πw on ζ and ζp) in these cases.
                if length(unique(X1)) != 1

                    X =[ones(length(Y)) X1 X2 X3]
                    β = (X'*X)\X'*Y
                    push!(outputs, [dot([1, RStar, ζ, ζp], β)])
                    push!(inputs, [RStar, ζ, ζp])

                else

                    X =[ones(length(Y)) X2 X3]
                    β = (X'*X)\X'*Y
                    push!(outputs, [dot([1, ζ, ζp], β)])
                    push!(inputs, [RStar, ζ, ζp])

                end

            end

        end

        # Compute normalization if it is not supplied
        if isempty(normFactors)
            EπwCond = getNormalization(P, EπwCondSim)
            RStar = getNormalization(P, RStarSim)
            ζ = getNormalization(P, ζSim)
            ζPrime = getNormalization(P, ζPrimeSim)
            normFactors = (EπwCond = EπwCond, RStar = RStar, ζ = ζ, ζPrime = ζPrime)
        end

        for ii in 1:length(outputs)

            outputs[ii][1] =  (outputs[ii][1] - normFactors.EπwCond.location) / normFactors.EπwCond.scale
            inputs[ii][1] =  (inputs[ii][1] - normFactors.RStar.location) / normFactors.RStar.scale
            inputs[ii][2] =  (inputs[ii][2] - normFactors.ζ.location) / normFactors.ζ.scale
            inputs[ii][3] =  (inputs[ii][3] - normFactors.ζPrime.location) / normFactors.ζPrime.scale

        end

    else

        # Compute normalization if it is not supplied
        if isempty(normFactors)
            EπwCond = getNormalization(P, EπwCondSim)
            RStar = getNormalization(P, RStarSim)
            ζ = getNormalization(P, ζSim)
            ζPrime = getNormalization(P, ζPrimeSim)
            normFactors = (EπwCond = EπwCond, RStar = RStar, ζ = ζ, ζPrime = ζPrime)
        end

        # Normalize data
        EπwCondSim = (EπwCondSim .- normFactors.EπwCond.location) / normFactors.EπwCond.scale
        RStarSim = (RStarSim .- normFactors.RStar.location) / normFactors.RStar.scale
        ζSim = (ζSim .- normFactors.ζ.location) / normFactors.ζ.scale
        ζPrimeSim = (ζPrimeSim .- normFactors.ζPrime.location) / normFactors.ζPrime.scale

        # Construct input/output matrices
        for ii in 1:length(EπwCondSim)
            push!(outputs, [EπwCondSim[ii]])
            push!(inputs, [RStarSim[ii], ζSim[ii], ζPrimeSim[ii]])
        end

    end

    return outputs, inputs, normFactors

end


"""
    getNormalization(P, sim)

Returns tuple with location and scale factor that are used for normalization.

"""
function getNormalization(P, sim)

    if P.IONormalization == :minMax01

        normFactors = (location = minimum(sim), scale = maximum(sim)-minimum(sim))

    elseif P.IONormalization == :minMax11

        normFactors = (location = (minimum(sim)+maximum(sim))/2 , scale = (maximum(sim)-minimum(sim))/2)

    elseif P.IONormalization == :minMax44

        normFactors = (location = (minimum(sim)+maximum(sim))/2 , scale = (1/4) * (maximum(sim)-minimum(sim))/2)

    else

        normFactors = (location = mean(sim), scale = std(sim))

    end

    return normFactors

end


"""
    activation(NN, x)

Activation function used by the Neural Network.

"""
function activation(NN, x)

    if NN.activationFunction == :softplus
        return log(1+exp(-abs(x))) + max(x, 0.0)    # Formula adjusted for numerical stability. 
                                                    # Note: log(1+exp(x)) = log(1+exp(x)) - log(exp(x)) + x = log(1+exp(-x)) + x
    elseif NN.activationFunction == :relu
        return (x < 0.0) ? 0.0 : x
    elseif NN.activationFunction == :sigmoid
        return 1/(1+exp(-x))
    end

end


"""
    activationPrime(NN, x)

Derivative of activation function used by the Neural Network.

"""
function activationPrime(NN, x)

    if NN.activationFunction == :softplus
        return 1/(1+exp(-x))
    elseif NN.activationFunction == :relu
        return (x < 0.0) ? 0.0 : 1
    elseif NN.activationFunction == :sigmoid
        return (1/(1+exp(-x))) * (1 - 1/(1+exp(-x)))
    end

end


"""
    updateπwALM!(P, NN::NeuralNetwork, πwALMUpdate, πwALM,  πwSim, RStarSim, ζSim)

Updates the ALM for inflation using a Neural Network.

"""
function updateπwALM!(P, NN::NeuralNetwork, πwALMUpdate, πwALM,  πwSim, RStarSim, ζSim)

    traininingSuccessful = false
    local stats, ALM, trainingData, validationData

    while !traininingSuccessful

        # Try to update the ALM 
        stats, ALM, trainingData, validationData = _updateπwALM!(P, NN, πwALMUpdate, πwALM,  πwSim, RStarSim, ζSim)

        # If there are issues during training (e.g. gradient explodes), reinitialize the neural network and try again
        if isnan(stats.training.R2)
            @warn "Updating ALM (πw) failed. Reinitializing neural network weights..."
            reinitializeNeuralNetwork!(P, NN, :πw)
        else
            traininingSuccessful = true
        end

    end

    return stats, ALM, trainingData, validationData

end


"""
    _updateπwALM!(P, NN::NeuralNetwork, πwALMUpdate, πwALM,  πwSim, RStarSim, ζSim)

Updates the ALM for wage inflation using a Neural Network.

"""
function _updateπwALM!(P, NN::NeuralNetwork, πwALMUpdate, πwALM,  πwSim, RStarSim, ζSim)

    # Initialize the gradient struct (this will be reused during gradient descent)
    NNGradient = NeuralNetworkGradient(NN)

    # Remove burn-in periods
    πwSim = πwSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]

    # Preprocess the training data
    outputs, inputs, NN.normFactors = prepareData(P, πwSim, RStarSim, ζSim)
    trainingData = (inputs = inputs, outputs = outputs, normFactors =  NN.normFactors)

    # Preprocess the validation data
    if P.projectDataOntoGridKnots
        outputsValidation, inputsValidation, _ = prepareData(P, πwSim, RStarSim, ζSim; normFactors = NN.normFactors, disableProjection = true)
        validationData = (inputs = inputsValidation, outputs = outputsValidation, normFactors =  NN.normFactors)
    else
        validationData = (inputs = inputs, outputs = outputs, normFactors =  NN.normFactors)
    end

    # Initalize progress indicator
    if P.showPolicyIterations
        p = Progress(P.epochs; desc = "Training πw ALM...", color = :grey, barlen = 0)
    else
        println("Training πw ALM...")
    end

    # Initialize vectors to keep track of training and validation loss
    trainingLosses = zeros(P.epochs)
    validationLosses = zeros(P.epochs)

    # Train the neural network
    for ii in 1:P.epochs

        trainNeuralNetwork!(P, NN, NNGradient, trainingData.inputs, trainingData.outputs)

        # Update training and validation losses
        if P.computeMSEDuringTraining_πw != :none

            displayedStats = []

            if P.computeMSEDuringTraining_πw in (:testOnly, :both)
                trainingLosses[ii] = lossWithoutRegularization(NN, trainingData.inputs, trainingData.outputs)
                push!(displayedStats, (:MSETraining, trainingLosses[ii]))
            end

            if P.computeMSEDuringTraining_πw in (:validationOnly, :both)
                validationLosses[ii] = lossWithoutRegularization(NN, validationData.inputs, validationData.outputs)
                push!(displayedStats, (:MSEValidation, validationLosses[ii]))
            end

        end

        # Update progress indicator
        if P.showPolicyIterations

            if P.computeMSEDuringTraining_πw == :none
                next!(p)
            else
                next!(p; showvalues = displayedStats, valuecolor = :grey)
            end

        end

    end

    # Evaluate the neural network on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]

        # Get the state variables for the current node in the dense state space
        RStar = (P.RDenseGrid[i_R] - NN.normFactors.RStar.location) / NN.normFactors.RStar.scale
        ζ = (P.ζDenseGrid[i_ζ] - NN.normFactors.ζ.location) / NN.normFactors.ζ.scale

        ALM[idx] = feedforward(NN, [RStar, ζ])[1] * NN.normFactors.πw.scale + NN.normFactors.πw.location

    end

    # Update the ALM
    @. πwALMUpdate = P.λALM * ALM + (1-P.λALM) * πwALM

    # Compute R^2 and MSE (loss)
    outputDeNormalize(x) = x * NN.normFactors.πw.scale + NN.normFactors.πw.location
    trainingStats = computeStatsALM(NN, trainingData.inputs, trainingData.outputs, outputDeNormalize)
    validationStats = computeStatsALM(NN, validationData.inputs, validationData.outputs, outputDeNormalize)

    # Add loss to training and validation stats
    trainingStats = (trainingStats..., loss = trainingLosses)
    validationStats = (validationStats..., loss = validationLosses)

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim)
    dist = computeALMDistance(πwALM, πwALMUpdate, visitedNodes)

    # Generate named tuple with all statistics
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    return stats, ALM, trainingData, validationData

end


"""
    updateEπwCondALM!(P, NN::NeuralNetwork, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

Updates the ALM for the term related to inflation expectations using a Neural Network.

"""
function updateEπwCondALM!(P, NN::NeuralNetwork, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

    traininingSuccessful = false
    local stats, ALM, trainingData, validationData

    while !traininingSuccessful

        # Try to update the ALM 
        stats, ALM, trainingData, validationData = _updateEπwCondALM!(P, NN, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

        # If there are issues during training (e.g. gradient explodes), reinitialize the neural network and try again
        if isnan(stats.training.R2)
            @warn "Updating ALM (EπwCond) failed. Reinitializing neural network weights..."
            reinitializeNeuralNetwork!(P, NN, :EπwCond)
        else
            traininingSuccessful = true
        end

    end

    return stats, ALM, trainingData, validationData

end


"""
    _updateEπwCondALM!(P, NN::NeuralNetwork, EπwCondALMUpate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

Updates the ALM for the term related to inflation expectations using a Neural Network.

"""
function _updateEπwCondALM!(P, NN::NeuralNetwork, EπwCondALMUpdate, EπwCondALM, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

    # Initialize the gradient struct (this will be reused during gradient descent)
    NNGradient = NeuralNetworkGradient(NN)

    # Remove burn-in periods
    EπwCondSim = EπwCondSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]
    ζPrimeSim = ζPrimeSim[P.burnIn+1:end]

    # Preprocess the data
    outputs, inputs, NN.normFactors = prepareData(P, EπwCondSim, RStarSim, ζSim, ζPrimeSim)
    trainingData = (inputs = inputs, outputs = outputs, normFactors =  NN.normFactors)

    # Preprocess the validation data
    if P.projectDataOntoGridKnots
        outputsValidation, inputsValidation, _ = prepareData(P, EπwCondSim, RStarSim, ζSim, ζPrimeSim; normFactors = NN.normFactors, disableProjection = true)
        validationData = (inputs = inputsValidation, outputs = outputsValidation, normFactors =  NN.normFactors)
    else
        validationData = (inputs = inputs, outputs = outputs, normFactors =  NN.normFactors)
    end

    # Initalize progress indicator
    if P.showPolicyIterations
        p = Progress(P.epochs; desc = "Training EπwCond ALM...", color = :grey, barlen = 0)
    else
        println("Training EπwCond ALM...")
    end

    # Initialize vectors to keep track of training and validation loss
    trainingLosses = zeros(P.epochs)
    validationLosses = zeros(P.epochs)

    # Train the neural network
    for ii in 1:P.epochs

        trainNeuralNetwork!(P, NN, NNGradient, trainingData.inputs, trainingData.outputs)

        # Update training and validation losses
        if P.computeMSEDuringTraining_EπwCond != :none

            displayedStats = []

            if P.computeMSEDuringTraining_EπwCond in (:testOnly, :both)
                trainingLosses[ii] = lossWithoutRegularization(NN, trainingData.inputs, trainingData.outputs)
                push!(displayedStats, (:MSETraining, trainingLosses[ii]))
            end

            if P.computeMSEDuringTraining_EπwCond in (:validationOnly, :both)
                validationLosses[ii] = lossWithoutRegularization(NN, validationData.inputs, validationData.outputs)
                push!(displayedStats, (:MSEValidation, validationLosses[ii]))
            end

        end

        # Update progress indicator
        if P.showPolicyIterations

            if P.computeMSEDuringTraining_EπwCond == :none
                next!(p)
            else
                next!(p; showvalues = displayedStats, valuecolor = :grey)
            end

        end

    end

    # Evaluate the neural network on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]
        i_ζp = idx[3]

        # Get the state variables for the current node in the dense state space
        RStar = (P.RDenseGrid[i_R] - NN.normFactors.RStar.location) / NN.normFactors.RStar.scale
        ζ = (P.ζDenseGrid[i_ζ] - NN.normFactors.ζ.location) / NN.normFactors.ζ.scale
        ζPrime = (P.ζDenseGrid[i_ζp] - NN.normFactors.ζPrime.location) / NN.normFactors.ζPrime.scale

        ALM[idx] = feedforward(NN, [RStar, ζ, ζPrime])[1] * NN.normFactors.EπwCond.scale + NN.normFactors.EπwCond.location

    end

    # Update the ALM
    @. EπwCondALMUpdate = P.λALM * ALM + (1-P.λALM) * EπwCondALM

    # Compute R^2 and MSE (loss)
    outputDeNormalize(x) = x * NN.normFactors.EπwCond.scale + NN.normFactors.EπwCond.location
    trainingStats = computeStatsALM(NN, trainingData.inputs, trainingData.outputs, outputDeNormalize)
    validationStats = computeStatsALM(NN, validationData.inputs, validationData.outputs, outputDeNormalize)

    # Add loss to training and validation stats
    trainingStats = (trainingStats..., loss = trainingLosses)
    validationStats = (validationStats..., loss = validationLosses)

    # Compute distance between ALMs
    visitedNodes = checkVisitedNodes(P, RStarSim, ζSim, ζPrimeSim)
    dist = computeALMDistance(EπwCondALM, EπwCondALMUpdate, visitedNodes)

    # Generate named tuple with all statistics
    stats = (training = trainingStats, validation = validationStats, dist = dist)

    return stats, ALM, trainingData, validationData

end


"""
    validateNNπwALM(P, NN::NeuralNetwork, πwSim, RStarSim, ζSim)

Evaluates the Neural Network on given dataset.

"""
function validateNNπwALM(P, NN::NeuralNetwork, πwSim, RStarSim, ζSim)

    # Remove burn-in periods
    πwSim = πwSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]

    # Preprocess the data
    outputs, inputs, _ = prepareData(P, πwSim, RStarSim, ζSim; normFactors = NN.normFactors, disableProjection = true)

    # Compute R^2 and MSE (loss)
    outputDeNormalize(x) = x * NN.normFactors.πw.scale + NN.normFactors.πw.location
    stats = computeStatsALM(NN, inputs, outputs, outputDeNormalize)

    return stats

end


"""
    validateNNEπwCondALM(P, NN::NeuralNetwork, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

Evaluates the Neural Network on given dataset.

"""
function validateNNEπwCondALM(P, NN::NeuralNetwork, EπwCondSim, RStarSim, ζSim, ζPrimeSim)

    # Remove burn-in periods
    EπwCondSim = EπwCondSim[P.burnIn+1:end]
    RStarSim = RStarSim[P.burnIn+1:end]
    ζSim = ζSim[P.burnIn+1:end]
    ζPrimeSim = ζPrimeSim[P.burnIn+1:end]

    # Preprocess the data
    outputs, inputs, _ = prepareData(P, EπwCondSim, RStarSim, ζSim, ζPrimeSim; normFactors = NN.normFactors, disableProjection = true)

    # Compute R^2 and MSE (loss)
    outputDeNormalize(x) = x * NN.normFactors.EπwCond.scale + NN.normFactors.EπwCond.location
    stats = computeStatsALM(NN, inputs, outputs, outputDeNormalize)

    return stats

end


"""
    computeStatsALM(NN, inputs, outputs, outputDeNormalize)

Computes R^2 and MSE (loss) of the neural network.

"""
function computeStatsALM(NN, inputs, outputs, outputDeNormalize)

    # Initialize outputs and fitted outputs
    yFit = zeros(length(inputs))
    y = similar(yFit)

    # Compute fitted values
    for ii in 1:length(inputs)
        yFit[ii] = outputDeNormalize(feedforward(NN, inputs[ii])[1])
        y[ii] = outputDeNormalize(outputs[ii][1])
    end

    # Compute R^2
    ȳ = mean(y)
    R2 = 1 - sum((yFit.-y).^2) / sum((y.-ȳ).^2)

    # Compute MSE
    MSE = lossWithoutRegularization(NN, inputs, outputs)

    # Get size of training sample
    nObs = length(outputs)

    return (R2 = R2, MSE = MSE, nObs = nObs)

end


"""
    NeuralNetworkGradient()

Struct that holds the gradient of the loss function of the neural network and
auxiliary matrices used during training.

"""
struct NeuralNetworkGradient
    w1::Array{Float64,2}
    w2::Array{Float64,2}
    b1::Array{Float64,1}
    b2::Array{Float64,1}
    z_Lm1::Array{Float64,1}
    a_Lm1::Array{Float64,1}
    z_L::Array{Float64,1}
    a_L::Array{Float64,1}
    δ_L::Array{Float64,1}
    δ_Lm1::Array{Float64,1}
end


"""
    NeuralNetworkGradient(NN::NeuralNetwork)

Initializes NeuralNetworkGradient struct based on dimensions of neural network NN.

"""
function NeuralNetworkGradient(NN::NeuralNetwork)

    NeuralNetworkGradient(
        zeros(size(NN.w1)),
        zeros(size(NN.w2)),
        zeros(size(NN.b1)),
        zeros(size(NN.b2)),
        zeros(NN.nHidden),
        zeros(NN.nHidden),
        zeros(NN.nOutputs),
        zeros(NN.nOutputs),
        zeros(NN.nOutputs),
        zeros(NN.nHidden)
    )

end


"""
    computeGradient!(NN, NNGradient, trainingInputs, trainingOutputs)

Computes gradient of loss function. Used when training the neural network.

"""
function computeGradient!(NN, NNGradient, trainingInputs, trainingOutputs)

    # Get the sample size
    sampleSize = size(trainingInputs, 1)

    # Reset the gradients
    NNGradient.w1 .= 0.0
    NNGradient.w2 .= 0.0
    NNGradient.b1 .= 0.0
    NNGradient.b2 .= 0.0

    # Create references for easier access
    z_Lm1 = NNGradient.z_Lm1
    a_Lm1 = NNGradient.a_Lm1
    z_L = NNGradient.z_L
    a_L = NNGradient.a_L
    δ_L = NNGradient.δ_L
    δ_Lm1 = NNGradient.δ_Lm1

    # Reset auxiliary matrices
    z_Lm1 .= 0.0
    a_Lm1 .= 0.0
    z_L .= 0.0
    a_L .= 0.0
    δ_L .= 0.0
    δ_Lm1 .= 0.0

    for ii in 1:sampleSize

        # Feed forward
        z_Lm1 .= mul!(z_Lm1, NN.w1, trainingInputs[ii]) .+ NN.b1
        a_Lm1 .= activation.(Ref(NN), z_Lm1)
        z_L .= mul!(z_L, NN.w2, a_Lm1) .+ NN.b2
        a_L .= z_L

        # Backpropagation
        @. δ_L = (a_L - trainingOutputs[ii])
        δ_Lm1 .= mul!(δ_Lm1, NN.w2', δ_L) .* activationPrime.(Ref(NN), z_Lm1)

        # Compute gradient and add it to the sum
        mul!(NNGradient.w2,  δ_L, a_Lm1', 1.0, 1.0)
        mul!(NNGradient.w1,  δ_Lm1, trainingInputs[ii]', 1.0, 1.0)
        NNGradient.b2 .+= δ_L
        NNGradient.b1 .+= δ_Lm1

    end

    # Divide by sample size
    rdiv!(NNGradient.w2, sampleSize)
    rdiv!(NNGradient.w1, sampleSize)
    rdiv!(NNGradient.b2, sampleSize)
    rdiv!(NNGradient.b1, sampleSize)

    # Add regularization term
    @. NNGradient.w1 = 2 * (NNGradient.w1 + NN.λ / sampleSize * NN.w1)
    @. NNGradient.w2 = 2 * (NNGradient.w2 + NN.λ / sampleSize * NN.w2)
    @. NNGradient.b1 = 2 * NNGradient.b1
    @. NNGradient.b2 = 2 * NNGradient.b2

    nothing

end


"""
    loss(NN, inputs, outputs)

Computes loss of neural network for given dataset.

"""
function loss(NN, inputs, outputs)
    return loss(NN, NN.w1, NN.w2, NN.b1, NN.b2, inputs, outputs)
end


"""
    loss(NN, w1, w2, b1, b2, inputs, outputs)

Computes loss of neural network for given dataset using alternative weights and biases.

"""
function loss(NN, w1, w2, b1, b2, inputs, outputs)

    # Get the sample size
    sampleSize = size(inputs, 1)

    # Initialiize the loss
    L = 0.0

    # Preallocate matrices to keep total allocations low
    z_Lm1 = zeros(NN.nHidden)
    a_Lm1 = zeros(NN.nHidden)
    z_L = zeros(NN.nOutputs)
    a_L = zeros(NN.nOutputs)
    tmp = zeros(NN.nOutputs)

    for ii in 1:length(inputs)

        # Feed forward
        z_Lm1 .= mul!(z_Lm1, w1, inputs[ii]) .+ b1
        a_Lm1 .= activation.(Ref(NN), z_Lm1)
        z_L .= mul!(z_L, w2, a_Lm1) .+ b2
        a_L .= z_L

        # Evalute loss for input
        @. tmp = (outputs[ii] - a_L)^2
        L += tmp[1] / sampleSize # Assumes that there is only one output

    end

    # Add regularization term
    L = L + NN.λ / sampleSize * (sum(w1.^2) + sum(w2.^2))

    return L

end


"""
    lossWithoutRegularization(NN, inputs, outputs)

Computes loss of neural network for given dataset without add regularization term.

"""
function lossWithoutRegularization(NN, inputs, outputs)

    # Get the sample size
    sampleSize = size(inputs, 1)

    # Compute loss with regularization
    L = loss(NN, NN.w1, NN.w2, NN.b1, NN.b2, inputs, outputs)

    # Remove regularization term
    L = L - NN.λ / sampleSize * (sum(NN.w1.^2) + sum(NN.w2.^2))

    return L

end


"""
    testWeights!(NNTmp, NN, NNGradient, learn)

Updates the weights of NNTmp using the gradient and a learning parameter starting
from the weights of NN. This function is used to test different learning rates
without changing the weights of the original neural network NN.

"""
function testWeights!(NNTmp, NN, NNGradient, learn)

    @. NNTmp.w1 = NN.w1 - learn * NNGradient.w1
    @. NNTmp.w2 = NN.w2 - learn * NNGradient.w2
    @. NNTmp.b1 = NN.b1 - learn * NNGradient.b1
    @. NNTmp.b2 = NN.b2 - learn * NNGradient.b2

    nothing

end


"""
    findLearningRate(NN, NNGradient, inputs, outputs)

Tries to find lerning rate which leads to the biggest decline of the loss in the
direction given by the gradient.

"""
function findLearningRate(NN, NNGradient, inputs, outputs)

    # Create temporary NN which is used to evaluate the different learning rates
    NNTmp = NeuralNetwork(nInputs = NN.nInputs, nHidden = NN.nHidden, nOutputs = NN.nOutputs,
        λ = NN.λ, learningSpeed = NN.learningSpeed, activationFunction = NN.activationFunction,
        normFactors = NN.normFactors)

    # Get th loss of the current neural network
    jumpsize0 = 0
    L0 = loss(NN, inputs, outputs)

    # Adjust weights in the direction of the gradient
    jumpsize1 = NN.learningSpeed
    testWeights!(NNTmp, NN, NNGradient, jumpsize1) # Updates weights of NNTmp with weights of NN - jumpsize1*gradient
    L1 = loss(NNTmp, inputs, outputs)

    # If the previous step has not lead to a decrease in the loss, decrease learning
    # speed until it leads to a decrease in the loss
    iter = 0

    while L1 > L0

        jumpsize1 = jumpsize1/2
        testWeights!(NNTmp, NN, NNGradient, jumpsize1)
        L1 = loss(NNTmp, inputs, outputs)

        iter += 1

        # If the learning rate that leads to a decrease in loss becomes too small,
        # we are probably in a minimum already
        if jumpsize1 < 1e-16
            return jumpsize0
        end

    end

    # Find learning rate which yields an increase in loss
    jumpsize2 = jumpsize1 * 1.412
    testWeights!(NNTmp, NN, NNGradient, jumpsize2)
    L2 = loss(NNTmp, inputs, outputs)

    iter = 0

    while L1 > L2

        jumpsize2 = jumpsize2*1.1892
        testWeights!(NNTmp, NN, NNGradient, jumpsize2)
        L2 = loss(NNTmp, inputs, outputs)

        iter += 1

        # If too many iterations are required to find an increase in the loss,
        # just return a learning rate that leads to a decrease
        if iter > 10000
            return jumpsize1
        end

    end

    # Given jumpsize0, jumpsize1, and jumpsize2, find the leanring rate which yields
    # the lowest loss
    for ii = 1:3

        if L2 > L0

            jumpsizeNew = (jumpsize2 + jumpsize0)/2
            testWeights!(NNTmp, NN, NNGradient, jumpsizeNew)
            LNew = loss(NNTmp, inputs, outputs)

            if LNew < L1

                jumpsize0 = jumpsize1
                L0 = L1

                jumpsize1 = jumpsizeNew
                L1 = LNew

            else

                jumpsize2 = jumpsizeNew
                L2 = LNew

            end

        else

            jumpsizeNew = (jumpsize2 + jumpsize0)/2
            testWeights!(NNTmp, NN, NNGradient, jumpsizeNew)
            LNew = loss(NNTmp, inputs, outputs)

            if LNew < L1

                jumpsize2 = jumpsize1
                L2 = L1

                jumpsize1 = jumpsizeNew
                L1 = LNew

            else

                jumpsize1 = jumpsizeNew
                L1 = LNew

            end

        end

    end


    return jumpsize1

end


"""
    learnMin(NN, NNTmp, NNGradient, inputs, outputs, jumpsize1)

Used to find learning speed that maximizes gradient descent.

"""
function learnMin(NN, NNTmp, NNGradient, inputs, outputs, jumpsize1)

    # Updates weights of NNTmp with weights of NN - jumpsize1*gradient
    testWeights!(NNTmp, NN, NNGradient, jumpsize1)
    L1 = loss(NNTmp, inputs, outputs)

    return L1

end


"""
    trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

Updates the weights of the neural network.

"""
function trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

    if P.gradientDescent == :fullbatch

        _trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

    else

        batchSize = (P.gradientDescent == :stochastic) ? 1 : P.batchSize
        batchPerEpoch = floor(Int64, length(outputs)/batchSize)
        # Note: floor essentially implies that the last batch will be discareded
        # if it is not at least of size "batchSize"

        for ii in 1:batchPerEpoch
            idx = (ii-1)*batchSize+1:ii*batchSize
            @views _trainNeuralNetwork!(P, NN, NNGradient, inputs[idx], outputs[idx])
        end

    end

    nothing

end


"""
    _trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

Auxiliary function called by trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

"""
function _trainNeuralNetwork!(P, NN, NNGradient, inputs, outputs)

    # Compute the gradient
    computeGradient!(NN, NNGradient, inputs, outputs)

    # Determine learning speed
    if P.learningSpeedType == :optimal
        NNTmp = NeuralNetwork(nInputs = NN.nInputs,
                              nHidden = NN.nHidden,
                              nOutputs = NN.nOutputs,
                              λ = NN.λ,
                              learningSpeed = NN.learningSpeed,
                              activationFunction = NN.activationFunction,
                              normFactors = NN.normFactors)
        f(x) = learnMin(NN, NNTmp, NNGradient, inputs, outputs, x)
        learn = optimize(f, [NN.learningSpeed], LBFGS()).minimizer[1]
    elseif P.learningSpeedType == :improved
        learn = findLearningRate(NN, NNGradient, inputs, outputs)
    else
        learn = P.baseLearningSpeed
    end

    # Update the weights of the neural network
    @. NN.w1 = NN.w1 - learn * NNGradient.w1
    @. NN.w2 = NN.w2 - learn * NNGradient.w2
    @. NN.b1 = NN.b1 - learn * NNGradient.b1
    @. NN.b2 = NN.b2 - learn * NNGradient.b2

end

"""
    feedforward(NN, inputs)

Computes the outputs of neural network NN for given inputs.

"""
function feedforward(NN, inputs)

    hidden = activation.(Ref(NN), NN.w1 * inputs .+ NN.b1)
    outputs = NN.w2 * hidden .+ NN.b2

    return outputs

end


"""

    checkVisitedNodes(P, RStarSim, ζSim)

Returns matrix with number of times each node was visited.

"""
function checkVisitedNodes(P, RStarSim, ζSim)

    visitedNodes = zeros(P.RDenseGridSize, P.ζDenseGridSize)

    RStarStep = convert(Float64, P.RDenseGrid.step)
    ζStep = convert(Float64, P.ζDenseGrid.step)

    for tt in 1:length(RStarSim)

        # Make sure it's within grid bounds
        RStar = max(RStarSim[tt], P.RMin+1e-6)
        RStar = min(RStar, P.RMax-1e-6)
        ζ = max(ζSim[tt], P.ζMin+1e-6)
        ζ = min(ζ, P.ζMax-1e-6)

        # Determine indices of closest gridpoints
        RStarPosD = floor(Int64, (RStar - P.RMin) / RStarStep) + 1
        RStarPosU = ceil(Int64, (RStar - P.RMin) / RStarStep) + 1
        ζPosD = floor(Int64, (ζ - P.ζMin) / ζStep) + 1
        ζPosU = ceil(Int64, (ζ - P.ζMin) / ζStep) + 1

        # Count "0.25 visits" for each of the four closest grid points to each simulated point
        visitedNodes[RStarPosD, ζPosD] += 0.25
        visitedNodes[RStarPosU, ζPosD] += 0.25
        visitedNodes[RStarPosD, ζPosU] += 0.25
        visitedNodes[RStarPosU, ζPosU] += 0.25

    end

    return visitedNodes

end


"""

    checkVisitedNodes(P, RStarSim, ζSim)

Returns matrix with number of times each node was visited.

"""
function checkVisitedNodes(P, RStarSim, ζSim, ζPrimeSim)

    visitedNodes = zeros(P.RDenseGridSize, P.ζDenseGridSize, P.ζDenseGridSize)

    RStarStep = convert(Float64, P.RDenseGrid.step)
    ζStep = convert(Float64, P.ζDenseGrid.step)

    for tt in 1:length(RStarSim)

        # Make sure it's within grid bounds
        RStar = max(RStarSim[tt], P.RMin+1e-6)
        RStar = min(RStar, P.RMax-1e-6)
        ζ = max(ζSim[tt], P.ζMin+1e-6)
        ζ = min(ζ, P.ζMax-1e-6)
        ζPrime = max(ζPrimeSim[tt], P.ζMin+1e-6)
        ζPrime = min(ζPrime, P.ζMax-1e-6)

        # Determine indices of closest gridpoints
        RStarPosD = floor(Int64, (RStar - P.RMin) / RStarStep) + 1
        RStarPosU = ceil(Int64, (RStar - P.RMin) / RStarStep) + 1
        ζPosD = floor(Int64, (ζ - P.ζMin) / ζStep) + 1
        ζPosU = ceil(Int64, (ζ - P.ζMin) / ζStep) + 1
        ζPrimePosD = floor(Int64, (ζPrime - P.ζMin) / ζStep) + 1
        ζPrimePosU = ceil(Int64, (ζPrime - P.ζMin) / ζStep) + 1

        # Count "0.125 visits" for each of the eight closest grid points to each simulated point
        visitedNodes[RStarPosD, ζPosD, ζPrimePosD] += 0.125
        visitedNodes[RStarPosD, ζPosD, ζPrimePosU] += 0.125
        visitedNodes[RStarPosD, ζPosU, ζPrimePosD] += 0.125
        visitedNodes[RStarPosD, ζPosU, ζPrimePosU] += 0.125
        visitedNodes[RStarPosU, ζPosD, ζPrimePosD] += 0.125
        visitedNodes[RStarPosU, ζPosD, ζPrimePosU] += 0.125
        visitedNodes[RStarPosU, ζPosU, ζPrimePosD] += 0.125
        visitedNodes[RStarPosU, ζPosU, ζPrimePosU] += 0.125

    end

    return visitedNodes

end


"""

    computeALMDistance(ALM, ALMUpdate, visitedNodes)

Computes different measures of distance between the new and old ALM.

"""
function computeALMDistance(ALM, ALMUpdate, visitedNodes)

    # Initialize measures of distance
    distFull = 0.0              # Distance between full two matrices
    distVisited = 0.0           # Distance between matrices for visited nodes only (weighted by number of visits)
    distVisitedUnweighted = 0.0 # Distance between matrices for visited nodes only  (unweighted)
    distNorm = norm(ALM-ALMUpdate) # This whould be equivalent to distFull but not divided by sqrt(length(ALM))

    # Compute the sum of squared errors
    for ii in 1:length(ALM)

        # Add to full distance
        distFull += (ALM[ii] - ALMUpdate[ii])^2

        # Add to distance only if node was visited
        if visitedNodes[ii] > 0.0
            distVisited += visitedNodes[ii] * (ALM[ii] - ALMUpdate[ii])^2
            distVisitedUnweighted += (ALM[ii] - ALMUpdate[ii])^2
        end

    end

    # Compute mean and take square root
    distFull = sqrt(distFull / length(ALM))
    distVisited = sqrt(distVisited / sum(visitedNodes .> 0.0))
    distVisitedUnweighted = sqrt(distVisitedUnweighted / sum(visitedNodes .> 0.0))

    return (norm = distNorm, full = distFull, visited = distVisited, visitedUnweighted = distVisitedUnweighted)

end
