"""
    computeShocks(P)

Simulates aggregate shocks based on the settings defined in P.

"""
function computeShocks(P)

    # Simulate aggregate shock
    ζ = P.ζ̄ * ones(P.T)

    for tt in 2:P.T
        ζ[tt] = P.ζ̄ * (ζ[tt-1]/P.ζ̄)^P.ρ_ζ * exp(P.σ̃_ζ * randn())
    end

    return (ζ = ζ, )

end


"""
    convertAggregateState(P, ζ)

Assigns aggregate shock to desired shock type.

"""
function convertAggregateState(P, ζ)

    if P.aggShockType == :Preference
        states = (ξ = ζ, q = P.q̄)
    elseif P.aggShockType == :DiscountFactor
        states = (ξ = P.ξ̄, q = ζ)
    else
        error("Unknown aggShockType: $(P.aggShockType)")
    end

    return states

end


"""
    solveModel(P, S)

Solves the model using the algorithm described in the notes.

"""
function solveModel(P, S, DSS;
    bPolicy = Array{Float64,4}(undef, 0, 0, 0, 0),
    πwALM = Array{Float64,2}(undef, 0, 0),
    EπwCondALM = Array{Float64,3}(undef, 0, 0, 0),
    bCross = Array{Float64,1}(undef, 0),
    NNπw = initializeALMApproximation(P).NNπw, # Neural Network or linear regression coefficients for wage inflation
    NNEπwCond = initializeALMApproximation(P).NNEπwCond, # Neural Network or linear regression coefficients for the wage inflation expectation term
    RStar = DSS.RStar,
    H = DSS.H,
    πw = DSS.πw)

    # Check whether the supplied settings are valid
    validateSettings(P)

    # Display type of solution algorithm
    if P.approximationTypeALM == :NeuralNetwork
        println("Solution algorithm based on neural networks")
    elseif P.approximationTypeALM == :LinearRegression
        println("Solution algorithm based on linear regressions")
    elseif P.approximationTypeALM in (:DualRegression, :QuadRegression)
        println("Solution algorithm based on linear regressions for ZLB and non-ZLB periods")
    end

    # Display type of aggregate shock
    println("Aggregate Shock Type: $(P.aggShockType)")

    # Initialize bond policy function
    if length(bPolicy) == 0
        bPolicy = zeros(P.bGridSize, P.RGridSize, P.sGridSize, P.ζGridSize)
        for idx in CartesianIndices(bPolicy)

            # Get indices in the grid
            i_b = idx[1]
            i_s = idx[3]

            # Assign steady state policy
            bPolicy[idx] = DSS.bPolicy[i_b, i_s]

        end
    end

    # Initial guess for the aggregate law of motion for wage inflation
    if length(πwALM) == 0
        πwALM = DSS.πw * ones(P.RDenseGridSize, P.ζDenseGridSize)
    end
    πwALMUpdate = similar(πwALM)

    # Initial guess for the aggregate law of motion for the wage inflation expectation term
    if length(EπwCondALM) == 0
        EπwCondALM = zeros(P.RDenseGridSize, P.ζDenseGridSize, P.ζDenseGridSize)
        # Expectation term is such that it cancels in the wage inflation equation
        # i.e. aggregate labor supply and output are equal to their steady state value
    end
    EπwCondALMUpdate = similar(EπwCondALM)

    # Initialize cross-sectional bond distribution
    if length(bCross) == 0
        bCross = copy(DSS.bCross)
    end

    # Interpolate the ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    # Initalize vector holidng progress indicator such R^2, MSE and dist,
    algorithmProgress = Any[]

    # Intialize
    iter = 1
    dist = 10.0

    while(dist > P.tolALM)

        println("-------------------------------------------------------------")
        println("Iteration ", iter)

        # Update ALM learning parameters
        if P.λALMAltIteration == 0 || (P.λALMAltIteration != 0 && iter % P.λALMAltIteration != 0)
            almUp = (P.λALMInit - P.λALMFinal) * P.λALMDecay^(iter-1) + P.λALMFinal
        else
            almUp = (P.λALMInitAlt - P.λALMFinalAlt) * P.λALMDecayAlt^(iter-1) + P.λALMFinalAlt
        end
        P = settings(P; λALM = almUp)

        # Update Neural Network learning parameters
        if P.approximationTypeALM == :NeuralNetwork

            if P.learningSpeedType == :decay
                learnUp = (P.learningSpeedInit - P.learningSpeedFinal) * P.learningSpeedDecay^(iter-1) + P.learningSpeedFinal
                P = settings(P; baseLearningSpeed = learnUp)
                NNπw.learningSpeed = learnUp
                NNEπwCond.learningSpeed = learnUp
            else
                learnUp = P.baseLearningSpeed
            end

        end

        # Display updated learning parameters
        if P.approximationTypeALM == :NeuralNetwork
            println("Learning: ", almUp, ", ", learnUp, ", ", P.λᵇ)
        else
            println("Learning: ", almUp, ", ", P.λᵇ)
        end

        # Find individual bond policy function given the aggregate laws of motion
        solveIndividualProblem!(P, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol)

        # Simulate the model (Stop updating bCross if ALM is sufficiently accurate)
        if dist < P.tolALM*100 || P.alwaysStartFromDSS
            println("bCross Fixed")
            πwSim, EπwCondSim, HSim, RStarSim, _, _ = simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCross, RStar, H, πw)
        else
            println("bCross Updated")
            πwSim, EπwCondSim, HSim, RStarSim, bCross, _  = simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCross, RStar, H, πw)
            RStar = RStarSim[end]
            H = HSim[end]
            πw = πwSim[end]
        end

        # Update ALMs
        if iter == 1  && P.enableMultipleNNStarts && P.approximationTypeALM == :NeuralNetwork

            # Initialize temporary arrays containing results realated to NNs
            NNπws = Array{Any,2}(undef, P.NNStarts, 2)
            NNEπwConds = Array{Any,2}(undef, P.NNStarts, 2)
            πwALMUpdateDummy = similar(πwALMUpdate)
            EπwCondALMUpdateDummy = similar(EπwCondALMUpdate)

            # Train several NNs
            for ii in 1:P.NNStarts

                println("Candidate NN $(ii):")

                # Initialize the NNs
                if ii == 1 # Use the NNs supplied to the function as one candidate
                    NNπws[ii, 1] = NNπw
                    NNEπwConds[ii, 1] = NNEπwCond
                else # Generate additional canidates as needed
                    NNπws[ii, 1], NNEπwConds[ii, 1] = initializeALMApproximation(P)
                end

                # Train the NNs
                NNπws[ii, 2] = updateπwALM!(P, NNπws[ii, 1], πwALMUpdateDummy, πwALM, πwSim[2:end], RStarSim[1:end-1], S.ζ[2:end])
                NNEπwConds[ii, 2] = updateEπwCondALM!(P, NNEπwConds[ii, 1], EπwCondALMUpdateDummy, EπwCondALM, EπwCondSim[2:end-2], RStarSim[1:end-3], S.ζ[2:end-2], S.ζ[3:end-1])

            end

            # Determine the evaluation criteria for each NN
            if P.criteriaMultipleNNstarts == :validationMSE # Use MSE for whole dataset in this case
                evalCriteriaNNπw = [NNπws[ii, 2][1].validation.MSE for ii in 1:P.NNStarts]
                evalCriteriaNNEπwCond = [NNEπwConds[ii, 2][1].validation.MSE for ii in 1:P.NNStarts]
            elseif P.criteriaMultipleNNstarts == :trainingMSE
                evalCriteriaNNπw = [NNπws[ii, 2][1].training.MSE for ii in 1:P.NNStarts]
                evalCriteriaNNEπwCond = [NNEπwConds[ii, 2][1].training.MSE for ii in 1:P.NNStarts]
            else
                error("Unknown criteriaMultipleNNstarts type: $(P.criteriaMultipleNNstarts)")
            end

            # Find NN with minimum MSE
            _, idxNNπw = findmin(evalCriteriaNNπw)
            _, idxNNEπwCond = findmin(evalCriteriaNNEπwCond)

            # Show the results
            println("MSEs (πw): ", evalCriteriaNNπw)
            println("NN with lowest MSE: ", idxNNπw)
            println("MSEs (EπwCond): ", evalCriteriaNNEπwCond)
            println("NN with lowest MSE: ", idxNNEπwCond)

            # Assign best NN and stats to the respective variables
            NNπw = NNπws[idxNNπw, 1]
            NNEπwCond = NNEπwConds[idxNNEπwCond, 1]
            NNπwStats, NNπwALMMat, NNπwTrainingData, _ = NNπws[idxNNπw, 2]
            NNEπwCondStats, NNEπwCondALMMat, NNEπwCondTrainingData, _ = NNEπwConds[idxNNEπwCond, 2]

            # Update the ALM manually (since we circumvented the updating in the updateALM functions)
            @. πwALMUpdate = P.λALM * NNπwALMMat + (1-P.λALM) * πwALM
            @. EπwCondALMUpdate = P.λALM * NNEπwCondALMMat + (1-P.λALM) * EπwCondALM

        else

            if P.approximationTypeALM in (:DualRegression, :QuadRegression) # Needs two additional arguments
                NNπwStats, NNπwALMMat, NNπwTrainingData, _ = updateπwALM!(P, NNπw, πwALMUpdate, πwALM, πwSim[2:end], RStarSim[1:end-1], S.ζ[2:end], DSS, EπwCondALM)
                NNEπwCondStats, NNEπwCondALMMat, NNEπwCondTrainingData, _ = updateEπwCondALM!(P, NNEπwCond, EπwCondALMUpdate, EπwCondALM, EπwCondSim[2:end-2], RStarSim[1:end-3], S.ζ[2:end-2], S.ζ[3:end-1], DSS, πwALM)
            else
                NNπwStats, NNπwALMMat, NNπwTrainingData, _ = updateπwALM!(P, NNπw, πwALMUpdate, πwALM, πwSim[2:end], RStarSim[1:end-1], S.ζ[2:end])
                NNEπwCondStats, NNEπwCondALMMat, NNEπwCondTrainingData, _ = updateEπwCondALM!(P, NNEπwCond, EπwCondALMUpdate, EπwCondALM, EπwCondSim[2:end-2], RStarSim[1:end-3], S.ζ[2:end-2], S.ζ[3:end-1])
            end

        end

        # Show distance
        println("ALM (πw) Distance: ", NNπwStats.dist)
        println("Training (N = ", NNπwStats.training.nObs, ") - R2 (πw): ", NNπwStats.training.R2, ", MSE (πw): ", NNπwStats.training.MSE)
        if P.projectDataOntoGridKnots
            println("Validation (N = ", NNπwStats.validation.nObs, ") - R2 (πw): ",
                NNπwStats.validation.R2, ", MSE (πw): ", NNπwStats.validation.MSE)
        end
        println("ALM (EπwCond) Distance: ", NNEπwCondStats.dist)
        println("Training (N = ", NNEπwCondStats.training.nObs, ") - R2 (EπwCond): ", NNEπwCondStats.training.R2, ", MSE (EπwCond): ", NNEπwCondStats.training.MSE)
        if P.projectDataOntoGridKnots
            println("Validation (N = ", NNEπwCondStats.validation.nObs, ") - R2 (EπwCond): ",
                NNEπwCondStats.validation.R2, ", MSE (πw): ", NNEπwCondStats.validation.MSE)
        end
        dist = max(NNπwStats.dist.visited, NNEπwCondStats.dist.visited)

        # Save algorithm progress results
        push!(algorithmProgress, (NNπwStats = NNπwStats, NNEπwCondStats = NNEπwCondStats))

        # Update interpolation of ALMs
        πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALMUpdate, extrapolation_bc = Line())
        EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALMUpdate, extrapolation_bc = Line())

        # Plot results for current iteration
        if P.showProgressPlots || (dist < P.tolALM && P.saveFigureAtEnd) || P.saveFigureEachIteration

            # Simulate the predictions of the ALM (and output) conditional on the equilibrium nominal interest rate and ζ
            πwSimALM = similar(πwSim)
            EπwCondSimALM = similar(πwSim)
            YSim = similar(HSim)
            for tt in 2:P.T-1
                πwSimALM[tt] = evaluateALM(πwALMInterpol, RStarSim[tt-1], S.ζ[tt])
                EπwCondSimALM[tt] = evaluateALM(EπwCondALMInterpol, RStarSim[tt-1], S.ζ[tt], S.ζ[tt+1])
                _, _, _, _, _, YSim[tt], _, _ = computeAggregateVariables(P, DSS, RStarSim[tt-1], S.ζ[tt], πwALMInterpol, EπwCondALMInterpol)
            end
            πwSimALM[end] = evaluateALM(πwALMInterpol, RStarSim[end-1], S.ζ[end])
            _, _, _, _, _, YSim[end], _, _ = computeAggregateVariables(P, DSS, RStarSim[end-1], S.ζ[end], πwALMInterpol, EπwCondALMInterpol)

            # Plot the series
            p = plotComparison(P, S, DSS, πwSim[2:end-1], πwSimALM[2:end-1], EπwCondSim[2:end-1],
                                EπwCondSimALM[2:end-1], RStarSim[2:end-1], HSim[2:end-1], YSim[2:end-1], S.ζ[2:end-1], bCross,
                                πwALMUpdate, EπwCondALMUpdate, NNπwALMMat, NNEπwCondALMMat, algorithmProgress,
                                NNπwTrainingData, NNEπwCondTrainingData)

            # Save figure
            if dist < P.tolALM && P.saveFigureAtEnd || P.saveFigureEachIteration
                if !isdir("Figures/$(P.filenameFolder)")
                    mkpath("Figures/$(P.filenameFolder)")
                end
                savefig(p, "Figures/$(P.filenameFolder)/$(P.filenamePrefix)_$(P.filenameExt).pdf")
            end

        end

        # Update the ALMs
        πwALM .= πwALMUpdate
        EπwCondALM .= EπwCondALMUpdate
        iter = iter + 1

        # Save results after each iteration
        if P.saveAlgorithmProgress

            if P.appendIterationToFilename
                filename = "Results/$(P.filenameFolder)/$(P.filenamePrefix)_$(P.filenameExt)_iter$(iter-1).bson"
            else
                filename = P.filename
            end

            println("Saving...")
            if !isdir(filename)
                mkpath("Results/$(P.filenameFolder)")
            end
            @save filename P S DSS bPolicy πwALM EπwCondALM NNEπwCond NNπw bCross RStar H πw algorithmProgress
            println("Saved")

        end

    end

    # Save the results
    println("Saving...")
    if !isdir(P.filename)
        mkpath("Results/$(P.filenameFolder)")
    end
    @save P.filename P S DSS bPolicy πwALM EπwCondALM NNEπwCond NNπw bCross RStar H πw algorithmProgress
    println("Saved")

    # Remove intermediate results
    if P.removeIntermediateResultsAtEnd && P.appendIterationToFilename

        println("Removing Intermediate Results...")

        for ii in 1:iter-1

            filename = "Results/$(P.filenameFolder)/$(P.filenamePrefix)_$(P.filenameExt)_iter$(ii).bson"

            if isfile(filename)
                rm(filename)
            end

        end

        println("Done")

    end

    return bPolicy, πwALM, EπwCondALM, NNEπwCond, NNπw, bCross, RStar, H, πw, algorithmProgress

end


"""
    initializeALMApproximation(P)

Initializes the Neural Network or Linear Regression coefficients depending on settings.

"""
function initializeALMApproximation(P)

    # Set seed for NN initialization
    if !P.enableMultipleNNStarts && P.initalizationSeed != -1
        Random.seed!(P.initalizationSeed)
    end

    if P.approximationTypeALM == :NeuralNetwork

        NNπw = NeuralNetwork(nInputs = 2,
            nOutputs = 1,
            nHidden = P.nHiddenNodes_πw,
            λ = P.reg_πw,
            learningSpeed = P.baseLearningSpeed,
            activationFunction = P.activationFunction,
            normFactors = prepareData(P, randn(10), randn(10), randn(10); disableProjection = true)[3])

        NNEπwCond = NeuralNetwork(nInputs = 3,
            nOutputs = 1,
            nHidden = P.nHiddenNodes_EπwCond,
            λ = P.reg_EπwCond,
            learningSpeed = P.baseLearningSpeed,
            activationFunction = P.activationFunction,
            normFactors = prepareData(P, randn(10), randn(10), randn(10), randn(10); disableProjection = true)[3])

        # By default, weights are initialized using standard normal distribution
        # Reinitialize if differently initialized weights are required
        if P.initalizationType == :he

            NNπw.w1 .= sqrt(2 / NNπw.nInputs) * randn(NNπw.nHidden, NNπw.nInputs)
            NNπw.w2 .= sqrt(2 / NNπw.nHidden) * randn(NNπw.nOutputs, NNπw.nHidden)
            NNEπwCond.w1 .= sqrt(2 / NNEπwCond.nInputs) * randn(NNEπwCond.nHidden, NNEπwCond.nInputs)
            NNEπwCond.w2 .= sqrt(2 / NNEπwCond.nHidden) * randn(NNEπwCond.nOutputs, NNEπwCond.nHidden)

        elseif P.initalizationType == :xavier

            NNπw.w1 .= sqrt(1 / NNπw.nInputs) * randn(NNπw.nHidden, NNπw.nInputs)
            NNπw.w2 .= sqrt(1 / NNπw.nHidden) * randn(NNπw.nOutputs, NNπw.nHidden)
            NNEπwCond.w1 .= sqrt(1 / NNEπwCond.nInputs) * randn(NNEπwCond.nHidden, NNEπwCond.nInputs)
            NNEπwCond.w2 .= sqrt(1 / NNEπwCond.nHidden) * randn(NNEπwCond.nOutputs, NNEπwCond.nHidden)

        elseif P.initalizationType == :xavier2

            NNπw.w1 .= sqrt(2 / (NNπw.nHidden + NNπw.nInputs)) * randn(NNπw.nHidden, NNπw.nInputs)
            NNπw.w2 .= sqrt(2 / (NNπw.nOutputs + NNπw.nHidden)) * randn(NNπw.nOutputs, NNπw.nHidden)
            NNEπwCond.w1 .= sqrt(2 / (NNEπwCond.nHidden + NNEπwCond.nInputs)) * randn(NNEπwCond.nHidden, NNEπwCond.nInputs)
            NNEπwCond.w2 .= sqrt(2 / (NNEπwCond.nOutputs + NNEπwCond.nHidden)) * randn(NNEπwCond.nOutputs, NNEπwCond.nHidden)

        end

    elseif  P.approximationTypeALM == :LinearRegression

        NNπw = zeros(4)
        NNEπwCond = zeros(7)

    elseif  P.approximationTypeALM == :DualRegression

        NNπw = zeros(4, 2)
        NNEπwCond = zeros(7, 2)

    elseif  P.approximationTypeALM == :QuadRegression

        NNπw = zeros(4, 2)
        NNEπwCond = zeros(7, 4)

    else

        error("approximationTypeALM option unknown ($(P.approximationTypeALM))")

    end

    return (NNπw = NNπw, NNEπwCond = NNEπwCond)

end


"""
    evaluateALM(coeff, A, B)

Computes the value predicted by the aggregate law of motion (ALM) for given inputs.

"""
function evaluateALM(coeff, A, B)

    res = coeff(A,B)
    return res

end


"""
    evaluateALM(coeff, A, B, C)

Computes the value predicted by the aggregate law of motion (ALM) for given inputs.

"""
function evaluateALM(coeff, A, B, C)

    res = coeff(A,B,C)
    return res

end


"""
    evaluateALMExpectation(P, ALM, RStar, ζ)

Computes (wage) inflation expectation term in the wage inflation equation

"""
function evaluateALMExpectation(P, ALM, RStar, ζ)

    expec = 0.0

    for ii in 1:P.nGHNodes

        # Aggregate shock in the next period
        ζp = P.ζ̄ * (ζ/P.ζ̄)^P.ρ_ζ * exp(P.eNodes[ii])

        # Evaluate ALM for each value of ζp to compute the expectation
        expec += evaluateALM(ALM, RStar, ζ, ζp) * P.eWeights[ii]

    end

    return expec

end


"""
    monetaryPolicyRule(P, DSS, π, Y, RStarPrev)

Taylor rule of the model.

"""
function monetaryPolicyRule(P, DSS, π, Y, RStarPrev)

    # Use R for inertia in the Taylor rule if necessary
    if !P.useRStarForTaylorInertia && P.bindingZLB && RStarPrev < P.ZLBLevel
        RStarPrev = P.ZLBLevel
    end

    # Compute nominal interest rate implied by asymmetric Taylor rule
    # Note ϕₕ = ϕₗ implies a standard Taylor rule
    RInf = (π < DSS.π) * (π/DSS.π)^(P.ϕₗ) + (π >= DSS.π) * (π/DSS.π)^(P.ϕₕ)
    RStar = DSS.R * (RStarPrev/DSS.R)^P.ρ_R * (RInf * (Y/DSS.Y)^P.ϕʸ)^(1-P.ρ_R)

    # Check whether ZLB is binding
    R = applyLowerBound(P, RStar)

    # Constrain RStar to get legacy behavior of model
    if P.useLegacyRStar
        RStar = R
    end

    return R, RStar

end


"""
    applyLowerBound(P, RStar)

Returns the nominal rate R for given desired nominal rate RStar.

"""
function applyLowerBound(P, RStar)

    # Check whether ZLB is binding
    if P.bindingZLB && RStar < P.ZLBLevel
        R = P.ZLBLevel
    else
        R = RStar
    end

    return R

end


"""
    checkZLB(P, DSS, π, Y, RStarPrev)

Checks whether the ZLB is binding for a particular node in the state space.

"""
function checkZLB(P, DSS, π, Y, RStarPrev)

    # Compute nominal interest rate
    R, RStar = monetaryPolicyRule(P, DSS, π, Y, RStarPrev)

    # Check whether the ZLB binds
    if R <= P.ZLBLevel
        ZLBBinds = 1
    else
        ZLBBinds = 0
    end

    return ZLBBinds

end


"""
    solveIndividualProblem!(P, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol)

Finds individual bond policy function given the aggregate laws of motion.

"""
function solveIndividualProblem!(P, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol)

    # Initialize matrices
    bPolicyUpdate = similar(bPolicy)
    bPolicyError = similar(bPolicy)

    # Precompute everything that is static during the policy function iteration
    RStarPrime, wealth, _, πPrime, wPrime, HPrime, TPrime, ζPrime =
        precomputeStaticPartsIndividualProblem(P, DSS, πwALMInterpol, EπwCondALMInterpol)

    # Interpolate bond policy function
    bPolicyInterpol = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), bPolicy, extrapolation_bc = Line())

    # Initialize policy function iteration
    iter = 1
    dist = 10.0

    # Initalize progress indicator
    if P.showPolicyIterations
        p = ProgressUnknown(desc = "Policy Function Iteration:",  color = :grey)
    else
        print("Policy Function Iteration: ")
    end

    # Do bond policy funtion iteration
    while(dist > P.tol)

        # Update the interpolation
        bPolicyInterpol = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), bPolicy, extrapolation_bc = Line())

        # Compute the proposal for new asset policy function
        @batch per=core  for idx in CartesianIndices(bPolicy)

            # Determne indices of 1D grids
            i_b = idx[1]
            i_R = idx[2]
            i_s = idx[3]
            i_ζ = idx[4]

            # Update the policy
            bPolicyUpdate[idx] = updateBondPolicy(P, DSS, bPolicyInterpol, bPolicy[idx],
                wealth[idx], RStarPrime[i_R, i_ζ],
                @view(πPrime[i_R, i_ζ, :]),
                wPrime,
                @view(HPrime[i_R, i_ζ, :]),
                @view(TPrime[i_R, i_ζ, :]),
                @view(ζPrime[i_ζ, :]),
                P.ζGrid[i_ζ],
                i_s)

        end

        # Check the distance between current iteration and the previous one
        @. bPolicyError = abs.(bPolicyUpdate - bPolicy)
        dist = maximum(bPolicyError)

        # Display current iteration
        if P.showPolicyIterations
            ProgressMeter.next!(p; showvalues = [(:Distance, dist), (:DistanceAlt, sum(bPolicyError.^2)/length(bPolicyError)), (:MaxErrorId, findall(bPolicyError .== dist)), (:MaxErrorPolicy, bPolicy[findall(bPolicyError .== dist)]), (:MaxErrorPolicyUpdate, bPolicyUpdate[findall(bPolicyError .== dist)])], valuecolor = :grey)
        end

        # Update the bond policy function
        @. bPolicy = P.λᵇ * bPolicyUpdate + (1-P.λᵇ) * bPolicy
        iter = iter+1

        if iter > P.maxPolicyIterations
            if P.showWarningsAndInfo
                @warn "Maximum number of iterations reached: dist = $(dist)"
            end
            break
        end

    end

    # Display final iteration
    if !P.showPolicyIterations
        println(iter, " (Distance bond policy: ", dist, ")")
    else
        ProgressMeter.finish!(p)
    end

    nothing

end


"""
    updateBondPolicy(P, DSS, bPolicyInterpol, bPrime, wealth, RStarPrime, πPrime, wPrime, HPrime, TPrime, ζPrime, ζ, i_s)

Auxiliary function called when solving the individual problem. Computes the
updated policy function for a particular node in the state space.

"""
function updateBondPolicy(P, DSS, bPolicyInterpol, bPrime, wealth, RStarPrime, πPrime, wPrime, HPrime, TPrime, ζPrime, ζ, i_s)

    # Initialize expectation term in the Euler equation
    expec = 0.0

    for ii in 1:P.nGHNodes, jj in 1:P.sGridSize

        # Get bond decision at t+1
        b2Prime = bPolicyInterpol(bPrime, RStarPrime, jj, ζPrime[ii])

        # Determine nominal interest rate set by central bank in previous period
        RPrime = applyLowerBound(P, RStarPrime)

        # Compute consumption from the budget constraint
        cPrime = computeIndividualCashOnHand(P, bPrime, P.sGrid[jj], RPrime, πPrime[ii], wPrime, HPrime[ii], TPrime[ii]) - b2Prime
        cPrime = cPrime < 0.0 ? 1e-10 : cPrime # Make sure that consumption is positive

        # Compute marginal utility of future consumption
        muPrime = cPrime^(-P.σ)

        # Determine preference shock
        ξPrime = convertAggregateState(P, ζPrime[ii]).ξ
        ξ = convertAggregateState(P, ζ).ξ

        # Determine discount factor shock
        q = convertAggregateState(P, ζ).q # Note that either q or ξ is going to be equal to one

        # Add to the expectation term
        expec += ξPrime/ξ * q * muPrime * RPrime / πPrime[ii] * P.Ω[i_s, jj] * P.eWeights[ii]

    end

    # Current consumption found from the Euler equation if the borrowing constraint is not binding
    cUpdate = (P.β*expec)^(-1/P.σ)

    # Compute the updated policy function
    bPrimeUpdate = wealth - cUpdate

    # Make sure that bPrime is within the grid bounds
    bPrimeUpdate = max(P.bMin, bPrimeUpdate)
    bPrimeUpdate = min(P.bMax, bPrimeUpdate)

    return bPrimeUpdate

end


"""
    simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCrossInit, RStarInit, HInit, πwInit)

Simulates the model for a given sequence of shocks, policy functions, and an
initial distribution of bonds.

"""
function simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCrossInit, RStarInit, HInit, πwInit; savebCrossSeries = false, useMINPACK = false)

    # Initialize simulated series
    πw = zeros(P.T)
    H = zeros(P.T)
    EπwCond = zeros(P.T)
    RStar = zeros(P.T)
    RStar[1] = RStarInit
    H[1] = HInit
    πw[1] = πwInit
    errorCodes = zeros(Int64, P.T)

    # Initialize the cross sectional distribution of bonds
    bCross = copy(bCrossInit)
    bPrimeCross = similar(bCross)

    # Initialize the series for the bond distribution
    if savebCrossSeries
        bCrossSeries = zeros(P.bDenseGridSize, P.sGridSize, P.T)
    else
        bCrossSeries = zeros(P.bDenseGridSize, P.sGridSize, 1) # This is initalized in 3 dimensions to always return the same type
    end
    bCrossSeries[:, :, 1] .= bCross

    # Create an interpolation function
    bPolicyInterpol = linear_interpolation((P.bGrid, P.RGrid, 1:P.sGridSize, P.ζGrid), bPolicy, extrapolation_bc = Line())

    # Initalize progress indicator
    if P.showPolicyIterations
        p = Progress(P.T-1; desc = "Simulating...", color = :grey, barlen=0)
    else
        println("Simulating...")
    end

    for tt in 2:P.T

        # Find market clearing (wage) inflation
        f!(res, πwGuess) = check!(res, P, DSS, πwGuess, RStar[tt-1], S.ζ[tt], bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

        if useMINPACK # Faster non-linear solver from MINPACK

            res = fsolve(f!, [evaluateALM(πwALMInterpol, RStar[tt-1], S.ζ[tt])], 3; tol = 1e-8)
            πw[tt] = res.x[1]
            errorCodes[tt] = res.converged ? 0 : 1  # Note 0 means the solver converged. 1 means that there was some issue

        else # Slower (but potentially more robust) non-linear solver

            res = nlsolve(f!, [evaluateALM(πwALMInterpol, RStar[tt-1], S.ζ[tt])], show_trace = false)
            πw[tt] = res.zero[1]
            errorCodes[tt] = converged(res) ? 0 : 1 # Note 0 means the solver converged. 1 means that there was some issue

        end

        # Get bond distribution at market clearing inflation
        _, B, bPrimeCross, RStar[tt], H[tt] = check(P, DSS, πw[tt], RStar[tt-1], S.ζ[tt], bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

        # Display an error message if the solver did not find a solution
        if errorCodes[tt] == 1
            println("\nSimulation error. πw = ", πw[tt], ", B = ", B, ", B_error = ", B - DSS.B)
        end

        # Compute the conditional expectation term
        if P.indexRotemberg
            EπwCond[tt-1] = log(πw[tt]/DSS.πw) * H[tt] / H[tt-1]
        else
            EπwCond[tt-1] = log(πw[tt]) * H[tt] / H[tt-1]
        end

        # Update the crossectional distribution
        if savebCrossSeries
            bCrossSeries[:, :, tt] .= bCross
        end
        bCross .= bPrimeCross

        # Update progress indicator
        if P.showPolicyIterations
            next!(p; showvalues = [(:Period, tt), (:PercZLBBinding, sum(RStar[1:tt] .<= P.ZLBLevel)/tt*100)], valuecolor = :grey)
        end

    end

    return πw, EπwCond, H, RStar, bCross, bCrossSeries

end


"""
    check!(res, P, DSS, πw, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

Auxiliary function called during simulation. Returns the equilibrium amount of bonds
for given wage inflation.

"""
function check!(res, P, DSS, πw, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

    errorB, _, _, _, _ = check(P, DSS, πw, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)
    res[1] = errorB

    nothing
end


"""
    check(res, P, DSS, πwVec, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

Auxiliary function called during simulation. Returns the equilibrium amount of bonds
for given wage inflation.

"""
function check(P, DSS, πwVec, RStar, ζ, bCross, bPolicyInterpol, πwALMInterpol, EπwCondALMInterpol)

    # Extract the guess for inflation
    πw = πwVec[1]
    π = πw

    # Check whether inflation is too negative (<-100%) such that there would be
    # numerical issues
    if !checkInflationFeasibility(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = πw)
        return NaN, NaN, bCross, NaN, NaN
    end

    # Compute required aggregate variables at time t
    _, _, _, w, H, _, T, RStarPrime = computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = πw)

    # Determine nominal interest rate set by central bank in previous period
    R = applyLowerBound(P, RStar)

    # Initialize time t+1 matrices
    πPrime = zeros(P.nGHNodes)
    wPrime = NaN
    HPrime = similar(πPrime)
    TPrime = similar(πPrime)
    ζPrime = similar(πPrime)

    # Compute time t+1 variables which are used for computing expectations
    for ii in 1:P.nGHNodes

        # Aggregate shock in the next period
        ζPrime[ii] = P.ζ̄ * (ζ/P.ζ̄)^P.ρ_ζ * exp(P.eNodes[ii])

        # Compute required aggregate variables at time t+1
        _, _, πPrime[ii], wPrime, HPrime[ii], _, TPrime[ii], _ =
            computeAggregateVariables(P, DSS, RStarPrime, ζPrime[ii], πwALMInterpol, EπwCondALMInterpol)

    end

    # Initialize new bond policy
    bPolicy = zeros(P.bGridSize, P.sGridSize)
    bPolicyUpdate = similar(bPolicy)
    bPolicyError = similar(bPolicy)
    wealth = similar(bPolicy)

    @threads for idx in CartesianIndices(bPolicyUpdate)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_s = idx[2]

        # Get the states variables for the current node in the state space
        b = P.bGrid[i_b]
        s = P.sGrid[i_s]

        # Compute wealth
        wealth[idx] = computeIndividualCashOnHand(P, b, s, R, π, w, H, T) 

        # Fill bond policy function
        bPolicy[idx] = bPolicyInterpol(b, RStar, i_s, ζ)

    end

    # Initialize policy function iteration
    iter = 1
    dist = 10.0

    # Do bond policy funtion iteration
    while(dist > P.tol)

        # Compute the proposal for new asset policy function
        @threads for idx in CartesianIndices(bPolicy)

            # Determne indices of 1D grids
            i_b = idx[1]
            i_s = idx[2]

            # Update the policy
            bPolicyUpdate[idx] = updateBondPolicy(P, DSS, bPolicyInterpol, bPolicy[idx],
                wealth[idx], RStarPrime, πPrime, wPrime, HPrime, TPrime, ζPrime, ζ, i_s)

        end

        # Check the distance between current iteration and the previous one
        @. bPolicyError = abs.(bPolicyUpdate - bPolicy)
        dist = maximum(bPolicyError)

        # Update the bond policy function
        @. bPolicy = P.λᵇ * bPolicyUpdate + (1-P.λᵇ) * bPolicy
        iter = iter+1

        # Limit the number of iterations (since in some cases it does not converge)
        if iter > 10000
            if P.showWarningsAndInfo
                @warn "Policy function iteration in check() has reached maximum number of iterations: dist = $(dist), π = $(π)"
            end
            break
        end

    end

    # Interpolate the new bond policy function
    bPolicyNewInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), bPolicy, extrapolation_bc = Line())

    # Update the bond distribution
    bPrimeCross = propagateBondDistribution(P, bCross, bPolicyNewInterpol)

    # Compute the mean of the distribution
    B = sum(P.bDenseGrid' * bPrimeCross)

    # Compute the error in the mean of the bond distribution
    errorB = B - DSS.B

    return errorB, B, bPrimeCross, RStarPrime, H

end


"""
    propagateBondDistribution(P, bCross, bPolicyInterpol)

Computes the bond distribution in the next period.

"""
function propagateBondDistribution(P, bCross, bPolicyInterpol)

    # Initialize the updated bond distribution
    bPrimeCross = copy(bCross)

    # Update the bond distribution
    propagateBondDistribution!(P, bPrimeCross, bCross, bPolicyInterpol)

    return bPrimeCross

end


"""
    propagateBondDistribution!(P, bPrimeCross, bCross, bPolicyInterpol)

Computes the bond distribution in the next period.

"""
function propagateBondDistribution!(P, bPrimeCross, bCross, bPolicyInterpol)

    # Update the bond distribution
    for idx in CartesianIndices(bPrimeCross)

        # Get 1D indices
        i_b = idx[1]
        i_s = idx[2]

        # Get the current bond policy
        b = bPolicyInterpol(P.bDenseGrid[i_b], i_s)

        # Get the grid points that are below and above the current bond policy
        i_bLow, i_bUp = getAdjacentBondGridPoints(P, b)

        # Compute the weight which depend on the distance from the adjacent grid points
        ω = getReassignmentWeight(P, b, i_bLow, i_bUp)

        # Redistribute the mass of the histogram according to the bond policy and transition probbilities
        for i_sp in 1:P.sGridSize

            bPrimeCross[i_bLow, i_sp] += P.Ω[i_s, i_sp] * ω * bCross[i_b, i_s]
            bPrimeCross[i_bUp, i_sp] += P.Ω[i_s, i_sp] * (1-ω) * bCross[i_b, i_s]
            bPrimeCross[i_b, i_s] -= P.Ω[i_s, i_sp] * bCross[i_b, i_s]

        end

    end

    nothing

end


"""
    propagateBondDistributionThreaded!(P, bPrimeCross, bCross, bPolicyInterpol)

Computes the bond distribution in the next period. This is a modified version of
propagateBondDistribution! which uses multiple threads. However, it is not or only
marginally faster than propagateBondDistribution! on an 8-core CPU.

"""
function propagateBondDistributionThreaded!(P, bPrimeCross, bCross, bPolicyInterpol)

    # Initilize a matrix for each available thread
    bPrimeCrossThreads = [zeros(size(bPrimeCross)) for ii in 1:Threads.nthreads()]

    @threads for idx in CartesianIndices(bPrimeCross)

        # Get 1D indices
        i_b = idx[1]
        i_s = idx[2]

        # Get the current bond policy
        b = bPolicyInterpol(P.bDenseGrid[i_b], i_s)

        # Get the grid points that are below and above the current bond policy
        i_bLow, i_bUp = getAdjacentBondGridPoints(P, b)

        # Compute the weight which depend on the distance from the adjacent grid points
        ω = getReassignmentWeight(P, b, i_bLow, i_bUp)

        # Redistribute the mass of the histogram according to the bond policy and transition probbilities
        for i_sp in 1:P.sGridSize

            bPrimeCrossThreads[Threads.threadid()][i_bLow, i_sp] += P.Ω[i_s, i_sp] * ω * bCross[i_b, i_s]
            bPrimeCrossThreads[Threads.threadid()][i_bUp, i_sp] += P.Ω[i_s, i_sp] * (1-ω) * bCross[i_b, i_s]
            bPrimeCrossThreads[Threads.threadid()][i_b, i_s] -= P.Ω[i_s, i_sp] * bCross[i_b, i_s]

        end

    end

    # Sum everything up
    bPrimeCross .= bPrimeCross .+ sum(bPrimeCrossThreads)

    nothing

end


"""
    computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))

Computes "aggregate" variables for given states and perceived laws of motion.

"""
function computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))
    # Use the ALM to nowcast inflation if πw is not supplied

    # Real wage
    w = 1.0

    # Inflation
    π = πw

    # Use the ALM to predict the wage inflation expectation term
    Eπw = evaluateALMExpectation(P, EπwCondALMInterpol, RStar, ζ)

    # Compute labor supply from the wage inflation equation
    if P.indexRotemberg
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 - P.τ) * w + P.θ/P.ε * (log(πw/DSS.πw) - P.β̃ * Eπw)))^(1/(P.σ+P.ν))
    else
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 - P.τ) * w + P.θ/P.ε * (log(πw) - P.β̃ * Eπw)))^(1/(P.σ+P.ν))
    end

    # Determine nominal interest rate set by central bank in previous period
    R = applyLowerBound(P, RStar)

    # Transfers
    T = P.τ * w * H - (R/π - 1) * DSS.B

    # Output
    Y = H

    # Compute the nominal interest rate
    _, RStarPrime = monetaryPolicyRule(P, DSS, π, Y, RStar) # Note: RPrime refers here to R_t while R refers to the state R_{t-1}

    return πw, Eπw, π, w, H, Y, T, RStarPrime

end


"""
    checkInflationFeasibility(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))

Checks whether (wage) inflation rate is feasible (i.e. whether it yields positive labor supply).
Negative marginal costs causes issues when computing η in computeAggregateVariables().

"""
function checkInflationFeasibility(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol; πw = evaluateALM(πwALMInterpol, RStar, ζ))

    # Real wage
    w = 1.0

    # Use the ALM to predict the inflation expectation term
    Eπw = evaluateALMExpectation(P, EπwCondALMInterpol, RStar, ζ)

    # Compute labor supply from the wage inflation equation
    if P.indexRotemberg
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 - P.τ) * w + P.θ/P.ε * (log(πw/DSS.πw) - P.β̃ * Eπw)))^(1/(P.σ+P.ν))
    else
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 - P.τ) * w + P.θ/P.ε * (log(πw) - P.β̃ * Eπw)))^(1/(P.σ+P.ν))
    end

    # Check whetherlabor supply is positive
    if H < 0.0 || πw < 0.0
        isInflationRateFeasible = false
    else
        isInflationRateFeasible = true
    end

    return isInflationRateFeasible

end


"""
    precomputeStaticPartsIndividualProblem(P, DSS, πwALMInterpol, EπwCondALMInterpol)

Precomputes all matrices that don't change during policy function iteration.

"""
function precomputeStaticPartsIndividualProblem(P, DSS, πwALMInterpol, EπwCondALMInterpol)

    # Initialize time t matrices
    π = zeros(P.RGridSize, P.ζGridSize)
    w = NaN # This is just a dummy value
    H = similar(π)
    T = similar(π)
    RStarPrime = similar(π)
    wealth = zeros(P.bGridSize, P.RGridSize, P.sGridSize, P.ζGridSize)

    # Initialize time t+1 matrices
    πPrime = zeros(P.RGridSize, P.ζGridSize, P.nGHNodes)
    wPrime = NaN # This is just a dummy value
    HPrime = similar(πPrime)
    TPrime = similar(πPrime)
    ζPrime = zeros(P.ζGridSize, P.nGHNodes)

    # Compute aggregate variables for each node in the state space
    @threads for idx in CartesianIndices(π)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]

        # Get the states variables for the current node in the state space
        RStar = P.RGrid[i_R]
        ζ = P.ζGrid[i_ζ]

        # Compute required aggregate variables
        _, _, π[idx], w, H[idx], _, T[idx], RStarPrime[idx] =
            computeAggregateVariables(P, DSS, RStar, ζ, πwALMInterpol, EπwCondALMInterpol)
            # Note: w will be continously overwrittern by the same value becauise w is always equal 1

        # Compute time t+1 variables which are used for computing expectations
        for ii in 1:P.nGHNodes

            # Aggregate shock in the next period
            ζPrime[i_ζ, ii] = P.ζ̄ * (ζ/P.ζ̄)^P.ρ_ζ * exp(P.eNodes[ii])

            # Compute required aggregate variables
            _, _, πPrime[i_R, i_ζ, ii], wPrime, HPrime[i_R, i_ζ, ii], _, TPrime[i_R, i_ζ, ii], _ =
                computeAggregateVariables(P, DSS, RStarPrime[i_R, i_ζ], ζPrime[i_ζ, ii], πwALMInterpol, EπwCondALMInterpol)
                # Note: wPrime will be continously overwrittern by the same value becauise wPrime is always equal 1

        end

    end

    # Compute individual variables for each node in the state space
    @threads for idx in CartesianIndices(wealth)

        # Determne indices of 1D grids
        i_b = idx[1]
        i_R = idx[2]
        i_s = idx[3]
        i_ζ = idx[4]

        # Get the states variables for the current node in the state space
        b = P.bGrid[i_b]
        RStar = P.RGrid[i_R]
        s = P.sGrid[i_s]
        ζ = P.ζGrid[i_ζ]

        # Determine nominal interest rate set by central bank in previous period
        R = applyLowerBound(P, RStar)

        # Compute cash on hand
        wealth[idx] = computeIndividualCashOnHand(P, b, s, R, π[i_R, i_ζ], w, H[i_R, i_ζ], T[i_R, i_ζ]) 

    end

    return RStarPrime, wealth, H, πPrime, wPrime, HPrime, TPrime, ζPrime

end


"""
    plotComparison(P, S, DSS, πwSim, πwSimALM, EπwCondSim, EπwCondSimALM, RStarSim,
        HSim, YSim, ζSim, bCross, πwALM, EπwCondALM, NNπwALMMat, NNEπwCondALMMat, algorithmProgress,
        NNπwTrainingData, NNEπwCondTrainingData)

Plots several of the simulated series to give an overview of how well the
solution algorithm is doing.

"""
function plotComparison(P, S, DSS, πwSim, πwSimALM, EπwCondSim, EπwCondSimALM, RStarSim,
    HSim, YSim, ζSim, bCross, πwALM, EπwCondALM, NNπwALMMat, NNEπwCondALMMat, algorithmProgress,
    NNπwTrainingData, NNEπwCondTrainingData)

    # Interpolation of PLMs
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())
    NNπwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), NNπwALMMat, extrapolation_bc = Line())
    NNEπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), NNEπwCondALMMat, extrapolation_bc = Line())

    # Simulate the predictions of the PLM conditional on the equilibrium nominal interest rate and ζ
    # Note these series will be 2 periods shorter than πwSim, etc.
    πwSimNN = similar(πwSim)
    EπwCondSimNN = similar(πwSim)
    for tt in 2:P.T-3
        πwSimNN[tt] = evaluateALM(NNπwALMInterpol, RStarSim[tt-1], ζSim[tt])
        EπwCondSimNN[tt] = evaluateALM(NNEπwCondALMInterpol, RStarSim[tt-1], ζSim[tt], ζSim[tt+1])
    end
    πwSimNN[end] = evaluateALM(NNπwALMInterpol, RStarSim[end-1], S.ζ[end])
    πwSimNN = πwSimNN[2:end-1]
    EπwCondSimNN = EπwCondSimNN[2:end-1]

    # Plot the perceived law of motion for inflation
    p1 = plot(πwSim, label = "Simulated", ylabel = "Inflation")
    plot!(p1, πwSimALM, label = "Perceived")
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)

    # Plot the perceived law of motion for inflation expectation term
    p2 = plot(EπwCondSim, label = "Simulated", ylabel = "Cond. Inflation Exp. Term", legend = :none)
    plot!(p2, EπwCondSimALM, label = "Perceived")
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)

    # Show errors for matrix that represents PLM
    p3 = plot(πwSimALM .- πwSim, label = "Inflation", ylabel = "Errors")
    plot!(p3, EπwCondSimALM .- EπwCondSim, label = "Inflation Exp. Term")
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)

    # Show errors for matrix that represents PLM
    p8 = histogram(πwSimALM[P.burnIn+1:end] .- πwSim[P.burnIn+1:end], label = "Inflation", ylabel = "Errors (Mat)", fillalpha = 0.6)
    histogram!(p8, EπwCondSimALM[P.burnIn+1:end] .- EπwCondSim[P.burnIn+1:end], label = "Inflation Exp. Term", fillalpha = 0.6)

    # Show errors for Neural Network PLM
    p9 = histogram(πwSimNN[P.burnIn+1:end] .- πwSim[P.burnIn+2:end-1], label = "Inflation", ylabel = "Errors (NN)", fillalpha = 0.6)
    histogram!(p9, EπwCondSimNN[P.burnIn+1:end] .- EπwCondSim[P.burnIn+2:end-1], label = "Inflation Exp. Term", fillalpha = 0.6)

    # Output
    p4 = plot(HSim, ylabel = "Agg. Labor", legend = :none)
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)

    # Nominal Interest Rate
    p5 = plot(RStarSim, ylabel = "Nominal Rate", label = L"R^{*}_{t-1}")
    plot!(applyLowerBound.(Ref(P), RStarSim), label = L"R_{t-1}")
    if P.ZLBLevel == 1.0
        plot!(NaN .* RStarSim, label = "$(P.bindingZLB ? "" : "Shadow ")ZLB Freq.: $(round(sum(RStarSim .<= 1.0) / length(RStarSim) * 100, digits = 2))%")
    else
        plot!(NaN .* RStarSim, label = "Shadow ZLB Freq.: $(round(sum(RStarSim .<= 1.0) / length(RStarSim) * 100, digits = 2))%")
        plot!(NaN .* RStarSim, label = "$(P.bindingZLB ? "" : "Shadow ")ELB ($(round(log(P.ZLBLevel)*400, digits = 2))%) Freq.: $(round(sum(RStarSim .<= P.ZLBLevel) / length(RStarSim) * 100, digits = 2))%")
    end
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)
    hline!([P.RMin], linestyle = :dot, color = :olive, label = :none)
    hline!([P.RMax], linestyle = :dot, color = :olive, label = :none)

    # Aggregate Shock
    p6 = plot(ζSim, ylabel = "Aggregate Shock ($(P.aggShockType))", legend = :none)
    vline!([P.burnIn], linestyle = :dash, color = :grey, label = :none)
    hline!([P.ζMin], linestyle = :dot, color = :grey, label = :none)
    hline!([P.ζMax], linestyle = :dot, color = :grey, label = :none)

    # Bond distribution
    p7 = bar(P.bDenseGrid, sum(DSS.bCross, dims = 2), ylabel = "Bond Distribution", linecolor = :steelblue, legend = :none)


    # Plot inflation PLM
    pp1 = surface(P.RDenseGrid, P.ζDenseGrid, πwALM',
        xlabel = L"R^{*}_{t-1}",
        ylabel = L"\zeta_t",
        zlabel = L"\pi_t",
        title = "Inflation",
        camera = (-60,30),
        legend = :none, margin = 5Plots.PlotMeasures.mm, cbar = :none)

    # Add the prediction of the NN
    surface!(pp1, P.RDenseGrid, P.ζDenseGrid, NNπwALMMat', c = cgrad(:blues), alpha = 0.5)

    # Add simulated data points and differentiate between cases where the ZLB is binding and where it's not
    ZLBCheck = checkZLB.(Ref(P), Ref(DSS), πwSim[2:end], YSim[2:end], RStarSim[1:end-1])
    RStarSimZLB =  RStarSim[1:end-1][ZLBCheck .== 1]
    πwSimZLB = πwSim[2:end][ZLBCheck .== 1]
    ζSimZLB =  ζSim[2:end][ZLBCheck .== 1]
    RStarSimNotZLB = RStarSim[1:end-1][ZLBCheck .== 0]
    πwSimNotZLB = πwSim[2:end][ZLBCheck .== 0]
    ζSimNotZLB = ζSim[2:end][ZLBCheck .== 0]

    scatter3d!(pp1, RStarSimZLB, ζSimZLB, πwSimZLB, color = :red, markersize = 3)
    scatter3d!(pp1, RStarSimNotZLB, ζSimNotZLB, πwSimNotZLB, color = :green, markersize = 3)

    # Plot another panel with the inflation PLM and training data
    pp2 = surface(P.RDenseGrid, P.ζDenseGrid, NNπwALMMat',
        xlabel = L"R^{*}_{t-1}",
        ylabel = L"\zeta_t",
        zlabel = L"\pi_t",
        camera = (-60,30),
        legend = :none,
        margin = 5Plots.PlotMeasures.mm,
        c = cgrad(:blues),
        cbar = :none,
        alpha = 0.5)

    if P.projectDataOntoGridKnots && P.approximationTypeALM != :LinearRegression
        RStarTrain = [x[1] * NNπwTrainingData.normFactors.RStar.scale + NNπwTrainingData.normFactors.RStar.location for x in NNπwTrainingData.inputs]
        ζTrain = [x[2] * NNπwTrainingData.normFactors.ζ.scale + NNπwTrainingData.normFactors.ζ.location for x in NNπwTrainingData.inputs]
        πwTrain = [x[1] * NNπwTrainingData.normFactors.πw.scale + NNπwTrainingData.normFactors.πw.location for x in NNπwTrainingData.outputs]
        scatter3d!(pp2, RStarTrain, ζTrain, πwTrain, markersize = 3)
    end

    # Add contour plots of inflation PLM
    pp3 = contour(P.RDenseGrid, P.ζDenseGrid, πwALM',
        xlabel = L"R^{*}_{t-1}",
        ylabel = L"\zeta_t",
        label = "",
        fill = true,
        margin = 5Plots.PlotMeasures.mm,
        cbar = :none)


    # Plot inflation expectation term ALM
    pp4 = surface(P.RDenseGrid, P.ζDenseGrid, (RR,ζζ) -> evaluateALMExpectation(P, EπwCondALMInterpol, RR, ζζ),
        xlabel = L"R^{*}_{t-1}",
        ylabel = L"\zeta_t",
        zlabel = L"E(\pi_{t+1}\frac{H_{t+1}}{H_t})",
        title = "Inflation Expectation",
        camera = (-60,30),
        legend = :none,
        margin = 5Plots.PlotMeasures.mm,
        cbar = :none)

    # Add the prediction of the NN
    surface!(pp4, P.RDenseGrid, P.ζDenseGrid, (RR,ζζ) -> evaluateALMExpectation(P, NNEπwCondALMInterpol, RR, ζζ), c = cgrad(:blues), alpha = 0.5)

    # Plot another panel with the inflation PLM and training data
    pp5 = surface(P.ζDenseGrid, P.ζDenseGrid, (ζζ, ζζp) -> NNEπwCondALMInterpol(DSS.R, ζζ, ζζp),
        xlabel = L"\zeta_{t}",
        ylabel = L"\zeta_{t+1}",
        zlabel = L"\pi_{t+1}\frac{H_{t+1}}{H_t}",
        camera = (-60,30),
        legend = :none,
        margin = 5Plots.PlotMeasures.mm,
        c = cgrad(:blues),
        cbar = :none,
        alpha = 0.5)

    if P.projectDataOntoGridKnots && P.approximationTypeALM != :LinearRegression
        RStarTrain = [x[1] * NNEπwCondTrainingData.normFactors.RStar.scale + NNEπwCondTrainingData.normFactors.RStar.location for x in NNEπwCondTrainingData.inputs]
        ζTrain = [x[2] * NNEπwCondTrainingData.normFactors.ζ.scale + NNEπwCondTrainingData.normFactors.ζ.location for x in NNEπwCondTrainingData.inputs]
        ζpTrain = [x[3] * NNEπwCondTrainingData.normFactors.ζPrime.scale + NNEπwCondTrainingData.normFactors.ζPrime.location for x in NNEπwCondTrainingData.inputs]
        EπwCondTrain = [x[1] * NNEπwCondTrainingData.normFactors.EπwCond.scale + NNEπwCondTrainingData.normFactors.EπwCond.location for x in NNEπwCondTrainingData.outputs]
        scatter3d!(pp5, ζTrain, ζpTrain, EπwCondTrain, markersize = 3)
    end

    # Add contour plots of inflation PLM
    pp6 = contour(P.RDenseGrid, P.ζDenseGrid, (RR,ζζ) -> evaluateALMExpectation(P, EπwCondALMInterpol, RR, ζζ),
        xlabel = L"R^{*}_{t-1}",
        ylabel = L"\zeta_t",
        label = "",
        fill = true,
        margin = 5Plots.PlotMeasures.mm,
        cbar = :none)


    # Plot convergence results
    processStats(x) = mean(x) # For the case where multiple statistics are stored in atuple we take the average
    R2πw = [processStats(x.NNπwStats.training.R2) for x in algorithmProgress]
    MSEπw = [processStats(x.NNπwStats.training.MSE) for x in algorithmProgress]
    R2EπwCond = [processStats(x.NNEπwCondStats.training.R2) for x in algorithmProgress]
    MSEEπwCond = [processStats(x.NNEπwCondStats.training.MSE) for x in algorithmProgress]
    R2πwValidation = [processStats(x.NNπwStats.validation.R2) for x in algorithmProgress]
    MSEπwValidation = [processStats(x.NNπwStats.validation.MSE) for x in algorithmProgress]
    R2EπwCondValidation = [processStats(x.NNEπwCondStats.validation.R2) for x in algorithmProgress]
    MSEEπwCondValidation = [processStats(x.NNEπwCondStats.validation.MSE) for x in algorithmProgress]
    distπwVisited = [processStats(x.NNπwStats.dist.visited) for x in algorithmProgress]
    distπwNorm = [processStats(x.NNπwStats.dist.norm) for x in algorithmProgress]
    distEπwCondVisited = [processStats(x.NNEπwCondStats.dist.visited) for x in algorithmProgress]
    distEπwCondNorm = [processStats(x.NNEπwCondStats.dist.norm) for x in algorithmProgress]

    ppp1 = plot(R2πw, label = "Wage Inflation (Training)", ylabel = "R^2 (NN)", xlabel = "Iteration", marker = :circle, markersize = 3, legend = :none)
    plot!(ppp1, R2EπwCond, label = "Wage Inflation Exp. Term (Training)", marker = :circle, markersize = 3)
    if P.projectDataOntoGridKnots
        plot!(ppp1, R2πwValidation, label = "Wage Inflation (Validation)", marker = :circle, markersize = 3)
        plot!(ppp1, R2EπwCondValidation, label = "Wage Inflation Exp. Term (Validation)", marker = :circle, markersize = 3)
    end

    ppp2 = plot(MSEπw, label = "Wage Inflation (Training)", ylabel = "MSE (NN)", xlabel = "Iteration", marker = :circle, markersize = 3)
    plot!(ppp2, MSEEπwCond, label = "Wage Inflation Exp. Term (Training)", marker = :circle, markersize = 3)
    if P.projectDataOntoGridKnots
        plot!(ppp2, MSEπwValidation, label = "Wage Inflation (Validation)", marker = :circle, markersize = 3)
        plot!(ppp2, MSEEπwCondValidation, label = "Wage Inflation Exp. Term (Validation)", marker = :circle, markersize = 3)
    end

    ppp3 = plot(distπwVisited, label = "Wage Inflation", legend = :none, ylabel = "Convergence criteria", xlabel = "Iteration", marker = :circle, markersize = 3)
    plot!(ppp3, distEπwCondVisited, label = "Wage Inflation Exp. Term", legend = :none, marker = :circle, markersize = 3)
    hline!([P.tolALM], linestyle = :dash)

    # Combine the plots
    l = @layout [grid(9,1){0.6w} grid(3,2);
     grid(1,2){0.1h} a{0.1h}]

    p = plot(p1, p2, p3, p8, p9, p4, p5, p6, p7, pp1, pp4, pp2, pp5, pp3, pp6, ppp1, ppp2, ppp3, layout = l,
        legendfontsize = 5, guidefontsize = 6, tickfontsize = 6, size = (1600, 900))

    # If plots are only saved at the end, don't display the plot
    if P.showProgressPlots
        display(p)
    end

    return p

end
