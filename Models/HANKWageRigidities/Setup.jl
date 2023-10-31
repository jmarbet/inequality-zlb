"""
    settings()

Defines model parameters and several settings for the solution algorithm.

"""
@with_kw struct settings

    # Model Parameters
    β::Float64 = 0.99398        # Discount factor
    σ::Float64 = 1.0            # Risk aversion coefficient
    ν::Float64 = 1.0            # Inverse Frisch labor supply elasticity
    χ::Float64 = (7.67-1)/7.67  # Labor disutility
    b̲::Float64 = -(7.67-1)/7.67 # Credit limit
    ϕₗ::Float64 = 2.0            # Taylor rule coefficient on inflation (π_t < π̃)
    ϕₕ::Float64 = ϕₗ             # Taylor rule coefficient on inflation (π_t >= π̃)
    ϕʸ::Float64 = 0.0           # Taylor rule coefficient on output
    ρ_R::Float64 = 0.0          # Inertia in Taylor rule
    useRStarForTaylorInertia::Bool = false # Determines whether R or RStar us used for inertia in the Taylor rule
    useLegacyRStar::Bool = true # Contsrains RStar to be larger or equal than ZLBLevel (can be used to get legacy model behavior)
    π̃::Float64 = exp(0.02/4)    # Gross steady state inflation / Inflation target
    indexRotemberg::Bool = true # If true, indexes adjustment cost to steady state inflation
    bindingZLB::Bool = true     # If true, zero lower bound on nominal interest rates exists in the economy
    ZLBLevel::Float64 = 1.0     # Gross level below which "zero" lower bound is binding
    β̃::Float64 = β              # Discount factor of unions
    θ::Float64 = 79.41          # Rotemberg wage adjustment cost coefficient
    ε::Float64 = 7.67           # Elasticity of substitution across differentiated labor services
    τ::Float64 = 0.0            # Labor tax rate
    B::Float64 = 1.0            # Net supply of bonds

    # Parameters for idiosyncratic and aggregate states
    nGHNodes::Int64 = 10        # Nodes for the Gauss-Hermite quadrature
    ξ̄::Float64 = 1.0            # Preference shock in DSS
    q̄::Float64 = 1.0            # Risk premium / discount factor shock in DSS
    aggShockType::Symbol = :Preference # Type of the aggregate shock: :Preference, :DiscountFactor
    ζ̄::Float64 = 1.0            # Aggregate shock in DSS
    ρ_ζ::Float64 = 0.6          # Persistence of aggregate log-AR(1) process
    σ̃_ζ::Float64 = 0.01175      # Standard deviation of aggregate shock

    # Compute Gauss-Hermite quadrature nodes and weights (Note: weights include the π^(-1/2) term)
    tmpGH::Tuple{Array{Float64,1},Array{Float64,1}} = hermite(nGHNodes)
    eNodes::Array{Float64,1} = sqrt(2) * σ̃_ζ * tmpGH[1]
    eWeights::Array{Float64,1} = tmpGH[2] / sqrt(pi)

    # Construct the idiosyncratic transition matrix
    Ω::Array{Float64,2} = convertAR1Rouwenhorst(0.966, 0.017, 3)[1]
    P0::Array{Float64,1} = computeStationaryDistribution(Ω, 1e-8) # Stationary distribution of the Markov chain

    # Simulation settings
    alwaysStartFromDSS::Bool = true # true: simulation starts from DSS
                                # false: simulation continues from last period from prev. simulation
    T::Int64 = 1100             # Number of periods
    burnIn::Int64 = 100         # Number of discared burn-in periods

    # Policy function iteration settings
    maxPolicyIterations::Int64 = 1000 # Maximum number of iterations for bond policy function
    λᵇ::Float64 = 0.7           # Learning parameter for bond policy function
    λALM::Float64 = 0.3         # Learning parameter for the aggregate law of motion (ALM)
    tol::Float64 = 1e-6         # Tolerance bond policy function
    tolALM::Float64 = 1e-5      # Tolerance aggregate law of motion
    tolDSSDistribution::Float64 = 1e-6 # Tolerance DSS bond distribution
    tolDSSRate::Float64 = 1e-8  # Tolerance DSS nominal interest rate

    # Relaxation parameters
    λALMInit::Float64 = λALM    # Initial value for the λALM learning parameter
    λALMDecay::Float64 = 0.9    # Decay of the λALM learning parameter
    λALMFinal::Float64 = 0.1    # Minimum value for the λALM learning parameter
    λALMAltIteration::Int64 = 3 # Every xth iteration alternative relaxation parameters are used (0 = never)
    λALMInitAlt::Float64 = λALM # Alternaive initial value for the λALM learning parameter
    λALMDecayAlt::Float64 = 0.95 # Alternaive decay of the λALM learning parameter
    λALMFinalAlt::Float64 = 0.1 # Alternaive minimum value for the λALM learning parameter

    # Additional settngs
    showWarningsAndInfo::Bool = true
    showPolicyIterations::Bool = true
    showProgressPlots::Bool = false
    saveAlgorithmProgress::Bool = true
    appendIterationToFilename::Bool = true
    removeIntermediateResultsAtEnd::Bool = true
    saveFigureAtEnd::Bool = true
    saveFigureEachIteration::Bool = true

    # Grid for bonds
    bGridSize::Int64 = 100
    bDenseGridSize::Int64 = 1001
    bMin::Float64 = b̲
    bMax::Float64 = updateBondGridBounds(bMin, 8.2, bDenseGridSize; showInfo = showWarningsAndInfo)
    bGridLin::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(bMin, stop=bMax, length=bGridSize)
    xGridLin::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
            range(0, stop=0.5, length=bGridSize)
    yGrid::Array{Float64,1} = xGridLin.^2/maximum(xGridLin.^2)
    bGrid::Array{Float64,1} = getBondGrid(bMin, bMax, bGridSize, 0.4)#@. bMin + (bMax-bMin) * yGrid

    # Grid for (desired) nominal interest rate RStar
    RGridSize::Int64 = 61
    RMin::Float64 = bindingZLB && useLegacyRStar ? ZLBLevel : 0.98
    RMax::Float64 = 1.05
    RGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(RMin, stop=RMax, length=RGridSize)

    # Aggregate shock grids
    ζGridSize::Int64 = 41
    ζMin::Float64 = 1 - max(1-exp(log(ζ̄) - 3.5 * σ̃_ζ/sqrt(1-ρ_ζ^2)), exp(log(ζ̄) + 3.5 * σ̃_ζ/sqrt(1-ρ_ζ^2)) - 1)
    ζMax::Float64 = 1 + max(1-exp(log(ζ̄) - 3.5 * σ̃_ζ/sqrt(1-ρ_ζ^2)), exp(log(ζ̄) + 3.5 * σ̃_ζ/sqrt(1-ρ_ζ^2)) - 1)
    ζGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(ζMin, stop=ζMax, length=ζGridSize)

    # Dense grids used for representing the PLM and the bond distribution
    RDenseGridSize::Int64 = 101
    RDenseGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(RMin, stop=RMax, length=RDenseGridSize)

    ζDenseGridSize::Int64 = 101
    ζDenseGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(ζMin, stop=ζMax, length=ζDenseGridSize)

    # Note: bDenseGridSize is defined after bGridSize
    bDenseGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(bMin, stop=bMax, length=bDenseGridSize)
    ΔbDense::Float64 = bDenseGrid[2] - bDenseGrid[1]
    bZero::Int64 = sum(bDenseGrid .< 0) + 1 # Index of zero wealth (used for assigning newborn agents)

    # Idiosyncratic productivity shock grid
    sGrid::Array{Float64,1} = 1 .+ collect(convertAR1Rouwenhorst(0.966, 0.017, 3)[2])
    sGridSize::Int64 = length(sGrid)

    # Approximation of aggregate law of motion
    approximationTypeALM::Symbol = :NeuralNetwork # :NeuralNetwork or :LinearRegression or :DualRegression or :QuadRegression

    # Neural Network data preparation settings
    projectDataOntoGridKnots::Bool = false
    RDataPrepGridSize::Int64 = 31
    ζDataPrepGridSize::Int64 = 31
    RDataPrepGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(RMin, stop = RMax, length = RDataPrepGridSize)
    ζDataPrepGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(ζMin, stop = ζMax, length = ζDataPrepGridSize)

    # Neural Network training settings
    enableMultipleNNStarts::Bool = false # If true, initializes multiple NNs and continues with best performing one
    NNStarts::Int64 = 10            # Number of NNs that are initialized if enableMultipleNNStarts is true
    criteriaMultipleNNstarts::Symbol = :validationMSE # Criteria used to choose between initialized NNs: :validationMSE or :trainingMSE
    initalizationSeed::Int64 = -1   # Random number seed for NN initialization (-1 -> use Julia default);
                                    # Note that this is ignored if enableMultipleNNStarts is set to true.
    initalizationType::Symbol = :standardnormal # How NN weights are initialized: :standardnormal, :he, :xavier
    gradientDescent::Symbol = :stochastic   # :fullbatch, :stochastic or :minibatch
    batchSize::Int64 = 100          # Batch size for minibatch gradient descent
    activationFunction::Symbol = :softplus # :softplus, :sigmoid, :relu
    nHiddenNodes_πw::Int64 = 16      # Hidden nodes in NN for πw PLM
    nHiddenNodes_EπwCond::Int64 = 16 # Hidden nodes in NN for EπwCond PLM
    reg_πw::Float64 = 0.0            # Regularization param. for πw PLM
    reg_EπwCond::Float64 = 0.0       # Regularization param. for EπwCond PLM
    baseLearningSpeed::Float64 = 0.005 # Base learning speed
    learningSpeedType::Symbol = :decay # :optimal: chooses speed that maximizes gradient descent
                                    # :improved: adjusts speed to improve gradient descent
                                    # :fixed: always uses baseLearningSpeed
                                    # :decay: uses decay settings below to adjust learning rate accross
                                    # iterations of the main algorithm (across epochs it's fixed)
    learningSpeedInit::Float64 = baseLearningSpeed
    learningSpeedDecay::Float64 = 0.9
    learningSpeedFinal::Float64 = 0.00005
    epochs::Int64 = 10000            # Number of times the whole data set is used to train the NN
    IONormalization::Symbol = :minMax44     # :standard: substract mean and divide by standard deviation
                                    # :minMax01: all values between 0 and 1
                                    # :minMax11: all values between -1 and 1
                                    # :minMax44: all values between -4 and 4
    computeMSEDuringTraining_πw::Symbol = :none  # Whether to compute MSE after each epoch (can be computationally costly for large datasets)
                                    # :testOnly, validationOnly, :both, :none
    computeMSEDuringTraining_EπwCond::Symbol = :none # Whether to compute MSE after each epoch (can be computationally costly for large datasets)
                                    # :testOnly, validationOnly, :both, :none

    # Filename under which the results are saved
    filenameExt::String = bindingZLB ? "ZLB" : "NoZLB"
    filenameFolder::String = "HANKWageRigidities"
    filenamePrefix::String = "HANKWageRigidities"
    filename::String = "Results/$(filenameFolder)/$(filenamePrefix)_$(filenameExt).bson"

end


"""
    validateSettings(P)

Checks for certain setting combinations that are not allowed.

"""
function validateSettings(P)

    # Check that multiple choice settings are valid
    if P.gradientDescent ∉ (:fullbatch, :stochastic, :minibatch)
        error("Settings error: gradientDescent option unknown ($(P.gradientDescent))")
    end

    if P.learningSpeedType ∉ (:decay, :fixed, :improved, :optimal)
        error("Settings error: learningSpeedType option unknown ($(P.learningSpeedType))")
    end

    if P.approximationTypeALM ∉ (:NeuralNetwork, :LinearRegression, :DualRegression, :QuadRegression)
        error("Settings error: approximationTypeALM option unknown ($(P.approximationTypeALM))")
    end

    if P.activationFunction ∉ (:softplus, :sigmoid, :relu)
        error("Settings error: activationFunction option unknown ($(P.activationFunction))")
    end

    # There is no code implemented for the following cases
    if P.projectDataOntoGridKnots && P.approximationTypeALM ∈ (:LinearRegression, :DualRegression, :QuadRegression)
        error("Settings error: approximationTypeALM option ($(P.approximationTypeALM)) not compatible with projectDataOntoGridKnots option ($(P.projectDataOntoGridKnots))")
    end

    if P.useLegacyRStar && P.useRStarForTaylorInertia
        error("Settings error: useLegacyRStar option ($(P.useLegacyRStar)) not compatible with useRStarForTaylorInertia option ($(P.useRStarForTaylorInertia))")
    end

    nothing

end


"""
    computeStationaryDistribution(Π, tolStationary)

Computes stationary distribution of Markov chain.

"""
function computeStationaryDistribution(Π, tolStationary)

    nStates = size(Π, 1)
    P0 = ones(nStates)/nStates
    P1 = similar(P0)
    dist = 10.0

    while(dist >= tolStationary)
        P1 .= Π' * P0
        dist = norm(P1 - P0)
        P0 .= P1
    end

    return P0

end


"""
    convertAR1Rouwenhorst(ρ, σ, N)

Converts AR(1) process to N state Markov chain using the Rouwenhorst method.

"""
function convertAR1Rouwenhorst(ρ, σ, N)

    #
    p = q = (1+ρ)/2
    ψ = σ * sqrt(N-1)

    # Compute the transition matrix
    Θ = [p 1-p; 1-q q]

    for ii in 2:N-1

        m1 = [Θ zeros(ii, 1); zeros(1, ii+1)]
        m2 = [zeros(ii, 1) Θ; zeros(1, ii+1)]
        m3 = [zeros(1, ii+1); Θ zeros(ii, 1)]
        m4 = [zeros(1, ii+1); zeros(ii, 1) Θ]
        Θ = (p * m1 + (1-p) * m2 + (1-q) * m3 + q * m4)
        Θ[2:ii, :] = Θ[2:ii, :] /2

    end

    # Compute the grid
    z = range(-ψ, stop=ψ, length=N)

    return Θ, z

end


"""
    getBondGrid(gridMin, gridMax, gridSize, negativeFraction)

Generates a grid that puts more points at the borrowing constraint and near zero.

"""
function getBondGrid(gridMin, gridMax, gridSize, negativeFraction)

    bZero = 0.0

    # Get grid points in the 3 regions
    n1 = floor(Int64, gridSize * negativeFraction / 2)
    n2 = floor(Int64, gridSize * negativeFraction / 2)
    n3 = n2
    n4 = gridSize - n1 - n2 - n3

    # Determine helper grid
    xGridLin1 = range(0, stop=0.5, length=n1)
    xGridLin2 = reverse(range(0, stop=0.5, length=n2+1))
    xGridLin3 = range(0, stop=0.5, length=n3+1)
    xGridLin4 = range(0, stop=0.5, length=n4+1)

    yGrid1 = xGridLin1.^2 / maximum(xGridLin1.^2)
    yGrid2 = (1 .- xGridLin2.^2 / maximum(xGridLin2.^2))[2:end]
    yGrid3 = (xGridLin3.^2 / maximum(xGridLin3.^2))[2:end]
    yGrid4 = (xGridLin4 / maximum(xGridLin4))[2:end]

    # Compute actual bond grid
    bGrid  = @. [gridMin + (gridMin/2-gridMin) * yGrid1;
                 gridMin/2 + (bZero-gridMin/2) * yGrid2;
                 bZero + (abs(gridMin)/2-bZero) * yGrid3;
                 abs(gridMin)/2 + (gridMax-abs(gridMin)/2) * yGrid4]

    return bGrid

end


"""
    getBondGrid(gridMin, gridMax, gridSize)

Generates a grid that puts more points at the borrowing constraint.

"""
function getBondGrid(gridMin, gridMax, gridSize)

    xGridLin = range(0, stop=0.5, length=gridSize)
    yGrid = xGridLin.^2/maximum(xGridLin.^2)
    bGrid = @. gridMin + (gridMax-gridMin) * yGrid

    return bGrid

end


"""
    updateBondGridBounds(bMin, desiredBMax, bDenseGridSize)

Computes the upper grid bound such that zero is a grid point in the dense bond grid.

"""
function updateBondGridBounds(bMin, desiredBMax, bDenseGridSize; showInfo = true)

    # Get the step size implied by the current settings
    initialStepSize = (desiredBMax - bMin) / (bDenseGridSize - 1)

    # Change the steps size to insure that zero is on of the grid points
    stepsToZero = floor(Int64, abs(bMin) / initialStepSize)
    updatedStepSize = abs(bMin) / stepsToZero

    # Compute the new upper grid bound
    bMax = updatedStepSize * (bDenseGridSize - 1) + bMin

    if bMax != desiredBMax && showInfo
        @info "Adjusted upper bond grid bound to $(bMax) from $(desiredBMax). This ensures that zero is a grid point in the dense bond grid."
    end

    return bMax

end


"""
    calibrateχ()

Computes labor disutility parameter χ such that H = 1 in the DSS.

"""
function calibrateχ(; config = tuple())

    function f!(res, x)
        P = settings(; config..., χ = x[1])
        DSS = computeDeterministicSteadyState(P)
        res[1] = DSS.H - 1.0
    end

    res = nlsolve(f!, [1.0])
    println("χ = ", res.zero[1])

end


"""
    calibrateβ()

Computes discount factor β such that r = z% in the DSS.

"""
function calibrateβ(; z = 1.5, config = tuple())

    function f!(res, x)
        P = settings(; config..., β = x[1])
        DSS = computeDeterministicSteadyState(P)
        res[1] = log(DSS.r)*400 - z
    end

    res = nlsolve(f!, [0.9975])
    println("β = ", res.zero[1])

    return res.zero[1]

end


"""
    calibrateB()

Computes B  such that the debt-to-output ratio is equal to a given value in the DSS.

"""
function calibrateB(BYRatio; config = tuple())

    function f!(res, x)
        P = settings(; config..., B = x[1])
        DSS = computeDeterministicSteadyState(P)
        res[1] = DSS.B / DSS.Y - BYRatio
    end

    res = nlsolve(f!, [1.0])
    println("B = ", res.zero[1])

    return res.zero[1]

end
