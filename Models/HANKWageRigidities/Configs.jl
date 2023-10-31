delete(nt::NamedTuple{names}, keys) where names =
    NamedTuple{filter(x -> x ∉ keys, names)}(nt)

function baselineConfigGeneral() 

    # Intermediate settings
    b̲ = -0.58
    bGridSize = 100
    bDenseGridSize = 1001
    bMin = b̲
    bMax = updateBondGridBounds(bMin, 24, bDenseGridSize; showInfo = true)
    bGrid = getBondGrid(bMin, bMax, bGridSize)

    generalConfig = (
        β = 0.9957739732021806, # 1.5% real rate
        σ = 1.0,
        ν = 1.0,
        χ = (11-1)/11, # (ε - 1) / ε  => Y = 1.0
        b̲ = b̲,
        ϕₗ = 2.5,
        ϕʸ = 0.1,
        ρ_R = 0.0,
        useRStarForTaylorInertia = false,
        indexRotemberg = true,
        θ = 100.0,
        ε = 11.0,
        τ = 0.0,
        B = 1.0,
        bGridSize = bGridSize,
        bDenseGridSize = bDenseGridSize,
        bMax = bMax,
        bMin = bMin,
        bGrid = bGrid,
        RMax = 1.03,
        ρ_ζ = 0.6, 
        σ̃_ζ = 0.01225, #for θ = 110: 0.0127,
        RDenseGridSize = 101,
        ζDenseGridSize = 101,
        λALM = 0.15,
        maxPolicyIterations = 1000,
        λALMDecay = 1.0,
        λALMAltIteration = 0,
        enableMultipleNNStarts = true,
        initalizationType = :he,
        baseLearningSpeed = 0.008,
        T = 5100,
        burnIn = 100,
        gradientDescent = :minibatch,
        batchSize = 20,
        initalizationSeed = 8723 # Neueral network random seed
    )

    return generalConfig

end


function baselineConfig() 

    configs = []
    noHomotopyList = []
    onlyUseShocksForHomotopyList = []
    skipList = []

    generalConfig = baselineConfigGeneral()

    # Model settings
    πTargets = reverse([1.7:0.1:1.9; collect(2.0:0.5:4.0)] / 100) # Inflation targets
    σLevels = [0.075, 0.095] # Idiosyncratic volatility
    skipExistingConfigs = false
    onlyUseShocksForHomotopy = true
    enableDrawingShockSeries = true

    # Generate list of settings which change from calibration to calibration
    calibrationList = vec(Any[(σ = σ, kk = kk, π = π) for (kk, π) in enumerate(πTargets), σ in σLevels])
    calibrationList[1] = merge(calibrationList[1], (redrawShocks = true, shockSeed = 2387, shockSequenceName = "Baseline")) 

    # Add additional settings
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, indZLB = false)) # 2% inflation target without ZLB
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, approximationTypeALM = :LinearRegression)) # Linear Regression instead of Neural Network
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, approximationTypeALM = :DualRegression)) # Two Linear Regressions instead of Neural Network
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, approximationTypeALM = :QuadRegression)) # Four Linear Regressions instead of Neural Network
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, T = 25100, redrawShocks = true, shockSeed = 2387, shockSequenceName = "HigherAccuracy",
        noHomotopy = true, appendix = "HigherAccuracy"))   # More accurate solution, less subject to randomness from simulation
                                                            # Note: this uses the same random seed as the shorter shock sequence
    push!(calibrationList, (σ = 0.075, kk = 1, π = 0.02, T = 25100, θ = 200.0, appendix = "HigherAccuracy_HighRotembergCost")) 

    # Generate all configs
    shockSeriesID = 0
    for (ii, calib) in enumerate(calibrationList)

        # Extract parameter settings
        π = calib.π
        σ = calib.σ
        kk = calib.kk

        # Default calibration settings (used if calib does not contain any other setting)
        indZLB = haskey(calib, :indZLB) ? calib.indZLB : true
        approximationTypeALM = haskey(calib, :approximationTypeALM) ? calib.approximationTypeALM : :NeuralNetwork
        appendix = haskey(calib, :appendix) ? "_" * calib.appendix : ""
        noHomotopy = haskey(calib, :noHomotopy) ? calib.noHomotopy : false
        redrawShocks = haskey(calib, :redrawShocks) ? calib.redrawShocks : false
        shockSeed = haskey(calib, :shockSeed) ? calib.shockSeed : -1
        shockSequenceName = haskey(calib, :shockSequenceName) ? calib.shockSequenceName : "$(shockSeriesID)"
        remainingCalibrationOptions = delete(calib, [:π, :σ, :kk, :indZLB, :approximationTypeALM, :appendix, :noHomotopy, :redrawShocks, :shockSeed, :shockSequenceName])

        # Generate filename
        πStr =  replace(string(round(π, digits = 4)), "." => "_")
        σStr = replace(string(round(σ, digits = 3)), "." => "_")
        ZLBStr = indZLB ? "ZLB" : "NoZLB"
        algorithmStr = approximationTypeALM == :NeuralNetwork ? "" : ("_" * string(approximationTypeALM))
        filenameExt = "BaselineConfig_$(ZLBStr)_pitilde_$(πStr)_sig_$(σStr)$(algorithmStr)$(appendix)"
        filename = "Results/HANKWageRigidities/HANKWageRigidities_$(filenameExt).bson"

        # Add current config
        config = (merge(generalConfig, remainingCalibrationOptions)...,
            approximationTypeALM = approximationTypeALM,
            Ω = convertAR1Rouwenhorst(0.94, σ, 3)[1],
            sGrid = 1 .+ collect(convertAR1Rouwenhorst(0.94, σ, 3)[2]),
            π̃ = exp(π/4),
            ZLBLevel = 1.0,
            bindingZLB = indZLB,
            filenameExt = filenameExt,
            filename = filename
        )

        # Redraw shocks 
        if redrawShocks

            # Update shock series counter
            shockSeriesID += 1

            # Fix seed if desired
            if shockSeed != -1
                Random.seed!(shockSeed)
            end

            # Draw new shocks series
            if enableDrawingShockSeries
                P = settings(; config...)
                S = computeShocks(P)
            end

            # Save the shock series
            if shockSeed != -1
                filenameShocks = "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ShockSeries_$(shockSequenceName)_Seed_$(shockSeed).bson"
            else
                filenameShocks = "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ShockSeries_$(shockSequenceName).bson"
            end
            
            if isfile(filenameShocks)
                @warn "Shock series already exists: $(filename). Loading existing shocks."
            else
                if enableDrawingShockSeries
                    @warn "New shock series generated: $(filename)"
                    @save filenameShocks S
                end
            end
            

            # Add the shock series to the config list such that it will be used for homotopy
            push!(configs, (filename = filenameShocks,))
            push!(skipList, length(configs))

        end

        # Add config to list of configs
        push!(configs, config)

        # If enabled, homotopy is used for changes in the inflation target
        # Note this assumes a particular order for calibrationList
        if noHomotopy
            push!(noHomotopyList, length(configs))
        else
            if kk == 1 || onlyUseShocksForHomotopy
                push!(onlyUseShocksForHomotopyList, length(configs))
            end
        end

        # Skip existing config files
        if isfile(filename) && skipExistingConfigs
            push!(skipList, length(configs))
        end

    end

    return configs, noHomotopyList, onlyUseShocksForHomotopyList, skipList

end
