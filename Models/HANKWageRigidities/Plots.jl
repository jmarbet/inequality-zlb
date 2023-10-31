"""
    recreateComparisonPlots()

Generates the plot that is created during the solution algorithm again.

"""
function recreateComparisonPlots()

    # Load results
    filename = "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_04_sig_0_075.bson"
    @load filename P S DSS bPolicy πwALM EπwCondALM NNEπwCond NNπw bCross RStar H πw algorithmProgress

    # Interpolate the ALMs
    πwALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid), πwALM, extrapolation_bc = Line())
    EπwCondALMInterpol = linear_interpolation((P.RDenseGrid, P.ζDenseGrid, P.ζDenseGrid), EπwCondALM, extrapolation_bc = Line())

    #
    bCross = copy(DSS.bCross)
    RStar = DSS.RStar
    H = DSS.H
    πw = DSS.πw

    #
    πwSim, EπwCondSim, HSim, RStarSim, _, _ = simulateModel(P, S, DSS, bPolicy, πwALMInterpol, EπwCondALMInterpol, bCross, RStar, H, πw)

    #
    NNπwALMMat = getπwALMMatrix(P, NNπw)
    NNEπwCondALMMat = getEπwCondALMMatrix(P, NNEπwCond)

    # Simulate the predictions of the ALM conditional on the equilibrium nominal interest rate and z
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
                        πwALM, EπwCondALM, NNπwALMMat, NNEπwCondALMMat, algorithmProgress, tuple(), tuple())

    savefig(p, "Figures/$(P.filenameFolder)/SolutionProgress_$(P.filenameExt).pdf")

    return p

end


"""
    getπwALMMatrix()

Auxiliary function of recreateComparisonPlots(). Computes the matrix representation
of the ALM for inflation.

"""
function getπwALMMatrix(P, NN)

    # Evaluate the neural network on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]

        if P.approximationTypeALM == :NeuralNetwork

            # Get the state variables for the current node in the dense state space
            RStar = (P.RDenseGrid[i_R] - NN.normFactors.RStar.location) / NN.normFactors.RStar.scale
            ζ = (P.ζDenseGrid[i_ζ] - NN.normFactors.ζ.location) / NN.normFactors.ζ.scale

            ALM[idx] = feedforward(NN, [RStar, ζ])[1] * NN.normFactors.πw.scale + NN.normFactors.πw.location

        elseif  P.approximationTypeALM == :LinearRegression

            # Get the state variables for the current node in the dense state space
            RStar = P.RDenseGrid[i_R]
            ζ = P.ζDenseGrid[i_ζ]

            ALM[idx] = exp(dot(NN, [1.0, log(RStar), log(ζ), log(RStar)*log(ζ)]))

        end

    end

    return ALM

end


"""
    getEπwCondALMMatrix()

Auxiliary function of recreateComparisonPlots(). Computes the matrix representation
of the ALM for the inflation expectation term.

"""
function getEπwCondALMMatrix(P, NN)

    # Evaluate the neural network on a grid of state variables
    ALM = zeros(P.RDenseGridSize, P.ζDenseGridSize, P.ζDenseGridSize)
    @threads for idx in CartesianIndices(ALM)

        # Determne indices of 1D grids
        i_R = idx[1]
        i_ζ = idx[2]
        i_ζp = idx[3]

        if P.approximationTypeALM == :NeuralNetwork

            # Get the state variables for the current node in the dense state space
            RStar = (P.RDenseGrid[i_R] - NN.normFactors.RStar.location) / NN.normFactors.RStar.scale
            ζ = (P.ζDenseGrid[i_ζ] - NN.normFactors.ζ.location) / NN.normFactors.ζ.scale
            ζPrime = (P.ζDenseGrid[i_ζp] - NN.normFactors.ζPrime.location) / NN.normFactors.ζPrime.scale

            ALM[idx] = feedforward(NN, [RStar, ζ, ζPrime])[1] * NN.normFactors.EπwCond.scale + NN.normFactors.EπwCond.location

        elseif  P.approximationTypeALM == :LinearRegression

            # Get the state variables for the current node in the dense state space
            RStar = P.RDenseGrid[i_R]
            ζ = P.ζDenseGrid[i_ζ]
            ζPrime = P.ζDenseGrid[i_ζp]

            ALM[idx] = dot(NN, [1.0, log(RStar), log(ζ), log(ζPrime), log(RStar)*log(ζ), log(RStar)*log(ζPrime), log(ζ)*log(ζPrime)])

        end

    end

    return ALM

end


function computeIncomeComponentsGrid(P, periods, bSet, IRFs, SSS)

    # Initialize income components
    incomeComponents = Dict()
    incomeComponents[:totalIncome] = zeros(length(bSet), P.sGridSize, length(periods))

    for incomeType in [:interestIncome, :laborIncome, :transferIncome]
        incomeComponents[incomeType] = similar(incomeComponents[:totalIncome])
    end

    # Compute income components at each point in time
    for ii in 1:length(periods)

        # Note tt and ii do not have to be the same (e.g. if periods = 2:3)
        tt = periods[ii]

        # Get values of all variables at time tt (except for R which is from tt-1)
        simVars = getSimulatedVariablesAtTime(IRFs, tt; t0Values = SSS)

        # Compute the income components and save results in incomeComponents
        for i_b in 1:length(bSet), i_s in 1:P.sGridSize

            currentComponents = computeIncomeComponents(P, bSet[i_b], i_s, simVars)

            for incomeType in keys(currentComponents)
                incomeComponents[incomeType][i_b, i_s, ii] = currentComponents[incomeType]
            end

        end

    end

    return incomeComponents

end


function computeIncomeComponentsGridSSS(P, bSet, SSS)

    # Initalize income components in SSS
    incomeComponentsSSS = Dict()
    incomeComponentsSSS[:totalIncome] = zeros(length(bSet), P.sGridSize)

    for incomeType in [:interestIncome, :laborIncome, :transferIncome]
        incomeComponentsSSS[incomeType] = similar(incomeComponentsSSS[:totalIncome])
    end

    # Compute the income components and save results in incomeComponentsSSS
    for i_b in 1:length(bSet), i_s in 1:P.sGridSize

        currentComponents = computeIncomeComponents(P, bSet[i_b], i_s, SSS)

        for incomeType in keys(currentComponents)
            incomeComponentsSSS[incomeType][i_b, i_s] = currentComponents[incomeType]
        end

    end

    return incomeComponentsSSS

end


function aggregateIncomeComponentsGrid(P, decompType, periods, percList, prodType, bSet, IRFs, SSS, incomeComponents, incomeComponentsSSS)

    # Aggregate if necessary or drop dimensons
    if decompType == :borrowersSavers

        for incomeType in keys(incomeComponents)

            # Get the distribution used for aggregation
            bCrossIRFs = cat([getSimulatedVariablesAtTime(IRFs, periods[ii]; t0Values = SSS).bCross for ii in 1:length(periods)]..., dims = 3)

            # Determine savers and borrowers
            saversSelector = bSet .>= 0.0
            borrowersSelector = bSet .< 0.0

            # Compute aggregated income component
            saversIncComp = incomeComponents[incomeType][saversSelector, :, :] .*
                (bCrossIRFs[saversSelector, :, :] ./ sum(bCrossIRFs[saversSelector, :, :], dims = (1, 2)))
            borrowersIncComp = incomeComponents[incomeType][borrowersSelector, :, :] .*
                (bCrossIRFs[borrowersSelector, :, :] ./ sum(bCrossIRFs[borrowersSelector, :, :], dims = (1, 2)))
            allIncComp = incomeComponents[incomeType] .* bCrossIRFs

            # Replace income components with aggregated values
            incomeComponents[incomeType] = hcat(dropdims(sum(borrowersIncComp, dims = (1, 2)), dims = (1, 2)),
                                            dropdims(sum(saversIncComp, dims = (1, 2)), dims = (1, 2)),
                                            dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))

            # Compute aggregated income component in SSS
            saversIncComp = incomeComponentsSSS[incomeType][saversSelector, :] .*
                (SSS[:bCross][saversSelector, :] ./ sum(SSS[:bCross][saversSelector, :], dims = (1, 2)))
            borrowersIncComp = incomeComponentsSSS[incomeType][borrowersSelector, :] .*
                (SSS[:bCross][borrowersSelector, :] ./ sum(SSS[:bCross][borrowersSelector, :, :], dims = (1, 2)))
            allIncComp = incomeComponentsSSS[incomeType] .* SSS[:bCross]

            # Replace income components with aggregated values
            incomeComponentsSSS[incomeType] = hcat(dropdims(sum(borrowersIncComp, dims = (1, 2)), dims = (1, 2)),
                                            dropdims(sum(saversIncComp, dims = (1, 2)), dims = (1, 2)),
                                            dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))

        end

    elseif decompType in [:percentile, :percentileNoAgg]

        for incomeType in keys(incomeComponents)

            # Get the distribution used for aggregation
            bCrossIRFs = cat([getSimulatedVariablesAtTime(IRFs, periods[ii]; t0Values = SSS).bCross for ii in 1:length(periods)]..., dims = 3)

            # Aggregate the components for the percentiles
            if prodType == :average

                # Temporarary vectors for response of different percentiles
                incomeComponentsTmp = zeros(length(periods), length(percList))
                incomeComponentsSSSTmp = zeros(1, length(percList))

                for ii in 1:length(percList)

                    # Get grid points below and above the current percentile bSet[ii]
                    bIdxLower = findlast(x -> x <= bSet[ii], P.bDenseGrid)
                    bIdxUpper = findfirst(x -> x > bSet[ii], P.bDenseGrid) # If bSet[ii] == P.bMax, this will return nothing

                    # Compute the income components by using a weighted average over the productivity types
                    # 1. uses marginal distribution for grid point below and above bSet[ii] to compute two averages
                    # 2. uses distance to grid point above and below bSet[ii] to interpolate between the two averages
                    for tt in 1:length(periods)
                        w1 = (bSet[ii] - P.bDenseGrid[bIdxLower]) / (P.bDenseGrid[bIdxUpper] - P.bDenseGrid[bIdxLower])
                        weightsLower = bCrossIRFs[bIdxLower, :, tt] / sum(bCrossIRFs[bIdxLower, :, tt])
                        weightsUpper = bCrossIRFs[bIdxUpper, :, tt] / sum(bCrossIRFs[bIdxUpper, :, tt])
                        comp = incomeComponents[incomeType][ii, :, tt]
                        incomeComponentsTmp[tt, ii] = w1 * dot(comp, weightsLower) + (1 - w1) * dot(comp, weightsUpper)
                    end

                    # Repeat previous step for the SSS
                    w1 = (bSet[ii] - P.bDenseGrid[bIdxLower]) / (P.bDenseGrid[bIdxUpper] - P.bDenseGrid[bIdxLower])
                    weightsLower = SSS[:bCross][bIdxLower, :] / sum(SSS[:bCross][bIdxLower, :])
                    weightsUpper = SSS[:bCross][bIdxUpper, :] / sum(SSS[:bCross][bIdxUpper, :])
                    comp = incomeComponentsSSS[incomeType][ii, :]
                    incomeComponentsSSSTmp[1, ii] = w1 * dot(comp, weightsLower) + (1 - w1) * dot(comp, weightsUpper)

                end

            elseif startswith("$(prodType)", r"only")

                # Get the productivity type
                prodTypeNo = parse(Int64, "$(prodType)"[5:end])

                # Temporarary vectors for response of different percentiles
                incomeComponentsTmp = incomeComponents[incomeType][1:length(percList), prodTypeNo, :]'
                incomeComponentsSSSTmp = incomeComponentsSSS[incomeType][1:length(percList), prodTypeNo]'

            elseif prodType == :all

                # This option only works for a single period
                @assert length(periods) == 1

                # Temporarary vectors for response of different percentiles
                incomeComponentsTmp = incomeComponents[incomeType][1:length(percList), :, 1]'
                incomeComponentsSSSTmp = incomeComponentsSSS[incomeType][1:length(percList), :]'

            end

            # Additionally, conpute the decomposition for the aggregate
            if decompType == :percentile

                # Compute aggregated income component
                allIncComp = incomeComponents[incomeType][length(percList)+1:end, :, :] .* bCrossIRFs

                # Replace income components with aggregated values
                if prodType != :all
                    incomeComponents[incomeType] = hcat(incomeComponentsTmp,
                                                dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))
                else
                    incomeComponents[incomeType] = hcat(incomeComponentsTmp,
                                                repeat(dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)), P.sGridSize))
                end

                # Compute aggregated income component in SSS
                allIncComp = incomeComponentsSSS[incomeType][length(percList)+1:end, :] .* SSS[:bCross]

                # Replace income components with aggregated values
                if prodType != :all
                    incomeComponentsSSS[incomeType] = hcat(incomeComponentsSSSTmp,
                                                    dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))
                else
                    incomeComponentsSSS[incomeType] = hcat(incomeComponentsSSSTmp,
                                                    repeat(dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)), P.sGridSize))
                end

            elseif decompType == :percentileNoAgg

                incomeComponents[incomeType] = incomeComponentsTmp
                incomeComponentsSSS[incomeType] = incomeComponentsSSSTmp

            end

        end

    end

    return incomeComponents, incomeComponentsSSS

end


function getConditionalProductivityDistributionAtWealthLevel(P, b, bCross)

    # Get grid points below and above the wealth level b
    bIdxLower = findlast(x -> x <= b, P.bDenseGrid)
    bIdxUpper = findfirst(x -> x > b, P.bDenseGrid) # If b == P.bMax, this will return nothing

    # Determine weights of the productivity types
    # 1. uses marginal distribution for grid point below and above b to compute two averages
    # 2. uses distance to grid point above and below b to interpolate between the two averages
    w1 = (b - P.bDenseGrid[bIdxLower]) / (P.bDenseGrid[bIdxUpper] - P.bDenseGrid[bIdxLower])
    weightsLower = bCross[bIdxLower, :] / sum(bCross[bIdxLower, :])
    weightsUpper = bCross[bIdxUpper, :] / sum(bCross[bIdxUpper, :])
    weights = w1 * weightsLower + (1 - w1) * weightsUpper

    return weights

end


function computeIncomeComponents(P, b, i_s, R, π, w, H, T)

    # Get individual productivity
    s = P.sGrid[i_s]
    
    # Determine income components
    interestIncome = b * (R/π-1)
    laborIncome = w * s * H
    transferIncome = T - P.τ * w * s * H
    totalIncome = interestIncome + laborIncome + transferIncome

    return (totalIncome = totalIncome,
        interestIncome  = interestIncome,
        laborIncome = laborIncome,
        transferIncome = transferIncome)

end


function computeIncomeComponents(P, b, i_s, vars)

    @unpack R, π, w, H, T = vars

    return computeIncomeComponents(P, b, i_s, R, π, w, H, T)

end


function getSimulatedVariablesAtTime(simSeries, tt; t0Values = NamedTuple(), laggedList = [:R, :RStar])

    varNames = Tuple(keys(simSeries))
    varValues = Array{Any,1}(undef, length(varNames))

    # Get particular time period for each variable
    for (ii, var) in enumerate(varNames)

        if var in laggedList # get value at time tt-1

            if tt > 1

                if ndims(simSeries[var]) == 3 # e.g. bCross
                    varValues[ii] = simSeries[var][:, :, tt-1]
                else
                    varValues[ii] = simSeries[var][tt-1]
                end

            else

                varValues[ii] = t0Values[var]

            end

        else # get value at time tt

            if ndims(simSeries[var]) == 3 # e.g. bCross
                varValues[ii] = simSeries[var][:, :, tt]
            else
                varValues[ii] = simSeries[var][tt]
            end

        end

    end

    return NamedTuple{varNames}(varValues)

end


function computeConsumptionGrid(P, periods, bSet, IRFs, SSS, bPolicyInterpol)

    # Initialize consumption "grid"
    consumptionGrid = zeros(length(bSet), P.sGridSize, length(periods))

    # Compute consumption at each point in time
    for ii in 1:length(periods)

        # Note tt and ii do not have to be the same (e.g. if periods = 2:3)
        tt = periods[ii]

        # Get values of all variables at time tt (except for R which is from tt-1)
        simVars = getSimulatedVariablesAtTime(IRFs, tt; t0Values = SSS)

        # Compute consumption and save results in consumptionGrid
        for i_b in 1:length(bSet), i_s in 1:P.sGridSize

            # Get individual variables for easier access
            b = bSet[i_b]
            s = P.sGrid[i_s]

            # Bond choice
            bp = bPolicyInterpol(b, simVars.RStar, i_s, simVars.ζ)

            # Determine consumption
            consumptionGrid[i_b, i_s, ii] = computeIndividualCashOnHand(P, b, s, simVars.R, simVars.π, simVars.w, simVars.H, simVars.T) - bp

        end

    end

    return consumptionGrid

end


function computeConsumptionGridSSS(P, bSet, SSS, bPolicyInterpol)

    # Initialize consumption "grid"
    consumptionGrid = zeros(length(bSet), P.sGridSize)

    # Compute the consumption and save results in consumptionGrid
    for i_b in 1:length(bSet), i_s in 1:P.sGridSize

        # Get individual variables for easier access
        b = bSet[i_b]
        s = P.sGrid[i_s]

        # Bond choice
        bp = bPolicyInterpol(b, SSS.RStar, i_s, SSS.ζ) 

        # Determine consumption
        consumptionGrid[i_b, i_s]  = computeIndividualCashOnHand(P, b, s, SSS.R, SSS.π, SSS.w, SSS.H, SSS.T) - bp

    end

    return consumptionGrid

end


function aggregateConsumptionComponentsGrid(P, decompType, periods, percList, prodType, bSet, IRFs, SSS, consumptionGrid, consumptionGridSSS)

    # Aggregate if necessary or drop dimensons
    if decompType == :borrowersSavers

        # Get the distribution used for aggregation
        bCrossIRFs = cat([getSimulatedVariablesAtTime(IRFs, periods[ii]; t0Values = SSS).bCross for ii in 1:length(periods)]..., dims = 3)

        # Determine savers and borrowers
        saversSelector = bSet .>= 0.0
        borrowersSelector = bSet .< 0.0

        # Compute aggregated consumption
        saversIncComp = consumptionGrid[saversSelector, :, :] .*
            (bCrossIRFs[saversSelector, :, :] ./ sum(bCrossIRFs[saversSelector, :, :], dims = (1, 2)))
        borrowersIncComp = consumptionGrid[borrowersSelector, :, :] .*
            (bCrossIRFs[borrowersSelector, :, :] ./ sum(bCrossIRFs[borrowersSelector, :, :], dims = (1, 2)))
        allIncComp = consumptionGrid .* bCrossIRFs

        # Replace consumption grid with aggregated values
        consumptionGrid = hcat(dropdims(sum(borrowersIncComp, dims = (1, 2)), dims = (1, 2)),
                            dropdims(sum(saversIncComp, dims = (1, 2)), dims = (1, 2)),
                            dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))

        # Compute aggregated consumption in SSS
        saversIncComp = consumptionGridSSS[saversSelector, :] .*
            (SSS[:bCross][saversSelector, :] ./ sum(SSS[:bCross][saversSelector, :], dims = (1, 2)))
        borrowersIncComp = consumptionGridSSS[borrowersSelector, :] .*
            (SSS[:bCross][borrowersSelector, :] ./ sum(SSS[:bCross][borrowersSelector, :, :], dims = (1, 2)))
        allIncComp = consumptionGridSSS .* SSS[:bCross]

        # Replace income components with aggregated values
        consumptionGridSSS = hcat(dropdims(sum(borrowersIncComp, dims = (1, 2)), dims = (1, 2)),
                            dropdims(sum(saversIncComp, dims = (1, 2)), dims = (1, 2)),
                            dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))


    elseif decompType in [:percentile, :percentileNoAgg]

        # Get the distribution used for aggregation
        bCrossIRFs = cat([getSimulatedVariablesAtTime(IRFs, periods[ii]; t0Values = SSS).bCross for ii in 1:length(periods)]..., dims = 3)

        # Aggregate consumption for the percentiles
        if prodType == :average

            # Temporarary vectors for response of different percentiles
            consumptionGridTmp = zeros(length(periods), length(percList))
            consumptionGridSSSTmp = zeros(1, length(percList))

            for ii in 1:length(percList)

                # Get grid points below and above the current percentile bSet[ii]
                bIdxLower = findlast(x -> x <= bSet[ii], P.bDenseGrid)
                bIdxUpper = findfirst(x -> x > bSet[ii], P.bDenseGrid) # If bSet[ii] == P.bMax, this will return nothing

                # Compute consumption by using a weighted average over the productivity types
                # 1. uses marginal distribution for grid point below and above bSet[ii] to compute two averages
                # 2. uses distance to grid point above and below bSet[ii] to interpolate between the two averages
                for tt in 1:length(periods)
                    w1 = (bSet[ii] - P.bDenseGrid[bIdxLower]) / (P.bDenseGrid[bIdxUpper] - P.bDenseGrid[bIdxLower])
                    weightsLower = bCrossIRFs[bIdxLower, :, tt] / sum(bCrossIRFs[bIdxLower, :, tt])
                    weightsUpper = bCrossIRFs[bIdxUpper, :, tt] / sum(bCrossIRFs[bIdxUpper, :, tt])
                    comp = consumptionGrid[ii, :, tt]
                    consumptionGridTmp[tt, ii] = w1 * dot(comp, weightsLower) + (1 - w1) * dot(comp, weightsUpper)
                end

                # Repeat previous step for the SSS
                w1 = (bSet[ii] - P.bDenseGrid[bIdxLower]) / (P.bDenseGrid[bIdxUpper] - P.bDenseGrid[bIdxLower])
                weightsLower = SSS[:bCross][bIdxLower, :] / sum(SSS[:bCross][bIdxLower, :])
                weightsUpper = SSS[:bCross][bIdxUpper, :] / sum(SSS[:bCross][bIdxUpper, :])
                comp = consumptionGridSSS[ii, :]
                consumptionGridSSSTmp[1, ii] = w1 * dot(comp, weightsLower) + (1 - w1) * dot(comp, weightsUpper)

            end

        elseif startswith("$(prodType)", r"only")

            # Get the productivity type
            prodTypeNo = parse(Int64, "$(prodType)"[5:end])

            # Temporarary vectors for response of different percentiles
            consumptionGridTmp = consumptionGrid[1:length(percList), prodTypeNo, :]'
            consumptionGridSSSTmp = consumptionGridSSS[1:length(percList), prodTypeNo]'

        elseif prodType == :all

            # This option only works for a single period
            @assert length(periods) == 1

            # Temporarary vectors for response of different percentiles
            consumptionGridTmp = consumptionGrid[1:length(percList), :, 1]'
            consumptionGridSSSTmp = consumptionGridSSS[1:length(percList), :]'

        end

        # Additionally, compute the decomposition for the aggregate
        if decompType == :percentile

            # Compute aggregated consumption
            allIncComp = consumptionGrid[length(percList)+1:end, :, :] .* bCrossIRFs

            # Combine percentiles with aggregated values
            if prodType != :all
                consumptionGrid = hcat(consumptionGridTmp,
                                    dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))
            else
                consumptionGrid = hcat(consumptionGridTmp,
                                    repeat(dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)), P.sGridSize))
            end

            # Compute aggregated consumption in SSS
            allIncComp = consumptionGridSSS[length(percList)+1:end, :] .* SSS[:bCross]

            # Combine percentiles with aggregated values
            if prodType != :all
                consumptionGridSSS = hcat(consumptionGridSSSTmp,
                                    dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)))
            else
                consumptionGridSSS = hcat(consumptionGridSSSTmp,
                                    repeat(dropdims(sum(allIncComp, dims = (1, 2)), dims = (1, 2)), P.sGridSize))
            end

        elseif decompType == :percentileNoAgg

            consumptionGrid = consumptionGridTmp
            consumptionGridSSS = consumptionGridSSSTmp

        end

    end

    return consumptionGrid, consumptionGridSSS

end


"""
    computeDSSOverview(P, DSS)

Displays summary statistics and generates an overview plot.

"""
function computeDSSOverview(P, DSS)

    # Bond distribution plot
    p1 = bar(P.bDenseGrid, sum(DSS.bCross, dims = 2),
        normalize = true,
        title = "Bond Distribution",
        xlabel = "Bonds (b)",
        linecolor = :steelblue,
        legend = :none)

    # Savings policy plots
    p2 = plot(P.bGrid, DSS.bPolicy .- P.bGrid,
        title = "Savings Policy",
        label = [latexstring("s_", ii) for jj in 1:1, ii in 1:3],
        ylabel = "b' - b", xlabel = "Bonds",
        legend = :best)

    percentiles = floor(sum(DSS.bCross, dims = 2)[1], digits = 2)+0.01:0.01:0.99
    wealthPerc = [computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, x) for x in percentiles]
    p3 = bar(percentiles * 100, wealthPerc,
        normalize = true,
        title = "Wealth Percentiles",
        xlabel = "Percentile",
        ylabel = "Bonds (b)",
        linecolor = :steelblue,
        legend = :none)

    lorenzCurve, giniCoeff, percentiles = computeLorenzCurve(P, DSS.B, DSS.bCross)
    p4 = plot(percentiles * 100, lorenzCurve * 100,
            normalize = true,
          title = "Lorenz Curve (Gini: $(round(giniCoeff, digits = 2)))",
          xlabel = "% of Population",
          ylabel = "% of Wealth",
          linecolor = :steelblue,
          label = "")
    plot!(0:10:100, 0:10:100, linestyle = :dash, color = :black, label = "")
    scatter!([90], [100*lorenzCurve[percentiles .== 0.9]], marker = :star, label = "Bottom 90%")

    # Compute MPCs in DSS
    MPCs = computeDSSMPCs(P, DSS)
    MPCsInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), MPCs, extrapolation_bc = Line())

    # Compute MPCs for percentiles that will be shown
    percPlotted = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99]
    MPCsPlotted = zeros(length(percPlotted), P.sGridSize)
    bCross = vec(sum(DSS.bCross, dims = 2))
    bCrossPerc = [computePercentile(bCross, P.bDenseGrid, perc) for perc in percPlotted]
    for i_s in 1:P.sGridSize
        MPCsPlotted[:, i_s] = MPCsInterpol.(bCrossPerc, i_s)
    end

    # Plot the MPCs
    #ctg = repeat(["Low Productivity", "Medium Productivity", "High Productivity"], inner = length(percPlotted))#repeat("s_" .* string.(1:P.sGridSize), inner = length(percPlotted))
    #=nam = repeat(string.(vec(convert.(Int64, 100*percPlotted))), outer = P.sGridSize)
    p5 = groupedbar(nam, MPCsPlotted,
        title = "MPCs",
        xlabel = "Wealth Percentile",
        ylabel = "MPC",
        bar_width = 0.67,
        lw = 0.2,
        color = [palette(:YlGnBu_3)[3] palette(:YlGnBu_3)[2]  palette(:YlGnBu_3)[1]],
        label = ["Low Labor Productivity" "Medium Labor Productivity" "High Labor Productivity"],
        margin = 5Plots.PlotMeasures.mm)
    ylims!(0, 0.3)=#

    # Get consumption policy function
    cPolicy = computeConsumptionPolicyDSS(P, DSS)
    cPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), cPolicy, extrapolation_bc = Line())

    p5 = plot(title = "MPCs", xlabel = "Bonds", ylabel = "MPC",)
    for i_s in 1:P.sGridSize
        plot!(P.bGrid, b -> MPCsInterpol(b, i_s), label = latexstring("s_", i_s))
    end

    # Add line for economy wide average MPC
    avgMPC = 0.0
    for  i_s in 1:P.sGridSize
        avgMPC += dot(DSS.bCross[:, i_s], MPCsInterpol.(P.bDenseGrid, i_s))
    end
    hline!([avgMPC], color = :black, linestyle = :dash, label = "Average MPC")

    wealthPoorestIndicator = cumsum(vec(sum(DSS.bCross, dims = 2))) .<= 0.3
    p6 = plot((1:3)', legend = :none, framestyle = :none)
    txt = string(
        "r = ", round(log(DSS.r)*400, digits = 3), "%, ",
        "Y = ", round(DSS.Y, digits = 3), ", ",
        "B/Y = ", round(DSS.B/DSS.Y, digits = 3), ", ",
        "w = ", round(DSS.w, digits = 3), "\n",
        "Borrowing constraint: ", round(P.b̲, digits = 3), " (≈ ", round(-P.b̲/DSS.w, digits = 3), " w)\n",
        "Agents with negative assets: ", round(100*sum(DSS.bCross[P.bDenseGrid .< 0, :]), digits = 3), "%\n",
        "Agents at borrowing constraint: ", round(100*sum(DSS.bCross[1, :]), digits = 3), "%\n",
        "Average wealth of poorest ~30%: ", round(sum(DSS.bCross[wealthPoorestIndicator,:] .* P.bDenseGrid[wealthPoorestIndicator]) / sum(DSS.bCross[wealthPoorestIndicator,:]), digits = 3), "\n",
        "Gini Coefficient: ", round(giniCoeff, digits = 3), "\n",
        "Average MPC: ", round(avgMPC, digits = 3), "\n\n",
        "Wealth Share:\n",
        "Top 10%: ", round(100*getWealthShare(:top, 0.1, lorenzCurve, percentiles), digits = 3), "%\n",
        "Top 1%: ", round(100*getWealthShare(:top, 0.01, lorenzCurve, percentiles), digits = 3), "%\n",
        "Bottom 50%: ", round(100*getWealthShare(:bottom, 0.5, lorenzCurve, percentiles), digits = 3), "%\n",
        "Bottom 25%: ", round(100*getWealthShare(:bottom, 0.25, lorenzCurve, percentiles), digits = 3), "%\n\n",
        "Wealth Distribution:\n",
        "99th Percentile: ", round(computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.99), digits = 3), "\n",
        "90th Percentile: ", round(computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.9), digits = 3), "\n",
        "50th Percentile: ", round(computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.5), digits = 3), "\n",
        "20th Percentile: ", round(computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.2), digits = 3), "\n",
        "10th Percentile: ", round(computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.1), digits = 3), "\n"
    )
    annotate!([(1, 2, Plots.text(txt, 9, :black, :left))])

    # Combine plots
    #l = @layout [a; b; c]
    p = plot(p1, p2, p3, p4, p5, p6, layout = 6, legendfontsize = 8, guidefontsize = 8, tickfontsize = 8, size = (1200, 700), margin = 5Plots.PlotMeasures.mm)

    # Display some statistics
    println("DSS Statistics")
    println("Real Rate: ", log(DSS.r)*400)
    println("Nominal Rate: ", log(DSS.R)*400)
    println("Output: ", DSS.Y)
    println("Bonds: ", DSS.B)
    println("Bond-Output Ratio: ", DSS.B/DSS.Y)
    println("Wage: ", DSS.w)
    println("Type shares: ", P.P0)
    println("Agents with negative assets (%): ", 100*sum(DSS.bCross[P.bDenseGrid .< 0, :]))
    for i_s in 1:P.sGridSize
        println("-> Type $(i_s) (s = $(round(P.sGrid[i_s], digits=3))): ", 100*sum(DSS.bCross[P.bDenseGrid .< 0, i_s]))
    end
    println("Agents at borrowing constraint (%): ", 100*sum(DSS.bCross[1, :]))
    for i_s in 1:P.sGridSize
        println("-> Type $(i_s) (s = $(round(P.sGrid[i_s], digits=3))): ", 100*DSS.bCross[1, i_s])
    end
    println("Average wealth of wealth poorest ~30% (exact $(100*sum(DSS.bCross[wealthPoorestIndicator,:]))%): ", sum(DSS.bCross[wealthPoorestIndicator,:] .* P.bDenseGrid[wealthPoorestIndicator]) / sum(DSS.bCross[wealthPoorestIndicator,:]))
    println("Gini Coefficient: ", giniCoeff)
    println("Wealth Share: ")
    println(" Top 10%: ", 100*getWealthShare(:top, 0.1, lorenzCurve, percentiles))
    println(" Top 1%: ", 100*getWealthShare(:top, 0.01, lorenzCurve, percentiles))
    println(" Bottom 50%: ", 100*getWealthShare(:bottom, 0.5, lorenzCurve, percentiles))
    println(" Bottom 25%: ", 100*getWealthShare(:bottom, 0.25, lorenzCurve, percentiles))
    println("Wealth Distribution: ")
    println(" 99th Percentile: ", computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.99))
    println(" 90th Percentile: ", computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.9))
    println(" 50th Percentile: ", computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.5))
    println(" 20th Percentile: ", computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.2))
    println(" 10th Percentile: ", computePercentile(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid, 0.1))

    #
    cCross, cGrid = computeConsumptionDistributionDSS(P, DSS; cGridSize = 10000)
    println("\nChecks:")
    println("Aggregate Consumption: ", DSS.C)
    println("Mean Consumption: ", sum(cCross' * cGrid))
    println("Agg. Consumption Error: ", DSS.C-sum(cCross' * cGrid))
    println("Output: ", DSS.Y)

    # Compute additional statistics for the model distributions
    println("\nWealth Distribution Statistics:")
    showDistributionStatistics(vec(sum(DSS.bCross, dims = 2)), P.bDenseGrid)

    println("\nIncome Distribution Statistics:")
    incomeHist, incomeGrid = computeIncomeDistributionDSS(P, DSS)
    showDistributionStatistics(vec(sum(incomeHist, dims = 2)), incomeGrid)

    return p

end
