using LaTeXStrings
using Plots
using StatsPlots
using StatsBase
using OrderedCollections
using Formatting
using Random
using LinearAlgebra

default(legend_font_halign=:left) # Align legend font left

pgfplotsx()

function generatePublicationPlots()

    # Set plotting backend
    pgfplotsx()

    # Settings
    inputFolder = "Results/HANKWageRigidities/Results"
    outputFolder = "Figures/HANKWageRigidities/PublicationPlotsInequalityAndZLB"

    # Inflation targets (in ascending order)
    πTargets = [1.7:0.1:1.8; collect(2.0:0.5:4.0)] / 100

    # Idiosyncratic volatility
    σLevels = [0.075, 0.0, 0.095]
    σLabels = ["ZLB-HANK" "ZLB-RANK" "ZLB-HANK with High Idiosyncratic Risk"]

    # Initalize filename and model lists
    filenames = []
    modelsCompStat = Array{String,2}(undef, length(πTargets), length(σLevels)) # Contains model definitions for comparative statics

    for (ii, σ) in enumerate(σLevels)

        for (jj, π) in enumerate(πTargets)

            # Generate filename
            πStr =  replace(string(round(π, digits = 4)), "." => "_")
            τStr = "0_0"
            ELBStr = "1_0"
            σStr = replace(string(round(σ, digits = 4)), "." => "_")
            ZLBStr = "ZLB"

            if σ > 0.0

                filenameExt = "BaselineConfig_$(ZLBStr)_pitilde_$(πStr)_sig_$(σStr)"
                filename = "Results/HANKWageRigidities/HANKWageRigidities_$(filenameExt).bson"

            elseif σ == 0.0

                filenameExt = "RANKWageRigidities_BaselineConfig_$(ZLBStr)_pitilde_$(πStr)"
                filename = "Results/RANKWageRigidities/$(filenameExt).bson"

            elseif σ == -1.0

                filenameExt = "RANKWageRigidities_BaselineConfig_$(ZLBStr)_pitilde_$(πStr)_Recalibrated"
                filename = "Results/RANKWageRigidities/$(filenameExt).bson"

            end

            # Add file to be loaded
            push!(filenames, filename)
            modelsCompStat[jj, ii] = filenameExt

        end

    end

    # Add RANK without ZLB
    modelsCompStatPlus = Array{String,2}(undef, length(πTargets), 1)
    for (jj, π) in enumerate(πTargets)

        # Generate filename
        πStr =  replace(string(round(π, digits = 4)), "." => "_")
        ZLBStr = "NoZLB"

        filenameExt = "RANKWageRigidities_BaselineConfig_$(ZLBStr)_pitilde_$(πStr)"
        filename = "Results/RANKWageRigidities/$(filenameExt).bson"
        
        # Add file to be loaded
        push!(filenames, filename)
        modelsCompStatPlus[jj, 1] = filenameExt

    end

    # Additional required files 
    push!(filenames,
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_NoZLB_pitilde_0_02_sig_0_075.bson",
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075_LinearRegression.bson",
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075_DualRegression.bson",
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075_QuadRegression.bson",
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075_HigherAccuracy.bson",
        "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075_HigherAccuracy_HighRotembergCost.bson",
        "Results/RANKWageRigidities/RANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_HighRotembergCost.bson",
        "Results/RANKWageRigidities/RANKWageRigidities_BaselineConfig_NoZLB_pitilde_0_02.bson",
    )

    # Main model definitions
    modelZLB = "BaselineConfig_ZLB_pitilde_0_02_sig_0_075_HigherAccuracy"
    modelZLBLinear = "BaselineConfig_ZLB_pitilde_0_02_sig_0_075_LinearRegression"
    modelZLBDualLinear = "BaselineConfig_ZLB_pitilde_0_02_sig_0_075_DualRegression"
    modelZLBQuadLinear = "BaselineConfig_ZLB_pitilde_0_02_sig_0_075_QuadRegression"
    modelNoZLB = "BaselineConfig_NoZLB_pitilde_0_02_sig_0_075"
    modelZLBRANK = "RANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02"
    modelNoZLBRANK = "RANKWageRigidities_BaselineConfig_NoZLB_pitilde_0_02"

    # Additional model definitions
    modelZLBHighRotembergCost = "BaselineConfig_ZLB_pitilde_0_02_sig_0_075_HigherAccuracy_HighRotembergCost" 
    modelZLBRANKHighRotembergCost = "RANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_HighRotembergCost"

    # Create the output folder
    if !isdir(outputFolder)
        mkpath(outputFolder)
    end

    # Create the output folder for additional figures
    outputFolderAdditional = outputFolder * "/Additional"
    if !isdir(outputFolderAdditional)
        mkpath(outputFolderAdditional)
    end

    # Load the results
    res = HANKWageRigidities.loadAllResults(filenames, inputFolder, loadSimulationResults = true, loadSimulationPlusResults = true)

    # PLM Inflation (Figure 1)
    plotPLMInflation(res, modelZLB, outputFolder)

    # Forecast errors inflation (Figure 2)
    plotForecastErrorsInflation(res, modelZLB, modelZLBLinear, outputFolder)

    # Ergodic distribution (Figure 3)
    plotErgodicDistribution(res, modelZLB, modelNoZLB, outputFolder)

    # Impulse responses (Figure 4)
    plotImpulseResponseFunctions(res, modelZLB, modelNoZLB, outputFolder)

    # Impact response decomposition (Figures 5-6)
    plotResponseDecomposition(res, modelZLB, modelNoZLB, outputFolder; prodTypes = [:all], skipAdditionalFigure = false, outputFolderAdditional)

    # Comparative statics (includes Figure 7-8)
    plotComparativeStatics(res, modelsCompStat, σLabels, outputFolder)

    # DSS and SSS comparison table (Table 2)
    generateDSSSSSErgodicMeanTable(res, modelZLB, modelNoZLB, modelZLBRANK, modelNoZLBRANK, outputFolder) 

    # Rate decomposition table (Table 3)
    generateSmallRateDecompositionTable(res, modelZLB, modelZLBRANK, outputFolder)

    # Additional figures
    plotForecastErrorsInflationExpectations(res, modelZLB, modelZLBLinear, outputFolderAdditional)
    plotForecastErrorsInflation(res, modelZLB, modelZLBDualLinear, outputFolderAdditional; labelLinear = "Piecewise Linear", filename = "ForecastErrorsInflationPiecewise")
    plotForecastErrorsInflationExpectations(res, modelZLB, modelZLBDualLinear, outputFolderAdditional; labelLinear = "Piecewise Linear", filename = "ForecastErrorsInflationExpectationPiecewise")
    plotForecastErrorsInflation(res, modelZLB, modelZLBQuadLinear, outputFolderAdditional; labelLinear = "Piecewise Linear (4 Regions)", filename = "ForecastErrorsInflationPiecewise4Regions")
    plotForecastErrorsInflationExpectations(res, modelZLB, modelZLBQuadLinear, outputFolderAdditional; labelLinear = "Piecewise Linear (4 Regions)", filename = "ForecastErrorsInflationExpectationPiecewise4Regions")
    plotErgodicDistribution(res, modelZLB, modelZLBRANK, outputFolderAdditional; labelNoZLB = "ZLB-RANK", filename = "ErgodicDistributionHANKvsRANK")
    plotImpulseResponseFunctions(res, modelZLB, modelZLBRANK, outputFolderAdditional; labelNoZLB = "ZLB-RANK", filename = "IRFComparisonHANKvsRANK")
    plotPhaseDiagram(res, modelZLB, outputFolderAdditional)
    plotPolicyFunctionComparison(res, modelZLB, modelNoZLB, outputFolderAdditional)
    plotComparativeStaticsAdditional(res, modelsCompStat, σLabels, outputFolderAdditional)
    plotResponseDecomposition(res, modelZLB, modelNoZLB, outputFolderAdditional; prodTypes = [:all], plotType = :modelAOnly, skipAdditionalFigure = true)
    plotResponseDecomposition(res, modelZLB, modelNoZLB, outputFolderAdditional; prodTypes = [:all], plotType = :modelBOnly, skipAdditionalFigure = true)

    # Additional tables
    generateDSSSSSErgodicMeanTable(res, modelZLB, modelNoZLB, outputFolderAdditional)
    generateDSSSSSErgodicMeanTable(res, modelZLB, modelZLBRANK, outputFolderAdditional; labelNoZLB = "ZLB-RANK", filename = "SteadyStatesAndErgodicMeansHANKvsRANK")
    generateRateDecompositionTable(res, modelZLB, modelZLBRANK, outputFolderAdditional,)
    generateRateDecompositionTable(res, modelZLBHighRotembergCost, modelZLBRANKHighRotembergCost, outputFolderAdditional, "HighRotembergCost")
    generateComparativeStaticsTable(res, modelZLB, modelsCompStat, modelsCompStatPlus, σLabels, πTargets, outputFolderAdditional)

    # Additional statistics only shown in REPL
    computeAdditionalStatistics(res, modelZLB)
    computeAdditionalStatistics(res, modelNoZLB)

    display("Plots generated!")

    nothing

end


"""
    plotPLMInflation(res, modelZLB, outputFolder)

Plots the PLM for inflation for a given solution file.

"""
function plotPLMInflation(res, modelZLB, outputFolder; filename = "PLMInflation")

    # Make some variables more easily accesible
    DSS = res[modelZLB][:DSS]
    simSeries = res[modelZLB][:simSeriesPlus] # To make it easier to draw use simSeriesPlus which only has 10000 samples
    P = res[modelZLB][:P]
    πwALM = res[modelZLB][:πwALM]

    # Add simulated data points and differentiate between cases where the ZLB is binding and where it's not
    ZLBCheck = HANKWageRigidities.checkZLB.(Ref(P), Ref(DSS), simSeries[:π][2:end], simSeries[:Y][2:end], simSeries[:RStar][1:end-1])
    RStarSimZLB =  simSeries[:RStar][1:end-1][ZLBCheck .== 1]
    πSimZLB = simSeries[:π][2:end][ZLBCheck .== 1]
    ζSimZLB =  simSeries[:ζ][2:end][ZLBCheck .== 1]
    RStarSimNotZLB = simSeries[:RStar][1:end-1][ZLBCheck .== 0]
    πSimNotZLB = simSeries[:π][2:end][ZLBCheck .== 0]
    ζSimNotZLB = simSeries[:ζ][2:end][ZLBCheck .== 0]
    smplZLB = 1:1:length(ζSimZLB)
    smplNotZLB = 1:1:length(ζSimNotZLB)

    p1 = scatter3d(log.(RStarSimZLB[smplZLB])*400, ζSimZLB[smplZLB], log.(πSimZLB[smplZLB])*400,
        color = palette(:RdYlGn_9)[1],
        markersize = 2,
        markerstrokewidth = 0.4,
        xlabel = L"Nominal Rate $R_{t-1}$ (\%)",
        ylabel = L"Preference Shock $\xi_t$",
        zlabel = L"Inflation $\pi_t$ (\%)",
        camera = (-60,20),
        legend = :none, margin = 5Plots.PlotMeasures.mm, cbar = :none)
    scatter3d!(p1, log.(RStarSimNotZLB[smplNotZLB])*400, ζSimNotZLB[smplNotZLB], log.(πSimNotZLB[smplNotZLB])*400,
        color = palette(:RdYlGn_9)[9],
        markersize = 2,
        markerstrokewidth = 0.4)
    zlims!(-4.0, 5.0)
    ylims!(0.945, 1.055)
    #display(p1)
    #savefig(p1, "$(outputFolder)/PLMInflationOnlySimulatedData.pdf")

    p2 = surface(log.(P.RDenseGrid)*400, P.ζDenseGrid, log.(πwALM')*400,
        xlabel = L"Nominal Rate $R_{t-1}$ (\%)",
        ylabel = L"Preference Shock $\xi_t$",
        zlabel = L"Inflation $\pi_t$ (\%)",
        camera = (-60,20),
        legend = :none, margin = 5Plots.PlotMeasures.mm, cbar = :none,
        colormap_name = "viridis",
        extra_kwargs = :subplot)
    zlims!(-4.0, 5.0)
    ylims!(0.945, 1.055)
    #display(p2)
    #savefig(p2, "$(outputFolder)/PLMInflation.pdf")

    title!(p1, L"\textrm{(b) Simulated Inflation }\pi(\xi_t,R_{t-1})")
    title!(p2, L"\textrm{(a) Perceived Inflation }\hat{\pi}(\xi_t,R_{t-1})")
    p = plot(p2, p1, layout = grid(1,2), size = (900, 300))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")
    savefig(p, "$(outputFolder)/$(filename).png")
    #savefig(p, "$(outputFolder)/$(filename).tex")

    nothing

end


"""
    plotForecastErrorsInflation(res, modelZLB, modelZLBLinear, outputFolder)

Plots the forecast errors for inflation for given solution files.

"""
function plotForecastErrorsInflation(res, modelZLB, modelZLBLinear, outputFolder; 
        nbins = -0.5:0.02:1.5, minError = -0.25, maxError = 1.0, filename = "ForecastErrorsInflation",
        labelZLB = "Neural Network", labelLinear = "Linear")

    # Make some variables more easily accesible
    P = res[modelZLB][:P]
    DSS = res[modelZLB][:DSS] # DSS is the same in both modelZLB and modelZLBLinear
    simSeries = res[modelZLB][:simSeries]
    simSeriesLinear = res[modelZLBLinear][:simSeries]
    πwALMInterpol = res[modelZLB][:πwALMInterpol]
    πwALMInterpolLinear = res[modelZLBLinear][:πwALMInterpol]

    # Forecast errors inflation PLM
    errorsNN, statsNN = computeForecastErrorsInflationPLM(simSeries[:πw], simSeries[:RStar], simSeries[:ζ], πwALMInterpol; rescaleErrors = true)
    errorsLinear, statsLinear = computeForecastErrorsInflationPLM(simSeriesLinear[:πw], simSeriesLinear[:RStar], simSeriesLinear[:ζ], πwALMInterpolLinear; rescaleErrors = true)

    p1 = histogram(errorsNN,
        label = L"%$labelZLB (R$^2$ = %$(round(statsNN.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        xlabel = "Errors (pp)",
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    histogram!(errorsLinear,
        label = L"%$labelLinear (R$^2$ = %$(round(statsLinear.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    # Forecast errors for ZLB periods only
    ZLBCheck = HANKWageRigidities.checkZLB.(Ref(P), Ref(DSS), simSeries[:π][2:end], simSeries[:Y][2:end], simSeries[:RStar][1:end-1])
    errorsNN, statsNN = computeForecastErrorsInflationPLMZLBOnly(simSeries[:πw], simSeries[:RStar], simSeries[:ζ], πwALMInterpol, ZLBCheck; rescaleErrors = true)
    ZLBCheckLinear = HANKWageRigidities.checkZLB.(Ref(P), Ref(DSS), simSeriesLinear[:π][2:end], simSeriesLinear[:Y][2:end], simSeriesLinear[:RStar][1:end-1])
    errorsLinear, statsLinear = computeForecastErrorsInflationPLMZLBOnly(simSeriesLinear[:πw], simSeriesLinear[:RStar], simSeriesLinear[:ζ], πwALMInterpolLinear, ZLBCheckLinear; rescaleErrors = true)

    p2 = histogram(errorsNN,
        label = L"%$labelZLB (R$^2$ = %$(round(statsNN.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        xlabel = "Errors (pp)",
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    histogram!(errorsLinear,
        label = L"%$labelLinear (R$^2$ = %$(round(statsLinear.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    #display(p2)
    #savefig(p2, "$(outputFolder)/PLMForecastErrorsInflationZLBOnly.pdf")

    plot!(p1, title = "(a) All Simulated Periods")
    plot!(p2, title = "(b) Only Periods with binding ZLB")
    p = plot(p1, p2, layout = grid(1,2), size = (900, 300))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")

    nothing

end


"""
    plotForecastErrorsInflationExpectations(res, modelZLB, modelZLBLinear, outputFolder)

Plots the forecast errors for inflation expectations for given solution files.

"""
function plotForecastErrorsInflationExpectations(res, modelZLB, modelZLBLinear, outputFolder;
        nbins = -0.5:0.02:1.5, minError = -0.25, maxError = 1.0, filename = "ForecastErrorsInflationExpectation",
        labelZLB = "Neural Network", labelLinear = "Linear")

    # Make some variables more easily accesible
    P = res[modelZLB][:P]
    DSS = res[modelZLB][:DSS] # DSS is the same in both modelZLB and modelZLBLinear
    simSeries = res[modelZLB][:simSeries]
    simSeriesLinear = res[modelZLBLinear][:simSeries]
    EπwCondALMInterpol = res[modelZLB][:EπwCondALMInterpol]
    EπwCondALMInterpolLinear = res[modelZLBLinear][:EπwCondALMInterpol]

    # Forecast errors inflation expectation PLM
    errorsNN, statsNN = computeForecastErrorsInflationExpectationPLM(simSeries[:EπwCond], simSeries[:RStar], simSeries[:ζ], EπwCondALMInterpol; rescaleErrors = true)
    errorsLinear, statsLinear = computeForecastErrorsInflationExpectationPLM(simSeriesLinear[:EπwCond], simSeriesLinear[:RStar], simSeriesLinear[:ζ], EπwCondALMInterpolLinear; rescaleErrors = true)

    p1 = histogram(errorsNN,
        label = L"%$labelZLB (R$^2$ = %$(round(statsNN.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        xlabel = "Errors (pp)",
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    histogram!(errorsLinear,
        label = L"%$labelLinear (R$^2$ = %$(round(statsLinear.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    #display(p1)
    #savefig(p1, "$(outputFolder)/PLMForecastErrorsInflationExpectation.pdf")

    # Forecast errors inflation expectations for ZLB periods only
    ZLBCheck = HANKWageRigidities.checkZLB.(Ref(P), Ref(DSS), simSeries[:π][2:end-1], simSeries[:Y][2:end-1], simSeries[:RStar][1:end-2])
    errorsNN, statsNN = computeForecastErrorsInflationExpectationPLMZLBOnly(simSeries[:EπwCond], simSeries[:RStar], simSeries[:ζ], EπwCondALMInterpol, ZLBCheck; rescaleErrors = true)
    ZLBCheckLinear = HANKWageRigidities.checkZLB.(Ref(P), Ref(DSS), simSeriesLinear[:π][2:end-1], simSeriesLinear[:Y][2:end-1], simSeriesLinear[:RStar][1:end-2])
    errorsLinear, statsLinear = computeForecastErrorsInflationExpectationPLMZLBOnly(simSeriesLinear[:EπwCond], simSeriesLinear[:RStar], simSeriesLinear[:ζ], EπwCondALMInterpolLinear, ZLBCheckLinear; rescaleErrors = true)

    p2 = histogram(errorsNN,
        label = L"%$labelZLB (R$^2$ = %$(round(statsNN.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        xlabel = "Errors (pp)",
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    histogram!(errorsLinear,
        label = L"%$labelLinear (R$^2$ = %$(round(statsLinear.R2*100, digits = 2))%)",
        xlim = (minError, maxError),
        linewidth = 0.2,
        fillalpha = 0.5,
        normalize = :density,
        nbins = nbins,
        legend = :topright)

    #display(p2)
    #savefig(p2, "$(outputFolder)/PLMForecastErrorsInflationExpectationZLBOnly.pdf")

    plot!(p1, title = "(a) All Simulated Periods")
    plot!(p2, title = "(b) Only Periods with binding ZLB")
    p = plot(p1, p2, size = (900, 300))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")

end


"""
    plotErgodicDistribution(res, modelZLB, modelNoZLB, outputFolder)

Plots a comparison of the ergodic distribution of two given solution files

"""
function plotErgodicDistribution(res, modelZLB, modelNoZLB, outputFolder; 
        labelZLB = "ZLB-HANK", labelNoZLB = "HANK", filename = "ErgodicDistributionZLBvsNoZLB")

    function plotHist(x1, x2, nbins, xlabel; normalize = :probability)

        p = histogram(x1,
            xlabel = xlabel,
            linewidth = 0.2,
            fillalpha = 0.5,
            normalize = normalize,
            nbins = nbins,
            label = "",
            legend = :topleft)

        histogram!(x2,
            linewidth = 0.2,
            fillalpha = 0.5,
            normalize = normalize,
            nbins = nbins,
            label = "")

        return p

    end

    # Inflation
    p1 = plotHist(log.(res[modelZLB][:simSeries][:π])*400,
                  log.(res[modelNoZLB][:simSeries][:π])*400,
                  -6:0.25:10,
                  L"Inflation $\pi_t$ (\%)")

    # Nominal Interest Rate
    p2 = plotHist(log.(res[modelZLB][:simSeries][:R])*400,
                  log.(res[modelNoZLB][:simSeries][:R])*400,
                  -6:0.25:10,
                  L"Nominal Rate $R_{t-1}$ (\%)")

    # Real Interest Rate
    p3 = plotHist(log.(res[modelZLB][:simSeries][:r])*400,
                  log.(res[modelNoZLB][:simSeries][:r])*400,
                  -6:0.25:10,
                  L"Real Rate $r_t$ (\%)")

    # Aggregate Consumption
    p4 = plotHist(res[modelZLB][:simSeries][:C],
                  res[modelNoZLB][:simSeries][:C],
                  0.9:0.0025:1.05,
                  L"Aggregate Consumption $C_t$")

    p1.series_list[1][:label] = labelZLB
    p1.series_list[3][:label] = labelNoZLB

    p = plot(p1, p2, p3, p4, layout = grid(2,2), size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")

    nothing

end


"""
    plotImpulseResponseFunctions(res, modelZLB, modelNoZLB, outputFolder)

Plot comparison of impulse responses.

"""
function plotImpulseResponseFunctions(res, modelZLB, modelNoZLB, outputFolder;
        labelZLB = "ZLB-HANK", labelNoZLB = "HANK", filename = "IRFComparisonZLBvsNoZLB")

    linewidth = 2

    # Collect the IRFs
    IRFs = [res[modelZLB][:IRFs1],
            res[modelZLB][:IRFs3],
            res[modelNoZLB][:IRFs1],
            res[modelNoZLB][:IRFs3]]

    # Collect the SSS associated to the IRFs
    SSSs = [res[modelZLB][:SSS],
            res[modelZLB][:SSS],
            res[modelNoZLB][:SSS],
            res[modelNoZLB][:SSS]]

    # Define labels for the IRFs
    IRFLabels = ["$labelZLB (1 std)",
                 "$labelZLB (3 std)",
                 "$labelNoZLB (1 std)",
                 "$labelNoZLB (3 std)"]

    # Define the IRF colors and line
    IRFStyles = [(color = palette(:Paired_10)[5], linestyle = :solid),
                 (color = palette(:Paired_10)[6], linestyle = :solid),
                 (color = palette(:Paired_10)[1], linestyle = :dash),
                 (color = palette(:Paired_10)[2], linestyle = :dash)]

    # Define variables to be plotted
    variables = [:π, :r, :R, :Y, :T, :ξ] # Note: τ_t = -T_t have the same IRFs
    variableNames = [L"Inflation $\pi_t$ (pp)",
                     L"Real Rate $r_t$ (pp)",
                     L"Nominal Rate $R_{t}$ (pp)",
                     L"Output $Y_t$ (\%)",
                     L"Taxes $\tau_t$ (\%)",
                     L"Preference Shock $\xi_t$ (\%)"]

    # Define which variables are interest rates
    interestRateList = [:r, :π, :R]

    # Shown period
    period = 1:8

    # Create the IRF plot
    pAll =  plotIRFComparison(IRFs, SSSs, IRFLabels, IRFStyles, variables, variableNames, interestRateList, period, linewidth)
    p = plot(pAll..., layout = grid(3,2), size = (720, 720))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")

    nothing

end


"""
    generateDSSSSSErgodicMeanTable(res, modelZLB, modelNoZLB, outputFolder)

Generates table with comparison of DSS, SSS and Ergodic Means for two given solution files.

"""
function generateDSSSSSErgodicMeanTable(res, modelZLB, modelNoZLB, outputFolder; 
        labelZLB = "ZLB-HANK", labelNoZLB = "HANK", filename = "SteadyStatesAndErgodicMeansZLBvsNoZLB")

    # Settings
    numberFormat = "%2.2f"
    addGroupLines = true

    # Define table caption
    caption = "Comparison of DSS, SSS and Ergodic Means ($labelZLB vs $labelNoZLB)"

    # Define the variables to be printed
    variables = OrderedDict(:r => "Real Rate (\$r_t\$; \\%)",
                            :π => "Inflation (\$\\pi_t\$; \\%)",
                            :R => "Nominal Rate (\$R_t\$; \\%)",
                            :Y => "Output (\$Y_t\$)",
                            :C => "Consumption (\$C_t\$)",
                            :w => "Wage (\$w_t\$)",
                            :ZLBFreq => "(Shadow) ZLB Frequency (\\%)",
                            :ZLBSpellMean => "(Shadow) ZLB Spell Duration"
                            )

    # Define which variables are (quarterly) interest rates
    interestRateList = [:r, :π, :R]

    # Define which variables are in percent
    percentList = [:ZLBFreq]

    # Function that returns a named tuple with simulation averages
    function getSimAverages(res, model)

        # Compute means of the existing variables
        varNames = Tuple(keys(res[model][:simSeries]))
        varValues = [mean(x) for x in values(res[model][:simSeries])]

        # Compute ZLB spell duration and ZLB frequency
        bindingZLBIndicator = res[model][:simSeries][:RStar] .<= 1.0
        freqZLB = sum(bindingZLBIndicator) / length(bindingZLBIndicator)
        durations = Int64[]
        tmpDur = 0

        for ii in 1:length(bindingZLBIndicator)

            if bindingZLBIndicator[ii]
                tmpDur += 1
            else

                if tmpDur != 0
                    push!(durations, tmpDur)
                    tmpDur = 0
                end

            end

        end

        meanDuration = mean(durations)

        # Add the new variables
        varNames = tuple(varNames..., :ZLBFreq, :ZLBSpellMean)
        varValues = [varValues; freqZLB; meanDuration]

        return NamedTuple{varNames}(varValues)

    end

    # Collect the SSS and DSS variables
    steadyStates = [res[modelZLB][:DSS],
                    res[modelZLB][:SSS],
                    getSimAverages(res, modelZLB),
                    res[modelNoZLB][:DSS],
                    res[modelNoZLB][:SSS],
                    getSimAverages(res, modelNoZLB)]

    # Define labels for the steady states
    steadyStateLabels = ["DSS",
                         "SSS",
                         "Mean",
                         "DSS",
                         "SSS",
                         "Mean"]
    superLabels = [(labelZLB, 3), (labelNoZLB, 3)]

    #
    allLines = []

    # Add headings
    push!(allLines, "\\begin{table}[h]")
    push!(allLines, "\\centering")
    push!(allLines, "\\caption{$(caption)}")
    push!(allLines, "\\begin{tabular}{l*{$(length(steadyStateLabels))}{S[table-format=2.2]}}")
    push!(allLines, "\\toprule")

    # Generate the column "super" labels
    if length(superLabels) > 0
        currentLine = " "
        for ii in 1:length(superLabels)

            if superLabels[ii][2] == 1
                currentLine *= " & {$(superLabels[ii][1])}"
            else
                currentLine *= " & \\multicolumn{$(superLabels[ii][2])}{c}{$(superLabels[ii][1])}"
            end

        end
        currentLine *= "\\\\"
        push!(allLines, currentLine)
    end

    # Add lines below HANK, RANK, etc.
    if addGroupLines
        currentLine = ""
        currentColumn = 2
        for ii in 1:length(superLabels)
            if superLabels[ii][1] != ""
                currentLine *= "\\cmidrule(lr){$(currentColumn)-$(currentColumn+superLabels[ii][2]-1)}"
            end
            currentColumn = currentColumn + superLabels[ii][2]
        end
        push!(allLines, currentLine)
    end

    # Generate the column labels
    currentLine = "Variables "
    for ii in 1:length(steadyStateLabels)
        currentLine *= " & {$(steadyStateLabels[ii])}"
    end
    currentLine *= "\\\\"
    push!(allLines, currentLine)
    push!(allLines, "\\midrule")

    # Generate the main contents of the table
    for var in keys(variables)

        currentLine = variables[var]

        for ii in 1:length(steadyStates)

            if haskey(steadyStates[ii], var)
                if var in interestRateList
                    currentValue = sprintf1(numberFormat, log(steadyStates[ii][var])*400)
                elseif var in percentList
                    currentValue = sprintf1(numberFormat, steadyStates[ii][var]*100)
                else
                    currentValue = sprintf1(numberFormat, steadyStates[ii][var])
                end
            else
                currentValue = "{-}"
            end

            currentLine *= " & $(currentValue)"

        end

        currentLine *= "\\\\"

        push!(allLines, currentLine)

    end

    # Finish the table
    push!(allLines, "\\bottomrule")
    push!(allLines, "\\end{tabular}")
    push!(allLines, "\\end{table}")
    display(allLines)

    # Write the table to a file
    open("$(outputFolder)/$(filename).tex", "w") do f
        for line in allLines
            println(f, line)
        end
    end

end


"""
    generateDSSSSSTable(res, modelZLB, modelNoZLB, modelZLBRANK, modelNoZLBRANK, outputFolder)

Generates table with comparison of DSS and SSS for four given solution files.

"""
function generateDSSSSSErgodicMeanTable(res, modelZLB, modelNoZLB, modelZLBRANK, modelNoZLBRANK, outputFolder; 
        labelZLB = "ZLB-HANK", labelNoZLB = "HANK", labelZLBRANK = "ZLB-RANK", labelNoZLBRANK = "RANK", 
        filename = "SteadyStatesZLBvsNoZLB")

    # Settings
    numberFormat = "%2.2f"
    addGroupLines = true

    # Define table caption
    caption = "Comparison of DSS and SSS in $labelZLB, $labelNoZLB, $labelZLBRANK, and $labelNoZLBRANK."

    # Define the variables to be printed
    variables = OrderedDict(
        :π => "Inflation",
        :R => "Nominal Rate",
        :r => "Real Rate",
        :ZLBFreq => "(Shadow) ZLB Frequency"
    )

    # Define which variables are (quarterly) interest rates
    interestRateList = [:r, :π, :R]

    # Define which variables are in percent
    percentList = [:ZLBFreq]

    # Define which variables arise from simulations
    simList = [:ZLBFreq]

    # Function that returns a named tuple with simulation averages
    function getSimAverages(res, model)

        # Compute means of the existing variables
        varNames = Tuple(keys(res[model][:simSeries]))
        varValues = [mean(x) for x in values(res[model][:simSeries])]

        # Compute ZLB spell duration and ZLB frequency
        bindingZLBIndicator = res[model][:simSeries][:RStar] .<= 1.0
        freqZLB = sum(bindingZLBIndicator) / length(bindingZLBIndicator)
        durations = Int64[]
        tmpDur = 0

        for ii in 1:length(bindingZLBIndicator)

            if bindingZLBIndicator[ii]
                tmpDur += 1
            else

                if tmpDur != 0
                    push!(durations, tmpDur)
                    tmpDur = 0
                end

            end

        end

        meanDuration = mean(durations)

        # Add the new variables
        varNames = tuple(varNames..., :ZLBFreq, :ZLBSpellMean)
        varValues = [varValues; freqZLB; meanDuration]

        return NamedTuple{varNames}(varValues)

    end

    # Collect the SSS and DSS variables
    steadyStates = [res[modelZLB][:DSS],
                    res[modelZLB][:SSS],
                    res[modelNoZLB][:DSS],
                    res[modelNoZLB][:SSS],
                    res[modelZLBRANK][:DSS],
                    res[modelZLBRANK][:SSS],
                    res[modelNoZLBRANK][:DSS],
                    res[modelNoZLBRANK][:SSS],]

    simStats = [(dummy=NaN,),
                getSimAverages(res, modelZLB),
                (dummy=NaN,),
                getSimAverages(res, modelNoZLB),
                (dummy=NaN,),
                getSimAverages(res, modelZLBRANK),
                (dummy=NaN,),
                getSimAverages(res, modelNoZLBRANK),]


    # Define labels for the steady states
    steadyStateLabels = ["DSS",
                         "SSS",
                         "DSS",
                         "SSS",
                         "DSS",
                         "SSS",
                         "DSS",
                         "SSS",]
    superLabels = [(labelZLB, 2), (labelNoZLB, 2), (labelZLBRANK, 2), (labelNoZLBRANK, 2)]

    #
    allLines = []

    # Add headings
    push!(allLines, "\\begin{table}[h]")
    push!(allLines, "\\centering")
    push!(allLines, "\\caption{$(caption)}")
    push!(allLines, "\\begin{tabular}{l*{$(length(steadyStateLabels))}{S[table-format=2.2]}}")
    push!(allLines, "\\toprule")

    # Generate the column "super" labels
    if length(superLabels) > 0
        currentLine = " "
        for ii in 1:length(superLabels)

            if superLabels[ii][2] == 1
                currentLine *= " & {$(superLabels[ii][1])}"
            else
                currentLine *= " & \\multicolumn{$(superLabels[ii][2])}{c}{$(superLabels[ii][1])}"
            end

        end
        currentLine *= "\\\\"
        push!(allLines, currentLine)
    end

    # Add lines below HANK, RANK, etc.
    if addGroupLines
        currentLine = ""
        currentColumn = 2
        for ii in 1:length(superLabels)
            if superLabels[ii][1] != ""
                currentLine *= "\\cmidrule(lr){$(currentColumn)-$(currentColumn+superLabels[ii][2]-1)}"
            end
            currentColumn = currentColumn + superLabels[ii][2]
        end
        push!(allLines, currentLine)
    end

    # Generate the column labels
    currentLine = "Variables "
    for ii in 1:length(steadyStateLabels)
        currentLine *= " & {$(steadyStateLabels[ii])}"
    end
    currentLine *= "\\\\"
    push!(allLines, currentLine)
    push!(allLines, "\\midrule")

    # Generate the main contents of the table
    for var in keys(variables)

        currentLine = variables[var]

        for ii in 1:length(steadyStates)

            if haskey(steadyStates[ii], var) && var ∉ simList
                if var in interestRateList
                    currentValue = sprintf1(numberFormat, log(steadyStates[ii][var])*400)
                    currentValue *= "\\%"
                elseif var in percentList
                    currentValue = sprintf1(numberFormat, steadyStates[ii][var]*100)
                    currentValue *= "\\%"
                else
                    currentValue = sprintf1(numberFormat, steadyStates[ii][var])
                end
            elseif haskey(simStats[ii], var) && var ∈ simList
                if var in interestRateList
                    currentValue = sprintf1(numberFormat, log(simStats[ii][var])*400)
                    currentValue *= "\\%"
                elseif var in percentList
                    currentValue = sprintf1(numberFormat, simStats[ii][var]*100)
                    currentValue *= "\\%"
                else
                    currentValue = sprintf1(numberFormat, simStats[ii][var])
                end
            else
                currentValue = "{-}"
            end

            currentLine *= " & $(currentValue)"

        end

        currentLine *= "\\\\"

        push!(allLines, currentLine)

    end

    # Finish the table
    push!(allLines, "\\bottomrule")
    push!(allLines, "\\end{tabular}")
    push!(allLines, "\\end{table}")
    display(allLines)

    # Write the table to a file
    open("$(outputFolder)/$(filename).tex", "w") do f
        for line in allLines
            println(f, line)
        end
    end

end


"""
    plotResponseDecomposition(res, modelB, modelA, outputFolder)

Plots the response decomposition.

"""
function plotResponseDecomposition(res, modelB, modelA, outputFolder; shortLabelA = "HANK", shortLabelB = "ZLB-HANK", 
        prodTypes = [:all, :average, :only1, :only2, :only3], saveOnlyCombinedPlots = false, skipAdditionalFigure = true, 
        outputFolderAdditional = outputFolder, plotType = :comparison)

    for prodType = prodTypes

        # Settings
        periods = 1:1       # Range of periods used for computations
        refIRF = :IRFs3     # Precomputed IRFs that are used for computations
        decompType = :percentile # Categories for which decomposition is shown: :borrowersSavers, :percentile, :percentileNoAgg
        #prodType = 1        # Productivity type used for :percentile and :percentileNoAgg
        percList = [10, 99] # Percentile list used for :percentile and :percentileNoAgg
        plotAvsB = false    # true: switch model A with model B in the final plot
        useCumulativeResponse = false  # true: sums the percentage response for all periods defined above

        # Labels of the components
        labels = Dict()
        labels[:interestIncome] = L"Interest$\;$" # Space at the end to improve legend spacing
        labels[:laborIncome] = L"Labor Income$\;$"
        labels[:transferIncome] = L"Taxes$\;$"

        # Initalize dict with results
        results = Dict()

        # Compute the income response decompositions for each model
        for model in [modelA, modelB]

            # Extract the settings, IRFs and SSS from the results for easier access
            P = res[model][:P]
            IRFs = res[model][refIRF]
            SSS = res[model][:SSS]

            # Define the wealth levels that need to be evaluated
            if decompType == :borrowersSavers
                bSet = P.bDenseGrid
            elseif decompType == :percentileNoAgg
                bSet = [HANKWageRigidities.computePercentile(vec(sum(SSS[:bCross], dims = 2)), P.bDenseGrid, perc / 100) for perc in percList]
            elseif decompType == :percentile
                bSet = [HANKWageRigidities.computePercentile(vec(sum(SSS[:bCross], dims = 2)), P.bDenseGrid, perc / 100) for perc in percList]
                bSet = [bSet; P.bDenseGrid]
            else
                error("Unknown decompType")
            end

            # Compute the income components at each point in time
            incomeComponents = HANKWageRigidities.computeIncomeComponentsGrid(P, periods, bSet, IRFs, SSS)

            # Compute the income components in the SSS
            incomeComponentsSSS = HANKWageRigidities.computeIncomeComponentsGridSSS(P, bSet, SSS)

            # Aggregate income components (e.g. to get borrowers and savers)
            incomeComponents, incomeComponentsSSS = HANKWageRigidities.aggregateIncomeComponentsGrid(P, decompType, periods, percList, prodType, bSet, IRFs, SSS, incomeComponents, incomeComponentsSSS)

            # Compute contributions of income components (analogous to computing "growth contributions")
            for incomeType in keys(incomeComponents)

                rate = incomeComponents[incomeType] ./ incomeComponentsSSS[incomeType] .- 1
                weights = incomeComponentsSSS[incomeType] ./ incomeComponentsSSS[:totalIncome]
                incomeComponents[incomeType] = rate .* weights * 100

            end

            # Check whether there are issues in the previous step (usually this is due to a component being zero in levels)
            for incomeType in keys(incomeComponents)

                if any(isnan.(incomeComponents[incomeType]))
                    @warn "$incomeType in $model has NaN values"
                    delete!(incomeComponents, incomeType)
                    #incomeComponents[incomeType][isnan.(incomeComponents[incomeType])] .= 0.0
                end

            end

            # Save the results for the current model
            results[model] = incomeComponents

        end

        # Compute the difference in the respone across the two models
        # or only show one of the two responses
        responseDiff = Dict()
        for incomeType in keys(results[modelA])

            if plotType == :comparison

                # Switch model A and B if required
                if plotAvsB
                    responseDiff[incomeType] = results[modelA][incomeType] - results[modelB][incomeType]
                else
                    responseDiff[incomeType] = results[modelB][incomeType] - results[modelA][incomeType]
                end

            elseif plotType == :modelAOnly

                responseDiff[incomeType] = results[modelA][incomeType]

            elseif plotType == :modelBOnly

                responseDiff[incomeType] = results[modelB][incomeType]

            end

            # Compute the cumulative response if required
            if useCumulativeResponse
                responseDiff[incomeType] = sum(responseDiff[incomeType], dims = 1)
            end

        end

        # Generate Plot

        # Some labels used in the plot
        cumLabel = useCumulativeResponse ? "Cumulative" : ""
        if plotType == :comparison
            modelComparisonLabel = plotAvsB ? "$shortLabelA - $shortLabelB" : "$shortLabelB - $shortLabelA"
            titleLabel = "Δ $cumLabel Income Response (pp; $modelComparisonLabel)"
        elseif plotType in [:modelAOnly, :modelBOnly]
            modelComparisonLabel = plotType == :modelAOnly ? "$shortLabelA" : "$shortLabelB"
            titleLabel = "$cumLabel Income Response (%; $modelComparisonLabel)"
        end

        if decompType == :borrowersSavers
            nam = repeat(["Borrowers", "Savers", "Aggregate"], outer = 3)
        elseif decompType == :percentileNoAgg
            nam = repeat(string.(percList) .* "th Wealth Percentile", outer = 3)
        elseif decompType == :percentile
            nam = repeat(vcat(string.(percList) .* "th Wealth Percentile", "Aggregate"), outer = 3)
        end

        nam = "\\textrm{".* nam .* "}"

        # Initialize the plot
        p1 = plot(ylabel = titleLabel, legend = (at = (0.5, -0.10), anchor = "north"), extra_kwargs=:subplot, legend_columns=-1)
        remainingTypes = setdiff(keys(responseDiff), tuple(:totalIncome)) # All income types except :totalIncome
        ii = 1
        bar_width = 0.67

        for incomeType in setdiff(keys(responseDiff), tuple(:totalIncome))

            # Determine the positive and negative responses
            ApInd = [responseDiff[t] .> 0.0 for t in remainingTypes]
            AmInd = [responseDiff[t] .<= 0.0 for t in remainingTypes]

            # Sum up positive and negative responses separately (accross income types)
            # Note: Since the bars will be overlayed by bars plotted in the next iteration,
            # the only part visible of the current iteration corresponds to the current incomeType
            # (eventhough we have summed up over all remainingTypes)
            Ap = sum([responseDiff[Tuple(remainingTypes)[jj]] .* ApInd[jj] for jj in 1:length(remainingTypes)])'
            Am = sum([responseDiff[Tuple(remainingTypes)[jj]] .* AmInd[jj] for jj in 1:length(remainingTypes)])'

            # Plot negative bars
            groupedbar!(p1, nam, Am,
                bar_width = bar_width,
                lw = 0.2,
                color = palette(:YlGnBu_5)[ii],
                label = [labels[incomeType] repeat([""], 1, 2*size(Am, 2))],
                extra_kwargs = :subplot, legend_columns = -1
            )

            # Plot positive bars
            groupedbar!(p1, nam, Ap,
                bar_width = bar_width,
                lw = 0.2,
                color = palette(:YlGnBu_5)[ii],
                label = repeat([""], 1, 2*size(Ap, 2) + 1),
                extra_kwargs = :subplot, legend_columns = -1
            )

            # Combine the bars for the aggregate since they are all the same anyways
            if prodType == :all && decompType == :percentile

                bar!(p1, [size(Am,1)-0.5], [Am[end, 1]],
                    bar_width = bar_width,
                    lw = 0.2,
                    color = palette(:YlGnBu_5)[ii],
                    label = repeat([""], 1, 2*size(Am, 2)),
                    extra_kwargs = :subplot, legend_columns = -1
                )

                bar!(p1, [size(Ap,1)-0.5], [Ap[end, 1]],
                    bar_width = bar_width,
                    lw = 0.2,
                    color = palette(:YlGnBu_5)[ii],
                    label = repeat([""], 1, 2*size(Ap, 2)),
                    extra_kwargs = :subplot, legend_columns = -1
                )

            end

            # Remove the current income type from remainingTypes
            remainingTypes = setdiff(remainingTypes, tuple(incomeType))
            ii += 1

        end

        # Compute the midpoint of each plotted bar
        positionTotal = []

        for ii in 1:size(responseDiff[:totalIncome], 2)

            dist = bar_width / size(responseDiff[:totalIncome], 1)
            halfDist = dist / 2

            pts = ((ii - 1 + 0.5) - bar_width / 2 + halfDist):dist:((ii - 1 + 0.5) + bar_width / 2)

            for jj in 1:size(responseDiff[:totalIncome], 1)
                push!(positionTotal, pts[jj])
            end

        end

        # Remove superflous total response vales
        vecResponseDiff = vec(responseDiff[:totalIncome])
        if prodType == :all && decompType == :percentile
            vecResponseDiff[end] = NaN
            vecResponseDiff[end-2] = NaN
        end

        # Plot the total response
        scatter!(positionTotal, vecResponseDiff,
            color = :white, marker = :diamond, markersize = 6, label = "Total",
            extra_kwargs = :subplot, legend_columns = -1)

        #
        hline!(p1, [0.0], linestyle = :dash, color = :black, label = "",
            extra_kwargs = :subplot, legend_columns = -1)
        if plotType == :comparison
            ylims!(p1, -2.0, 2.2)
        else
            ylims!(p1, -4.0, 6.0)
        end
        plot!(p1, foreground_color_legend = nothing, background_color_legend = nothing)

        # Save the figure
        filenameExt = "ProdType" * uppercase("$(prodType)"[1:1]) * "$(prodType)"[2:end] * (plotType == :modelAOnly ? "" : "_$(shortLabelA)") * (plotType == :modelBOnly ? "" : "_$(shortLabelB)")
        if !saveOnlyCombinedPlots
            savefig(p1, "$(outputFolder)/IncomeResponse$(filenameExt).pdf")
        end


        ## Consumption Response ################################################

        # Reuse settings from income response

        # Initalize dict with results
        results = Dict()

        # Compute the income response decompositions for each model
        for model in [modelA, modelB]

            # Extract the settings, IRFs and SSS from the results for easier access
            P = res[model][:P]
            IRFs = res[model][refIRF]
            SSS = res[model][:SSS]
            bPolicyInterpol = res[model][:bPolicyInterpol]

            # Define the wealth levels that need to be evaluated
            if decompType == :borrowersSavers
                bSet = P.bDenseGrid
            elseif decompType == :percentileNoAgg
                bSet = [HANKWageRigidities.computePercentile(vec(sum(SSS[:bCross], dims = 2)), P.bDenseGrid, perc / 100) for perc in percList]
            elseif decompType == :percentile
                bSet = [HANKWageRigidities.computePercentile(vec(sum(SSS[:bCross], dims = 2)), P.bDenseGrid, perc / 100) for perc in percList]
                bSet = [bSet; P.bDenseGrid]
            else
                error("Unknown decompType")
            end

            # Compute consumption at each point in time
            consumptionGrid = HANKWageRigidities.computeConsumptionGrid(P, periods, bSet, IRFs, SSS, bPolicyInterpol)

            # Compute consumption in the SSS
            consumptionGridSSS = HANKWageRigidities.computeConsumptionGridSSS(P, bSet, SSS, bPolicyInterpol)

            # Aggregate income components (e.g. to get borrowers and savers)
            consumption, consumptionSSS = HANKWageRigidities.aggregateConsumptionComponentsGrid(P, decompType, periods, percList, prodType, bSet, IRFs, SSS, consumptionGrid, consumptionGridSSS)

            # Compute consumption IRF for the percentiles and aggregates
            consumptionIRF = (consumption ./ consumptionSSS .- 1) * 100

            # Save the results for the current model
            results[model] = consumptionIRF

        end

        # Compute the difference in the respone across the two models
        # or only show one of the two responses
        if plotType == :comparison

            # Switch model A and B if required
            if plotAvsB
                responseDiff = results[modelA] - results[modelB]
            else
                responseDiff = results[modelB] - results[modelA]
            end

        elseif plotType == :modelAOnly

            responseDiff = results[modelA]

        elseif plotType == :modelBOnly

            responseDiff = results[modelB]

        end

        # Compute the cumulative response if required
        if useCumulativeResponse
            responseDiff = sum(responseDiff, dims = 1)
        end


        # Generate Plot

        # Some labels used in the plot
        cumLabel = useCumulativeResponse ? "Cumulative" : ""
        if plotType == :comparison
            modelComparisonLabel = plotAvsB ? "$shortLabelA - $shortLabelB" : "$shortLabelB - $shortLabelA"
            titleLabel = "Δ $cumLabel Consumption Response (pp; $modelComparisonLabel)"
        elseif plotType in [:modelAOnly, :modelBOnly]
            modelComparisonLabel = plotType == :modelAOnly ? "$shortLabelA" : "$shortLabelB"
            titleLabel = "$cumLabel Consumption Response (%; $modelComparisonLabel)"
        end

        if decompType == :borrowersSavers
            nam = repeat(["Borrowers", "Savers", "Aggregate"], outer = 3)
        elseif decompType == :percentileNoAgg
            nam = repeat(string.(percList) .* "th Wealth Percentile", outer = 3)
        elseif decompType == :percentile
            nam = repeat(vcat(string.(percList) .* "th Wealth Percentile", "Aggregate"), outer = 3)
        end

        nam = "\\textrm{".* nam .* "}"

        # Initialize the plot
        p2 = plot(legend = :none, ylabel = titleLabel)
        ii = 1
        bar_width = 0.67

        # Plot bars
        groupedbar!(p2, nam, responseDiff',
            bar_width = bar_width,
            lw = 0.2,
            color = palette(:YlGnBu_5)[ii],
            label = ""
        )

        # Combine the bars for the aggregate since they are all the same anyways
        if prodType == :all && decompType == :percentile

            bar!(p2, [size(responseDiff,2)-0.5], [responseDiff[1, end]],
                bar_width = bar_width,
                lw = 0.2,
                color = palette(:YlGnBu_5)[ii],
                label = ""
            )

        end

        #
        hline!(p2, [0.0], linestyle = :dash, color = :black, label = "")
        if plotType == :comparison
            ylims!(p2, -1.5, 0.0)
        else
            ylims!(p2, -2.5, 0.0)
        end

        # Save the figure
        #filenameExt = "ProdType" * uppercase("$(prodType)"[1:1]) * "$(prodType)"[2:end]
        if !saveOnlyCombinedPlots
            savefig(p2, "$(outputFolder)/ConsumptionResponse$(filenameExt).pdf")
        end

        if plotType == :comparison
            modelComparisonLabel = plotAvsB ? "$shortLabelA - $shortLabelB" : "$shortLabelB - $shortLabelA"
        elseif plotType in [:modelAOnly, :modelBOnly]
            modelComparisonLabel = plotType == :modelAOnly ? "$shortLabelA" : "$shortLabelB"
        end        
        title!(p1, "(a) $(plotType == :comparison ? "Δ" : "") Income Response ($modelComparisonLabel)")
        ylabel!(p1, plotType == :comparison ? "pp" : "%")
        title!(p2, "(b) $(plotType == :comparison ? "Δ" : "") Consumption Response ($modelComparisonLabel)")
        ylabel!(p2, plotType == :comparison ? "pp" : "%")
        p = plot(p1, p2, layout = grid(1,2), size = (900, 300))
        display(p)
        savefig(p, "$(outputFolder)/IncomeAndConsumptionResponse$(filenameExt).pdf")


        # Additional figure
        if prodType == :all && !skipAdditionalFigure

            nam = repeat(["10th Wealth Percentile - 99th Wealth Percentile"], outer = 3)
            #nam = repeat(vcat("10th Wealth Percentile - 99th Wealth Percentile", "10th Wealth Percentile - Aggregate"), outer = 3)
            nam = "\\textrm{".* nam .* "}"

            #bb = hcat(responseDiff[:, 1] .- responseDiff[:, end-1], responseDiff[:, 1] .- responseDiff[:, end])'
            bb = (responseDiff[:, 1] .- responseDiff[:, end-1])'

            p3 = groupedbar(nam, bb,
                bar_width = bar_width,
                lw = 0.2,
                color = palette(:YlGnBu_5)[ii],
                label = ""
            )
            hline!(p3, [0.0], linestyle = :dash, color = :black, label = "")
            ylabel!(p3, "pp")
            if !saveOnlyCombinedPlots
                display(p3)
                savefig(p3, "$(outputFolderAdditional)/ConsumptionResponseComparison$(filenameExt).pdf")
            end

        end

    end

    nothing

end


"""
    plotPhaseDiagram(res, modelZLB, outputFolder)

Plot phase diagram for given model solution file.

"""
function plotPhaseDiagram(res, modelZLB, outputFolder; filename = "PhaseDiagramSimulation")

    # Make some variables more easily accesible
    P = res[modelZLB][:P]
    DSS = res[modelZLB][:DSS]
    SSS = res[modelZLB][:SSS]
    bPolicy = res[modelZLB][:bPolicy]
    πwALM = res[modelZLB][:πwALM]
    EπwCondALM = res[modelZLB][:EπwCondALM]

    # Phase diagram settings
    T = 11
    RGridSize = 11
    ζGridSize = 11
    RGrid = range(P.RMin, stop=P.RMax, length=RGridSize)
    ζGrid = range(P.ζMin, stop=P.ζMax, length=ζGridSize)


    # Initialize result matrices
    RStarPaths = NaN * zeros(T, RGridSize * ζGridSize)
    ζPaths = NaN * zeros(T, RGridSize * ζGridSize)
    ii = 1

    for i_ζ in 1:ζGridSize, i_R in 1:RGridSize

        # Skip all grid points except those on the boundary
        if !((i_ζ == 1 || i_ζ == ζGridSize) || (i_R == 1 || i_R == RGridSize))
            continue
        end

        # Simulate preference shock
        ζ = ζGrid[i_ζ] * ones(T)

        for tt in 2:T
            ζ[tt] = P.ζ̄ * (ζ[tt-1]/P.ζ̄)^P.ρ_ζ
        end

        # Simulate remaining series
        simSeries, _ = HANKWageRigidities.simulateAllSeries(P, DSS, bPolicy, πwALM, EπwCondALM;
                bCrossInit = SSS.bCross,
                RStarInit = RGrid[i_R],
                T = T,
                ζ = ζ,
                simulateAggregateShock = false)

        # Collect the simulated paths
        RStarPaths[:, ii] = simSeries[:RStar]
        ζPaths[:, ii] = simSeries[:ζ]

        ii += 1

    end

    # Phase diagram
    p = plot(log.(RStarPaths)*400, ζPaths, color = 1,
        label = "",
        xlabel = L"Nominal Rate $R_{t-1}$ (\%)",
        ylabel = L"Preference Shock $\xi_t$",
        legend = :topleft)

    # Add random (but fixed every for rerun) arrows
    ζPaths[rand(MersenneTwister(1234), 1:3, size(ζPaths)...) .== 1] .= NaN
    ζPaths[6:end, :] .= NaN

    # Remove line segments with single element
    for tt in 1:size(ζPaths,1), ii in 1:size(ζPaths,2)
        if tt == 1 && isnan(ζPaths[tt+1, ii])
            ζPaths[tt, ii] = NaN
        elseif tt == T && isnan(ζPaths[tt-1, ii])
            ζPaths[tt, ii] = NaN
        elseif tt != T && tt != 1 && isnan(ζPaths[tt+1, ii]) && isnan(ζPaths[tt-1, ii])
            ζPaths[tt, ii] = NaN
        end
    end

    #plot!(p, log.(RPaths)*400, ζPaths, color = 1, arrow = true, label = "")

    # Adjust plot limits
    xlims!(log(P.RMin)*400, log(P.RMax)*400)
    ylims!(P.ζMin, P.ζMax)

    # Add steady states
    scatter!([log(DSS[:RStar])*400], [DSS[:ζ]], label = "DSS", color = :green, markersize = 7)
    scatter!([log(SSS[:RStar])*400], [SSS[:ζ]], label = "SSS", color = :red, markersize = 7)

    # Save the figure
    savefig(p, "$(outputFolder)/$(filename).pdf")
    display(p)

    nothing

end


"""
    generateRateDecompositionTable(res, modelHANK, modelRANK, outputFolder, compName = "")

Generates table with rate decomposition for given solution files.

"""
function generateRateDecompositionTable(res, modelHANK, modelRANK, outputFolder, compName = "";
        filename = "RateDecomposition$(compName)", captionAddition = (compName != "" ? " (\$\\theta = " * string(res[modelHANK][:P].θ) * "\$)" : ""))

    # Number formatting
    numberFormat = "%2.2f"

    # Define the caption
    caption = "Rate Decomposition" * captionAddition

    # Define which interest rates are shown
    rates = [:r, :R, :π]
    labels = ["Real Rate", "Nominal Rate", "Inflation"]

    # Compute the required SSS and DSS values
    RANKDSS = [log(res[modelRANK][:DSS][rate])*400 for rate in rates]
    RANKSSS = [log(res[modelRANK][:SSS][rate])*400 for rate in rates]
    HANKDSS = [log(res[modelHANK][:DSS][rate])*400 for rate in rates]
    HANKSSS = [log(res[modelHANK][:SSS][rate])*400 for rate in rates]

    # Initialize vector containg each line of the table
    allLines = []

    # Add headings
    push!(allLines, "\\begin{table}[h]")
    push!(allLines, "\\centering")
    push!(allLines, "\\caption{$(caption)}")
    push!(allLines, "\\begin{tabular}{l*{$(length(rates))}{S[table-format=2.2]}}")
    push!(allLines, "\\toprule")

    push!(allLines, "  & {$(join(labels, "} & {"))} \\\\")
    push!(allLines, "\\midrule")

    push!(allLines, "RANK DSS & $(join(sprintf1.(numberFormat, RANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "RANK SSS & $(join(sprintf1.(numberFormat, RANKSSS), "\\% & "))\\% \\\\")
    push!(allLines, "\\midrule")
    push!(allLines, "(i) Deflationary Bias & $(join(sprintf1.(numberFormat, RANKDSS .- RANKSSS), "pp & "))pp \\\\")
    push!(allLines, "\\midrule\\\\")

    push!(allLines, "\\midrule")
    push!(allLines, "RANK DSS & $(join(sprintf1.(numberFormat, RANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "HANK DSS & $(join(sprintf1.(numberFormat, HANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "\\midrule")
    push!(allLines, "(ii) Prec. Savings (Idiosync. Risk) & $(join(sprintf1.(numberFormat, RANKDSS .- HANKDSS), "pp & "))pp \\\\")
    push!(allLines, "\\midrule\\\\")

    push!(allLines, "\\midrule")
    push!(allLines, "RANK DSS & $(join(sprintf1.(numberFormat, RANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "HANK SSS & $(join(sprintf1.(numberFormat, HANKSSS), "\\% & "))\\% \\\\")
    push!(allLines, "\\midrule")
    push!(allLines, "(iii) Total & $(join(sprintf1.(numberFormat, RANKDSS .- HANKSSS), "pp & "))pp \\\\")
    push!(allLines, "\\midrule\\\\")

    push!(allLines, "\\midrule")
    push!(allLines, "(iii) - (ii) - (i) Prec. Savings (Agg. Risk) & $(join(sprintf1.(numberFormat, RANKDSS .- HANKSSS .- (RANKDSS .- RANKSSS) .- (RANKDSS .- HANKDSS)), "pp & "))pp \\\\")

    # Finish the table
    push!(allLines, "\\bottomrule")
    push!(allLines, "\\end{tabular}")
    push!(allLines, "\\end{table}")
    display(allLines)

    # Write the table to a file
    open("$(outputFolder)/$(filename).tex", "w") do f
        for line in allLines
            println(f, line)
        end
    end

end


"""
    generateSmallRateDecompositionTable(res, modelHANK, modelRANK, outputFolder)

Generates table with rate decomposition (without precautionary savings due to aggregate risk) for given solution files.

"""
function generateSmallRateDecompositionTable(res, modelHANK, modelRANK, outputFolder;
        filename = "RateDecompositionSmall")

    # Number formatting
    numberFormat = "%2.2f"

    # Define the caption
    caption = "Decomposition Exercise."

    # Define which interest rates are shown
    rates = [:r, :R, :π]
    labels = ["Real Rate", "Nominal Rate", "Inflation"]

    # Compute the required SSS and DSS values
    RANKDSS = [log(res[modelRANK][:DSS][rate])*400 for rate in rates]
    RANKSSS = [log(res[modelRANK][:SSS][rate])*400 for rate in rates]
    HANKDSS = [log(res[modelHANK][:DSS][rate])*400 for rate in rates]
    HANKSSS = [log(res[modelHANK][:SSS][rate])*400 for rate in rates]

    # Initialize vector containg each line of the table
    allLines = []

    # Add headings
    push!(allLines, "\\begin{table}[h]")
    push!(allLines, "\\centering")
    push!(allLines, "\\caption{$(caption)}")
    push!(allLines, "\\begin{tabular}{l*{$(length(rates))}{S[table-format=2.2]}}")
    push!(allLines, "\\toprule")

    push!(allLines, "  & {$(join(labels, "} & {"))} \\\\")
    push!(allLines, "\\midrule")

    push!(allLines, "ZLB-RANK DSS & $(join(sprintf1.(numberFormat, RANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "ZLB-HANK SSS & $(join(sprintf1.(numberFormat, HANKSSS), "\\% & "))\\% \\\\")
    push!(allLines, "\\midrule")
    push!(allLines, "(i) Total & $(join(sprintf1.(numberFormat, RANKDSS .- HANKSSS), "pp & "))pp \\\\")
    push!(allLines, "\\midrule\\\\")

    push!(allLines, "\\midrule")
    push!(allLines, "ZLB-RANK DSS & $(join(sprintf1.(numberFormat, RANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "ZLB-HANK DSS & $(join(sprintf1.(numberFormat, HANKDSS), "\\% & "))\\% \\\\")
    push!(allLines, "\\midrule")
    push!(allLines, "(ii) Precautionary Savings & $(join(sprintf1.(numberFormat, RANKDSS .- HANKDSS), "pp & "))pp \\\\")
    push!(allLines, "\\phantom{(ii)} Idiosyncratic Risk & $(repeat("& ",length(rates)-1)) \\\\")
    push!(allLines, "\\midrule\\\\")

    push!(allLines, "\\midrule")
    push!(allLines, "(i)-(ii) Deflationary Bias & $(join(sprintf1.(numberFormat, HANKDSS .- HANKSSS), "pp & "))pp \\\\")

    # Finish the table
    push!(allLines, "\\bottomrule")
    push!(allLines, "\\end{tabular}")
    push!(allLines, "\\end{table}")
    display(allLines)

    # Write the table to a file
    open("$(outputFolder)/$(filename).tex", "w") do f
        for line in allLines
            println(f, line)
        end
    end

end


"""
    plotPolicyFunctionComparison(res, modelZLB, modelNoZLB, outputFolder)

Plots comparison of policy functions for given solution files.

"""
function plotPolicyFunctionComparison(res, modelZLB, modelNoZLB, outputFolder;
    labelZLB = "ZLB", labelNoZLB = "No ZLB", filename = "PolicyFunctionComparison")

    local p

    for (ii, model) in enumerate([modelZLB, modelNoZLB])
        
        # Extract important equilibrium objects
        P = res[model][:P]
        DSS = res[model][:DSS]
        SSS = res[model][:SSS]
        bPolicy = res[model][:bPolicy]
        πwALM = res[model][:πwALM]
        EπwCondALM = res[model][:EπwCondALM]
        πwALMInterpol = res[model][:πwALMInterpol]
        EπwCondALMInterpol = res[model][:EπwCondALMInterpol]
        bPolicyInterpol = res[model][:bPolicyInterpol]
        
        # Define the shocks that are compared
        ζ_small = exp(-1 * res[model][:P].σ̃_ζ)
        ζ_large = exp(1 * res[model][:P].σ̃_ζ)
        ζ_extralarge = exp(-3.0 * res[model][:P].σ̃_ζ)
        RStar = exp(0.1/400) # SSS.RStar
        
        println("R(R_SSS, ζ_SSS) = ", log(SSS.R)*400)
        println("R(RStar, ζ_SSS) (alt. computation) = ", log(HANKWageRigidities.computeAllAggregateVariables(P, DSS, RStar, SSS.ζ, 
            SSS.bCross, πwALMInterpol, EπwCondALMInterpol).R) * 400)
        println("R(RStar, ζ_small) = ", log(HANKWageRigidities.computeAllAggregateVariables(P, DSS, RStar, ζ_small, 
            SSS.bCross, πwALMInterpol, EπwCondALMInterpol).R) * 400)
        println("R(RStar, ζ_large) = ", log(HANKWageRigidities.computeAllAggregateVariables(P, DSS, RStar, ζ_large, 
            SSS.bCross, πwALMInterpol, EπwCondALMInterpol).R) * 400)
        println("R(RStar, ζ_extralarge) = ", log(HANKWageRigidities.computeAllAggregateVariables(P, DSS, RStar, ζ_extralarge, 
            SSS.bCross, πwALMInterpol, EπwCondALMInterpol).R) * 400)
        
        # Create plot
        pAll = []
        titles =  ["Low Labor Productivity" "Medium Labor Productivity" "High Labor Productivity"]
        colors = palette(:Paired_10)[2:2:end]'
        
        for i_s in 1:P.sGridSize
            
            #  legend = (at = (0.5, -0.10), anchor = "north"), extra_kwargs=:subplot, legend_columns=-1
            p0 = plot(res[model][:P].bDenseGrid, bPolicyInterpol.(P.bDenseGrid, SSS.RStar, i_s, SSS.ζ) .- res[model][:P].bDenseGrid, 
                label = L"R_{t-1} = R_{SSS},\, \xi_t = \xi_{SSS}\;", title = titles[i_s], 
                linewidth = 2,
                color = colors[1],
                legend = (i_s == 2) ? (at = (0.5, -0.25), anchor = "north") : :none,
                extra_kwargs = :subplot, 
                legend_columns = -1, 
                foreground_color_legend = nothing,
                background_color_legend = nothing)
            plot!(p0, res[model][:P].bDenseGrid, bPolicyInterpol.(P.bDenseGrid, RStar, i_s, ζ_small) .- res[model][:P].bDenseGrid, 
                label = L"R_{t-1} = 0.1\%,\, \xi_t = \xi_{low}\;",
                linewidth = 2,
                color = colors[2],
                extra_kwargs = :subplot, 
                legend_columns = -1)
            plot!(p0, res[model][:P].bDenseGrid, bPolicyInterpol.(P.bDenseGrid, RStar, i_s, ζ_large) .- res[model][:P].bDenseGrid, 
                label = L"R_{t-1} = 0.1\%,\, \xi_t = \xi_{high}\;",
                linewidth = 2,
                color = colors[3],
                extra_kwargs = :subplot, 
                legend_columns = -1)
            #=plot!(p0, res[model][:P].bDenseGrid, bPolicyInterpol.(P.bDenseGrid, RStar, i_s, ζ_extralarge) .- res[model][:P].bDenseGrid, 
                label = L"R_{t-1} = 0.1\%,\, \xi_t = \xi_{extra low}\;",
                linewidth = 2,
                color = colors[4],
                extra_kwargs = :subplot, 
                legend_columns = -1)=#
            hline!([0.0], color = :black, linestyle = :dot, label = "", extra_kwargs = :subplot, legend_columns = -1)
            xlabel!(L"Wealth $b_{t-1}$")
            ylabel!(L"Savings $b_t - b_{t-1}$")
            xlims!(P.bMin*1.5, 15.0)
            #ylims!(-0.15, 0.075)
            
            push!(pAll, p0)

        end

        p = plot(pAll..., layout = grid(1, 3), size = (1080, 240))
        savefig(p, "$(outputFolder)/$(filename)$(ii == 1 ? labelZLB : replace(labelNoZLB, " " => "")).pdf")

    end

    # Same figure as before as difference betwen model 
    modelA = modelZLB
    modelB = modelNoZLB

    # Define the shocks that are compared
    ζ_small = exp(-1 * res[modelA][:P].σ̃_ζ)
    ζ_large = exp(1 * res[modelA][:P].σ̃_ζ)
    ζ_extralarge = exp(-3.0 * res[modelA][:P].σ̃_ζ)
    RStar = exp(0.1/400) # SSS.RStar

    # Create plot
    pAll = []
    titles =  ["Low Labor Productivity" "Medium Labor Productivity" "High Labor Productivity"]
    colors = palette(:Paired_10)[2:2:end]'
    
    for i_s in 1:res[modelA][:P].sGridSize
        
        # Compute differences in policy functions
        bPolicyDiffSSS = res[modelA][:bPolicyInterpol].(res[modelA][:P].bDenseGrid, res[modelA][:SSS].RStar, i_s, res[modelA][:SSS].ζ) - 
                         res[modelB][:bPolicyInterpol].(res[modelB][:P].bDenseGrid, res[modelB][:SSS].RStar, i_s, res[modelB][:SSS].ζ)

        bPolicyDiffSmall = res[modelA][:bPolicyInterpol].(res[modelA][:P].bDenseGrid, RStar, i_s, ζ_small) - 
                           res[modelB][:bPolicyInterpol].(res[modelB][:P].bDenseGrid, RStar, i_s, ζ_small)

        bPolicyDiffLarge = res[modelA][:bPolicyInterpol].(res[modelA][:P].bDenseGrid, RStar, i_s, ζ_large) - 
                           res[modelB][:bPolicyInterpol].(res[modelB][:P].bDenseGrid, RStar, i_s, ζ_large)
        
        # 
        p0 = plot(res[modelA][:P].bDenseGrid, bPolicyDiffSSS, 
            label = L"R_{t-1} = R_{SSS},\, \xi_t = \xi_{SSS}\;", title = titles[i_s], 
            linewidth = 2,
            color = colors[1],
            legend = (i_s == 2) ? (at = (0.5, -0.25), anchor = "north") : :none,
            extra_kwargs = :subplot, 
            legend_columns = -1, 
            foreground_color_legend = nothing,
            background_color_legend = nothing)
        plot!(p0, res[modelA][:P].bDenseGrid, bPolicyDiffSmall, 
            label = L"R_{t-1} = 0.1\%,\, \xi_t = \xi_{low}\;",
            linewidth = 2,
            color = colors[2],
            extra_kwargs = :subplot, 
            legend_columns = -1)
        plot!(p0, res[modelA][:P].bDenseGrid, bPolicyDiffLarge, 
            label = L"R_{t-1} = 0.1\%,\, \xi_t = \xi_{high}\;",
            linewidth = 2,
            color = colors[3],
            extra_kwargs = :subplot, 
            legend_columns = -1)  
        hline!([0.0], color = :black, linestyle = :dot, label = "", extra_kwargs = :subplot, legend_columns = -1)
        xlabel!(L"Wealth $b_{t-1}$")
        ylabel!(L"$\Delta$Savings ($b_t^{%$labelZLB} - b_t^{%$labelNoZLB}$)")
        xlims!(res[modelA][:P].bMin*1.5, 15.0)
        
        push!(pAll, p0)

    end

    p = plot(pAll..., layout = grid(1, 3), size = (1080, 240), margin = 5Plots.PlotMeasures.mm,)
    savefig(p, "$(outputFolder)/$(filename)$(labelZLB)minus$(replace(labelNoZLB, " " => "")).pdf")

    nothing

end


"""
    computeComparativeStatics(res, modelsCompStat)

Computes comparative statics for given solution files.

"""
function computeComparativeStatics(res, modelsCompStat)
    
    # Generate vectors to be plotted
    πDSS = [log(res[model][:DSS][:π])*400 for model in modelsCompStat]
    πSSS = [log(res[model][:SSS][:π])*400 for model in modelsCompStat]

    rDSS = [log(res[model][:DSS][:r])*400 for model in modelsCompStat]
    rSSS = [log(res[model][:SSS][:r])*400 for model in modelsCompStat]

    RDSS = [log(res[model][:DSS][:R])*400 for model in modelsCompStat]
    RSSS = [log(res[model][:SSS][:R])*400 for model in modelsCompStat]

    freqZLB = [sum(res[model][:simSeries][:RStar] .<= 1.0) / length(res[model][:simSeries][:RStar]) for model in modelsCompStat]

    return πDSS, πSSS, rDSS, rSSS, RDSS, RSSS, freqZLB

end


"""
    plotComparativeStatics(res, modelsCompStat, σLabels, outputFolder)

Plots comparative statics for given solution files.

"""
function plotComparativeStatics(res, modelsCompStat, σLabels, outputFolder; filename = "ComparativeStatics")

    # Compute comparative statics
    πDSS, πSSS, rDSS, rSSS, RDSS, RSSS, freqZLB = computeComparativeStatics(res, modelsCompStat)

    # Plot the results
    sel = [1, 2, 3]
    labels = [σLabels[1] σLabels[2] σLabels[3]]
    p1 = plot(πDSS[:, sel], πSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Inflation ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)

    p2 = plot(πDSS[:, sel], RSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Nominal Rate ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)

    p3 = plot(πDSS[:, sel], rSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Real Rate ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)

    p4 = plot(πDSS[:, sel], 100*freqZLB[:, sel],
        label = labels,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"ZLB Frequency ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :topright)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p = plot(p1, p2, p3, p4, size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename).pdf")

    # Plot the results
    sel = [1, 2]
    colorSSS = palette(:Paired_10)[2:2:end]'
    colorDSS = palette(:Paired_10)[1:2:end]'
    labels = [σLabels[1] σLabels[2]]
    p1 = plot(πDSS[:, sel], πSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Inflation ($\% $)",
        color = colorSSS,
        linewidth = 2,
        legend = :none)

    plot!(πDSS[:, sel], πDSS[:, sel],
        linestyle = :longdash,
        color = colorDSS,
        linewidth = 2)

    p2 = plot(πDSS[:, sel], RSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Nominal Rate ($\% $)",
        color = colorSSS,
        linewidth = 2,
        legend = :none)

    plot!(πDSS[:, sel], RDSS[:, sel],
        linestyle = :longdash,
        color = colorDSS,
        linewidth = 2)

    p3 = plot(πDSS[:, sel], rSSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"Real Rate ($\% $)",
        color = colorSSS,
        linewidth = 2,
        legend = :none)

    plot!(πDSS[:, sel], rDSS[:, sel],
        linestyle = :longdash,
        color = colorDSS,
        linewidth = 2)

    p4 = plot(πDSS[:, sel], 100*freqZLB[:, sel],
        label = length(labels) > 1 ? labels .* " (SSS)" : "SSS",
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"ZLB Frequency ($\% $)",
        color = colorSSS,
        linewidth = 2,
        legend = :topright)

    plot!(πDSS[1, sel]', freqZLB[1, sel]',
        label = length(labels) > 1 ? labels .* " (DSS)" : "DSS",
        linestyle = :longdash,
        color = colorDSS)

    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p = plot(p1, p2, p3, p4, size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename)Baseline.pdf")

    # Plot the results
    sel = [1, 2]
    color = palette(:Paired_10)[2:2:end]'
    linestyle = [:solid :longdash]
    labels = [σLabels[1] σLabels[2]]
    p1 = plot(πDSS[:, sel], πSSS[:, sel] - πDSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Inflation Bias (pp)",
        color = color,
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p2 = plot(πDSS[:, sel], RSSS[:, sel] - RDSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Nominal Rate Bias (pp)",
        color = color,
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p3 = plot(πDSS[:, sel], rSSS[:, sel] - rDSS[:, sel],
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Real Rate Bias (pp)",
        color = color,
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p4 = plot(πDSS[:, sel], 100*freqZLB[:, sel],
        label = labels,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"ZLB Frequency ($\% $)",
        color = color,
        linestyle = linestyle,
        linewidth = 2,
        legend = :topright)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p = plot(p1, p2, p3, p4, size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename)BiasBaseline.pdf")

    nothing

end


"""
    plotComparativeStaticsAdditional(res, modelsCompStat, σLabels, outputFolder)

Plots comparative statics for given solution files.

"""
function plotComparativeStaticsAdditional(res, modelsCompStat, σLabels, outputFolder; filename = "ComparativeStatics")

    # Compute comparative statics
    πDSS, πSSS, rDSS, rSSS, RDSS, RSSS, freqZLB = computeComparativeStatics(res, modelsCompStat)

    # Plot the results
    p1 = plot(πDSS, πSSS - πDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Inflation Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p2 = plot(πDSS, RSSS - RDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Nominal Rate Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p3 = plot(πDSS, rSSS - rDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Real Rate Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p4 = plot(πDSS, 100*freqZLB,
        label = σLabels,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"ZLB Frequency ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linewidth = 2,
        legend = :topright)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p = plot(p1, p2, p3, p4, size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename)Bias.pdf")

    # Plot the results
    linestyle = [:solid :longdash :longdashdot :extralongdash]
    p1 = plot(πDSS, πSSS - πDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Inflation Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p2 = plot(πDSS, RSSS - RDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Nominal Rate Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p3 = plot(πDSS, rSSS - rDSS,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = "Real Rate Bias (pp)",
        color = palette(:Paired_10)[2:2:end]',
        linestyle = linestyle,
        linewidth = 2,
        legend = :none)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p4 = plot(πDSS, 100*freqZLB,
        label = σLabels,
        xlabel = L"Inflation Target ($\% $)",
        ylabel = L"ZLB Frequency ($\% $)",
        color = palette(:Paired_10)[2:2:end]',
        linestyle = linestyle,
        linewidth = 2,
        legend = :topright)
    hline!([0.0], color = :black, linestyle = :dot, label = "")

    p = plot(p1, p2, p3, p4, size = (900, 600))
    display(p)
    savefig(p, "$(outputFolder)/$(filename)BiasAlt.pdf")

    nothing

end


"""
    generateComparativeStaticsTable(res, modelZLB, modelsCompStat, modelsCompStatPlus, σLabels, πTargets, outputFolder)

Generates table with comparative statics for given solution files.

"""
function generateComparativeStaticsTable(res, modelZLB, modelsCompStat, modelsCompStatPlus, σLabels, πTargets, outputFolder;
        filename = "ComparativeStatics")

    # Compute comparative statics
    πDSS, πSSS, rDSS, rSSS, RDSS, RSSS, freqZLB = computeComparativeStatics(res, modelsCompStat)

    # Add higher accuracy solution to comparative statics for table
    idxHighAcc = findfirst(πTargets .== 2.0 / 100)
    rSSSPlus = hcat(rSSS, fill(NaN, length(πTargets), 1))
    rSSSPlus[idxHighAcc, end] = log(res[modelZLB][:SSS][:r])*400 
    rDSSPlus = hcat(rDSS, fill(NaN, length(πTargets), 1))
    rDSSPlus[idxHighAcc, end] = log(res[modelZLB][:DSS][:r])*400 
    
    πSSSPlus = hcat(πSSS, fill(NaN, length(πTargets), 1))
    πSSSPlus[idxHighAcc, end] = log(res[modelZLB][:SSS][:π])*400 
    πDSSPlus = hcat(πDSS, fill(NaN, length(πTargets), 1))
    πDSSPlus[idxHighAcc, end] = log(res[modelZLB][:DSS][:π])*400 
    
    RSSSPlus = hcat(RSSS, fill(NaN, length(πTargets), 1))
    RSSSPlus[idxHighAcc, end] = log(res[modelZLB][:SSS][:R])*400 
    RDSSPlus = hcat(RDSS, fill(NaN, length(πTargets), 1))
    RDSSPlus[idxHighAcc, end] = log(res[modelZLB][:DSS][:R])*400 

    freqZLBPlus = hcat(freqZLB, fill(NaN, length(πTargets), 1))
    freqZLBPlus[idxHighAcc, end] = sum(res[modelZLB][:simSeries][:RStar] .<= 1.0) / length(res[modelZLB][:simSeries][:RStar])

    # Add RANK without ZLB to comparative statics for table
    rSSSPlus = hcat(rSSSPlus, [log(res[model][:SSS][:r])*400 for model in modelsCompStatPlus])
    rDSSPlus = hcat(rDSSPlus, [log(res[model][:DSS][:r])*400 for model in modelsCompStatPlus])

    πSSSPlus = hcat(πSSSPlus, [log(res[model][:SSS][:π])*400 for model in modelsCompStatPlus])
    πDSSPlus = hcat(πDSSPlus, [log(res[model][:DSS][:π])*400 for model in modelsCompStatPlus]) 

    RSSSPlus = hcat(RSSSPlus, [log(res[model][:SSS][:R])*400 for model in modelsCompStatPlus])
    RDSSPlus = hcat(RDSSPlus, [log(res[model][:DSS][:R])*400 for model in modelsCompStatPlus])

    freqZLBPlus = hcat(freqZLBPlus, [sum(res[model][:simSeries][:RStar] .<= 1.0) / length(res[model][:simSeries][:RStar]) for model in modelsCompStatPlus])

    # Model labels for table
    modelLabels = hcat(σLabels, "ZLB-HANK (Higher Accuracy)", "RANK")

    # Settings
    numberFormat = "%2.2f"
    addGroupLines = true

    # Define table caption
    caption = "Comparative Statics"

    #
    allLines = []

    # Add headings
    push!(allLines, "\\begin{table}[h]")
    push!(allLines, "\\centering")
    push!(allLines, "\\footnotesize")
    push!(allLines, "\\caption{$(caption)}")
    push!(allLines, "\\begin{tabular}{l*{$(size(modelsCompStat,1))}{S[table-format=2.2]}}")
    push!(allLines, "\\toprule")
    push!(allLines, " & \\multicolumn{$(size(modelsCompStat,1))}{c}{Inflation Target (\\%)}\\\\")
    push!(allLines, "\\cmidrule{2-$(1+size(modelsCompStat,1))}")

    # Inflation target levels
    currentLine = ""
    for (jj, π) in enumerate(πTargets)
        currentLine *= " & $(sprintf1(numberFormat, 100*π))"
    end
    currentLine *= "\\\\"
    push!(allLines, currentLine)
    
    # 
    for (steadyStateTitle, steadyStateVarsAndLabels) in [("Stochastic Steady State (SSS)", [(rSSSPlus, "(Ex-Post) Real Rate (\\%)"), (πSSSPlus, "Inflation Rate (\\%)"), (RSSSPlus, "Nominal Rate (\\%)"), (100*freqZLBPlus, "ZLB Frequency (\\%)")]),
                                                        ("Deterministic Steady State (DSS)", [(rDSSPlus, "(Ex-Post) Real Rate (\\%)"), (πDSSPlus, "Inflation Rate (\\%)"), (RDSSPlus, "Nominal Rate (\\%)")])]

        push!(allLines, "\\\\")
        push!(allLines, "\\multicolumn{$(1+size(modelsCompStat,1))}{c}{$(steadyStateTitle)}\\\\")

        for (currentVar, currentLabel) in steadyStateVarsAndLabels

            push!(allLines, "\\midrule")
            push!(allLines, "\\multicolumn{$(1+size(modelsCompStat,1))}{c}{$(currentLabel)}\\\\")
            push!(allLines, "\\midrule")

            for ii in eachindex(modelLabels)

                currentLine = modelLabels[ii]

                for (jj, π) in enumerate(πTargets)
                    currentLine *= " & $(isnan(currentVar[jj, ii]) ? "{-}" : sprintf1(numberFormat, currentVar[jj, ii]))"
                end

                currentLine *= "\\\\"
                push!(allLines, currentLine)

            end

        end
        
    end

    # Finish the table
    push!(allLines, "\\bottomrule")
    push!(allLines, "\\end{tabular}")
    push!(allLines, "\\end{table}")
    display(allLines)

    # Write the table to a file
    open("$(outputFolder)/$(filename).tex", "w") do f
        for line in allLines
            println(f, line)
        end
    end

    nothing

end


"""
    computeAdditionalStatistics(res, model)

Computes additional statistics for given solution files.

"""
function computeAdditionalStatistics(res, model)

    # Settings
    percList = [10, 90, 99]

    # Extract required variables
    P = res[model][:P]
    outputSeries = res[model][:simSeriesPlus][:Y]
    bCrossSeries = res[model][:simSeriesPlus][:bCross]
    RStarSeries = res[model][:simSeriesPlus][:RStar]
    πSeries = res[model][:simSeriesPlus][:π]
    wSeries = res[model][:simSeriesPlus][:w]
    HSeries = res[model][:simSeriesPlus][:H]
    TSeries = res[model][:simSeriesPlus][:T]

    # Compute correlations
    individualIncomeSeries = zeros(size(outputSeries, 1), length(percList), P.sGridSize)
    individualIncomeSeriesFullDist = zeros(size(outputSeries, 1), P.bDenseGridSize, P.sGridSize)
    sumIncomeBottomWealth90 = zeros(size(outputSeries, 1))
    sumIncomeTopWealth10 = zeros(size(outputSeries, 1))
    averageIncomeBottomWealth90 = zeros(size(outputSeries, 1))
    averageIncomeTopWealth10 = zeros(size(outputSeries, 1))

    for tt in 2:size(outputSeries, 1)

        # Compute the percentile
        bSet = [HANKWageRigidities.computePercentile(vec(sum(bCrossSeries[:, :, tt], dims = 2)), P.bDenseGrid, perc / 100) for perc in percList]

        # Compute individual income series
        for i_s in 1:P.sGridSize, i_perc in 1:length(percList)
            individualIncomeSeries[tt, i_perc, i_s] = HANKWageRigidities.computeIndividualCashOnHand(P, bSet[i_perc], P.sGrid[i_s], RStarSeries[tt-1], πSeries[tt], wSeries[tt], HSeries[tt], TSeries[tt]) - bSet[i_perc]
        end

        # Compute additional individual income series for full distribution
        for i_s in 1:P.sGridSize, i_b in 1:P.bDenseGridSize
            individualIncomeSeriesFullDist[tt, i_b, i_s] = HANKWageRigidities.computeIndividualCashOnHand(P, P.bDenseGrid[i_b], P.sGrid[i_s], RStarSeries[tt-1], πSeries[tt], wSeries[tt], HSeries[tt], TSeries[tt]) - P.bDenseGrid[i_b]
        end

        bottomWealth90Ind = (P.bDenseGrid .< bSet[first(findall(x -> x==90, percList))])

        sumIncomeBottomWealth90[tt] = sum(individualIncomeSeriesFullDist[tt, bottomWealth90Ind, :] .* bCrossSeries[bottomWealth90Ind, :, tt])
        sumIncomeTopWealth10[tt] = sum(individualIncomeSeriesFullDist[tt, .!bottomWealth90Ind, :] .* bCrossSeries[.!bottomWealth90Ind, :, tt])
        averageIncomeBottomWealth90[tt] = sumIncomeBottomWealth90[tt] / sum(bCrossSeries[bottomWealth90Ind, :, tt])
        averageIncomeTopWealth10[tt] = sumIncomeTopWealth10[tt] / sum(bCrossSeries[.!bottomWealth90Ind, :, tt])

    end

    # Remove nan values (arise if low percentiles cannot be computed)
    nanInd = dropdims(reduce(|, isnan.(individualIncomeSeries), dims = (2,3)), dims = (2,3))

    if sum(nanInd) > 0
        @warn "NaN values in individualIncomeSeries will be ignored for computing correlations."
    end

    individualIncomeSeriesWithoutNaN = individualIncomeSeries[.!nanInd, :, :]
    outputSeriesWithoutNaN = outputSeries[.!nanInd]

    # Show correlations
    println(model)
    println("--------------------------------------------------------------------------------")
    println("Correlations with Output:")

    for i_perc in 1:length(percList)

        println("\nIncome of $(percList[i_perc])th percentile of wealth distribution")

        for i_s in 1:P.sGridSize
            println("i_s = $i_s: $(HANKWageRigidities.cor(individualIncomeSeriesWithoutNaN[2:end, i_perc, i_s], outputSeriesWithoutNaN[2:end]))")
        end

    end

    println("\nAverage income (Bottom 90%): $(HANKWageRigidities.cor(averageIncomeBottomWealth90[2:end], outputSeries[2:end]))")
    println("Sum of income (Bottom 90%): $(HANKWageRigidities.cor(sumIncomeBottomWealth90[2:end], outputSeries[2:end]))")
    println("Average income (Top 10%): $(HANKWageRigidities.cor(averageIncomeTopWealth10[2:end], outputSeries[2:end]))")
    println("Sum of income (Top 10%): $(HANKWageRigidities.cor(sumIncomeTopWealth10[2:end], outputSeries[2:end]))")

    println("--------------------------------------------------------------------------------")

    nothing

end


# Extension to Plots to adjust the legend positions
Plots.convertLegendValue(v::NamedTuple) = v


"""
    Plots.pgfx_get_linestyle(k::Symbol) 

This overwrites the line options for the pgfplotsx backend of Plots.jl and adds a longdash option. Note that this might
fail if Plots.jl is updated.

"""
Plots.pgfx_get_linestyle(k::Symbol) = get(
    (
        solid = "solid",
        dash = "dashed",
        dot = "dotted",
        dashdot = "dashdotted",
        dashdotdot = "dashdotdotted",
        longdash = "dashed, dash_pattern = on 8pt off 2pt",
        extralongdash = "dashed, dash_pattern = on 12pt off 2pt",
        longdashdot = "dash pattern = on 8pt off 2pt on 2pt off 2pt"
    ),
    k,
    "solid",
)

# Disables warnings (added to silence warninges related to new linestyles)
Plots.should_warn_on_unsupported(::Plots.PGFPlotsXBackend) = false


function computeForecastErrorsInflationPLM(πw, RStar, ζ, πwALMInterpol; rescaleErrors = false)

    # Compute the errors implied by the NN
    errors = zeros(length(πw)-1)
    for tt in 2:length(πw)
        errors[tt-1] = πwALMInterpol(RStar[tt-1], ζ[tt]) - πw[tt]
    end

    # Compute R^2 and MSE
    π̄w = mean(πw[2:end])
    R2 = 1 - sum(errors.^2) / sum((πw[2:end] .- π̄w).^2)
    MSE = mean(errors.^2)
    stats = (MSE = MSE, R2 = R2)

    # Convert the errors to (annual) percentage points
    if rescaleErrors
        @. errors = errors * 400
    end

    return errors, stats

end


function computeForecastErrorsInflationPLMZLBOnly(πw, RStar, ζ, πwALMInterpol, ZLBCheck; rescaleErrors = false)

    # Compute errors for all periods
    errors, _ = computeForecastErrorsInflationPLM(πw, RStar, ζ, πwALMInterpol)

    # Select ZLB Periods
    errors = errors[ZLBCheck .== 1]
    πw = πw[2:end]
    πw = πw[ZLBCheck .== 1]

    # Compute R^2 and MSE
    π̄w = mean(πw)
    R2 = 1 - sum(errors.^2) / sum((πw .- π̄w).^2)
    MSE = mean(errors.^2)
    stats = (MSE = MSE, R2 = R2)

    # Convert the errors to (annual) percentage points
    if rescaleErrors
        @. errors = errors * 400
    end

    return errors, stats

end


function computeForecastErrorsInflationExpectationPLM(EπwCond, RStar, ζ, EπwCondALMInterpol; rescaleErrors = false)

    # Compute the errors implied by the NN
    errors = zeros(length(EπwCond)-2)
    for tt in 2:length(EπwCond)-1
        errors[tt-1] = EπwCondALMInterpol(RStar[tt-1], ζ[tt], ζ[tt+1]) - EπwCond[tt]
    end

    # Compute R^2 and MSE
    Eπ̄ = mean(EπwCond[2:end-1])
    R2 = 1 - sum(errors.^2) / sum((EπwCond[2:end-1] .- Eπ̄).^2)
    MSE = mean(errors.^2)
    stats = (MSE = MSE, R2 = R2)

    # Convert the errors to (annual) percentage points
    if rescaleErrors
        @. errors = errors * 400
    end

    return errors, stats

end


function computeForecastErrorsInflationExpectationPLMZLBOnly(EπwCond, RStar, ζ, EπwCondALMInterpol, ZLBCheck; rescaleErrors = false)

    # Compute errors for all periods
    errors, _ = computeForecastErrorsInflationExpectationPLM(EπwCond, RStar, ζ, EπwCondALMInterpol)

    # Select ZLB Periods
    errors = errors[ZLBCheck .== 1]
    EπwCond = EπwCond[2:end-1]
    EπwCond = EπwCond[ZLBCheck .== 1]

    # Compute R^2 and MSE
    π̄ = mean(EπwCond)
    R2 = 1 - sum(errors.^2) / sum((EπwCond .- π̄).^2)
    MSE = mean(errors.^2)
    stats = (MSE = MSE, R2 = R2)

    # Convert the errors to (annual) percentage points
    if rescaleErrors
        @. errors = errors * 400
    end

    return errors, stats

end


function plotIRFComparison(IRFs, SSSs, IRFLabels, IRFStyles, variables, variableNames, interestRateList, period, linewidth; levelDifferencList = [])

    pAll = Any[]

    for (ii, (IRF, SSS, IRFStyle)) in enumerate(zip(IRFs, SSSs, IRFStyles))

        if ii == 1
            for (var, name) in zip(variables, variableNames)
                isInterestRate = (var in interestRateList)
                inLevelDifference = (var in levelDifferencList)
                if var ∉ keys(SSS) || var ∉ keys(IRF)
                    @warn "IRF ii = $(ii): No IRF found for $(var)."
                    p = plotIRF(NaN.*collect(period), NaN, name; isInterestRate, inLevelDifference, inDeviations = true, linewidth, IRFStyle...)
                else
                    p = plotIRF(IRF[var][period], SSS[var], name; isInterestRate, inLevelDifference, inDeviations = true, linewidth, IRFStyle...)
                end
                push!(pAll, p)
            end
        else
            jj = 1
            for (var, name) in zip(variables, variableNames)
                isInterestRate = (var in interestRateList)
                inLevelDifference = (var in levelDifferencList)
                if var ∉ keys(SSS) || var ∉ keys(IRF)
                    @warn "IRF ii = $(ii): No IRF found for $(var)."
                    plotIRF!(pAll[jj], NaN.*collect(period), NaN; isInterestRate, inLevelDifference, inDeviations = true, linewidth, IRFStyle...)
                else
                    plotIRF!(pAll[jj], IRF[var][period], SSS[var]; isInterestRate, inLevelDifference, inDeviations = true, linewidth, IRFStyle...)
                end
                jj += 1
            end
        end

    end

    plot!(pAll[1], legend = :bottomright)
    for (ii, label) in enumerate(IRFLabels)
        jj = (ii==1) ? ii : ii+1 # Skip jj = 2, which corresponds to the zero line
        pAll[1].series_list[jj][:label] = label
    end
    pAll[1].series_list[2][:label] = ""

    return pAll

end


function plotIRF(IRF, SSS, label; isInterestRate = false, inLevelDifference = false, inDeviations = true, color = :default, linestyle = :default, linewidth = 2, xlabel = "Quarter")

    # Transform the series
    if isInterestRate && inDeviations && !inLevelDifference
        IRF = log.(IRF)*400 .- log(SSS)*400
        hlineLevel = 0
    elseif isInterestRate && inDeviations && inLevelDifference
        IRF = (IRF .- SSS) * 100
        hlineLevel = 0
    elseif isInterestRate
        IRF = log.(IRF)*400
        hlineLevel = log.(SSS)*400
    elseif inDeviations && !inLevelDifference
        IRF = (IRF / SSS .- 1) * 100
        hlineLevel = 0
    elseif inDeviations && inLevelDifference
        IRF = IRF .- SSS
        hlineLevel = 0
    else
        hlineLevel = SSS
    end

    # Plot the IRFs and horizontal line at SSS or at zero
    if linestyle != :default && color != :default
        p = plot(0:length(IRF)-1, IRF, legend = :none, linewidth = linewidth, xlabel = xlabel, ylabel = label, color = color, linestyle = linestyle)
    elseif linestyle != :default
        p = plot(0:length(IRF)-1, IRF, legend = :none, linewidth = linewidth, xlabel = xlabel, ylabel = label, linestyle = linestyle)
    elseif color != :default
        p = plot(0:length(IRF)-1, IRF, legend = :none, linewidth = linewidth, xlabel = xlabel, ylabel = label, color = color)
    else
        p = plot(0:length(IRF)-1, IRF, legend = :none, linewidth = linewidth, xlabel = xlabel, ylabel = label)
    end
    hline!([hlineLevel], linestyle = :dot, color = :black)

    return p

end


function plotIRF!(p, IRF, SSS; isInterestRate = false, inLevelDifference = false, inDeviations = true, color = :default, linestyle = :default, linewidth = 2)

    # Transform the series
    if isInterestRate && inDeviations && !inLevelDifference
        IRF = log.(IRF)*400 .- log(SSS)*400
    elseif isInterestRate && inDeviations && inLevelDifference
        IRF = (IRF .- SSS) * 100
    elseif isInterestRate
        IRF = log.(IRF)*400
    elseif inDeviations && !inLevelDifference
        IRF = (IRF / SSS .- 1) * 100
    elseif inDeviations && inLevelDifference
        IRF = IRF .- SSS
    end

    # Plot the IRFs and horizontal line at SSS or at zero
    if linestyle != :default && color != :default
        plot!(p, 0:length(IRF)-1, IRF, linewidth = linewidth, color = color, linestyle = linestyle)
    elseif linestyle != :default
        plot!(p, 0:length(IRF)-1, IRF, linewidth = linewidth, linestyle = linestyle)
    elseif color != :default
        plot!(p, 0:length(IRF)-1, IRF, linewidth = linewidth, color = color)
    else
        plot!(p, 0:length(IRF)-1, IRF, linewidth = linewidth)
    end

    nothing

end

