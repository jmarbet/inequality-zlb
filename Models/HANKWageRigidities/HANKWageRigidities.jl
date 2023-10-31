module HANKWageRigidities

## Preliminaries ###############################################################

using Parameters
using Interpolations
using Statistics
using StatsBase
using Distributions
using GaussQuadrature
using BSON: @load, @save
using Plots
using StatsPlots
using LaTeXStrings
using NLsolve
using Optim
using ProgressMeter
using Measures
using Random
using OrderedCollections
using BSON
using Polyester
using MINPACK

using Base.Threads
using LinearAlgebra
using Printf
using Dates
using Revise
using Formatting

include("../../Library/tools.jl")
include("Setup.jl")
include("Common.jl")
include("SteadyState.jl")
include("Solve.jl")
include("NeuralNetwork.jl")
include("LinearRegression.jl")
include("DualRegression.jl")
include("QuadRegression.jl")
include("Analysis.jl")
include("Configs.jl")
include("Plots.jl")


## Main Functions ##############################################################

"""
    main()

Solves the model for the current parameter settings.

"""
function main(; config = tuple(), forceGRPlots = true)

    tstart = time()

    # Set plotting backend
    if forceGRPlots
        gr()
    end

    # Load settings
    @time P = settings(; config...)

    # Compute the deterministic steady state
    @time DSS = computeDeterministicSteadyState(P)

    # Compute the aggregate shocks used for solving the model
    @time S = computeShocks(P)

    # Solve the model
    solveModel(P, S, DSS)

    # Display runtime
    displayTimeElapsed(tstart)

    nothing

end


"""
    mainSteadyState()

Solves the steady state model for the current parameter settings.

"""
function mainSteadyState(; config = tuple(), filenameExt = "", loadSettingsFromFile = "", 
        outputFolder = "Figures/HANKWageRigidities/DSS", forceGRPlots = true, showPlot = true)

    tstart = time()

    # Set plotting backend
    if forceGRPlots
        gr()
    end

    # Create the output folder
    if !isdir(outputFolder)
        mkpath(outputFolder)
    end

    # Load settings
    if !isempty(loadSettingsFromFile)

        # Load the results
        res = loadAllResults([loadSettingsFromFile], "Results/HANKWageRigidities/Results")

        # Get settings
        P = res[first(keys(res))][:P]

    else
        P = settings(; config...)
    end

    # Solve the steady state model
    DSS = computeDeterministicSteadyState(P)

    # Display summary statistics and generate overview plot
    pp = computeDSSOverview(P, DSS)
    if showPlot
        display(pp)
    end
    savefig("$(outputFolder)/DSSOverview$(filenameExt).pdf")
    
    # Display runtime
    displayTimeElapsed(tstart)

    nothing

end


function startFromPreviousSolution()

    tstart = time()

    # Load results
    filename = "Results/HANKWageRigidities/HANKWageRigidities_ZLB_pitilde_0_02.bson"
    @load filename P S DSS bPolicy πwALM EπwCondALM NNEπwCond NNπw bCross RStar H πw algorithmProgress

    # Load settings
    @time P = settings(ϕₕ = 1.5, filenameExt = "ZLB_pitilde_0_02_AsymmetricTaylorRule",
        filename = "Results/HANKWageRigidities/HANKWageRigidities_ZLB_pitilde_0_02_AsymmetricTaylorRule.bson")

    # Compute the deterministic steady state
    @time DSS = computeDeterministicSteadyState(P)

    # Solve the model staring from the solution that was loaded
    solveModel(P, S, DSS,
               bPolicy = bPolicy,
               πwALM = πwALM,
               EπwCondALM = EπwCondALM,
               NNEw = NNEw,
               NNEπwCond = NNEπwCond)

    # Display runtime
    displayTimeElapsed(tstart)

    nothing

end


"""
    solveDifferentConfigs()

Solves the model for different configurations.

"""
function solveDifferentConfigs(; 
        input = baselineConfig(), 
        configs = input[1], 
        noHomotopyList = input[2],
        onlyUseShocksForHomotopyList = input[3],
        skipList = input[4],
        useHomotopy = true,
        skipInitalConfig = false,
        prepResults = true,
        prepAdditionalResults = true,
        forceGRPlots = true
    )

    tstart = time()

    # Set plotting backend
    if forceGRPlots
        gr()
    end

    # Solve the model for the configurations defined in configs
    for (ii, config) in enumerate(configs)

        println("-------------------------------------------------------------")
        println("Configuration No. $(ii):")
        println(config)

        if (skipInitalConfig && ii == 1) || ii ∈ skipList
            println("Skipping config...")
            continue
        end

        if useHomotopy && ii != 1 && ii ∉ noHomotopyList

            println("Using previous solution as initial guess...")

            # Use the new configuration (Note: default settings are reused and not the loaded P)
            P = settings(; config...)

            # Compute the deterministic steady state
            DSS = computeDeterministicSteadyState(P)

            # Solve the model starting from the solution that was loaded
            if ii in onlyUseShocksForHomotopyList

                println("Only reusing shocks...")

                # Load previous results
                @load configs[ii-1].filename S

                solveModel(P, S, DSS)

            else

                # Load previous results
                @load configs[ii-1].filename S bPolicy πwALM EπwCondALM NNEπwCond NNπw bCross RStar H πw algorithmProgress

                solveModel(P, S, DSS,
                           bPolicy = bPolicy,
                           πwALM = πwALM,
                           EπwCondALM = EπwCondALM,
                           NNπw = NNπw, # Note: it might be better to not pass NNπw and NNEπwCond and just let the NN reinitialize randomly
                           NNEπwCond = NNEπwCond)

            end

        else

            println("Using default initial guess...")

            # Use the new configuration (Note: default settings are used for undefined settings)
            P = settings(; config...)

            # Compute the deterministic steady state
            DSS = computeDeterministicSteadyState(P)

            # Compute the aggregate shocks used for solving the model
            S = computeShocks(P)

            # Solve the model
            solveModel(P, S, DSS)

        end

        # Display runtime
        displayTimeElapsed(tstart)

    end

    # Prepare results require for analysis of the model
    if prepResults || prepAdditionalResults

        # Files containing solution (skip those that are only used as an initial guess
        filenames = unique([x.filename for x in configs if length(x) > 1])

        # Settings
        outputFolder = "Results/HANKWageRigidities/Results"

        # Create the output folder
        if !isdir(outputFolder)
            mkpath(outputFolder)
        end

        # Prepare results
        if prepResults
            prepareResults(filenames, outputFolder)
        end

        if prepAdditionalResults
            prepareAdditonalResults(filenames, outputFolder)
        end

    end

    # Display runtime
    displayTimeElapsed(tstart)

    nothing

end

end
