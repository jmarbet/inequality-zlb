module RANKWageRigidities

## Preliminaries ###############################################################

using Parameters
using Interpolations
using Statistics
using Distributions
using BSON: @load, @save
using BSON
using Plots
using LaTeXStrings
using NLsolve
using GaussQuadrature
using OrderedCollections
using ProgressMeter

using Base.Threads
using LinearAlgebra
using Printf
using Dates
using Revise

include("../../Library/tools.jl")
include("Setup.jl")
include("Solve.jl")
include("Configs.jl")
include("Analysis.jl")

include("../HANKWageRigidities/HANKWageRigidities.jl")
using .HANKWageRigidities

theme(:default)
#theme(:juno)


## Main Functions ##############################################################

"""
    main()

Solves the model for the current parameter settings.

"""
function main(; config = tuple(), disableFigures = false)

    tstart = time()

    # Load settings
    P = settings(; config ...)

    # Draw the individual and aggregate shocks
    S = computeShocks(P)

    # Compute deterministic steady state
    DSS = solveSteadyState(P)

    # Solve representative agent model
    πPolicy, HPolicy = solveModel(P, DSS)

    # Simulate remaining variables
    simSeries = simulateRemainingVariables(P, DSS, S, πPolicy, HPolicy)

    # Compute stochastic steady state
    SSS = computeStochasticSteadyState(P, DSS, S, πPolicy, HPolicy)

    # Plot important variables
    if !disableFigures
        p = plotComparison(P, DSS, SSS, S, simSeries)
        display(p)
    end

    # Compute and plot IRFs
    IRFs1 = computeIRFs(P, DSS, SSS, πPolicy, HPolicy; std = -1)
    IRFs2 = computeIRFs(P, DSS, SSS, πPolicy, HPolicy; std = -2)
    IRFs3 = computeIRFs(P, DSS, SSS, πPolicy, HPolicy; std = -3)
    if !disableFigures
        _, p = plotIRFs(P, IRFs3, SSS)
        display(p)
    end

    # Compute value function
    V = computeValueFunction(P, S, DSS, πPolicy, HPolicy; showSteps = true)
    VInterpol = linear_interpolation((P.RGrid, P.ξGrid), V, extrapolation_bc = Line())

    # Display runtime
    displayTimeElapsed(tstart)

    # Save the results
    @save string(P.filename[1:end-5], ".bson") P DSS SSS πPolicy HPolicy simSeries IRFs1 IRFs2 IRFs3 V VInterpol

    nothing

end


function solveDifferentConfigs()

    for config in baselineConfig()
        println("Inflation (%): ", log(config[:π̃])*400)
        main(; config = config, disableFigures = true)
        println("\n")
    end

end


function displayTimeElapsed(tstart)

    dec = 10^3
    tt = time()
    T = tt - tstart
    hh = floor(Int64,T/3600)
    mm = floor(Int64,T/60)-60*hh
    ss = round(Int64,(T-60*mm-3600*hh)*dec)/dec
    println("Time Elapsed: $(hh)h$(mm)m$(ss)s")

    nothing

end

end
