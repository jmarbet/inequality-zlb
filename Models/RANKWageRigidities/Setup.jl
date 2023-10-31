"""
    settings()

Defines model parameters and several settings for the solution algorithm.

"""
@with_kw struct settings

    # Model Parameters
    β::Float64 = 0.99398    # Discount factor
    σ::Float64 = 1.0        # Risk aversion coefficient
    ν::Float64 = 1.0        # Inverse Frisch labor supply elasticity
    ϕₗ::Float64 = 2.0        # Taylor rule coefficient on inflation (π_t < π̃)
    ϕₕ::Float64 = ϕₗ         # Taylor rule coefficient on inflation (π_t >= π̃)
    ϕʸ::Float64 = 0.0       # Taylor rule coefficient on output
    ρ_R::Float64 = 0.0      # Inertia in Taylor rule
    useRStarForTaylorInertia::Bool = false # Determines whether R or RStar us used for inertia in the Taylor rule
    π̃::Float64 = exp(0.02/4)# Steady state inflation
    χ::Float64 = (7.67-1)/7.67 # Disutility of labor
    indexRotemberg::Bool = true # If true, indexes adjustment cost to steady state inflation
    bindingZLB::Bool = true # If true, zero lower bound on nominal interest rates exists in the economy
    ZLBLevel::Float64 = 1.0 # Level below which "zero" lower bound is binding
    θ::Float64 = 79.41      # Rotemberg wage adjustment cost coefficient
    ε::Float64 = 7.67       # Elasticity of substitution across differentiated labor services
    λ::Float64 = 1.0        # Learning parameter for policy functions
    β̃::Float64 = β          # Discount factor of firms if stochastic discount factor is not used
    B::Float64 = 0.25       # Net supply of bonds
    τ::Float64 = 0.0        # Labor tax rate

    # Parameters for aggregate states
    nGHNodes::Int64 = 10    # Nodes for the Gauss-Hermite quadrature
    ξ̄::Float64 = 1.0        # Mean disocunt factor shock
    ρ::Float64 = 0.60       # Persistence of aggregate AR(1) process
    σ̃::Float64 = 0.01175    # Standard deviation of aggregate shock

    # Compute Gauss-Hermite quadrature nodes and weights (Note: weights include the π^(-1/2) term)
    tmpGH::Tuple{Array{Float64,1},Array{Float64,1}} = hermite(nGHNodes)
    eNodes::Array{Float64,1} = sqrt(2) * σ̃ * tmpGH[1]
    eWeights::Array{Float64,1} = tmpGH[2] / sqrt(pi)

    # Simulation settings
    T::Int64 = 1100         # Number of periods
    burnIn::Int64 = 100     # Number of discared burn-in periods

    # Policy function iteration settings
    tol::Float64 = 1e-8     # Tolerance policy function
    showPolicyIterations::Bool = false

    # Grid for nominal interest rate
    RGridSize::Int64 = 61
    RMin::Float64 = bindingZLB && !useRStarForTaylorInertia ? ZLBLevel : 0.98
    RMax::Float64 = 1.05
    RGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
        range(RMin, stop=RMax, length=RGridSize)

    # Aggregate shock grid
    ξGridSize::Int64 = 101
    ξGridMin::Float64 = 1 - max(1 - exp(-3.5*σ̃/sqrt(1-ρ^2)), exp(3.5*σ̃/sqrt(1-ρ^2)) - 1)
    ξGridMax::Float64 = 1 + max(1 - exp(-3.5*σ̃/sqrt(1-ρ^2)), exp(3.5*σ̃/sqrt(1-ρ^2)) - 1)
    ξGrid::StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64},Int64} =
            range(ξGridMin, stop=ξGridMax, length=ξGridSize)

    # Filename under which the results are saved
    filenameExt::String = "Baseline"
    filenameFolder::String = "RANKWageRigidities"
    filenamePrefix::String = "RANKWageRigidities"
    filename::String = "Results/$(filenameFolder)/$(filenamePrefix)_$(filenameExt).bson"

end
