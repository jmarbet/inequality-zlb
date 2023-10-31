"""
    computeDeterministicSteadyState(P)

Solves for the steady state quantities assuming that there are no aggregate
shocks but agents still face idiosyncratic risk.

"""
function computeDeterministicSteadyState(P)

    # Check whether the supplied settings are valid
    validateSettings(P)

    # Equilibrium nominal interest rate
    R = solveSteadyState(P)
    RStar = R # Since we assume the ZLB is not binding in the DSS

    # Steady state preference shock
    ξ = P.ξ̄

    # Steady state risk premium / discount factor shock
    q = P.q̄

    # Steady state aggregate shock
    ζ = P.ζ̄

    # Price inflation
    π = P.π̃

    # Wage inflation
    πw = P.π̃

    # Real wage
    w = 1.0

    # Aggregate labor supply
    if P.indexRotemberg
        H = (1/P.χ * (P.ε-1)/P.ε * (1 - P.τ) * w)^(1/(P.σ+P.ν))
    else
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 - P.τ) * w + P.θ * (1-P.β̃) / P.ε * log(πw)))^(1/(P.σ+P.ν))
    end

    # Output and aggregate consumption
    Y = H
    C = Y

    # Government debt
    B = P.B

    # Transfers
    T = P.τ * w * H - (R/π - 1) * B

    # Return on bonds
    r = R / π

    # Get the crossectional distribution of assets
    _, _, bCross, bPolicy = getSteadyStateAggregates(P, R)

    # (Wage) inflation expectation term
    if P.indexRotemberg
        Eπw = 0.0
        EπwCond = 0.0
    else
        Eπw = log(πw)
        EπwCond = log(πw)
    end

    return (R = R,
            RStar = RStar,
            r = r,
            ξ = ξ,
            q = q,
            ζ = ζ,
            w = w,
            H = H,
            Y = Y,
            C = C,
            B = B,
            T = T,
            π = π,
            πw = πw,
            Eπw = Eπw,
            EπwCond = EπwCond,
            bCross = bCross,
            bPolicy = bPolicy)
end


"""
    solveSteadyState(P)

Solves for the steady state nominal interest rate assuming that there are no aggregate
shocks but agents still face idiosyncratic risk.

"""
function solveSteadyState(P)

    # Solve for equilibrium nominal interest rate
    f(x) = getSteadyStateAggregates(P, x)[1]
    #res = bisect(f, P.RMin, P.RMax; tol = P.tolDSSRate, maxIter = 100, showSteps = false)
    res = falseposition(f, P.RMin, P.RMax; tol = P.tolDSSRate, maxIter = 100, showSteps = false)

    return res.x

end


"""
    getSteadyStateAggregates(P, R)

Computes (net) supply/demand of bonds for a given level of nominal interest rate R.

"""
function getSteadyStateAggregates(P, R)

    # Price infation
    π = P.π̃
    
    # Wage inflation
    πw = P.π̃

    # Real wage
    w = 1.0

    # Aggregate labor supply
    if P.indexRotemberg
        H = (1/P.χ * (P.ε-1)/P.ε * (1 -P.τ) * w)^(1/(P.σ+P.ν))
    else
        H = (1/P.χ * ((P.ε-1)/P.ε * (1 -P.τ) * w + P.θ * (1-P.β̃) / P.ε * log(πw)))^(1/(P.σ+P.ν))
    end

    # Output
    Y = H

    # Government debt
    B = P.B

    # Transfers
    T = P.τ * w * H - (R/π - 1) * B

    # Make an initial guess for the individual bond policy function
    bPolicy = zeros(P.bGridSize, P.sGridSize)
    for ii in 1:P.sGridSize
        bPolicy[:, ii] .= 0.9 * P.bGrid
    end

    # Find the asset policy function
    solveIndividualSteadyStateProblem!(P, bPolicy, R, π, w, H, T)

    # Simulate the model
    bCrossInit = ones(P.bDenseGridSize, P.sGridSize) / (P.bDenseGridSize * P.sGridSize)
    bCross = simulateSteadyStateModel(P, bPolicy, bCrossInit)

    # Amount of bonds supplied/demanded by households
    effectiveB = sum(P.bDenseGrid' * bCross)

    # Compute the error in the mean of the bond distribution compared to the
    # targeted bond supply
    errorB = effectiveB - B

    return errorB, B, bCross, bPolicy

end


"""
    solveIndividualSteadyStateProblem!(P, bPolicy, R, π, w, H, T)

Finds individual bond policy function given prices.

"""
function solveIndividualSteadyStateProblem!(P, bPolicy, R, π, w, H, T)

    # Initialize matrices
    bPolicyError = similar(bPolicy)
    bPolicyUpdate = similar(bPolicy)
    wealth = similar(bPolicy)

    # Precompute wealth
    for idx in CartesianIndices(bPolicy)

        # Get indices in the grid
        i_b = idx[1]
        i_s = idx[2]

        # Compute cash on hand
        wealth[idx] = computeIndividualCashOnHand(P, P.bGrid[i_b], P.sGrid[i_s], R, π, w, H, T)

    end

    # Interpolate bPolicy
    bPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), bPolicy, extrapolation_bc = Line())

    # Initialize asset policy function iteration
    iter = 1
    dist = 10.0

    # Do bond policy funtion interation
    while(dist > P.tol)

        # Update the interpolation
        bPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), bPolicy, extrapolation_bc = Line())

        # Compute the proposal for new asset policy function
        @threads for idx in CartesianIndices(bPolicy)
            bPolicyUpdate[idx] =
                updateSteadyStateBondPolicy(P, bPolicyInterpol, bPolicy[idx], wealth[idx], R, π, w, H, T, idx)
        end

        # Check the distance between current iteration and the previous one
        @. bPolicyError = abs.(bPolicyUpdate - bPolicy)
        dist = maximum(bPolicyError)

        # Update the asset policy function
        @. bPolicy = P.λᵇ * bPolicyUpdate + (1-P.λᵇ) * bPolicy
        iter = iter+1

        if iter > P.maxPolicyIterations
            if P.showWarningsAndInfo
                @warn "Maximum number of iterations reached: dist = $(dist), R = $(R)"
            end
            break
        end

    end

    nothing

end


"""
    updateSteadyStateBondPolicy(P, bPolicyInterpol, bPrime, wealth, R, π, w, H, T, idx)

Auxiliary function called when solving the individual steady state problem. Computes the
updated policy function for a particular node in the state space.

"""
function updateSteadyStateBondPolicy(P, bPolicyInterpol, bPrime, wealth, R, π, w, H, T, idx)

    # Get indices in the grid
    i_b = idx[1]
    i_s = idx[2]

    # Initialize expectation term in the Euler equation
    expec = 0.0

    for jj in 1:P.sGridSize

        # Get bond decision at t+1
        b2Prime = bPolicyInterpol(bPrime, jj)

        # Compute consumption from the budget constraint
        cPrime = computeIndividualCashOnHand(P, bPrime, P.sGrid[jj], R, π, w, H, T) - b2Prime
        cPrime = cPrime < 0.0 ? 1e-10 : cPrime # Make sure that consumption is positive

        # Compute marginal utility of future consumption
        muPrime = cPrime^(-P.σ)

        # Add to the expectation term
        expec += muPrime * R/π * P.Ω[i_s, jj]

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
    simulateSteadyStateModel(P, bPolicy, bCrossInit)

Simulates the model for a given sequence of shocks, policy functions, and an
initial distribution of bonds.

"""
function simulateSteadyStateModel(P, bPolicy, bCrossInit)

    # Set the cross sectional distribution of bonds
    bCross = copy(bCrossInit)
    bPrimeCross = copy(bCrossInit)
    bCrossError = similar(bCross)

    # Create an interpolation function
    bPolicyInterpol = linear_interpolation((P.bGrid, 1:P.sGridSize), bPolicy, extrapolation_bc = Line())

    for tt in 1:P.T

        # Update the bond distribution
        propagateBondDistribution!(P, bPrimeCross, bCross, bPolicyInterpol)

        # Compute the difference between the new and old bond distribution
        @. bCrossError = abs(bCross - bPrimeCross)
        dist = maximum(bCrossError)

        # If the distribution hasn't changed much, we are at the DSS
        if dist < P.tolDSSDistribution
            bCross .= bPrimeCross
            break
        else
            bCross .= bPrimeCross
        end

    end

    return bCross

end


"""
    getAdjacentBondGridPoints(P, b)

Computes the adjacent grid points in P.bDenseGrid and returns their indices. If b is
outside of the grid, both indices correspond to either the minimum or maximum index.

"""
function  getAdjacentBondGridPoints(P, b)

    if b < P.bMin # b is not within the grid (too low)
                  # This should not happen because of the borrowing constraint
        return 1, 1

    elseif b > P.bMax # b is not within the grid (too high)

        return P.bDenseGridSize, P.bDenseGridSize

    else

        # Make sure that b is within grid bounds
        # Note: this is done to ensure that the returned indices are always valid
        b = min(P.bMax - 1e-6, b)
        b = max(P.bMin + 1e-6, b)

        i_bLow = floor(Int64, (b - P.bMin) / P.ΔbDense) + 1
        i_bUp = ceil(Int64, (b - P.bMin) / P.ΔbDense) + 1

        # Note: If a grid point is hit exactly, i_bLow and i_bUp should be the same

        return i_bLow, i_bUp

    end

end


"""
    getReassignmentWeight(P, b, i_bLow, i_bUp)

Computes the weights for updating the bond disribution as in Young (2010).

"""
function getReassignmentWeight(P, b, i_bLow, i_bUp)

    if i_bLow == i_bUp
        ω = 1.0
    else
        ω = 1 - (b - P.bDenseGrid[i_bLow]) / (P.bDenseGrid[i_bUp] - P.bDenseGrid[i_bLow])
    end

    return ω

end
