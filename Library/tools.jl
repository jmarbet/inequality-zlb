"""
    bisect(f,xLow,xUp; tol = 10e-8, maxIter = 100, showSteps = false)

Does bisection on given function.
"""
function bisect(f,xLow,xUp; tol = 10e-8, maxIter = 100, showSteps = false)

    # Compute initial values
    fLow = f(xLow)
    fUp = f(xUp)
    iter = 1

    # Check if endpoints have different signs
    if fLow*fUp > 0
        error("The function values at the interval endpoints must differ in sign.")
    end

    # Initilaize mid point
    local fMid, xMid

    while true # Stop when error is small enough

        xMid = (xUp+xLow)/2
        fMid::Float64 = f(xMid) # Added type to make function type stable

        if fLow * fMid > 0 # i.e. both have same sign
            xLow = xMid
            fLow = fMid
        else
            xUp = xMid
            #fUp = fMid
        end

        # Display current iteration
        if showSteps
            println("Bisection step ", iter, " (x: ", xMid, ", fval: ", fMid, ")")
        end

        # End the loop if the tolerance level is met
        if abs(fMid)<tol || iter>maxIter
            break
        end

        iter = iter + 1

    end

    # Solution state
    if iter >= maxIter && showSteps
       println("Maximum number of iterations reached")
    end

    return (x = xMid, fx = fMid, iter = iter)

end


"""
    falseposition(f,xLow,xUp; tol = 10e-8, maxIter = 100, showSteps = false)

Uses a false position algorithm on given function.
"""
function falseposition(f, xLow, xUp; tol = 10e-8, maxIter = 100, showSteps = false, showError = false)

    # Compute initial values
    fLow = f(xLow)
    fUp = f(xUp)
    iter = 1

    # Check if endpoints have different signs
    if fLow*fUp > 0
        error("The function values at the interval endpoints must differ in sign.")
    end

    # Initilaize mid point
    local fNew, xNew

    while true # Stop when error is small enough

        # Find next value
        xNew = xLow - fLow * (xUp - xLow)/(fUp-fLow)
        fNew = f(xNew)

        if fNew == 0.0
            # Exact root found
        elseif fLow * fNew < 0.0
            xUp = xNew
            fUp = fNew
        elseif fUp * fNew < 0.0
            xLow = xNew
            fLow = fNew
        end

        # Display current iteration
        if showSteps
            println("False position step ", iter, " (x: ", xNew, ", fval: ", fNew, ")")
        end

        # End the loop if the tolerance level is met
        if abs(fNew)<tol || iter>maxIter
            break
        end

        iter = iter + 1

    end

    # Solution state
    if iter >= maxIter && (showSteps || showError)
       println("Maximum number of iterations reached")
    end

    return (x = xNew, fx = fNew, iter = iter)

end


function tauchen(ρ, σ, n; nStd = 3)

    # Endpoints of discretization are nStd standard deviations away from the mean
    yMinMax = nStd*σ/sqrt(1-ρ^2)

    # Compute the grid values
    y = range(-yMinMax, length=n, stop=yMinMax)

    # Determine the distance between y's
    stepSize = y[2] - y[1]

    # Construct the transition matrix
    Π = zeros(n, n)

    for j in 1:n, i in 1:n
        if j == 1
            Π[i, j] = cdf(Normal(0, σ), y[j] - ρ*y[i] + stepSize/2)
        elseif j == n
            Π[i, j] = 1.0 - cdf(Normal(0, σ), y[j] - ρ*y[i] - stepSize/2)
        else
            Π[i, j] = cdf(Normal(0, σ), y[j] - ρ*y[i] + stepSize/2) -
                      cdf(Normal(0, σ), y[j] - ρ*y[i] - stepSize/2)
        end
    end

    return y, Π

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
