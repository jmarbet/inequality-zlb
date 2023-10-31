"""
    computeIndividualCashOnHand(P, b, s, R, π, w, H, T)

Computes individual resources available given individual states/choices and aggregates.

"""
function computeIndividualCashOnHand(P, b, s, R, π, w, H, T)

    return R / π * b + (1 - P.τ) * w * s * H + T

end

"""
    computeIndividualCashOnHand(P, DSS, b, s)

Computes individual resources available given individual states/choices and aggregates in the DSS.

"""
function computeIndividualCashOnHand(P, DSS, b, s)

    return computeIndividualCashOnHand(P, b, s, DSS.R, DSS.π, DSS.w, DSS.H, DSS.T)

end
