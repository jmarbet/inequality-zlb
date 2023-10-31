function baselineConfigGeneral(; customParameters = tuple())

    # Get common parameters form HANK calibration
    config = HANKWageRigidities.baselineConfigGeneral()
    commonParametersHANK = (:β, :σ, :ν, :χ, :ϕₗ, :ϕʸ, :ρ_R, :indexRotemberg, :θ, :ε, :RMax, :τ, :B, :useRStarForTaylorInertia, :ρ_ζ, :σ̃_ζ)
    commonParametersRANK = (:β, :σ, :ν, :χ, :ϕₗ, :ϕʸ, :ρ_R, :indexRotemberg, :θ, :ε, :RMax, :τ, :B, :useRStarForTaylorInertia, :ρ, :σ̃) # Note: RANK names are different for some paramteres
    generalConfig = NamedTuple{commonParametersRANK}([getfield(config, x) for x in commonParametersHANK])

    # Add RANK specific parameters
    configRANKSpecififc = (
        T = 101000,
        burnIn = 1000,
        ZLBLevel = 1.0,
        bindingZLB = true
        )
    generalConfig = merge(generalConfig, configRANKSpecififc)
    generalConfig = merge(generalConfig, customParameters)

    return generalConfig

end


function baselineConfig(; customParameters = tuple(), appendix = "")

    configs = []

    # Get the baseline parameters
    generalConfig = baselineConfigGeneral(; customParameters)
    
    # Generate configs for different inflation targets
    πTargets = [1.7:0.1:1.9; collect(2.0:0.25:4.0)] / 100

    for π in reverse(πTargets)

        # Generate filename
        πStr =  replace(string(round(π, digits = 4)), "." => "_")
        ZLBStr = generalConfig[:bindingZLB] ? "ZLB" : "NoZLB"
        filenameExt = "BaselineConfig_$(ZLBStr)_pitilde_$(πStr)$(appendix)"
        
        config = (generalConfig...,
            π̃ = exp(π/4),
            filenameExt = filenameExt)
        push!(configs, config)

    end

    # Recalibrated configs (r=1.5% at 2.0% inflation)
    for config in copy(configs)
        push!(configs, merge(config, (β = 1/exp(1.5/400), filenameExt = config[:filenameExt] * "_Recalibrated")))
    end

    # Add no ZLB versions for baseline configs
    for config in copy(configs[1:length(πTargets)])
        push!(configs, merge(config, (bindingZLB = false, filenameExt = replace(config[:filenameExt], "_ZLB_" => "_NoZLB_"))))
    end

    # Add high and low Rotemberg cost configs
    refConfig = configs[findfirst(reverse(πTargets) .== 0.02)] # 2% Target config
    push!(configs, merge(refConfig, (θ = 200.0, filenameExt = refConfig[:filenameExt] * "_HighRotembergCost")))
    push!(configs, merge(refConfig, (θ = 80.0, filenameExt = refConfig[:filenameExt] * "_LowRotembergCost")))

    return configs

end
