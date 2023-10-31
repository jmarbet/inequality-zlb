## Preliminaries ###############################################################

include("Models/HANKWageRigidities/HANKWageRigidities.jl")
using .HANKWageRigidities

include("Models/RANKWageRigidities/RANKWageRigidities.jl")
using .RANKWageRigidities

include("PublicationPlots.jl")

using PolyesterWeave

## Main Functions ##############################################################

function main()

    # Make sure that all threads are free 
    PolyesterWeave.reset_workers!()

    # Solve all HANK and RANK configurations
    HANKWageRigidities.solveDifferentConfigs()
    RANKWageRigidities.solveDifferentConfigs()

    # Generate DSS overview figures
    HANKWageRigidities.mainSteadyState(; 
        filenameExt = "_Baseline", 
        loadSettingsFromFile = "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_075.bson",
        outputFolder = "Figures/HANKWageRigidities/PublicationPlotsInequalityAndZLB/Additional",
        showPlot = false
    )

    HANKWageRigidities.mainSteadyState(; 
        filenameExt = "_HighRisk", 
        loadSettingsFromFile = "Results/HANKWageRigidities/HANKWageRigidities_BaselineConfig_ZLB_pitilde_0_02_sig_0_095.bson",
        outputFolder = "Figures/HANKWageRigidities/PublicationPlotsInequalityAndZLB/Additional",
        showPlot = false
    )

    # Generate all plots required for the publication
    generatePublicationPlots()

    display("DONE!")

    nothing

end

