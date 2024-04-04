# Inequality and the Zero Lower Bound

This repository contains the Julia code to solve the model developed in "Inequality and the Zero Lower Bound" by Jesús Fernández-Villaverde, Joël Marbet, Galo Nuño, and Omar Rachedi. 


## Replication

The code has been developed and tested on Julia 1.9.3. However, it is recommended that you use Julia 1.10.2 (or later) due to the large memory usage under Julia 1.9 for long-running processes.


### Setting Up the Environment in Julia

The basic procedure to set up the required environment in Julia is as follows:

1. Install Julia and clone this repository.
2. Open Julia and type `]` in the Julia REPL to enter package mode.
3. Type `activate .` to activate the environment in the current folder.
4. Type `instantiate` to download and install all required Julia packages.

This setup is sufficient to solve all model parametrizations. However, to generate all the figures shown in the paper, one also needs a working installation of LaTeX.


### Generating the Results From Scratch

To produce all figures and tables used in the paper, execute `main()` found in `Main.jl`. The code then

1. Solves all HANK model parametrizations,
2. Solves all RANK model parametrizations, and
3. Generates all figures and tables in the paper as well as some auxiliary figures.

The resulting figures and tables can then be found in `Figures/HANKWageRigidities/PublicationPlotsInequalityAndZLB`.


### Downloading the Published Results

The results used in the published version of the paper are hosted [here](https://www.archive.org/details/inequality-zlb) due to GitHub's file size limitations.
