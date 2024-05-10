# simLifSys


This repository contains all the code necessary to implement the experiments in

* Antoine Aspeel, Necmiye Ozay. [A Simulation Preorder for Koopman-like Lifted Control Systems](https://arxiv.org/abs/2401.14909). Presentation in _The 8th IFAC Conference on Analysis and Design of Hybrid Systems_, 2024.

The code uses Julia-1.10 and requires a Mosek license. Free Mosek licenses are available for academics. More information [here](https://www.mosek.com/).

In the folder « simLifSys-main », use the Julia REPL to activate the environment by running:

`import Pkg; Pkg.activate("Project.toml")`

Then, to reproduce the experiments, run:

`include("main.jl")`.

Other files:
* simulate.jl contains functions to construct an affine lifted system simulating a polynomial unlifted system.
* refine.jl contains the functions to check if an affine lifted system simulates another.
* backward_reachability.jl contains functions related to presets computation.
* utils.jl contains plot-related functions, and a structure to represent affine systems.
* test_functions.jl contains functions to test every function separetely.