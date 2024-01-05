# simLifSys
Repository for the paper "A Simulation Preorder for Koopman-like Lifted Control Systems" from Antoine Aspeel and Necmiye Ozay.

Run the script main.jl to reproduce the results from the paper.

* simulate.jl contains functions to construct an affine lifted system simulating a polynomial unlifted system.
* refine.jl contains the functions to check if an affine lifted system simulates another.
* backward_reachability.jl contains functions related to presets computation.
* utils.jl contains plot-related functions, and a structure to represent affine systems.
* test_functions.jl contains functions to test every function separetely.
