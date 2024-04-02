#=
Given:
    * a nonlinear (polynomial) dynamics
    * a domain
    * a set of inputs
    * a list of lifting functions
for each lifting function in the list, this scipt:
    * construct an affine lifted system simulating the nonlinear system
    * compute the backward reachable sets (BRS) for each affine lifted system
for each pair (i,j) of lifted systems (with i!=j), this script
    * try to synthesize a refinement map such to prove LifSys_i <= LifSys_j.
=#
using Plots
using DynamicPolynomials
using LaTeXStrings
using Missings

include("utils.jl")
include("simulate.jl")
include("backward_reachability.jl")
include("refine.jl")

Random.seed!(1234) # useful for sample_polytope() in utils.jl

# create a time-stamped folder to store data.
#time_stamp = Dates.format(now(), "yyyy-mm-dd-HH-MM-SS")
#mkdir("saved_files")
folder_name="saved_files/" 

skip_BRS=false
skip_plot=false
skip_refinement=false

println("======================")
println("= PROBLEM DEFINITION =")
println("======================")

n_x=2
n_u=1

@polyvar x[1:n_x] u[1:n_u]

dt=0.1
f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation

# H-rep of the safety set
H_Sx = [1 0; -1 0; 0 1; 0 -1]
h_Sx = [2; 2; 3; 3] # [2; 2; 3; 3]
Sx = hrep(H_Sx,h_Sx)

# H-rep of the input set
H_Su = [1; -1;;] # must be a matrix
h_Su = [50; 50]
Su = hrep(H_Su,h_Su)

target_set = hrep( Polyhedra.polyhedron(BallInf(zeros(2), 0.5)) )

lifting_1 = [x;] 
lifting_2 = [x; x[1]^3]
lifting_3 = [x; x[1]*x[2]]

list_liftings = [lifting_1, lifting_2, lifting_3] # list_liftings[i] = lifting_i

cone_type_simulation = "sos"
maxdegree_certificate_simulation = 10
cone_type_refinement = "sdsos"
maxdegree_certificate_refinement = 10

n_presets = 10

number_of_liftings = length(list_liftings)

###### Compute linear dynamics for each lifting function #####
println("===================")
println("== LINEAR SYSTEM ==")
println("===================")

list_linear_systems=[]
println("Compute $number_of_liftings affine lifted systems...")
model_simulate = missings(Any, number_of_liftings)
for (i,lifting) in enumerate(list_liftings)
    linear_system, model_simulate_loc = koopman_over_approx(x, u, f, lifting, Sx, Su, maxdegree_certificate = maxdegree_certificate_simulation, cone_type = cone_type_simulation)
    push!(list_linear_systems,linear_system)
    println("$i/$number_of_liftings affine lifted systems computed.")
    model_simulate[i]=model_simulate_loc
end

if !skip_BRS
    ##### COMPUTE BRS #####
    println("===============")
    println("===== BRS =====")
    println("===============")

    list_BRS=[]
    println("Compute $number_of_liftings BRS (each with $n_presets steps)...")
    for (i,linear_system) in enumerate(list_linear_systems)
        implicit_presets = compute_implicit_presets(linear_system, Sx, Su, target_set, n_presets)
        push!(list_BRS,implicit_presets)
        println("BRS computed for $i/$number_of_liftings lifted systems.")
    end
end

if !skip_plot
    ##### PLOT BRS #####
    println("=================")
    println("===== PLOTS =====")
    println("=================")

    n_samples=2*1e3

    fig = plot(polyhedron(Sx), alpha=0.1, label=L"\mathcal{X}")
    plot!(fig,polyhedron(target_set), color=:yellow, alpha=0.3, label=L"Target\ set")

    # generate samples used to plot implicit BRS
    if Sx.A == [1 0; -1 0; 0 1; 0 -1] # must be hyperbox in canonical form
        samples_x = grid_polytope(Sx, ceil(Int64,sqrt(n_samples)))
    else
        println("CAUTION: random sampling was used (instead of grid sampling) for plot_implicit_representations_hull!() because domain_x was not a 2D hyperbod in canonical form")
        samples_x = [sample_polytope(vrep(polyhedron(domain_x))) for i=1:n_samples]
    end

    list_colors = [:blue; :red; :green; :purple; :orange; :pink; :brown; :black; :grey23]
    list_linestyles = [:solid; :dash; :dot; :dashdot; :dashdotdot; :auto; :auto; :auto]
    println("Plot $number_of_liftings implicit plots (each with $n_presets steps)...")
    for (i,lifting) in enumerate(list_liftings)
        plot_implicit_representations_hull!(fig, x, Sx, list_BRS[i], lifting, samples_x, n_samples, 50; color=list_colors[i], linestyle=list_linestyles[i], label=L"BRS_%$i")
        println("$i/$number_of_liftings BRS plotted.")
    end
    plot!(fig,legend=:bottomright)
    xlabel!(L"x_1")
    ylabel!(L"x_2")

    display(fig)
    savefig(fig, folder_name*"brs.pdf" )
end

if !skip_refinement
    ##### TRY TO REFINE #####
    println("==================")
    println("===== REFINE =====")
    println("==================")

    #models = missings(Any, number_of_liftings, number_of_liftings)
    status = missings(Any, number_of_liftings, number_of_liftings)
    duration = missings(Any, number_of_liftings, number_of_liftings)
    for (i_good,lifting_good) in enumerate(list_liftings)
        linear_system_good = list_linear_systems[i_good]
        for (i_bad,lifting_bad) in enumerate(list_liftings) # try to verify LifSys_good ≤ LifSys_bad
            if i_good != i_bad
                linear_system_bad = list_linear_systems[i_bad]
                println("Try to prove LifSys_$i_good ≤ LifSys_$i_bad ...")

                model_loc = compute_refinement(x, Sx, Su, lifting_bad, linear_system_bad, lifting_good, linear_system_good, maxdegree_certificate=maxdegree_certificate_refinement, cone_type=cone_type_refinement)
                set_time_limit_sec(model_loc, 60.0)
                duration[i_good,i_bad] = @elapsed optimize!(model_loc)

                status_loc = termination_status(model_loc)
                println("TERMINATION_STATUS: ", status_loc)

                status[i_good,i_bad] = status_loc
            end
        end
    end
    println()
    println("===================")
    println("===== RESULTS =====")
    println("===================")
    println("Results when trying to verify LifSys_i ≤ LifSys_j for i!=j.")
    println("A refinement map has been found for LifSys_i ≤ LifSys_j iff termination_status[i,j]=='OPTIMAL::TerminationStatusCode = 1'.")
    println("termination_status=")
    display(status)
    #println("optimization time")
    #display(duration)
end
