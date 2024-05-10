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

Random.seed!(1234) # for reproducibility.

folder_name="saved_files/" 

skip_BRS=false
skip_plot=false
skip_pre_feasibility=false
skip_refinement=false

println("""
    ======================
    = PROBLEM DEFINITION =
    ======================
    """)

n_x=2
n_u=1

@polyvar x[1:n_x] u[1:n_u]

dt=0.1
f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation

# H-rep of the safety set
H_Sx = [1 0; -1 0; 0 1; 0 -1]
h_Sx = [2; 2; 2; 2] #[2; 2; 3; 3]
Sx = hrep(H_Sx,h_Sx)

# H-rep of the input set
H_Su = [1; -1;;] # must be a matrix
h_Su = [50; 50]
Su = hrep(H_Su,h_Su)

target_set = hrep( Polyhedra.polyhedron(BallInf(zeros(2), 1.0)) )

lifting_1 = [x;]
lifting_2 = [x;]
lifting_3 = [x; x[1]^2]
lifting_4 = [x; x[1]^2]
lifting_5 = [x; x[1]^2; x[1]^3]
lifting_6 = [x; x[1]^2; x[1]^3]
# The dynamics associated with even liftings (that is lifting_2, lifting_4, lifting 6) will be artificially increased. That is why each lifting is considered twice.

# list of feasible liftings:
# [x;]
# [x; x[1]^2]
# [x; x[1]^3]
# [x; x[1]^2*x[2]]
# [x; x[1]^4]

list_liftings = [lifting_1, lifting_2, lifting_3, lifting_4, lifting_5, lifting_6]

cone_type_simulation = "sos"
maxdegree_certificate_simulation = 10 #10
cone_type_refinement = "sdsos" # sdsos # note that sampling is used instead.
maxdegree_certificate_refinement = 10 # 10

n_presets = 10 # number of presets in the computation of backward reachable sets.

number_of_liftings = length(list_liftings)

###### Compute linear dynamics for each lifting function #####
println("""
    ===================
    == LINEAR SYSTEM ==
    ===================
    """)

list_linear_systems=[]
println("Compute $number_of_liftings affine lifted systems...")
model_simulate = missings(Any, number_of_liftings)

for (i,lifting) in enumerate(list_liftings)
    println("Compute linear system for lifting i=$i.")
    linear_system, model_simulate_loc = koopman_over_approx(x, u, f, lifting, Sx, Su, maxdegree_certificate = maxdegree_certificate_simulation, cone_type = cone_type_simulation)
    
    # Hack: Increase the noise for even liftings. !
    if iseven(i)
        println("Hacking i == $i")
        println("linear_system.W.b before hack:", linear_system.W.b)
        linear_system.W.b = linear_system.W.b .+10
        println("linear_system.W.b after hack:", linear_system.W.b)
    end

    push!(list_linear_systems,linear_system)
    println("$i/$number_of_liftings affine lifted systems computed.")
    model_simulate[i]=model_simulate_loc
    println("linear_system.A_xbar:\n "); display(linear_system.A)
end
return

if !skip_BRS
    ##### COMPUTE BRS #####
    println("""
        ===============
        ===== BRS =====
        ===============
        """)

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
    println("""
        =================
        ===== PLOTS =====
        =================
        """)

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

    list_colors = [:black; :pink; :blue; :orange; :red; :green; :purple; :brown; :grey23; :auto]
    list_linestyles = [:solid; :dashdotdot; :dash; :dot; :dashdot; :auto; :auto; :auto; :auto; :auto]
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

if !skip_pre_feasibility
    println("""
            ==================
            ==== PRE-FEAS ====
            ==================
            """)
    pre_feasibility = missings(Any, number_of_liftings, number_of_liftings)
    for (i_good,linear_system_good) in enumerate(list_linear_systems)
        A_good = linear_system_good.A
        for (i_bad,linear_system_bad) in enumerate(list_linear_systems)
            A_bad = linear_system_bad.A
            pre_feasibility[i_good,i_bad] = dynamics_feasibility_zbar(A_bad, A_good, n_x)
        end
    end
    println()
    println("""
            Pre-feasibility for LifSys_i ≤ LifSys_j.
            Condition for LifSys_i ≤ LifSys_j was pre-feasible iff pre_feasibility[i,j] is true.
            """)
        println("pre_feasibility=")
        display(pre_feasibility)
end

if !skip_refinement
    ##### TRY TO REFINE #####
    println("""
        ==================
        ===== REFINE =====
        ==================
        """)

    #models = missings(Any, number_of_liftings, number_of_liftings)
    status = missings(Any, number_of_liftings, number_of_liftings)
    duration = missings(Any, number_of_liftings, number_of_liftings)
    model = missings(Any, number_of_liftings, number_of_liftings)
    for (i_good,lifting_good) in enumerate(list_liftings)
        linear_system_good = list_linear_systems[i_good]
        for (i_bad,lifting_bad) in enumerate(list_liftings) # try to verify LifSys_good ≤ LifSys_bad
            if i_good != i_bad && pre_feasibility[i_good,i_bad]
                linear_system_bad = list_linear_systems[i_bad]
                println("Try to prove LifSys_$i_good ≤ LifSys_$i_bad ...")

                model_loc = compute_refinement(x, Sx, Su, lifting_bad, linear_system_bad, lifting_good, linear_system_good, maxdegree_certificate=maxdegree_certificate_refinement, cone_type=cone_type_refinement)
                set_time_limit_sec(model_loc, 10*60)
                set_attribute(model_loc, "FeasibilityTol", 1e-6)

                duration[i_good,i_bad] = @elapsed optimize!(model_loc)
                model[i_good,i_bad] = model_loc
                status_loc = termination_status(model_loc)
                println("TERMINATION_STATUS: ", status_loc)

                status[i_good,i_bad] = status_loc
            end
        end
    end
    println()
    println("""
        ===================
        ===== RESULTS =====
        ===================
        Results when trying to verify LifSys_i ≤ LifSys_j for i!=j.
        A refinement map has been found for LifSys_i ≤ LifSys_j iff termination_status[i,j]=='OPTIMAL::TerminationStatusCode = 1'.
        """)
    println("termination_status=")
    display(status)
end