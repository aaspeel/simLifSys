# this file contains test functions.

using Plots
using DynamicPolynomials
using LazySets

include("utils.jl")
include("simulate.jl")
include("backward_reachability.jl")
include("refine.jl")

function test_forward_image()
    Random.seed!(1234) # useful for sample_polytope() in utils.jl

    n_x=2
    n_u=1

    @polyvar x[1:n_x] u[1:n_u]

    dt=1
    f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation
    
    #f = x + 0.01*[x[2]; x[1]^2+x[1]^3+u[1]]

    lifting = [x; x[1]^3]

    # H-rep of the safety set
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [5; 5; 5 ; 5]
    Sx = hrep(H_Sx,h_Sx)

    # H-rep of the input set
    H_Su = [1; -1;;] # must be a matrix
    h_Su = [50; 50]
    Su = hrep(H_Su,h_Su)

    linear_system = koopman_over_approx(x, u, f, lifting, Sx, Su, maxdegree_certificate = nothing)

    x0=[0; 0]
    lifted_image = forward_image(linear_system, x, lifting, Su, x0)

    p = plot(polyhedron(Sx), alpha=0.1, title="bad", label="safe set")
    plot_implicit_representation!(p, x, Sx, polyhedron(lifted_image), lifting, 1e5)
    display(p)
end

#test_forward_image()

function compare_BRS()
    Random.seed!(1234) # useful for sample_polytope() in utils.jl

    n_x=2
    n_u=1

    @polyvar x[1:n_x] u[1:n_u]

    #dt=0.025
    #f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation
    f = x + 0.01*[x[2]; x[1]^2+x[1]^3+u[1]]

    lifting_bad = [x;]
    lifting_good = [x; x[1]^3]

    # H-rep of the safety set
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [5; 5; 10 ; 10]
    Sx = hrep(H_Sx,h_Sx)

    # H-rep of the input set
    H_Su = [1; -1;;] # must be a matrix
    h_Su = [50; 50]
    Su = hrep(H_Su,h_Su)

    linear_system_bad = koopman_over_approx(x, u, f, lifting_bad, Sx, Su, maxdegree_certificate = 10)
    linear_system_good = koopman_over_approx(x, u, f, lifting_good, Sx, Su, maxdegree_certificate = 10)

    """
    model = compute_refinement(x, Sx, Su, lifting_bad, linear_system_bad, lifting_good, linear_system_good)
    optimize!(model)
    println("TERMINATION_STATUS: ", termination_status(model))
    """

    n_presets = 50

    target_set = hrep( Polyhedra.polyhedron(BallInf(zeros(2), 1.0)) )

    implicit_presets_bad = compute_implicit_presets(linear_system_bad, Sx, Su, target_set, n_presets)
    implicit_presets_good = compute_implicit_presets(linear_system_good, Sx, Su, target_set, n_presets)

    # plots
    p_bad = plot(polyhedron(Sx), alpha=0.1, title="bad", label="safe set")
    p_good = plot(polyhedron(Sx), alpha=0.1, title="good")
    p_both = plot(polyhedron(Sx), alpha=0.1, title="both")
    plot!(p_bad,polyhedron(target_set), color=:yellow, alpha=0.3, label="target set")
    plot!(p_good,polyhedron(target_set), color=:yellow, alpha=0.3)
    plot!(p_both,polyhedron(target_set), color=:yellow, alpha=0.3)

    alpha=1
    alpha_low=0.05
    for k=1:n_presets
        plot_implicit_representation!(p_bad, x, Sx, implicit_presets_bad[k], lifting_bad, 1e4; markershape = :square, markersize=1, alpha=alpha, color=:red,primary=false)
        plot_implicit_representation!(p_good, x, Sx, implicit_presets_good[k], lifting_good, 1e4; markershape = :square, markersize=1, alpha=alpha, color=:green)
        plot_implicit_representation!(p_both, x, Sx, implicit_presets_bad[k], lifting_bad, 1e4; markershape = :square, markersize=1, alpha=alpha, color=:red)
    end
    for k=1:n_presets
        plot_implicit_representation!(p_both, x, Sx, implicit_presets_good[k], lifting_good, 1e4; markershape = :square, markersize=1, alpha=alpha_low, color=:green)
    end
    plot!(p_bad,legend=:bottomleft)
    p=plot(p_bad,p_good,p_both)
    display(p)
end

#compare_BRS()

function test_refine()
    Random.seed!(1234) # useful for sample_polytope() in utils.jl

    n_x=2
    n_u=1

    @polyvar x[1:n_x] u[1:n_u]

    dt=0.025
    f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation

    lifting_bad = [x; x[1]^2]
    lifting_good = [x; x[1]^3; x[2]^2]

    # H-rep of the safety set
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [0.5; 0.5; 1.5 ; 1.5]
    Sx = hrep(H_Sx,h_Sx)

    # H-rep of the input set
    H_Su = [1; -1;;] # must be a matrix
    h_Su = [5; 5]
    Su = hrep(H_Su,h_Su)

    linear_system_bad = koopman_over_approx(x, u, f, lifting_bad, Sx, Su, maxdegree_certificate = nothing)
    linear_system_good = koopman_over_approx(x, u, f, lifting_good, Sx, Su, maxdegree_certificate = nothing)

    model = compute_refinement(x, Sx, Su, lifting_bad, linear_system_bad, lifting_good, linear_system_good)
    optimize!(model)
    println("TERMINATION_STATUS: ", termination_status(model))
end

#test_refine()

function test_implicit_BRS()
    n_x=2
    n_u=1

    @polyvar x[1:n_x] u[1:n_u]

    dt=0.025
    f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation

    lifting = [x; x[1]^3]

    # H-rep of the safety set
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [0.5; 0.5; 1.5 ; 1.5]
    Sx = hrep(H_Sx,h_Sx)

    H_Su = [1; -1;;] # must be a matrix
    h_Su = [5; 5]
    Su = hrep(H_Su,h_Su)

    target_set = hrep( Polyhedra.polyhedron(BallInf(zeros(2), 0.5)) )

    n_presets = 3

    linear_system = koopman_over_approx(x, u, f, lifting, Sx, Su, maxdegree_certificate=nothing)

    implicit_presets = compute_implicit_presets(linear_system, Sx, Su, target_set, n_presets)

    p = plot(polyhedron(target_set), alpha=0.3)
    for k=1:n_presets
        #plot_implicit_representation!(p, x, Sx, implicit_presets[k], lifting, 1e5; markershape = :hexagon, ma=1-k/(n_presets+1))
        plot_implicit_representation_hull!(p, x, Sx, implicit_presets[k], lifting, 1e2)
    end
    display(p)

end

#test_implicit_BRS()

function test_simulate()
    n_x=2
    n_u=1

    @polyvar x[1:n_x] u[1:n_u]

    dt=0.025
    f = x + dt*[x[2] ; 2*x[1]-2*x[1]^3-0.5*x[2]+u[1]] # Duffing equation

    lifting = [x; x[1]^3]

    # H-rep of the safety set
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [0.5; 0.5; 1.5 ; 1.5]
    Sx = hrep(H_Sx,h_Sx)

    H_Su = [1; -1;;] # must be a matrix
    h_Su = [5; 5]
    Su = hrep(H_Su,h_Su)

    linear_system = koopman_over_approx(x, u, f, lifting, Sx, Su, maxdegree_certificate=10)

    print("A: ")
    display(linear_system.A)
    print("B: ")
    display(linear_system.B)
    print("W: ")
    display(linear_system.W)
end

#test_simulate()

function test_implicit_plot()
    n_x=2
    @polyvar x[1:n_x]
    lifting = [x; .5+x[1]^2; .5+x[2]^2]
    #domain_x = hrep(Polyhedra.polyhedron(LazySets.BallInf(zeros(2), 1.0)))
    H_Sx = [1 0; -1 0; 0 1; 0 -1]
    h_Sx = [5; 5; 5 ; 5]
    domain_x = hrep(H_Sx,h_Sx)
    
    lifted_set = domain_x*domain_x

    p = plot(polyhedron(domain_x), alpha=0.3)
    plot_implicit_representation_hull!(p, x, domain_x, lifted_set, lifting, 100; markershape = :hexagon)
    display(p)
end

#test_implicit_plot()

function test_BRS()
    # Define the linear dynamics
    A = [1 0; 1 1]
    B = [1 0; 0 1]
    W = hrep(Polyhedra.polyhedron(LazySets.BallInf(zeros(2), 1.0)))
    linearSystem = LinearDynamics(A,B,W)

    # Define the sets
    safe_set = Polyhedra.polyhedron(BallInf(zeros(2), 6.0))
    input_set = Polyhedra.polyhedron(BallInf(zeros(2), 1.0))
    target_set = Polyhedra.polyhedron(BallInf(zeros(2), 5.0))

    n_presets = 8

    # compute the presets
    presets = compute_presets(linearSystem, safe_set, input_set, target_set, n_presets)

    #println("COMPUTED POLYHEDRONS:")
    #display(presets)

    # plot the target set and the presets
    plot(target_set, color="blue")

    solver = Gurobi.Optimizer

    for k=1:n_presets
        if isempty(presets[k], solver)
            println("presets for k≥$k are empty.")
            break
        end
        plot!(presets[k], color="blue", alpha=1-k/(n_presets+1))
    end
    current()
end

#test_BRS()