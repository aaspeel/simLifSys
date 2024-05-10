# Contain functions to compute a Koopman over approx., i.e., a linear lifted system

using Polyhedra
using DynamicPolynomials
using JuMP
using SumOfSquares
using CSDP
using Gurobi
using Mosek
using MosekTools

include("utils.jl")

function koopman_over_approx(x, u, f, lifting, domain_X::HRepresentation, domain_U::HRepresentation; template_W=nothing, maxdegree_certificate=nothing, cone_type = "sos")
    # Compute an affine lifted system using sum-of-squares

    n_lifting = length(lifting)
    n_u = fulldim(domain_U)

    lifting_of_f = subs(lifting, x=>f)

    if cone_type=="dsos"
        cone=DSOSCone()
        model = Model(Gurobi.Optimizer)
    elseif cone_type=="sdsos"
        cone=SDSOSCone()
        model = Model(Gurobi.Optimizer)
    elseif cone_type=="sos"
        cone=SOSCone()
        model = Model(Mosek.Optimizer)
    else
        error("Invalid cone.")
    end

    @variable(model, A[1:n_lifting,1:n_lifting])
    @variable(model, B[1:n_lifting,1:n_u])

    Err = lifting_of_f - A*lifting - B*u

    # Define the polytope W := {w | template_W*w <= h_W }
    # By default, template_W is an axis-aligned hyperbox
    if template_W === nothing
        # Construct the H matrix of the H-rep of an hypercube, e.g., if n_lifting=2:  H_W = [1 0;-1 0; 0 1; 0 -1]
        template_W = zeros(2*n_lifting,n_lifting)
        template_W[1:2*n_lifting+2:end].=1
        template_W[2:2*n_lifting+2:end].=-1
    end

    @variable(model, h_W[1:size(template_W,1)])

    # Define a SemiAlgebraicSet defining the safe set S_x times S_u from H-representations
    H_Sx=domain_X.A
    h_Sx=domain_X.b
    H_Su=domain_U.A
    h_Su=domain_U.b
    S = intersect([@set (H_Sx*x)[i]<=h_Sx[i] for i in eachindex(h_Sx)] ...)
    S = intersect(S, [@set (H_Su*u)[i]<=h_Su[i] for i in eachindex(h_Su)] ...)

    # The error Err must be in W: H_W * Err <= h_W
    if maxdegree_certificate === nothing
        @constraint(model, [i = 1:length(h_W)], (h_W - template_W*Err)[i] in cone, domain = S) # use default maxdegree
    else
        println("CAUTION: overwritting the default value for 'maxdegree_certificate' can create an error in the function 'koopman_over_approx'.")
        @constraint(model, [i = 1:length(h_W)], (h_W - template_W*Err)[i] in cone, domain = S, maxdegree=maxdegree_certificate)
    end

    @objective(model, Min, sum(h_W))

    optimize!(model)
    println(solution_summary(model))
    println("""
            termination_status : $(termination_status(model))
            primal_status      : $(primal_status(model))
            dual_status        : $(dual_status(model))
            raw_status         : $(raw_status(model))
            """)

    if (primal_status(model) != FEASIBLE_POINT) || (dual_status(model) != FEASIBLE_POINT)
        error("Model is not feasible")
    end
    linear_system = LinearDynamics(value.(A), value.(B), hrep(template_W,value.(h_W)))

    println("linear_system:\n",linear_system)

    return linear_system, model
end
