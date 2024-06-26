using LinearAlgebra
using Polyhedra
using DynamicPolynomials
using Gurobi
using JuMP
using Random

include("simulate.jl")
include("utils.jl")

function compute_refinement(x, domain_X, domain_U, lifting_z, linear_system_z, lifting_y, linear_system_y; template_rho_xbar=nothing, maxdegree_certificate=nothing, cone_type="sdsos")
    #=  Try to verify that lifted_system_y <= lifted_system_z, i.e., try to find an *affine refinement map* rho such that lifted_system_y <=_rho lifted_system_z.
        rho(z) := { y | Rz + W_rho } where W_rho is a H-polytope W_rho := { y | template_rho*y <= h_rho }.
        To be a refinement map, W_rho must satisfy W_rho = {0} x W_rho_xbar.
        The following H-representation is used: W_rho_xbar := {y | template_rho_xbar*y <= h_rho_xbar}
        To be a refinement map, the matric R must satisfy R = [I zeros(n_x,n_zbar) ; R_xbar_all].
    =#

    n_x = length(x)
    n_y = length(lifting_y)
    n_z = length(lifting_z)
    #n_u = size(linear_system_z.B)[2]

    n_ybar = n_y-n_x
    n_zbar = n_z-n_x

    # By default, template_rho_xbar is an axis-aligned hyperbox
    if template_rho_xbar === nothing
        # Construct the H matrix of the H-rep of an hypercube, e.g., if n_lifting=2:  H_W = [1 0;-1 0; 0 1; 0 -1]
        template_rho_xbar = zeros(2*n_ybar,n_ybar)
        template_rho_xbar[1:2*n_ybar+2:end].=1
        template_rho_xbar[2:2*n_ybar+2:end].=-1
    end

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    #set_attribute(model, "Presolve", 0)

    @variable(model, R_xbar_all[1:n_ybar,1:n_z])
    @variable(model, h_rho_xbar[1:size(template_rho_xbar,1)]) # the dimension should match the number of lines in template_rho_xbar

    #=  Add the constaint on the liftings. Two functions can be used for that:
        * add_lifting_constraint_sampling!() enforces the condition with sampling. Note that this is a necessary sufficient condition (and not a sufficient one) but it leads to linear constraints - recommended
        * add_lifting_constraint_cone!() enforces the condition using polynomial optimization. Three options can be used here:
            - cone_type="sos": uses sum-of squares.
            - cone_type="dsos": uses diagonally sum-of-squares.
            - cone_type="sdsos": uses scalled diagonally sum-of-squares.
            Note that "sos" is less conservative than "sdsos" which is less conservative than "dsos". However, "sos" leads to SDP constraints (and only few solvers can handle SDP constraints and bilinear constraints).
    =#
    if n_ybar > 0 # the constraint on the lifting holds trivially when n_ybar == 0
        #add_lifting_constraint_cone!(model, x, lifting_z, lifting_y, domain_X, template_rho_xbar, h_rho_xbar, R_xbar_all; cone_type=cone_type, maxdegree_certificate=maxdegree_certificate)
        add_lifting_constraint_sampling!(model, x, lifting_z, lifting_y, domain_X, template_rho_xbar, h_rho_xbar, R_xbar_all; n_samples=1000)
        @warn "lifting constraint enforced with sampling"
    end

    R = [I zeros(n_x,n_zbar) ; R_xbar_all]

    Ay = linear_system_y.A
    By = linear_system_y.B
    Hy = linear_system_y.W.A
    hy = linear_system_y.W.b

    # NB: Wy must be the cartesian product of a polytope in R^n_x and another one in the remaining dim. In addition, we assume that H_W is block diag with 2 blocks. The first one has size (2*n_x,n_x)
    @assert all(Hy[1:2*n_x, n_x+1:end] .== 0.0) and all(Hy[2*n_x+1:end,1:n_x] .== 0.0)
    Hy_x = Hy[1:2*n_x,1:n_x]
    hy_x = hy[1:2*n_x]
    Hy_xbar = Hy[2*n_x+1:end,n_x+1:end]
    hy_xbar = hy[2*n_x+1:end]

    Az = linear_system_z.A
    Bz = linear_system_z.B
    Hz = linear_system_z.W.A
    hz = linear_system_z.W.b
    H_domain_X = domain_X.A
    h_domain_X = domain_X.b
    H_domain_U = domain_U.A
    h_domain_U = domain_U.b

    # linear constraint
    if n_zbar > 0 # this constraint is trivial when n_zbar == 0
        @constraint(model, (Ay*R-R*Az)[:,n_x+1:n_z] .== 0 )
    end

    # containment for the components 1:n_x
    # X of the Minkowski sum of AH-polytopes: X*HPol(H,h)
    if n_ybar > 0
        X_in_x = hcat(  (Ay*R-R*Az)[1:n_x,1:n_x],
                        (By-R*Bz)[1:n_x,:],
                        Ay[1:n_x,n_x+1:end], # only if n_ybar > 0
                        I
                    )
        
        H_in_x = cat(   H_domain_X,
                        H_domain_U,
                        template_rho_xbar, # only if n_ybar > 0
                        Hy_x,
                    dims=(1,2)) # block diagonal concatenation

        h_in_x = vcat(  h_domain_X,
                        h_domain_U,
                        h_rho_xbar, # only if n_ybar > 0
                        hy_x
                    )
    else
        X_in_x = hcat(  (Ay*R-R*Az)[1:n_x,1:n_x],
                        (By-R*Bz)[1:n_x,:],
                        I
                    )
        
        H_in_x = cat(   H_domain_X,
                        H_domain_U,
                        Hy_x,
                    dims=(1,2)) # block diagonal concatenation

        h_in_x = vcat(  h_domain_X,
                        h_domain_U,
                        hy_x
                    )
    end

    X_out_x = R[1:n_x,:]
    H_out_x = Hz
    h_out_x = hz

    #=  This constaints enforces the AH-polytope containment. Two functions can be used for that:
            * 'add_AHpolytope_containment_vert!()' for vertex enumeration (which is exact) - recommended
            * 'add_AHpolytope_containmen()' which uses a sufficient condition for AH polytope containment
    =#
    add_AHpolytope_containment_vert!(model, X_in_x, H_in_x, h_in_x, X_out_x, H_out_x, h_out_x, variables_name = "_x")

    if n_ybar > 0
    # containment for the components n_x+1:n_y
        X_in_xbar = hcat(   (Ay*R-R*Az)[n_x+1:end,1:n_x],
                            (By-R*Bz)[n_x+1:end,:],
                            Ay[n_x+1:end,n_x+1:end],
                            I
                        ) # X of the Minkowski sum

        H_in_xbar = cat(    H_domain_X,
                            H_domain_U,
                            template_rho_xbar,
                            Hy_xbar,
                        dims=(1,2)) # block diagonal concatenation

        h_in_xbar = vcat(   h_domain_X,
                            h_domain_U,
                            h_rho_xbar,
                            hy_xbar
                        )

        X_out_xbar = hcat(  R[n_x+1:end,:],
                            I
                        )
        
        H_out_xbar = cat(   Hz,
                            template_rho_xbar,
                        dims=(1,2))

        h_out_xbar = vcat(  hz,
                            h_rho_xbar
                        )

        add_AHpolytope_containment_vert!(model, X_in_xbar, H_in_xbar, h_in_xbar, X_out_xbar, H_out_xbar, h_out_xbar, variables_name = "_xbar")
    end
    
    return model
end

function add_lifting_constraint_sampling!(model, x, lifting_z, lifting_y, domain_X, template_rho_xbar, h_rho_xbar, R_xbar_all; n_samples=1000)
    # add the constraint lifting_y \in rho( lifting_z ) using sampling of the domain
    
    n_x = length(x)
    n_y = length(lifting_y)

    # sample data
    samples_x = [sample_polytope(vrep(polyhedron(domain_X))) for i=1:n_samples]
    samples_y = convert(Array{Array{Float64}}, [subs(lifting_y, x=>sample_x) for sample_x in samples_x]) 
    samples_z = convert(Array{Array{Float64}}, [subs(lifting_z, x=>sample_x) for sample_x in samples_x])
    samples_x = reduce(hcat, samples_x) # n_x x n_samples
    samples_y = reduce(hcat, samples_y) # n_y x n_samples
    samples_z = reduce(hcat, samples_z) # n_z x n_samples

    # constraint on liftings
    @constraint(model, template_rho_xbar * (samples_y[n_x+1:n_y,:] - R_xbar_all*samples_z) .<= h_rho_xbar)
end

function add_lifting_constraint_cone!(model, x, lifting_z, lifting_y, domain_X, template_rho_xbar, h_rho_xbar, R_xbar_all; cone_type="sdsos", maxdegree_certificate=nothing)
    # add the constraint lifting_y\in rho( lifting_z ) using polynomial optimization (sdsos or dsos)

    n_x = length(x)
    n_y = length(lifting_y)

    # Define a SemiAlgebraicSet defining the safe set domain_X from H-representations
    H_Sx=domain_X.A
    h_Sx=domain_X.b
    S = intersect([@set (H_Sx*x)[i]<=h_Sx[i] for i in eachindex(h_Sx)] ...)

    if cone_type=="dsos"
        cone=DSOSCone()
    elseif cone_type=="sdsos"
        cone=SDSOSCone()
    elseif cone_type=="sos"
        cone=SOSCone()
    else
        error("Invalid cone.")
    end

    Err = lifting_y[n_x+1:end] - R_xbar_all*lifting_z
    if maxdegree_certificate === nothing
        @constraint(model, [i = 1:length(h_rho_xbar)], (h_rho_xbar - template_rho_xbar*Err)[i] in cone, domain = S)
    else
        println("CAUTION: overwritting the default value for 'maxdegree_certificate' can create an error in the function 'add_lifting_constraint_cone!'.")
        @constraint(model, [i = 1:length(h_rho_xbar)], (h_rho_xbar - template_rho_xbar*Err)[i] in cone, domain = S, maxdegree=maxdegree_certificate)
    end
end

function add_AHpolytope_containment!(model, X_in, H_in, h_in, X_out, H_out, h_out; variables_name="")
    # add the following AH polytope containment constraint: X_in*Pol_in includeded in X_out*Pol_in, with Pol_in={x| H_in * x <= h_in}, and Pol_out={x|H_out * x <= h_out}.

    @assert size(h_in,1) == size(H_in,1)
    @assert size(h_out,1) == size(H_out,1)
    @assert size(H_in,2) == size(X_in,2)
    @assert size(H_out,2) == size(X_out,2)
    @assert size(X_in,1) == size(X_out,1)

    q_in, n_in = size(H_in)
    q_out, n_out = size(H_out)

    # anonymous variables in case of multiple calls of this function over the same model
    Gamma = @variable(model, [1:n_out,1:n_in], base_name="Gamma"*variables_name)
    beta = @variable(model, [1:n_out], base_name = "beta"*variables_name)
    Lambda = @variable(model, [1:q_out,1:q_in], lower_bound=0, base_name = "Lambda"*variables_name)

    @constraint(model, X_in .== X_out*Gamma )
    @constraint(model, X_out*beta .== 0) # added a "." here
    @constraint(model, Lambda*H_in .== H_out*Gamma)
    @constraint(model, Lambda*h_in .<= h_out + H_out*beta) # added a "." here
end

function add_AHpolytope_containment_vert!(model, X_in, H_in, h_in, X_out, H_out, h_out; variables_name="")
    # Add an AH-polytope containment constraint to model. Use vertex enumeration of the inside H-polytope.
    # CAUTION: A specific structure is assumed for H_in

    @assert size(h_in,1) == size(H_in,1)
    @assert size(h_out,1) == size(H_out,1)
    @assert size(H_in,2) == size(X_in,2)
    @assert size(H_out,2) == size(X_out,2)
    @assert size(X_in,1) == size(X_out,1)
    
    q_in, n_in = size(H_in)
    q_out, n_out = size(H_out)
    
    # enumerate inner vertices. This is assuming that H_in has the form
    # H=[[1 0 ...    0]
    #    [-1 0 ...   0]
    #    [0 1 0 ...  0]
    #    [0 -1 0 ... 0]]
    @assert q_in == 2*n_in
    h_aux = reshape(h_in,(2,n_in))
    h_aux[2,:] = -h_aux[2,:] # flip sign of h[i] when H[i,:]=[0 ... 0 -1 0 ... 0]
    list_vertices = reshape( collect(Iterators.product(eachcol(h_aux)...)) ,2^n_in) # all combinations
    list_vertices = [collect(vertex) for vertex in list_vertices] # convert a list of tuples into a list of vectors

    for (vertex_id, vertex) in enumerate(list_vertices)
        pol_out_var = @variable(model, [1:n_out], base_name="pol_out_var_$(vertex_id)"*variables_name)
        @constraint(model, X_in*vertex .== X_out*pol_out_var )
        @constraint(model, H_out*pol_out_var .<= h_out)
    end
end

function dynamics_feasibility_zbar(Az, Ay, n_x)
    #=
    This function checks if it exists a matrix R such that
        (Ay*R - R*Az)[:,nx+1:n_z] == 0
        R[:,1:n_x] == [I 0]
    =#
    n_y = size(Ay)[1]
    n_z = size(Az)[1]
    n_ybar = n_y-n_x
    n_zbar = n_z-n_x

    model = Model(Gurobi.Optimizer)

    @variable(model, R_xbar_all[1:n_ybar,1:n_z])
    R = [I zeros(n_x,n_zbar) ; R_xbar_all]

    @constraint(model, (Ay*R-R*Az)[:,n_x+1:n_z] .== 0 )

    optimize!(model)

    model_is_feasible = is_solved_and_feasible(model; dual = true)

    if !model_is_feasible
        @warn """
            The model was not solved correctly:
            termination_status : $(termination_status(model))
            primal_status      : $(primal_status(model))
            dual_status        : $(dual_status(model))
            raw_status         : $(raw_status(model))
            """
    else
        println("""
                Solution is optimal
                primal solution: R_xbar_all = $( value.(R_xbar_all) )
                """)
    end

    return model_is_feasible
end
