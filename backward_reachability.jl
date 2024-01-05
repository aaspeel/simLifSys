using DynamicPolynomials
using Gurobi
using LazySets
using Polyhedra

function compute_preset(linearSystem, safe_set, input_set, target_set)
    # compute the one-step preset.

    solver = Gurobi.Optimizer
    if isempty(target_set, solver)
        return target_set
    end

    (n_x,n_u)=size(linearSystem.B)

    # compute the Minkowski difference 'target-W' using LazySets
    target_lazy = LazySets.HPolytope(target_set)
    W_lazy = LazySets.HPolytope(linearSystem.W)
    target_minus_W_lazy = LazySets.minkowski_difference(target_lazy,W_lazy)
    target_minus_W = Polyhedra.polyhedron(target_minus_W_lazy)

    # compute the polyhedron {(x,u) | Ax+Bu is in target minus W}
    H_unconstrained_preset_xu = hrep(target_minus_W).A*[linearSystem.A linearSystem.B]
    h_unconstrained_preset_xu = hrep(target_minus_W).b
    unconstrained_preset_xu = hrep(H_unconstrained_preset_xu,h_unconstrained_preset_xu)

    # compute intersection with S x U
    preset_xu = polyhedron(intersect(unconstrained_preset_xu, safe_set*input_set))

    # project over the x dimension (i.e., eliminate the input u)
    preset = eliminate(preset_xu, n_x+1:n_x+n_u)

    #removehredundancy!(preset)

    return preset
end

function compute_presets(linearSystem, safe_set, input_set, target_set, n_steps)
    # compute multiple steps of presets
    # return a list containing the polyhedrons
    presets = Vector(undef, n_steps)
    println("Start computing $n_steps presets...")
    for k in 1:n_steps
        target_set = compute_preset(linearSystem, safe_set, input_set, target_set)
        presets[k] = target_set
        println("$k/$n_steps presets computed.")
    end
    return presets
end

function compute_implicit_presets(linearSystem, safe_set_x, input_set, target_set_x, n_steps)
    # lift the target set and compute multiple steps of presets
    # return a list of presets (in the lifted domain)
    (n_lifting, n_u) = size(linearSystem.B)
    n_x = fulldim(safe_set_x)

    # lift sets
    full_space = hrep( Array{Float64}(undef, 0, n_lifting - n_x) , Array{Float64}(undef, 0) ) # R^{n_lifting - n_x}, i.e., polytope without any constraint
    safe_set_lifted = safe_set_x*full_space
    target_set_lifted = target_set_x*full_space

    # compute lifted presets
    implicit_presets = compute_presets(linearSystem, safe_set_lifted, input_set, target_set_lifted, n_steps)

    return implicit_presets
end

function forward_image(linearSystem, x_variable, lifting, input_set, x0)

    input_set_vrep = vrep(polyhedron(input_set))
    W_vrep = vrep(polyhedron(linearSystem.W))

    z0 = convert(Array{Float64},subs(lifting, x_variable=>x0)) # z0 = lifting( x0 )

    Z1 = translate(linearSystem.B*input_set_vrep + W_vrep, linearSystem.A*z0 )

    return Z1
end
