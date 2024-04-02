using Polyhedra
using Random
using Plots
using ConcaveHull

# structure to represent a linear system of the form x(t+1) = A*x(t)+B*u(t) + W, where W is a polytope.
struct LinearDynamics
    A
    B
    W::HRepresentation
end

function hyperbox(radius)
    # Return the H-representation of a n_dim dimensional hyperbox centered around the origin.
    
    n_dims=length(radius)
    
    halfSpaces=Array{HalfSpace{Float64,Array{Float64,1}}}(undef, 2*n_dims)
    for i=1:n_dims
        normalVector1=zeros(n_dims)
        normalVector1[i]=1
        halfSpaces[2*i-1]=HalfSpace(normalVector1, radius[i])

        normalVector2=zeros(n_dims)
        normalVector2[i]=-1
        halfSpaces[2*i]=HalfSpace(normalVector2, radius[i])
    end
    return hrep(halfSpaces)
end
    
"""
sample_polytope (not sure that is a uniform distribution over the polytope)
Description:
    This function samples a polytope as specified by polytope_in. If 'fromVertices', return one vertex.
"""
function sample_polytope( v_polytope_in::VRepresentation, fromVertices::Bool=false )
    # sample the interior of a polytope if fromVertices is false; sample vertices of the polytope if fromVertices=true.

    # To have field .V in v_polytope_in
    v_polytope_in=convert(MixedMatVRep{Float64,Array{Float64,2}},v_polytope_in)
    
    # Constants
    num_vertices = size(v_polytope_in.V,1)
    
    if fromVertices # return a random vertex
        return v_polytope_in.V[rand(1:num_vertices),:]
    else
        # Algorithm
        theta = randexp(Float64,num_vertices)
        theta = theta/sum(theta) # This vector sums to one. Normalized exponential random variables are uniformly distributed over the symplex.

        # Return our sample a random and convex combination of the vertices
        return transpose(v_polytope_in.V)*theta
    end
end

function grid_polytope(pol::HRepresentation, n_steps=10)
    @assert pol.A == [1 0; -1 0; 0 1; 0 -1] # must be hyperbox in canonical form
    h = pol.b
    x=range(-h[2],h[1],n_steps)
    y=range(-h[4],h[3],n_steps)

    samples = reshape([[x[i],y[j]] for i=1:n_steps,  j=1:n_steps],n_steps^2)
    return samples
end


function plot_implicit_representation!(current_plot, x, domain_x, lifted_set, lifting, n_samples=1000; kwargs...)
    # plot of an implicit representation: { x\in domain_x | lifting(x)\in lifted_set }. Use sample rejection (with n_samples).

    if domain_x.A == [1 0; -1 0; 0 1; 0 -1] # must be hyperbox in canonical form
        samples_x = grid_polytope(domain_x, ceil(Int64,sqrt(n_samples)))
    else
        println("CAUTION: random sampling was used (instead of grid sampling) for plot_implicit_representation!() because domain_x was not a 2D hyperbod in canonical form")
        samples_x = [sample_polytope(vrep(polyhedron(domain_x))) for i=1:n_samples]
    end
    n_samples=length(samples_x)

    samples_z = convert(Array{Array{Float64}},[subs(lifting, x=>sample_x) for sample_x in samples_x])

    ind = in.(samples_z,lifted_set)

    samples_x = mapreduce(permutedims, vcat, samples_x) # convert Vector of Vector by Matrix

    scatter!(current_plot, samples_x[ind,1], samples_x[ind,2], markerstrokewidth = 0; kwargs...)
end

function plot_implicit_representation_hull!(current_plot, x, domain_x, lifted_set, lifting, n_samples=1000, knn_param=50; kwargs...)
    # plot the concave hull of the implicit representation.

    if domain_x.A == [1 0; -1 0; 0 1; 0 -1] # must be hyperbox in canonical form
        samples_x = grid_polytope(domain_x, ceil(Int64,sqrt(n_samples)))
    else
        println("CAUTION: random sampling was used (instead of grid sampling) for plot_implicit_representation!() because domain_x was not a 2D hyperbod in canonical form")
        samples_x = [sample_polytope(vrep(polyhedron(domain_x))) for i=1:n_samples]
    end
    n_samples=length(samples_x)

    samples_z = convert(Array{Array{Float64}},[subs(lifting, x=>sample_x) for sample_x in samples_x])

    ind = in.(samples_z,lifted_set)

    hull = concave_hull(samples_x[ind],knn_param) # second argument is the number of points used in knn algorithm
    plot!(current_plot, hull; kwargs...)
end

function plot_implicit_representations_hull!(current_plot, x, domain_x, lifted_sets, lifting, samples_x=nothing, n_samples=1000, knn_param=50; kwargs...)
    # plot the concave hull of multiple implicit representations (using the same samples)

    if samples_x === nothing
        if domain_x.A == [1 0; -1 0; 0 1; 0 -1] # must be hyperbox in canonical form
            samples_x = grid_polytope(domain_x, ceil(Int64,sqrt(n_samples)))
        else
            println("CAUTION: random sampling was used (instead of grid sampling) for plot_implicit_representations_hull!() because domain_x was not a 2D hyperbod in canonical form")
            samples_x = [sample_polytope(vrep(polyhedron(domain_x))) for i=1:n_samples]
        end
    end
    n_samples=length(samples_x)

    samples_z = convert(Array{Array{Float64}},[subs(lifting, x=>sample_x) for sample_x in samples_x])

    n_sets = length(lifted_sets)

    # find the indices of the samples that are in at least one lisfted set.
    ind=BitArray(zeros(n_samples)) # list of indices if z that are contained in one of the lifted sets
    for i=1:n_sets
        new_ind = in.(samples_z,lifted_sets[i])
        ind = any( [ind new_ind], dims=2)
    end
    ind=ind[:,1] # to get a vector instead of a matrix

    # plot only if there are more that 2 points (concave huull not defined otherwise)
    if sum(ind)>2
        print("Computing concave hull... ")
        hull = concave_hull(samples_x[ind],knn_param)
        println("Done.")
        plot!(current_plot, hull; kwargs...)
        
        # Display an extra plot to check if the concave hull is representative of the samples
        samples_x = mapreduce(permutedims, vcat, samples_x) # convert Vector of Vector by Matrix
        plot_check = plot(polyhedron(domain_x), alpha=0.1)
        scatter!(plot_check, samples_x[ind,1], samples_x[ind,2], markerstrokewidth = 0; kwargs...)
        plot!(plot_check, hull; kwargs)
        display(plot_check)
    else
        println("Concave hull not plotted! Concave hull cannot be computed from less than 3. There was $(sum(ind)) points.")
    end
end