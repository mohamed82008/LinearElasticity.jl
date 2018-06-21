## Credit to Simon Danisch for most of the following code

function AbstractPlotting.to_vertices(cells::AbstractVector{<: JuAFEM.Node{N, T}}) where {N, T}
    convert_arguments(nothing, cells)[1]
end

function AbstractPlotting.to_gl_indices(cells::AbstractVector{<: JuAFEM.Cell})
    tris = GLTriangle[]
    for cell in cells
        to_triangle(tris, cell)
    end
    tris
end

function to_triangle(tris, cell::JuAFEM.Hexahedron)
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[5]))
    push!(tris, GLTriangle(nodes[5], nodes[2], nodes[6]))

    push!(tris, GLTriangle(nodes[6], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[6], nodes[7]))

    push!(tris, GLTriangle(nodes[7], nodes[8], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[8], nodes[4]))

    push!(tris, GLTriangle(nodes[4], nodes[8], nodes[5]))
    push!(tris, GLTriangle(nodes[5], nodes[4], nodes[1]))

    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[1], nodes[4]))
end

function to_triangle(tris, cell::JuAFEM.Tetrahedron)
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[3], nodes[2]))
    push!(tris, GLTriangle(nodes[3], nodes[4], nodes[2]))
    push!(tris, GLTriangle(nodes[4], nodes[3], nodes[1]))
    push!(tris, GLTriangle(nodes[4], nodes[1], nodes[2]))
end

function to_triangle(tris, cell::JuAFEM.Quadrilateral)
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
    push!(tris, GLTriangle(nodes[3], nodes[4], nodes[1]))
end

function to_triangle(tris, cell::JuAFEM.Triangle)
    nodes = cell.nodes
    push!(tris, GLTriangle(nodes[1], nodes[2], nodes[3]))
end

function AbstractPlotting.convert_arguments(P, x::AbstractVector{<: JuAFEM.Node{N, T}}) where {N, T}
    convert_arguments(P, reinterpret(Point{N, T}, x))
end

function visualize(mesh::JuAFEM.Grid{dim}, u) where {dim}
    T = eltype(u)
    nnodes = length(mesh.nodes)
    #TODO make this work without creating a Node
    if dim == 2
        nodes = broadcast(1:nnodes, mesh.nodes) do i, node
            JuAFEM.Node(ntuple(Val{3}) do j
                if j < 3
                    node.x[j]
                else
                    zero(T)
                end
            end)
        end
        u = [u; zeros(T, 1, nnodes)]
    else
        nodes = mesh.nodes
    end

    cnode = AbstractPlotting.Node(zeros(Float32, length(mesh.nodes)))
    scene = Makie.mesh(nodes, mesh.cells, color = cnode, colorrange = (0.0, 33.0), shading = false);
    mplot = scene[end]
    displacevec = reinterpret(GeometryTypes.Vec{3, Float64}, u, (size(u, 2),))
    displace = norm.(displacevec)
    new_nodes = broadcast(1:length(mesh.nodes), nodes) do i, node
        JuAFEM.Node(ntuple(Val{3}) do j
            node.x[j] + u[j, i]
        end)
    end
    mesh!(nodes, mesh.cells, color = (:gray, 0.4))

    scatter!(Point3f0.(getfield.(new_nodes, :x)), markersize = 0.1);
    # TODO make mplot[1] = new_nodes work
    mplot.input_args[1][] = new_nodes
    # TODO make mplot[:color] = displace work
    push!(cnode, displace)
    points = reinterpret(Point{3, Float64}, nodes)
    #arrows!(points, displacevec, linecolor = (:black, 0.3))

    scene
end

function visualize(problem::LinearElasticityProblem{dim, T}, u) where {dim, T}
    mesh = problem.ch.dh.grid
    node_dofs = problem.metadata.node_dofs
    nnodes = JuAFEM.getnnodes(mesh)
    node_displacements = reshape(u[node_dofs], dim, nnodes)
    visualize(mesh, node_displacements)
end
