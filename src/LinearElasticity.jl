module LinearElasticity

using Einsum
using JuAFEM
using StaticArrays
using GeometryTypes
using Makie

abstract type AbstractFEAProblem end

include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("utils.jl")
include("matrices_and_vectors.jl")
include("assemble.jl")
include("buckling.jl")
include("makie.jl")

export PointLoadCantilever, HalfMBB, CompressedBeam, InpLinearElasticity, AbstractFEAProblem, GlobalFEAInfo, ElementFEAInfo, assemble, buckling, visualize

end # module
