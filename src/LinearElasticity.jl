module LinearElasticity

using Einsum
using JuAFEM
using StaticArrays

abstract type AbstractFEAProblem end

include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("utils.jl")
include("matrices_and_vectors.jl")
include("assemble.jl")
include("buckling.jl")

export PointLoadCantilever, HalfMBB, InpLinearElasticity, AbstractFEAProblem, GlobalFEAInfo, ElementFEAInfo, assemble, buckling

end # module
