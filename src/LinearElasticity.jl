module LinearElasticity

using JuAFEM
using StaticArrays

abstract type AbstractFEAProblem end

include("grids.jl")
include("metadata.jl")
include("problem_types.jl")
include("utils.jl")
include("matrices_and_vectors.jl")
include("assemble.jl")

export PointLoadCantilever, HalfMBB, InpLinearElasticity, AbstractFEAProblem, GlobalFEAInfo, ElementFEAInfo, assemble

end # module
