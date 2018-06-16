# LinearElasticity

This is a WIP application package for stress and buckling analysis relying heavily on JuAFEM.jl.

## To Do List
1. Finding stress tensor in each cell,
2. Building the element stress stiffness matrices,
3. Assembling the element stiffness matrices for buckling analysis,
4. Visualizing the displacement, stress and eigenmodes.

## Example

```julia
using LinearElasticity

# Define problem, can also be imported from .inp files
nels = (60,20)
sizes = (1.0,1.0)
E = 1.0;
ν = 0.3;
force = -1.0;
problem = PointLoadCantilever(nels, sizes, E, ν, force)

# Build element stiffness matrices and force vectors
einfo = ElementFEAInfo(problem);

# Assemble global stiffness matrix and force vector
ginfo = assemble(problem, einfo);

# Solve for node displacements
u = ginfo.K \ ginfo.f
```
