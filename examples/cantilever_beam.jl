using LinearElasticity

# 2D cantilever beam
nels = (60, 20)
sizes = (1.0, 1.0)

# 3D cantilever beam
#nels = (60, 20, 4)
#sizes = (1.0, 1.0, 1.0)

E = 1.0;
ν = 0.3;
force = 1.0;
problem = PointLoadCantilever(nels, sizes, E, ν, force);

# Build element stiffness matrices and force vectors
einfo = ElementFEAInfo(problem);
# Assemble global stiffness matrix and force vector
ginfo = assemble(problem, einfo);

# Solve for deformation
u = ginfo.K \ ginfo.f
# Visualize deformation using Makie
visualize(problem, u)
