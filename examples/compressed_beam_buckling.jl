using LinearElasticity

# 2D compressed vertical beam
nels = (2, 80)
sizes = (1.0, 1.0)

# 3D compressed vertical beam
#nels = (2, 80, 2)
#sizes = (1.0, 1.0, 1.0)

E = 1.0;
ν = 0.3;
force = 1.0;
problem = CompressedBeam(nels, sizes, E, ν, force);

# Build element stiffness matrices and force vectors
einfo = ElementFEAInfo(problem);
# Assemble global stiffness matrix and force vector
ginfo = assemble(problem, einfo);

# Solve for deformation
u = ginfo.K \ ginfo.f
# Visualize deformation using Makie
visualize(problem, u)

# Buckling analysis

K, Kσ = LinearElasticity.buckling(problem, ginfo, einfo);
using IterativeSolvers
X = rand(size(K, 1), 1)

n_mode_shapes = 10 # Can change that
r = lobpcg(Kσ.data, K.data, true, X, n_mode_shapes, tol = 1e-3, maxiter = 5000)

# Mode shape index, you can vary i from 1 to n_mode_shapes
i = 3
u = r.X[:,i];
visualize(problem, u)
