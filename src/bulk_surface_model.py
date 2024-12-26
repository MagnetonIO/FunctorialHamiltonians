from dolfinx import mesh, fem, plot
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import pyvista as pv

# Create a 3D Unit Sphere Mesh and Boundary Mesh
domain = mesh.create_unit_sphere(MPI.COMM_WORLD, 20)
boundary_mesh = mesh.exterior_facet_mesh(domain)

# Define Function Spaces for Bulk and Surface
V_bulk = fem.FunctionSpace(domain, ("Lagrange", 1))
V_surf = fem.FunctionSpace(boundary_mesh, ("Lagrange", 1))

# Initial Conditions
u_bulk = fem.Function(V_bulk)
u_surf = fem.Function(V_surf)
u_bulk.interpolate(lambda x: np.full(x.shape[1], 0.1))
u_surf.interpolate(lambda x: np.full(x.shape[1], 0.0))

# Test and Trial Functions
u_bulk_trial = ufl.TrialFunction(V_bulk)
u_surf_trial = ufl.TrialFunction(V_surf)
v_bulk = ufl.TestFunction(V_bulk)
v_surf = ufl.TestFunction(V_surf)

# Parameters
D_bulk = fem.Constant(domain, PETSc.ScalarType(1.0))
D_surf = fem.Constant(boundary_mesh, PETSc.ScalarType(0.5))
k_react = fem.Constant(boundary_mesh, PETSc.ScalarType(0.05))
dt = fem.Constant(domain, PETSc.ScalarType(0.01))

# Bulk Weak Form (Time-discrete Diffusion)
F_bulk = ((u_bulk_trial - u_bulk) / dt) * v_bulk * ufl.dx \
       + D_bulk * ufl.inner(ufl.grad(u_bulk_trial), ufl.grad(v_bulk)) * ufl.dx

# Surface Weak Form (Surface Diffusion + Reaction)
F_surf = ((u_surf_trial - u_surf) / dt) * v_surf * ufl.dS \
       + D_surf * ufl.inner(ufl.grad(u_surf_trial), ufl.grad(v_surf)) * ufl.dS \
       - k_react * u_surf * v_surf * ufl.dS

# Plot Mesh
topology, cell_types, geometry = plot.create_vtk_mesh(domain, 0)
grid = pv.UnstructuredGrid(topology, cell_types, geometry)
grid.plot(show_edges=True)
