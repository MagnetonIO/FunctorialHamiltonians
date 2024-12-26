from bulk_surface_model import *
from surface_hamiltonian import SurfaceHamiltonian

sh = SurfaceHamiltonian(q0=0.5, p0=0.0, curvature=0.1, order=3)

u_bulk_ = Function(V_bulk)
u_surf_ = Function(V_surf)
T = 5.0
t = 0.0

while t < T:
    solve(lhs(F_bulk) == rhs(F_bulk), u_bulk_)
    solve(lhs(F_surf) == rhs(F_surf), u_surf_)

    surface_energy = assemble(u_surf_ * ds)
    curvature_correction = sh.RnH(surface_energy, 0.0)

    u_surf_.vector()[:] += curvature_correction * dt
    u_bulk.assign(u_bulk_)
    u_surf.assign(u_surf_)
    t += dt

    plot(u_surf_, title="Surface Transport with Hamiltonian Correction")
