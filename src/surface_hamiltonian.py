import jax.numpy as jnp
from jax import grad

class SurfaceHamiltonian:
    def __init__(self, q0, p0, curvature, order=3):
        self.q = jnp.array(q0)
        self.p = jnp.array(p0)
        self.curvature = curvature
        self.order = order
        self.hbar = 1.0
        self.alpha = [1.0, 0.3, 0.05]

    def H0(self, q, p):
        return 0.5 * p**2 + 0.5 * q**2 + self.curvature * q**2

    def Hn(self, q, n):
        return self.alpha[n - 1] * self.hbar**n * q**n

    def RnH(self, q, p):
        H = self.H0(q, p)
        for i in range(1, self.order + 1):
            H += self.Hn(q, i)
        return H

    def evolve(self, dt):
        dq_dt = grad(self.RnH, argnums=1)(self.q, self.p)
        dp_dt = -grad(self.RnH, argnums=0)(self.q, self.p)
        self.q += dq_dt * dt
        self.p += dp_dt * dt
        return self.q, self.p
