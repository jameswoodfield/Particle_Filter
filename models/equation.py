import os
os.environ["JAX_ENABLE_X64"] = "true"
from dataclasses import dataclass
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from .base import BaseModel
from .ETD_KT_CM_JAX_Vectorised import ETD_KT_CM_JAX_Vectorised

Solver1D = type("Solver1D", (ETD_KT_CM_JAX_Vectorised,), {})
## this represents an abstract solver for other examples for other models.

@dataclass(frozen=True)
class Equation1D(BaseModel):
    def __init__(self, params):
        self.params = params
        self.x = jnp.linspace(0, 1, self.params.nx, endpoint=False)
        self.dx = 1 / self.params.nx
        self.k = jnp.fft.fftfreq(self.params.nx, self.dx)
        self.L = -1j * self.k * self.params.c_0 + self.k**2 * self.params.c_2 + 1j * self.k**3 * self.params.c_3 - self.k**4 * self.params.c_4
        self.g = -0.5j * self.k * self.params.c_1
        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        self.stochastic_basis = self.params.sigma*jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * self.x / (p + 1)) for p in range(self.params.P)])

    def draw_noise(self, n_steps, n_ens, key):
        dW = jax.random.normal(key, shape=(n_steps, n_ens, self.params.P))
        return dW

    def step(self, initial_state, noise):
        v = jnp.fft.fft(initial_state, axis=1)
        v, u = step_ETDRK4(v, self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3, self.g, self.stochastic_basis, 0, noise)
        return u

    def run(self, initial_state, n_steps, noise):

        def scan_fn(y, i):
            y_next = self.step(y, noise[i])
            return y_next, y_next
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))
        return u_out

    def ic(self, n_ens):
        """Initial condition for all ensemble members."""
        return jnp.tile(jnp.sin(2 * jnp.pi * self.x), (n_ens, 1))
