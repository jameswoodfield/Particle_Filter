import os
os.environ["JAX_ENABLE_X64"] = "true"
from dataclasses import dataclass
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from .base import BaseModel
Solver1D = type("Solver1D", (ETD_KT_CM_JAX_Vectorised,), {})
## this represents an abstract solver for other examples for other models.

@dataclass(frozen=True)
class Equation1D(BaseModel):
    def __init__(self, params):
        self.params = params
        
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
