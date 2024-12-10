import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from base import BaseModel
#from models.base import BaseModel
jax.config.update("jax_enable_x64", True)

class ETD_KT_CM_JAX_Vectorised(BaseModel):
    def __init__(self, params):

        self.params = params
        self.x = jnp.linspace(0, 1, self.params.nx, endpoint=False)
        self.dx = 1 / self.params.nx
        self.k = jnp.fft.fftfreq(self.params.nx, self.dx)
        self.L = -1j * self.k * self.params.c_0 + self.k**2 * self.params.c_2 + 1j * self.k**3 * self.params.c_3 - self.k**4 * self.params.c_4
        self.g = -0.5j * self.k * self.params.c_1

        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        self.nmax = round(self.params.tmax / self.params.dt)
        self.stochastic_basis = self.params.sigma*jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * self.x / (p + 1)) for p in range(self.params.P)])

        self.noise_key = jax.random.PRNGKey(0)

    def draw_noise(self, n_steps, key):
        dW = jax.random.normal(key, shape=(n_steps, self.params.E, self.params.P))
        return dW

    def step(self, initial_state, noise):
        v = jnp.fft.fft(initial_state, axis=1)
        v, u = step_ETDRK4(v, self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3, self.g, self.stochastic_basis, 0, noise)
        return u

    def run(self, initial_state, n_steps, noise):
        
        if noise is None:
            self.noise_key, noise_key = jax.random.split(self.noise_key)
            noise = self.draw_noise(n_steps, noise_key)

        def scan_fn(y, i):
            y_next = self.step(y, noise[i])
            return y_next, y_next
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))
        return u_out

def ic(x, E):
    """Initial condition for all ensemble members."""
    return jnp.tile(jnp.sin(2 * jnp.pi * x), (E, 1))

def Kassam_Trefethen(dt, L, nx, M=32):
    """Precompute weights for ETDRK4."""
    E = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)
    r = jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * jnp.transpose(jnp.tile(L, (M, 1))) + jnp.tile(r, (nx, 1))
    Q = dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * jnp.real(jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * jnp.real(jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * jnp.real(jnp.mean((-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=1))
    return E, E_2, Q, f1, f2, f3

def step_ETDRK4(v, E, E_2, Q, f1, f2, f3, g, stochastic_basis, sqrtdt, dW_n):
    """Perform one ETDRK4 time step with stochastic forcing for all ensemble members."""
    Nv = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(v,axis=-1))**2, axis=1)
    a = E_2 * v + Q* Nv
    Na = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(a,axis=-1))**2, axis=1)
    b = E_2 * v + Q * Na
    Nb = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(b,axis=-1))**2, axis=1)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(c,axis=-1))**2, axis=1)
    v_next = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
    u_next = jnp.real(jnp.fft.ifft(v_next, axis=1))
    stochastic_field = jnp.einsum('pd,ep->ed', stochastic_basis, dW_n)
    u_next += stochastic_field
    return jnp.fft.fft(u_next, axis=1), u_next

# Simulation parameters
KS_params = {# KS equation. 
    "c_0": 0, "c_1": 1, "c_2": 0.1, "c_3": 0.0, "c_4": 0.001,
    "nx": 256, "P": 1, "E": 1, "tmax": 64, "dt": 1 / 128, "sigma": 0.01,
}
Heat_params = {# Heat equation. 
    "c_0": 0, "c_1": 0, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "nx": 256, "P": 1, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.01,
}
Burgers_params={# Burgers equation. 
    "c_0": 0, "c_1": 1, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "nx": 256, "P": 1, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.01,
}
KDV_params = {# KdV equation. 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 0.01, "c_4": 0.0,
    "nx": 256, "P": 1, "E": 1, "tmax": 16, "dt": 1 / 128, "sigma": 0.01,
}

if __name__ == "__main__":
    params = KDV_params
    nx, P, E, tmax, dt = params["nx"], params["P"], params["E"], params["tmax"], params["dt"]
    dx = 1 / nx
    x = jnp.linspace(0, 1, nx, endpoint=False)
    u = ic(x, E)
    v = jnp.fft.fft(u, axis=1)
    k = jnp.fft.fftfreq(nx, dx)
    sqrtdt = jnp.sqrt(dt)

    L = -1j * k * params["c_0"] + k**2 * params["c_2"] + 1j * k**3 * params["c_3"] - k**4 * params["c_4"]
    g = -0.5j * k * params["c_1"]

    E_weights, E_2, Q, f1, f2, f3 = Kassam_Trefethen(dt, L, nx)
    nmax = round(tmax / dt)
    uu = jnp.zeros([E, nmax, nx])
    uu = uu.at[:, 0, :].set(u)

    stochastic_basis = 0.0001*jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * x / (p + 1)) for p in range(P)])
    key = jax.random.PRNGKey(0)
    dW = jax.random.normal(key, shape=(nmax, E, P))

    for n in range(1, nmax):
        v, u = step_ETDRK4(v, E_weights, E_2, Q, f1, f2, f3, g, stochastic_basis, sqrtdt, dW[n, :, :])
        uu = uu.at[:, n, :].set(u)
        if n % 10 == 0:  # Plot every 10 steps for better performance
            plt.clf()
            for e in range(E):
                plt.plot(x, u[e, :], label=f'Ensemble {e + 1}')
            plt.legend()
            plt.pause(0.001)

    plt.show()
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.pcolormesh(x, jnp.arange(0, nmax) * dt, uu.mean(axis=0))
    plt.colorbar(label='Mean Solution')
    plt.title("Ensemble Average Solution")
    plt.show()