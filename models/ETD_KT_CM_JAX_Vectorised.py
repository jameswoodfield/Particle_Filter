import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel
from jax.config import config
config.update("jax_enable_x64", True)

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
        self.stochastic_advective_basis = self.params.sigma*jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * self.x / (p + 1)) for p in range(self.params.P)])
        self.stochastic_forcing_basis = self.params.sigma*jnp.array([1 / (s + 1) * jnp.sin(2 * jnp.pi * self.x / (s + 1)) for s in range(self.params.S)])

        self.noise_key = jax.random.PRNGKey(0)
        self.key1, self.key2 = jax.random.split(self.noise_key)


    def draw_noise(self, n_steps, key1, key2):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.P))
        dZ = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S))
        return dW, dZ

    def step(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Strang_split(u, 
                         self.E_weights, 
                         self.E_2, 
                         self.Q, 
                         self.f1, 
                         self.f2, 
                         self.f3, 
                         self.g, 
                         self.params.dt,
                         self.stochastic_advective_basis,
                         noise_advective,
                         self.stochastic_forcing_basis,
                         noise_forcing)
        return u

    def run(self, initial_state, n_steps, noise):

        if noise is None:
            self.key1, key1 = jax.random.split(self.key1, 2)
            self.key2, key2 = jax.random.split(self.key2, 2)
            noise_advective,noise_forcing = self.draw_noise(n_steps, key1, key2)

        def scan_fn(y, i):
            y_next = self.step(y, noise_advective[i], noise_forcing[i])
            return y_next, y_next
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out

def initial_condition(x, E , name):
    """_Initial condition specifier, creates a function of x, 
    and then tiles it to E ensemble members_

    Args:
        x (_type_): _mesh x array_
        E (_type_): _number of ensemble members to initialise_
        name (_type_): _name of initial condition_

    Returns:
        _type_: _array of shape (E,nx) 
        consisting of E coppies of the initial condition specified_
    """
    if name == 'sin':
        function_of_x= jnp.sin(2 * jnp.pi * x)
    elif name == 'compact_bump':
        x_min = 0.2; x_max = 0.3
        s = (2 * (x - x_min) / (x_max - x_min)) - 1
        function_of_x = jnp.exp(-1 / (1 - s**2)) * (jnp.abs(s) < 1)
    else:
        raise ValueError(f'Initial condition {name} not implemented')
    
    _ic = jnp.tile(function_of_x, (E, 1))
    return _ic

def Kassam_Trefethen(dt, L, nx, M=64,R=1):
    """ Precompute weights for use in ETDRK4.
    Hard to evaluate functions are computed by a 
    complex contour integration technique in 
    Kassam and Trefethen Siam 2005 
    @article{kassam2005fourth,
    title={Fourth-order time-stepping for stiff PDEs},
    author={Kassam, Aly-Khan and Trefethen, Lloyd N},
    journal={SIAM Journal on Scientific Computing},
    volume={26},
    number={4},
    pages={1214--1233},
    year={2005},
    publisher={SIAM}
    }"""
    E = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    #LR = dt * jnp.transpose(jnp.tile(L, (M, 1))) + jnp.tile(r, (nx, 1))
    LR = dt * L[:, None] + r[None, :] # broadcasting (nx,M)
    Q  = dt * jnp.mean( (jnp.exp(LR / 2) - 1) / LR, axis=-1)# trapesium rule performed by mean in the M variable.
    f1 = dt * jnp.mean( (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * jnp.mean( (4 + 2 * LR + jnp.exp(LR) * (-4 + 2*LR)) / LR**3, axis=-1)# 2 times the KT one. 
    f3 = dt * jnp.mean( (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=-1)
    # In the original papers code, one exploits LR being over 1j, and KS equation has real L,
    # The above helps to deal with equations like KdV, and one also can absorb the factor of 2 into f2.
    # Q  = dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=1))
    # f1 = dt * jnp.real(jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    # f2 = dt * jnp.real(jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    # f3 = dt * jnp.real(jnp.mean((-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=1))
    return E, E_2, Q, f1, f2, f3

# def step_ETDRK4_original(v, E, E_2, Q, f1, f2, f3, g, stochastic_basis, dt, dW_n):
#     """ Perform one ETDRK4 time step with stochastic forcing for all ensemble members
#     original, this is the original, could be improved upon"""

#     Nv = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(v,axis=-1))**2, axis=1)
#     a = E_2 * v + Q* Nv
#     Na = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(a,axis=-1))**2, axis=1)
#     b = E_2 * v + Q * Na
#     Nb = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(b,axis=-1))**2, axis=1)
#     c = E_2 * a + Q * (2 * Nb - Nv)
#     Nc = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(c,axis=-1))**2, axis=1)
#     v_next = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
#     u_next = jnp.real(jnp.fft.ifft(v_next, axis=1))

#     stochastic_field = jnp.einsum('pd,ep->ed', stochastic_basis, dW_n)
#     u_next += stochastic_field

#     return jnp.fft.fft(u_next, axis=1), u_next # v,u, we do a

def step_ETDRK4(u, E, E_2, Q, f1, f2, f3, g):
    """ Perform one ETDRK4 time step, following Cox and Matthews,
    precomputed weights are computed using Kassam_Trefethen, with a fixed dt.
    @article{cox2002exponential,
    title={Exponential time differencing for stiff systems},
    author={Cox, Steven M and Matthews, Paul C},
    journal={Journal of Computational Physics},
    volume={176},
    number={2},
    pages={430--455},
    year={2002},
    publisher={Elsevier}
    }"""
    # TODO: Nonlinearity, why real then square?
    # Answer: We are performing multiplication(nonlinearity) in real space, and then converting to spectral space.
    # TODO: Nonlinearity, implement an optional dealiasing module. 
    # TODO: Timestepping, implement an optional time filter exponential cutoff. 
    v= jnp.fft.fft(u, axis=1)# convert to fourier space
    # u = jnp.real(jnp.fft.ifft(v,axis=-1))
    Nv = g * jnp.fft.fft(u**2, axis=-1)# nonlinearity computed in real space.
    a = E_2 * v + Q * Nv # linearity computed in fourier space.
    Na = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(a,axis=-1))**2, axis=-1)
    b = E_2 * v + Q * Na
    Nb = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(b,axis=-1))**2, axis=-1)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = g * jnp.fft.fft(jnp.real(jnp.fft.ifft(c,axis=-1))**2, axis=-1)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

def step_Dealiased_ETDRK4(u, E, E_2, Q, f1, f2, f3, g, k, cutoff_ratio=1/3):
    v = jnp.fft.fft(u, axis=1)# get complex representation
    u_sqrd = jnp.real( jnp.fft.ifft(v, axis=-1) )**2
    u_sqrd_hat = jnp.fft.fft(u_sqrd, axis=-1)
    Nv = g * dealias_using_k(u_sqrd_hat, k, cutoff_ratio=cutoff_ratio)
    a = E_2 * v + Q * Nv
    Na = g * dealias_using_k(jnp.fft.fft( jnp.real(jnp.fft.ifft(a,axis=-1))**2, axis=-1), k, cutoff_ratio=cutoff_ratio)
    b = E_2 * v + Q * Na
    Nb = g * dealias_using_k(jnp.fft.fft( jnp.real(jnp.fft.ifft(b,axis=-1))**2, axis=-1), k, cutoff_ratio=cutoff_ratio)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = g * dealias_using_k(jnp.fft.fft( jnp.real(jnp.fft.ifft(c,axis=-1))**2, axis=-1), k, cutoff_ratio=cutoff_ratio)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next



def Muscl_Limiter(r):
    """_We implement the Koren Limiter
    @book{koren1993robust,
    title={A robust upwind discretization method for advection, diffusion and source terms},
    author={Koren, Barry},
    volume={45},
    year={1993},
    publisher={Centrum voor Wiskunde en Informatica Amsterdam}
    }_

    Args:
        r (_type_): _ratio of successive gradients_

    Returns:
        _array_: _\phi(r), implementation of the limiter function on r_
    """
    _ans =  jnp.maximum(1/3*r+2/3,0)
    _ans =  jnp.minimum(jnp.minimum(_ans,2*r),2)
    return _ans

def getEdgeGradients(f):
    """_Compute f_{i+1} - f_{i} in last axis_

    Args:
        f (_type_): _multidimensional array_

    Returns:
        _type_: _multidimensional array_
    """
    fex = (xroll(f, -1) - f)
    return fex

def DV(a,b):
    return a/(b + 1e-16)

def pos(c):
    return jnp.maximum(c, 0*c)

def neg(c):
    return jnp.minimum(c, 0*c)

def xroll(x,shift):
    return jnp.roll(a=x, shift=shift, axis=-1)

def Muscl_Sweby(f):
    # get gradients.
    fex = getEdgeGradients(f)
    # ratio of Sweby.
    rsx = DV( xroll(fex, 1), fex)
    # ratio of Roe.
    rrx = DV( fex, xroll(fex, 1) )
    # limit the ratios, acording to a limiter function
    rsx = Muscl_Limiter(rsx)
    rrx = Muscl_Limiter(rrx)
    # reconstruction 
    f_R = f + 1 / 2 * rsx * fex
    f_L = f - 1 / 2 * rrx * xroll(fex, 1)
    return f_R, f_L

def Upwind(f_R, f_L, Ufx):
    f_R = pos(Ufx) * f_R + neg(Ufx) * xroll(f_L, -1)
    return f_R

def Euler_Maruyama(q,dt,xi_p,dW_t,f_s,dZ_t):
    """_Euler Maruyama scheme
    $$ q^{n+1} = q^n - (F_{i+1/2}-F_{i-1/2}) \Delta W + f_s \Delta Z $$_

    Args:
        q (_type_): _description_
        dt (_type_): 
        xi_p (_type_): _description_
        dW_t (_type_): _description_
        f_s (_type_): _description_
        dZ_t (_type_): _description_

    Returns:
        _type_: _description_
    """
    dW_t = dW_t*jnp.sqrt(dt)
    dZ_t = dZ_t*jnp.sqrt(dt)
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t) 
    RFT = jnp.einsum('jk,lj ->lk',f_s,dZ_t)
    f_R, f_L = Muscl_Sweby(q)
    f_R = Upwind(f_R, f_L, RVF) 
    _q = q - (f_R - xroll(f_R, 1)) + RFT
    return _q

def SSP33(q,dt,xi_p,dW_t,f_s,dZ_t):
    _q1 = Euler_Maruyama(q,dt,xi_p,dW_t,f_s,dZ_t)
    _q1 = 1/3*q+2/3*Euler_Maruyama(_q1,dt,xi_p,dW_t,f_s,dZ_t)
    _q1 = 1/4*q+3/4*Euler_Maruyama(_q1,dt,xi_p,dW_t,f_s,dZ_t)
    return _q1

def Strang_split(u, E, E_2, Q, f1, f2, f3, g, dt, xi_p, dW_t, f_s, dZ_t):
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    u = step_ETDRK4(u, E, E_2, Q, f1, f2, f3, g)
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    return u

# Simulation parameters
# Simulation parameters
KS_params = {# KS equation. 
    "c_0": 0, "c_1": 1, "c_2": 0.1, "c_3": 0.0, "c_4": 0.001,
    "nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 64, "dt": 1 / 128, "sigma": 0.001,
    "ic": 'sin',
}
Heat_params = {# Heat equation. 
    "c_0": 0, "c_1": 0, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.001,
    "ic": 'sin',
}
Burgers_params={# Burgers equation. 
    "c_0": 0, "c_1": 1, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.001,
    "ic": 'sin',
}
KDV_params = {# KdV equation. 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 0.01, "c_4": 0.0,
    "nx": 256, "P": 10, "S": 9, "E": 2, "tmax": 8, "dt": 1 / 128, "sigma": 0.0001,
    "ic": 'sin',
}

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    params = KDV_params
    nx, P, S, E, tmax, dt = params["nx"], params["P"],params["S"], params["E"], params["tmax"], params["dt"]
    dx = 1 / nx
    x = jnp.linspace(0, 1, nx, endpoint=False)
    u = initial_condition(x, E, params["ic"])
    k = jnp.fft.fftfreq(nx, dx)

    L = -1j * k * params["c_0"] + k**2 * params["c_2"] + 1j * k**3 * params["c_3"] - k**4 * params["c_4"]
    g = -0.5j * k * params["c_1"]

    
    E_weights, E_2, Q, f1, f2, f3 = Kassam_Trefethen(dt, L, nx)
    nmax = round(tmax / dt)
    uu = jnp.zeros([E, nmax, nx])
    uu = uu.at[:, 0, :].set(u)

    stochastic_advection_basis = 0.01*jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * x / (p + 1)) for p in range(P)])
    stochastic_forcing_basis = 0.01*jnp.array([1 / (s + 1) * jnp.sin(2 * jnp.pi * x / (s + 1)) for s in range(S)])

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    dW = jax.random.normal(key1, shape=(nmax, E, P))
    dZ = jax.random.normal(key2, shape=(nmax, E, S))

    for n in range(1, nmax):
        u = Strang_split(u, E_weights, E_2, Q, f1, f2, f3, g, dt , stochastic_advection_basis, dW[n, :, :], stochastic_forcing_basis, dZ[n, :, :])
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