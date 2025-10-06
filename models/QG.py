import os
os.environ["JAX_ENABLE_X64"] = "true"
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel
from jax import vmap
from ml_collections import ConfigDict

class QG_SETD_KT_CM_JAX(BaseModel):
    def __init__(self, params):
        self.params = params
        self.derived_params = derived_params(params)
        self.params.L = self.derived_params["L"]
        self.params.Nt = self.derived_params["Nt"]
        self.params.dx = self.derived_params["dx"]
        self.x = jnp.linspace(self.params.xmin + 0.5 * self.params.dx, self.params.xmax - 0.5 * self.params.dx, self.params.nx)
        self.xx, self.yy = jnp.meshgrid(self.x, self.x)
        # --- Wavenumber grid (Fourier) ---
        self.k = 2 * jnp.pi * jnp.fft.fftfreq(self.params.nx, d=self.params.dx)# note no 1j i^4 = 1
        self.kx, self.ky = jnp.meshgrid(self.k, self.k)
        self.ksq = self.kx**2 + self.ky**2
        # --- Linear operator L =  Δ  ---
        self.Lhat = (-1) * (self.ksq) * self.params.mu + (-1) * (self.ksq)**2 * self.params.nu 
        self.E_1, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.Lhat, self.params.nx)

        self.psi0 = initial_condition(self.xx,self.yy,self.params.E,self.params.initial_condition)

        self.basis_1 = self.params["noise_magnitude_1"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_1, self.params.SALT_basis_name)
        self.basis_2 = self.params["noise_magnitude_2"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_2, self.params.SFLT_basis_name)
        self.basis_3 = self.params["noise_magnitude_3"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_3, self.params.Forcing_basis_name)

        self.mask = dealias_mask(self.kx, self.ky,cutoff=2/3)

    def timestep_validatate(self):
        if self.params.dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {self.params.dt}")
        if self.params.nt * self.params.dt != self.params.tmax :
            raise ValueError(f"nt {self.params.nt} does not match tmax {self.params.tmax} and dt {self.params.dt}. ")

    def validate_params(self):
        if self.params.E < 1:
            raise ValueError(f"Number of ensemble members E must be greater than or equal to 1")
        pass
    
    def print_timestepping_methods(self):
        available_methods = ['Dealiased_SETDRK4']
        print(available_methods)
        return available_methods
        
    def draw_noise(self, n_steps, key1, key2, key3):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.S_1))
        dB = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S_2))
        dZ = jax.random.normal(key3, shape=(n_steps, self.params.E, self.params.S_3))
        return dW, dB, dZ
    
    def step_Dealiased_SETDRK4(self, initial_state, noise_salt, noise_sflt, noise_forcing):
        u = initial_state
        u = SETDRK4(u,
                    self.kx, 
                    self.ky,
                    self.xx,
                    self.yy,
                    self.params.beta,
                    self.params.Ld,
                    self.mask,
                    self.E_1, 
                    self.E_2,
                    self.Q,
                    self.f1, 
                    self.f2,
                    self.f3,
                    self.basis_1,
                    self.basis_2,
                    self.basis_3,
                    self.params.dt,
                    noise_salt,
                    noise_sflt,
                    noise_forcing)
        return u
    ###########################
    ##    Running options    ##
    ###########################
    def run(self, initial_state, n_steps, noise, key):

        if noise is None:
            key, key1, key2, key3 = jax.random.split(key, 4)
            noise_salt, noise_sflt, noise_forcing = self.draw_noise(n_steps, key1, key2, key3)
        else:
            noise_salt, noise_sflt, noise_forcing = noise, noise, noise# this assumes only one is selected.

        self.validate_params()
        self.timestep_validatate()    

        if self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_salt[i],noise_forcing[i],noise_sflt[i])
                return y_next, y_next
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
    def final_time_run(self, initial_state, n_steps, noise, key):
        if noise is None:
            key, key1, key2, key3 = jax.random.split(key, 4)
            noise_salt, noise_sflt, noise_forcing = self.draw_noise(n_steps, key1, key2, key3)
        else:
            noise_salt, noise_sflt, noise_forcing = noise,noise,noise
        self.validate_params()
        self.timestep_validatate()  
        if self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_salt[i], noise_sflt[i], noise_forcing[i])
                return y_next, None
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        u_out,_ = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
    def run_2(self, initial_state, n_steps, noise, key, save_every=1):
        if noise is None:
            key, key1, key2, key3 = jax.random.split(key, 4)
            noise_salt, noise_sflt, noise_forcing = self.draw_noise(n_steps, key1, key2, key3)
        else:
            noise_salt, noise_sflt, noise_forcing = noise, noise, noise

        self.validate_params()
        self.timestep_validatate()    

        def k_steps(state, noise_chunk):
            def step_fn(s, inputs):
                adv, forc, sflt = inputs
                s = self.step_Dealiased_SETDRK4(s, adv, forc, sflt)
                return s, None
            state, _ = jax.lax.scan(step_fn, state, noise_chunk)
            return state

        # reshape noise into chunks of size 
        salt_chunks = noise_salt.reshape(-1, save_every, *noise_salt.shape[1:])
        sflt_chunks = noise_sflt.reshape(-1, save_every, *noise_sflt.shape[1:])
        forcing_chunks = noise_forcing.reshape(-1, save_every, *noise_forcing.shape[1:])

        def outer_scan(state, chunk):
            salt_chunk, sflt_chunk,forcing_chunk = chunk
            state = k_steps(state, (salt_chunk, sflt_chunk, forcing_chunk))
            return state, state  # save only every k steps

        u_out = jax.lax.scan(outer_scan,
                            initial_state,
                            (salt_chunks, sflt_chunks, forcing_chunks))
        return u_out # consists of final_state, saved_states
    
def initial_condition(xx,yy,E,name):
    if name == 'random':
        import numpy as np
        N = xx.shape[0]
        ans = 0.1 * (np.random.randn(N, N))
    elif name == 'smoz':
        ans = 1/4*(2*jnp.cos(4*jnp.pi*xx*2) + jnp.cos(4*jnp.pi*yy)) 
    elif name == 'smoz_cos':
        import numpy as np
        N = xx.shape[0]
        ans = 0.1 * (np.random.randn(N, N))
        ans = 1/4*(2*jnp.cos(4*jnp.pi*xx*2) + jnp.cos(4*jnp.pi*yy)) +  0.1 * (np.random.randn(N, N))
    elif name == 'wei_spinup':
        ans = jnp.sin(8*jnp.pi*xx)*jnp.sin(8*jnp.pi*yy)   +  0.4*jnp.cos(6*jnp.pi*xx)*jnp.cos(6*jnp.pi*yy)  \
            + 0.3*jnp.cos(10*jnp.pi*xx)*jnp.cos(4*jnp.pi*yy) + 0.02*jnp.sin(2*jnp.pi*xx) + 0.02*jnp.sin(2*jnp.pi*yy)
    elif name == 'sinsin':
        ans = jnp.sin(2 * jnp.pi * xx ) * jnp.sin(2 * jnp.pi * yy ) + 1.5 * jnp.sin(4 * jnp.pi * xx ) * jnp.sin(4 * jnp.pi * yy ) 
    else:
        raise ValueError(f"Initial condition {name} not recognised")
    ic = jnp.tile(ans, (E, 1, 1))
    return ic

def stochastic_basis_specifier(x,y,P,name):
    if name == 'sin':
        ans = jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * (p + 1) * x) * jnp.sin(2 * jnp.pi * (p + 1) * y) for p in range(P)])
    elif name == 'sin_sin':
        ans = jnp.array([(p + 1) * jnp.sin(2 * jnp.pi * (p + 1) * x) * jnp.sin(2 * jnp.pi * (p + 1) * y) for p in range(P)])
    elif name == 'x_sin':
        ans = jnp.array([ jnp.sin(2 * jnp.pi * (p + 1) * x ) for p in range(P)])
    elif name == 'y_sin':
        ans = jnp.array([ jnp.sin(2 * jnp.pi * (p + 1) * y ) for p in range(P)])
    elif name == 'none':
        ans = jnp.zeros((P, x.shape[0], x.shape[1]))
    else:
        raise ValueError(f"Stochastic basis {name} not recognised")
    return ans

def small_scale_noise_basis(XX, YY, P, scale=10.0, key=jax.random.PRNGKey(0)):
    Nx, Ny = XX.shape
    keys = jax.random.split(key, P)
    basis = jnp.stack([
        jax.random.normal(k, (Nx, Ny)) * jnp.sin(2 * jnp.pi * scale * XX) *
        jnp.sin(2 * jnp.pi * scale * YY)
        for k in keys
    ])
    return basis


####----------------------------------------------------------####
####-----------------start ETDRK method----------------------####
####----------------------------------------------------------####

def Kassam_Trefethen(dt, Lhat, nx, M=128, R=1):
    E_1 = jnp.exp(dt * Lhat)
    E_2 = jnp.exp(dt * Lhat / 2)	
    # --- ETDRK4 φ-function precomputation using contour integral ---
    M = 64
    r = jnp.exp(2j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)  # unit circle
    LR = dt * Lhat[..., None] + r                          # shape (N,N,M)
    Q  = dt * jnp.mean((jnp.exp(LR/2) - 1) / LR, axis=-1)
    f1 = dt * jnp.mean((-4 - LR + jnp.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * jnp.mean((4 + 2*LR + jnp.exp(LR)*(-4 + 2*LR)) / LR**3, axis=-1)
    f3 = dt * jnp.mean((-4 - 3*LR - LR**2 + jnp.exp(LR)*(4 - LR)) / LR**3, axis=-1)
    return E_1, E_2, Q, f1, f2, f3

@jax.jit
def SETDRK4(u, kx, ky, xx, yy, beta, Ld, mask, E, E_2, Q, f1, f2, f3, basis_1, basis_2, basis_3, dt, dW, dB, dZ):
    # this computes the solution to the SNSE, with stochastic advection, stochastic forcing and  
    rvf1 = jnp.einsum('...ijk,...i->...jk', basis_1, dW )
    rvf2 = jnp.einsum('...ijk,...i->...jk', basis_2, dB )
    rvf3 = jnp.einsum('...ijk,...i->...jk', basis_3, dZ )
    rvf1_hat = jnp.fft.fftn(rvf1,axes=(-1,-2))# salt noise
    rvf2_hat = jnp.fft.fftn(rvf2,axes=(-1,-2))# forcing noise
    rvf3_hat = jnp.fft.fftn(rvf3,axes=(-1,-2))# sflt noise

    def grad_perp(psi_hat,kx,ky,mask):
        u_hat =  1j * ky * (psi_hat) * mask
        v_hat = -1j * kx * (psi_hat) * mask
        u = jnp.fft.ifftn(u_hat,axes=(-1,-2)) 
        v = jnp.fft.ifftn(v_hat,axes=(-1,-2)) 
        return u,v
    
    def Jacobian(psi_hat,q_hat,kx,ky,mask):
        """ Nonlinear part: N(q) = div(u q)= (uq)_x+(vq)_y ,"""
        u,v = grad_perp(psi_hat,kx,ky,mask)
        q = jnp.fft.ifftn(q_hat, axes=(-1,-2))
        flux_x_hat = jnp.fft.fftn(u*q,axes=(-1,-2)) * mask
        flux_y_hat = jnp.fft.fftn(v*q,axes=(-1,-2)) * mask
        J_hat = 1j * ( kx * flux_x_hat + ky * flux_y_hat )
        return J_hat
    
    def eliptic_solver(q_hat,kx,ky,Ld):
        denom = kx**2 + ky**2 + 1.0 / Ld**2
        safe_denom = jnp.where(denom == 0, 1.0, denom)  # probably unnecessary for Ld>0
        psi_hat = q_hat / safe_denom
        return psi_hat 
    
    def N(q_hat,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,Ld):
        psi_hat = eliptic_solver(q_hat,kx,ky,Ld)
        J_hat = -Jacobian(psi_hat + rvf1_hat,q_hat + rvf2_hat,kx,ky,mask) #-Jacobian(psi_hat,q_hat+beta_y_hat+rvf2_hat*500,kx,ky,mask)# SFLT.
        beta_term_hat = 1j * kx * psi_hat * beta # = J(psi_hat,beta_y_hat = fft(beta*y_grid)).#J(\psi, \beta y) = \beta \psi_x.
        J_hat = J_hat + beta_term_hat
        b = J_hat + rvf3_hat # additive forcing 
        return b 
    
    v = jnp.fft.fftn(u,axes=(-1,-2))
    Nv = N(v,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,Ld) #  N takes fourier -> physical space -> nonlinarity -> fourier
    a = E_2 * v + Q * Nv
    Na = N(a,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,Ld)
    b = E_2 * v + Q * Na
    Nb = N(b,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,Ld)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = N(c,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,Ld)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.fft.ifftn( v_next,axes=(-1,-2)) # back to real space
    return jnp.real(u_next)

def dealias_mask(kx, ky,cutoff=2/3):
    """ Create a 2/3 rule dealiasing mask for 2D Fourier space. """
    kx_max = jnp.max(jnp.abs(kx))
    ky_max = jnp.max(jnp.abs(ky))
    mask = (jnp.abs(kx) <= (cutoff) * kx_max) & (jnp.abs(ky) <= (cutoff) * ky_max)
    return mask

QG_params_euler = {# Ld= 100000, beta = 0, gives euler, 
    "equation_name": 'QuasiGeostrophic',  
    "nx": int(128*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, too large from the theoretical perspective.
    "tmax": 160,    # Final time
    "mu": 0.00000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "beta": 0.0,    # beta plane parameter # 0 is barotropic, 1 is visible rosby waves, 5 is strong beta. on earth beta 0.5–5 
    "Ld": 1000000.0,      # deformation radiusL_d is 20–50 km, while the planetary radius is ~6370 km, 0.05 is strong deformation short rosby, 0.2 is moderate, L= large barotropic limit.
    "S_1": 1,
    "S_2": 1,
    "S_3": 4,
    "E": 1,
    "nt": int(160/0.1),
    "noise_magnitude_1": 0.0, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.0, #0.0001, # forcing noise magnitude
    "noise_magnitude_3": 0.01,#0.1, # sflt noise magnitude
    "SALT_basis_name": 'none',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'none',#'sin_sin',
    "initial_condition": 'smoz',  # Initial condition 
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}
QG_params_deformation_no_rossby = {#(0.2,0) gives deformation effects, but no Rossby waves
# #1,1 Balanced QG, Rossby waves and turbulence coexist #,(0.2,5) Strong beta-effect, zonal jet formation (inf,5)Barotropic turbulence with beta-effect (classic beta-plane jet benchmark)
    "equation_name": 'QuasiGeostrophic',  
    "nx": int(128*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, too large from the theoretical perspective.
    "tmax": 160,    # Final time
    "mu": 0.00000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "beta": 0.0,    # beta plane parameter # 0 is barotropic, 1 is visible rosby waves, 5 is strong beta. on earth beta 0.5–5 
    "Ld": 0.2,      # deformation radiusL_d is 20–50 km, while the planetary radius is ~6370 km, 0.05 is strong deformation short rosby, 0.2 is moderate, L= large barotropic limit.
    "S_1": 1,
    "S_2": 1,
    "S_3": 4,
    "E": 1,
    "nt": int(160/0.1),
    "noise_magnitude_1": 0.0, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.0, #0.0001, # forcing noise magnitude
    "noise_magnitude_3": 0.01,#0.1, # sflt noise magnitude
    "SALT_basis_name": 'none',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'none',#'sin_sin',
    "initial_condition": 'wei_spinup',#'smoz_cos',#'smoz',  # Initial condition 
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}

QG_params_balanced = {#1,1 Balanced QG, Rossby waves and turbulence coexist #,(0.2,5) Strong beta-effect, zonal jet formation (inf,5)Barotropic turbulence with beta-effect (classic beta-plane jet benchmark)
    "equation_name": 'QuasiGeostrophic',  
    "nx": int(128*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, too large from the theoretical perspective.
    "tmax": 160,    # Final time
    "mu": 0.00000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "beta": 1.0,    # beta plane parameter # 0 is barotropic, 1 is visible rosby waves, 5 is strong beta. on earth beta 0.5–5 
    "Ld": 1.0,      # deformation radiusL_d is 20–50 km, while the planetary radius is ~6370 km, 0.05 is strong deformation short rosby, 0.2 is moderate, L= large barotropic limit.
    "S_1": 1,
    "S_2": 1,
    "S_3": 1,
    "E": 1,
    "nt": int(160/0.1),
    "noise_magnitude_1": 0.000, #0.0001 salt noise magnitude
    "noise_magnitude_2": 0.00001, #0.01, #sflt noise magnitude
    "noise_magnitude_3": 0.00,#0.01,
    "SALT_basis_name": 'sin_sin',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'sin_sin',#'sin_sin',
    "initial_condition": 'smoz',  # Initial condition 
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}
QG_params_unbalanced = {#(0.2,5) Strong beta-effect, zonal jet formation 
    "equation_name": 'QuasiGeostrophic',  
    "nx": int(128*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, too large from the theoretical perspective.
    "tmax": 160,    # Final time
    "mu": 0.00000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "beta": 5.0,    # beta plane parameter # 0 is barotropic, 1 is visible rosby waves, 5 is strong beta. on earth beta 0.5–5 
    "Ld": 0.2,      # deformation radiusL_d is 20–50 km, while the planetary radius is ~6370 km, 0.05 is strong deformation short rosby, 0.2 is moderate, L= large barotropic limit.
    "S_1": 1,
    "S_2": 1,
    "S_3": 4,
    "E": 1,
    "nt": int(160/0.1),
    "noise_magnitude_1": 0.0, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.0, #0.0001, # forcing noise magnitude
    "noise_magnitude_3": 0.01,#0.1, # sflt noise magnitude
    "SALT_basis_name": 'none',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'none',#'sin_sin',
    "initial_condition": 'smoz',  # Initial condition 
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}

QG_params_beta_jet = {# (inf,5)Barotropic turbulence with beta-effect (classic beta-plane jet benchmark)
    "equation_name": 'QuasiGeostrophic',  
    "nx": int(128*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, too large from the theoretical perspective.
    "tmax": 160,    # Final time
    "mu": 0.00000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "beta": 5.0,    # beta plane parameter # 0 is barotropic, 1 is visible rosby waves, 5 is strong beta. on earth beta 0.5–5 
    "Ld": 1000000.0,      # deformation radiusL_d is 20–50 km, while the planetary radius is ~6370 km, 0.05 is strong deformation short rosby, 0.2 is moderate, L= large barotropic limit.
    "S_1": 1,
    "S_2": 1,
    "S_3": 4,
    "E": 1,
    "nt": int(160/0.1),
    "noise_magnitude_1": 0.0, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.01, #0.0001, # forcing noise magnitude
    "noise_magnitude_3": 0.00,#0.1, # sflt noise magnitude
    "SALT_basis_name": 'none',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'none',#'sin_sin',
    "initial_condition": 'smoz',  # Initial condition 
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}

def derived_params(params):
    L = params["xmax"] - params["xmin"]  # Domain size
    Nt = int(params["tmax"] / params["dt"])  # Number of time steps
    dx = (params["xmax"] - params["xmin"]) / params["nx"]
    return {"L": L, "Nt": Nt, "dx": dx}

def compute_energy(q, kx, ky, Ld):
    # FFT of q
    q_hat = jnp.fft.fftn(q)
    denom = kx**2 + ky**2 + 1.0 / Ld**2
    psi_hat = q_hat / denom
    # Energy spectrum
    energy_density = 0.5 * (kx**2 + ky**2 + 1.0 / Ld**2) * jnp.abs(psi_hat)**2
    return jnp.sum(energy_density).real

def casimir1(q):
    return jnp.sum(q).real

def casimir2(q):
    return 0.5 * jnp.sum(q**2).real

def main():
    import numpy as np
    params = ConfigDict(QG_params_deformation_no_rossby)
    signal_model = QG_SETD_KT_CM_JAX(params) 
    xmin, xmax, nx, S_1,S_2,S_3, E, tmax, dt, mu = signal_model.params["xmin"],signal_model.params["xmax"],signal_model.params["nt"],signal_model.params["S_1"],signal_model.params["S_2"],signal_model.params["S_3"],signal_model.params["E"],signal_model.params["tmax"],signal_model.params["dt"],signal_model.params["mu"]
    L,Nt,dx = signal_model.params["L"], signal_model.params["Nt"], signal_model.params["dx"]
    x = signal_model.x
    xx, yy = signal_model.xx, signal_model.yy
    kx, ky = signal_model.kx, signal_model.ky
    ksq = signal_model.ksq
    Lhat = signal_model.Lhat
    E_1,E_2,Q,f1,f2,f3 = signal_model.E_1, signal_model.E_2, signal_model.Q, signal_model.f1, signal_model.f2, signal_model.f3
    basis_1 = signal_model.basis_1
    basis_2 = signal_model.basis_2
    basis_3 = signal_model.basis_3

    q_ic = initial_condition(xx,yy,E,name=params['initial_condition']) # this is the initial condition, which is a complex array of shape (E,nx,nx)
    # Plot the basis functions
    for basis, title in zip([basis_1, basis_2, basis_3], ["Basis 1", "Basis 2", "Basis 3"]):
        num_to_plot = min(9, basis.shape[0])  # Plot at most 9 basis functions
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(num_to_plot):
            ax = axs.flat[i]
            im = ax.imshow(basis[i, :, :], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="RdBu", aspect='equal')
            ax.set_title(f"{title} {i+1}")
            fig.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.show()

    dWs = np.random.randn(Nt, basis_1.shape[0])* jnp.sqrt(dt)/dt 
    output = signal_model.run_2(initial_state=signal_model.psi0, n_steps=signal_model.params['Nt'], noise=None, key=jax.random.PRNGKey(0), save_every=16)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    times_to_plot = [0, signal_model.params['Nt'] // 3, 2 * signal_model.params['Nt'] // 3, signal_model.params['Nt'] - 1]
    for i, time_idx in enumerate(times_to_plot):
        ax = axs.flat[i]
        im = ax.imshow(jnp.real(output[1][time_idx, 0]), extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="seismic", aspect='equal')
        ax.set_title(f"Time step {time_idx} (Ensemble 1)")
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()

    u_out = output[1]   # shape (Nt, E, nx, nx)
    Ld = signal_model.params["Ld"]

  
    E_series, C1_series, C2_series = [], [], []

    for t in range(u_out.shape[0]):
        q = u_out[t, 0]   # ensemble member 0
        E_series.append(compute_energy(q, kx, ky, Ld))
        C1_series.append(casimir1(q))
        C2_series.append(casimir2(q))

    E_series = jnp.array(E_series)
    C1_series = jnp.array(C1_series)
    C2_series = jnp.array(C2_series)
    # Make relative to the initial value
    E_series_rel  = E_series  / E_series[0]
    C1_series_rel = C1_series / C1_series[0] 
    C2_series_rel = C2_series / C2_series[0]

    time = jnp.arange(u_out.shape[0]) * signal_model.params["dt"]

    plt.figure(figsize=(8, 5))
    plt.plot(time, E_series_rel, label="Energy")
    plt.plot(time, C1_series_rel, label="Casimir 1")
    plt.plot(time, C2_series_rel, label="Casimir 2")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("relative change in Energy-Casimir Evolution (Ensemble 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(jnp.real(output[1][0, 0]), extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="seismic", aspect='equal')
    ax.set_title("Time Evolution (Ensemble 1)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    subsample_rate = 1  # Adjust this value to control the speed of the animation
    subsampled_frames = jnp.arange(0, signal_model.params['Nt'], subsample_rate)

    def update(frame_idx):
        frame = subsampled_frames[frame_idx]
        im.set_data(jnp.real(output[1][frame, 0]))
        ax.set_title(f"Time step {frame} (Ensemble 1)")
        return im,

    anim = FuncAnimation(fig, update, frames=len(subsampled_frames), interval=50, blit=True)
    plt.show()
    
if __name__ == "__main__":
    main()
    
