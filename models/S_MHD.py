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

class MHD_SETD_KT_CM_JAX(BaseModel):
    # todo: implement seperate functions for the different noise options. 
    # handle case by case considerations. 
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

        self.q0 = initial_condition(self.xx,self.yy,self.params.E,self.params.initial_condition_q)
        self.A0 = initial_condition(self.xx,self.yy,self.params.E,self.params.initial_condition_a)

        self.basis_1 = self.params["noise_magnitude_1"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_1, self.params.SALT_basis_name)
        self.basis_2 = self.params["noise_magnitude_2"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_2, self.params.SFLT_basis_name)
        self.basis_3 = self.params["noise_magnitude_3"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S_3, self.params.Forcing_basis_name)

        self.mask = dealias_mask(self.kx, self.ky,cutoff=2/3)

    def timestep_validate(self):
        if self.params.dt <= 0:
            raise ValueError(f"Time step dt must be positive, got {self.params.dt}")
        if self.params.nt * self.params.dt != self.params.tmax :
            raise ValueError(f"nt {self.params.nt} does not match tmax {self.params.tmax} and dt {self.params.dt}. ")

    def validate_params(self):
        if self.params.E < 1:
            raise ValueError(f"Number of ensemble members E must be greater than or equal to 1")
        pass
    
    def print_timestepping_methods(self):
        available_methods = ['Dealiased_SETDRK4', 'Dealiased_SETDRK4_forced']
        print(available_methods)
        return available_methods
        
    def draw_noise(self, n_steps, key1, key2, key3):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.S_1))
        dB = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S_2))
        dZ = jax.random.normal(key3, shape=(n_steps, self.params.E, self.params.S_3))
        return dW, dB, dZ
    
    def step_Dealiased_SETDRK4(self, initial_state, noise_salt, noise_sflt, noise_forcing):
        q, a = initial_state
        #q, A, kx, ky, mask, E, E_2, Q, f1, f2, f3, basis_1, basis_2, basis_3, dt, dW, dB, dZ
        q,a = SETDRK4(q,
                    a,
                    self.kx, 
                    self.ky,
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
        final_state = (q,a)
        return final_state
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
        self.timestep_validate()    

        if self.params.method in ('Dealiased_SETDRK4', 'Dealiased_SETDRK4_forced'):
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_salt[i], noise_sflt[i],noise_forcing[i])
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
        self.timestep_validate()  
        if self.params.method in ('Dealiased_SETDRK4', 'Dealiased_SETDRK4_forced'):
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_salt[i], noise_sflt[i],noise_forcing[i])
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
        self.timestep_validate()    

        def k_steps(state, noise_chunk):
            def step_fn(s, inputs):
                salt, sflt, forc = inputs
                s = self.step_Dealiased_SETDRK4(s, salt, sflt, forc)
                return s, None
            state, _ = jax.lax.scan(step_fn, state, noise_chunk)
            return state

        # reshape noise into chunks of size 
        salt_chunks = noise_salt.reshape(-1, save_every, *noise_salt.shape[1:])
        sflt_chunks = noise_sflt.reshape(-1, save_every, *noise_sflt.shape[1:])
        forc_chunks = noise_forcing.reshape(-1, save_every, *noise_forcing.shape[1:])

        def outer_scan(state, chunk):
            salt_chunk, sflt_chunk, forc_chunk = chunk
            state = k_steps(state, (salt_chunk, sflt_chunk, forc_chunk))
            return state, state  # save only every k steps

        u_out = jax.lax.scan(outer_scan,
                            initial_state,
                            (salt_chunks, sflt_chunks, forc_chunks))
        return u_out # consists of final_state, saved_states
    
def initial_condition(xx,yy,E,name):
    if name == 'random':
        import numpy as np
        N = xx.shape[0]
        ans = 0.1 * (np.random.randn(N, N))
    elif name == 'smoz':
        ans = 1/4*(2*jnp.cos(4*jnp.pi*xx*2) + jnp.cos(4*jnp.pi*yy))
    elif name == 'sinsin':
        ans = jnp.sin(2 * jnp.pi * xx ) * jnp.sin(2 * jnp.pi * yy ) + 1.5 * jnp.sin(4 * jnp.pi * xx ) * jnp.sin(4 * jnp.pi * yy ) 
    elif name == 'friedel_q':
        ans = -jnp.cos((xx + 1.4)) - jnp.cos((yy + 2.0))
    elif name== 'friedel_a':
        ans = 1/3*( jnp.cos((xx + 2.3)) + jnp.cos((yy + 6.2)) )
    elif name=='orzag_modified_q':#Bizkamp and Welter 198
        ans = -4*jnp.pi**2*(jnp.cos(2*jnp.pi*xx)+jnp.cos(2*jnp.pi*yy))#-8*jnp.pi**2*
    elif name=='orzag_modified_a':
        ans = 1/2*jnp.cos(4*jnp.pi*yy)+jnp.cos(2*jnp.pi*xx)
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
####-----------------start ETDRK methods----------------------####
####----------------------------------------------------------####

def Kassam_Trefethen(dt, Lhat, nx, M=128, R=1):
    # --- ETDRK4 φ-function precomputation using contour integral ---
    E_1 = jnp.exp(dt * Lhat)
    E_2 = jnp.exp(dt * Lhat / 2)	
    M = 64
    r = jnp.exp(2j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)  # unit circle
    LR = dt * Lhat[..., None] + r                          # shape (N,N,M)
    Q  = dt * jnp.mean((jnp.exp(LR/2) - 1) / LR, axis=-1)
    f1 = dt * jnp.mean((-4 - LR + jnp.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * jnp.mean((4 + 2*LR + jnp.exp(LR)*(-4 + 2*LR)) / LR**3, axis=-1)
    f3 = dt * jnp.mean((-4 - 3*LR - LR**2 + jnp.exp(LR)*(4 - LR)) / LR**3, axis=-1)
    return E_1, E_2, Q, f1, f2, f3

@jax.jit
def SETDRK4(q, A, kx, ky, mask, E, E_2, Q, f1, f2, f3, basis_1, basis_2, basis_3, dt, dW, dB, dZ):
    rvf1 = jnp.einsum('...ijk,...i->...jk', basis_1, dW )
    rvf2 = jnp.einsum('...ijk,...i->...jk', basis_2, dB )
    rvf3 = jnp.einsum('...ijk,...i->...jk', basis_3, dZ )
    rvf1_hat = jnp.fft.fftn(rvf1,axes=(-1,-2))#*0.1 #* mask 
    rvf2_hat = jnp.fft.fftn(rvf2,axes=(-1,-2))#*0.001 #* mask
    rvf3_hat = jnp.fft.fftn(rvf3,axes=(-1,-2))

    def Solve_psi_hat_from_q_hat(q_hat,kx,ky,mask):
        """ Solve q = Δψ for ψ in Fourier space. """
        denom = kx**2 + ky**2
        safe_denom = jnp.where(denom == 0, 1.0, denom)
        psi_hat = q_hat / safe_denom 
        psi_hat = psi_hat.at[..., 0, 0].set(0.0)  
        return psi_hat * mask  
    
    def solve_j_hat_from_A_hat(A_hat,kx,ky,mask):
        """ Solve j = -ΔA for A in Fourier space. """
        denom = kx**2 + ky**2
        j_hat = - A_hat * denom 
        return j_hat * mask  # Apply dealiasing mask
    
    def J(psi_hat,q_hat,kx,ky,mask):
        " Take psi_hat and q_hat, return J(psi,q) = psi_x q_y - psi_y q_x"
        u_hat =  1j * ky * (psi_hat) * mask
        v_hat = -1j * kx * (psi_hat) * mask
        u = jnp.fft.ifftn(u_hat,axes=(-1,-2)) 
        v = jnp.fft.ifftn(v_hat,axes=(-1,-2)) 
        q = jnp.fft.ifftn(q_hat, axes=(-1,-2)) 
        flux_x_hat = jnp.fft.fftn(u*q,axes=(-1,-2)) * mask
        flux_y_hat = jnp.fft.fftn(v*q,axes=(-1,-2)) * mask
        b = -1j * ( kx * flux_x_hat + ky * flux_y_hat )
        return b
    
    def N_RHS(q_hat,psi_hat,A_hat,j_hat,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,dt):
        """ Nonlinear part: N_1 = -J(psi,q) + J(A,j), in addition to """
        J1 = J(psi_hat+rvf1_hat, q_hat,kx,ky,mask)# salt on the psi. 
        J2 = J(A_hat, j_hat,kx,ky,mask)
        J3 = J(psi_hat+rvf1_hat, A_hat,kx,ky,mask)
        rhs1 = -J1 + J2 #+ rvf3_hat*0.01
        rhs2 = -J3 #+ rvf2_hat*500
        return rhs1*dt, rhs2*dt

    
    qhat = jnp.fft.fftn(q,axes=(-1,-2))# go to Fourier space
    Ahat = jnp.fft.fftn(A,axes=(-1,-2))# go to Fourier space
    Jhat = solve_j_hat_from_A_hat(Ahat,kx,ky,mask)
    Psihat = Solve_psi_hat_from_q_hat(qhat,kx,ky,mask)
    v = (qhat,Ahat)
    Nv1_hat, Nv2_hat = N_RHS(qhat,Psihat,Ahat,Jhat,kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,dt)
    a_1,a_2 = E_2 * qhat + Q * Nv1_hat , E_2 * Ahat + Q * Nv2_hat
    Na1_hat, Na2_hat = N_RHS(a_1,Solve_psi_hat_from_q_hat(a_1,kx,ky,mask),a_2,solve_j_hat_from_A_hat(a_2,kx,ky,mask),kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,dt)
    b_1,b_2 = E_2 * qhat + Q * Na1_hat , E_2 * Ahat + Q * Na2_hat
    Nb1_hat, Nb2_hat = N_RHS(b_1,Solve_psi_hat_from_q_hat(b_1,kx,ky,mask),b_2,solve_j_hat_from_A_hat(b_2,kx,ky,mask),kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,dt)
    c_1,c_2 = E_2 * a_1 + Q * (2 * Nb1_hat - Nv1_hat) , E_2 * a_2 + Q * (2 * Nb2_hat - Nv2_hat)
    Nc1_hat, Nc2_hat = N_RHS(c_1,Solve_psi_hat_from_q_hat(c_1,kx,ky,mask),c_2,solve_j_hat_from_A_hat(c_2,kx,ky,mask),kx,ky,mask,rvf1_hat,rvf2_hat,rvf3_hat,dt)
    qhat_next = E * qhat + Nv1_hat * f1 + (Na1_hat + Nb1_hat) * f2 + Nc1_hat * f3
    Ahat_next = E * Ahat + Nv2_hat * f1 + (Na2_hat + Nb2_hat) * f2 + Nc2_hat * f3
    q_next = jnp.fft.ifftn( qhat_next,axes=(-1,-2)) # back to real space
    A_next = jnp.fft.ifftn( Ahat_next,axes=(-1,-2)) # back to real space
    return jnp.real(q_next), jnp.real(A_next)




def dealias_mask(kx, ky,cutoff=2/3):
    """ Create a 2/3 rule dealiasing mask for 2D Fourier space. """
    kx_max = jnp.max(jnp.abs(kx))
    ky_max = jnp.max(jnp.abs(ky))
    mask = (jnp.abs(kx) <= (cutoff) * kx_max) & (jnp.abs(ky) <= (cutoff) * ky_max)
    return mask


MHD_params = {
    "equation_name": 'MHD',  
    "nx": int(64*2),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.001,       # Time step, .
    "tmax": 10.0,    # Final time 2.7 blow up
    "mu": 0.000000,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "S_1": 1,
    "S_2": 1,
    "S_3": 1,
    "E": 1,
    "nt": int(10.0/0.001),
    "noise_magnitude_1": 0.0, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.0, #0.0001, # forcing noise magnitude
    "noise_magnitude_3": 0.0,#0.1, # sflt noise magnitude
    "SALT_basis_name": 'sin_sin',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'sin_sin',#'sin_sin',
    "initial_condition_q": 'orzag_modified_q',  # Initial condition type specified by chebfuns example
    "initial_condition_a": 'orzag_modified_a',  # Initial condition type specified by chebfuns example
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}
S_MHD_params = {
    "equation_name": 'MHD',  
    "nx": int(128),        # Grid size
    "xmin": 0.0,     # Minimum x-coordinate
    "xmax": 1.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step, .
    "tmax": 8000,    # Final time
    "mu": 0.000001,       # diffusion parameter
    "nu": 0.00000000001, #(1/nx)**4      # hyperdiffusion parameter
    "S_1": 1,
    "S_2": 1,
    "S_3": 1,
    "E": 1,
    "nt": int(8000/0.1),
    "noise_magnitude_1": 0.0001, #0.0001 # salt noise magnitude
    "noise_magnitude_2": 0.0,
    "noise_magnitude_3": 0.0,
    "SALT_basis_name": 'sin_sin',#'sin_sin',
    "SFLT_basis_name": 'sin_sin',#'sin_sin',
    "Forcing_basis_name": 'sin_sin',#'sin_sin',
    "initial_condition_q": 'friedel_q',  # Initial condition type specified by chebfuns example
    "initial_condition_a": 'friedel_a',  # Initial condition type specified by chebfuns example
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}



def derived_params(params):
    L = params["xmax"] - params["xmin"]  # Domain size
    Nt = int(params["tmax"] / params["dt"])  # Number of time steps
    dx = (params["xmax"] - params["xmin"]) / params["nx"]
    return {"L": L, "Nt": Nt, "dx": dx}

def main():
    import numpy as np
    from matplotlib.animation import FuncAnimation

    params = ConfigDict(MHD_params)
    signal_model = MHD_SETD_KT_CM_JAX(params) 
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

    q_ic = initial_condition(xx,yy,E,name='orzag_modified_q')#*0 
    A_ic = initial_condition(xx,yy,E,name='orzag_modified_a')#*0.000001# this gives vortex sheets and blow up as expected.
    # Plot the basis functions
    num_to_plot = min(9, basis_1.shape[0])  # Plot at most 9 basis functions
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(num_to_plot):
        ax = axs.flat[i]
        #print('basis[i]',basis[i].shape,basis[i].dtype,basis[i])
        im = ax.imshow(basis_1[i,:,:], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="RdBu", aspect='equal')
        ax.set_title(f"Basis {i+1}")
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()

    dWs = np.random.randn(Nt, basis_1.shape[0])* jnp.sqrt(dt)/dt 
    output = signal_model.run_2(initial_state=(q_ic, A_ic), n_steps=signal_model.params['Nt'], noise=None, key=jax.random.PRNGKey(0), save_every=1)
    (A0, B0), (A_hist, B_hist) = output

    # Convert JAX -> NumPy, drop the singleton channel axis (1 ensemble member)
    framesA = np.asarray(A_hist)[:, 0, :, :]   # (T, 128, 128)
    framesB = np.asarray(B_hist)[:, 0, :, :]   # (T, 128, 128)
    T = framesA.shape[0]
    print(T)
    fig, ax = plt.subplots()
    ax.set_title("A(t)")
    # Fix color scale across time (usually what you want)
    vmin = framesA.min()
    vmax = framesA.max()
    for i in range(T):
        if i % 10 == 0:
            plt.clf()
            plt.imshow(framesA[i], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="RdBu", aspect='equal')
            plt.pause(0.001)
            plt.draw()
    plt.show()
    for i in range(T):
        if i % 10 == 0:
            plt.clf()
            plt.imshow(framesB[i], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="PuRd", aspect='equal')
            plt.pause(0.001)
            plt.draw()
    plt.show()
    
if __name__ == "__main__":
    main()