import os
os.environ["JAX_ENABLE_X64"] = "true"
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel
import yaml
from jax import vmap
print(jnp.array(1.0).dtype)
from ml_collections import ConfigDict
class CGLE_SETD_KT_CM_JAX(BaseModel):
    def __init__(self, params):
        self.params = params
        self.derived_params = derived_params(params)
        self.params.L = self.derived_params["L"]
        self.params.Nt = self.derived_params["Nt"]
        self.params.dx = self.derived_params["dx"]

        self.x = jnp.linspace(self.params.xmin, self.params.xmax, self.params.nx, endpoint=False)
        self.xx, self.yy = jnp.meshgrid(self.x, self.x)
        # --- Wavenumber grid (Fourier) ---
        self.k = 2 * jnp.pi * jnp.fft.fftfreq(self.params.nx, d=self.params.dx)
        self.kx, self.ky = jnp.meshgrid(self.k, self.k)
        self.ksq = self.kx**2 + self.ky**2
        # --- Linear operator L = (1 + i alpha) Δ + 1 ---
        self.Lhat = (1 + 1j * self.params.alpha) * (-self.ksq) + 1
        self.E_1, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.Lhat, self.params.nx)

        self.psi0 = initial_condition(self.xx,self.yy,self.params.E,self.params.initial_condition)

        self.basis = self.params["noise_magnitude"]*stochastic_basis_specifier(self.xx, self.yy, self.params.S, self.params.Forcing_basis_name)


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
        available_methods = ['Dealiased_SETDRK4_forced']
        print(available_methods)
        return available_methods
        
    def draw_noise(self, n_steps, key1, key2):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.S))
        dZ = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S))
        return dW, dZ
    #############################
    # After testing: this method emerged as being useful, 
    # and is given here with additive forcing and stochastic advection options. 
    #############################
    def step_Dealiased_SETDRK4_forced(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = SETDRK4(u,
                    self.E_1, 
                    self.E_2,
                    self.Q,
                    self.f1, 
                    self.f2,
                    self.f3,
                    self.params.beta,
                    self.basis,
                    self.params.dt,
                    noise_advective,
                    noise_forcing )
        return u
    ###########################
    ##    Running options    ##
    ###########################
    def run(self, initial_state, n_steps, noise, key):
        # design choice1: specify a scan function based on a conditional,
        # this allows the jitting over a single function without the conditional statement call.
        ## design choice2: key is passed in ans creates the RNG for all time. 
        
        if noise is None:
            key, key1, key2 = jax.random.split(key, 3)
            noise_advective, noise_forcing = self.draw_noise(n_steps, key1, key2)
        else:
            noise_advective, noise_forcing = noise, noise# this assumes only one is selected.

        self.validate_params()
        self.timestep_validatate()    
        ###SETDRK###
        if self.params.method == 'Dealiased_SETDRK4_forced':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4_forced(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
    def final_time_run(self, initial_state, n_steps, noise, key):
        if noise is None:
            key, key1, key2 = jax.random.split(key, 3)
            noise_advective, noise_forcing = self.draw_noise(n_steps, key1, key2)
        else:
            noise_advective, noise_forcing = noise,noise
        self.validate_params()
        self.timestep_validatate()  
        ###SETDRK###
        if self.params.method == 'Dealiased_SETDRK4_forced':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4_forced(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        u_out,_ = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
def initial_condition(xx,yy,E,name):
    """_Initial condition specifier_

    Args:
        x (_type_): _mesh x array_
        E (_type_): _number of ensemble members to initialise_
        name (_type_): _name of initial condition_

    Returns:
        _type_: _array of shape (E,nx)
        consisting of E coppies of the initial condition specified
    """
    if name == 'random':
        import numpy as np
        N = xx.shape[0]  # Assuming xx and yy are square grids
        ans = 0.1 * (np.random.randn(N, N) + 1j * jnp.random.randn(N, N))
    elif name == 'zero':
        ans = jnp.zeros_like(xx) + 1j * jnp.zeros_like(yy)
    elif name == 'chebfun':
        ans = (xx*1j + yy) * jnp.exp(-0.03 * (xx**2 + yy**2))
    

    else:
        raise ValueError(f"Initial condition {name} not recognised")
    
    ic = jnp.tile(ans, (E, 1, 1))
    return ic

def stochastic_basis_specifier(x,y,P,name):
    """_Stochastic Basis specifier_

    Args:
        x (_type_): _mesh array_
        P (_type_): _number of basis functions_
        name (_type_): _name of ic_

    Raises:
        ValueError: _if name not specified_

    Returns:
        _array_: _(P,nx)_
    """
    if name == 'sin':
        ans = jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * (p + 1) * x/100) * jnp.sin(2 * jnp.pi * (p + 1) * y/100) for p in range(P)])
    elif name == 'sin_sin':
        ans = jnp.array([(p + 1) * jnp.sin(2 * jnp.pi * (p + 1) * x/100) * jnp.sin(2 * jnp.pi * (p + 1) * y/100) for p in range(P)])
    elif name == 'x_sin':
        ans = jnp.array([ jnp.sin(2 * jnp.pi * (p + 1) * x / 100) for p in range(P)])
    elif name == 'y_sin':
        ans = jnp.array([ jnp.sin(2 * jnp.pi * (p + 1) * y / 100) for p in range(P)])


    # elif name == 'LSP':
    #     ans = MD_ST_Basis(P,parameters,kx, ky, dt, alpha_noise, kappa)
    else:
        raise ValueError(f"Stochastic basis {name} not recognised")
    return ans


####----------------------------------------------------------####
####-----------------start ETDRK methods----------------------####
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


# @jax.jit
# def Dealiased_SETDRK4_forced(u, E, E_2, Q, f1, f2, f3, g, k, xi_p, dW_t, eta_p, dB_t, dt, cutoff_ratio=2/3):
#     dW_t = dW_t*jnp.sqrt(dt)/dt# the timestep is done by the exponential operators, and we require this for the increment,
#     dB_t = dB_t*jnp.sqrt(dt)/dt
#     RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
#     RFT = jnp.einsum('jk,lj ->lk',eta_p,dB_t)
#     hat_RFT = jnp.fft.fft(RFT, axis=-1)
#     v = jnp.fft.fft(u, axis=-1)

#     def N(_v,_RVF,g,_hat_RFT):
#         r = jnp.real( jnp.fft.ifft( _v ) )# return u in real space
#         n = (r + 2*_RVF)*r # g contains 1/2 in it. 
#         nhat = jnp.fft.fft(n , axis=-1)# go back to spectral space. 
#         n = dealias_using_k(nhat, k, cutoff_ratio=cutoff_ratio)# dealiasing in spectral space
#         ans = (g * n) - _hat_RFT 
#         return ans
    
#     Nv = N(v,RVF,g,hat_RFT) 
#     a = E_2 * v + Q * Nv
#     Na =  N(a,RVF,g,hat_RFT) 
#     b = E_2 * v + Q * Na
#     Nb =  N(b,RVF,g,hat_RFT) 
#     c = E_2 * a + Q * (2 * Nb - Nv)
#     Nc =  N(c,RVF,g,hat_RFT) 
#     v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
#     u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
#     u_next = u_next #- RFT # add the forcing term in real space.
#     return u_next

#@jax.jit
def SETDRK4(u, E, E_2, Q, f1, f2, f3, beta, basis, dt, dW, dB):
    print(u.shape)
    print(basis.shape,dW.shape,dB.shape)
    def N(in_field,beta,basis,dt,dW,dB):
        """ Nonlinear part: N(psi) = - (1 + i*beta) * |psi|^2 * psi + noise """
        psi =  jnp.fft.ifftn(in_field)# real space (complex)
        b = - (1 + 1j * beta) * jnp.abs(psi)**2 * (psi) 
        b = b + jnp.einsum('...ijk,...i->...jk', basis, dW )*1
        b = b + jnp.einsum('...ijk,...i->...jk', basis, dB )*1
        return jnp.fft.fftn( b ) # back to fourier space (complex)
    v = jnp.fft.fftn(u)
    Nv = N(v,beta,basis,dt,dW,dB) #  N takes fourier -> physical space -> nonlinarity -> fourier
    a = E_2 * v + Q * Nv
    Na = N(a,beta,basis,dt,dW,dB)
    b = E_2 * v + Q * Na
    Nb = N(b,beta,basis,dt,dW,dB)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = N(c,beta,basis,dt,dW,dB)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.fft.ifftn( v_next ) 
    return u_next


def get_2d_bj(lambdaxx, lambdayy, dtref, alpha_noise):
    # alpha is the smoothing parameter associated with cutting off the spectrum.
    root_qj = jnp.exp( alpha_noise * (lambdaxx**2+lambdayy**2)) # I have a built in imaginary that makes this positive.
    bj = root_qj*jnp.sqrt(dtref) 
    bj = bj
    return bj

def get_twod_dW(bj,kappa,M):
    import numpy as np
    nx,ny = bj.shape
    nnr = jnp.squeeze( jnp.sum( np.random.randn(nx,ny,M,kappa),3) )
    nnc = jnp.squeeze( jnp.sum( np.random.randn(nx,ny,M,kappa),3) ) # sum over kappa dimension.
    nn2 = nnr + 1j*nnc 
    ans = jnp.zeros_like(nn2)
    tmp = jnp.zeros_like(nn2)
    ans = ans.at[:,:].set( bj*nn2[:,:])
    tmp = tmp.at[:,:].set(jnp.fft.ifft2(ans[:,:]))# complex valued function, each are smooth approximations of white noise, alpha
    return tmp

def MD_ST_Basis(P, parameters, kx2d, ky2d, dt, alpha_noise,kappa):
    nx, ny = jnp.shape(kx2d)
    xi_streams = jnp.zeros([P,nx,ny])
    for p in range(0,P):
        xi_streams= xi_streams.at[p,:,:].set(parameters[p]*get_twod_dW( bj=get_2d_bj(kx2d, ky2d, dt, alpha_noise), kappa=kappa, M=1 ))
    return xi_streams




CGLE_params = {# Heat equation. 
    "equation_name" : 'Complex Ginzburg-Landau',  
    "nx": 256,        # Grid size
    "xmin": -50.0,    # Minimum x-coordinate
    "xmax": 50.0,     # Maximum x-coordinate
    "dt": 0.1,       # Time step
    "tmax": 64.0,     # Final time
    "alpha": 0.0,     # CGLE parameter
    "beta": 1.5,      # CGLE parameter	
    "S": 1,
    "E": 2,
    "nt": int(64/0.1),
    "noise_magnitude": 0.001, 
    "Forcing_basis_name": 'sin_sin',
    "initial_condition": 'chebfun',  # Initial condition type
    "method": 'Dealiased_SETDRK4_forced',  # Time-stepping method
}

def derived_params(params):
    L = params["xmax"] - params["xmin"]  # Domain size
    Nt = int(params["tmax"] / params["dt"])  # Number of time steps
    dx = (params["xmax"] - params["xmin"]) / params["nx"]
    return {"L": L, "Nt": Nt, "dx": dx}

def main():
    import numpy as np

    # --- Parameters chosen to match chebfun example---
    params = ConfigDict(CGLE_params)
    signal_model = CGLE_SETD_KT_CM_JAX(params)
    print(params['Forcing_basis_name'])
    print(signal_model.params['Forcing_basis_name']) # LSP is the name of the basis used in the example.
    #pu # LSP is the name of the basis used in the example.
    xmin, xmax, nx, S, E, tmax, dt, alpha, beta = signal_model.params["xmin"],signal_model.params["xmax"],signal_model.params["nt"],signal_model.params["S"],signal_model.params["E"],signal_model.params["tmax"],signal_model.params["dt"],signal_model.params["alpha"],signal_model.params["beta"]
    L,Nt,dx = signal_model.params["L"], signal_model.params["Nt"], signal_model.params["dx"]
    x = signal_model.x
    xx, yy = signal_model.xx, signal_model.yy
    kx, ky = signal_model.kx, signal_model.ky
    ksq = signal_model.ksq
    Lhat = signal_model.Lhat
    E_1,E_2,Q,f1,f2,f3 = signal_model.E_1, signal_model.E_2, signal_model.Q, signal_model.f1, signal_model.f2, signal_model.f3
    #signal_model.
    basis = signal_model.basis

    print(signal_model.psi0.shape)
    psi_ic = initial_condition(xx,yy,E,name=params['initial_condition']) # this is the initial condition, which is a complex array of shape (E,nx,nx)
    print(psi_ic.shape)
    # Plot the basis functions
    num_to_plot = min(9, basis.shape[0])  # Plot at most 9 basis functions
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(num_to_plot):
        ax = axs.flat[i]
        im = ax.imshow(basis[i], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="RdBu", aspect='equal')
        ax.set_title(f"Basis {i+1}")
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()
    dWs = np.random.randn(Nt, basis.shape[0])* jnp.sqrt(dt)/dt 
    output = signal_model.run(initial_state=signal_model.psi0, n_steps=signal_model.params['Nt'], noise=dWs, key=jax.random.PRNGKey(0))
    # Plot the solution of the output
    # Plot the solution of the output for the first ensemble member
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    times_to_plot = [0, signal_model.params['Nt'] // 3, 2 * signal_model.params['Nt'] // 3, signal_model.params['Nt'] - 1]
    for i, time_idx in enumerate(times_to_plot):
        ax = axs.flat[i]
        im = ax.imshow(jnp.real(output[1][time_idx, 0]), extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="seismic", aspect='equal')
        ax.set_title(f"Time step {time_idx} (Ensemble 1)")
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()

    # Create an animation of the output for the first ensemble member
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(jnp.real(output[1][0, 0]), extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="seismic", aspect='equal')
    ax.set_title("Time Evolution (Ensemble 1)")
    fig.colorbar(im, ax=ax, shrink=0.7)

    def update(frame):
        im.set_data(jnp.real(output[1][frame, 0]))
        ax.set_title(f"Time step {frame} (Ensemble 1)")
        return im,

    anim = FuncAnimation(fig, update, frames=signal_model.params['Nt'], interval=50, blit=True)
    plt.show()
    




if __name__ == "__main__":
    main()
    
