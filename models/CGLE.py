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
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.P))
        dZ = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S))
        return dW, dZ
    #############################
    # After testing: this method emerged as being useful, 
    # and is given here with additive forcing and stochastic advection options. 
    #############################
    def step_Dealiased_SETDRK4_forced(self,initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SETDRK4_forced(u, 
                                    self.E_weights, 
                                    self.E_2,
                                    self.Q,
                                    self.f1, 
                                    self.f2,
                                    self.f3,
                                    self.g,
                                    self.k,
                                    self.stochastic_advection_basis,
                                    noise_advective,
                                    self.stochastic_forcing_basis,
                                    noise_forcing,
                                    self.params.dt)
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
    
    ic = jnp.tile(ans, (E, 1))
    return ic

def stochastic_basis_specifier(x, P, name):
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
        ans = jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * x * (p + 1)) for p in range(P)])
    elif name =='none':
        ans = jnp.zeros([P,x.shape[0]])
    elif name == 'cos':
        ans = jnp.zeros((P, nx))
        for p in range(P):
            ans = ans.at[p, :].set(jnp.cos((p+1)*2*jnp.pi*x))
    elif name == 'random':
        ans = jax.random.normal(jax.random.PRNGKey(0), (P, len(x)))

    elif name == 'constant':
        nx = len(x)
        ans = jnp.ones((P, nx))# each basis function is a constant
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
    # R = 2 * jnp.max(jnp.abs(L*dt))# contain the spectrum of L.
    # E_1 = jnp.exp(dt * L)
    # E_2 = jnp.exp(dt * L / 2)
    # r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    # LR = dt * L[:, None] + r[None, :]# dt*L[:] is the center. Radius in new dim
    # Q  = dt * jnp.mean( (jnp.exp(LR / 2) - 1) / LR, axis=-1)# trapesium rule performed by mean in the M variable.
    # f1 = dt * jnp.mean( (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1)
    # f2 = dt * jnp.mean( (4 + 2 * LR + jnp.exp(LR) * (-4 + 2*LR)) / LR**3, axis=-1)# 2 times the KT one. 
    # f3 = dt * jnp.mean( (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=-1)
    return E_1, E_2, Q, f1, f2, f3


@jax.jit
def Dealiased_SETDRK4_forced(u, E, E_2, Q, f1, f2, f3, g, k, xi_p, dW_t, eta_p, dB_t, dt, cutoff_ratio=2/3):
    dW_t = dW_t*jnp.sqrt(dt)/dt# the timestep is done by the exponential operators, and we require this for the increment,
    dB_t = dB_t*jnp.sqrt(dt)/dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    RFT = jnp.einsum('jk,lj ->lk',eta_p,dB_t)
    hat_RFT = jnp.fft.fft(RFT, axis=-1)
    v = jnp.fft.fft(u, axis=-1)

    def N(_v,_RVF,g,_hat_RFT):
        r = jnp.real( jnp.fft.ifft( _v ) )# return u in real space
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        nhat = jnp.fft.fft(n , axis=-1)# go back to spectral space. 
        n = dealias_using_k(nhat, k, cutoff_ratio=cutoff_ratio)# dealiasing in spectral space
        ans = (g * n) - _hat_RFT 
        return ans
    
    Nv = N(v,RVF,g,hat_RFT) 
    a = E_2 * v + Q * Nv
    Na =  N(a,RVF,g,hat_RFT) 
    b = E_2 * v + Q * Na
    Nb =  N(b,RVF,g,hat_RFT) 
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc =  N(c,RVF,g,hat_RFT) 
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    u_next = u_next #- RFT # add the forcing term in real space.
    return u_next






def SETDRK4(u, E, E_2, Q, f1, f2, f3, beta, basis, dt, dW, dB):
    def N(in_field,beta,basis,dt,dW,dB):
        """ Nonlinear part: N(psi) = - (1 + i*beta) * |psi|^2 * psi + noise """
        psi =  jnp.fft.ifftn(in_field)# real space (complex)
        b = - (1 + 1j * beta) * jnp.abs(psi)**2 * (psi) 
        b = b + jnp.einsum('ijk,i->jk', basis, dW )*1
        b = b + jnp.einsum('ijk,i->jk', basis, dB )*1
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
    "S": 0,
    "E": 1,
    "nt": int(64/0.1),
    "noise_magnitude": 0.1, 
    "Forcing_basis_name": 'LSP',
    "initial_condition": 'chebfun',  # Initial condition type
    "method": 'Dealiased_SETDRK4'
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
    xmin, xmax, nx, S, E, tmax, dt, alpha, beta = signal_model.params["xmin"],signal_model.params["xmax"],signal_model.params["nt"],signal_model.params["S"],signal_model.params["E"],signal_model.params["tmax"],signal_model.params["dt"],signal_model.params["alpha"],signal_model.params["beta"]
    L,Nt,dx = signal_model.params["L"], signal_model.params["Nt"], signal_model.params["dx"]
    x = signal_model.x
    xx, yy = signal_model.xx, signal_model.yy
    kx, ky = signal_model.kx, signal_model.ky
    ksq = signal_model.ksq
    Lhat = signal_model.Lhat
    E_1,E_2,Q,f1,f2,f3 = signal_model.E_1, signal_model.E_2, signal_model.Q, signal_model.f1, signal_model.f2, signal_model.f3
    
    np.random.seed(1)
    #psi0 = 0.1 * (np.
    psi = signal_model.psi0

    # --- Time loop ---
    t = 0.0
    max_mode = 9 # number of modes in the basis. 
    P = max_mode
    #x, basis = real_fourier_basis_2d(N, xmin, xmax, max_mode)
    parameters = 10**-5 * jnp.asarray(jnp.ones(P)); 
    alpha_noise = 0.25
    kappa = 5# terms used in the generation of white noise process.
    basis = MD_ST_Basis(P,parameters,kx, ky, dt, alpha_noise, kappa)
    basis = basis / jnp.linalg.norm(basis, axis=(1, 2), keepdims=True)*1# normalise basis. 
    basis = basis*0
    num_to_plot = min(9, basis.shape[0])  # plot at most 9
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(num_to_plot):
        ax = axs.flat[i]
        im = ax.imshow(basis[i], extent=[xmin, xmax, xmin, xmax], origin="lower", cmap="RdBu", aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.show()

    dWs = np.random.randn(Nt, basis.shape[0])
    dBs = np.random.randn(Nt, basis.shape[0])
    # in our solver we require a rescaling of the noise by 1/dt 
    dWs = dWs * jnp.sqrt(dt)/dt  # Scale by sqrt(dt) for Wiener increments
    dBs = dBs * jnp.sqrt(dt)/dt  # Scale by sqrt(dt) for Wiener increments 
    print("Shape of dWs:", dWs.shape)
    fig = plt.figure(figsize=(4, 4), dpi=150)	
    for i in range(Nt):
        psi = SETDRK4(psi, E_1, E_2, Q, f1, f2, f3, beta, basis, dt, dWs[i], dBs[i])	
        t += dt	
        # Save the state at specific times
        if i in [0, Nt // 3, 2 * Nt // 3, Nt - 1]:
            plt.clf()
            plt.suptitle(f"Time = {t:.2f}", fontsize=14)
            plt.title("Real part", fontsize=10)
            im = plt.imshow(jnp.real(psi), cmap='seismic', origin='lower',extent=[xmin, xmax, xmin, xmax])
            im.set_clim(-1, 1)  # Set color scale to be fixed between -1 and 1
            plt.colorbar(im, shrink=0.6)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.pause(0.001)
            #plt.savefig(f"psi_snapshot_t{i}_real.", np.real(psi))
            # plt.savefig(f"psi_snapshot_t{i}_imag.npy", np.imag(psi))
        if i % 1 == 0:
            plt.clf()
            plt.suptitle(f"Time = {t:.2f}", fontsize=14)
            plt.title("Real part", fontsize=10)
            im = plt.imshow( np.real(psi), cmap='seismic', origin='lower',extent=[xmin, xmax, xmin, xmax])
            im.set_clim(-1, 1)  # Set color scale to be fixed between -1 and 1
            plt.colorbar(im, shrink=0.6)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.pause(0.001)
    plt.show(block=True)
    #plt.savefig("SETDRK4_2D_CGLE.png", dpi=300)

if __name__ == "__main__":
    main()
    
