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

class ETD_KT_CM_JAX_Vectorised(BaseModel):
    """Exponential Time Differencing-Kassam Trefethen-Cox Mathews JAX Vectorised class
    """
    def __init__(self, params):
        self.params = params


        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        
    def update_params(self, new_params):
        """Update the parameters of the model."""
        self.params = new_params
        self.xf = jnp.linspace(self.params.xmin, self.params.xmax, self.params.nx+1)
        self.x  = 0.5 * ( self.xf[1:] + self.xf[:-1] )
        self.k = jnp.fft.fftfreq(self.params.nx, self.dx) * 2 * jnp.pi # additional multiplication by 2pi
        self.L = -1j * self.k * self.params.c_0 + self.k**2 * self.params.c_2 + 1j * self.k**3 * self.params.c_3 - self.k**4 * self.params.c_4
        self.g = -0.5j * self.k * self.params.c_1
        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        self.stochastic_advection_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.P, self.params.Advection_basis_name)
        self.stochastic_forcing_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.S, self.params.Forcing_basis_name)
        self.A1, self.A2, self.B1, self.B2, self.B3, self.E1, self.E2, self.E3, self.E4, self.E5, self.E6, self.E7 = Complex_integration_technique(self.params.dt, self.L, self.params.nx)


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
    if name == 'sin':
        ans = jnp.sin(2 * jnp.pi * x)

    elif name == 'compact_bump':
        x_min = 0.2; x_max = 0.3
        s = (2 * (x - x_min) / (x_max - x_min)) - 1
        ans = jnp.exp(-1 / (1 - s**2)) * (jnp.abs(s) < 1)

    elif name == 'Kassam_Trefethen_KS_IC':
        ans = jnp.cos( x / 16 ) * ( 1 + jnp.sin(x / 16) )

    elif name == 'Kassam_Trefethen_KdV_IC_eq3pt1':
        A = 25
        B = 16
        ans  = 3 * A**2 * jnp.cosh( 0.5 * A * (x+2) )**-2
        ans += 3 * B**2 * jnp.cosh( 0.5 * B * (x+1) )**-2

    elif name == 'traveling_wave':
        beta = 3
        ans  =  12 * beta**2 * jnp.cosh( beta * ( x ) )**-2
        ans =  jnp.where(ans<1e-7,0,ans)# remove small values little bit wrong.
    elif name == 'new_traveling_wave':
        beta = 6*6 # ensure that this is used for x - beta t traveling wave solution. 
        ans  =  3 * beta * jnp.cosh( (jnp.sqrt(beta)/2) * ( x ) )**-2
        #ans =  jnp.where(ans<1e-8,0,ans)#
    elif name == 'steep_traveling_wave':
        beta = 8*8 # ensure that this is used for x - beta t traveling wave solution. 
        ans  =  3 * beta * jnp.cosh( (jnp.sqrt(beta)/2) * ( x ) )**-2
    elif name == 'very_steep_traveling_wave':
        beta = 9*9
        ans  =  3 * beta * jnp.cosh( (jnp.sqrt(beta)/2) * ( x ) )**-2

    elif name == 'ultra_steep_traveling_wave':
        beta = 15*15
        ans  =  3 * beta * jnp.cosh( (jnp.sqrt(beta)/2) * ( x ) )**-2

    elif name == 'gaussian':
        A = 1; x0 = 0.5; sigma = 0.1
        ans = A * jnp.exp(-((x - x0)**2) / (2 * sigma**2))

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

def Kassam_Trefethen(dt, L, nx, M=128, R=1):
    #R = 2 * jnp.max(jnp.abs(L*dt))# contain the spectrum of L.
    E_1 = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]# dt*L[:] is the center. Radius in new dim
    Q  = dt * jnp.mean( (jnp.exp(LR / 2) - 1) / LR, axis=-1)# trapesium rule performed by mean in the M variable.
    f1 = dt * jnp.mean( (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * jnp.mean( (4 + 2 * LR + jnp.exp(LR) * (-4 + 2*LR)) / LR**3, axis=-1)# 2 times the KT one. 
    f3 = dt * jnp.mean( (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=-1)
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


CGLE_params = {# Heat equation. 
    "equation_name" : 'Complex Ginzburg-Landau',  
    "xmin": 0, "xmax": 1,"nx": 256, "S": 0, "E": 1, "tmax": 16, "dt": 1 / 4,  "noise_magnitude": 0.1, "nt": int(16*4),
    "initial_condition": 'chebfun', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'fourier'
}


if __name__ == "__main__":
    #### Below is a testing script to run the code, without instantiating the class.
    jax.config.update("jax_enable_x64", True)
    
    params = CGLE_params
    cwd = os.getcwd()
    xmin, xmax, nx, P, S, E, tmax, dt = params["xmin"],params["xmax"], params["nx"], params["P"],params["S"], params["E"], params["tmax"], params["dt"]
    E = 1
    E = params["E"]
    nt = params["nt"]
    assert nt*dt == tmax, "nt must be equal to tmax/dt"
    #dx = (xmax - xmin) / nx
    #x = jnp.linspace(xmin, xmax, nx, endpoint=False)
    xf = jnp.linspace(xmin, xmax, nx+1)
    x = 0.5 * ( xf[1:] + xf[:-1] ) # cell centers
    dx = x[1]-x[0] # cell width
    u = initial_condition(x, E, params["initial_condition"])
    
    k = jnp.fft.fftfreq(nx, dx, dtype=jnp.complex128) * 2 * jnp.pi
    L = -1j * k * params["c_0"] + k**2 * params["c_2"] + 1j*k**3 * params["c_3"] - k**4 * params["c_4"]
    g = -0.5j * k * params["c_1"]

    E_1, E_2, Q, f1, f2, f3 = Kassam_Trefethen(dt, L, nx, M=64, R=10)# we rename the weights
    nt = round(tmax / dt)
    uu = jnp.zeros([E, nt, nx])
    uu = uu.at[:, 0, :].set(u)
    UU = jnp.zeros([E, nt, nx])
    UU = UU.at[:, 0, :].set(u)
    print(params["Forcing_basis_name"])
    stochastic_advection_basis = params["noise_magnitude"] * stochastic_basis_specifier(x, P, params["Advection_basis_name"])
    stochastic_forcing_basis   = params["noise_magnitude"] * stochastic_basis_specifier(x, S, params["Forcing_basis_name"])
    stochastic_advection_basis =  1 * stochastic_advection_basis # scaling the noise for testing purposes.
    stochastic_forcing_basis   =  1 * stochastic_forcing_basis # scaling the noise for testing purposes.
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    dW = jax.random.normal(key1, shape=(nt, E, P),dtype=jnp.float64)
    dZ = jax.random.normal(key2, shape=(nt, E, S),dtype=jnp.float64)
    A1, A2 = SETDRK11_params()
    B1, B2, B3 = SETDRK22_params()
    E1, E2, E3, E4, E5, E6, E7 = SETDRK33_params()
    

    u_f = u
    u1 = u
    u2 = u
    u3 = u
    # u4 = u
    u5 = u
    u6 = u
    u7 = u
    u8 = u 
    u9 = u
    u10 = u
    u11 = u

    W = jnp.cumsum(dW, axis=0)
    W = jnp.sqrt(dt) * params["noise_magnitude"] * W
    W_new = jnp.zeros([nt+1, E, P])
    W_new = W_new.at[1:,:,:].set(W)

    analytic = jnp.zeros([nt+1, E, nx])

    initial_condition_jitted = jax.jit(initial_condition, static_argnums=(1,2))# jitted with last two arguments frozen
    def compute_ans(n, x, dt, W_new, xmax, E, initial_condition):
        return initial_condition_jitted((x - 6*6 * (dt * n) - W_new[n, :, :] + xmax) % (xmax * 2) - xmax, E, initial_condition)
    
    compute_ans_vmap = vmap(compute_ans, in_axes=(0, None, None, None, None, None, None))
    
    # Generate the range of n values
    n_values = jnp.arange(nt + 1)
    # Compute analytic using the vectorized function
    if P>0:
        analytic = compute_ans_vmap(n_values, x, dt, W_new, xmax, E, 'new_traveling_wave')
    print(stochastic_forcing_basis)


    for n in range(1, nt):
        u_f = Dealiased_SETDRK4_forced(u_f,E_1, E_2, Q, f1, f2, f3, g, k, stochastic_advection_basis, dW[n, :, :], stochastic_forcing_basis, 1*dZ[n, :, :], dt, cutoff_ratio=2/3)
        #u_f = u_f + 1*stochastic_forcing_basis*dZ[n, :, :]
        # # SETDRK methods:
        #ue = analytic[n,0,:]
        # #u1 = Dealiased_SETDRK11(u1, A1, A2, g, k, stochastic_advection_basis, dW[n, :, :], dt, cutoff_ratio=2/3)
        # u2 = Dealiased_SETDRK22(u2, B1, B2, B3, g, k, stochastic_advection_basis, dW[n, :, :], dt, cutoff_ratio=2/3)
        # u3 = Dealiased_SETDRK33(u3, E1, E2, E3, E4, E5, E6, E7, g, k, stochastic_advection_basis, dW[n, :, :], dt, cutoff_ratio=2/3)
        # # u4 = Dealiased_CSSSPETDRK33(u4, A1, A2, g, k,  stochastic_advection_basis, dW[n, :, :], dt, cutoff_ratio=2/3)
        # u5 = Dealiased_SETDRK4(u5, E_1, E_2, Q, f1, f2, f3, g, k, stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        # #IFSRK methods: 
        # u6 = Dealiased_IFSRK4(u6,E_1,E_2,g,k,L,stochastic_advection_basis, dW[n, :, :],dt,cutoff_ratio=2/3)
        # u7 = Dealiased_eSSPIFSRK_P_33(u7,E_1,g,k,L,stochastic_advection_basis, dW[n, :, :],dt,cutoff_ratio=2/3)
        # u8 = Dealiased_eSSPIFSRK_P_22(u8,E_1,g,k,L,stochastic_advection_basis, dW[n, :, :],dt,cutoff_ratio=2/3)
        # ##u = Dealiased_eSSPIFSRK_P_11(u,E_1,g,k,L,stochastic_advection_basis, dW[n, :, :],dt,cutoff_ratio=2/3)
        # ##SRK methods:
        # u9 = Dealiased_SRK4(u9,L,g,k,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        # #uq = Dealiased_EM(uq,L,g,k,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        # u10 = Dealiased_SSP22(u10,L,g,k,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        # u11 = Dealiased_SSP33(u11,L,g,k,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        # uu = uu.at[:, n, :].set(u)
        #UU = UU.at[:, n, :].set(U)
        if n % 10 == 0:  # Plot every 10 steps for better performance
            plt.clf()
            for e in range(E):
                plt.plot(x,u_f[e, :],label=f'{e + 1} Forced SETDRK4', linewidth=0.5, c='b', linestyle='--', marker='o', markerfacecolor='none',markersize=1)
                #plt.plot(x, analytic[n,0,:], label=f'{e + 1} Analytic',linewidth=1,c='k',linestyle='-')
                # plt.plot(x, u2[e, :], label=f'{e + 1} SETDRK22', linewidth=0.25, c='r', linestyle='--', marker='s', markerfacecolor='none')
                # plt.plot(x, u3[e, :], label=f'{e + 1} SETDRK33', linewidth=0.25, c='g', linestyle='--', marker='^', markerfacecolor='none')

                # plt.plot(x, u5[e, :], label=f'{e + 1} SETDRK4', linewidth=0.5, c='c', linestyle='--', marker='d', markerfacecolor='none')
                # plt.plot(x, u6[e, :], label=f'{e + 1} IFSRK4', linewidth=0.5, c='y', linestyle='--', marker='p', markerfacecolor='none')
                # plt.plot(x, u7[e, :], label=f'{e + 1} eSSPIFSRK_P_33', linewidth=0.5, c='k', linestyle='--', marker='+', markerfacecolor='none')
                # plt.plot(x, u8[e, :], label=f'{e + 1} eSSPIFSRK_P_22', linewidth=0.5, c='orange', linestyle='--', marker='x', markerfacecolor='none')
                
                # plt.plot(x, u9[e, :], label=f'{e + 1} SRK4', linewidth=0.5, c='brown', linestyle='--', marker='h', markerfacecolor='none')
                # plt.plot(x, u10[e, :], label=f'{e + 1} SSP22', linewidth=0.5, c='gray', linestyle='--', marker='D', markerfacecolor='none')
                # plt.plot(x, u11[e, :], label=f'{e + 1} SSP33', linewidth=0.5, c='olive', linestyle='--', marker='X', markerfacecolor='none')
            plt.legend()
            plt.pause(0.001)
