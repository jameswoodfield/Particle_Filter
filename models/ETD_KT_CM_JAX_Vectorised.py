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

        self.xf = jnp.linspace(self.params.xmin, self.params.xmax, self.params.nx+1)
        self.x  = 0.5 * ( self.xf[1:] + self.xf[:-1] ) # cell centers
        self.dx = self.x[1]-self.x[0] # cell width
        #self.x = jnp.linspace(self.params.xmin, self.params.xmax , self.params.nx, endpoint=False)
        #self.dx = (self.params.xmax - self.params.xmin) / self.params.nx

        self.k = jnp.fft.fftfreq(self.params.nx, self.dx) * 2 * jnp.pi # additional multiplication by 2pi
        self.L = -1j * self.k * self.params.c_0 + self.k**2 * self.params.c_2 + 1j * self.k**3 * self.params.c_3 - self.k**4 * self.params.c_4
        self.g = -0.5j * self.k * self.params.c_1

        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        
        self.stochastic_advection_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.P, self.params.Advection_basis_name)
        self.stochastic_forcing_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.S, self.params.Forcing_basis_name)
        self.A1, self.A2, self.B1, self.B2, self.B3, self.E1, self.E2, self.E3, self.E4, self.E5, self.E6, self.E7 = Complex_integration_technique(self.params.dt, self.L, self.params.nx)

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
        available_methods = ['Dealiased_SETDRK4_forced',
            'Dealiased_ETDRK4',
            'Dealiased_SETDRK4',
            'Dealiased_SETDRK33',
            'Dealiased_SETDRK22',
            'Dealiased_IFSRK4',
            'Dealiased_eSSPIFSRK_P_33',
            'Dealiased_eSSPIFSRK_P_22',
            'Dealiased_SRK4',
            'Dealiased_SSP33',
            'Dealiased_SSP22']
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
    #  Time stepping methods  #
    ###########################
    def step_Dealiased_ETDRK4(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_ETDRK4(u, 
                   self.E_weights, 
                   self.E_2, 
                   self.Q, 
                   self.f1, 
                   self.f2, 
                   self.f3, 
                   self.g,
                   self.k)
        return u
    ############################
    #  Dealiased SETDRK methods #
    ############################
    def step_Dealiased_SETDRK4(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SETDRK4(u,
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
                            self.params.dt
                            )
        return u
    def step_Dealiased_SETDRK33(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SETDRK33(u,
                            self.E1,
                            self.E2,
                            self.E3, 
                            self.E4,
                            self.E5, 
                            self.E6, 
                            self.E7,
                            self.g,
                            self.k,
                            self.stochastic_advection_basis,
                            noise_advective,
                            self.params.dt
                            )
        return u
    def step_Dealiased_CSSSPETDRK33(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_CSSSPETDRK33(u,
                            self.A1,
                            self.A2,
                            self.g,
                            self.k,
                            self.stochastic_advection_basis,
                            noise_advective,
                            self.params.dt
                            )
        return u
    def step_Dealiased_SETDRK22(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SETDRK22(u,
                            self.B1,
                            self.B2,
                            self.B3,
                            self.g,
                            self.k,
                            self.stochastic_advection_basis,
                            noise_advective,
                            self.params.dt
                            )
        return u
    def step_Dealiased_SETDRK11(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SETDRK11(u,
                            self.A1,
                            self.A2,
                            self.g,
                            self.k,
                            self.stochastic_advection_basis,
                            noise_advective,
                            self.params.dt
                            )
        return u
    ############################
    #  Dealiased IFSRK methods  #
    ############################
    def Dealiased_IFSRK4(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_IFSRK4(u,
                             self.E_weights,
                             self.E_2,
                             self.g,
                             self.k,
                             self.L,
                             self.stochastic_advection_basis,
                             noise_advective,
                             self.params.dt
                             )
        return u
    def Dealiased_eSSPIFSRK_P_33(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_eSSPIFSRK_P_33(u,
                                     self.E_weights,# loss of E_2
                                     self.g,
                                     self.k,
                                     self.L,
                                     self.stochastic_advection_basis,
                                     noise_advective,
                                     self.params.dt
                                     )
        return u
    def Dealiased_eSSPIFSRK_P_22(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_eSSPIFSRK_P_22(u,
                                     self.E_weights,
                                     self.g,
                                     self.k,
                                     self.L,
                                     self.stochastic_advection_basis,
                                     noise_advective,
                                     self.params.dt
                                     )
        return u
    def Dealiased_eSSPIFSRK_P_11(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_eSSPIFSRK_P_11(u,
                                     self.E_weights,# loss of E_2
                                     self.g,
                                     self.k,
                                     self.L,
                                     self.stochastic_advection_basis,
                                     noise_advective,
                                     self.params.dt
                                     )
        return u
    ############################
    #  Dealiased SRK methods  #
    ############################
    def Dealiased_SRK4(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SRK4(u,
                 self.L,
                 self.g,
                 self.k,
                 self.stochastic_advection_basis,
                 noise_advective,
                 self.params.dt
                 )
        return u
    def Dealiased_SSP33(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SSP33(u,
                 self.L,
                 self.g,
                 self.k,
                 self.stochastic_advection_basis,
                 noise_advective,
                 self.params.dt
                 )
        return u
    def Dealiased_SSP22(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_SSP22(u,
                 self.L,
                 self.g,
                 self.k,
                 self.stochastic_advection_basis,
                 noise_advective,
                 self.params.dt
                 )
        return u
    def Dealiased_EM(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_EM(u,
                 self.L,
                 self.g,
                 self.k,
                 self.stochastic_advection_basis,
                 noise_advective,
                 self.params.dt
                 )
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
        if self.params.method == 'Dealiased_ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SETDRK4_forced':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4_forced(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SETDRK33':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK33(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SETDRK22':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK22(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SETDRK11':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK11(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        ###IFSTK###
        elif self.params.method == 'Dealiased_IFSRK4':
            def scan_fn(y,i):
                y_next = self.Dealiased_IFSRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_33':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_33(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_22':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_22(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_11':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_11(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        
        ###SRK###
        elif self.params.method == 'Dealiased_SRK4':
            def scan_fn(y,i):
                y_next = self.Dealiased_SRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SSP33':
            def scan_fn(y,i):
                y_next = self.Dealiased_SSP33(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SSP22':
            def scan_fn(y,i):
                y_next = self.Dealiased_SSP22(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_EM':
            def scan_fn(y,i):
                y_next = self.Dealiased_EM(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
    def final_time_run(self, initial_state, n_steps, noise, key):
        """_run whilst only giving final timestep,
            this approach does not hold data in local memory.
            _

        Args:
            initial_state (_type_): _description_
            n_steps (_type_): _description_
            noise (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if noise is None:
            key, key1, key2 = jax.random.split(key, 3)
            noise_advective, noise_forcing = self.draw_noise(n_steps, key1, key2)
        else:
            noise_advective, noise_forcing = noise,noise

        self.validate_params()
        self.timestep_validatate()    
        ###SETDRK###
        if self.params.method == 'Dealiased_ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SETDRK4_forced':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SETDRK4_forced(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SETDRK33':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK33(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SETDRK22':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK22(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SETDRK11':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK11(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        ###IFSTK###
        elif self.params.method == 'Dealiased_IFSRK4':
            def scan_fn(y,i):
                y_next = self.Dealiased_IFSRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_33':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_33(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_22':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_22(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_eSSPIFSRK_P_11':
            def scan_fn(y,i):
                y_next = self.Dealiased_eSSPIFSRK_P_11(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        ###SRK###
        elif self.params.method == 'Dealiased_SRK4':
            def scan_fn(y,i):
                y_next = self.Dealiased_SRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SSP33':
            def scan_fn(y,i):
                y_next = self.Dealiased_SSP33(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_SSP22':
            def scan_fn(y,i):
                y_next = self.Dealiased_SSP22(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        elif self.params.method == 'Dealiased_EM':
            def scan_fn(y,i):
                y_next = self.Dealiased_EM(y, noise_advective[i], noise_forcing[i])
                return y_next, None
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        u_out,_ = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
def initial_condition(x, E , name):
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
    elif name == 'cdm':
        """https://arxiv.org/pdf/2507.17685"""
        # could not reproduce this papers initial condition setup,
        def sech(x):
            return 1.0 / jnp.cosh(x)
        def u0_rescaled(x):  # x in [0, 4]
            x_scaled = 10.0 * x
            return 0.2 * sech(x_scaled - 403.0 / 15.0) + 0.5 * sech(x_scaled - 203.0 / 15.0)
        ans = u0_rescaled(x)
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
    elif name == 'nsin':
        ans = jnp.array([jnp.sin(2 * jnp.pi * x * (p + 1)) for p in range(P)])
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

###----------------------------------------------------------####
###--------------------------SRK-----------------------------####
###----------------------------------------------------------####

@jax.jit
def Dealiased_SRK4(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)
    #g = g * dt # ensure g is multiplied by dt, as it is -.5j*dt*k

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    def L_function(_v,L):
        return L * _v
    
    def f(_v):
        return (L_function(_v, L) + N(_v, RVF, g))

    k1 = f(v)
    k2 = f(v + dt * k1 / 2)
    k3 = f(v + dt * k2 / 2)
    k4 = f(v + dt * k3)
    v_next = v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    u_next = jnp.real( jnp.fft.ifft(v_next, axis=-1) )
    return u_next

@jax.jit
def Dealiased_SSP33(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio=2/3):
    k1 = Dealiased_EM(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio)
    k2 = 3/4*u + 1/4*Dealiased_EM(k1,L,g,k,xi_p,dW_t,dt,cutoff_ratio)
    u_next = 1/3*u + 2/3*Dealiased_EM(k2,L,g,k,xi_p,dW_t,dt,cutoff_ratio)
    return u_next

@jax.jit
def Dealiased_SSP22(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio=2/3):
    k1 = Dealiased_EM(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio)
    u_next = 1/2*(u+Dealiased_EM(k1,L,g,k,xi_p,dW_t,dt,cutoff_ratio))
    return u_next

@jax.jit
def Dealiased_EM(u,L,g,k,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    def L_function(_v,L):
        return L * _v
    
    def f(_v):
        return (L_function(_v, L) + N(_v, RVF, g))

    k1 = f(v)
    v_next = v + dt * k1 
    u_next = jnp.real( jnp.fft.ifft(v_next, axis=-1) )
    return u_next

####----------------------------------------------------------####
####-----------------------SIFRK methods----------------------####
####----------------------------------------------------------####

@jax.jit
def IFRK4(u,E,E_2,g):
    """_Deterministic integrating factor Runge-Kutta 4_

    Args:
        u   =  solution at time n
        E   = jnp.exp(dt * L)
        E_2 = jnp.exp(dt * L / 2)
        g   = -.5j*dt*k, ensure dt is put in. 
    Out:
        u_next = solution at the next timestep, 
    """
    v = jnp.fft.fft(u,axis=1)

    def N(_in,g):
        """_Nonlinearity computation in real space_"""
        return g * jnp.fft.fft(jnp.real(jnp.fft.ifft(_in,axis=-1))**2, axis=-1)
    
    a = N(v,g)
    u1 = E_2*(v + a/2)
    b = N(u1,g)
    u2 = E_2*v + b/2# not correct 
    c = N(u2,g)
    u3 = E*v + E_2*c
    d = N(u3,g)
    v = E*v + (E*a + 2*E_2*(b+c) +d)/6
    u_next = jnp.real( jnp.fft.ifft(v) )

    # k_1 = N(v,g)
    # k_2 = N(E_2*(v + k_1/2),

    
    return u_next

@jax.jit
def IFSRK4(u,E,E_2,g,k,L,xi_p,dW_t,dt):
    dW_t = dW_t*jnp.sqrt(dt) / dt 
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t) # xi = (P,nx), dW_t = (E,P) so RVF = (E,nx)
    v = jnp.fft.fft(u,axis=1)
    g = g*dt# note g = -.5j*dt*k,
    #-0.5j * k * params["c_1"]
    # E = jnp.exp(dt * L)
    # E_2 = jnp.exp(dt * L / 2)
    print('error, this is not supported, use Dealiased_IFSRK4 instead')
    def N(v,RVF,g):
        r = jnp.real( jnp.fft.ifft( v ) )# convert to real space
        n = (u + 2 * RVF)*u # nonlinearity in real space 
        return g * jnp.fft.fft(n , axis=-1)# compute derivative in spectral space

    a  = N(v,RVF,g)
    v1 = E_2*(v + a/2)
    b  = N(v1,RVF,g)
    v2 = E_2*v + b/2
    c  = N(v2,RVF,g)
    v3 = E*v + E_2*c
    d  = N(v3,RVF,g)
    v  = E*v + (E*a + 2*E_2*(b+c) +d)/6

    u_next = jnp.real( jnp.fft.ifft(v) )
    return u_next

@jax.jit
def Dealiased_IFSRK4(u,E,E_2,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    #g = -.5j * dt * k 
    g = g*dt
    E = jnp.exp(dt * L)# we replace them 
    E_2 = jnp.exp(dt * L / 2) # replaced them
    
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    a = N(v,RVF,g)
    v1 = E_2*(v + a/2) 
    b = N(v1,RVF,g)
    v2 = (E_2*v + b/2)
    c = N(v2,RVF,g)
    v3 = E*v + E_2*c
    d = N(v3,RVF,g)
    v = E*v + ( E*a + 2*E_2*(b+c) +d )/6

    u_next = jnp.real(jnp.fft.ifft(v))
    return u_next



@jax.jit
def Dealiased_eSSPIFSRK_P_33(u,E,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    # we require new things.
    g = -.5j * dt * k
    E = jnp.exp(dt * L)
    E_13 = jnp.exp(dt * L * 1/3)
    E_23 = jnp.exp(dt * L * 2/3)

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    ans_1 = (v + 4/3*N(v,RVF,g))
    v1 = 1/2*E_23*(v+ans_1)
    v2 = 2/3*E_23*v + 1/3*(v1 + 4/3*N(v1,RVF,g))
    v3 = 59/128*E*v + 15/128*E*ans_1 + 27/64*E_13*(v2 + 4/3*N(v2,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v3))
    return u_next

@jax.jit
def Dealiased_eSSPIFSRK_P_22(u,E,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    g = -.5j * dt * k
    E = jnp.exp(dt * L)

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    v1 = E*(v + N(v,RVF,g))
    v1 = 1/2*(E*v + v1 + N(v1,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v1))
    return u_next

@jax.jit
def Dealiased_eSSPIFSRK_P_11(u,E,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    g = -.5j * dt * k
    E = jnp.exp(dt * L)

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    v1 = E*(v + N(v,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v1))
    return u_next

####----------------------------------------------------------####
####----------------end of SIFRK methods----------------------####
####----------------------------------------------------------####

####----------------------------------------------------------####
####-----------------start ETDRK methods----------------------####
####----------------------------------------------------------####

def Kassam_Trefethen(dt, L, nx, M=128, R=1):
    """_Precompute weights for use in ETDRK4.
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
    } 
    Trapezoidal rule has exponential convergence in the complex plane.
    This code differs from the original in the paper, in that KT note that:
    "When L is real, we can exploit the symmetry and evaluate only in equally spaced points on the upper half of a circle,
    centered on the real axis, then take the real part of the result, this is not done here"
    Here here we leave in the more general form, this allows us treat
    complex L associated with dispersion terms, such as in the KdV equation.
    However, the KS equation, has machine precision error complex values.
    Furthermore, we also absorb the factor of 2 into f2.
    _

    Args:
        dt (_type_): _timestep_
        L (_type_): _Linear operator_
        nx (_type_): _number of spatial points_
        M (int, optional): _number of points for integration_. Defaults to 32.
        R (int, optional): _Radius of circle used_. Defaults to 1.

    Returns:
        E : computed manually
        E_2 : computed manually
        Q : approximation of (e^{z}-1)/z. 
        f1 : _See eq2.5 in KassamTrefethen, alpha=f1 _
        f2 :  beta=f2
        f3 :  gamma=f3._
    """
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
def ETDRK4(u, E, E_2, Q, f1, f2, f3, g):
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
    def N(in_field,g):
        return g * jnp.fft.fft( jnp.real(jnp.fft.ifft(in_field,axis=-1))**2, axis=-1)
    
    v = jnp.fft.fft(u, axis=-1)
    Nv = N(v,g)
    a = E_2 * v + Q * Nv
    Na = N(a,g)
    b = E_2 * v + Q * Na
    Nb = N(b,g)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = N(c,g)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3

    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

@jax.jit
def Dealiased_ETDRK4(u, E, E_2, Q, f1, f2, f3, g, k, cutoff_ratio=2/3):
    v = jnp.fft.fft(u, axis=-1)
    # perhaps reasigning in the scope of the function allows faster access, requires testing.
    E=E;E_2=E_2;Q=Q;f1=f1;f2=f2;f3=f3;g=g;k=k;cutoff_ratio=cutoff_ratio
    def N(_v,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = r*r
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    Nv = N(v,g)
    a = E_2 * v + Q * Nv
    Na = N(a,g)
    b = E_2 * v + Q * Na
    Nb = N(b,g)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc = N(c,g)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3

    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

@jax.jit
def Dealiased_SETDRK4(u, E, E_2, Q, f1, f2, f3, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    dW_t = dW_t*jnp.sqrt(dt)/dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    Nv = N(v,RVF,g)
    a = E_2 * v + Q * Nv
    Na =  N(a,RVF,g)
    b = E_2 * v + Q * Na
    Nb =  N(b,RVF,g)
    c = E_2 * a + Q * (2 * Nb - Nv)
    Nc =  N(c,RVF,g)
    v_next = E * v + Nv * f1 + (Na + Nb) * f2 + Nc * f3
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

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

def dealias_using_k(spectral_field, k, cutoff_ratio=2/3):
    """_ Dealiasing using wavenumber cutoff.
        (Philips, 1959) suggested zeroing, upper half of wavenumbers,
        (Orzag 1971), instead suggested keeping 2/3rds of the wavenumbers, 
        This remains specific to the quadratic nonlinearity.
        One performs dealiasing on the nonlinearity, 
        then differentiates after in spectral space by multiplication._

    Args:
        spectral_field (_type_): _spectral field_
        k (_type_): _wavenumbers_
        cutoff_ratio (_type_, optional): _Ratio of cuttoff wavenumbers_. Defaults to 2/3.

    Returns:
        _type_: _dealiased spectral field_
    """
    k_magnitude = jnp.abs(k)
    spectral_field = jnp.where(k_magnitude < cutoff_ratio * jnp.amax(k_magnitude), spectral_field, 0)
    return spectral_field


def Complex_integration_technique(dt, L, nx, M=128, R=10):
    """ for SETDRK1, SETDRK2, SETDRK3,
    Args:
        dt (_type_): _timestep_
        L (_type_): _Linear operator_
        nx (_type_): _number of spatial points_
        M (int, optional): _number of points for integration_. Defaults to 32.
        R (int, optional): _Radius of circle used_. Defaults to 1.
    """
    #R = 0.005 * jnp.max(jnp.abs(L))
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    # For SETDRK1.
    A1 = jnp.exp(dt * L)
    A2 = dt * jnp.mean( (jnp.exp(LR) - 1) / LR, axis=-1)  # trapesium rule performed by mean in the M variable.
    # for SETDRK2.
    B1 = jnp.exp(dt * L)
    B2 = dt * jnp.mean( (jnp.exp(LR) - 1) / LR, axis=-1)
    B3 = dt * jnp.mean( (jnp.exp(LR) - 1 - LR ) / LR**2, axis=-1)
    # for SETDRK3.
    E1 = jnp.exp(dt * L / 2)
    E2 = dt * jnp.mean( (jnp.exp(LR/2) - 1) / LR, axis=-1)
    E3 = jnp.exp(dt * L)
    E4 = dt * jnp.mean( (jnp.exp(LR) - 1) / LR, axis=-1)
    E5 = dt * jnp.mean( (-4 - LR + jnp.exp(LR)*(4 - 3*LR + LR**2)  ) / LR**3, axis=-1)
    E6 = dt * 4* jnp.mean( (2 + LR + jnp.exp(LR)*(-2 + LR)) / LR**3, axis=-1)
    E7 = dt * jnp.mean( (-4 - 3*LR - LR**2 + jnp.exp(LR)*(4 - LR)) / LR**3, axis=-1)
    
    return A1, A2, B1, B2, B3, E1, E2, E3, E4, E5, E6, E7

def SETDRK11_params():
    """_Precompute weights for use in SETDRK11."""
    A1, A2, B1, B2, B3, E1, E2, E3, E4, E5, E6, E7 = Complex_integration_technique(dt, L, nx, M=128, R=1)
    return A1, A2

def SETDRK22_params():
    """_Precompute weights for use in SETDRK22."""
    A1, A2, B1, B2, B3, E1, E2, E3, E4, E5, E6, E7 = Complex_integration_technique(dt, L, nx, M=128, R=1)
    return B1, B2, B3
def SETDRK33_params():
    """_Precompute weights for use in SETDRK33."""
    A1, A2, B1, B2, B3, E1, E2, E3, E4, E5, E6, E7 = Complex_integration_technique(dt, L, nx, M=128, R=1)
    return E1, E2, E3, E4, E5, E6, E7 

@jax.jit
def Dealiased_SETDRK11(u, A1, A2, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    dW_t = dW_t*jnp.sqrt(dt)/dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    Nv = N(v,RVF,g)
    v_next  = A1 * v + A2 * Nv
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

@jax.jit
def Dealiased_SETDRK22(u, B1, B2, B3, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    dW_t = dW_t*jnp.sqrt(dt)/dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    Nv = N(v,RVF,g)
    k1  = B1 * v + B2 * Nv
    v_next = k1 + B3 * ( N(k1,RVF,g) - Nv )
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

@jax.jit
def Dealiased_SETDRK33(u, E1, E2, E3, E4, E5, E6, E7, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    dW_t = dW_t*jnp.sqrt(dt)/dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u, axis=1)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + 2*_RVF)*r # g contains 1/2 in it. 
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    Nv = N(v,RVF,g)
    k1 = E1 * v + E2 * Nv
    Nk1 = N(k1,RVF,g)
    k2 = E3 * v + E4*(2*Nk1 - Nv) 
    Nk2 = N(k2,RVF,g)
    v_next = E3 * v + E5*Nv + E6*Nk1+ E7*Nk2
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

@jax.jit
def Dealiased_CSSSPETDRK33(u, A1, A2, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    # naive attempt for a ETDRK33 method, only using the A1,A2
    k1 = Dealiased_SETDRK11(u, A1, A2, g, k, xi_p, dW_t, dt, cutoff_ratio)
    k2 = 3/4*u + (1/4)*Dealiased_SETDRK11(k1, A1, A2, g, k, xi_p, dW_t, dt, cutoff_ratio)
    u_next = 1/3*u + (2/3)*Dealiased_SETDRK11(k2, A1, A2, g, k, xi_p, dW_t, dt, cutoff_ratio)
    return u_next

Heat_params = {# Heat equation. 
    "equation_name" : 'Heat', 
    "c_0": 0, "c_1": 0, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 16, "dt": 1 / 128,  "noise_magnitude": 0.1, "nt": int(16 * 128),
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

LinearAdvection_params = {# Linear Advection equation.
    "equation_name" : 'Linear-Advection', 
    "c_0": 0.5, "c_1": 0, "c_2": 0, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 2, "tmax": 16, "dt": 1 / 128,  "noise_magnitude": 0.001, "nt": int(16 * 128),
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

Burgers_params={# Burgers equation. 
    "equation_name" : 'Burgers', 
    "c_0": 0, "c_1": 1, "c_2": -1/256, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 0.5, "dt": 1 / 128,  "noise_magnitude": 0.001, "nt": int(0.5 * 128),
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

# Simulation parameters 
KS_params = {# KS equation, from Kassam Krefethen deterministic.
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0., "xmax": 32*jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 150., "dt": 0.25 , "noise_magnitude": 0.0, "nt": 600,
    "initial_condition": 'Kassam_Trefethen_KS_IC', "method": 'Dealiased_ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KS_params_SALT = {# KS equation, from Kassam Krefethen but with transport noise
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 150, "dt": 0.25, "noise_magnitude": 0.001, "nt": 600,
    "initial_condition": 'Kassam_Trefethen_KS_IC', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}


KDV_params = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 0.01, "dt": 2e-6, "noise_magnitude": 0.0, "nt": int(0.01 / 2e-6),
    "initial_condition": 'Kassam_Trefethen_KdV_IC_eq3pt1', "method": 'Dealiased_ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KDV_params_2 = {# KdV equation. gaussian initial condition, small dispersion, no noise
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 4, "dt": 0.001, "noise_magnitude": 0.0, "nt": 4000,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KDV_params_2_SALT = {# KdV equation. gaussian initial condition, small dispersion, constant advevctive noise
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 4.0, "dt": 0.001, "noise_magnitude": 0.01, "nt": 4000,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}

KDV_params_2_SALT_LEARNING = {# KdV equation. gaussian initial condition, small dispersion, used for learning.
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 0.4, "dt": 0.001, "noise_magnitude": 0.01, "nt": 400,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}
KDV_params_noise = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 256, "P": 1, "S": 0, "E": 2, "tmax": 0.01, "dt": 2e-6, "noise_magnitude": 2,"nt": int(0.01 / 2e-6),
    "initial_condition": 'Kassam_Trefethen_KdV_IC_eq3pt1', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}
KDV_params_traveling = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/pdectb/kdv2.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 64, "P": 1, "S": 0, "E": 30, "tmax": 1.0, "dt": 0.0001, "noise_magnitude": 1.0, "nt": int(1.0 / 0.0001),
    "initial_condition": 'traveling_wave', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}
KDV_params_traveling_demo = {# KdV equation. traveling wave initial condition
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 64, "P": 1, "S": 0, "E": 30, "tmax": 10.0, "dt": 0.0001, "noise_magnitude": 1.0, "nt": int(10.0 / 0.0001),
    "initial_condition": 'traveling_wave', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}




KDV_params_SALT = {# KdV equation. gaussian initial condition, small dispersion.
    "equation_name": 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1e-4, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 128, "P": 1, "S": 0, "E": 3, "tmax": 10, "dt": 1e-5, "noise_magnitude": 0.1, "nt": 1000000,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}

KDV_params_exact_traveling = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/pdectb/kdv2.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 64, "P": 1, "S": 0, "E": 1, "tmax": 1.0, "dt": 0.0001, "noise_magnitude": 1.0, "nt": int(1.0 / 0.0001),
    "initial_condition": 'new_traveling_wave', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}


##### examples with forcing. 

KDV_params_traveling_demo_forced = {
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 128, "P": 1, "S": 1, "E": 1, "tmax": 10.0, "dt": 0.0001, "noise_magnitude": 1.0, "nt": int(10.0 / 0.0001),
    "initial_condition": 'traveling_wave', "method": 'Dealiased_SETDRK4_forced', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'sin'
}

KDV_params_2_FORCE = {# KdV equation. gaussian initial condition, small dispersion.
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 0, "S": 1, "E": 2, "tmax": 4, "dt": 0.001, "noise_magnitude": 0.1, "nt": 4000,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4_forced', 
    "Advection_basis_name": 'none', "Forcing_basis_name": 'sin'
}

KS_params_Force = {# KS equation, from Kassam Krefethen but with transport noise
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 0, "S": 1, "E": 1, "tmax": 150, "dt": 0.25, "noise_magnitude": 0.001, "nt": 600,
    "initial_condition": 'Kassam_Trefethen_KS_IC', "method": 'Dealiased_SETDRK4_forced',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'sin'
}


KS_params_force_cdm = {# KS equation, under initial conditions and parameters similar to Colin, Dan and Manish
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 0.03, "c_3": 0.0, "c_4": 1.1,
    "xmin": 0, "xmax": 4, "nx": 256, "P": 0, "S": 1, "E": 1, "tmax": 150, "dt": 0.25, "noise_magnitude": 0.001, "nt": 600,
    "initial_condition": 'cdm', "method": 'Dealiased_SETDRK4_forced',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'sin'
}

if __name__ == "__main__":
    #### Below is a testing script to run the code, without instantiating the class.
    # Currently setup to demonstrate the advantages of the ETDRK4 method for the KdV equation, at high cfl low res.
    jax.config.update("jax_enable_x64", True)
    #params = LinearAdvection_params#KDV_params_SALT#KDV_params_SALT
    #params = KDV_params
    #params = KDV_params_noise
    #params = Heat_params#KDV_params_2
    #params = KS_params_SALT
    #params = Burgers_params
    #params = LinearAdvection_params
    #params = KDV_params_traveling_demo
    #params = KDV_params_traveling_demo_forced

    params = KDV_params_2_FORCE 
    #params = KDV_params_2_FORCE
    #params = KS_params
    cwd = os.getcwd()
    # _equation_name = params['equation_name'];_initial_condition = params['initial_condition']
    # run_file_name = f'{cwd}/config/auto_yaml/{_equation_name}_{_initial_condition}.yml'
    # with open(f'{run_file_name}', 'w') as outfile:
    #     yaml.dump(params, outfile, default_flow_style=False)
    # params = LinearAdvection_params
    xmin, xmax, nx, P, S, E, tmax, dt = params["xmin"],params["xmax"], params["nx"], params["P"],params["S"], params["E"], params["tmax"], params["dt"]
    E = 1#
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
