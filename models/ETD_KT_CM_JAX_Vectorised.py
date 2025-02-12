import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel
try:
    from jax.config import config
    config.update("jax_enable_x64", True)
except ImportError:
    jax.config.update("jax_enable_x64", True)

import yaml



class ETD_KT_CM_JAX_Vectorised(BaseModel):
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
        self.nmax = round(self.params.tmax / self.params.dt)
        #self.dt = self.params.tmax/self.nmax
        self.noise_key = jax.random.PRNGKey(0)
        self.key1, self.key2 = jax.random.split(self.noise_key)

        self.stochastic_advection_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.P, self.params.Advection_basis_name)
        self.stochastic_forcing_basis = self.params.noise_magnitude * stochastic_basis_specifier(self.x, self.params.S, self.params.Forcing_basis_name)
        

    def validate_params(self):
        if self.params.method not in ['Dealiased_ETDRK4','step_Dealiased_SETDRK4']:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        if self.params.method == 'Dealiased_ETDRK4':
            if self.params.P != 0 or self.params.S != 0:
                raise ValueError(f"Method {self.params.method} requires P and S to be 0")
            
        if self.params.method == 'Dealiased_SETDRK4':
            pass

        if self.params.E < 1:
            raise ValueError(f"Number of ensemble members E must be greater than or equal to 1")
        
        pass

    def draw_noise(self, n_steps, key1, key2):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.P))
        dZ = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S))
        return dW, dZ
    
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
    
    # def Dealiased_IFSRK4(self, initial_state, noise_advective, noise_forcing):
    #     u = initial_state
    #     u = Dealiased_IFSRK4(u,
    #                          self.E_weights,
    #                          self.E_2,
    #                          self.g,
    #                          self.k,
    #                          self.L,
    #                          self.stochastic_advection_basis,
    #                          noise_advective,
    #                          self.params.dt
    #                          )
    #     return u

    def run(self, initial_state, n_steps, noise):

        if noise is None:
            self.key1, key1 = jax.random.split(self.key1, 2)
            self.key2, key2 = jax.random.split(self.key2, 2)
            noise_advective,noise_forcing = self.draw_noise(n_steps, key1, key2)
        else:
            noise_advective = noise
            noise_forcing = noise
        #self.validate_params()
            
        if self.params.method == 'Dealiased_ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        elif self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

        return u_out
    
    def final_time_run(self, initial_state, n_steps, noise):
        """_run whilst only giving final timestep
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
            self.key1, key1 = jax.random.split(self.key1, 2)
            self.key2, key2 = jax.random.split(self.key2, 2)
            noise_advective,noise_forcing = self.draw_noise(n_steps, key1, key2)

        if self.params.method == 'Dealiased_ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, None
            
        elif self.params.method == 'Dealiased_SETDRK4':
            def scan_fn(y,i):
                y_next = self.step_Dealiased_SETDRK4(y, noise_advective[i], noise_forcing[i])
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
        ans =  jnp.where(ans<1e-7,0,ans)#
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
    elif name == 'constant':
        nx = len(x)
        ans = jnp.ones((P, nx))# each basis function is a constant
    else:
        raise ValueError(f"Stochastic basis {name} not recognised")
    return ans

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
        """_Nonlinearity computation in real space_

        Args:
            _in (_type_): _description_
            g (_type_): _description_

        Returns:
            _type_: _description_
        """
        return g * jnp.fft.fft(jnp.real(jnp.fft.ifft(_in,axis=-1))**2, axis=-1)

    a = N(v,g)
    u1 = E_2*(v + a/2)
    b = N(u1,g)
    u2 = E_2*(v + b/2)
    c = N(u2,g)
    u3 = E*v + E_2*c
    d = N(u3,g)
    v = E*v + (E*a + 2*E_2*(b+c) +d)/6
    u_next = jnp.real( jnp.fft.ifft(v) )
    
    return u_next

@jax.jit
def IFSRK4(u,E,E_2,g,k,L,xi_p,dW_t,dt):
    dW_t = dW_t*jnp.sqrt(dt) / dt 
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t) # xi = (P,nx), dW_t = (E,P) so RVF = (E,nx)
    v = jnp.fft.fft(u,axis=1)
    g = g*dt#-.5j*dt*k
    #-0.5j * k * params["c_1"]
    # E = jnp.exp(dt * L)
    # E_2 = jnp.exp(dt * L / 2)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )# convert to real space
        n = (u + RVF)*u # nonlinearity in real space
        return g * jnp.fft.fft(n , axis=-1)# compute derivative in spectral space

    a  = N(v,RVF,g)
    v1 = E_2*(v + a/2)
    b  = N(v1,RVF,g)
    v2 = E_2*(v + b/2)
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
    # E = jnp.exp(dt * L)
    # E_2 = jnp.exp(dt * L / 2)
    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + _RVF)*r
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    a = N(v,RVF,g)
    v1 = E_2*(v + a/2)
    b = N(v1,RVF,g)
    v2 = E_2*(v + b/2)
    c = N(v2,RVF,g)
    v3 = E*v + E_2*c
    d = N(v3,RVF,g)
    v = E*v + ( E*a + 2*E_2*(b+c) +d )/6

    u_next = jnp.real(jnp.fft.ifft(v))
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
        n = (r + _RVF)*r
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    v1 = E*(v + N(v,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v1))
    return u_next

@jax.jit
def Dealiased_eSSPIFSRK_P_22(u,E,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    "SSP"
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    g = -.5j * dt * k
    E = jnp.exp(dt * L)

    def N(_v,_RVF,g):
        r = jnp.real( jnp.fft.ifft( _v ) )
        n = (r + _RVF)*r
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    v1 = E*(v + N(v,RVF,g))
    v1 = 1/2*(E*v + v1 + N(v1,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v1))
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
        n = (r + _RVF)*r
        n = dealias_using_k(jnp.fft.fft(n , axis=-1), k, cutoff_ratio=cutoff_ratio)
        return g * n
    
    ans_1 = (v + 4/3*N(v,RVF,g))
    v1 = 1/2*E_23*(v+ans_1)
    v2 = 2/3*E_23*v + 1/3*(v1 + 4/3*N(v1,RVF,g))
    v3 = 59/128*E*v + 15/128*E*ans_1 + 27/64*E_13*(v2 + 4/3*N(v2,RVF,g))
    u_next = jnp.real(jnp.fft.ifft(v3))
    return u_next

####----------------------------------------------------------####
####----------------end of SIFRK methods----------------------####
####----------------------------------------------------------####

####----------------------------------------------------------####
####-----------------start ETDRK methods----------------------####
####----------------------------------------------------------####

def Kassam_Trefethen(dt, L, nx, M=64, R=1):
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
    centered on the real axis, then take the real part of the result."
    Here here we leave in the more general form, this allows us 
    complex L associated with dispersion terms, such as in the KdV equation.
    Furthermore, we also absorb the factor of 2 into f2.
    _

    Args:
        dt (_type_): _timestep_
        L (_type_): _Linear operator_
        nx (_type_): _number of spatial points_
        M (int, optional): _number of points for integration_. Defaults to 32.
        R (int, optional): _Radius of circle used_. Defaults to 1.

    Returns:
        E : 
        E_2 :
        Q :
        f1 : _See eq2.5 in KassamTrefethen, alpha=f1 _
        f2 :  beta=f2
        f3 :  gamma=f3._
    """
    E_1 = jnp.exp(dt * L) 
    E_2 = jnp.exp(dt * L / 2)
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
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
        n = (r + _RVF)*r
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

# Simulation parameters 
KS_params = {# KS equation, from Kassam Krefethen
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 150, "dt": 0.25 , "noise_magnitude": 0.0,
    "initial_condition": 'Kassam_Trefethen_KS_IC', "method": 'Dealiased_ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KS_params_SALT = {# KS equation, from Kassam Krefethen with transport noise
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 150, "dt": 0.25, "noise_magnitude": 0.001,
    "initial_condition": 'Kassam_Trefethen_KS_IC', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}
KDV_params_2 = {# KdV equation. gaussian initial condition, small dispersion.
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 1, "dt": 0.001, "noise_magnitude": 0.0,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KDV_params_2_SALT = {# KdV equation. gaussian initial condition, small dispersion.
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 1, "S": 0, "E": 10, "tmax": 1, "dt": 0.001, "noise_magnitude": 0.01,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}



Heat_params = {# Heat equation. 
    "equation_name" : 'Heat', 
    "c_0": 0, "c_1": 0, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 16, "dt": 1 / 128,  "noise_magnitude": 0.1,
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

LinearAdvection_params = {# Linear Advection equation.
    "equation_name" : 'Linear-Advection', 
    "c_0": 0.5, "c_1": 0, "c_2": 0, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 2, "tmax": 16, "dt": 1 / 128,  "noise_magnitude": 0.001,
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

Burgers_params={# Burgers equation. 
    "equation_name" : 'Burgers', 
    "c_0": 0, "c_1": 1, "c_2": -1/256, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 0.5, "dt": 1 / 128,  "noise_magnitude": 0.001,
    "initial_condition": 'sin', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

KDV_params = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 0.01, "dt": 2e-6, "noise_magnitude": 0.0,
    "initial_condition": 'Kassam_Trefethen_KdV_IC_eq3pt1', "method": 'Dealiased_ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KDV_params_noise = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 256, "P": 1, "S": 0, "E": 2, "tmax": 0.01, "dt": 2e-6, "noise_magnitude": 2,
    "initial_condition": 'Kassam_Trefethen_KdV_IC_eq3pt1', "method": 'Dealiased_SETDRK4',
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}



KDV_params_traveling = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/pdectb/kdv2.pdf
    "equation_name" : 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 64, "P": 1, "S": 0, "E": 30, "tmax": 1.0, "dt": 0.0001, "noise_magnitude": 1.0,
    "initial_condition": 'traveling_wave', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'constant', "Forcing_basis_name": 'none'
}

KDV_params_SALT = {# KdV equation. gaussian initial condition, small dispersion.
    "equation_name": 'KdV', 
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1e-4, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 128, "P": 1, "S": 0, "E": 3, "tmax": 10, "dt": 1e-5, "noise_magnitude": 0.1,
    "initial_condition": 'gaussian', "method": 'Dealiased_SETDRK4', 
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    #params = LinearAdvection_params#KDV_params_SALT#KDV_params_SALT
    #params = KDV_params
    #params = KDV_params_noise
    #params = Heat_params#KDV_params_2
    #params = KS_params_SALT
    #params = Burgers_params
    params = LinearAdvection_params
    #params = KDV_params_traveling
    #params = KS_params
    cwd = os.getcwd()

    _equation_name = params['equation_name'];_initial_condition = params['initial_condition']
    run_file_name = f'{cwd}/config/auto_yaml/{_equation_name}_{_initial_condition}.yml'
    with open(f'{run_file_name}', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)
    # params = LinearAdvection_params
    xmin, xmax, nx, P, S, E, tmax, dt = params["xmin"],params["xmax"], params["nx"], params["P"],params["S"], params["E"], params["tmax"], params["dt"]
    
    #dx = (xmax - xmin) / nx
    #x = jnp.linspace(xmin, xmax, nx, endpoint=False)
    xf = jnp.linspace(xmin, xmax, nx+1)
    x = 0.5 * ( xf[1:] + xf[:-1] ) # cell centers
    dx = x[1]-x[0] # cell width
    u = initial_condition(x, E, params["initial_condition"])
    u_benchmark = u
    
    k = jnp.fft.fftfreq(nx, dx, dtype=jnp.complex128) * 2 * jnp.pi
    L = -1j * k * params["c_0"] + k**2 * params["c_2"] + 1j*k**3 * params["c_3"] - k**4 * params["c_4"]
    g = -0.5j * k * params["c_1"]

    E_1, E_2, Q, f1, f2, f3 = Kassam_Trefethen(dt, L, nx, M=64, R=1)# we rename the weights
    #E_weights, Q, LW = LW_contour_trick(L, dt, M=64, R=1)
    nmax = round(tmax / dt)
    uu = jnp.zeros([E, nmax, nx])
    uu = uu.at[:, 0, :].set(u)
    UU = jnp.zeros([E, nmax, nx])
    UU = UU.at[:, 0, :].set(u)

    stochastic_advection_basis = params["noise_magnitude"] * stochastic_basis_specifier(x, P, params["Advection_basis_name"])
    stochastic_forcing_basis   = params["noise_magnitude"] * stochastic_basis_specifier(x, S, params["Forcing_basis_name"])

    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    dW = jax.random.normal(key1, shape=(nmax, E, P))
    dZ = jax.random.normal(key2, shape=(nmax, E, S))
    print(stochastic_advection_basis.shape,dW.shape)

    # W = jnp.cumsum(dW, axis=0)
    # W = jnp.sqrt(dt) * params["magnitude"] * W
    # plt.plot(W[:,0,0].T)
    # plt.show()

    for n in range(1, nmax):
        beta = 3
        # some number instead of 0.125. 
        #U = initial_condition((x - 4*beta**2 * (dt*n) - 1/2*W[n,:,:] + xmax)%(xmax*2) - xmax, E, params["initial_condition"])
        #u = Dealiased_Strang_split(u, E_weights, E_2, Q, f1, f2, f3, g, dt, 0*stochastic_advection_basis,dW[n, :, :], stochastic_forcing_basis,dZ[n, :, :], k)
        #u = IFRK4(u,E_weights,E_2,g,k,L)
        #u = Dealiased_IFSRK4(u,E_1,E_2,g,k,L,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        #u = SETDRK1(u, E, Q, LW, g, k, stochastic_advection_basis,dW[n, :, :], cutoff_ratio=2/3)
        #u = Dealiased_SETDRK4(u, E_weights, E_2, Q, f1, f2, f3, g, k, stochastic_advection_basis,dW[n, :, :],dt)
        #u = Dealiased_eSSPIFSRK_P_11(u,E,g,k,L,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        #u = Dealiased_IFSRK4(u,E_weights,E_2,g,k,L,stochastic_advection_basis,dW[n, :, :],dt)
        # dealiasing is important for the IFSRK4 method.
        u = Dealiased_SETDRK4(u, E_1, E_2, Q, f1, f2, f3, g, k, stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        #U = Dealiased_SETDRK4(U, E_1, E_2, Q, f1, f2, f3, g, k,stochastic_advection_basis,dW[n, :, :],dt,cutoff_ratio=2/3)
        
        #u_benchmark = Dealiased_ETDRK4(u_benchmark, E_weights, E_2, Q, f1, f2, f3, g, k, cutoff_ratio=2/3)
        #u, E, E_2, Q, f1, f2, f3, g
        #u_benchmark = Dealiased_IFSRK4(u_benchmark,E_weights,E_2,g,k,L,stochastic_advection_basis,dW[n, :, :],dt)
        #u = Strang_split(u, E_weights, E_2, Q, f1, f2, f3, g, dt , stochastic_advection_basis, dW[n, :, :], stochastic_forcing_basis, dZ[n, :, :])
        
        
        uu = uu.at[:, n, :].set(u)
        #UU = UU.at[:, n, :].set(U)
        if n % 10 == 0:  # Plot every 10 steps for better performance
            plt.clf()
            for e in range(E):
                plt.plot(x, u[e, :], label=f'SEDTRK4 {e + 1}',linewidth=0.5,c='b')
                #plt.plot(x, U[e, :], label=f'SEDTRK4',linewidth=0.5,c='k')

                #plt.plot(x, u_benchmark[e, :], label=f'bench{e + 1}')
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

    plt.show()
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.pcolormesh(x, jnp.arange(0, nmax) * dt, UU.mean(axis=0))
    plt.colorbar(label='Mean Solution')
    plt.title("Ensemble Average Solution")
    plt.show()