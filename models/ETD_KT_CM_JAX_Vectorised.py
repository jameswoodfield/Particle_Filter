import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
try:
    from .base import BaseModel
except ImportError:
    from base import BaseModel
from jax.config import config
import yaml

config.update("jax_enable_x64", True)

class ETD_KT_CM_JAX_Vectorised(BaseModel):
    def __init__(self, params):
        self.params = params
        self.xf = jnp.linspace(self.params.xmin, self.params.xmax, self.params.nx+1)
        self.x = 0.5 * ( self.xf[1:] + self.xf[:-1] ) # cell centers
        self.dx = self.x[1]-self.x[0] # cell width
        #self.x = jnp.linspace(self.params.xmin, self.params.xmax , self.params.nx, endpoint=False)
        #self.dx = (self.params.xmax - self.params.xmin) / self.params.nx

        self.k = jnp.fft.fftfreq(self.params.nx, self.dx) * 2 * jnp.pi # additional multiplication by 2pi
        self.L = -1j * self.k * self.params.c_0 + self.k**2 * self.params.c_2 + 1j * self.k**3 * self.params.c_3 - self.k**4 * self.params.c_4
        self.g = -0.5j * self.k * self.params.c_1

        self.E_weights, self.E_2, self.Q, self.f1, self.f2, self.f3 = Kassam_Trefethen(self.params.dt, self.L, self.params.nx)
        self.nmax = round(self.params.tmax / self.params.dt)
        self.noise_key = jax.random.PRNGKey(0)
        self.key1, self.key2 = jax.random.split(self.noise_key)


    def create_basis(self):
        self.stochastic_advection_basis = self.params.sigma * stochastic_basis_specifier(self.x, self.P, self.params.Advection_basis_name)
        self.stochastic_forcing_basis = self.params.sigma * stochastic_basis_specifier(self.x, self.S, self.params.Forcing_basis_name)
        

    def validate_params(self):
        if self.params.method not in ['SS_ETDRK4_SSP33','Dealiased_SS_ETDRK4_SSP33', 'ETDRK4', 'Dealiased_ETDRK4']:
            raise ValueError(f"Method {self.params.method} not recognised")
        if self.params.method == 'SS_ETDRK4_SSP33':
            if self.params.P == 0 or self.params.S == 0:
                raise ValueError(f"Method {self.params.method} requires P and S to be greater than 0")
        if self.params.method == 'Dealiased_SS_ETDRK4_SSP33':
            if self.params.P == 0 or self.params.S == 0:
                raise ValueError(f"Method {self.params.method} requires P and S to be greater than 0")
        if self.params.method == 'ETDRK4':
            if self.params.P != 0 or self.params.S != 0:
                raise ValueError(f"Method {self.params.method} requires P and S to be 0")
        if self.params.method == 'Dealiased_ETDRK4':
            if self.params.P != 0 or self.params.S != 0:
                raise ValueError(f"Method {self.params.method} requires P and S to be 0")
        if self.params.E < 1:
            raise ValueError(f"Number of ensemble members E must be greater than or equal to 1")
        pass


    def draw_noise(self, n_steps, key1, key2):
        dW = jax.random.normal(key1, shape=(n_steps, self.params.E, self.params.P))
        dZ = jax.random.normal(key2, shape=(n_steps, self.params.E, self.params.S))
        return dW, dZ

    def step_SS_ETDRK4_SSP33(self, initial_state, noise_advective, noise_forcing):
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

    def step_Dealiased_SS_ETDRK4_SSP33(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = Dealiased_Strang_split(u, 
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
                         noise_forcing,
                         self.k)
        return u
    
    def step_ETDRK4(self, initial_state, noise_advective, noise_forcing):
        u = initial_state
        u = ETDRK4(u, 
                   self.E_weights, 
                   self.E_2, 
                   self.Q, 
                   self.f1, 
                   self.f2, 
                   self.f3, 
                   self.g)
        return u
    
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

    def run(self, initial_state, n_steps, noise):

        if noise is None:
            self.key1, key1 = jax.random.split(self.key1, 2)
            self.key2, key2 = jax.random.split(self.key2, 2)
            noise_advective,noise_forcing = self.draw_noise(n_steps, key1, key2)

        #self.validate_params()
        # dependent on the method define the scan function.
        if self.params.method == 'SS_ETDRK4_SSP33':
            def scan_fn(y, i):
                y_next = self.step_SS_ETDRK4_SSP33(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
        elif self.params.method == 'Dealiased_SS_ETDRK4_SSP33':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_SS_ETDRK4_SSP33(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        elif self.params.method == 'ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        elif self.params.method == 'Dealiased_ETDRK4':
            def scan_fn(y, i):
                y_next = self.step_Dealiased_ETDRK4(y, noise_advective[i], noise_forcing[i])
                return y_next, y_next
            
        else:
            raise ValueError(f"Method {self.params.method} not recognised")
        
        u_out = jax.lax.scan(scan_fn, initial_state, jnp.arange(n_steps))

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

    elif name == 'Kassam_Trefethen_KdV_IC_shift':
        A = 25
        B = 16
        ans  = 3 * A**2 * jnp.cosh( 0.5 * A * (x+3) )**-2
        ans += 3 * B**2 * jnp.cosh( 0.5 * B * (x+2) )**-2

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
        ValueError: _description_

    Returns:
        _array_: _(P,nx)_
    """
    if name == 'sin':
        ans = jnp.array([1 / (p + 1) * jnp.sin(2 * jnp.pi * x * (p + 1)) for p in range(P)])
    elif name =='none':
        ans = jnp.zeros([P,x.shape[0]])
    else:
        raise ValueError(f"Stochastic basis {name} not recognised")
    return ans

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
    centered on the real axis, then take the real part of the result.", where here we leave in the more general form, 
    as to allow complex L associated with dispersion terms, such as in the KdV equation. Furthermore, one also can absorb the factor of 2 into f2.
    _

    Args:
        dt (_type_): _timestep_
        L (_type_): _Linear operator_
        nx (_type_): _number of spatial points_
        M (int, optional): _number of points for integration_. Defaults to 32.
        R (int, optional): _Radius of circle used_. Defaults to 1.

    Returns:
        _E : 
        E_2 :
        Q :
        f1 : _See eq2.5 in KassamTrefethen, alpha=f1 _
        f2 :  beta=f2
        f3 :  gamma=f3._
    """
    E = jnp.exp(dt * L) 
    E_2 = jnp.exp(dt * L / 2)
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * jnp.transpose(jnp.tile(L, (M, 1))) + jnp.tile(r, (nx, 1))
    #LR = dt * L[:, None] + r[None, :] # broadcasting (nx,M) TODO: this would allow the removal of the nx arguement
    Q  = dt * jnp.mean( (jnp.exp(LR / 2) - 1) / LR, axis=-1)# trapesium rule performed by mean in the M variable.
    f1 = dt * jnp.mean( (-4 - LR + jnp.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * jnp.mean( (4 + 2 * LR + jnp.exp(LR) * (-4 + 2*LR)) / LR**3, axis=-1)# 2 times the KT one. 
    f3 = dt * jnp.mean( (-4 - 3 * LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=-1)
    return E, E_2, Q, f1, f2, f3

def LW_contour_trick(L, dt, M=64, R=1):
    # new speculative stuff
    E = jnp.exp(dt * L)
    r = R * jnp.exp(2j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * jnp.transpose(jnp.tile(L, (M, 1))) + jnp.tile(r, (nx, 1))
    Q  = dt * jnp.mean( (jnp.exp(LR / 2) - 1) / LR, axis=-1)
    LW  = jnp.sqrt(dt) * jnp.mean( jnp.sqrt( (jnp.exp(2 * LR) - 1 ) / (2*LR)), axis=-1)
    return E, Q, LW

def SETDRK1(u, E, Q, LW, g, k, xi_p, dW_t, cutoff_ratio=2/3):    
    # new speculative stuff
    v = jnp.fft.fft(u, axis=-1)
    dW_t = dW_t
    print(xi_p.shape,dW_t.shape)
    RVF = jnp.einsum('jk,lj ->lk',xi_p*0,dW_t)
    u_squared = u**2
    u_squared_hat = jnp.fft.fft(u_squared, axis=-1)
    Nv = g * dealias_using_k(u_squared_hat, k, cutoff_ratio=cutoff_ratio)# computes (u^2)_x# f
    u_times_a =  u * RVF # computes (u\xi)
    Ng = 2 * g * dealias_using_k(u_times_a, k, cutoff_ratio=cutoff_ratio)# compute (u\xi)_x# g
    v_next = E * v + Nv * Q +  Ng * LW
    u_next = jnp.real( jnp.fft.ifft( v_next, axis=-1 ) )
    return u_next

def IFRK4(u,E,E_2,g,k,L):
    """_Integrating factor Runge Kutta 4_

    Args:
        u (_type_): _description_
        E (_type_): _description_
        E_2 (_type_): _description_
    """
    # it may be possible to define these out of the scope, then reassign them in the scope
    # this would allow for the function not to have to recompute them each time.
    g = -.5j*dt*k
    E = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)

    v = jnp.fft.fft(u,axis=1)
    def N(_in,g):
        return g * jnp.fft.fft(jnp.real(jnp.fft.ifft(_in,axis=-1))**2, axis=-1)

    a = N(v,g)
    u1 = E_2*(v + a/2)
    b = N(u1,g)
    u2 = E_2*(v + b/2)
    c = N(u2,g)
    u3 = E*v + E_2*c
    d = N(u3,g)
    v = E*v + (E*a + 2*E_2*(b+c) +d)/6
    u_next = jnp.fft.ifft(v).real
    return u_next

def IFSRK4(u,E,E_2,g,k,L,xi_p,dW_t,dt):
    dW_t = dW_t*jnp.sqrt(dt) / dt 
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t) # xi = (P,nx), dW_t = (E,P) so RVF = (E,nx)
    v = jnp.fft.fft(u,axis=1)
    g = -.5j*dt*k
    E = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)

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

def Dealiased_IFSRK4(u,E,E_2,g,k,L,xi_p,dW_t,dt,cutoff_ratio=2/3):
    dW_t = dW_t * jnp.sqrt(dt) / dt
    RVF = jnp.einsum('jk,lj ->lk',xi_p,dW_t)
    v = jnp.fft.fft(u,axis=1)
    g = -.5j * dt * k
    E = jnp.exp(dt * L)
    E_2 = jnp.exp(dt * L / 2)

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

    u_next = jnp.fft.ifft(v).real
    return u_next


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


def Dealiased_SETDRK4(u, E, E_2, Q, f1, f2, f3, g, k, xi_p, dW_t, dt, cutoff_ratio=2/3):
    # new speculative stuff
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
    #max_k = jnp.max(k_magnitude)
    #cutoff = max_k * cutoff_ratio
    #mask = k_magnitude <= cutoff
    #spectral_field = spectral_field * mask
    spectral_field = jnp.where(k_magnitude < cutoff_ratio * jnp.amax(k_magnitude), spectral_field, 0)
    return spectral_field


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
    """_Divides with error handling_

    Args:
        a (_type_): _numerator_
        b (_type_): _denominator_

    Returns:
        _type_: _answer_
    """
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
    """_SSP33, timestepper originally proposed by Shu and Osher in 1988, 
    this stochastic version converges to the stratonovich equation, 
    proven as a subcase of the arguements presented in Ruemelin 1982._

    Args:
        q (_type_): _description_
        dt (_type_): _description_
        xi_p (_type_): _description_
        dW_t (_type_): _description_
        f_s (_type_): _description_
        dZ_t (_type_): _description_

    Returns:
        _type_: _description_
    """
    _q1 = Euler_Maruyama(q,dt,xi_p,dW_t,f_s,dZ_t)
    _q1 = 1/3*q+2/3*Euler_Maruyama(_q1,dt,xi_p,dW_t,f_s,dZ_t)
    _q1 = 1/4*q+3/4*Euler_Maruyama(_q1,dt,xi_p,dW_t,f_s,dZ_t)
    return _q1

def Strang_split(u, E, E_2, Q, f1, f2, f3, g, dt, xi_p, dW_t, f_s, dZ_t):
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    u = ETDRK4(u, E, E_2, Q, f1, f2, f3, g)
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    return u

def Dealiased_Strang_split(u, E, E_2, Q, f1, f2, f3, g, dt, xi_p, dW_t, f_s, dZ_t, k):
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    u = Dealiased_ETDRK4(u, E, E_2, Q, f1, f2, f3, g, k)
    u = SSP33(u,dt/2,xi_p,dW_t,f_s,dZ_t)
    return u

# Simulation parameters
KS_params = {# KS equation, from Kassam Krefethen
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 150, "dt": 0.25 , "sigma": 0.001,
    "ic": 'Kassam_Trefethen_KS_IC', "method": 'ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}
KS_params_SALT = {# KS equation, from Kassam Krefethen with transport noise
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 1, "c_3": 0.0, "c_4": 1,
    "xmin": 0, "xmax": 32*jnp.pi, "nx": 256, "P": 1, "S": 1, "E": 1, "tmax": 150, "dt": 0.25, "sigma": 0.001,
    "ic": 'Kassam_Trefethen_KS_IC', "method": 'SS_ETDRK4_SSP33',
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}

KS_params_2 = {# KS equation, not from Kassam Krefethen 
    "equation_name" : 'Kuramoto-Sivashinsky', 
    "c_0": 0, "c_1": 1, "c_2": 0.1, "c_3": 0.0, "c_4": 1e-5,
    "xmin": 0, "xmax": 1, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 1, "dt": 0.0025 , "sigma": 0.001,
    "ic": 'gaussian', "method": 'SS_ETDRK4_SSP33',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

Heat_params = {# Heat equation. 
    "equation_name" : 'Heat equation', 
    "c_0": 0, "c_1": 0, "c_2": -0.1, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 1, "S": 0, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.1,
    "ic": 'sin', "method": 'SS_ETDRK4_SSP33',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

LinearAdvection_params = {# Linear Advection equation.
    "equation_name" : 'Linear Advection equation', 
    "c_0": 0.5, "c_1": 0, "c_2": 0, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 2, "tmax": 16, "dt": 1 / 128,  "sigma": 0.001,
    "ic": 'sin', "method": 'SS_ETDRK4_SSP33',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

Burgers_params={# Burgers equation. 
    "c_0": 0, "c_1": 1, "c_2": -1/256, "c_3": 0.0, "c_4": 0.0, 
    "xmin": 0, "xmax": 1,"nx": 256, "P": 10, "S": 9, "E": 1, "tmax": 16, "dt": 1 / 128,  "sigma": 0.001,
    "ic": 'sin', "method": 'SS_ETDRK4_SSP33',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

KDV_params = {# KdV equation. https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1, "c_4": 0.0,
    "xmin": -jnp.pi, "xmax": jnp.pi, "nx": 256, "P": 0, "S": 0, "E": 1, "tmax": 0.01, "dt": 2e-6, "sigma": 0.0,
    "ic": 'Kassam_Trefethen_KdV_IC_eq3pt1', "method": 'ETDRK4',
    "Advection_basis_name": 'none', "Forcing_basis_name": 'none'
}

KDV_params_2 = {# KdV equation. gaussian initial condition, small dispersion.
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 2e-5, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 256, "P": 10, "S": 1, "E": 5, "tmax": 10, "dt": 2e-3, "sigma": 0.1,
    "ic": 'gaussian', "method": 'SS_ETDRK4_SSP33', 
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'sin'
}

KDV_params_SALT = {# KdV equation. gaussian initial condition, small dispersion.
    "c_0": 0, "c_1": 1, "c_2": 0.0, "c_3": 1e-4, "c_4": 0.0,
    "xmin":0, "xmax":1, "nx": 128, "P": 1, "S": 0, "E": 2, "tmax": 10, "dt": 1e-5, "sigma": 0.0,
    "ic": 'gaussian', "method": 'SS_ETDRK4_SSP33', 
    "Advection_basis_name": 'sin', "Forcing_basis_name": 'none'
}

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    params = LinearAdvection_params#KDV_params_SALT#KDV_params_SALT
    params = KDV_params
    params = KS_params


    run_file_name = f'/Users/jmw/Documents/GitHub/Particle_Filter/config/auto_yaml/data.yml'
    with open(f'{run_file_name}', 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)

    # params = LinearAdvection_params
    xmin, xmax, nx, P, S, E, tmax, dt = params["xmin"],params["xmax"], params["nx"], params["P"],params["S"], params["E"], params["tmax"], params["dt"]
    #dx = (xmax - xmin) / nx
    #x = jnp.linspace(xmin, xmax, nx, endpoint=False)
    xf = jnp.linspace(xmin, xmax, nx+1)
    x = 0.5 * ( xf[1:] + xf[:-1] ) # cell centers
    dx = x[1]-x[0] # cell width
    u = initial_condition(x, E, params["ic"])
    u_benchmark = u
    
    k = jnp.fft.fftfreq(nx, dx, dtype=jnp.complex128) * 2 * jnp.pi
    L = -1j * k * params["c_0"] + k**2 * params["c_2"] + 1j*k**3 * params["c_3"] - k**4 * params["c_4"]
    g = -0.5j * k * params["c_1"]

    E_weights, E_2, Q, f1, f2, f3 = Kassam_Trefethen(dt, L, nx, M=64, R=1)# we rename the weights
    E_weights, Q, LW = LW_contour_trick(L, dt, M=64, R=1)
    nmax = round(tmax / dt)
    uu = jnp.zeros([E, nmax, nx])
    uu = uu.at[:, 0, :].set(u)

    stochastic_advection_basis = params["sigma"] * stochastic_basis_specifier(x, P, params["Advection_basis_name"])
    stochastic_forcing_basis   = params["sigma"] * stochastic_basis_specifier(x, S, params["Forcing_basis_name"])

    
    P = 23
    stochastic_advection_basis = params["sigma"] * stochastic_basis_specifier(x, P, 'sin')
    print("sin_shape",stochastic_advection_basis.shape)
    stochastic_advection_basis = params["sigma"] * stochastic_basis_specifier(x, P, 'none')
    print("none_shape",stochastic_advection_basis.shape)


    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    dW = jax.random.normal(key1, shape=(nmax, E, P))
    dZ = jax.random.normal(key2, shape=(nmax, E, S))
    print(stochastic_advection_basis.shape,dW.shape)

    for n in range(1, nmax):
        u = IFRK4(u,E_weights,E_2,g,k,L)
        #u = SETDRK1(u, E, Q, LW, g, k, stochastic_advection_basis,dW[n, :, :], cutoff_ratio=2/3)
        #u = Dealiased_SETDRK4(u, E_weights, E_2, Q, f1, f2, f3, g, k, stochastic_advection_basis,dW[n, :, :],dt)
        #u = Dealiased_IFSRK4(u,E_weights,E_2,g,k,L,stochastic_advection_basis,dW[n, :, :],dt)
        # dealiasing is important for the IFSRK4 method.

        #u = ETDRK4(u, E_weights, E_2, Q, f1, f2, f3, g)
        #u_benchmark = Dealiased_ETDRK4(u_benchmark, E_weights, E_2, Q, f1, f2, f3, g, k, cutoff_ratio=2/3)
        #u, E, E_2, Q, f1, f2, f3, g
        #u_benchmark = Dealiased_IFSRK4(u_benchmark,E_weights,E_2,g,k,L,stochastic_advection_basis,dW[n, :, :],dt)
        #u = Strang_split(u, E_weights, E_2, Q, f1, f2, f3, g, dt , stochastic_advection_basis, dW[n, :, :], stochastic_forcing_basis, dZ[n, :, :])
        uu = uu.at[:, n, :].set(u)
        if n % 10 == 0:  # Plot every 10 steps for better performance
            plt.clf()
            for e in range(E):
                plt.plot(x, u[e, :], label=f'Ensemble {e + 1}')
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