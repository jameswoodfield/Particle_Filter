import pytest
import jax
from jax import config
import jax.numpy as jnp
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.ETD_KT_CM_JAX_Vectorised import initial_condition
from models.ETD_KT_CM_JAX_Vectorised import dealias_using_k
from models.ETD_KT_CM_JAX_Vectorised import stochastic_basis_specifier

###################### Initial Condition ######################

def test_initial_condition_sin():
    x = jnp.linspace(0, 1, 5)
    E = 3
    name = 'sin'
    expected_output = jnp.tile(jnp.sin(2 * jnp.pi * x), (E, 1))
    assert jnp.array_equal(initial_condition(x, E, name), expected_output)

def test_initial_condition_compact_bump():
    x = jnp.linspace(0, 1, 5)
    E = 3
    name = 'compact_bump'
    x_min = 0.2; x_max = 0.3
    s = (2 * (x - x_min) / (x_max - x_min)) - 1
    function_of_x = jnp.exp(-1 / (1 - s**2)) * (jnp.abs(s) < 1)
    expected_output = jnp.tile(function_of_x, (E, 1))
    assert jnp.array_equal(initial_condition(x, E, name), expected_output)

def test_initial_condition_invalid_name():
    x = jnp.linspace(0, 1, 5)
    E = 3
    name = 'invalid'
    with pytest.raises(ValueError):
        initial_condition(x, E, name)

def test_initial_condition_empty_x():
    x = jnp.array([])
    E = 3
    name = 'sin'
    expected_output = jnp.tile(jnp.sin(2 * jnp.pi * x), (E, 1))
    assert jnp.array_equal(initial_condition(x, E, name), expected_output)
    assert initial_condition(x, E, name).shape == (E, 0)

def test_initial_condition_zero_ensemble():
    nx = 5
    x = jnp.linspace(0, 1, nx)
    E = 0
    name = 'sin'
    expected_output = jnp.tile(jnp.sin(2 * jnp.pi * x), (E, 1))
    assert jnp.array_equal(initial_condition(x, E, name), expected_output)
    assert initial_condition(x, E, name).shape == (0, nx)

###################### stochastic basis ######################

def test_stochastic_basis_dimensions():
    nx = 5
    x = jnp.linspace(0, 1, nx)
    P = 23
    ans_1 =  stochastic_basis_specifier(x, P, 'sin')
    ans_2 =  stochastic_basis_specifier(x, P, 'none')
    assert ans_1.shape == (P, nx)
    assert ans_2.shape == (P, nx)

def test_stochastic_basis_invalid_name():
    nx = 5
    x = jnp.linspace(0, 1, nx)
    P = 23
    name = 'invalid'
    with pytest.raises(ValueError):
        stochastic_basis_specifier(x, P, name)
    
###################### Dealiasing tests ######################
def dealias(u_hat, k, cutoff_ratio=2/3):# for testing dealiasing
    max_k = jnp.max(jnp.abs(k))
    cutoff = max_k * cutoff_ratio
    mask = jnp.abs(k) <= cutoff
    u_hat = u_hat * mask
    return u_hat

def test_dealiasing_equivalence():
    jax.config.update("jax_enable_x64", True)# set to True for double precision
    E = 1  # Number of ensemble members
    N = 128  # Size of the 1D grid
    cutoff_ratio = 1  # Common cutoff ratio for dealiasing
    xmin = 0; xmax = 1; 
    xf = jnp.linspace(xmin, xmax, N+1)
    x = 0.5 * ( xf[1:] + xf[:-1] )# this is better, 
    dx = x[1]-x[0]
    u = initial_condition(x , E, 'sin') # (E,N) sized array
    u_squared = jnp.real(u)**2  # Compute u^2 in real space
    u_hat_squared = jnp.fft.fft(u_squared,axis=-1)  # Compute u^2 in spectral space
    k = 2 * jnp.pi * jnp.fft.fftfreq(N, dx)# (-N/2 to N/2) * 2 * pi, spaced by dx.
    spectral_field_dealiased = dealias_using_k(u_hat_squared, k, cutoff_ratio=cutoff_ratio)
    spectral_field_dealiased_new = dealias(u_hat_squared, k, cutoff_ratio=cutoff_ratio)
    # Plot the dealiased fields
    # (u^2)_x = 2 * u * u_x 
    # u is sin(2 * pi * x), 
    # (u^2)_x, (sin(2 * pi * x)^2)_x = 2 * sin(2 * pi * x) * (2 * pi) * cos (2 pi x)
    ans = 2 * jnp.sin(2 * jnp.pi * x) * 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)
    m1 = jnp.real(jnp.fft.ifft(k * 1j * spectral_field_dealiased_new, axis=-1))[0,:]
    m2 = jnp.real(jnp.fft.ifft(k * 1j * spectral_field_dealiased, axis=-1))[0,:]
    m3 = jnp.real(jnp.fft.ifft(k * 1j * u_hat_squared, axis=-1))[0,:]
    # visualise the results: uncomment the following lines to plot. 
    # plt.plot(x, m2)
    # plt.plot(x, m1)
    # plt.plot(x, ans) 
    # plt.show()
    # plt.clf()
    # plt.plot(x,m3-ans)
    # plt.show()
    # Check that the dealiased fields are equivalent up to a tolerance
    assert jnp.allclose(m1, m2, atol=1e-6), \
         "test 1"
    assert jnp.allclose(m3, ans, atol=1e-6), \
         "test 2"
    assert jnp.allclose(m1, ans, atol=1e-6), \
         "test 3"

#test_initial_condition_sin()
if __name__ == "__main__":
    pytest.main()