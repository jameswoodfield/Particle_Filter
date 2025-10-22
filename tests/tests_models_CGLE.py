import os
os.environ["JAX_ENABLE_X64"] = "true"
import sys
import pytest
from jax import random
import jax.numpy as jnp
from ml_collections import ConfigDict

# Add the models directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

from CGLE import CGLE_SETD_KT_CM_JAX, initial_condition, stochastic_basis_specifier, derived_params

# Define a fixture for the default parameters
@pytest.fixture
def default_params():
    return ConfigDict({
        "nx": 128,
        "xmin": -50.0,
        "xmax": 50.0,
        "dt": 0.1,
        "tmax": 1.0,
        "alpha": 0.0,
        "beta": 1.0,
        "S": 9,
        "E": 2,
        "nt": 10,
        "noise_magnitude": 0.001,
        "Forcing_basis_name": 'sin_sin',
        "initial_condition": 'chebfun',
        "method": 'Dealiased_SETDRK4_forced',
    })

# Test the CGLE_SETD_KT_CM_JAX class
def test_cgle_model_initialization(default_params):
    model = CGLE_SETD_KT_CM_JAX(default_params)
    assert model.params.nx == 128
    assert model.params.E == 2
    assert model.x.shape == (128,)
    assert model.ksq.shape == (128, 128)
    assert model.psi0.shape == (2, 128, 128)

def test_cgle_model_validate_params(default_params):
    with pytest.raises(ValueError, match="Number of ensemble members E must be greater than or equal to 1"):
        default_params.E = 0
        model = CGLE_SETD_KT_CM_JAX(default_params)
        model.validate_params()

def test_cgle_model_timestep_validate(default_params):
    with pytest.raises(ValueError, match="Time step dt must be positive"):
        default_params.dt = 0
        model = CGLE_SETD_KT_CM_JAX(default_params)
        model.timestep_validatate()
    with pytest.raises(ValueError, match="does not match tmax"):
        default_params.dt = 0.1
        default_params.nt = 5
        model = CGLE_SETD_KT_CM_JAX(default_params)
        model.timestep_validatate()

def test_cgle_model_run(default_params):
    model = CGLE_SETD_KT_CM_JAX(default_params)
    key = random.PRNGKey(0)
    n_steps = default_params.nt
    noise = (
        random.normal(key, shape=(n_steps, default_params.E, default_params.S)),
        random.normal(key, shape=(n_steps, default_params.E, default_params.S))
    )
    output = model.run(model.psi0, n_steps, noise, key)
    assert output[1].shape == (n_steps, default_params.E, default_params.nx, default_params.nx)

# Test the initial_condition function
@pytest.mark.parametrize("name", ['random', 'zero', 'chebfun'])
def test_initial_condition(name):
    xx, yy = jnp.meshgrid(jnp.linspace(-1, 1, 10), jnp.linspace(-1, 1, 10))
    E = 2
    ic = initial_condition(xx, yy, E, name)
    assert ic.shape == (E, 10, 10)

def test_initial_condition_invalid():
    xx, yy = jnp.meshgrid(jnp.linspace(-1, 1, 10), jnp.linspace(-1, 1, 10))
    with pytest.raises(ValueError, match="not recognised"):
        initial_condition(xx, yy, 1, 'invalid_name')

# Test the stochastic_basis_specifier function
@pytest.mark.parametrize("name", ['sin', 'sin_sin', 'x_sin', 'y_sin'])
def test_stochastic_basis_specifier(name):
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, 10), jnp.linspace(-1, 1, 10))
    P = 4
    basis = stochastic_basis_specifier(x, y, P, name)
    assert basis.shape == (P, 10, 10)

def test_stochastic_basis_specifier_invalid():
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, 10), jnp.linspace(-1, 1, 10))
    with pytest.raises(ValueError, match="not recognised"):
        stochastic_basis_specifier(x, y, 1, 'invalid_name')

# Test the derived_params function
def test_derived_params():
    params = {"xmax": 50.0, "xmin": -50.0, "tmax": 10.0, "dt": 0.1, "nx": 128}
    d_params = derived_params(params)
    assert d_params["L"] == 100.0
    assert d_params["Nt"] == 100
    assert d_params["dx"] == 100.0 / 128
