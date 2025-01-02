import pytest
import jax.numpy as jnp
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.ETD_KT_CM_JAX_Vectorised import initial_condition

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

#test_initial_condition_sin()
if __name__ == "__main__":
    pytest.main()