import os
os.environ["JAX_ENABLE_X64"] = "true"
import pytest
from jax import config
import jax.numpy as jnp
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.ETD_KT_CM_JAX_Vectorised import stochastic_basis_specifier

@pytest.mark.parametrize("x", [
    jnp.linspace(0, 1, 50),
    jnp.linspace(-1, 1, 100),
])
@pytest.mark.parametrize("P", [1, 3, 5])
@pytest.mark.parametrize("name", ["sin", "nsin", "none", "cos", "random", "constant"])
def test_shape(x, P, name):
    basis = stochastic_basis_specifier(x, P, name)
    assert basis.shape == (P, len(x))


def test_sin_basis(x=jnp.linspace(0, 1, 10), P=2):
    basis = stochastic_basis_specifier(x, P, "sin")
    expected0 = 1/(1) * jnp.sin(2*jnp.pi*x*1)
    expected1 = 1/(2) * jnp.sin(2*jnp.pi*x*2)
    assert jnp.allclose(basis[0], expected0)
    assert jnp.allclose(basis[1], expected1)


def test_constant_basis(x=jnp.linspace(0, 1, 10), P=3):
    basis = stochastic_basis_specifier(x, P, "constant")
    assert jnp.all(basis == 1.0)


def test_random_is_reproducible(x=jnp.linspace(0, 1, 10), P=4):
    b1 = stochastic_basis_specifier(x, P, "random")
    b2 = stochastic_basis_specifier(x, P, "random")
    # Because the key is fixed at 0, both outputs should be identical
    assert jnp.allclose(b1, b2)


def test_invalid_name(x=jnp.linspace(0, 1, 10)):
    with pytest.raises(ValueError, match="not recognised"):
        stochastic_basis_specifier(x, 2, "not_a_valid_basis")



if __name__ == "__main__":

    pytest.main()