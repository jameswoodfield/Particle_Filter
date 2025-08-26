import os
os.environ["JAX_ENABLE_X64"] = "true"
import pytest
# import jax
from jax import config
import jax.numpy as jnp
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.ETD_KT_CM_JAX_Vectorised import initial_condition
from models.ETD_KT_CM_JAX_Vectorised import stochastic_basis_specifier

###################### Initial Condition ######################

@pytest.mark.parametrize("x", [
    jnp.linspace(0, 1, 64),       # coarse grid on [0,1]
    jnp.linspace(-1, 1, 201),     # symmetric domain
    jnp.linspace(0, 2*jnp.pi, 128) # periodic-like domain
])

@pytest.mark.parametrize("E", [1, 3, 5])

@pytest.mark.parametrize("name", [
    "sin",
    "compact_bump",
    "Kassam_Trefethen_KS_IC",
    "Kassam_Trefethen_KdV_IC_eq3pt1",
    "traveling_wave",
    "new_traveling_wave",
    "steep_traveling_wave",
    "very_steep_traveling_wave",
    "ultra_steep_traveling_wave",
    "gaussian",
])

def test_shape_and_determinism(x, E, name):
    ic1 = initial_condition(x, E, name)
    ic2 = initial_condition(x, E, name)
    # shape is (E, len(x)), and deterministic. 
    assert ic1.shape == (E, len(x)), f"Shape {ic1.shape} not as expected {(E, len(x))}"
    assert jnp.allclose(ic1, ic2, rtol=1e-12, atol=1e-12), f"IC not deterministic for {name} with x={x} and E={E}"


def test_sin_ic(x):
    E = 2
    ic = initial_condition(x, E, "sin")
    expected = jnp.sin(2 * jnp.pi * x)
    for row in ic:
        assert jnp.allclose(row, expected, rtol=1e-12, atol=1e-12)

def test_gaussian_ic(x):
    E = 1
    ic = initial_condition(x, E, "gaussian")
    A, x0, sigma = 1, 0.5, 0.1
    expected = A * jnp.exp(-((x - x0) ** 2) / (2 * sigma**2))
    assert jnp.allclose(ic[0], expected, rtol=1e-12, atol=1e-12)

def plot_basis(x, P, name):
    import matplotlib.pyplot as plt
    basis = stochastic_basis_specifier(x, P, name)
    plt.figure(figsize=(8, 5))
    for p in range(P):
        plt.plot(x, basis[p], label=f"$\\phi_{p+1}(x)$")
    plt.title(f"Stochastic basis: {name}, P={P}")
    plt.xlabel("x")
    plt.ylabel("basis functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_named_basis(name):
    x = jnp.linspace(0, 1, 256)
    plot_basis(x, P=2, name=name)

if __name__ == "__main__":
    #plot_named_basis('compact_bump')
    pytest.main()