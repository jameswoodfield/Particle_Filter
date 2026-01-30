import os
os.environ["JAX_ENABLE_X64"] = "true"
import pytest
import jax
from jax import config
import jax.numpy as jnp
import sys
import xarray as xr
import xskillscore as xs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from metrics.ensemble import (
    convert_jnp_to_xarray,
    rmse,
    bias,
    crps,
    crps_internal,
)

# ----------------------------
# Fixtures for reproducibility
# ----------------------------
@pytest.fixture
def sample_data():
    time, member, space = 4, 3, 2
    key = jax.random.PRNGKey(0)
    true = jax.random.normal(key, shape=(time, space))
    ensemble = jax.random.normal(key, shape=(time, member, space))
    return true, ensemble

# ----------------------------
# convert_jnp_to_xarray
# ----------------------------
def test_convert_jnp_to_xarray_shapes(sample_data):
    true, ensemble = sample_data
    true_x, ensemble_x = convert_jnp_to_xarray(true, ensemble)

    assert isinstance(true_x, xr.DataArray)
    assert isinstance(ensemble_x, xr.DataArray)
    assert true_x.dims == ("time", "space")
    assert ensemble_x.dims == ("time", "member", "space")
    assert true_x.shape == true.squeeze().shape
    assert ensemble_x.shape == ensemble.shape


# ----------------------------
# RMSE
# ----------------------------
def test_rmse_shape_and_values(sample_data):
    true, ensemble = sample_data
    result = rmse(true, ensemble)
    assert result.shape == (true.shape[0],)
    # RMSE must be non-negative
    assert jnp.all(result >= 0)


# ----------------------------
# Bias
# ----------------------------
def test_bias_shape_and_mean_sign(sample_data):
    true, ensemble = sample_data
    result = bias(true, ensemble)
    assert result.shape == (true.shape[0],)
    # Bias can be positive or negative but finite
    assert jnp.all(jnp.isfinite(result))


# ----------------------------
# CRPS using xskillscore
# ----------------------------
def test_crps_xskillscore_shape(sample_data):
    true, ensemble = sample_data
    result = crps(true, ensemble)
    assert result.shape == (true.shape[0],)
    assert jnp.all(result >= 0)


# ----------------------------
# Internal CRPS vs xskillscore
# ----------------------------
def test_crps_internal_consistency(sample_data):
    """Compare internal CRPS implementation with xskillscore CRPS."""
    true, ensemble = sample_data
    crps_val = crps(true, ensemble)
    crps_int_val = crps_internal(true[:, None, :], ensemble)  # match shape handling

    assert crps_int_val.shape == (true.shape[0],)
    # Values should be close (within tolerance)
    assert jnp.allclose(crps_int_val, crps_val, rtol=1e-7, atol=1e-7)


# ----------------------------
# Deterministic consistency check
# ----------------------------
def test_zero_error_cases():
    """When ensemble perfectly matches truth, metrics should be zero."""
    time, member, space = 5, 4, 3
    signal = jnp.ones((time, space))
    ensemble = jnp.ones((time, member, space))

    rmse_val = rmse(signal, ensemble)
    bias_val = bias(signal, ensemble)
    crps_int_val = crps_internal(signal[:, None, :], ensemble)

    assert jnp.allclose(rmse_val, 0.0)
    assert jnp.allclose(bias_val, 0.0)
    assert jnp.allclose(crps_int_val, 0.0, atol=1e-7)


if __name__ == "__main__":
    pytest.main()