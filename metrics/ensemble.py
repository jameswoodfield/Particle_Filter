import xarray as xr
import xskillscore as xs
import jax.numpy as jnp

def convert_jnp_to_xarray(true, ensemble):
    """Convert JAX arrays to xarray DataArrays for CRPS calculation.
    Args:
        true (jnp.ndarray): True signal with shape (time, space).
        ensemble (jnp.ndarray): Ensemble predictions with shape (time, member, space).
    Returns:
        tuple: Two xarray DataArrays, one for the true signal and one for the ensemble.
    """
    true_x = xr.DataArray(true.squeeze(), dims=['time', 'space'])
    ensemble_x = xr.DataArray(ensemble, dims=['time', 'member', 'space'])
    return true_x, ensemble_x

def rmse(observations, ensemble):
    """Calculate the Root Mean Square Error (RMSE) between observations and ensemble predictions.
    Args:
        observations (jnp.ndarray): Observed signal with shape (time, space).
        ensemble (jnp.ndarray): Ensemble predictions with shape (time, member, space).
    Returns:
        jnp.ndarray: Sparial Mean (RMSE over member dimension) for each time step, shape (time,).
    """
    square_error = (ensemble - observations)**2
    mean_square_error = jnp.mean(square_error, axis=1)# average over ensemble members
    root_mean_square_error = jnp.sqrt(mean_square_error)
    mean_rmse_over_space = jnp.mean(root_mean_square_error, axis=1)# average over space dimension
    return mean_rmse_over_space


def bias(signal, ensemble):
    ensemble_mean = jnp.mean(ensemble, axis=1)
    truths = signal.squeeze()
    bias = jnp.mean(ensemble_mean - truths, axis=1)
    return bias

def crps(signal, ensemble):
    signal_x, ensemble_x = convert_jnp_to_xarray(signal, ensemble)
    crps = xs.crps_ensemble(signal_x, ensemble_x, dim=['space'])
    return jnp.array(crps.values)