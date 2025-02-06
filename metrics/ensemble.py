import xarray as xr
import xskillscore as xs
import jax.numpy as jnp

def convert_jnp_to_xarray(true, ensemble):
    true_x = xr.DataArray(true.squeeze(), dims=['time', 'space'])
    ensemble_x = xr.DataArray(ensemble, dims=['time', 'member', 'space'])
    return true_x, ensemble_x

def rmse(observations, ensemble):
    square_error = (ensemble - observations)**2
    mean_square_error = jnp.mean(square_error, axis=1)
    root_mean_square_error = jnp.sqrt(mean_square_error)
    mean_rmse_over_space = jnp.mean(root_mean_square_error, axis=1)
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