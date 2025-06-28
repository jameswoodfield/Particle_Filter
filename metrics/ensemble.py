import xarray as xr
import xskillscore as xs
import jax.numpy as jnp
import jax

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
    this can be the signal used, or the observations. 
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

def crps_internal(signal, ensemble):
    """Calculate the Continuous Ranked Probability Score (CRPS) for an ensemble prediction,
    this is the internal version, which does not use xskillscore. Based on some code for https://arxiv.org/pdf/2502.12775
    this allows jax implementation and is more efficient and differentiable."""
    #print("CRPS internal shape:", signal.shape, ensemble.shape)

    #reshape from (time, member, space), into (time, space, member)
    ensemble = jnp.moveaxis(ensemble, 1, 2)  # Move member dimension to the last position
    signal = jnp.moveaxis(signal, 1, 2)[:,:,0]
    # print("CRPS internal shape:", signal.shape, ensemble.shape)
    # print("required shape: (time, space, member)")
    # print("required shape: (time, space)")

    @jax.jit
    def My_CRPS(obs,forcast):
        #Gneiting
        # assumes for example:  (10,10),(10,10,E) 
        # based on the following representation of the crps score. 
        observations = jnp.asarray(obs)
        forecasts = jnp.asarray(forcast)
        observations = observations[... , jnp.newaxis]
        score = jnp.mean(jnp.abs(forecasts - observations), -1)
        forecasts_diff = (jnp.expand_dims(forecasts, -1) -
                      jnp.expand_dims(forecasts, -2))
        score = score - 0.5 * jnp.mean( jnp.abs(forecasts_diff),
                                       axis=(-2, -1))
        return score

    crps_values = My_CRPS(signal, ensemble)
    return jnp.mean(crps_values, axis=1)  # Average over space dimension