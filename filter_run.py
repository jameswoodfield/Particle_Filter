import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from models.ETD_KT_CM_JAX_Vectorised import ETD_KT_CM_JAX_Vectorised, ic, KS_params, KDV_params
from filters.filter import ParticleFilter

def run_filter(resampling, observation_locations=None):
    signal_params = ConfigDict(KDV_params)
    signal_params.update(E=1)
    ensemble_params = ConfigDict(KDV_params)
    ensemble_params.update(E=10)
    ensemble_params.update(sigma=0.001)
    signal_model = ETD_KT_CM_JAX_Vectorised(signal_params)
    ensemble_model = ETD_KT_CM_JAX_Vectorised(ensemble_params)

    initial_signal = ic(signal_model.x, signal_params.E)
    initial_ensemble = ic(ensemble_model.x, ensemble_params.E) 
    #initial_ensemble += 0.01 * jax.random.normal(jax.random.PRNGKey(56), initial_ensemble.shape) # adding this is totally "unphysical"
    pf = ParticleFilter(
        n_particles = ensemble_params.E,
        n_steps = 100,
        n_dim = initial_signal.shape[-1],
        forward_model = ensemble_model,
        signal_model = signal_model,
        sigma = 0.1,
        seed = 11,
        resampling=resampling,
        observation_locations=observation_locations,
    )
    _, all = pf.run(initial_ensemble, initial_signal, 10)
    signal = jnp.concatenate([initial_signal[None,...], all[1]], axis=0)
    particles = jnp.concatenate([initial_ensemble[None,...], all[0]], axis=0)
    observations = all[2]
    return signal, particles, signal_model.x, observations