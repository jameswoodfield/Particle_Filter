import jax
import jax.numpy as jnp

@jax.jit
def observation(data,particles,sigma):
    """data[:,:] returns noisy observations, and output. """
    _,nx_h = data.shape
    E,nx = particles.shape
    freq_x = int(nx_h/nx)# 4
    observation_slice_data = jnp.s_[:,::freq_x] # e,::
    observations = data[observation_slice_data] + sigma*np.random.randn(1,nx)
    out = particles[:,:] - observations
    return observations, out

@jax.jit
def update_weights(q,data,weights,sigma):
    particles = q
    E,nx = particles.shape
    observations, out = observation(data,particles,sigma)
    distance = jnp.linalg.norm(out, ord=2,axis=1)
    weights = weights*jnp.exp(-distance / sigma) #  exp(log(weights)*temperature)) # prior*likelyhood, normalised.
    weights += 1.e-16     # avoid round-off to zero
    _weights = weights/jnp.sum(weights) # normalize
    return _weights

def neff(weights):
    return 1. / jnp.sum(weights**2)


def multinomial_resample(weights):
    """
    #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
    """
    E = len(weights)
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum = cumulative_sum.at[-1].set(1.)  
    index = jnp.searchsorted(cumulative_sum, np.random.rand(E))  
    return index

def systematic_resample(weights):
    """The systematic resampling algorithm as implmented in: 
    #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
    """
    cumulative_sum = np.cumsum(weights)
    E = len(weights)
    positions = ( np.arange(E) + np.random.rand()*np.ones(E) ) / E
    indexes = np.zeros(E, 'i')
    i, j = 0, 0
    while i < E:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def systematic_resample_2(weights):
    """The systematic resampling algorithm, where search sorted performs the above, 
    Currently the np.random.rand"""
    cumulative_sum = jnp.cumsum(weights)
    E = len(weights)
    positions = ( jnp.arange(E) + np.random.rand()*jnp.ones(E) ) / E
    #key, subkey = random.split(key)
    #positions = ( jnp.arange(E) + jax.random.uniform()*jnp.ones(E) ) / E
    indexes = jnp.searchsorted(cumulative_sum, positions, side='left')
    return indexes


def resample_from_index(particles, weights, indexes):
    """resample particles from indexes and reset weights"""
    E = len(weights)
    particles = particles.at[:,:].set(particles[indexes,:])
    weights = weights.at[:].set( jnp.ones_like(weights)/E )
    return particles, weights