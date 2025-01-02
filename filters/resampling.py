import jax
import jax.numpy as jnp

resamplers = {}

def register_resampler(name):
    def register_fn(fn):
        resamplers[name] = fn
        return fn
    return register_fn

# @jax.jit
# def observation(data,particles,sigma):
#     """data[:,:] returns noisy observations, and output. """
#     _,nx_h = data.shape
#     E,nx = particles.shape
#     freq_x = int(nx_h/nx)# 4
#     observation_slice_data = jnp.s_[:,::freq_x] # e,::
#     observations = data[observation_slice_data] + sigma*np.random.randn(1,nx)
#     out = particles[:,:] - observations
#     return observations, out

# @jax.jit
# def update_weights(q,data,weights,sigma):
#     particles = q
#     E,nx = particles.shape
#     observations, out = observation(data,particles,sigma)
#     distance = jnp.linalg.norm(out, ord=2,axis=1)
#     weights = weights*jnp.exp(-distance / sigma) #  exp(log(weights)*temperature)) # prior*likelyhood, normalised.
#     weights += 1.e-16     # avoid round-off to zero
#     _weights = weights/jnp.sum(weights) # normalize
#     return _weights

# def neff(weights):
#     return 1. / jnp.sum(weights**2)

@register_resampler('multinomial')
def multinomial_resample(particles, weights, key):
    """
    #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
    """
    E = len(weights)
    cumulative_sum = jnp.cumsum(weights)
    cumulative_sum = cumulative_sum.at[-1].set(1.)  
    index = jnp.searchsorted(cumulative_sum, jax.random.uniform(key, shape=(E,)))

    return particles[index]

# def systematic_resample(weights):
#     """The systematic resampling algorithm as implmented in: 
#     #https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
#     """
#     cumulative_sum = np.cumsum(weights)
#     E = len(weights)
#     positions = ( np.arange(E) + np.random.rand()*np.ones(E) ) / E
#     indexes = np.zeros(E, 'i')
#     i, j = 0, 0
#     while i < E:
#         if positions[i] < cumulative_sum[j]:
#             indexes[i] = j
#             i += 1
#         else:
#             j += 1
#     return indexes

@register_resampler('systematic')
def systematic_resample_2(particles, weights, key):
    """The systematic resampling algorithm"""
    cumulative_sum = jnp.cumsum(weights)
    E = len(weights)
    rand_positions = ( jnp.arange(E) + jax.random.uniform(key, shape=(E,))*jnp.ones(E) ) / E
    indexes = jnp.searchsorted(cumulative_sum, rand_positions, side='left')
    return particles[indexes]

# def resample_from_index(particles, weights, indexes):
#     """resample particles from indexes and reset weights"""
#     E = len(weights)
#     particles = particles.at[:,:].set(particles[indexes,:])
#     weights = weights.at[:].set( jnp.ones_like(weights)/E )
#     return particles, weights

@register_resampler('no_resampling')
def no_resampling(particles, weights, key):
    return particles

@register_resampler('default')
def default_resampler(particles, weights, key):
    """Default resampler from Dan's book (need to find out if it has a name...)"""
    parents = branching(weights, key)
    return particles[parents]

# Helper functions for the default resampler
@jax.jit
def branching(weights, key):
    """Perform a branching step"""
    #TODO Change to eliminate the positions array
    positions = jnp.arange(len(weights), dtype=jnp.int32)
    offspring = compute_offspring(weights, key)
    parent = reindex(positions, offspring)

    return parent

@jax.jit
def compute_offspring(weights, key):
    """Compute the number of offspring for each particle"""

    def inner_cond_true(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(unif[j] < 1. - (((n*weights[j]) % 1)/((g % 1)+1e-10)),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j])),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + h - jnp.floor(g)), # pylint: disable=line-too-long
                    offspr, j, n, weights, h, g  )
        return offspr

    def inner_cond_false(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(unif[j] < 1. - (1. - ((n*weights[j]) % 1) ) / (1 - (g % 1)+1e-10),
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + 1.), # pylint: disable=line-too-long
                    lambda offspr, j, n, weights, h, g: offspr.at[j].set(jnp.floor(n*weights[j]) + h - jnp.floor(g)), # pylint: disable=line-too-long
                    offspr, j, n, weights, h, g  )
        return offspr

    def outer_cond(unif, offspr, j, n, weights, h, g):
        offspr = jax.lax.cond(((n*weights[j]) % 1) + ((g - n*weights[j]) % 1) < 1.,
                    inner_cond_true,
                    inner_cond_false,
                    unif, offspr, j, n, weights, h, g  )
        return offspr

    def body_fun(i, val):
        unif, offspr, n, weights, h, g = val
        offspr = outer_cond(unif, offspr, i, n, weights, h, g)
        g = g - n*weights[i]
        h = h - offspr[i]
        return unif, offspr, n, weights, h, g

    n = weights.shape[0]
    unif = jax.random.uniform(key, shape=(n-1,))
    g = n
    h = n
    offspr = jnp.empty_like(weights)

    unif, offspr, n, weights, h, g = jax.lax.fori_loop(0, n-1, body_fun, (unif, offspr, n, weights, h, g)) # pylint: disable=line-too-long

    offspr = offspr.at[n-1].set(h)
    offspr = offspr.astype(int)
    return offspr

@jax.jit
def reindex(positions, offspring):
    """Reindex the particles according to their offspring"""

    # TODO Change to eliminate the positions array

    def body_fun_inner(i, val):
        pos, parent, r, positions, j = val
        pos = pos.at[r+i].set(positions[j])
        parent = parent.at[r+i].set(j) # parent of particle r+i is particle j
        return pos, parent, r, positions, j

    def body_fun_outer(j, val):
        pos, parent, r, positions, offspring = val
        pos, parent, r, positions, _ = jax.lax.fori_loop(0, offspring[j], body_fun_inner, (pos, parent, r, positions, j)) # pylint: disable=line-too-long
        r = r + offspring[j]
        return pos, parent, r, positions, offspring

    n = len(positions)
    pos = jnp.empty_like(positions)
    parent = jnp.empty_like(offspring)
    r = 0
    pos, parent, r, positions, offspring = jax.lax.fori_loop(0, n, body_fun_outer, (pos, parent, r, positions, offspring)) # pylint: disable=line-too-long

    return parent