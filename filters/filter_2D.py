import os
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
from .resampling import resamplers



class EnsembleKalmanFilter_2D:
    # Simple implementation of the Stochastic Ensemble Kalman Filter (EnKF) as introduced by Evensen (1994).
    def __init__(self, n_particles, n_steps, n_dim, forward_model, signal_model, sigma, observation_locations=None, inflation_factor=1.0,localization_radius=1.0,relaxation_factor=1.0):
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.n_dim = n_dim
        self.fwd_model = forward_model
        self.signal_model = signal_model
        self.sigma = sigma
        # self.observation_locations = slice(observation_locations) if observation_locations is None else tuple(observation_locations)
        self.inflation_factor = inflation_factor # Default inflation factor, can be adjusted later
        self.localization_radius = localization_radius # Localisation radius for Gaspari-Cohn localisation function
        self.relaxation_factor = relaxation_factor # Relaxation factor for particle updates
        # Observation locations as list of (i, j) tuples
        if observation_locations is None:
            self.observation_locations = []
        else:
            self.observation_locations = jnp.array(observation_locations) 
        # self.obs_indices = [i * ny + j for (i, j) in self.observation_locations]

    def advance_signal(self, signal_position, key):
        signal, _ = self.signal_model.run(signal_position, self.n_steps, None, key)
        return signal

    def predict(self, particles, key):
        prediction, _ = self.fwd_model.run(particles, self.n_steps, None, key)# final,all.
        return prediction

    def observation_from_signal(self, signal, key):
        key, subkey = jax.random.split(key)

        def H(x):
            i_idx = self.observation_locations[:, 0]
            j_idx = self.observation_locations[:, 1]
            return jnp.abs(x[0, 0, i_idx, j_idx])
        
        obs = H(signal)  # Apply the observation operator H
        obs = obs + self.sigma * jax.random.normal(subkey, shape=obs.shape)
        return obs

    def update(self, particles, observation, key):
        """
        particles: (n_particles, nx, ny)
        observation: (n_obs,)
        """
        initial_particles = particles
        #print(f"Updating particles with shape: {particles.shape} and observation shape: {observation.shape}")
        _, n_particles, nx, ny = particles.shape #(1, n_particles, nx, ny)
        flat_particles = particles.reshape(n_particles, nx * ny)
        #print(f"Flattened particles shape: {flat_particles.shape}") # (n_particles, nx*ny)

        # Observation operator H
        H = lambda x: jnp.abs(x[...,self.observation_locations[:, 0], self.observation_locations[:, 1]])
        n_obs = len(self.observation_locations[:, 0])

        mean = jnp.mean(flat_particles, axis=0)
        X = flat_particles - mean  # anomalies

        Y = jax.vmap(H)(particles).reshape(n_particles, n_obs )  # shape (n_particles, n_obs)
        #print(f"Y shape after applying observation operator: {Y.shape}")  # (n_particles, n_obs)
        y_mean = jnp.mean(Y, axis=0)
        Y_perturb = Y - y_mean
        # Observation noise perturbation
        key, subkey = jax.random.split(key)
        obs_perturb = self.sigma * jax.random.normal(subkey, shape=observation.shape)
        #y_obs = Y + obs_perturb
        y_obs = observation + obs_perturb # shape(64) associated with the 64 gridpoints specified by self.observation_locations = 64,2
        #jax.vmap(H)(observation).reshape(n_particles, n_obs )

        # Compute covariances
        Pf_HT = X.T @ Y_perturb / (n_particles - 1)  # (nx*ny, n_obs)
        S = Y_perturb.T @ Y_perturb / (n_particles - 1) + self.sigma**2 * jnp.eye(Y.shape[1])
        K = Pf_HT @ jnp.linalg.inv(S)  # (nx*ny, n_obs)

        # Update
        innovations = y_obs - Y
        update = jax.vmap(lambda innov: K @ innov)(innovations)  # (n_particles, nx*ny)
        flat_particles = flat_particles + update

        particles = flat_particles.reshape(n_particles, nx, ny)
        particles = particles[jnp.newaxis, ...]  # Add time dimension back

        # relaxation option
        particles = initial_particles + (particles-initial_particles) * self.relaxation_factor  # Apply inflation factor
        
        # inflate particles
        mean_unflat = mean.reshape(1, nx, ny)  # Reshape mean to match particles shape
        particles = mean_unflat + (particles - mean_unflat) * self.inflation_factor
        
        return particles #initial_particles#particles

    def run_step(self, particles, signal, key):
        key, obs_key, update_key, pred_key, sig_key = jax.random.split(key, 5)
        signal = self.advance_signal(signal, sig_key)# signal model run
        particles = self.predict(particles, pred_key)# forward model run
        observation = self.observation_from_signal(signal, obs_key)
        particles = self.update(particles, observation, update_key)
        return particles, signal, observation

    def run(self, initial_particles, initial_signal, n_total, key):
        def scan_fn(val, i):
            particles, signal, key = val
            key, next_key = jax.random.split(key)
            particles, signal, observation = self.run_step(particles, signal, key)
            return (particles, signal, next_key), (particles, signal, observation)
        
        final, all = jax.lax.scan(scan_fn, (initial_particles, initial_signal, key), jnp.arange(n_total))
        return final, all

    def run_step_no_da(self, particles, signal, key):
        key, pred_key, sig_key = jax.random.split(key, 3)
        signal = self.advance_signal(signal, sig_key)
        particles = self.predict(particles, pred_key)
        observation = signal*0
        return particles, signal, observation
    
    def run_no_da(self, initial_particles, initial_signal, n_total, key):
        def scan_fn(val, i):
            particles, signal, key = val
            key, next_key = jax.random.split(key)
            particles, signal, observation = self.run_step_no_da(particles, signal, key)
            return (particles, signal, next_key), (particles, signal, observation)
        
        final, all = jax.lax.scan(scan_fn, (initial_particles, initial_signal, key), jnp.arange(n_total))
        return final, all
        