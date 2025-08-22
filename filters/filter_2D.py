import os
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
from .resampling import resamplers

## TODO: generalise to allow an observation operator H, as an input function. 

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
        forecast_particles = particles
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
        particles = forecast_particles + (particles-forecast_particles) * self.relaxation_factor  # Apply inflation factor
        
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
    



class Ensemble_Transform_Kalman_Filter_2D:
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
            #jnp.abs(
            return x[0, 0, i_idx, j_idx]
        
        obs = H(signal)  # Apply the observation operator H
        obs = obs #+ self.sigma * jax.random.normal(subkey, shape=obs.shape)
        return obs

    def update(self, particles, observation, key):
        """
        particles: (1, n_particles, nx, ny)
        observation: (n_obs,)
        let E = ensemble size
        let n = state dimension = nx*ny, flattened.
        let p = observation dimension
        let H = observation operator, maps from R^n to R^p
        let R = observation noise covariance, p x p
        let Pf = forecast covariance, n x n
        let K = Kalman gain, n x p

        step zero: flatten particles to shape (E, n)
        step one: forecast mean: x_f = 1/E sum(x_i) in R^n, (n,) 
        step two: forecast anomalies: X = [x_1 - x_f, ..., x_E - x_f] in R^(n x E), 
        step three: predicted observation ensemble:
        """
        forecast_particles = particles
        _, E, nx, ny = particles.shape #(1, n_particles, nx, ny)
        n = nx * ny
        flat_particles = particles.reshape(E, n) 
        #jnp.abs()
        H = lambda x: x[...,self.observation_locations[:, 0], self.observation_locations[:, 1]]
        p = len(self.observation_locations[:, 0])
        
        # step 1: Compute the  forecast mean and forecast anomalies
        mean_f = jnp.mean(flat_particles, axis=0) # (n,)
        X_f = (flat_particles - mean_f).T / jnp.sqrt(E-1)  # (n, E) 
        print(f"X shape: {X_f.shape}, {n}, {E}")  # (n, n_particles)
        
        # step 2: Compute the predicted observation ensemble in obs space.
        Y = jax.vmap(H)(particles).reshape(E,p) # (E, p)
        y_mean = jnp.mean(Y, axis=0)# (p,)
        Y_perturb = ((Y - y_mean).T)/ jnp.sqrt(E-1) 
        print(f"Y shape: {Y_perturb.shape}, {p}, {E}")  # (n_particles)

        # step 3: Compute the “analysis” weights
        S = Y_perturb # (p, E)

        M = jnp.eye(E) + 1/self.sigma**2 * S.T @ S # (E,p)(p,E) = (E,E)

        def inv_psd(A,eps=1e-12):
            """ Compute the inverse of a positive semi-definite matrix using its eigen decomposition. """
            eigvals, eigvecs = jnp.linalg.eigh(A)
            eigvals_clipped = jnp.maximum(eigvals, eps)
            inv_sqrt = 1.0 / jnp.sqrt(eigvals_clipped)        
            A_inv_half = (eigvecs * inv_sqrt) @ eigvecs.T
            # one could do (eigvecs * inv_sqrt**2) @ eigvecs.T for the explicit inverse, for w =M^{-1}b 
            return A_inv_half
        T = inv_psd(M)  # (E, E)

        # here is the first use of the observation:
        inovation_obs = observation - y_mean  # (p,)
        #print(f"d shape: {inovation_obs}")
        # we solbe M^{-1}b , by multiplying by the inverse.

        # approach one, explicitely forming the inverse:
        #M_1 = 1/self.sigma**2 * inovation_obs #(p,)
        #M_2 = S.T @ M_1  # (E,) = (E, p)(p, )
        #w = (M@M) @ M_2  # (E,) = (E, E)(E, ) #vector of ensemble space weights
        
        # approach 2, not sure if legit.
        #w = T & (S.T @ (1/self.sigma**2 * inovation_obs)) # equivalent to above line, but more efficient.
        # approach 3 solve linear system:
        w = jnp.linalg.solve(M, S.T @ (1/self.sigma**2 * inovation_obs))  # (E,) = (E,E)(E,)
        
        # X_f @ w can be thought of as Kalman gain times innovation, but in ensemble subspace.
        mean_a = mean_f + X_f @ w  # (n,) = (n,)+(n,E)(E,)
        X_a = X_f @ T  # (n, E) = (n, E)(E, E)
        e = jnp.ones(E)
        x_a = jnp.einsum('i,j->ij',mean_a,e ) + jnp.sqrt(E-1) * X_a  # (n, E) = (n,)@(E,).T+(n,E)
        print(f"x_a shape: {x_a.shape}")  # (n, E)
        particles = flat_particles.reshape(E, nx, ny)
        particles = particles[jnp.newaxis, ...]  # Add time dimension back
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