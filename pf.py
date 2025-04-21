""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
    Modified by Wentao Yuan for CSE571: Probabilistic Robotics (Spring 2022)
"""

import numpy as np

from utils import minimized_angle
from soccer_field import Field


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def move_particles(self, env: Field, u):
        """Update particles after taking an action
        param:
            env: environment object
            u: action, shape (3, 1)
        return:
            new_particles: particle states (2D position + 1D orientation), shape (n, 3)
        Env API may be useful:
            env.forward(x, u)
                x: [x, y, theta]
                u: [rot1, trans, rot2]
            env.sample_noisy_action(u, alpha)
                u: desired action
                alphas: noise parameters for odometry motion model (default: data alphas)
        """
        new_particles = self.particles

        # YOUR IMPLEMENTATION HERE

        new_particles = np.zeros_like(self.particles)
        for i in range(self.num_particles):
            u_noisy = env.sample_noisy_action(u, self.alphas)
            x_new_col = env.forward(self.particles[i, :], u_noisy)
            new_particles[i, :] = x_new_col.ravel()
        self.particles = new_particles
        
        # YOUR IMPLEMENTATION END HERE
        return new_particles

    def update(self, env: Field, u, z, marker_id):
        """Update the state estimate after taking an action and receiving 
        a landmark observation.
        param:
            u: action, shape (3, 1)
            z: landmark observation, shape (1, 1)
            marker_id: landmark ID, int
        return: 
            mean: updated mean, shape (3, 1)
            cov: updated covariance matrix, shape (3, 3)
        Env API may be useful:
            env.observe(x, marker_id)
                x: [x, y, theta]
                marker_id: int
            env.likelihood(innovation, beta)
                innovation: difference between expected and observed bearing angle
                beta: noise parameters for landmark observation model            
        """
        particles = self.move_particles(env, u)
        mean, cov = None, None
        # YOUR IMPLEMENTATION HERE

        # compute weights
        weights = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            z_hat = env.observe(particles[i, :], marker_id)
            innov = minimized_angle(z - z_hat)
            weights[i] = env.likelihood(innov, self.beta)

        # normalize
        weights += 1e-300
        weights /= np.sum(weights)
        self.weights = weights

        # resample
        new_particles = self.resample(particles, weights)
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

        mean, cov = self.mean_and_variance(self.particles)
        
        # YOUR IMPLEMENTATION END HERE
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.
        param:
            particles: particle states (2D position + 1D orientation), shape (n, 3)
            weights: particle weights, shape (n,)
        return:
            new_particles: particle states (2D position + 1D orientation), shape (n, 3)
        """
        new_particles = None
        # YOUR IMPLEMENTATION HERE

        M = self.num_particles
        new_particles = np.zeros_like(particles)

        r = np.random.rand() / M
        c = weights[0]
        i = 0

        for m in range(M):
            u = r + m * (1.0 / M)
            while u > c:
                i += 1
                c += weights[i]
            new_particles[m, :] = particles[i, :]
        
        # YOUR IMPLEMENTATION END HERE
        return new_particles

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.
        param: 
            particles: particle states (2D position + 1D orientation), shape (n, 3)
        return:
            mean: mean, shape (3, 1)
            cov: covariance matrix, shape (3, 3)
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum(),
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles
        cov += np.eye(particles.shape[1]) * 1e-6  # Avoid bad conditioning 

        mean = mean.reshape((-1, 1))
        return mean, cov
