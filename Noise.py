import numpy as np
from math import sqrt

class OU_Noise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2,):
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def sample_noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x