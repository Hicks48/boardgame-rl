import random

def linear_decay(num_episodes):
    return lambda i_episode: float(i_episode) / float(num_episodes)

class EpsilonGreedy:
    def __init__(
        self,
        initial_epsilon,
        epsilon_decay_fn = lambda i: 0,
        random_sampler = lambda: random.uniform(0, 1)
    ):
        self.epsilon = initial_epsilon
        self.epsilon_decay_fn = epsilon_decay_fn
        self.random_sampler = random_sampler

    def sample_should_explore(self, i_episode = 0):
        return self.random_sampler() < (self.epsilon - self.epsilon_decay_fn(i_episode))

    def sample_should_exploit(self, iteration = 0):
        return not self.sample_should_explore(iteration)
