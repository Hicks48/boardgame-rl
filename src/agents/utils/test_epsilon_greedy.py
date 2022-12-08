import unittest
from epsilon_greedy import EpsilonGreedy

class WhenRandomNumberIsLessThanEpsilon(unittest.TestCase):
    def test_should_exploit(self):
        epsilon_greedy = EpsilonGreedy(0.1, random_sampler=lambda: 0.2)
        self.assertTrue(epsilon_greedy.sample_should_exploit())

    def test_should_not_explore(self):
        epsilon_greedy = EpsilonGreedy(0.1, random_sampler=lambda: 0.2)
        self.assertFalse(epsilon_greedy.sample_should_explore())

class WhenRandomNumberIsGreaterThanEpsilon(unittest.TestCase):
    def test_should_not_exploit(self):
        epsilon_greedy = EpsilonGreedy(0.1, random_sampler=lambda: 0.05)
        self.assertFalse(epsilon_greedy.sample_should_exploit())

    def test_should_explore(self):
        epsilon_greedy = EpsilonGreedy(0.1, random_sampler=lambda: 0.05)
        self.assertTrue(epsilon_greedy.sample_should_explore())

if __name__ == '__main__':
    unittest.main()
