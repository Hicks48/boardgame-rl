from collections import namedtuple
import numpy as np

def makeEmptyQTable(observation_space_size, action_space_size):
    return np.zeros([observation_space_size, action_space_size])

QTableAgentHyperParameters = namedtuple('QTableAgentHyperParameters', ('learning_rate', 'discount_factor', 'epsilon_greedy'))

class QTableAgent:
    def __init__(self, q_table):
        self.q_table = q_table

    def select_inference_action(self, state):
        # In inference always return the best value.
        return np.argmax(self.q_table[state])

    def select_train_action(self, state, hyper_parameters):
        # Check if should exploit current policy and return best value if should exploit.
        if hyper_parameters.epsilon_greedy.sample_should_exploit():
            return np.argmax(self.q_table[state])

        # Return random action as an exploration action.
        return np.random.choice(len(self.q_table[state]))

    def train(self, transition, hyper_parameters):
        # Extract relevant hyper parameters.
        learning_rate = hyper_parameters.learning_rate
        discount_factor = hyper_parameters.discount_factor

        # Extract information from the transition.
        state = transition.state
        action = transition.action
        next_state = transition.next_state
        reward = transition.reward if transition.reward is not None else 0

        # Calculate Q(s_t0, a_t0) and max_a_t1(Q(s_t1, a_t1)).
        previous_q_value = self.q_table[state, action]
        next_max_q_value = np.max(self.q_table[next_state])

        # Update Q value based on the Bellman equation.
        self.q_table[state, action] = previous_q_value + learning_rate * (reward + discount_factor * next_max_q_value - previous_q_value)

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)
