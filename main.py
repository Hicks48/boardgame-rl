import gym
import numpy as np
from src.env_simulator import TrainingProcess, InferenceProcess, RewardAccumulator, simulate
from src.agents import QTableAgent, QTableAgentHyperParameters, makeEmptyQTable, EpsilonGreedy, linear_decay
from src.data_visualisation import visualise_rewards

# Temp solution need come up with something better
def makeEnv(render_mode = None):
    # return gym.make('Taxi-v3', render_mode=render_mode)
    return gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False, render_mode=render_mode)

if __name__ == '__main__':
    NUM_TRAIN_EPISODES = 50000
    NUM_INFERENCE_EPISODES = 8

    env = makeEnv()
    agent = QTableAgent(makeEmptyQTable(env.observation_space.n, env.action_space.n))

    # Train phase.
    hyper_parameters = QTableAgentHyperParameters(0.1, 0.6, EpsilonGreedy(0.3, epsilon_decay_fn=linear_decay(NUM_TRAIN_EPISODES)))
    training_reward_accumulator = RewardAccumulator()
    training_process = TrainingProcess(agent, hyper_parameters, reward_accumulator=training_reward_accumulator)

    simulate(env, training_process, number_of_episodes=NUM_TRAIN_EPISODES)
    visualise_rewards(training_reward_accumulator.get_rewards())

    env.close()

    env = env = makeEnv('human')

    # Inference phase.
    inference_reward_accumulator = RewardAccumulator()
    inference_process = InferenceProcess(agent, reward_accumulator=inference_reward_accumulator)
    simulate(env, inference_process, number_of_episodes=NUM_INFERENCE_EPISODES)

    env.close()

    print('Q-Table: {}'.format(agent.q_table))

    agent.save('./builds/q-table-taxi.npy')
