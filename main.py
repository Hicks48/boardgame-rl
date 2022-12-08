import gym
import numpy as np
from src.env_simulator import TrainingProcess, InferenceProcess, RewardAccumulator, simulate
from src.agents import QTableAgent, QTableAgentHyperParameters, makeEmptyQTable, EpsilonGreedy
from src.data_visualisation import visualise_rewards

if __name__ == '__main__':
    env_name = 'Taxi-v3'
    env = gym.make(env_name, render_mode = None)

    agent = QTableAgent(makeEmptyQTable(env.observation_space.n, env.action_space.n))

    # Train phase.
    hyper_parameters = QTableAgentHyperParameters(0.1, 0.6, EpsilonGreedy(0.1))
    training_reward_accumulator = RewardAccumulator()
    training_process = TrainingProcess(agent, hyper_parameters, reward_accumulator=training_reward_accumulator)

    simulate(env, training_process, number_of_episodes=50000)
    visualise_rewards(training_reward_accumulator.get_rewards())

    env.close()

    env = gym.make(env_name, render_mode = 'human')

    # Inference phase.
    inference_reward_accumulator = RewardAccumulator()
    inference_process = InferenceProcess(agent, reward_accumulator=inference_reward_accumulator)
    simulate(env, inference_process, number_of_episodes=8)

    env.close()

    print('Q-Table: {}'.format(agent.q_table))

    agent.save('./builds/q-table-taxi.npy')
