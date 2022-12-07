# Reinforcement Learning for Boardgames

## Background and Theory
This section provides overview on Reinforcement Learning and the specific algorithms used on projects of this repository.

### What is Reinforcement Learning?
In Reinforcement Learning setup an agent takes actions in an environment with a goal of maximizing rewards. The agent observes the state of the environment and takes an action. The agent is then provided with a new state of the environment and a reward. The reward can be positive to signal whether the current state is desirable or not. It is typical for Reinforcement Learning problems that the rewards are delayd meaning that the reason for the reward is result of many actions that the agent took earlier not just the most recent one. It also common that the agent can only observe environment partially and the results of actions can also be stochastic.

TODO: Make image here.

### Markov Decision Process
Reinforcement Learning environment can be modeled as a Markov Decision Process. Markov Decision Process consists of states, actions and transitions. Each state has a set of actions which are available at the state. An action has a set of transitions one of which will occur as a result of the action. Transition has a probability that it will occur and a reward.

TODO: Make image here.

### Modeling the reward
The agents goal is to maximize the reward overtime steps. 

### Q-Learning
Q-Learning algorithm tries to find an optimal policy $\pi_*(s, a)$.


