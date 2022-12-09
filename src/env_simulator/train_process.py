class TrainingProcess:
    def __init__(self, agent, hyper_parameters, reward_accumulator):
        self.agent = agent
        self.hyper_parameters = hyper_parameters
        self.reward_accumulator = reward_accumulator
        self.current_episode = 0

    def select_action(self, state):
        return self.agent.select_train_action(state, self.hyper_parameters, self.current_episode)

    def observe_transition(self, transition):
        self.reward_accumulator.register_reward(transition.reward)
        self.agent.train(transition, self.hyper_parameters)

    def on_end_episode(self, i_episode):
        self.current_episode = i_episode + 1
        self.reward_accumulator.end_episode()
