class InferenceProcess:
    def __init__(self, agent, reward_accumulator):
        self.agent = agent
        self.reward_accumulator = reward_accumulator

    def select_action(self, state):
        return self.agent.select_inference_action(state)

    def observe_transition(self, transition):
        self.reward_accumulator.register_reward(transition.reward)

    def on_end_episode(self):
        self.reward_accumulator.end_episode()
