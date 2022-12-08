class RewardAccumulator:
    def __init__(self):
        self.current_episode_reward = 0.0
        self.episode_rewards = []
    
    def get_rewards(self):
        return self.episode_rewards

    def register_reward(self, reward):
        self.current_episode_reward += reward

    def end_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0
