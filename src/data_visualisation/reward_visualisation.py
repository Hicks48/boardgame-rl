import matplotlib.pyplot as plt
import numpy as np

def visualise_rewards(rewards):
    episodes = np.linspace(1, len(rewards), len(rewards))

    fig, ax = plt.subplots()
    ax.plot(episodes, rewards)
    plt.show()
