from tqdm import tqdm
from src.env_simulator.transition import Transition

def simulate(env, agent_process, number_of_episodes, max_steps = -1, show_progress_bar = True):
    progress_bar = tqdm(total=number_of_episodes) if show_progress_bar else None

    for i_episode in range(number_of_episodes):
        step_count = 0
        has_episode_ended = False

        state, info = env.reset()
        while not has_episode_ended:
            # Select and take action.
            action = agent_process.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Inform on the transition.
            transition = Transition(state, action, next_state, reward)
            agent_process.observe_transition(transition)
            
            # Handle step counts.
            step_count = step_count + 1
            was_max_step_count_reached = max_steps >= 0 and step_count >= max_steps
            
            # Check for episode termination conditions.
            if terminated or truncated or was_max_step_count_reached:
                agent_process.on_end_episode()
                has_episode_ended = True

            # Next state is new state for next step.
            state = next_state

        # Update progress bar at the end of episode.        
        if progress_bar is not None:
            progress_bar.update(1)

    # Clean up progress bar at the end of simulation.
    if progress_bar is not None:
        progress_bar.close()
