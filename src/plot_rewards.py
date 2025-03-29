import time
import matplotlib.pyplot as plt
import numpy as np
import os

rewards_dir_name = "reinforce_rewards"
# rewards_dir_name = "sac_rewards"

def get_latest_reward_list():
    # Get the list of reward files in the directory
    reward_files = [f for f in os.listdir(rewards_dir_name)]

    # Sort the files by modification time
    reward_files = sorted(reward_files, key=lambda f: os.path.getmtime(f"{rewards_dir_name}/{f}"), reverse=True)

    # Return the path of the most recent file
    return f"{rewards_dir_name}/{reward_files[0]}" if reward_files else None

# reward_list_path = "HW3/rewards/rewards_list_2024.12.13-15:05:30.pth"
reward_list_path = get_latest_reward_list()

if not reward_list_path:
    raise ValueError("reward list could not be found")

# Read the rewards from file: each line is expected to be a float value for each episode
with open(reward_list_path, 'r') as file:
    rewards = [float(line.strip()) for line in file if line.strip()]

# Function for moving average smoothing
def moving_average(data, window_size=10):
    if window_size < 1:
        return data
    # Using 'valid' mode to avoid boundary effects
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Set your desired smoothing window size (adjust as needed)
window_size = 100
smoothed_rewards = moving_average(rewards, window_size)

# Create x values for episodes
episodes = np.arange(1, len(rewards) + 1)
# Adjust the x-axis for the smoothed rewards (which has fewer points due to convolution 'valid' mode)
smoothed_episodes = np.arange(window_size, len(rewards) + 1)

# Plot the raw and smoothed rewards
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label='Raw Reward', alpha=0.5)
plt.plot(smoothed_episodes, smoothed_rewards, label='Smoothed Reward', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward Graph')
plt.legend()
plt.show()