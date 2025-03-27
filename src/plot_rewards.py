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

# Create subplots: one for raw data, one for smoothed data
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Initialize the line objects for raw and smoothed data
line_raw, = ax1.plot([], [], label="Raw Training Reward", color="blue")
ax1.set_title("Raw Training Reward")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reward")
ax1.legend()

line_smoothed, = ax2.plot([], [], label="Smoothed Training Reward", color="red")
ax2.set_title("Smoothed Training Reward")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Reward")
ax2.legend()


def smooth(data, window_size = 100):
    smoothed = []
    current = data[0]
    for i in range(len(data)):
        start = max(0, i - window_size)
        smoothed.append(np.mean(data[start:i+1]))
    return np.array(smoothed)


# Update the plots
def update_plots(data, line_raw, line_smoothed, ax1, ax2):
    # Update the raw data plot
    line_raw.set_xdata(range(len(data)))
    line_raw.set_ydata(data)
    ax1.relim()
    ax1.autoscale_view()

    # Smooth the data with a larger window size or exponential smoothing
    smoothed_data = smooth(data, window_size=200)

    # Update the smoothed data plot
    line_smoothed.set_xdata(range(len(smoothed_data)))
    line_smoothed.set_ydata(smoothed_data)
    ax2.relim()
    ax2.autoscale_view()

    # draw a vertical line at data point 4280
    ax1.axvline(x=4280, color='r', linestyle='--')
    ax2.axvline(x=4280, color='r', linestyle='--')

    # Refresh the plots
    plt.draw()
    fig.canvas.flush_events()


while True:
    try:
        # Read the data from file
        data = []
        with open(reward_list_path, "r") as f:
            for ln in f:
                if ln == "\n":
                    continue
                data.append(float(ln.strip()))  # Assuming each line is a single float value
        data = np.array(data)

        print(f"Read {len(data)} data points from {reward_list_path}")

        # Update plots
        update_plots(data, line_raw, line_smoothed, ax1, ax2)

        # Save the updated figure as PNG
        plt.savefig("latest_plot.png", bbox_inches='tight')

        # Flush the file explicitly to ensure immediate writing
        with open("latest_plot.png", "rb") as png_file:
            os.fsync(png_file.fileno())

    except Exception as e:
        print(f"Error reading or plotting data: {e}")

    # Pause for 60 seconds before the next update
    time.sleep(5)