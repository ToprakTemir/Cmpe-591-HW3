import subprocess
import os

SWEEP_ID = "topraktemir-/Cmpe-591-HW3-src/152rl4z8"
NUM_AGENTS = 20

processes = []

# Set environment variables for headless running
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

for i in range(NUM_AGENTS):
    print(f"launching agent {i}")
    p = subprocess.Popen(["wandb", "agent", SWEEP_ID])
    processes.append(p)

# Optional: wait for all to finish
for p in processes:
    p.wait()