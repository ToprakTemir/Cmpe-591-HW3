import multiprocessing
import os
import wandb

SWEEP_ID = "topraktemir_team/Cmpe-591-HW3-src/63eymr2h"
NUM_AGENTS = 10

def run_agent(i):
    print(f"Launching agent {i}")
    wandb.agent(SWEEP_ID)
    print(f"agent {i} is finished")

if __name__ == "__main__":
    # Set environment variables globally for all child processes
    # os.environ["PYOPENGL_PLATFORM"] = "egl"
    # os.environ["MUJOCO_GL"] = "egl"

    processes = []

    for i in range(NUM_AGENTS):
        p = multiprocessing.Process(target=run_agent, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()