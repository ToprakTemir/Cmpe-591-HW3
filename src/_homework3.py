# Using multiple processes to collect data for training a model
import os
import time
from collections import deque
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import numpy as np
from torch.ao.quantization.utils import activation_is_int8_quantized

import environment

from model import VPG
from model import SAC
from agents import REINFORCE_Agent
from agents import SAC_agent


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # divide the action space into n_actions
        self._delta = 0.05

        self._goal_thresh = 0.075
        self._max_timesteps = 50
        self._prev_obj_pos = None  # track object movement

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based rewards
        r_ee_to_obj = -0.1 * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = -0.2 * d_obj_to_goal  # moving object to goal

        # direction bonus
        if self._prev_obj_pos is None:
            self._prev_obj_pos = obj_pos.copy()

        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = 0.5 * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
            r_direction = 0.0

        # terminal bonus
        r_terminal = 10.0 if self.is_terminal() else 0.0

        r_step = -0.1  # penalty for each step

        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        action = action.clamp(-1, 1).cpu().numpy() * self._delta
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.06]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        result = self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        if result:  # If the action is successful
            truncated = self.is_truncated()
        else:  # If didn't realize the action
            truncated = True

        return state, reward, terminal, truncated


class Memory:
    def __init__(self, keys, buffer_length=None):
        self.buffer = {}
        self.keys = keys
        for key in keys:
            self.buffer[key] = deque(maxlen=buffer_length)

    def clear(self):
        for key in self.keys:
            self.buffer[key].clear()

    def append(self, dic):
        for key in self.keys:
            self.buffer[key].append(dic[key])

    def sample_n(self, n):
        r = torch.randperm(len(self))
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        res = {}
        for key in self.keys:
            res[key] = torch.stack([self.buffer[key][i] for i in idx])
        return res

    def get_all(self):
        idx = list(range(len(self)))
        return self.get_by_idx(idx)

    def __len__(self):
        return len(self.buffer[self.keys[0]])


def collecter(agent, shared_queue_for_episodes, is_collecting, all_episodes_collected, device):
    env = Hw3Env(render_mode="offscreen")

    while True:
        is_collecting.wait()

        while is_collecting.is_set():
            env.reset()
            state = env.high_level_state()
            done = False
            cum_reward = 0.0
            transitions = {"state": [], "action": [], "reward": [], "next_state": [], "done": [], "log_prob": []}
            while not done:
                if isinstance(agent, REINFORCE_Agent):
                    action, action_log_prob = agent.predict(torch.tensor(state, dtype=torch.float32).to(device))
                elif isinstance(agent, SAC_agent):
                    raise NotImplementedError

                next_state, reward, is_terminal, is_truncated = env.step(action)
                cum_reward += reward
                done = is_terminal or is_truncated
                # transitions["state"].append(state)
                # transitions["action"].append(action)
                transitions["reward"].append(reward)
                # transitions["next_state"].append(next_state)
                # transitions["done"].append(done)
                transitions["log_prob"].append(action_log_prob)
                state = next_state

            transitions["reward"] = np.array(transitions["reward"])
            transitions["log_prob"] = torch.stack(transitions["log_prob"])

            shared_queue_for_episodes.put(transitions)

            all_episodes_collected.wait() # wait for the other processes to finish their episodes

def reinforce_main():
    mp.set_start_method("spawn")

    device = torch.device("cpu")
    num_episodes_per_update = 1
    agent = REINFORCE_Agent(model=VPG(), device=device, num_episodes_per_update=num_episodes_per_update)

    reward_save_dir = "./reinforce_rewards/"
    if not os.path.exists(reward_save_dir):
        os.makedirs(reward_save_dir)
    model_save_dir = "./reinforce_models/"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    time_label = datetime.now().strftime("%Y%m%d-%H%M%S")
    reward_save_path = reward_save_dir + "rewards_" + time_label + ".npy"
    model_save_path = model_save_dir + "model_" + time_label

    episode_rewards_file = open(reward_save_path, "w")

    agent.model.share_memory()  # share model parameters across processes
    shared_queue_for_episode_lists = mp.Queue()

    is_collecting = mp.Event()
    all_episodes_collected = mp.Event()

    procs = []
    for i in range(num_episodes_per_update):
        p = mp.Process(target=collecter, args=(agent, shared_queue_for_episode_lists, is_collecting, all_episodes_collected, device))
        p.start()
        procs.append(p)

    num_episodes = 10_000
    for i in range(num_episodes):
        start = time.time()

        is_collecting.set()
        collected_episode_count = 0
        while collected_episode_count < agent.episodes_per_update:
            if not shared_queue_for_episode_lists.empty():
                # state, action, reward, next_state, done, action_log_prob = shared_queue.get()
                # agent.add_step_to_episode({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done, "log_prob": action_log_prob})
                # del state, action, reward, next_state, done, action_log_prob

                agent.add_episode(shared_queue_for_episode_lists.get())
                collected_episode_count += 1

        is_collecting.clear()

        episode_rewards_mean = np.array([episode["reward"].sum().item() for episode in agent.episodes]).mean()
        episode_rewards_file.write(f"{episode_rewards_mean}\n") # save the mean reward of the episodes
        episode_rewards_file.flush()

        end = time.time()
        print(f"episode {i} completed in {end - start} seconds, updating the model...")

        # do your update
        agent.update_model()
        agent.save_model(model_save_path)
        print(f"model updated in {time.time() - end} seconds")


    for p in procs:
        p.join()


def sac_main():
    raise NotImplementedError

if __name__ == "__main__":
    reinforce_main()
    # sac_main()
