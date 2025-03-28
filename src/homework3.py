import os
from datetime import datetime
import time

import torch
import torchvision.transforms as transforms
import numpy as np

import environment
from agents import REINFORCE_Agent

from model import VPG
from model import SAC


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._delta = 0.05
        self._goal_thresh = 0.075  # easier goal detection
        self._max_timesteps = 300  # allow more steps
        self.c_ee_to_obj = 0.1
        self.c_obj_to_target = 0.2
        self.c_direction = 0.5
        self.completion_reward = 10
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
    
    def reset(self):
        super().reset()
        self._prev_obj_pos = self.data.body("obj1").xpos[:2].copy()  # initialize previous position
        self._t = 0

        try:
            return self.high_level_state()
        except:
            return None

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

    # def reward(self):
    #     state = self.high_level_state()
    #     ee_pos = state[:2]
    #     obj_pos = state[2:4]
    #     goal_pos = state[4:6]
    #     ee_to_obj = max(10*np.linalg.norm(ee_pos - obj_pos), 1)
    #     obj_to_goal = max(10*np.linalg.norm(obj_pos - goal_pos), 1)
    #     goal_reward = 100 if self.is_terminal() else 0
    #     return 1/(ee_to_obj) + 1/(obj_to_goal) + goal_reward

    def reward(self):
        
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]

        d_ee_to_obj = np.linalg.norm(ee_pos - obj_pos)
        d_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

        # distance-based rewards
        r_ee_to_obj = - self.c_ee_to_obj * d_ee_to_obj  # getting closer to object
        r_obj_to_goal = - self.c_obj_to_target * d_obj_to_goal  # moving object to goal

        # direction bonus
        obj_movement = obj_pos - self._prev_obj_pos
        dir_to_goal = (goal_pos - obj_pos) / (np.linalg.norm(goal_pos - obj_pos) + 1e-8)
        r_direction = self.c_direction * max(0, np.dot(obj_movement / (np.linalg.norm(obj_movement) + 1e-8), dir_to_goal))
        if np.linalg.norm(obj_movement) < 1e-6:  # Avoid division by zero
            r_direction = 0.0

        # terminal bonus
        r_terminal = self.completion_reward if self.is_terminal() else 0.0

        r_step = -0.1  # penalty for each step
        # r_step = 0

        self._prev_obj_pos = obj_pos.copy()
        return r_ee_to_obj + r_obj_to_goal + r_direction + r_terminal + r_step

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

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


    # def step(self, action):
    #     action = action.clamp(-1, 1).cpu().numpy() * self._delta
    #     ee_pos = self.data.site(self._ee_site).xpos[:2]
    #     target_pos = np.concatenate([ee_pos, [1.06]])
    #     target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
    #     self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
    #     self._t += 1

    #     state = self.high_level_state()
    #     reward = self.reward()
    #     terminal = self.is_terminal()
    #     truncated = self.is_truncated()
    #     return state, reward, terminal, truncated


def reinforce_main(
        num_episodes_per_update = 1,
        env_max_timesteps = 200,
        model_lr = 1e-4,
        goal_tresh = 0.075,
        c_ee_to_obj = 0.1,
        c_obj_to_target = 0.2,
        c_direction = 0.5,
        completion_reward = 10,
):

    env = Hw3Env(render_mode="offscreen")
    env._max_timesteps = env_max_timesteps
    env._goal_thresh = goal_tresh
    env.c_ee_to_obj = c_ee_to_obj
    env.c_obj_to_target = c_obj_to_target
    env.c_direction = c_direction
    env.completion_reward = completion_reward

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    agent = REINFORCE_Agent(model=VPG(), device=device, num_episodes_per_update=num_episodes_per_update, lr=model_lr)



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

    num_episodes = 5_000
    episode_lengths = []
    for i in range(num_episodes):
        start_time = time.time()
        for episode in range(num_episodes_per_update):
            env.reset()
            state = env.high_level_state()
            done = False

            ep_lengths = []
            steps_this_episode = 0

            cumulative_reward = 0.0
            total_log_prob = 0.0

            while not done:
                action, log_prob = agent.predict(torch.tensor(state, dtype=torch.float32).to(device))
                next_state, reward, is_terminal, is_truncated = env.step(action)

                total_log_prob += log_prob
                cumulative_reward += reward

                done = is_terminal or is_truncated
                state = next_state
                steps_this_episode += 1

            agent.add_episode({"total_log_prob": total_log_prob,
                               "cumulative_reward": cumulative_reward,
                               "steps_per_episode": env._max_timesteps})
            ep_lengths.append(steps_this_episode)


        episode_lengths.append(np.mean(ep_lengths))

        episode_rewards_file.write(f"{cumulative_reward}\n")
        episode_rewards_file.flush()
        data_collect_end_time = time.time()
        print(f"Episode {i} collected in {data_collect_end_time - start_time} seconds, reward={cumulative_reward}, flush=True")

        agent.update_model()
        agent.save_model(model_save_path)
        print(f"model updated in {time.time() - data_collect_end_time} seconds", flush=True)
        print()

        avg_episode_length = np.mean(episode_lengths)
        wandb.log({
            "episode": i,
            "cumulative_reward": cumulative_reward,
            "avg_episode_length": avg_episode_length,
        })


def sweep_reinforce(config=None):
    with wandb.init(config=config):
        config = wandb.config

        reinforce_main(
            # num_episodes_per_update = config.num_episodes_per_update, constant 1
            # env_max_timesteps = config.env_max_timesteps, constant 200
            # goal_tresh = config.goal_tresh, constant 0.075
            model_lr = config.model_lr,
            c_ee_to_obj = config.c_ee_to_obj,
            c_obj_to_target = config.c_obj_to_target,
            c_direction = config.c_direction,
            completion_reward = config.completion_reward,
        )


import wandb
if __name__ == "__main__":

    default_config = {
        "model_lr": 1e-4,  # default learning rate
        "c_ee_to_obj": 0.1,
        "c_obj_to_target": 0.2,
        "c_direction": 0.5,
        "completion_reward": 10,
    }

    # reinforce_main()
    sweep_reinforce(default_config)


    # sac_main()


