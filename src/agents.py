from collections import deque

import torch
from torch import optim
import numpy as np

from model import VPG
from model import Q_Network
import time

class REINFORCE_Agent():
    def __init__(self, model=None, device="cpu", num_episodes_per_update=1, lr=1e-4):
        if model is None:
            self.model = VPG().to(device)
        else:
            self.model = model.to(device)

        self.device = device
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.episodes_per_update = num_episodes_per_update

        # self.episode_keys = ["state", "action", "reward", "next_state", "done", "log_prob"]
        self.episode_keys = ["total_log_prob", "cumulative_reward", "steps_per_episode"]
        self.episodes = []

    def reset_episodes(self):
        self.episodes = []

    def add_episode(self, episode_data_dict):
        assert list(episode_data_dict.keys()) == self.episode_keys

        if len(self.episodes) == self.episodes_per_update:
            raise ValueError("Episodes are full. Update the model before adding more episodes.")

        self.episodes.append(episode_data_dict)

    def predict(self, state):
        mean, std = self.model(state)
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob
    
    def update_model(self):

        total_loss = torch.tensor(0.).to(self.device)
        for episode in self.episodes:
            # episode_log_probs = episode["log_prob"]
            # episode_rewards = episode["reward"]
            #
            # episode_log_prob_sum = episode_log_probs.sum()
            # episode_rewards_sum = episode_rewards.sum()

            log_prob_sum = episode["total_log_prob"]
            rewards_sum = episode["cumulative_reward"]
            steps_per_episode = episode["steps_per_episode"]

            loss = - log_prob_sum * rewards_sum
            loss /= steps_per_episode
            total_loss += loss.mean()

        total_loss /= self.episodes_per_update

        total_loss = total_loss.mean()
        print(f"total loss: {total_loss}")
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.reset_episodes()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

# ----------------------------------------------------------------------------------------------------------------------

class SAC_agent():
    def __init__(self,
                 device="cpu",
                 obs_dim = 6,
                 action_dim = 2,
                 memory_capacity = 10**5,
                 num_steps_per_iteration = 1,
                 num_updates_per_iteration = 1,
                 gamma = 0.99,
                 batch_size = 128,
                 polyak_tau = 0.005,
                 initial_entropy_temp = 0.2,
                 policy_lr = 1e-4,
                 q_lr = 1e-4,
                 entropy_lr = 5e-5,
                 target_entropy_scale = 1
                 ):
        self.device = device

        # networks to be learned
        self.policy = VPG(obs_dim).to(device)
        self.q1 = Q_Network(obs_dim).to(device)
        self.q2 = Q_Network(obs_dim).to(device)
        self.target_q1 = Q_Network(obs_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2 = Q_Network(obs_dim).to(device)
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.entropy_temp = torch.tensor(initial_entropy_temp, requires_grad=True).to(device)

        # optimizers
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.entropy_lr = entropy_lr
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.q_lr)
        self.entropy_optimizer = torch.optim.Adam([self.entropy_temp], lr=self.entropy_lr)

        # other hyperparameters
        self.num_steps_per_iteration = num_steps_per_iteration
        self.num_updates_per_iteration = num_updates_per_iteration
        self.gamma = gamma
        self.batch_size = batch_size
        self.polyak_tau = polyak_tau
        self.initial_entropy_temp = initial_entropy_temp
        self.target_entropy = - action_dim * target_entropy_scale

        self.transition_keys = ["state", "action", "reward", "next_state", "done"]
        self.memory_buffer = {key: deque(maxlen=memory_capacity) for key in self.transition_keys}

    def add_transition_to_memory(self, state, action, reward, next_state, done):
        self.memory_buffer["state"].append(state)
        self.memory_buffer["action"].append(action)
        self.memory_buffer["reward"].append(reward)
        self.memory_buffer["next_state"].append(next_state)
        self.memory_buffer["done"].append(done)

    def predict(self, state):
        mean, std = self.policy(state)

        # reparametrization trick: sample noise, make action differentiable (since it is not "sampled" anymore)
        epsilon = torch.randn_like(mean)
        action = mean + epsilon * std

        action_dist = torch.distributions.Normal(mean, std)
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def update(self):
        mse = torch.nn.MSELoss()

        # sample batch from memory to update based on that
        memory_size = len(self.memory_buffer["state"])
        indices = np.random.choice(memory_size, self.batch_size)
        states = torch.tensor(self.memory_buffer["state"][indices], device=self.device)
        actions = torch.tensor(self.memory_buffer["action"][indices], device=self.device)
        rewards = torch.tensor(self.memory_buffer["reward"][indices], device=self.device).unsqueeze(-1)
        next_states = torch.tensor(self.memory_buffer["next_state"][indices], device=self.device)
        dones = torch.tensor(self.memory_buffer["done"][indices], device=self.device).unsqueeze(-1)

        # updating Q functions (both separately).
        # The main point is deriving the value function for the next state using target networks,
        # and updating the Q functions to fit the value function
        with torch.no_grad():
            next_policy_actions, next_policy_log_probs = self.predict(next_states)
        q1_next = self.target_q1(next_states, next_policy_actions)
        q2_next = self.target_q2(next_states, next_policy_actions)
        q_next = torch.min(q1_next, q2_next)

        entropy_penalty = self.entropy_temp * next_policy_log_probs
        value_of_next_state = q_next - entropy_penalty
        q_targets = rewards + self.gamma * value_of_next_state * (1 - dones)

        q_loss_1 = mse(self.q1(states, actions), q_targets)
        q_loss_2 = mse(self.q2(states, actions), q_targets)
        q_loss = q_loss_1 + q_loss_2

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # updating policy with new Q functions
        # the policy is fitted so that P(a ~ π(a|s)) ~ exp(1/entropy * Q(s, a)) i.e. entropy * log(π) = Q
        policy_actions, policy_log_probs = self.predict(states)

        q1_pi = self.q1(states, policy_actions)
        q2_pi = self.q2(states, policy_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.entropy_temp * policy_log_probs - q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # updating entropy temperature
        entropy_loss = - self.entropy_temp * (policy_log_probs + self.target_entropy).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # updating target Q networks (via polyak averaging)
        for q, target_q in [(self.q1, self.target_q1), (self.q2, self.target_q2)]:
            for target_param, param in zip(target_q.parameters(), q.parameters()):
                target_param.data.copy_(self.polyak_tau * param.data + (1 - self.polyak_tau) * target_param.data)



    def train(self, env, total_updates, reward_log_path, model_save_path, best_model_save_path):
        episode_rewards_file = open(reward_log_path, "w")
        episode_lengths = []
        for episode in range(total_updates):

            # data collecting
            start_time = time.time()
            env.reset()
            state = env.high_level_state()
            env_done = False
            cumulative_reward = 0.0
            for _ in range(self.num_steps_per_iteration):


                while not done:
                    action, log_prob = self.predict(torch.tensor(state, dtype=torch.float32).to(self.device))
                    next_state, reward, is_terminal, is_truncated = env.step(action)

                    cumulative_reward += reward

                    done = is_terminal or is_truncated
                    state = next_state

                self.add_episode({"total_log_prob": total_log_prob,
                                   "cumulative_reward": cumulative_reward,
                                   "steps_per_episode": env._max_timesteps})

            episode_lengths.append(np.mean(ep_lengths))
            episode_rewards_file.write(f"{cumulative_reward}\n")
            episode_rewards_file.flush()
            data_collect_end_time = time.time()
            print(f"Episode {i} collected in {data_collect_end_time - start_time} seconds, reward={cumulative_reward}",
                  flush=True)

            # model updating
            for _ in range(self.num_updates_per_iteration):
                self.update()



