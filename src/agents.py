from collections import deque

import torch
from torch import optim
import numpy as np

from model import VPG
from model import Q_Network
import time
import wandb

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

class Memory():
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done

        self.index += 1
        self.index %= self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.full = self.size == self.capacity

    def sample(self, batch_size):
        if self.full:
            indices = np.random.choice(self.capacity, batch_size)
        else:
            indices = np.random.choice(self.index, batch_size)

        states = torch.tensor(self.states[indices], dtype=torch.float32)
        actions = torch.tensor(self.actions[indices], dtype=torch.float32)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32)

        return states, actions, rewards, next_states, dones



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
        self.q1 = Q_Network(obs_dim, action_dim).to(device)
        self.q2 = Q_Network(obs_dim, action_dim).to(device)
        self.target_q1 = Q_Network(obs_dim, action_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2 = Q_Network(obs_dim, action_dim).to(device)
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

        self.memory_buffer = Memory(memory_capacity, obs_dim, action_dim)

    def predict(self, state):
        mean, std = self.policy(state)

        # reparametrization trick: sample noise, make action differentiable (since it is not "sampled" anymore)
        epsilon = torch.randn_like(mean)
        action = mean + epsilon * std

        action_dist = torch.distributions.Normal(mean, std)
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'target_q1_state_dict': self.target_q1.state_dict(),
            'target_q2_state_dict': self.target_q2.state_dict(),
            'entropy_temp': self.entropy_temp,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q2.load_state_dict(checkpoint['target_q2_state_dict'])
        self.entropy_temp = checkpoint['entropy_temp']

    def update(self):
        mse = torch.nn.MSELoss()

        # sample batch from memory to update based on that
        states, actions, rewards, next_states, dones = self.memory_buffer.sample(self.batch_size)

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
        policy_loss.backward() # retain graph to not lose the gradients of policy_log_probs
        self.policy_optimizer.step()

        # updating entropy temperature
        policy_actions, policy_log_probs = self.predict(states)
        entropy_loss = - self.entropy_temp * (policy_log_probs + self.target_entropy).mean()
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # updating target Q networks (via polyak averaging)
        for q, target_q in [(self.q1, self.target_q1), (self.q2, self.target_q2)]:
            for target_param, param in zip(target_q.parameters(), q.parameters()):
                target_param.data.copy_(self.polyak_tau * param.data + (1 - self.polyak_tau) * target_param.data)


    def train(self, env, total_updates, reward_log_path, model_save_path, best_model_save_path):
        env.reset()
        env_done = False
        reward_this_episode = 0.0

        highest_reward = -np.inf
        episode_rewards_file = open(reward_log_path, "w")
        latest_episode_end_time = time.time()
        latest_10_rewards = deque(maxlen=10)

        for i in range(total_updates):

            # data collecting
            start_time = time.time()
            for _ in range(self.num_steps_per_iteration):
                if env_done:
                    env.reset()
                    print(
                        f"Episode {i} collected in {time.time() - latest_episode_end_time} seconds, reward={reward_this_episode}",
                        flush=True)
                    latest_episode_end_time = time.time()
                    episode_rewards_file.write(f"{reward_this_episode}\n")
                    episode_rewards_file.flush()
                    if reward_this_episode > highest_reward:
                        highest_reward = reward_this_episode
                        self.save_model(best_model_save_path)
                        print(f"Best model saved with reward {highest_reward}", flush=True)
                    reward_this_episode = 0.0

                state = env.high_level_state()
                action, _ = self.predict(torch.tensor(state, dtype=torch.float32).to(self.device))
                next_state, reward, is_terminal, is_truncated = env.step(action.detach())
                reward_this_episode += reward
                latest_10_rewards.append(reward_this_episode)
                env_done = is_terminal or is_truncated
                self.memory_buffer.add(state, action.detach(), reward, next_state, env_done)

            # model updating
            for _ in range(self.num_updates_per_iteration):
                self.update()

            self.save_model(model_save_path)

            # if i % 10 == 0:
            #     wandb.log({
            #         "update_step": i,
            #         "reward": sum(latest_10_rewards),
            #     })