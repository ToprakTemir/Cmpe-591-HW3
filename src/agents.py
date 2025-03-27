import torch
from torch import optim

from model import VPG

class REINFORCE_Agent():
    def __init__(self, model=None, device="cpu", num_episodes_per_update=6):
        if model is None:
            self.model = VPG().to(device)
        else:
            self.model = model.to(device)

        self.device = device

        # HYPERPARAMETERS
        self.episodes_per_update = num_episodes_per_update
        self.lr = 3e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)


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



class SAC_agent():
    pass
        
