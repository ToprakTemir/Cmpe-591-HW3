import torch.nn as nn
import torch
import torch.nn.functional as F

class VPG(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]) -> None:
        super(VPG, self).__init__()
        self.act_dim = act_dim
        layers = []
        layers.append(nn.Linear(obs_dim, hl[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hl)):
            layers.append(nn.Linear(hl[i-1], hl[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hl[-1], act_dim*2))  # act_dim * (1 for mean + 1 for std)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        mean_and_std = self.model(x)
        mean = mean_and_std[:, :self.act_dim]
        std = mean_and_std[:, self.act_dim:]
        std = F.softplus(std) + 0.05
        return mean, std




class SAC(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hl=[256, 512, 256]) -> None:
        raise NotImplementedError
    