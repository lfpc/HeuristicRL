import torch
from torch import nn
from torch.distributions import Categorical


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions)   # policy logits
        self.critic = mlp(obs_dim, 1)          # state value V(s)

    def logits(self, obs, mask):
        return self.actor(obs).masked_fill(~mask, -1e9)

    def dist(self, obs, mask):
        return Categorical(logits=self.logits(obs, mask))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, mask, greedy=False):
        """numpy in, numpy out."""
        obs, mask = torch.as_tensor(obs), torch.as_tensor(mask)
        if greedy:
            return self.logits(obs, mask).argmax(-1).numpy()
        return self.dist(obs, mask).sample().numpy()
