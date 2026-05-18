import math
import numpy as np
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DistType = Literal["gaussian", "dirichlet"]

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        state = torch.tensor(self.states[idx], dtype=torch.float32, device=device)
        action = torch.tensor(self.actions[idx], dtype=torch.float32, device=device)
        reward = torch.tensor(self.rewards[idx], dtype=torch.float32, device=device)
        next_state = torch.tensor(self.next_states[idx], dtype=torch.float32, device=device)
        done = torch.tensor(self.dones[idx], dtype=torch.float32, device=device)

        return state, action, reward, next_state, done

    def __len__(self):
        return self.size


# -----------------------------
# Critic
# -----------------------------
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.q1_l1, self.q1_l2, self.q2_l1, self.q2_l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.q1_l3.weight, gain=1.0)
        nn.init.constant_(self.q1_l3.bias, 0.0)
        nn.init.orthogonal_(self.q2_l3.weight, gain=1.0)
        nn.init.constant_(self.q2_l3.bias, 0.0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

class GaussianActor(nn.Module):
    """
    Continuous action in [-1, 1] using Tanh-squashed Gaussian.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        # Output layer: small init để action ban đầu gần 0
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic: bool = False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            z = mean
        else:
            z = normal.rsample()

        action = z

        # No Tanh correction needed for unconstrained logits
        log_prob = normal.log_prob(z)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class DirichletActor(nn.Module):
    """
    Action is a probability simplex: non-negative and sums to 1.
    Best for routing/offloading proportions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.l1, self.l2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.alpha_head.weight, gain=0.01)
        nn.init.constant_(self.alpha_head.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        alpha = F.softplus(self.alpha_head(x)) + 0.01
        return alpha

    def sample(self, state, deterministic: bool = False):
        alpha = self.forward(state)

        if deterministic:
            if (alpha > 1).all():
                action = (alpha - 1) / (alpha.sum(dim=-1, keepdim=True) - self.action_dim)
            else:
                action = alpha / alpha.sum(dim=-1, keepdim=True)  # fallback to mean
            dist = torch.distributions.Dirichlet(alpha)
            log_prob = dist.log_prob(torch.clamp(action, 1e-6, 1.0 - 1e-6)).unsqueeze(-1)
        else:
            gamma_dist = torch.distributions.Gamma(alpha, 1.0)
            gamma_sample = gamma_dist.rsample()  # có rsample!
            action = gamma_sample / gamma_sample.sum(dim=-1, keepdim=True)

            action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
            action = action / action.sum(dim=-1, keepdim=True)
            log_prob = torch.distributions.Dirichlet(alpha).log_prob(action).unsqueeze(-1)

        return action, log_prob

class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device,
        dist_type: DistType = "gaussian",
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.dist_type = dist_type
        self.gamma = gamma
        self.tau = tau

        if dist_type == "gaussian":
            self.actor = GaussianActor(state_dim, action_dim).to(device)
            self.target_entropy = -float(action_dim)
        elif dist_type == "dirichlet":
            self.actor = DirichletActor(state_dim, action_dim).to(device)
            uniform_alpha = torch.ones(action_dim)
            uniform_dist = torch.distributions.Dirichlet(uniform_alpha)
            max_entropy = uniform_dist.entropy().item()
            self.target_entropy = max_entropy * 0.5
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate: bool = False):
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_t = torch.tensor(state_arr, device=self.device).unsqueeze(0)
            squeeze = True
        else:
            state_t = torch.tensor(state_arr, device=self.device)
            squeeze = False

        with torch.no_grad():
            action, _ = self.actor.sample(state_t, deterministic=evaluate)

        action = action.cpu().numpy()
        return action[0] if squeeze else action

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 256):
        if len(buffer) < batch_size:
            return {}

        state, action, reward, next_state, done = buffer.sample(batch_size, self.device)

        # Target Q
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, deterministic=False)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1.0 - done) * self.gamma * target_q

        # Critic update
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(state, deterministic=False)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target critic
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q_new.mean().item()),
        }

