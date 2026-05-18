import math
import numpy as np
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm, MF
from matrix_source.agents.buffer.sac_buffer import MultiAgentSACReplayBuffer

DistType = Literal["gaussian", "dirichlet"]

# -----------------------------
# Networks
# -----------------------------
class MFCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mf_dim:int=None, hidden_dim: int = 256, num_instances=1):
        super().__init__()
        self.num_instances = num_instances
        
        # Q1 architecture
        if not mf_dim:
            mf_dim= action_dim
        self.q1_l1 = MultiInstanceLinear(num_instances, state_dim + action_dim + mf_dim, hidden_dim) # Action + MF Action
        self.q1_norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.q1_l2 = MultiInstanceLinear(num_instances, hidden_dim, hidden_dim)
        self.q1_norm2 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.q1_l3 = MultiInstanceLinear(num_instances, hidden_dim, 1)

        # Q2 architecture
        self.q2_l1 = MultiInstanceLinear(num_instances, state_dim + action_dim + mf_dim, hidden_dim)
        self.q2_norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.q2_l2 = MultiInstanceLinear(num_instances, hidden_dim, hidden_dim)
        self.q2_norm2 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.q2_l3 = MultiInstanceLinear(num_instances, hidden_dim, 1)

    def forward(self, state, action, pred_mf, indices=None):
        sa = torch.cat([state, action, pred_mf], dim=-1)
        
        q1 = F.silu(self.q1_norm1(self.q1_l1(sa, indices), indices))
        q1 = F.silu(self.q1_norm2(self.q1_l2(q1, indices), indices))
        q1 = self.q1_l3(q1, indices)

        q2 = F.silu(self.q2_norm1(self.q2_l1(sa, indices), indices))
        q2 = F.silu(self.q2_norm2(self.q2_l2(q2, indices), indices))
        q2 = self.q2_l3(q2, indices)

        return q1, q2

class MFGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,mf_dim:int=None, hidden_dim: int = 256, num_instances=1, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.num_instances = num_instances
        if not mf_dim:
            mf_dim=action_dim

        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, hidden_dim) # State + MF
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.l2 = MultiInstanceLinear(num_instances, hidden_dim, hidden_dim)
        self.norm2 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        
        self.mean = MultiInstanceLinear(num_instances, hidden_dim, action_dim)
        self.log_std = MultiInstanceLinear(num_instances, hidden_dim, action_dim)

    def forward(self, state, pred_mf, indices=None):
        x = torch.cat([state, pred_mf], dim=-1)
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        
        mean = self.mean(x, indices)
        log_std = torch.clamp(self.log_std(x, indices), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, pred_mf, indices=None, deterministic: bool = False):
        mean, log_std = self.forward(state, pred_mf, indices)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)

        if deterministic:
            z = mean
        else:
            z = normal.rsample()

        action = torch.tanh(z)

        # Tanh correction
        log_prob = normal.log_prob(z) - (2 * (math.log(2) - z - F.softplus(-2 * z)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class MFDirichletActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mf_dim:int= None, hidden_dim: int = 256, num_instances=1):
        super().__init__()
        self.action_dim = action_dim
        self.num_instances = num_instances
        if not mf_dim:
            mf_dim=action_dim
        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, hidden_dim)
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.l2 = MultiInstanceLinear(num_instances, hidden_dim, hidden_dim)
        self.norm2 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        
        self.alpha_head = MultiInstanceLinear(num_instances, hidden_dim, action_dim)

    def forward(self, state, pred_mf, indices=None):
        x = torch.cat([state, pred_mf], dim=-1)
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        alpha = F.softplus(self.alpha_head(x, indices)) + 0.01
        return alpha

    def sample(self, state, pred_mf, indices=None, deterministic: bool = False):
        alpha = self.forward(state, pred_mf, indices)

        if deterministic:
            alpha_sum = alpha.sum(dim=-1, keepdim=True)
            mask = (alpha > 1).all(dim=-1, keepdim=True)
            action_mode = (alpha - 1) / (alpha_sum - self.action_dim).clamp(min=1e-8)
            action_mean = alpha / alpha_sum
            action = torch.where(mask, action_mode, action_mean)
            
            dist = torch.distributions.Dirichlet(alpha)
            log_prob = dist.log_prob(torch.clamp(action, 1e-6, 1.0 - 1e-6)).unsqueeze(-1)
        else:
            gamma_dist = torch.distributions.Gamma(alpha, 1.0)
            gamma_sample = gamma_dist.rsample()
            action = gamma_sample / gamma_sample.sum(dim=-1, keepdim=True)
            action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
            action = action / action.sum(dim=-1, keepdim=True)
            
            log_prob = torch.distributions.Dirichlet(alpha).log_prob(action).unsqueeze(-1)

        return action, log_prob





class MFSACAgent:
    def __init__(
        self,
        node_id,
        node_type,
        state_dim: int,
        action_dim: int,
        mf_dim: int= None,
        mf_hidden_sizes=(128,),
        hidden_sizes=(256, 256),
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        mf_lr: float = 1e-4,
        alpha_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.005,      # Used dynamically sometimes, overridden by learning alpha
        buffer_size: int = 100000,
        batch_size: int = 256,
        buffer_min_size: int = 4096,
        exclude_zero: bool = False,
        num_instances: int = 1,
        device=None,
        use_per: bool = False,
        logs_q: bool = False,
        dist_type: DistType = "gaussian",
    ):
        if not mf_dim:
            mf_dim = action_dim
        self.node_id = node_id
        self.node_type = node_type
        self.state_dim = state_dim
        self.action_dim = action_dim # For continuous actions in MFSAC, action is vector of action_dim dimension.
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist_type = dist_type
        self.gamma = gamma
        self.tau = alpha if alpha < 1.0 else tau # Some strategies pass alpha as tau.
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.num_instances = num_instances
        self.logs_q = logs_q
        self.exclude_zero = exclude_zero
        
        hidden_dim = hidden_sizes[0]

        if dist_type == "gaussian":
            self.actor = MFGaussianActor(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
            self.target_entropy = -float(self.action_dim)
        elif dist_type == "dirichlet":
            self.actor = MFDirichletActor(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
            uniform_alpha = torch.ones(self.action_dim, device=self.device)
            uniform_dist = torch.distributions.Dirichlet(uniform_alpha)
            max_entropy = uniform_dist.entropy().item()
            self.target_entropy = max_entropy * 0.5
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        self.critic = MFCritic(state_dim, self.action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.critic_target = MFCritic(state_dim, self.action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.mf_net = MF(state_dim + mf_dim, mf_dim, mf_hidden_sizes, num_instances).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)

        self.log_alpha = torch.zeros(num_instances, 1, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.use_per = use_per
        if use_per:
            # PER not fully implemented in sac_buffer yet, fallback to normal MultiAgentSACReplayBuffer or pass.
            self.memory = MultiAgentSACReplayBuffer(num_instances, node_type, buffer_size, state_dim, self.action_dim,mf_dim, self.device)
        else:
            self.memory = MultiAgentSACReplayBuffer(num_instances, node_type, buffer_size, state_dim, self.action_dim,mf_dim, self.device)

        self.mf_loss_fn = nn.MSELoss(reduction='none')
        self.learn_step_counter = 0

    @property
    def get_alpha(self):
        return self.log_alpha.exp()

    def choose_action_batch(self, states_batch, prev_mfs_batch, agent_indices=None, deterministic: bool = False):
        # Zeta is ignored for continuous, passed for interface compatibility
        batch_size = states_batch.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)

        if not torch.is_tensor(states_batch):
            states_batch = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        else:
            states_batch = states_batch.to(self.device)
            
        if not torch.is_tensor(prev_mfs_batch):
            prev_mfs_batch = torch.as_tensor(prev_mfs_batch, dtype=torch.float32, device=self.device)
        else:
            prev_mfs_batch = prev_mfs_batch.to(self.device)

        is_policy_agent = torch.tensor([
            self.memory.get_len(aid.item()) >= self.min_batch_size 
            for aid in agent_indices
        ], device=self.device)
        
        final_actions = torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=self.device)

        cold_mask = ~is_policy_agent
        if cold_mask.any():
            indices = cold_mask.nonzero(as_tuple=True)[0]
            if self.dist_type == "dirichlet":
                if deterministic:
                    final_actions[indices] = 1.0 / self.action_dim
                else:
                    d = torch.distributions.Dirichlet(torch.ones((len(indices), self.action_dim), device=self.device))
                    final_actions[indices] = d.sample()
            else:
                if deterministic:
                    final_actions[indices] = 0.0
                else:
                    final_actions[indices] = torch.rand((len(indices), self.action_dim), device=self.device) * 2 - 1.0

        policy_mask = is_policy_agent
        if policy_mask.any():
            indices = policy_mask.nonzero(as_tuple=True)[0]
            s_subset = states_batch[indices]
            mf_subset = prev_mfs_batch[indices]
            aid_subset = agent_indices[indices]

            with torch.no_grad():
                pred_mf = self.mf_net(torch.cat([s_subset, mf_subset], dim=-1), indices=aid_subset)
                action, _ = self.actor.sample(s_subset, pred_mf, indices=aid_subset, deterministic=deterministic)
                final_actions[indices] = action

        # MFSAC agent should return the batch of actions directly since it's continuous
        return final_actions.cpu().numpy()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids):
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(self.device).float()
            
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids)

        # train mf
        pred_curr_mf = self.mf_net(torch.cat([states, prev_mfs], dim=-1), indices=agent_ids)
        mf_loss = (self.mf_loss_fn(pred_curr_mf, curr_mfs).mean(dim=1)).mean()
        self.mf_optimizer.zero_grad()
        mf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), 5.0)
        self.mf_optimizer.step()
        return mf_loss.item()

    def learn(self, agents_ids: torch.Tensor = None):
        ready_pool = (self.memory.buffer_sizes >= self.min_batch_size).nonzero(as_tuple=True)[0]
        
        if agents_ids is not None:
            if not isinstance(agents_ids, torch.Tensor):
                agents_ids = torch.tensor(agents_ids, device=self.device)
            agents_ids = agents_ids.to(self.device).view(-1)
            mask = torch.isin(agents_ids, ready_pool)
            target_agents = agents_ids[mask]
        else:
            target_agents = ready_pool

        if len(target_agents) == 0:
            return None

        if self.use_per:
            states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, indices, weights = \
                self.memory.sample(self.batch_size, agent_ids=target_agents)
        else:
            states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids = \
                self.memory.sample(self.batch_size, agent_ids=target_agents)
            weights = torch.ones_like(rewards).view(-1)

        agent_ids = agent_ids.view(-1)
        actions = actions.float()
        if actions.shape[-1] == 1 and self.action_dim > 1: # Adjust shape if incorrectly recorded as 1D
            actions = actions.view(-1, self.action_dim)
            
        pred_curr_mf = self.mf_net(torch.cat([states, prev_mfs], dim=-1), indices=agent_ids)

        # 2. Train Critic
        with torch.no_grad():
            next_pred_mf = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            next_action, next_log_prob = self.actor.sample(next_states, next_pred_mf, indices=agent_ids)
            
            target_q1, target_q2 = self.critic_target(next_states, next_action, next_pred_mf, indices=agent_ids)
            target_q = torch.min(target_q1, target_q2) - self.get_alpha[agent_ids] * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions, pred_curr_mf.detach(), indices=agent_ids)
        
        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q
        
        critic_loss1 = (0.5 * (td_error1 ** 2) * weights.view(-1, 1)).mean()
        critic_loss2 = (0.5 * (td_error2 ** 2) * weights.view(-1, 1)).mean()
        critic_loss = critic_loss1 + critic_loss2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        if self.use_per:
            new_priorities = (torch.abs(td_error1) + torch.abs(td_error2)).view(-1) / 2.0 + 1e-6
            self.memory.update_priorities(agent_ids, indices, new_priorities.detach())

        # 3. Train Actor
        new_action, log_prob = self.actor.sample(states, pred_curr_mf.detach(), indices=agent_ids)
        q1_new, q2_new = self.critic(states, new_action, pred_curr_mf.detach(), indices=agent_ids)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.get_alpha[agent_ids].detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 4. Train Alpha
        alpha_loss = -(self.log_alpha[agent_ids] * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 5. Soft update
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.learn_step_counter += 1
        
        q_min = q_new.min().item()
        q_max = q_new.max().item()
        q_mean = q_new.mean().item()
        
        step=10
        if self.node_type=="Offload_Group":step=100
        if self.learn_step_counter % step == 0:
            print(f"[{self.node_type} Group] Step {self.learn_step_counter:5d} | C-Loss: {critic_loss.item():.5f} | A-Loss: {actor_loss.item():.5f} | Q [Min: {q_min:.3f}, Max: {q_max:.3f}, Mean: {q_mean:.3f}]")

        if self.logs_q:
            return {
                "loss": critic_loss.item(),
                "q_min": q_min,
                "q_max": q_max,
                "q_mean": q_mean
            }
        return critic_loss.item()
