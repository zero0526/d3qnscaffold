import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm
from matrix_source.agents.buffer.td3_buffer import MultiAgentTD3ReplayBuffer

# -----------------------------
# Networks
# -----------------------------

class MFNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.out = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.fc1(x, indices), indices))
        x = F.silu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices))

class MFCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mf_dim: int, hidden_dim: int = 256, num_instances=1):
        super().__init__()
        self.num_instances = num_instances

        # Q1 architecture
        self.q1_l1 = MultiInstanceLinear(num_instances, state_dim + action_dim + mf_dim, hidden_dim)
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

class MFSimplexActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, mf_dim: int, hidden_dim: int = 256, num_instances=1):
        super().__init__()
        self.action_dim = action_dim
        self.num_instances = num_instances

        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, hidden_dim)
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.l2 = MultiInstanceLinear(num_instances, hidden_dim, hidden_dim)
        self.norm2 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        self.out = MultiInstanceLinear(num_instances, hidden_dim, action_dim)

    def forward(self, state, mf, indices=None, action_mask=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        logits = self.out(x, indices)

        if action_mask is not None:
            # Mask invalid nodes by setting logits to very low value
            logits = logits.masked_fill(action_mask == 0, -1e10)
        
        return torch.softmax(logits, dim=-1)

# -----------------------------
# Agent
# -----------------------------

class MFTD3Agent:
    def __init__(
        self,
        node_id,
        node_type,
        num_comp_node: int,
        state_dim: int,
        action_dim: int,
        mf_dim: int,
        mf_hidden_sizes=(50, 50),
        hidden_sizes=(256, 256),
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        expl_noise: float = 0.1,
        buffer_size: int = 100000,
        batch_size: int = 256,
        buffer_min_size: int = 4096,
        num_instances: int = 1,
        device=None,
        logs_q: bool = False,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.action_dim = action_dim
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.num_instances = num_instances
        self.logs_q = logs_q
        self.total_it = 0

        hidden_dim = hidden_sizes[0]

        self.actor = MFSimplexActor(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.actor_target = MFSimplexActor(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = MFCritic(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.critic_target = MFCritic(state_dim, action_dim, mf_dim, hidden_dim, num_instances).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.mf = MFNetwork(mf_dim + state_dim, mf_dim, mf_hidden_sizes, num_instances).to(self.device)
        self.mf_optimizer = optim.Adam(self.mf.parameters(), lr=actor_lr)

        self.memory = MultiAgentTD3ReplayBuffer(num_instances, node_type, buffer_size, state_dim, num_comp_node, mf_dim, self.device)
        self.learn_step_counter = 0

    def choose_action_batch(self, states_batch, mf, agent_indices=None, action_masks=None, deterministic: bool = False):
        batch_size = states_batch.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)

        if not torch.is_tensor(states_batch):
            states_batch = torch.as_tensor(states_batch, dtype=torch.float32, device=self.device)
        else:
            states_batch = states_batch.to(self.device)

        if action_masks is not None and not torch.is_tensor(action_masks):
            action_masks = torch.as_tensor(action_masks, dtype=torch.float32, device=self.device)
        elif action_masks is not None:
            action_masks = action_masks.to(self.device)

        is_policy_agent = torch.tensor([
            self.memory.get_len(aid.item()) >= self.min_batch_size 
            for aid in agent_indices
        ], device=self.device)
        
        final_actions = torch.zeros((batch_size, self.action_dim), dtype=torch.float32, device=self.device)

        cold_mask = ~is_policy_agent
        if cold_mask.any():
            indices = cold_mask.nonzero(as_tuple=True)[0]
            if deterministic:
                if action_masks is not None:
                    mask_subset = action_masks[indices]
                    sum_mask = mask_subset.sum(dim=-1, keepdim=True).clamp(min=1.0)
                    final_actions[indices] = mask_subset / sum_mask
                else:
                    final_actions[indices] = 1.0 / self.action_dim
            else:
                raw = torch.rand((len(indices), self.action_dim), device=self.device)
                if action_masks is not None:
                    raw = raw * action_masks[indices]
                final_actions[indices] = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)

        policy_mask = is_policy_agent
        if policy_mask.any():
            indices = policy_mask.nonzero(as_tuple=True)[0]
            s_subset = states_batch[indices]
            aid_subset = agent_indices[indices]
            mf_subset = mf[indices] if mf.dim() > 1 else mf
            mask_subset = action_masks[indices] if action_masks is not None else None

            with torch.no_grad():
                probs = self.actor(s_subset, mf_subset, indices=aid_subset, action_mask=mask_subset)
                
                if not deterministic:
                    noise = torch.randn_like(probs) * self.expl_noise
                    probs = (probs + noise).clamp(min=1e-6)
                    if mask_subset is not None:
                        probs = probs * mask_subset
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                final_actions[indices] = probs

        return final_actions.cpu().numpy()

    def store_transition_train_mf_batch(self, states, prev_mf, actions, rewards, curr_states, curr_mf, dones, agent_ids, action_masks=None):
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        else:
            actions = actions.to(self.device).float()
            
        self.memory.add_batch(states, prev_mf, actions, rewards, curr_states, curr_mf, dones, agent_ids, action_masks=action_masks)
        return True

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

        samples = self.memory.sample(self.batch_size, agent_ids=target_agents)
        if samples is None: return None
        states, prev_mfs, actions, action_masks, rewards, curr_states, curr_mfs, dones, agent_ids = samples

        agent_ids = agent_ids.view(-1)
        actions = actions.float()
        action_masks = action_masks.float()

        # 1. Train MF Predictor
        self.mf_optimizer.zero_grad()
        combined_s_mf = torch.cat([states, prev_mfs], dim=-1)
        pred_curr_mf = self.mf(combined_s_mf, agent_ids)
        mf_loss = F.mse_loss(pred_curr_mf, curr_mfs)
        mf_loss.backward()
        self.mf_optimizer.step()

        # 2. Train Critics
        with torch.no_grad():
            # Target Policy Smoothing
            next_actions = self.actor_target(curr_states, curr_mfs, indices=agent_ids, action_mask=action_masks)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_actions = (next_actions + noise).clamp(min=1e-6)
            next_actions = next_actions * action_masks # Re-apply mask
            next_actions = next_actions / (next_actions.sum(dim=-1, keepdim=True) + 1e-8)

            target_q1, target_q2 = self.critic_target(curr_states, next_actions, curr_mfs, indices=agent_ids)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions, prev_mfs, indices=agent_ids)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)

        # 3. Delayed Policy Update
        if self.total_it % self.policy_delay == 0:
            new_actions = self.actor(states, prev_mfs, indices=agent_ids, action_mask=action_masks)
            q1_new, _ = self.critic(states, new_actions, prev_mfs, indices=agent_ids)
            actor_loss = -q1_new.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Soft Update
            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.total_it += 1
        self.learn_step_counter += 1

        if self.logs_q:
            # For consistent logging with strategy expectations
            q1_new, _ = self.critic(states, actions, prev_mfs, indices=agent_ids)
            return {
                "loss": critic_loss.item(),
                "q_min": q1_new.min().item(),
                "q_max": q1_new.max().item(),
                "q_mean": q1_new.mean().item(),
                "mf_loss": mf_loss.item()
            }
        return critic_loss.item()

    def save(self, path):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'mf': self.mf.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'mf_opt': self.mf_optimizer.state_dict(),
            'total_it': self.total_it,
            'learn_step': self.learn_step_counter
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        if 'mf' in checkpoint: self.mf.load_state_dict(checkpoint['mf'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        if 'mf_opt' in checkpoint: self.mf_optimizer.load_state_dict(checkpoint['mf_opt'])
        self.total_it = checkpoint.get('total_it', 0)
        self.learn_step_counter = checkpoint.get('learn_step', 0)