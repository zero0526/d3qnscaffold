import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm, MultiInstanceNoisyLinear
from matrix_source.agents.buffer.PrioritizedReplayBuffer import MultiAgentPrioritizedReplayBuffer

class RainbowNetwork(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_atoms, num_instances=1):
        super().__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.num_instances = num_instances
        
        h1, h2 = hidden_sizes
        
        # Shared Backbone
        self.l1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.l2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        
        # Dueling & Noisy Heads
        # Value stream
        self.v1 = MultiInstanceNoisyLinear(num_instances, h2, h2)
        self.v2 = MultiInstanceNoisyLinear(num_instances, h2, num_atoms)
        
        # Advantage stream
        self.a1 = MultiInstanceNoisyLinear(num_instances, h2, h2)
        self.a2 = MultiInstanceNoisyLinear(num_instances, h2, action_dim * num_atoms)

    def forward(self, state, pred_mf, indices=None):
        x = torch.cat([state, pred_mf], dim=-1)
        
        # Shared backbone (Linear)
        x = F.silu(self.norm1(self.l1(x, indices), indices))
        x = F.silu(self.norm2(self.l2(x, indices), indices))
        
        # Heads (Noisy)
        v = F.silu(self.v1(x, indices))
        v = self.v2(v, indices).view(-1, 1, self.num_atoms) # (Batch, 1, Atoms)
        
        a = F.silu(self.a1(x, indices))
        a = self.a2(a, indices).view(-1, self.action_dim, self.num_atoms) # (Batch, Actions, Atoms)
        
        # Combine Dueling: Q = V + (A - mean(A))
        q_atoms = v + (a - a.mean(dim=1, keepdim=True))
        
        # Return probability distribution per action
        return F.softmax(q_atoms, dim=-1)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, MultiInstanceNoisyLinear):
                m.reset_noise()

class MFNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, num_instances=1):
        super().__init__()
        h = hidden_sizes[0]
        self.l1 = MultiInstanceLinear(num_instances, input_size, h)
        self.norm = MultiInstanceRMSNorm(num_instances, h)
        self.l2 = MultiInstanceLinear(num_instances, h, output_size)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.l1(x, indices), indices))
        return torch.sigmoid(self.l2(x, indices))

class RainbowMultiAgent:
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim, mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64), lr=1e-4, gamma=0.99, buffer_size=100000, batch_size=64,
                 num_atoms=51, v_min=-10.0, v_max=10.0, n_step=3,
                 num_instances=1, device=None):
        self.node_id = node_id
        self.node_type = node_type
        self.action_dim = action_dim
        self.u_action_dim = u_action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_instances = num_instances
        self.min_batch_size = buffer_min_size
        
        # Categorical parameters
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, num_atoms, device=self.device)
        
        # Networks
        self.eval_net = RainbowNetwork(state_dim, action_dim, u_action_dim, hidden_sizes, num_atoms, num_instances).to(self.device)
        self.target_net = RainbowNetwork(state_dim, action_dim, u_action_dim, hidden_sizes, num_atoms, num_instances).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        
        self.mf_net = MFNet(state_dim + action_dim, action_dim, mf_hidden_sizes, num_instances).to(self.device)
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        
        # Buffer
        self.memory = MultiAgentPrioritizedReplayBuffer(num_instances, node_type, buffer_size, state_dim, action_dim, self.device, n_step=n_step)
        
        self.learn_step_counter = 0

    def choose_action(self, state, prev_mf, zeta, mask=None, agent_idx=0):
        idx_tensor = torch.tensor([agent_idx], device=self.device)
        actions = self.choose_action_batch(
            state.unsqueeze(0) if not torch.is_tensor(state) else state.detach().unsqueeze(0),
            prev_mf.unsqueeze(0) if not torch.is_tensor(prev_mf) else prev_mf.detach().unsqueeze(0),
            zeta, 
            masks_batch=mask.unsqueeze(0) if mask is not None else None,
            agent_indices=idx_tensor
        )
        return int(actions[0])

    def choose_action_batch(self, states_batch, prev_mfs_batch, masks_batch=None, agent_indices=None):
        batch_size = states_batch.shape[0]
        if agent_indices is None:
            agent_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        else:
            agent_indices = agent_indices.to(self.device).view(-1)
            
        states_batch = states_batch.to(self.device)
        prev_mfs_batch = prev_mfs_batch.to(self.device)

        is_policy_agent = torch.tensor([
            self.memory.get_len(aid.item()) >= self.min_batch_size 
            for aid in agent_indices
        ], device=self.device)
        
        final_actions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        cold_mask = ~is_policy_agent
        if cold_mask.any():
            indices = cold_mask.nonzero(as_tuple=True)[0]
            if masks_batch is not None:
                m = masks_batch[indices].to(self.device)
                probs = m / m.sum(dim=1, keepdim=True).clamp(min=1e-8)
                final_actions[indices] = torch.multinomial(probs, 1).squeeze(1)
            else:
                final_actions[indices] = torch.randint(0, self.u_action_dim, (len(indices),), device=self.device)

        policy_mask = is_policy_agent
        if policy_mask.any():
            indices = policy_mask.nonzero(as_tuple=True)[0]
            s_sub = states_batch[indices]
            mf_sub = prev_mfs_batch[indices]
            aid_sub = agent_indices[indices]
            
            with torch.no_grad():
                pred_mf = self.mf_net(torch.cat([s_sub, mf_sub], dim=-1), indices=aid_sub)
                dist = self.eval_net(s_sub, pred_mf, indices=aid_sub)
                q_values = (dist * self.support).sum(dim=2)
                
                if masks_batch is not None:
                    m = masks_batch[indices].to(self.device)
                    q_values = q_values + (m - 1.0) * 1e10
                
                # Rainbow with Noisy Nets uses greedy action selection
                final_actions[indices] = q_values.argmax(dim=1)
        
        return final_actions.tolist()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids):
        # res: (states, prev_mfs, curr_mfs, agent_ids) of finalized N-step transitions
        res = self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids)
        if res is not None:
            loss = self.learn_mf_batch(res[0], res[1], res[2], res[3])
            return loss
        return 0.0

    def learn_mf_batch(self, states, prev_mf, gt_mf, agent_ids):
        s = states.to(self.device)
        pmf = prev_mf.to(self.device)
        gt = gt_mf.to(self.device)
        pred = self.mf_net(torch.cat([s, pmf], dim=-1), indices=agent_ids)
        loss = F.mse_loss(pred, gt)
        self.mf_optimizer.zero_grad()
        loss.backward()
        self.mf_optimizer.step()
        return loss.item()

    def learn(self, requested_agent_ids=None):
        ready_pool = (self.memory.buffer_sizes >= self.min_batch_size).nonzero(as_tuple=True)[0]
        if requested_agent_ids is not None:
            a_ids = requested_agent_ids.to(self.device).view(-1)
            target_agents = a_ids[torch.isin(a_ids, ready_pool)]
        else:
            target_agents = ready_pool
            
        if len(target_agents) == 0: return None
        
        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids, indices, weights = \
            self.memory.sample(self.batch_size, agent_ids=target_agents)
            
        with torch.no_grad():
            next_mfs = self.mf_net(torch.cat([next_states, curr_mfs], dim=-1), indices=agent_ids)
            next_dist = self.target_net(next_states, next_mfs, indices=agent_ids)
            
            next_eval_dist = self.eval_net(next_states, next_mfs, indices=agent_ids)
            next_eval_q = (next_eval_dist * self.support).sum(dim=2)
            best_actions = next_eval_q.argmax(dim=1)
            
            next_best_dist = next_dist[range(self.batch_size), best_actions]
            
            n_step_gamma = self.gamma ** self.memory.buffers[0].n_step
            tz = rewards + n_step_gamma * self.support.unsqueeze(0) * (1 - dones)
            tz = tz.clamp(self.v_min, self.v_max)
            
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            
            target_dist = torch.zeros((self.batch_size, self.num_atoms), device=self.device)
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size, device=self.device).long().unsqueeze(1)
            
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_best_dist * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_best_dist * (b - l.float())).view(-1))

        curr_mfs_pred = self.mf_net(torch.cat([states, curr_mfs], dim=-1), indices=agent_ids)
        eval_dist = self.eval_net(states, curr_mfs_pred, indices=agent_ids)
        eval_action_dist = eval_dist[range(self.batch_size), actions.squeeze()]
        
        loss_samples = -(target_dist * (eval_action_dist + 1e-8).log()).sum(dim=1)
        loss = (loss_samples * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        new_priorities = loss_samples.detach() + 1e-6
        self.memory.update_priorities(agent_ids, indices, new_priorities)
        
        self.learn_step_counter += 1
        self.eval_net.reset_noise()
        self.target_net.reset_noise()
        
        if self.learn_step_counter % 10 == 0:
            self._soft_update()
            
        if self.learn_step_counter % 20 == 0:
            avg_q = (eval_action_dist * self.support).sum(dim=1).mean().item()
            print(f"[{self.node_type} Rainbow] Step {self.learn_step_counter:5d} | Loss: {loss.item():.5f} | Avg Q: {avg_q:.3f}")
            
        return loss.item()

    def _soft_update(self, tau=0.005):
        with torch.no_grad():
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)