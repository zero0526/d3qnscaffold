import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from matrix_source.agents.d3qn import MultiInstanceLinear
from matrix_source.agents.policy_replay_buffer import MultiAgentPolicyBuffer

class MultiInstanceRMSNorm(nn.Module):
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma = self.weight[indices]
        return x_norm * gamma

class MultiInstanceActorCritic(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances
        
        # Shared Feature Extractor
        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        
        # Actor head
        self.actor_logits = MultiInstanceLinear(num_instances, h2, action_dim)
        
        # Critic head
        self.critic = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, state, mf, indices=None):
        x = torch.cat([state, mf], dim=-1)
        x = F.relu(self.norm1(self.fc1(x, indices), indices))
        x = F.relu(self.norm2(self.fc2(x, indices), indices))
        
        logits = self.actor_logits(x, indices)
        value = self.critic(x, indices)
        return logits, value

class MFNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1 = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.out = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.relu(self.norm(self.fc1(x, indices), indices))
        x = F.relu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices)) # Constrain MF to [0, 1] range

class PPOAgent:
    def __init__(self, node_id, node_type, state_dim, action_dim, u_action_dim, 
                 mf_hidden_sizes, mf_lr, buffer_min_size, hidden_sizes, lr, gamma, alpha, 
                 buffer_size, batch_size, num_instances=1, device="cpu"):
        
        self.node_id = node_id
        self.node_type = node_type
        self.device = device
        self.num_instances = num_instances
        self.action_dim = u_action_dim
        self.mf_dim = action_dim # Dimensions of mean field (usually num_services)
        
        self.gamma = gamma
        self.lmbda = 0.95 # GAE lambda
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coef = 0.01
        self.min_batch_size = buffer_min_size
        self.batch_size = batch_size

        # Networks
        self.ac_net = MultiInstanceActorCritic(state_dim, self.mf_dim, self.action_dim, hidden_sizes, num_instances).to(device)
        self.mf_net = MFNetwork(state_dim + self.mf_dim, self.mf_dim, mf_hidden_sizes, num_instances).to(device)
        
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.MSELoss()

        self.memory = MultiAgentPolicyBuffer(num_instances, buffer_size, state_dim, self.mf_dim, device)
        self.learn_step_counter = 0

    def choose_action_batch(self, states, mfs, zeta=1.0, agent_indices=None):
        # zeta is used here to match d3qn signature, can act as temperature for policy
        if agent_indices is None:
            agent_indices = torch.zeros(states.shape[0], dtype=torch.long, device=self.device)
            
        states = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        mfs = torch.as_tensor(mfs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            # 1. Predict current MF
            pred_mfs = self.mf_net(torch.cat([states, mfs], dim=-1), indices=agent_indices)
            
            # 2. Get Actor Login and Critic Values
            logits, values = self.ac_net(states, pred_mfs, indices=agent_indices)
            
            # 3. Sample actions
            probs = torch.softmax(logits * zeta, dim=-1)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def store_transition_train_mf_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, log_probs, values, agent_ids):
        # 1. Train MF (supervised learning)
        loss_mf = self.learn_mf_batch(states, prev_mfs, curr_mfs, agent_ids)
        
        # 2. Store in Buffer
        self.memory.add_batch(states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, log_probs, values, agent_ids)
        return loss_mf

    def learn_mf_batch(self, states, prev_mfs, ground_truth_mfs, agent_ids):
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        pmf = torch.as_tensor(prev_mfs, device=self.device, dtype=torch.float32)
        gt_mf = torch.as_tensor(ground_truth_mfs, device=self.device, dtype=torch.float32)
        
        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), indices=agent_ids)
        loss = F.mse_loss(pred_mf, gt_mf)
        
        self.mf_optimizer.zero_grad()
        loss.backward()
        self.mf_optimizer.step()
        return loss.item()

    def learn(self, agents_ids: torch.Tensor = None):
        # PPO Learning from collected buffer
        data = self.memory.get_all_ready(min_size=self.min_batch_size, agent_ids_pool=agents_ids)
        if data is None:
            return None
        
        # Unpack data
        states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, old_log_probs, old_values, agent_ids = data
        
        # 1. Compute Advantages and Targets (Vectorized per agent)
        with torch.no_grad():
            # Get next values for bootstrap
            _, next_values = self.ac_net(next_states, curr_mfs, indices=agent_ids)
            
            # GAE Calculation (Vectorized over multiple agents)
            deltas = rewards + self.gamma * next_values * (1 - dones) - old_values
            advantages = torch.zeros_like(deltas)
            
            advantage = 0
            # Identify agent boundaries to prevent advantage bleed
            for t in reversed(range(len(deltas))):
                # If we are at the end of the batch OR the next transition belongs to a different agent, reset advantage
                if t < len(deltas) - 1 and agent_ids[t] != agent_ids[t+1]:
                    advantage = 0
                
                # Standard GAE update
                advantage = deltas[t] + self.gamma * self.lmbda * (1 - dones[t]) * advantage
                advantages[t] = advantage
            
            returns = advantages + old_values
            # Standardize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. PPO Update Epochs
        epoch_loss = 0
        for _ in range(self.k_epochs):
            # Evaluate current policy
            logits, values = self.ac_net(states, curr_mfs, indices=agent_ids)
            dist = Categorical(logits=logits)
            
            new_log_probs = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
            entropy = dist.entropy().mean()
            
            # Ratio for clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Actor Loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic Loss (MSE)
            critic_loss = F.mse_loss(values, returns)
            
            # Total Loss
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()

        self.learn_step_counter += 1
        if self.learn_step_counter % 10 == 0:
            print(f"[{self.node_type} PPO] Step {self.learn_step_counter} | Loss: {epoch_loss/self.k_epochs:.4f}")

        # Clear buffer after learning (PPO is on-policy)
        self.memory.clear()
        
        return epoch_loss / self.k_epochs
