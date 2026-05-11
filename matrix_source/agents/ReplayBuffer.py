import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        # Pre-allocate with torch tensors on the specified device
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.prev_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.curr_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        # Tracking which agent generated the transition
        self.agent_id = torch.zeros((max_size, 1), dtype=torch.int64, device=device)

    def _to_tensor(self, x, dtype):
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, state, prev_mf, curr_mf, action, reward, next_state, done, agent_id=0):
        self.state[self.ptr] = self._to_tensor(state, torch.float32)
        self.prev_mf[self.ptr] = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[self.ptr] = self._to_tensor(curr_mf, torch.float32)
        self.action[self.ptr] = self._to_tensor(action, torch.int64)
        self.reward[self.ptr] = self._to_tensor(reward, torch.float32)
        self.next_state[self.ptr] = self._to_tensor(next_state, torch.float32)
        self.done[self.ptr] = self._to_tensor(done, torch.float32)
        self.agent_id[self.ptr] = self._to_tensor(agent_id, torch.int64)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids):
        batch_size = states.shape[0]
        if batch_size == 0: return
        
        indices = torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.max_size
        
        self.state[indices] = states.detach().to(device=self.device, dtype=torch.float32)
        self.prev_mf[indices] = prev_mfs.detach().to(device=self.device, dtype=torch.float32)
        self.curr_mf[indices] = curr_mfs.detach().to(device=self.device, dtype=torch.float32)
        
        # Ensure actions/rewards/dones/agent_ids are 2D (Batch, 1)
        self.action[indices] = actions.detach().to(device=self.device, dtype=torch.int64).view(-1, 1)
        self.reward[indices] = rewards.detach().to(device=self.device, dtype=torch.float32).view(-1, 1)
        self.next_state[indices] = next_states.detach().to(device=self.device, dtype=torch.float32)
        self.done[indices] = dones.detach().to(device=self.device, dtype=torch.float32).view(-1, 1)
        self.agent_id[indices] = agent_ids.detach().to(device=self.device, dtype=torch.int64).view(-1, 1)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            return None
        
        num_to_sample = min(batch_size, self.size)
        ind = torch.randint(0, self.size, (num_to_sample,), device=self.device)
            
        return (
            self.state[ind],
            self.prev_mf[ind],
            self.curr_mf[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
            self.agent_id[ind].squeeze(1) # Return as (Batch,) for indexing
        )

    def __len__(self):
        return self.size

class MultiAgentReplayBuffer:
    def __init__(self, num_agents, max_size_per_agent, state_dim, action_dim, device="cpu"):
        self.num_agents = num_agents
        self.device = device
        self.buffers = [
            ReplayBuffer(max_size_per_agent, state_dim, action_dim, device)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long, device=device)
        self.total_size = 0

    def add_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids):
        # Maintain everything in tensor form
        a_ids = agent_ids.view(-1)
        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            self.buffers[a_id].add(
                states[i], prev_mfs[i], curr_mfs[i], actions[i], rewards[i], next_states[i], dones[i], a_id
            )
            # Update tracking tensor
            self.buffer_sizes[a_id] = self.buffers[a_id].size
            
        self.total_size = self.buffer_sizes.sum().item()

    def sample(self, batch_sizes, agent_ids=None):
        # 1. Determine which agents to sample from
        if agent_ids is None:
            # Default: Sample from all agents that have any data
            agent_ids = (self.buffer_sizes > 0).nonzero(as_tuple=True)[0]
        
        if len(agent_ids) == 0:
            return None
        
        if not isinstance(agent_ids, torch.Tensor):
            agent_ids = torch.tensor(agent_ids, device=self.device)

        # 2. To avoid duplicates and ensure fairness:
        num_agents_to_pick = min(batch_sizes, len(agent_ids))
        
        # Use torch for random selection (uniform without replacement)
        perm = torch.randperm(len(agent_ids), device=self.device)[:num_agents_to_pick]
        final_agent_ids = agent_ids[perm]
        
        # 3. Collect 1 sample from each chosen agent's buffer
        samples = [self.buffers[int(a_id)].sample(1) for a_id in final_agent_ids]
        
        # 4. Collate (Stack tensors)
        collated = []
        for i in range(8): # 8 fields in transition
            collated.append(torch.cat([s[i] for s in samples if s is not None], dim=0))
        
        collated[7] = collated[7].squeeze(-1)
        return tuple(collated)

    def get_len(self, agent_id=None):
        if agent_id is None:
            return self.total_size
        return self.buffer_sizes[agent_id].item()

    def __len__(self):
        return self.total_size