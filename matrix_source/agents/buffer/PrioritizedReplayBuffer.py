import torch
import collections

class PrioritizedReplayBuffer:
    def __init__(self, max_size, node_type, state_dim, action_dim, device="cpu", alpha=0.6, beta_start=0.4, beta_frames=100000, n_step=1, gamma=0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.node_type = node_type
        
        # PER parameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0
        
        # N-step parameters
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = collections.deque(maxlen=n_step)

        # Pre-allocate with torch tensors
        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.prev_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.curr_mf = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, 1), dtype=torch.int64, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.agent_id = torch.zeros((max_size, 1), dtype=torch.int64, device=device)
        
        # Priorities as tensor
        self.priorities = torch.zeros((max_size,), dtype=torch.float32, device=device)

    def _to_tensor(self, x, dtype):
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, state, prev_mf, curr_mf, action, reward, next_state, done, agent_id=0):
        self.n_step_buffer.append((state, prev_mf, curr_mf, action, reward, next_state, done, agent_id))
        if len(self.n_step_buffer) < self.n_step:
            return None

        state, prev_mf, curr_mf, action, curr_reward, next_state, done, agent_id = self._get_n_step_info()

        idx = self.ptr
        self.state[idx] = self._to_tensor(state, torch.float32)
        self.prev_mf[idx] = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[idx] = self._to_tensor(curr_mf, torch.float32)
        self.action[idx] = self._to_tensor(action, torch.int64)
        self.reward[idx] = self._to_tensor(curr_reward, torch.float32)
        self.next_state[idx] = self._to_tensor(next_state, torch.float32)
        self.done[idx] = self._to_tensor(done, torch.float32)
        self.agent_id[idx] = self._to_tensor(agent_id, torch.int64)

        max_prio = self.priorities.max().item() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return (state, prev_mf, curr_mf, agent_id)

    def _get_n_step_info(self):
        state, prev_mf, curr_mf, action, reward, next_state, done, agent_id = self.n_step_buffer[0]
        curr_reward = float(reward)
        curr_gamma = self.gamma
        
        for i in range(1, len(self.n_step_buffer)):
            s, pmf, cmf, a, r, ns, d, aid = self.n_step_buffer[i]
            curr_reward += curr_gamma * float(r)
            next_state, curr_mf, done = ns, cmf, d
            if d: break
            curr_gamma *= self.gamma
            
        return state, prev_mf, curr_mf, action, curr_reward, next_state, done, agent_id

    def sample(self, batch_size):
        if self.size == 0: return None
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return (
            self.state[indices].to(self.device),
            self.prev_mf[indices].to(self.device),
            self.curr_mf[indices].to(self.device),
            self.action[indices].to(self.device),
            self.reward[indices].to(self.device),
            self.next_state[indices].to(self.device),
            self.done[indices].to(self.device),
            self.agent_id[indices].squeeze(1).to(self.device),
            indices.to(self.device),
            weights.to(device=self.device, dtype=torch.float32)
        )

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities.squeeze().to(self.device)

    def __len__(self):
        return self.size

class MultiAgentPrioritizedReplayBuffer:
    def __init__(self, num_agents, node_type, max_size_per_agent, state_dim, action_dim, device="cpu", alpha=0.6, n_step=1):
        self.num_agents = num_agents
        self.device = device
        self.node_type = node_type
        self.buffers = [
            PrioritizedReplayBuffer(max_size_per_agent, node_type, state_dim, action_dim, device, alpha=alpha, n_step=n_step)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long, device=device)
        self.total_size = 0

    def add_batch(self, states, prev_mfs, curr_mfs, actions, rewards, next_states, dones, agent_ids):
        a_ids = agent_ids.view(-1)
        ready_states = []
        ready_prev_mfs = []
        ready_curr_mfs = []
        ready_aids = []

        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            res = self.buffers[a_id].add(
                states[i], prev_mfs[i], curr_mfs[i], actions[i], rewards[i], next_states[i], dones[i], a_id
            )
            self.buffer_sizes[a_id] = self.buffers[a_id].size
            if res is not None:
                ready_states.append(self.buffers[a_id]._to_tensor(res[0], torch.float32))
                ready_prev_mfs.append(self.buffers[a_id]._to_tensor(res[1], torch.float32))
                ready_curr_mfs.append(self.buffers[a_id]._to_tensor(res[2], torch.float32))
                ready_aids.append(torch.tensor([res[3]], device=self.device, dtype=torch.long))
            
        self.total_size = self.buffer_sizes.sum().item()

        if len(ready_states) > 0:
            return (
                torch.stack(ready_states),
                torch.stack(ready_prev_mfs),
                torch.stack(ready_curr_mfs),
                torch.cat(ready_aids)
            )
        return None

    def sample(self, batch_sizes, agent_ids=None):
        if agent_ids is None:
            agent_ids = (self.buffer_sizes > 0).nonzero(as_tuple=True)[0]
        if len(agent_ids) == 0: return None
        
        num_agents_to_pick = min(batch_sizes, len(agent_ids))
        perm = torch.randperm(len(agent_ids), device=self.device)[:num_agents_to_pick]
        final_agent_ids = agent_ids[perm]
        
        samples = [self.buffers[int(a_id)].sample(1) for a_id in final_agent_ids]
        collated = []
        for i in range(10): 
            collated.append(torch.cat([s[i] for s in samples if s is not None], dim=0))
        return tuple(collated)

    def update_priorities(self, agent_ids, indices, priorities):
        for i in range(len(agent_ids)):
            a_id = int(agent_ids[i])
            self.buffers[a_id].update_priorities(indices[i:i+1], priorities[i:i+1])

    def __len__(self):
        return self.total_size
