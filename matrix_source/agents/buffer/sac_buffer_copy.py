import torch

class SACReplayBuffer:
    def __init__(self, max_size, node_type, state_dim, node_dim, mf_dim, device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        self.node_type = node_type

        self.prev_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.prev_mf = torch.zeros((max_size, mf_dim), dtype=torch.float32, device=device)

        # ACTION IS FLOAT32 IN SAC
        self.action = torch.zeros((max_size, node_dim), dtype=torch.float32, device=device)

        # --- THÊM MASK VÀO ĐÂY ---
        # Mặc định bằng 1.0 (tất cả đều hợp lệ) để tránh lỗi khi backward
        self.action_mask = torch.ones((max_size, node_dim), dtype=torch.float32, device=device)

        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.curr_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.curr_mf = torch.zeros((max_size, mf_dim), dtype=torch.float32, device=device)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.agent_id = torch.zeros((max_size, 1), dtype=torch.int64, device=device)

    def _to_tensor(self, x, dtype):
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, prev_state, prev_mf, action, reward, curr_state, curr_mf, done, agent_id=0, action_mask=None):
        self.prev_state[self.ptr] = self._to_tensor(prev_state, torch.float32)
        self.prev_mf[self.ptr] = self._to_tensor(prev_mf, torch.float32)

        act_t = self._to_tensor(action, torch.float32)
        if act_t.dim() == 0:
            act_t = act_t.view(1)
        self.action[self.ptr] = act_t

        # --- LƯU MASK ---
        if action_mask is not None:
            self.action_mask[self.ptr] = self._to_tensor(action_mask, torch.float32)
        else:
            self.action_mask[self.ptr] = 1.0  # Fallback an toàn

        self.reward[self.ptr] = self._to_tensor(reward, torch.float32)
        self.curr_state[self.ptr] = self._to_tensor(curr_state, torch.float32)
        self.curr_mf[self.ptr] = self._to_tensor(curr_mf, torch.float32)
        self.done[self.ptr] = self._to_tensor(done, torch.float32)
        self.agent_id[self.ptr] = self._to_tensor(agent_id, torch.int64)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, prev_states, prev_mfs, actions, rewards, curr_states, curr_mfs, dones, agent_ids,
                  action_masks=None):
        batch_size = prev_states.shape[0]
        if batch_size == 0: return

        indices = torch.arange(self.ptr, self.ptr + batch_size, device=self.device) % self.max_size

        self.prev_state[indices] = prev_states.detach().to(device=self.device, dtype=torch.float32)
        self.prev_mf[indices] = prev_mfs.detach().to(device=self.device, dtype=torch.float32)

        acts = actions.detach().to(device=self.device, dtype=torch.float32)
        if acts.dim() == 1:
            acts = acts.view(-1, 1)
        self.action[indices] = acts

        # --- LƯU MASK BATCH ---
        if action_masks is not None:
            self.action_mask[indices] = action_masks.detach().to(device=self.device, dtype=torch.float32)
        else:
            self.action_mask[indices] = 1.0  # Fallback an toàn

        self.reward[indices] = rewards.detach().to(device=self.device, dtype=torch.float32).view(-1, 1)
        self.curr_state[indices] = curr_states.detach().to(device=self.device, dtype=torch.float32)
        self.curr_mf[indices] = curr_mfs.detach().to(device=self.device, dtype=torch.float32)
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
            self.prev_state[ind].to(self.device),
            self.prev_mf[ind].to(self.device),
            self.action[ind].to(self.device),
            self.action_mask[ind].to(self.device),  # <--- THÊM VÀO TUPLE TRẢ VỀ
            self.reward[ind].to(self.device),
            self.curr_state[ind].to(self.device),
            self.curr_mf[ind].to(self.device),
            self.done[ind].to(self.device),
            self.agent_id[ind].squeeze(-1).to(self.device)
        )

    def __len__(self):
        return self.size

class MultiAgentSACReplayBuffer:
    def __init__(self, num_agents, node_type, max_size_per_agent, state_dim,node_dim, mf_dim, device="cpu"):
        self.num_agents = num_agents
        self.device = device
        self.node_type = node_type
        self.buffers = [
            SACReplayBuffer(max_size_per_agent, node_type, state_dim, node_dim, mf_dim, device)
            for _ in range(num_agents)
        ]
        self.buffer_sizes = torch.zeros(num_agents, dtype=torch.long, device=device)
        self.total_size = 0
        self.total_adds = 0
        self.log_interval = 5000*num_agents if node_type=="Terminal_Group" else 500*num_agents

    def add_batch(self, states, prev_mf, actions, rewards, curr_states, curr_mf, dones, agent_ids, action_masks=None):
        a_ids = agent_ids.view(-1)
        for i in range(len(a_ids)):
            a_id = int(a_ids[i])
            self.buffers[a_id].add(
                states[i], prev_mf[i], actions[i], rewards[i], curr_states[i], curr_mf[i], dones[i], a_id,
                action_mask=action_masks[i] if action_masks is not None else None
            )
            self.buffer_sizes[a_id] = self.buffers[a_id].size

        self.total_size = self.buffer_sizes.sum().item()
        self.total_adds += len(a_ids)

    def sample(self, batch_sizes, agent_ids=None):
        if agent_ids is None:
            agent_ids = (self.buffer_sizes > 0).nonzero(as_tuple=True)[0]

        if len(agent_ids) == 0:
            return None

        if not isinstance(agent_ids, torch.Tensor):
            agent_ids = torch.tensor(agent_ids, device=self.device)

        num_agents = len(agent_ids)
        samples_per_agent = batch_sizes // num_agents
        remainder = batch_sizes % num_agents

        perm = torch.randperm(num_agents, device=self.device)
        shuffled_agent_ids = agent_ids[perm]

        samples = []
        for i, a_id in enumerate(shuffled_agent_ids):
            n = samples_per_agent + (1 if i < remainder else 0)
            valid_n = min(n, int(self.buffer_sizes[a_id].item()))
            if valid_n > 0:
                agent_samples = self.buffers[int(a_id)].sample(valid_n)
                if agent_samples is not None:
                    samples.append(agent_samples)

        if not samples:
            return None

        collated = []
        # SỬA: Tăng từ 8 lên 9 vì có thêm action_mask
        for i in range(9):
            collated.append(torch.cat([s[i] for s in samples], dim=0))

        return tuple(collated)

    def get_len(self, agent_id=None):
        if agent_id is None:
            return self.total_size
        return self.buffer_sizes[agent_id].item()

    def __len__(self):
        return self.total_size

