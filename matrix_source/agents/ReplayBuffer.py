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

    def _to_tensor(self, x, dtype):
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def add(self, state, prev_mf, curr_mf, action, reward, next_state, done):
        self.state[self.ptr] = self._to_tensor(state, torch.float32)
        self.prev_mf[self.ptr] = self._to_tensor(prev_mf, torch.float32)
        self.curr_mf[self.ptr] = self._to_tensor(curr_mf, torch.float32)
        self.action[self.ptr] = self._to_tensor(action, torch.int64)
        self.reward[self.ptr] = self._to_tensor(reward, torch.float32)
        self.next_state[self.ptr] = self._to_tensor(next_state, torch.float32)
        self.done[self.ptr] = self._to_tensor(done, torch.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Sample indices using torch for speed
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
            self.done[ind]
        )

    def __len__(self):
        return self.size