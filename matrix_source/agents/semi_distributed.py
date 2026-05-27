import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random

# ==========================================
# 1. CÁC LỚP MULTI-INSTANCE CƠ BẢN (Giữ nguyên từ code cũ)
# ==========================================
class MultiInstanceLinear(nn.Module):
    # ... (Giữ nguyên code của bạn) ...
    def __init__(self, num_instances, in_features, out_features, bias=True):
        super().__init__()
        self.num_instances = num_instances
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_instances, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_instances, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_instances):
            nn.init.orthogonal_(self.weight[i], gain=1.0)
            if self.bias is not None:
                nn.init.zeros_(self.bias[i])

    def forward(self, x, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        w = self.weight[indices]
        out = torch.bmm(x.unsqueeze(1), w).squeeze(1)
        if self.bias is not None:
            out += self.bias[indices]
        return out

class MultiInstanceRMSNorm(nn.Module):
    # ... (Giữ nguyên code của bạn) ...
    def __init__(self, num_instances, normalized_shape, eps=1e-6):
        super().__init__()
        self.num_instances = num_instances
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_instances, normalized_shape))

    def forward(self, x, indices):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        gamma = self.weight[indices]
        return x_norm * gamma

# ==========================================
# 2. THÊM MULTI-INSTANCE GRU CELL (Mới)
# ==========================================
class MultiInstanceGRUCell(nn.Module):
    def __init__(self, num_instances, input_size, hidden_size):
        super().__init__()
        self.num_instances = num_instances
        self.hidden_size = hidden_size
        
        # Gate weights cho GRU
        self.weight_ih = nn.Parameter(torch.Tensor(num_instances, input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_instances, hidden_size, 3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(num_instances, 3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(num_instances, 3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_instances):
            nn.init.orthogonal_(self.weight_ih[i])
            nn.init.orthogonal_(self.weight_hh[i])
            nn.init.zeros_(self.bias_ih[i])
            nn.init.zeros_(self.bias_hh[i])

    def forward(self, x, hx, indices=None):
        if indices is None:
            indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            
        # Lấy weight tương ứng với instance
        w_ih = self.weight_ih[indices]
        w_hh = self.weight_hh[indices]
        b_ih = self.bias_ih[indices]
        b_hh = self.bias_hh[indices]

        # Tính toán gate: x_t * W_ih + h_t * W_hh + bias
        gi = torch.bmm(x.unsqueeze(1), w_ih).squeeze(1) + b_ih
        gh = torch.bmm(hx.unsqueeze(1), w_hh).squeeze(1) + b_hh
        i_r, i_i, i_n = gi.chunk(3, dim=-1)
        h_r, h_i, h_n = gh.chunk(3, dim=-1)

        # Công thức GRU
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = new_gate + input_gate * (hx - new_gate)

        return hy

# ==========================================
# 3. ACTOR & CRITIC MỚI CHO BÀI TOÁN TUẦN TỰ
# ==========================================
class SequentialGRUActor(nn.Module):
    def __init__(self, state_dim, mf_dim, action_dim, hidden_dim, num_instances=1):
        super().__init__()
        self.num_instances = num_instances
        
        # Nhận state (P_i, Q_s, F_s) và Mean Field
        self.fc1 = MultiInstanceLinear(num_instances, state_dim + mf_dim, hidden_dim)
        self.norm1 = MultiInstanceRMSNorm(num_instances, hidden_dim)
        
        # Thay FC2 bằng GRU Cell để nhớ chuỗi
        self.gru_cell = MultiInstanceGRUCell(num_instances, hidden_dim, hidden_dim)
        
        # Lớp xuất logits
        self.actor_logits = MultiInstanceLinear(num_instances, hidden_dim, action_dim)

    def forward_step(self, state, mf, hidden_state, indices=None):
        """Chạy 1 bước thời gian"""
        x = torch.cat([state, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        new_hidden = self.gru_cell(x, hidden_state, indices)
        logits = self.actor_logits(new_hidden, indices)
        return logits, new_hidden

    def evaluate(self, state, mf, hidden_state, action, masks=None, indices=None):
        logits, new_hidden = self.forward_step(state, mf, hidden_state, indices)
        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)
            
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, new_hidden


class GlobalAggregatedCritic(nn.Module):
    """
    Critic không dùng GRU. Nó nhận aggregated state cuối timeslot:
    [Mask, Q_final, F, Workload_sent, MF]
    """
    def __init__(self, global_state_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.num_instances = num_instances

        self.fc1 = MultiInstanceLinear(num_instances, global_state_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2 = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.critic = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, global_state, indices=None):
        x = F.silu(self.norm1(self.fc1(global_state, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.critic(x, indices).squeeze(-1)


class MFNetwork(nn.Module):
    # ... (Giữ nguyên code của bạn) ...
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


# ==========================================
# 4. SEQUENTIAL BUFFER (Mới)
# ==========================================

class SequentialMultiAgentBuffer:
    """
    Lưu trữ N trajectories (N time slots) để train PPO chuẩn.
    """
    def __init__(self, num_instances):
        self.num_instances = num_instances
        self.completed_trajectories = [] # Lưu danh sách các time slots đã hoàn thành
        self.current_trajectory = {}     # Đang chứa dữ liệu của time slot hiện tại
        self.reset_current()

    def reset_current(self):
        """Làm sạch bộ nhớ tạm cho time slot mới"""
        self.current_trajectory = {i: {} for i in range(self.num_instances)}

    def start_episode(self):
        """Gọi ở đầu mỗi time slot"""
        self.reset_current()

    def add_step(self, agent_id, state, hidden, action, log_prob, mf):
        """Gọi tại mỗi bước k khi GRU chọn xong 1 task"""
        aid = int(agent_id)
        if 'states' not in self.current_trajectory[aid]:
            self.current_trajectory[aid] = {
                'states': [], 'hiddens': [], 'actions': [], 'log_probs': [], 'mfs': []
            }
        self.current_trajectory[aid]['states'].append(state)
        self.current_trajectory[aid]['hiddens'].append(hidden)
        self.current_trajectory[aid]['actions'].append(action)
        self.current_trajectory[aid]['log_probs'].append(log_prob)
        self.current_trajectory[aid]['mfs'].append(mf)

    def end_episode(self, agent_id, mf, mask, global_state, reward):
        """Gọi khi hết 1 service trong time slot"""
        self.current_trajectory[int(agent_id)]['mf'] = mf
        self.current_trajectory[int(agent_id)]['mask'] = mask
        self.current_trajectory[int(agent_id)]['global_state'] = global_state
        self.current_trajectory[int(agent_id)]['reward'] = reward

    def finalize_episode(self):
        """Gọi khi KẾT THÚC time slot, đẩy dữ liệu vào list completed"""
        # Chỉ lưu nếu time slot đó thực sự có task đi qua
        has_data = any(len(v.get('states', [])) > 0 for v in self.current_trajectory.values())
        if has_data:
            self.completed_trajectories.append(self.current_trajectory)
        self.reset_current()

    def get_batch(self, batch_size):
        """Lấy ngẫu nhiên batch_size trajectories nguyên vẹn (ko cắt xén chuỗi)"""
        if len(self.completed_trajectories) < batch_size:
            return None
        return random.sample(self.completed_trajectories, batch_size)

    def clear(self):
        """Xóa toàn bộ sau khi học xong (On-policy characteristic)"""
        self.completed_trajectories = []
        self.reset_current()

    def __len__(self):
        return len(self.completed_trajectories)

# ==========================================
# 5. PPO AGENT CHÍNH
# ==========================================
class SequentialGRU_PPOAgent:
    def __init__(self, agent_id, node_type, actor_state_dim, critic_global_dim,
                 mf_action_dim, mf_hidden_sizes, mf_lr, action_dim= 32,
                 hidden_dim=128, critic_hidden=(256, 128), lr=3e-4, 
                 clip_eps=0.2, k_epochs=5, entropy_coef=0.01, 
                 target_entropy_ratio=0.8, total_train_steps=1000,
                 num_instances=1, device=None):

        self.agent_id = agent_id
        self.node_type = node_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_instances = num_instances
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.learn_step_counter = 0
        self.total_train_steps = total_train_steps

        # Networks
        self.actor = SequentialGRUActor(actor_state_dim, mf_action_dim, action_dim, hidden_dim, num_instances).to(self.device)
        self.critic = GlobalAggregatedCritic(critic_global_dim, critic_hidden, num_instances).to(self.device)
        self.mf_net = MFNetwork(actor_state_dim + mf_action_dim, mf_action_dim, mf_hidden_sizes, num_instances).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.mf_optimizer = optim.Adam(self.mf_net.parameters(), lr=mf_lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = SequentialMultiAgentBuffer(num_instances)
        
        # Entropy Auto-tuning
        self.log_alpha = torch.tensor([np.log(entropy_coef)], requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy_ratio = target_entropy_ratio
        self.entropy_coef = entropy_coef

    def choose_action(self, state, mf, hidden, mask, agent_idx=0):
        """Dùng trong môi trường (Data Collection) - Single Instance"""
        res = self.choose_action_batch(
            torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mf, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(hidden, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(mask, device=self.device, dtype=torch.float32).unsqueeze(0),
            torch.tensor([agent_idx], device=self.device)
        )
        return res[0][0].item(), res[1][0].item(), res[2][0].cpu().numpy()

    def choose_action_batch(self, states, mfs, hiddens, masks, agent_indices):
        """Dùng trong môi trường (Data Collection) - Phiên bản Vectorized"""
        # states: (B, DIM)
        # mfs: (B, MF_DIM)
        # hiddens: (B, HIDDEN_DIM)
        # masks: (B, ACTION_DIM)
        # agent_indices: (B,)
        
        with torch.no_grad():
            logits, new_hiddens = self.actor.forward_step(states, mfs, hiddens, agent_indices)
            logits = logits.masked_fill(masks == 0, -1e9)
            
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return actions, log_probs, new_hiddens

    def learn_mf(self, state, prev_mf, ground_truth_mf, agent_ids):
        """Học Mean Field (Supervised)"""
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        pmf = torch.as_tensor(prev_mf, device=self.device, dtype=torch.float32).unsqueeze(0)
        gt_mf = torch.as_tensor(ground_truth_mf, device=self.device, dtype=torch.float32).unsqueeze(0)
        idx = torch.tensor([agent_ids], device=self.device)

        pred_mf = self.mf_net(torch.cat([s, pmf], dim=-1), indices=idx)
        loss = self.loss_fn(pred_mf, gt_mf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        self.mf_optimizer.step()

    def save(self, checkpoint_path):
        directory = os.path.dirname(checkpoint_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'mf_net_state_dict': self.mf_net.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'mf_optimizer_state_dict': self.mf_optimizer.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.mf_net.load_state_dict(checkpoint['mf_net_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.mf_optimizer.load_state_dict(checkpoint['mf_optimizer_state_dict'])

    def learn(self, batch_size=128, k_epochs=4):
        """Học PPO với BPTT qua chuỗi trên 1 batch dữ liệu"""
        data_list = self.memory.get_batch(batch_size)
        if data_list is None: return None  # Chưa đủ 256 trajectories

        epoch_v_loss = 0

        # Lặp lại K epochs trên CÙNG 1 bộ dữ liệu (Đặc trưng của PPO)
        for _ in range(k_epochs):
            for trajectory_data in data_list:
                # trajectory_data là 1 dict chứa dữ liệu của tất cả các service trong 1 time slot
                # VD: {0: {'states':..., 'reward':...}, 1: {...}, ...}

                for aid, data in trajectory_data.items():
                    if len(data.get('states', [])) == 0:
                        continue  # Bỏ qua service không có task ở slot này

                    idx = torch.tensor([aid], device=self.device)
                    reward_tensor = torch.tensor([data['reward']], dtype=torch.float32, device=self.device)

                    # 1. TÍNH ADVANTAGE (Không gradient)
                    with torch.no_grad():
                        V_final_no_grad = self.critic(data['global_state'].unsqueeze(0).to(self.device), indices=idx)
                        advantages = reward_tensor - V_final_no_grad
                        if advantages.shape[0] > 1:
                            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # 2. Lấy hằng số
                    mf_const = data['mf'].to(self.device).unsqueeze(0)
                    mask_const = data['mask'].to(self.device).unsqueeze(0)

                    # 3. BPTT UPDATE CHO ACTOR
                    h_gru = torch.zeros(1, self.actor.gru_cell.hidden_size, device=self.device)
                    total_actor_loss = 0
                    total_entropy = 0
                    seq_len = len(data['states'])

                    for k in range(seq_len):
                        state_k = data['states'][k].to(self.device).unsqueeze(0)
                        
                        # Bọc action bằng torch.tensor (int)
                        action_k = torch.as_tensor(data['actions'][k], device=self.device, dtype=torch.long).unsqueeze(0)
                        
                        # Đổi 'old_log_probs' thành 'log_probs' và bọc bằng tensor (float)
                        old_log_prob_k = torch.as_tensor(data['log_probs'][k], device=self.device, dtype=torch.float32).unsqueeze(0)

                        mf_k = data['mfs'][k].to(self.device).unsqueeze(0)
                        
                        log_prob, entropy, h_gru = self.actor.evaluate(
                            state_k, mf_k, h_gru, action_k, masks=mask_const, indices=idx
                        )

                        ratio = torch.exp(log_prob - old_log_prob_k)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                        actor_loss = -torch.min(surr1, surr2).mean()

                        total_actor_loss += actor_loss
                        total_entropy += entropy.mean()

                    total_actor_loss /= seq_len
                    total_entropy /= seq_len

                    # Backprop 1 lần cho cả chuỗi
                    self.optimizer_actor.zero_grad()
                    (total_actor_loss - self.entropy_coef * total_entropy).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                    self.optimizer_actor.step()

                    # 4. CẬP NHẬT CRITIC
                    V_pred = self.critic(data['global_state'].unsqueeze(0).to(self.device), indices=idx)
                    critic_loss = F.mse_loss(V_pred, reward_tensor)

                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                    self.optimizer_critic.step()

                    epoch_v_loss += critic_loss.item()

        # Không xóa memory ở đây, Strategy sẽ gọi hàm clear() sau khi train xong
        return epoch_v_loss / (len(data_list) * k_epochs) if data_list else 0