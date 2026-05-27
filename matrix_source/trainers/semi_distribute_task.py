import torch
from matrix_source.agents.ppo import PPOAgent
# IMPORT AGENT MỚI Thay cho PPOSCAFFOLDREPAgent cũ
from matrix_source.agents.semi_distributed import SequentialGRU_PPOAgent

from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.trainers.train import log_transform
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime


def compute_gae(rewards, next_values, values, dones, agent_ids, gamma, lmbda):
    """ (GIỮ NGUYÊN) Generalized Advantage Estimation (GAE) """
    device = rewards.device
    num_steps = rewards.size(0)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(deltas)
    masks = (1 - dones) * (gamma * lmbda)
    boundary_mask = torch.ones(num_steps, device=device)
    if num_steps > 1:
        boundary_mask[:-1] = (agent_ids[:-1] == agent_ids[1:]).float()
    combined_mask = masks * boundary_mask
    curr_advantage = 0
    for t in reversed(range(num_steps)):
        curr_advantage = deltas[t] + curr_advantage * (combined_mask[t] if t < num_steps - 1 else 0)
        advantages[t] = curr_advantage
    return advantages


class PPOSCAFFOLDREPStrategy(AlgorithmStrategy):
    def __init__(self):
        super().__init__()
        self.lower_train_num = 0
        self.upper_train_num = 0
        self.alt_train_num = 0
        self.alt_next = 'UPPER'
        self.upper_mf_ema = None
        self.mf_ema_alpha = 0.7

        self.lower_cfg = {'min_size': 4096, 'batch': 128, 'epochs': 7}
        self.upper_cfg = {'min_size': 512, 'batch': 64, 'epochs': 5}

        self.upper_warmup_steps = 5  
        self.lower_warmup_steps = 15  
        self.max_cycles = 20

        self.phase = 'LOWER_ONLY'
        self.cycle_num = 1
        self.current_phase_updates = 0
        self.entropy_decay_rate = 0.99
        self.is_evaluating = False
        self.lower_collect_size= 256
        self.lower_batch_size= 128
        self.lower_train_epochs= 4

    def initialize_agents(self, trainer):
        # ==========================================
        # 1. UPPER AGENT (GIỮ NGUYÊN 100%)
        # ==========================================
        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_cfg['min_size'],
            target_entropy_ratio=0.8, target_entropy_end_ratio=0.05, total_train_steps=30,
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            lam=trainer.config.hyper_neural.get('LAMBDA', 0.95),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.upper_cfg['epochs'], batch_size=self.upper_cfg['batch'],
            num_instances=trainer.num_edge_agents, device=trainer.device
        )

        # ==========================================
        # 2. LOWER AGENT (THAY ĐỔI THÀNH SEQUENTIAL GRU)
        # ==========================================
        # Instance ID của Lower giờ là SERVICE ID (Mỗi service có 1 GRU riêng)
        lower_hidden_dim = trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'][0]
        
        # Tính toán chiều của Global State cho Critic mới
        # Gồm: Mask(num_nodes) + Q_final(num_nodes) + F(num_nodes) + Workload_sent(num_nodes) + MF(mf_dim)
        lower_mf_dim = trainer.num_nodes + trainer.max_models
        critic_global_dim = (trainer.num_nodes * 4) + lower_mf_dim

        trainer.shared_lower_agent = SequentialGRU_PPOAgent(
            agent_id=-1, node_type="Service_Sequential",
            actor_state_dim=trainer.lower_state_dim, # P_i, Q_s, F_s, MF
            critic_global_dim=critic_global_dim,     # Trạng thái tổng hợp cuối timeslot
            mf_action_dim=lower_mf_dim,              # Chỉ số Mean Field
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            hidden_dim=lower_hidden_dim,              # 128
            critic_hidden=(256, 128),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            entropy_coef=0.01,
            target_entropy_ratio=0.5, target_entropy_end_ratio=0.01, total_train_steps=60,
            num_instances=trainer.num_services,       # ĐỔI THÀNH SỐ SERVICE
            device=trainer.device
        )

        if self.phase == 'LOWER_ONLY' and self.lower_warmup_steps == 0:
            self.phase = 'UPPER_ONLY'

    # ==========================================
    # UPPER LEVEL FUNCTIONS (GIỮ NGUYÊN TOÀN BỘ)
    # ==========================================
    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        if self.upper_mf_ema is None:
            self.upper_mf_ema = mf_global.clone()
        else:
            self.upper_mf_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * mf_global

        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = self.upper_mf_ema[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
        is_det = self.is_evaluating

        batch_a_ids, log_probs, values = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, agent_indices=instance_indices, deterministic=is_det
        )
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix, log_probs, values

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, log_probs, values, is_done):
        reward = next_res['reward_global']
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        avg_mf_loss = 0.0
        is_frozen = (self.phase == 'LOWER_ONLY')
        edge_states = s_all[trainer.edge_node_ids] if s_all is not None else None

        if not is_frozen and not self.is_evaluating:
            edge_next_states = ns_all[trainer.edge_node_ids]
            dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32, device=trainer.device)
            next_raw_mf = next_res['mean_fields']
            if self.upper_mf_ema is None: self.upper_mf_ema = next_raw_mf
            next_ema = (1 - self.mf_ema_alpha) * self.upper_mf_ema + self.mf_ema_alpha * next_raw_mf
            edge_c_mfs = self.upper_mf_ema[trainer.edge_node_ids]
            edge_n_mfs = next_ema[trainer.edge_node_ids]
            edge_acts = acts_matrix[trainer.edge_node_ids]
            pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
            edge_a_ids = (edge_acts * pw2).sum(dim=1).long()
            rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
            instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
            avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
                edge_states, edge_c_mfs, next_raw_mf[trainer.edge_node_ids], edge_a_ids, rewards, edge_next_states,
                dones, agent_ids=instance_indices, log_prob=log_probs, value=values
            )
        agg_state = edge_states[0] if (edge_states is not None and len(edge_states) > 0) else None
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=agg_state)
        if self.is_evaluating:
            return {'reward': next_res['reward_global'], 'backlog': next_res['obs']['backlog'].sum().item(), 'energy': next_res['info'].get('energy', 0.0)}
        return None

    # ==========================================
    # LOWER LEVEL HELPER FUNCTIONS (THAY ĐỔI CHO SEQUENTIAL)
    # ==========================================
    def _get_service_mask(self, trainer, s_idx):
        """Lấy mask cho 1 service cụ thể"""
        placement = trainer.env.engine.placement_matrix[:, s_idx].T # (1, num_nodes)
        return placement.squeeze(0)

    def _build_simulated_state(self, trainer, task_req, sim_backlog_s, sim_capacity_s, mf):
        """Xây dựng state cho GRU từ dữ liệu mô phỏng (Qs đã bị thay đổi)"""
        # Chuẩn hóa tương tự build_lower_state cũ
        st = torch.cat([task_req, sim_backlog_s, sim_capacity_s], dim=-1).unsqueeze(0)
        st = st.clone()
        st[:, 0] /= trainer.config.norm_data_size
        st[:, 1] /= 100.0
        if st.shape[1] > 4:
            st[:, 4:4 + 2 * trainer.num_nodes] /= trainer.config.norm_gflop
        return st

    def _build_critic_global_state(self, trainer, mask_s, q_final_s, f_s, workload_sent_s, mf_s):
        """Xây dựng Global State cho Critic cuối timeslot của 1 Service"""
        return torch.cat([mask_s, q_final_s, f_s, workload_sent_s, mf_s])

    # ==========================================
    # MAIN TRAINING LOOP (THAY ĐỔI CORE LOGIC LOWER)
    # ==========================================
    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        ep = 0
        pbar = tqdm(total=self.max_cycles, desc="Sequential GRU Progress")
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            for slot in range(max_slots):
                # 1. UPPER ACTIONS (Giữ nguyên)
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix, u_log_probs, u_values = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                
                if len(t_idx) > 0:
                    # ==========================================
                    # A. SEQUENTIAL DRY-RUN (MO PHONG GRU)
                    # ==========================================
                    trainer.shared_lower_agent.memory.start_episode()
                    
                    # Tạo bản sao Backlog toàn cục để mô phỏng việc cập nhật Q_s
                    sim_backlog = obs['lower']['backlog'].clone()
                    sim_capacity = (obs['lower']['cpu_alloc'] * trainer.env.engine.placement_matrix).clone()
                    
                    # Mảng lưu action thật để đẩy xuống env
                    final_n_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
                    final_m_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)

                    # Duyệt theo từng SERVICE (Mỗi service là 1 GRU instance)
                    unique_services = torch.unique(s_idx)
                    
                    for s_id in unique_services:
                        # Lấy index của các task thuộc service hiện tại
                        task_mask = (s_idx == s_id)
                        curr_t_idx = t_idx[task_mask]
                        curr_deadlines = task_deadlines[task_mask]
                        curr_task_reqs = obs['lower']['task_reqs'][curr_t_idx]
                        
                        # SẮP XẾP THAM LAM THEO DEADLINE
                        sorted_order = torch.argsort(curr_deadlines)
                        curr_t_idx = curr_t_idx[sorted_order]
                        curr_deadlines = curr_deadlines[sorted_order]
                        curr_task_reqs = curr_task_reqs[sorted_order]
                        
                        # Lấy trạng thái cố định của service này
                        mask_s = self._get_service_mask(trainer, s_id)
                        q_s_sim = sim_backlog[:, s_id].clone() # Q_s của service này
                        f_s = sim_capacity[:, s_id] * mask_s    # F_s của service này
                        mf_s = torch.zeros(lower_mf_dim, device=trainer.device) # MF tạm thời
                        
                        # Khởi tạo Hidden State cho GRU của Service này
                        h_gru = np.zeros(trainer.shared_lower_agent.actor.gru_cell.hidden_size)
                        workload_sent_s = torch.zeros(trainer.num_nodes, device=trainer.device)
                        
                        # Duyệt tuần tự từng task
                        for i in range(len(curr_t_idx)):
                            state_k = self._build_simulated_state(trainer, curr_task_reqs[i], q_s_sim, f_s, mf_s)
                            
                            # GRU chọn hành động
                            a_id, log_prob, h_gru = trainer.shared_lower_agent.choose_action(
                                state_k, mf_s, h_gru, mask_s, agent_idx=int(s_id.item())
                            )
                            
                            n_k = a_id // trainer.max_models
                            m_k = a_id % trainer.max_models
                            
                            # Lưu vào Buffer
                            trainer.shared_lower_agent.memory.add_step(
                                int(s_id.item()), state_k.squeeze(0), h_gru, a_id, log_prob
                            )
                            
                            # CẬP NHẬT Q_s NGAY LẬP TỨC (TRỌNG TÂM THUẬT TOÁN)
                            load_k = curr_task_reqs[i, 0].item() / trainer.config.norm_gflop # Giả sử cột 0 là dung lượng
                            q_s_sim[n_k] += load_k
                            workload_sent_s[n_k] += load_k
                            
                            # Lưu lại hành động đúng vị trí ban đầu (trước khi sort)
                            orig_idx = curr_t_idx[i]
                            final_n_idx[orig_idx] = n_k
                            final_m_idx[orig_idx] = m_k

                        # Kết thúc service: Tính Global State cho Critic và lưu vào buffer
                        global_state_s = self._build_critic_global_state(
                            trainer, mask_s, q_s_sim, f_s, workload_sent_s, mf_s
                        )
                        # Lưu tạm thời reward = 0, sẽ update bằng reward thật sau khi step env
                        trainer.shared_lower_agent.memory.end_episode(
                            int(s_id.item()), mf_s, mask_s, global_state_s, reward=0.0
                        )

                    # ==========================================
                    # B. THỰC SỰ TƯƠNG TÁC VỚI MÔI TRƯỜNG
                    # ==========================================
                    results = trainer.env.step_lower(
                        t_idx, s_idx, batch_sizes, final_n_idx, final_m_idx, 
                        task_deadlines, tasks_min_accuracy
                    )
                    true_reward = results['reward']
                    norm_rew = log_transform(true_reward / (trainer.config.norm_lower_rw if trainer.config.norm_lower_rw != 0 else 1.0))

                    # ==========================================
                    # C. GÁN REWARD THẬT VÀ TRAIN LOWER
                    # ==========================================
                    if self.phase == 'LOWER_ONLY' and not self.is_evaluating:
                        # Gán reward thật cho tất cả các service vừa xử lý
                        for s_id in unique_services:
                            trainer.shared_lower_agent.memory.rewards[int(s_id.item())] = norm_rew
                        
                        # Gọi Learn (BPTT sẽ chạy ở đây)
                        trainer.shared_lower_agent.memory.finalize_episode()
                        if len(trainer.shared_lower_agent.memory) >= self.lower_collect_size:

                            # Train PPO: Lấy 128 trajectories, lặp 4 lần
                            loss = trainer.shared_lower_agent.learn(
                                batch_size=self.lower_batch_size,
                                k_epochs=self.lower_train_epochs
                            )

                            if loss is not None:
                                self.lower_train_num += 1
                                self.current_phase_updates += 1
                                trainer.aggregator.record_td_losses(lower_losses=loss)

                                # Checkpoint
                                if self.lower_train_num % 10 == 0:
                                    trainer.shared_lower_agent.save(
                                        f'checkpoints/ppo_lower_seq_{self.lower_train_num}.pth')

                                if self.current_phase_updates >= self.lower_warmup_steps:
                                    self.phase = 'UPPER_ONLY'
                                    self.current_phase_updates = 0
                                    print(f"\n[Cycle {self.cycle_num}] LOWER Phase Complete.")

                            # XÓA BUFFER SAU KHI HỌC XONG (RẤT QUAN TRỌNG ĐỂ PPO ON-POLICY)
                            trainer.shared_lower_agent.memory.clear()

                    trainer.aggregator.add_lower(results, mf_loss=0.0, state=None)
                else:
                    trainer.env.time_manager.tick()

                # 2. UPPER LEVEL METRICS & TRAINING (GIỮ NGUYÊN)
                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    is_ep_done = (slot == max_slots - 1)
                    self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper,
                                                 u_acts_matrix, u_log_probs, u_values, is_ep_done)

                    if self.phase == 'UPPER_ONLY':
                        loss = trainer.shared_upper_agent.learn(torch.arange(trainer.num_edge_agents, device=trainer.device))
                        if loss is not None:
                            self.upper_train_num += 1
                            self.current_phase_updates += 1
                            if self.current_phase_updates >= self.upper_warmup_steps:
                                self.phase = 'LOWER_ONLY'
                                self.current_phase_updates = 0
                                pbar.update(1)
                                self.cycle_num += 1
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            trainer.aggregator.store_history()
            ep += 1
        pbar.close()
        self.run_evaluation(trainer, num_episodes=5)

    # ==========================================
    # EVALUATION LOOP (ĐÃ ĐIỀU CHỈNH CHO SEQUENTIAL)
    # ==========================================
    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Starting Post-Training Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps
        lower_mf_dim = trainer.num_nodes + trainer.max_models

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper, init_lower_obs = res['upper'], res['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            ep_reward = 0

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix, _, _ = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    # Tương tự như Training, nhưng không lưu buffer và không tính gradient
                    sim_backlog = init_lower_obs['backlog'].clone()
                    sim_capacity = (init_lower_obs['cpu_alloc'] * trainer.env.engine.placement_matrix).clone()
                    final_n_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
                    final_m_idx = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)

                    unique_services = torch.unique(s_idx)
                    for s_id in unique_services:
                        task_mask = (s_idx == s_id)
                        curr_t_idx = t_idx[task_mask]
                        curr_deadlines = task_deadlines[task_mask]
                        curr_task_reqs = init_lower_obs['task_reqs'][curr_t_idx]
                        
                        sorted_order = torch.argsort(curr_deadlines)
                        curr_t_idx = curr_t_idx[sorted_order]
                        curr_deadlines = curr_deadlines[sorted_order]
                        curr_task_reqs = curr_task_reqs[sorted_order]
                        
                        mask_s = self._get_service_mask(trainer, s_id)
                        q_s_sim = sim_backlog[:, s_id].clone()
                        f_s = sim_capacity[:, s_id] * mask_s
                        mf_s = torch.zeros(lower_mf_dim, device=trainer.device)
                        h_gru = np.zeros(trainer.shared_lower_agent.actor.gru_cell.hidden_size)
                        
                        for i in range(len(curr_t_idx)):
                            state_k = self._build_simulated_state(trainer, curr_task_reqs[i], q_s_sim, f_s, mf_s)
                            a_id, _, h_gru = trainer.shared_lower_agent.choose_action(
                                state_k, mf_s, h_gru, mask_s, agent_idx=int(s_id.item())
                            )
                            n_k = a_id // trainer.max_models
                            m_k = a_id % trainer.max_models
                            load_k = curr_task_reqs[i, 0].item() / trainer.config.norm_gflop
                            q_s_sim[n_k] += load_k
                            orig_idx = curr_t_idx[i]
                            final_n_idx[orig_idx] = n_k
                            final_m_idx[orig_idx] = m_k

                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, final_n_idx, final_m_idx, task_deadlines, tasks_min_accuracy)
                    init_lower_obs = results # Cập nhật obseravtion
                    ep_reward += results['reward_global']
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    obs_upper = res_upper
                    current_upper_state = self.build_upper_state(trainer, res_upper)

            print(f"Eval Episode {ep + 1}: Total Reward={ep_reward:.2f}")

        self.is_evaluating = False