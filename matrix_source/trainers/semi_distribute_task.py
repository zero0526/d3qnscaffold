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
        self.lower_mf_prev = None
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
        self.model_workloads= None

    def initialize_agents(self, trainer):
        # ==========================================
        # 1. UPPER AGENT (GIỮ NGUYÊN 100%)
        # ==========================================
        self.model_workloads = trainer.env.metadata["model_workloads"]
        trainer.shared_upper_agent = PPOAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=self.upper_cfg['min_size'],
             total_train_steps=30,
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
        lower_mf_dim = trainer.num_nodes + trainer.max_models
        critic_global_dim = (trainer.num_nodes * 4 + lower_mf_dim)

        # Nhận state (P_i, Q_s, F_s, Workload_dist) và Mean Field
        actor_state_dim = trainer.lower_state_dim + trainer.num_nodes
        trainer.shared_lower_agent = SequentialGRU_PPOAgent(
            agent_id=-1, node_type="Service_Sequential",
            actor_state_dim=actor_state_dim, # P_i, Q_s, F_s, Workload_dist, MF
            critic_global_dim=critic_global_dim,     # Trạng thái tổng hợp cuối timeslot
            mf_action_dim=lower_mf_dim,              # Chỉ số Mean Field
            action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            hidden_dim=lower_hidden_dim,              # 128
            critic_hidden=(256, 128),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            clip_eps=trainer.config.hyper_neural.get('CLIP_EPS', 0.2),
            k_epochs=self.lower_cfg['epochs'],
            entropy_coef=0.01,
            target_entropy_ratio=0.5,  total_train_steps=60,
            num_instances=trainer.num_services,       # ĐỔI THÀNH SỐ SERVICE
            device=trainer.device
        )

        if self.lower_mf_prev is None:
            self.lower_mf_prev = torch.zeros(trainer.num_services, trainer.num_nodes, lower_mf_dim, device=trainer.device)

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
        return trainer.env.engine.placement_matrix[:, s_idx].float()

    def _build_simulated_state_batch(self, trainer, task_reqs, sim_backlog_batch, sim_capacity_batch, workload_sent_batch):
        """Xây dựng state cho GRU - Phiên bản Vectorized"""
        st = torch.cat([task_reqs, sim_backlog_batch, sim_capacity_batch, workload_sent_batch], dim=-1)
        st[:, 0] /= trainer.config.norm_data_size
        st[:, 1] /= 100.0
        if st.shape[1] > 4:
            st[:, 4:] /= trainer.config.norm_gflop
        return st

    def _build_critic_global_state(self, trainer, mask_s, q_final_s, f_s, workload_sent_s, mean_mf_s):
        """Xây dựng Global State cho Critic cuối timeslot của 1 Service"""
        return torch.cat([mask_s, q_final_s, f_s, workload_sent_s, mean_mf_s])

    def _compute_node_wise_mfs(self, trainer, t_idx, s_idx, n_idx, m_idx, src_node_indices):
        """
        Tính toán Mean Field cho từng service, loại trừ nhóm task đến từ node đang xét.
        """
        B = len(t_idx)
        mf_dim = trainer.num_nodes + trainer.max_models
        device = trainer.device
        
        two_hot = torch.zeros(B, mf_dim, device=device)
        two_hot[torch.arange(B), n_idx.long()] = 1.0
        two_hot[torch.arange(B), trainer.num_nodes + m_idx.long()] = 1.0
        
        num_services = trainer.num_services
        num_nodes = trainer.num_nodes
        
        group_sums_sn = torch.zeros(num_services, num_nodes, mf_dim, device=device)
        group_counts_sn = torch.zeros(num_services, num_nodes, device=device)
        
        service_node_flat_idx = s_idx * num_nodes + src_node_indices
        group_sums_sn.view(-1, mf_dim).index_add_(0, service_node_flat_idx.long(), two_hot)
        group_counts_sn.view(-1).index_add_(0, service_node_flat_idx.long(), torch.ones(B, device=device))
        
        group_sums_s = group_sums_sn.sum(dim=1) 
        group_counts_s = group_counts_sn.sum(dim=1) 
        
        denom = (group_counts_s.unsqueeze(1) - group_counts_sn).clamp(min=1)
        node_wise_mfs = (group_sums_s.unsqueeze(1) - group_sums_sn) / denom.unsqueeze(2)
        
        node_wise_mfs = torch.where(
            (group_counts_s.unsqueeze(1) > group_counts_sn).unsqueeze(2),
            node_wise_mfs,
            torch.zeros_like(node_wise_mfs)
        )
        
        return node_wise_mfs 

    # ==========================================
    # MAIN TRAINING LOOP (THAY ĐỔI CORE LOGIC LOWER)
    # ==========================================
    def run_training(self, trainer):
        max_slots = trainer.env.time_manager.max_steps
        ep = 0
        pbar = tqdm(total=self.max_cycles, desc="Sequential GRU Progress")

        while self.cycle_num <= self.max_cycles:
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            obs_lower = obs['lower']
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix, u_log_probs, u_values = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                
                if len(t_idx) > 0:
                    trainer.shared_lower_agent.memory.start_episode()
                    
                    sim_backlog = obs_lower["obs"]['backlog'].clone()
                    sim_capacity = (obs_lower["obs"]['cpu_alloc'] * trainer.env.engine.placement_matrix).clone()
                    
                    unique_services = torch.unique(s_idx)
                    service_h_gru = {int(s): np.zeros(trainer.shared_lower_agent.actor.gru_cell.hidden_size) for s in unique_services}
                    service_workload_sent = {int(s): torch.zeros(trainer.num_nodes, device=trainer.device) for s in unique_services}

                    src_node_indices_all = torch.argmax(trainer.env.engine.terminal_to_node_map[t_idx], dim=1)
                    unique_src_nodes = torch.unique(src_node_indices_all)
                    node_priority = torch.zeros(trainer.num_nodes, dtype=torch.long, device=trainer.device)
                    node_priority[unique_src_nodes] = torch.arange(len(unique_src_nodes), device=trainer.device)
                    
                    sort_key = s_idx.long() * 1000000 + node_priority[src_node_indices_all] * 1000 + torch.argsort(torch.argsort(task_deadlines))
                    sorted_indices = torch.argsort(sort_key)
                    
                    t_idx_sorted = t_idx[sorted_indices]
                    s_idx_sorted = s_idx[sorted_indices]
                    batch_sizes_sorted = batch_sizes[sorted_indices]
                    task_reqs_sorted = obs_lower["obs"]['task_reqs'][t_idx_sorted]
                    src_node_sorted = src_node_indices_all[sorted_indices]
                    
                    unique_serv_sorted, counts = torch.unique(s_idx_sorted, return_counts=True)
                    num_active_serv = len(unique_serv_sorted)
                    max_seq_len = counts.max().item()
                    offsets = torch.zeros(num_active_serv + 1, dtype=torch.long, device=trainer.device)
                    offsets[1:] = torch.cumsum(counts, dim=0)

                    h_gru_batch = torch.stack([torch.as_tensor(service_h_gru[int(s)], device=trainer.device, dtype=torch.float32) for s in unique_serv_sorted])
                    workload_sent_batch = torch.stack([service_workload_sent[int(s)] for s in unique_serv_sorted])
                    mask_s_node_batch = torch.stack([self._get_service_mask(trainer, int(s)) for s in unique_serv_sorted])
                    mask_s_action_batch = mask_s_node_batch.repeat_interleave(trainer.max_models, dim=1)
                    
                    # --- Filter out services with no placement (all-zero mask) ---
                    valid_service_mask = mask_s_node_batch.sum(dim=1) > 0  # (num_active_serv,)
                    if not valid_service_mask.all():
                        # Mark tasks for unplaced services as failed and skip them
                        unplaced_svc_ids = unique_serv_sorted[~valid_service_mask]
                        unplaced_task_mask = torch.isin(s_idx_sorted, unplaced_svc_ids)
                        # Keep only tasks for placed services
                        keep_mask = ~unplaced_task_mask
                        t_idx_sorted = t_idx_sorted[keep_mask]
                        s_idx_sorted = s_idx_sorted[keep_mask]
                        batch_sizes_sorted = batch_sizes_sorted[keep_mask]
                        task_reqs_sorted = task_reqs_sorted[keep_mask]
                        src_node_sorted = src_node_sorted[keep_mask]
                        # Map final_n/m_idx for kept positions
                        original_indices = sorted_indices[keep_mask]
                        
                        # Recompute unique_serv_sorted, counts, etc
                        h_gru_batch = h_gru_batch[valid_service_mask]
                        workload_sent_batch = workload_sent_batch[valid_service_mask]
                        mask_s_node_batch = mask_s_node_batch[valid_service_mask]
                        mask_s_action_batch = mask_s_action_batch[valid_service_mask]
                        unique_serv_sorted = unique_serv_sorted[valid_service_mask]
                        counts = torch.stack([( s_idx_sorted == s).sum() for s in unique_serv_sorted])
                        if counts.max().item() == 0 or len(t_idx_sorted) == 0:
                            trainer.env.time_manager.tick()
                            goto_next = True
                        else:
                            max_seq_len = counts.max().item()
                            offsets = torch.zeros(len(unique_serv_sorted) + 1, dtype=torch.long, device=trainer.device)
                            offsets[1:] = torch.cumsum(counts, dim=0)
                            final_n_idx = torch.zeros(len(t_idx_sorted), dtype=torch.long, device=trainer.device)
                            final_m_idx = torch.zeros(len(t_idx_sorted), dtype=torch.long, device=trainer.device)
                            goto_next = False
                    else:
                        original_indices = sorted_indices
                        goto_next = False
                        final_n_idx = torch.zeros(len(t_idx_sorted), dtype=torch.long, device=trainer.device)
                        final_m_idx = torch.zeros(len(t_idx_sorted), dtype=torch.long, device=trainer.device)
                    
                    if goto_next:
                        continue

                    for k in range(max_seq_len):
                        active_mask = (counts > k)
                        if not active_mask.any(): break
                        
                        curr_batch_idx = (offsets[:-1][active_mask] + k)
                        curr_s_ids = unique_serv_sorted[active_mask]
                        curr_node_src = src_node_sorted[curr_batch_idx]
                        
                        q_s_sim_batch = sim_backlog[:, curr_s_ids.long()].T
                        f_s_batch = sim_capacity[:, curr_s_ids.long()].T * mask_s_node_batch[active_mask]
                        
                        state_batch = self._build_simulated_state_batch(
                            trainer, task_reqs_sorted[curr_batch_idx], q_s_sim_batch, f_s_batch, workload_sent_batch[active_mask]
                        )
                        mf_input_batch = self.lower_mf_prev[curr_s_ids.long(), curr_node_src.long()]
                        
                        a_ids, log_probs, new_h_batch = trainer.shared_lower_agent.choose_action_batch(
                            state_batch, mf_input_batch, h_gru_batch[active_mask], 
                            mask_s_action_batch[active_mask], curr_s_ids.long()
                        )
                        
                        h_gru_batch[active_mask] = new_h_batch
                        n_k, m_k = a_ids // trainer.max_models, a_ids % trainer.max_models
                        
                        load_q_batch = task_reqs_sorted[curr_batch_idx, 0] / trainer.config.norm_gflop
                        workload_batch = torch.tensor([self.model_workloads[int(s)][m] for s, m in zip(curr_s_ids, m_k)], device=trainer.device, dtype=torch.float32)
                        workload_batch = workload_batch * batch_sizes_sorted[curr_batch_idx] / trainer.config.norm_gflop
                        
                        sim_backlog.index_put_((n_k, curr_s_ids.long()), load_q_batch, accumulate=True)
                        workload_sent_batch[active_mask, n_k] += workload_batch
                        
                        for i, sid in enumerate(curr_s_ids):
                            trainer.shared_lower_agent.memory.add_step(int(sid), state_batch[i], new_h_batch[i], a_ids[i], log_probs[i], mf_input_batch[i])
                        
                        final_n_idx[curr_batch_idx] = n_k
                        final_m_idx[curr_batch_idx] = m_k

                    for i, s in enumerate(unique_serv_sorted):
                        service_h_gru[int(s)] = h_gru_batch[i].cpu().numpy()
                        service_workload_sent[int(s)] = workload_sent_batch[i]

                    # Build full-size env arrays (scatter filtered results into original slots)
                    final_n_idx_env = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
                    final_m_idx_env = torch.zeros(len(t_idx), dtype=torch.long, device=trainer.device)
                    final_n_idx_env[original_indices] = final_n_idx
                    final_m_idx_env[original_indices] = final_m_idx

                    curr_slot_mfs = self._compute_node_wise_mfs(
                        trainer, t_idx_sorted, s_idx_sorted, final_n_idx, final_m_idx, src_node_sorted
                    )

                    for s_id in unique_serv_sorted:
                        sid_val = int(s_id.item())
                        mask_s_node = self._get_service_mask(trainer, sid_val)
                        mask_s_action = mask_s_node.repeat_interleave(trainer.max_models)
                        q_final_s = sim_backlog[:, sid_val]
                        f_s = sim_capacity[:, sid_val] * mask_s_node
                        workload_final_s = service_workload_sent[sid_val]
                        mean_mf_s = curr_slot_mfs[sid_val].mean(dim=0)
                        global_state_s = self._build_critic_global_state(
                            trainer, mask_s_node, q_final_s, f_s, workload_final_s, mean_mf_s
                        )
                        dummy_mf = torch.zeros(trainer.num_nodes + trainer.max_models, device=trainer.device)
                        trainer.shared_lower_agent.memory.end_episode(
                            sid_val, dummy_mf, mask_s_action, global_state_s, reward=0.0
                        )

                    self.lower_mf_prev = curr_slot_mfs.clone()

                    results = trainer.env.step_lower(
                        t_idx, s_idx, batch_sizes, final_n_idx_env, final_m_idx_env, 
                        task_deadlines, tasks_min_accuracy
                    )
                    obs_lower = results 
                    true_reward = results['reward']
                    true_reward -= results['obs']['virtual_drift']
                    norm_rew = log_transform(true_reward / (trainer.config.norm_lower_rw if trainer.config.norm_lower_rw != 0 else 1.0))

                    if self.phase == 'LOWER_ONLY' and not self.is_evaluating:
                        for s_id in unique_serv_sorted:
                            sid_val = int(s_id.item())
                            trainer.shared_lower_agent.memory.current_trajectory[sid_val]['reward'] = norm_rew
                        
                        trainer.shared_lower_agent.memory.finalize_episode()
                        if len(trainer.shared_lower_agent.memory) >= self.lower_collect_size:
                            loss = trainer.shared_lower_agent.learn(
                                batch_size=self.lower_batch_size,
                                k_epochs=self.lower_train_epochs
                            )

                            if loss is not None:
                                self.lower_train_num += 1
                                self.current_phase_updates += 1
                                trainer.aggregator.record_td_losses(lower_losses=loss)

                                if self.current_phase_updates >= self.lower_warmup_steps:
                                    self.phase = 'UPPER_ONLY'
                                    self.current_phase_updates = 0
                                    print(f"\n[Cycle {self.cycle_num}] LOWER Phase Complete.")

                            trainer.shared_lower_agent.memory.clear()

                    trainer.aggregator.add_lower(results, mf_loss=0.0, state=None)
                    
                    if slot % 1000 == 0 and slot > 0:
                        success = sum(trainer.aggregator.episode_success_qos)
                        fail = sum(trainer.aggregator.episode_violate_qos)
                        trainer.aggregator.log(f"  > Slot {slot:4d} | Partial OK: {success:5.0f} | FAIL: {fail:5.0f}")
                else:
                    trainer.env.time_manager.tick()

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
            trainer.aggregator.report_episode(ep)
            trainer.aggregator.reset_episode()
            ep += 1
        pbar.close()
        self.run_evaluation(trainer, num_episodes=5)

    def run_evaluation(self, trainer, num_episodes=5):
        print(f"\n--- Starting Post-Training Evaluation ({num_episodes} Episodes) ---")
        self.is_evaluating = True
        max_slots = trainer.env.time_manager.max_steps
        
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
                    sim_backlog = init_lower_obs["obs"]['backlog'].clone()
                    sim_capacity = (init_lower_obs["obs"]['cpu_alloc'] * trainer.env.engine.placement_matrix).clone()
                    
                    unique_services_at_slot = torch.unique(s_idx)
                    service_h_gru = {int(s): np.zeros(trainer.shared_lower_agent.actor.gru_cell.hidden_size) for s in unique_services_at_slot}
                    service_workload_sent = {int(s): torch.zeros(trainer.num_nodes, device=trainer.device) for s in unique_services_at_slot}

                    src_node_indices_all = torch.argmax(trainer.env.engine.terminal_to_node_map[t_idx], dim=1)
                    unique_src_nodes = torch.unique(src_node_indices_all)
                    node_priority = torch.zeros(trainer.num_nodes, dtype=torch.long, device=trainer.device)
                    node_priority[unique_src_nodes] = torch.arange(len(unique_src_nodes), device=trainer.device)
                    
                    sort_key = s_idx.long() * 1000000 + node_priority[src_node_indices_all] * 1000 + torch.argsort(torch.argsort(task_deadlines))
                    sorted_indices = torch.argsort(sort_key)
                    
                    t_idx_sorted = t_idx[sorted_indices]
                    s_idx_sorted = s_idx[sorted_indices]
                    batch_sizes_sorted = batch_sizes[sorted_indices]
                    task_reqs_sorted = init_lower_obs["obs"]['task_reqs'][t_idx_sorted]
                    src_node_sorted = src_node_indices_all[sorted_indices]
                    
                    unique_serv_sorted, counts = torch.unique(s_idx_sorted, return_counts=True)
                    num_active_serv = len(unique_serv_sorted)
                    max_seq_len = counts.max().item()
                    offsets = torch.zeros(num_active_serv + 1, dtype=torch.long, device=trainer.device)
                    offsets[1:] = torch.cumsum(counts, dim=0)

                    h_gru_batch = torch.stack([torch.as_tensor(service_h_gru[int(s)], device=trainer.device, dtype=torch.float32) for s in unique_serv_sorted])
                    workload_sent_batch = torch.stack([service_workload_sent[int(s)] for s in unique_serv_sorted])
                    mask_s_node_batch = torch.stack([self._get_service_mask(trainer, int(s)) for s in unique_serv_sorted])
                    mask_s_action_batch = mask_s_node_batch.repeat_interleave(trainer.max_models, dim=1)
                    
                    final_n_idx = torch.zeros_like(t_idx)
                    final_m_idx = torch.zeros_like(t_idx)

                    for k in range(max_seq_len):
                        active_mask = (counts > k)
                        if not active_mask.any(): break
                        
                        curr_batch_idx = (offsets[:-1][active_mask] + k)
                        curr_s_ids = unique_serv_sorted[active_mask]
                        curr_node_src = src_node_sorted[curr_batch_idx]
                        
                        q_s_sim_batch = sim_backlog[:, curr_s_ids.long()].T
                        f_s_batch = sim_capacity[:, curr_s_ids.long()].T * mask_s_node_batch[active_mask]
                        
                        state_batch = self._build_simulated_state_batch(
                            trainer, task_reqs_sorted[curr_batch_idx], q_s_sim_batch, f_s_batch, workload_sent_batch[active_mask]
                        )
                        mf_input_batch = self.lower_mf_prev[curr_s_ids.long(), curr_node_src.long()]
                        
                        a_ids, _, new_h_batch = trainer.shared_lower_agent.choose_action_batch(
                            state_batch, mf_input_batch, h_gru_batch[active_mask], 
                            mask_s_action_batch[active_mask], curr_s_ids.long()
                        )
                        
                        h_gru_batch[active_mask] = new_h_batch
                        n_k, m_k = a_ids // trainer.max_models, a_ids % trainer.max_models
                        
                        load_q_batch = task_reqs_sorted[curr_batch_idx, 0] / trainer.config.norm_gflop
                        workload_batch = torch.tensor([self.model_workloads[int(s)][m] for s, m in zip(curr_s_ids, m_k)], device=trainer.device, dtype=torch.float32)
                        workload_batch = workload_batch * batch_sizes_sorted[curr_batch_idx] / trainer.config.norm_gflop
                        
                        sim_backlog.index_put_((n_k, curr_s_ids.long()), load_q_batch, accumulate=True)
                        workload_sent_batch[active_mask, n_k] += workload_batch
                        
                        final_n_idx[curr_batch_idx] = n_k
                        final_m_idx[curr_batch_idx] = m_k

                    for i, s in enumerate(unique_serv_sorted):
                        service_h_gru[int(s)] = h_gru_batch[i].cpu().numpy()
                        service_workload_sent[int(s)] = workload_sent_batch[i]

                    # 5. Cập nhật Personalized MF cho Slot sau (Evaluation)
                    src_node_indices_eval = torch.argmax(trainer.env.engine.terminal_to_node_map[t_idx], dim=1)
                    curr_eval_mfs = self._compute_node_wise_mfs(
                        trainer, t_idx, s_idx, final_n_idx, final_m_idx, src_node_indices_eval
                    )
                    self.lower_mf_prev = curr_eval_mfs.detach()

                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, final_n_idx, final_m_idx, task_deadlines, tasks_min_accuracy)
                    init_lower_obs = results 
                    ep_reward += results['reward']
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    obs_upper = res_upper
                    current_upper_state = self.build_upper_state(trainer, res_upper)

            print(f"Eval Episode {ep + 1}: Total Reward={ep_reward:.2f}")

        self.is_evaluating = False