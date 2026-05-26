import torch
import os

from networkx.classes import neighbors
from termcolor import colored
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.agents.td3 import MFTD3Agent
from matrix_source.trainers.train import Trainer
from matrix_source.utils.math_utils import to_binary
from matrix_source.trainers.train import log_transform

class D3QNTD3(AlgorithmStrategy):

    def initialize_agents(self, trainer: Trainer):
        self.upper_state_dim = trainer.upper_state_dim
        self.upper_action_dim = trainer.upper_action_dim
        # omega + workload_s + backlog_s + cpu_allo_s + avg_hops
        self.lower_state_dim = 1 + 1 + trainer.num_nodes*4
        self.lower_action_dim = trainer.num_nodes
        mf_lower_dim= trainer.num_nodes

        # Standard keys from metadata/static_matrices
        self.model_accuracies = trainer.env.metadata["model_accuracies"]
        self.model_workloads = trainer.env.metadata["model_workloads"]
        self.service_deadlines = trainer.env.metadata["service_deadlines"]
        self.adj_matrix = trainer.env.static_matrices["adj_matrix"]
        self.hops= trainer.env.static_matrices["hops"]

        # list edge_id in comp_node_id
        self.edge_ids = trainer.env.static_matrices["edge_ids"]
        self.num_edges = len(self.edge_ids)
        self.edge_id_to_agent_idx = trainer.env.static_matrices["edge_id_to_agent_idx"]
        self.agent_id_to_edge_idx = {agent_id: edge_id for edge_id, agent_id in self.edge_id_to_agent_idx.items()}
        self.agent_adj_matrix = trainer.env.static_matrices["agent_adj_matrix"].to(trainer.device)
        # node_id to access
        self.distance_matrix = trainer.env.static_matrices["transmission_delay_matrix"]
        self.upper_agent = D3QNAgent(
            node_id=-2, node_type="Edge_Group",
            state_dim=trainer.upper_state_dim,
            action_dim=trainer.upper_action_dim,
            u_action_dim=trainer.upper_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=float(trainer.config.hyper_neural["BUFFER_MIN_SIZE"][0]),
            hidden_sizes=trainer.config.hyper_neural['AGENT_HIDDEN_LAYER'],
            lr=float(trainer.config.hyper_neural['UPPER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=trainer.config.hyper_neural['BATCH_SIZE'],
            num_instances=trainer.num_edge_agents,
            device=trainer.device,
            logs_q=True
        )

        self.lower_agent = MFTD3Agent(
            node_id="shared_lower",
            node_type="Offload_Group",
            num_comp_node=trainer.num_nodes,
            state_dim=self.lower_state_dim,
            action_dim=self.lower_action_dim,
            mf_dim=mf_lower_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            hidden_sizes=trainer.config.hyper_neural["AGENT_HIDDEN_LAYER"],
            actor_lr=float(trainer.config.hyper_neural["LOWER_LR"]),
            critic_lr=float(trainer.config.hyper_neural["LOWER_LR"]),
            gamma=trainer.config.hyper_neural["DISCOUNT_FACTOR"],
            tau=trainer.config.hyper_neural["UPDATE_TARGET_COEF"],
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            expl_noise=0.1,
            buffer_size=trainer.config.hyper_neural["MEMORY_SIZE"],
            batch_size=trainer.config.hyper_neural["BATCH_SIZE"],
            buffer_min_size=trainer.config.hyper_neural["BUFFER_MIN_SIZE"][1],
            num_instances=len(self.edge_ids),
            device=trainer.device,
            logs_q=True
        )
        # num_agent x num_node x num_service
        self.distributed_task = torch.zeros((len(self.edge_ids)*trainer.num_services, trainer.num_nodes), device=trainer.device)

    def build_upper_state(self, trainer, obs_upper):
        actions = obs_upper['actions'] # (N, S)
        phi = obs_upper['phi_prob']    # (N, S)
        return torch.cat([actions, phi], dim=-1)

    def get_upper_actions(self, trainer, current_upper_state, obs_upper, deterministic=False):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        sc_mfs = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        
        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = sc_mfs[trainer.edge_node_ids]
        instance_indices = torch.tensor(range(self.num_edges), device=trainer.device)
        zeta = trainer.zeta_upper if not deterministic else 100.0
        
        batch_a_ids = self.upper_agent.choose_action_batch(
            edge_states, edge_mfs,epsilon=trainer.eps_upper,  zeta=zeta, agent_indices=instance_indices
        )

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)

        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def _build_min_wl(self, trainer, task_agent_ids, s_idx, tasks_min_accuracy, data_sizes):
        num_groups = self.num_edges * trainer.num_services
        srv_acc_all = self.model_accuracies[s_idx]
        diffs = srv_acc_all - tasks_min_accuracy.view(-1, 1)
        diffs[diffs < 0] = float('inf')
        best_models = diffs.argmin(dim=1)

        min_workloads = self.model_workloads[s_idx, best_models] * data_sizes
        service_min_workloads = torch.zeros((num_groups,), device=trainer.device)

        combined_indices = task_agent_ids * trainer.num_services + s_idx
        service_min_workloads.index_add_(0, combined_indices, min_workloads)
        
        active_indices = torch.unique(combined_indices)

        return service_min_workloads, active_indices

    def _build_unified_state(self, trainer, res_lower, min_wl_flat, active_indices=None):
        """State thống nhất: Bao gồm cả Omega, Hops, Backlog, CPU và WORKLOAD CẦN CHIA"""
        obs_dict = res_lower['obs']
        meta = trainer.env.metadata
        num_groups = self.num_edges * trainer.num_services

        omegas = meta['service_omega'].squeeze(-1)
        full_omegas = omegas.unsqueeze(0).repeat(self.num_edges, 1).view(num_groups, 1)

        avg_hops = self.hops[self.edge_ids].unsqueeze(1).repeat(1, trainer.num_services, 1).view(num_groups, -1)

        s_backlogs = obs_dict['backlog']
        backlog_coll = s_backlogs.T.unsqueeze(0).expand(self.num_edges, -1, -1).reshape(num_groups, -1)

        s_cpus = obs_dict['cpu_alloc']
        masking = trainer.env.engine.placement_matrix
        cpu_allo = (s_cpus.T.unsqueeze(0).expand(self.num_edges, -1, -1).reshape(num_groups, -1)) * \
                   (masking.T.unsqueeze(0).expand(self.num_edges, -1, -1).reshape(num_groups, -1))

        # Thêm cột Workload cần chia
        wl_feat = (min_wl_flat / trainer.config.norm_gflop).unsqueeze(-1)
        external_snack = obs_dict['external_snack']
        external_snack_feat = (
            external_snack.T.unsqueeze(0)
            .expand(self.num_edges, -1, -1)
            .reshape(num_groups, -1)
        )
        # Concat: 2 + 4*N
        state_all = torch.cat([
            full_omegas,
            wl_feat,
            avg_hops,
            backlog_coll / trainer.config.norm_gflop,
            cpu_allo / trainer.config.norm_gflop,
            external_snack_feat/ trainer.config.norm_gflop,
        ], dim=-1)
        
        if active_indices is not None:
            return state_all[active_indices], state_all
        return state_all

    def get_lower_actions(self, trainer, current_state, active_indices, min_wl_flat, t_idx, s_idx, tasks_min_accuracy, task_deadlines,
                          batch_sizes, deterministic=False):
        active_agent_indices = active_indices // trainer.num_services
        
        # 1. Continuous Probabilities from TD3 Actor
        curr_masks = trainer.env.engine.placement_matrix.T.unsqueeze(0).expand(self.num_edges, -1, -1).reshape(-1, trainer.num_nodes)
        active_masks = curr_masks[active_indices]
        
        active_probs = self.lower_agent.choose_action_batch(
            current_state, self.distributed_task[active_indices], 
            agent_indices=active_agent_indices, action_masks=active_masks, deterministic=deterministic
        )
        active_probs = torch.as_tensor(active_probs, device=trainer.device, dtype=torch.float32)

        # ==================================================================
        # 2. CONTINUOUS PART (FOR RL LEARNING): PHANTOM WORKLOAD
        # ==================================================================
        # Công thức: Workload thực tế = Tổng yêu cầu * Xác suất chia sẻ
        # min_wl_flat[active_indices]: (num_active_groups,), active_probs: (num_active_groups, num_nodes)
        continuous_workloads = min_wl_flat[active_indices].unsqueeze(-1) * active_probs 
        
        # Grid to store workloads for the entire system
        actual_workloads_rl = torch.zeros((self.num_edges, trainer.num_services, trainer.num_nodes), device=trainer.device)
        # Map to correct (Edge, Service) positions
        actual_workloads_rl.view(-1, trainer.num_nodes)[active_indices] = continuous_workloads

        # ==================================================================
        # 3. DISCRETE PART (FOR ENVIRONMENT): DISCRETE ASSIGNMENT
        # ==================================================================
        # We replace Water-filling with simple Argmax based on probabilities.
        # (Node with highest probability receives the entire task load for that group)
        all_probs = torch.zeros((self.num_edges * trainer.num_services, trainer.num_nodes), device=trainer.device)
        all_probs[active_indices] = active_probs

        T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
        task_edge_ids = T_E_map[t_idx].argmax(dim=1)
        task_agent_ids = torch.tensor([self.edge_id_to_agent_idx.get(eid.item(), 0) for eid in task_edge_ids], device=trainer.device)

        # Get probabilities corresponding to each task
        placed_mask = trainer.env.engine.placement_matrix.T[s_idx]
        task_probs = all_probs[task_agent_ids * trainer.num_services + s_idx] # (num_tasks, num_nodes)
        safe_task_probs = task_probs * placed_mask.float()
        
        # Choose node with highest probability for Environment execution
        assigned_nodes = torch.argmax(safe_task_probs, dim=-1)
        assigned_models = torch.zeros_like(assigned_nodes) # Default to smallest model (0)

        # ==================================================================
        # 4. MEAN FIELD CALCULATION (Using continuous actual_workloads_rl)
        # ==================================================================
        sum_w = actual_workloads_rl.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        current_dist = actual_workloads_rl / sum_w
        neighbor_rate = self.agent_adj_matrix / self.agent_adj_matrix.sum(dim=-1, keepdim=True)
        curr_mf = torch.einsum("ij, jsn -> isn", neighbor_rate, current_dist).flatten(0, 1)
        self.distributed_task = curr_mf
        
        # Return: Discrete nodes for Env, Continuous workloads for Reward
        return assigned_nodes, assigned_models, task_agent_ids, active_probs, actual_workloads_rl, curr_mf

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        # res_upper keys: 'actions', 'phi_prob', 'mean_fields', 'resources'

        sc_mfs = current_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        sc_ns_mfs = next_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        reward = next_res['reward_global']

        edge_mfs = sc_mfs[trainer.edge_node_ids]
        edge_ns_mfs = sc_ns_mfs[trainer.edge_node_ids]

        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
        dones = torch.full((trainer.num_edge_agents,), float(is_done), device=trainer.device)
        a_ids = acts_matrix[trainer.edge_node_ids].sum(dim=-1).long() # Placeholder logic
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)

        loss = self.upper_agent.store_transition_train_mf_batch(
            s_all, edge_mfs, edge_ns_mfs, a_ids, rewards, ns_all, dones, instance_indices
        )
        return loss

    def run_training(self, trainer_obj: Trainer):
        """
        # collect 100 eps random for lower agent
        train 1000 eps for stable lower agent
        train upper agent from 1100 eps 
        """
        num_eps = trainer_obj.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = trainer_obj.env.time_manager.max_steps
        print(colored("="*50, "blue", attrs=["bold"]))
        print(colored(f"RB-SAC-CEN Training ({num_eps} frames)", "blue", attrs=["bold"]))
        print(colored("="*50, "blue", attrs=["bold"]))

        for ep in range(num_eps):
            res = trainer_obj.env.reset()
            prev_upper_res, prev_lower_res = res["upper"], res["lower"]
            prev_upper_state = self.build_upper_state(trainer_obj, prev_upper_res)
            prev_mf = torch.zeros((self.num_edges*trainer_obj.num_services, trainer_obj.num_nodes),
                                  device=trainer_obj.device)
            transition_cache = None
            for slot in range(max_slots):
                if trainer_obj.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer_obj, prev_upper_state, prev_upper_res)
                    trainer_obj.env.step_upper(u_acts_matrix)
                # Get tasks
                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines  = trainer_obj.workload_gen.generate_step()
                current_lower_res= None
                if len(t_idx) > 0:
                    is_done = (slot == max_slots - 1)
                    T_E_map = trainer_obj.env.static_matrices["terminal_to_comp_node_map"]
                    task_edge_ids = T_E_map[t_idx].argmax(dim=1)
                    task_agent_ids = torch.tensor(
                        [self.edge_id_to_agent_idx.get(eid.item(), 0) for eid in task_edge_ids],
                        device=trainer_obj.device)

                    min_wl_flat, active_indices = self._build_min_wl(trainer_obj, task_agent_ids, s_idx, tasks_min_accuracy,
                                                                    batch_sizes)
                    current_state_t, full_state_t = self._build_unified_state(trainer_obj, prev_lower_res, min_wl_flat, active_indices)
                    # get action (Removed invalid_logits_penalty)
                    n_idx, m_idx, _, active_probs, actual_wl, curr_mf = self.get_lower_actions(
                        trainer_obj, current_state_t, active_indices, min_wl_flat, t_idx, s_idx, tasks_min_accuracy, task_deadlines,
                        batch_sizes
                    )
                    # step env
                    current_lower_res = trainer_obj.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    current_lower_res["is_done"] = is_done
                    trainer_obj.aggregator.add_lower(current_lower_res)

                    if transition_cache is not None:
                        # 3. UNPACK THÊM CACHE_MASKS VÀ CACHE_ACTIVE_INDICES
                        cache_S_t, cache_A_t, cache_R_t, cache_prev_mf, cache_curr_mf, cache_is_done, cache_masks, cache_active_indices = transition_cache

                        # next_state_for_cache của slot cũ là full_state_t hiện tại lọc theo index cũ
                        next_state_for_cache = full_state_t[cache_active_indices]

                        cache_flat_agent_indices = cache_active_indices // trainer_obj.num_services

                        self.lower_agent.memory.add_batch(
                            cache_S_t, cache_prev_mf, cache_A_t, cache_R_t.view(-1, 1),
                            next_state_for_cache, cache_curr_mf,
                            torch.full((cache_S_t.shape[0], 1), float(cache_is_done), device=trainer_obj.device),
                            cache_flat_agent_indices,
                            action_masks=cache_masks
                        )
                    rewards = self._calculate_reward(trainer_obj, current_lower_res, actual_wl, task_agent_ids, t_idx, s_idx)

                    # 1. TÍNH MASK TẠI THỜI ĐIỂM HIỆN TẠI (Full 25 groups)
                    curr_masks = trainer_obj.env.engine.placement_matrix.T.unsqueeze(0).expand(self.num_edges, -1,
                                                                                               -1).reshape(-1,
                                                                                                           trainer_obj.num_nodes)

                    # 2. Lọc thông tin ACTIVE để lưu vào cache
                    rewards_active = rewards[active_indices]
                    prev_mf_active = prev_mf[active_indices]
                    curr_mf_active = curr_mf[active_indices]
                    curr_masks_active = curr_masks[active_indices]

                    # Cache (Tăng thành 8 phần tử để lưu active_indices)
                    transition_cache = (current_state_t, active_probs, rewards_active, prev_mf_active, curr_mf_active, is_done,
                                        curr_masks_active, active_indices)
                    prev_mf = curr_mf.clone()
                    prev_lower_res = current_lower_res

                    learn_res = self.lower_agent.learn()
                    if learn_res:
                        trainer_obj.total_lower_steps += 1
                        if isinstance(learn_res, dict):
                            trainer_obj.aggregator.record_q_stats("Lower", learn_res.get("q_min", 0.0), learn_res.get("q_max", 0.0), learn_res.get("q_mean", 0.0))
                            trainer_obj.aggregator.record_td_losses(lower_losses=learn_res.get("loss", 0.0))
                else:
                    trainer_obj.env.time_manager.tick()

                if trainer_obj.env.time_manager.is_new_frame():
                    current_upper_res = trainer_obj.env.collect_upper_metrics()
                    current_upper_state = self.build_upper_state(trainer_obj, current_upper_res)

                    is_ep_done = (slot == max_slots - 1)
                    mf_upper_loss = 0.0
                    if trainer_obj.total_lower_steps >= trainer_obj.lower_stable_threshold:
                        mf_upper_loss = self.store_upper_transitions(trainer_obj, prev_upper_state, current_upper_state, prev_upper_res, current_upper_res,
                                                     u_acts_matrix, is_ep_done)
                    prev_upper_state = current_upper_state
                    if current_lower_res is not None:
                        prev_lower_res = current_lower_res
                    trainer_obj.aggregator.add_upper(current_upper_res, mf_loss=mf_upper_loss)

                    # train upper
                    res = self.upper_agent.learn(torch.arange(trainer_obj.num_edge_agents, device=trainer_obj.device))

                    if res is not None:
                        trainer_obj.total_upper_steps += 1
                        if isinstance(res, dict):
                            loss = res["loss"]
                            trainer_obj.aggregator.record_q_stats("Edge_Group", res["q_min"], res["q_max"], res["q_mean"])
                        else:
                            loss = res
                        trainer_obj.aggregator.record_td_losses(upper_losses=loss)

            # XỬ LÝ CACHE CUỐI CÙNG CỦA EPISODE
            if transition_cache is not None:
                # 5. UNPACK THÊM CACHE_MASKS VÀ CACHE_ACTIVE_INDICES
                cache_S_t, cache_A_t, cache_R_t, cache_prev_mf, cache_curr_mf, cache_is_done, cache_masks, cache_active_indices = transition_cache
                dummy_next_state = cache_S_t.clone().zero_()
                
                cache_flat_agent_indices = cache_active_indices // trainer_obj.num_services

                # 6. THÊM action_masks=cache_masks VÀO ĐÂY
                self.lower_agent.memory.add_batch(
                    cache_S_t, cache_prev_mf, cache_A_t, cache_R_t.view(-1, 1),
                    dummy_next_state, cache_curr_mf,
                    torch.full((cache_S_t.shape[0], 1), 1.0, device=trainer_obj.device),  # Done = True
                    cache_flat_agent_indices,
                    action_masks=cache_masks
                )
            trainer_obj.aggregator.report_episode(ep)
            trainer_obj.aggregator.store_history()
            trainer_obj.update_rates(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {trainer_obj.total_lower_steps} | Upper Samples: {trainer_obj.total_upper_steps}")
            print(f"Zeta Lower: {trainer_obj.zeta_lower:.4f} | Zeta Upper: {trainer_obj.zeta_upper:.4f}")
            print(f"Current Epsilon (upper): {trainer_obj.eps_upper} (lower): {trainer_obj.lower_epsilons}")

        # Final Checkpoint Saving
        checkpoint_dir = getattr(trainer_obj.config, 'checkpoints', 'data/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.lower_agent.save(os.path.join(checkpoint_dir, "lower_td3_final.pth"))
        self.upper_agent.save(os.path.join(checkpoint_dir, "upper_d3qn_final.pth"))
        print(colored(f"\n[Final] Checkpoints saved to {checkpoint_dir}", "green", attrs=["bold"]))

    def _calculate_reward(self, trainer, current_lower_res, actual_workloads, task_agent_ids, t_idx, s_idx):
        """
        Calculates Reward for Lower Agent with:
        1. Energy cost (Base)
        2. Queue Pressure Penalty
        3. QoS Fail Squared Penalty
        """
        num_sample = self.num_edges * trainer.num_services
        device = trainer.device

        # 1. Base Energy Reward + QoS Penalty + Queue diff
        pre_reward = current_lower_res['reward']
        rewards = torch.full((num_sample,), pre_reward, dtype=torch.float32, device=device)

        # 2. Queue Pressure Penalty
        LAMBDA_QUEUE = 0.5
        obs_dict = current_lower_res['obs']
        Q = obs_dict['backlog']

        for e_idx in range(self.num_edges):
            for s_idx_loop in range(trainer.num_services):
                local_idx = e_idx * trainer.num_services + s_idx_loop
                A_e = actual_workloads[e_idx, s_idx_loop, :]
                Q_v = Q[:, s_idx_loop]
                queue_pressure_penalty = (Q_v * A_e).sum()
                rewards[local_idx] -= LAMBDA_QUEUE * queue_pressure_penalty

        # 4. Log Transform
        rew_divisor = trainer.config.norm_lower_rw
        rewards = log_transform(rewards / rew_divisor if rew_divisor != 0 else 1.0)

        return rewards

    def load_checkpoints(self, trainer, lower_path=None, upper_path=None):
        """Loads checkpoints for evaluation."""
        if lower_path and os.path.exists(lower_path):
            self.lower_agent.load(lower_path)
            print(f"[RB_SAC_CEN_STRA] Lower agent loaded from {lower_path}")
        
        if upper_path and os.path.exists(upper_path):
            self.upper_agent.load(upper_path)
            print(f"[RB_SAC_CEN_STRA] Upper agent loaded from {upper_path}")

    def run_evaluation(self, trainer, num_episodes=5):
        """Runs a deterministic evaluation loop."""
        print(f"\n>>> Starting Evaluation TD3 ({num_episodes} episodes) <<<")
        max_slots = trainer.env.time_manager.max_steps

        for ep in range(num_episodes):
            res = trainer.env.reset()
            obs_upper, prev_lower_res = res["upper"], res["lower"]
            current_upper_state = self.build_upper_state(trainer, obs_upper)

            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper, deterministic=True)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
                    task_edge_ids = T_E_map[t_idx].argmax(dim=1)
                    task_agent_ids = torch.tensor(
                        [self.edge_id_to_agent_idx.get(eid.item(), 0) for eid in task_edge_ids], device=trainer.device)

                    min_wl_flat, active_indices = self._build_min_wl(trainer, task_agent_ids, s_idx, tasks_min_accuracy, batch_sizes)
                    current_state, _ = self._build_unified_state(trainer, prev_lower_res, min_wl_flat, active_indices)

                    n_idx, m_idx, _, active_probs, actual_wl, _ = self.get_lower_actions(
                        trainer, current_state, active_indices, min_wl_flat, t_idx, s_idx,
                        tasks_min_accuracy, task_deadlines, batch_sizes, deterministic=True
                    )

                    next_res = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines,
                                                      tasks_min_accuracy)

                    trainer.aggregator.add_lower(next_res)
                    prev_lower_res = next_res
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    trainer.aggregator.add_upper(res_upper)

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            trainer.aggregator.report_episode(ep)
            trainer.aggregator.store_history()

        print(f"\n>>> Evaluation Complete <<<")

