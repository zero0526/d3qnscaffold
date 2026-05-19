import torch
from termcolor import colored
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.agents.masac import MFSACAgent
from matrix_source.trainers.train import Trainer
from matrix_source.utils.math_utils import to_binary
from matrix_source.trainers.train import log_transform

class RB_SAC_CEN_STRA(AlgorithmStrategy):

    def initialize_agents(self, trainer: Trainer):
        self.upper_state_dim = trainer.upper_state_dim
        self.upper_action_dim = trainer.upper_action_dim

        # historigram_task + historigram_dl + omega + task_size + f_allo_vs + back_log_vs
        self.lower_state_dim = trainer.max_models + 3 + 1 + 1 + 2*trainer.num_nodes
        self.lower_action_dim = trainer.max_models*trainer.num_nodes

        # Standard keys from metadata/static_matrices
        self.model_accuracies = trainer.env.metadata["model_accuracies"]
        self.service_deadlines = trainer.env.metadata["service_deadlines"]
        self.adj_matrix = trainer.env.static_matrices["adj_matrix"]

        self.edge_ids = trainer.env.static_matrices["edge_ids"]
        self.num_edges = len(self.edge_ids)
        self.edge_id_to_agent_idx = trainer.env.static_matrices["edge_id_to_agent_idx"]
        self.agent_id_to_edge_idx = {agent_id: edge_id for edge_id, agent_id in self.edge_id_to_agent_idx.items()}
        self.agent_adj_matrix = trainer.env.static_matrices["agent_adj_matrix"].to(trainer.device)
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

        self.lower_agent = MFSACAgent(
            node_id="shared_lower",
            node_type="Offload_Group",
            state_dim=self.lower_state_dim,
            action_dim=self.lower_action_dim,
            mf_dim= trainer.num_nodes,
            mf_hidden_sizes=trainer.config.hyper_neural["MF_SAC_HIDDEN_LAYER"],
            hidden_sizes=trainer.config.hyper_neural["AGENT_HIDDEN_LAYER"],
            actor_lr=float(trainer.config.hyper_neural["LOWER_LR"]),
            critic_lr=float(trainer.config.hyper_neural["LOWER_LR"]),
            mf_lr=float(trainer.config.hyper_neural["MF_LR"]),
            alpha_lr=float(trainer.config.hyper_neural["LOWER_LR"]),
            gamma=trainer.config.hyper_neural["DISCOUNT_FACTOR"],
            tau=trainer.config.hyper_neural["UPDATE_TARGET_COEF"],
            buffer_size=trainer.config.hyper_neural["MEMORY_SIZE"],
            batch_size=trainer.config.hyper_neural["BATCH_SIZE"],
            buffer_min_size=trainer.config.hyper_neural["BUFFER_MIN_SIZE"][1],
            num_instances=len(self.edge_ids),
            device=trainer.device,
            dist_type="gaussian",
            logs_q=True
        )

    def build_upper_state(self, trainer, obs_upper):
        actions = obs_upper['actions'] # (N, S)
        phi = obs_upper['phi_prob']    # (N, S)
        # Combine actions and phi for each node
        return torch.cat([actions, phi], dim=-1)

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        sc_mfs = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        
        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = sc_mfs[trainer.edge_node_ids]
        instance_indices = torch.tensor(range(self.num_edges), device=trainer.device)

        batch_a_ids = self.upper_agent.choose_action_batch(
            edge_states, edge_mfs, zeta=trainer.zeta_upper, agent_indices=instance_indices
        )

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)

        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def _build_lower_state_for_tasks(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, data_sizes):
        T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
        task_edge_ids = T_E_map[t_idx].argmax(dim=1)
        
        num_tasks = len(t_idx)
        num_services = trainer.num_services
        max_models = trainer.max_models
        
        obs_dict = res_lower['obs']
        meta = trainer.env.metadata
        
        task_agent_ids = torch.tensor([self.edge_id_to_agent_idx.get(eid.item(), 0) for eid in task_edge_ids], device=trainer.device)
        combined_indices = task_agent_ids * num_services + s_idx
        num_groups = self.num_edges * num_services
        
        task_counts = torch.zeros(num_groups, device=trainer.device)
        task_counts.index_add_(0, combined_indices, torch.ones(num_tasks, device=trainer.device))
        
        task_srv_dls = self.service_deadlines[s_idx]
        dl_match = (torch.abs(task_deadlines.view(-1, 1) - task_srv_dls) < 1e-4).float()
        dl_counts = torch.zeros((num_groups, 3), device=trainer.device)
        dl_counts.index_add_(0, combined_indices, dl_match)
        task_dl_hists = dl_counts / task_counts.view(-1, 1).clamp(min=1.0)
        
        srv_accs_all = self.model_accuracies[s_idx]
        diffs = srv_accs_all - tasks_min_accuracy.view(-1, 1)
        diffs[diffs < 0] = float('inf')
        best_models = diffs.argmin(dim=1)
        acc_one_hot = torch.zeros((num_tasks, max_models), device=trainer.device)
        valid_mask = diffs.min(dim=1).values != float('inf')
        acc_one_hot[torch.arange(num_tasks)[valid_mask], best_models[valid_mask]] = 1.0
        
        acc_counts = torch.zeros((num_groups, max_models), device=trainer.device)
        acc_counts.index_add_(0, combined_indices, acc_one_hot)
        task_acc_hists = acc_counts / task_counts.view(-1, 1).clamp(min=1.0)
        
        data_sums = torch.zeros(num_groups, device=trainer.device)
        data_sums.index_add_(0, combined_indices, data_sizes)
        
        omegas = meta['service_omega'].squeeze(-1)
        s_cpus = obs_dict['cpu_alloc']
        s_backlogs = obs_dict['backlog']
        
        full_omegas = omegas.unsqueeze(0).repeat(self.num_edges, 1).view(num_groups, 1)
        full_data_sums = data_sums.view(num_groups, 1)

        cpu_allo = (
            s_cpus.T.unsqueeze(0)  # (1, num_services, num_nodes)
            .expand(self.num_edges, -1, -1)  # (num_edges, num_services, num_nodes)
            .reshape(num_groups, -1)  # (num_groups, num_nodes)
        )

        backlog_coll = (
            s_backlogs.T.unsqueeze(0)
            .expand(self.num_edges, -1, -1)
            .reshape(num_groups, -1)
        )
        full_state_tensor = torch.cat([
            task_acc_hists,
            task_dl_hists, 
            full_omegas,
            full_data_sums/trainer.config.norm_data_size,
            cpu_allo/trainer.config.norm_gflop,
            backlog_coll/trainer.config.norm_gflop
        ], dim=-1)
        # num_agent x num_service x num_node
        state_3d = full_state_tensor.view(self.num_edges, num_services, -1)
        return state_3d, task_agent_ids, task_edge_ids

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        meta = trainer.env.metadata
        unit_sizes = meta['service_input_size']
        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        state_3d, task_agent_ids, task_edge_ids = self._build_lower_state_for_tasks(trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, data_sizes)
        
        # Aggregate terminal Mean Field to Edge Level
        mf = res_lower['mean_field'] # (num_terminals, action_dim)

        num_edges = self.num_edges
        num_services = trainer.num_services
        # num_agent x num_service x lower_feature
        flat_states = state_3d.view(num_edges * num_services, -1)
        # Repeat edge MF across all services
        #  num_agent, -> num_agent x 1 --> num_agent x num_service  --> num_agent_service
        flat_agent_indices = torch.arange(num_edges, device=trainer.device).unsqueeze(1).expand(-1, num_services).reshape(-1)
        
        all_probs = self.lower_agent.choose_action_batch(
            flat_states,
            mf,
            agent_indices=flat_agent_indices
        )
        all_probs = torch.as_tensor(all_probs, device=trainer.device, dtype=torch.float32)

        node_ids, model_ids, masked_probs = self.heuristic(trainer, task_agent_ids, s_idx, all_probs, tasks_min_accuracy, task_deadlines, data_sizes)
        return node_ids, model_ids, task_agent_ids, masked_probs

    def compute_next_mean_field(self, trainer:Trainer, source_edge_ids, s_idx, target_edge_ids, target_model_ids, batch_sizes):
        num_edges = self.num_edges
        num_services = trainer.num_services
        num_nodes= trainer.num_nodes
        model_workloads= trainer.env.metadata["model_workloads"]
        actions= torch.zeros((num_edges, num_nodes, num_services), device=trainer.device)
        # num_edge x num_edge
        task_workloads = model_workloads[s_idx, target_model_ids]*batch_sizes
        flat_indices = (
                source_edge_ids * (num_nodes * num_services) +
                target_edge_ids * num_services +
                s_idx
        )
        actions.view(-1).index_add_(0, flat_indices, task_workloads)

        # num_edge
        num_neibor = self.agent_adj_matrix.sum(dim=1)
        num_neibor = torch.clamp(num_neibor, min=1.0)
        normalized_adj = self.agent_adj_matrix / num_neibor.unsqueeze(1)
        # num_agent x num_node x num_service
        action_mf = torch.einsum('ij,jnk->ink', normalized_adj, actions)

        action_mf = action_mf.permute(0, 2, 1).contiguous()
        action_mf= action_mf.reshape(num_edges*num_services, -1)
        row_sum = action_mf.sum(dim=-1, keepdim=True)

        normalized_action_mf = (
                action_mf /
                (row_sum + 1e-6)
        )
        # num_agent_service x num_node
        return normalized_action_mf

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx, tasks_min_accuracy, task_deadlines, batch_sizes, all_probs):
        state_3d, task_agent_ids, _ = self._build_lower_state_for_tasks(trainer, current_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        next_state_3d, _, _ = self._build_lower_state_for_tasks(trainer, next_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        
        mf_term = current_res['mean_field']
        ns_mf_term = next_res['mean_field']

        # Global base reward calculations
        reward = next_res['reward']
        rew_divisor = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        num_sample = state_3d.shape[0] * state_3d.shape[1]

        # --- Agent-Specific Reward Decomposition ---
        # Instead of a single scalar to all, we give credit where credit is due.
        # success_qos: {node_idx: [service_counts]}
        # violations_qos: {node_idx: [service_counts]}
        info = next_res['info']
        success_qos = info.get('success_qos', {})
        violate_qos = info.get('violate_qos', {})
        
        # Flattened rewards for all (edge, service) pairs
        # Initialize with the continuous drift/energy base (norm_rew)
        rewards = torch.full((num_sample,), norm_rew, dtype=torch.float32, device=trainer.device)
        
        # Add discrete success/fail rewards per agent-service group
        for e_idx in range(self.num_edges):
            e_node_id = self.edge_ids[e_idx] # The physical node ID of this agent
            node_succ = success_qos.get(e_node_id, [0]*trainer.num_services)
            node_fail = violate_qos.get(e_node_id, [0]*trainer.num_services)
            
            for s_idx in range(trainer.num_services):
                local_idx = e_idx * trainer.num_services + s_idx
                # +1 reward for success, -10 for violation (matching environment scaling)
                # We normalize these relative to the log-transformed global reward
                agent_perf = (node_succ[s_idx] * 1.0 - node_fail[s_idx] * 10.0) / 100.0
                rewards[local_idx] += agent_perf

        # --- Q-Value Slope (Smooth Gradient) shaping ---
        # Penalize placing probability on unplaced nodes to guide the Q-Gradient
        placement = trainer.env.engine.placement_matrix.T # (num_services, num_nodes)
        placed_mask = placement.unsqueeze(-1).expand(-1, -1, trainer.max_models).reshape(trainer.num_services, -1)
        placed_mask_all = placed_mask.unsqueeze(0).expand(self.num_edges, -1, -1).reshape(-1, self.lower_action_dim)
        
        probs_for_penalty = all_probs
        if hasattr(trainer, "strategy") and getattr(trainer.strategy, "lower_agent", None):
            if trainer.strategy.lower_agent.dist_type == "gaussian":
                # Note: all_probs passed here is now the masked_probs from heuristic (already normalized)
                probs_for_penalty = all_probs 
                
        # 1. Placement penalty
        invalid_mass = (probs_for_penalty * (1.0 - placed_mask_all)).sum(dim=-1)
        rewards = rewards - (invalid_mass * 5.0)

        # 2. Deadline Urgency Penalty — mirrors heuristic's delay_score for Critic learning
        # (E, num_nodes) = row-index each edge's row from distance matrix
        edge_node_row = [self.agent_id_to_edge_idx[i] for i in range(self.num_edges)]
        agent_dist = self.distance_matrix[edge_node_row]  # (E, N)

        # service_input_size (S, 1), service_deadlines (S, 3) -> use mean column
        svc_input = trainer.env.metadata["service_input_size"].squeeze(-1)    # (S,)
        svc_dl    = trainer.env.metadata["service_deadlines"][:, 1]           # (S,) mean deadline

        # trans_delay (E, S, N) = dist (E,1,N) * input_size (1,S,1)
        trans_delay  = agent_dist.unsqueeze(1) * svc_input.view(1, -1, 1)     # (E, S, N)
        # remaining time ratio = clamp( (dl - delay) / dl )
        safety       = (svc_dl.view(1,-1,1) - trans_delay).clamp(min=0)       # (E, S, N)
        delay_score  = safety / (svc_dl.view(1,-1,1) + 1e-8)                  # (E, S, N) in [0,1]

        # expand to (E*S, N*M) — same shape as probs_for_penalty
        delay_score_all = (
            delay_score.unsqueeze(-1)
            .expand(-1, -1, -1, trainer.max_models)        # (E, S, N, M)
            .reshape(self.num_edges * trainer.num_services, -1)  # (E*S, N*M)
        )
        # penalise probability mass placed on SLOW nodes  (1 - delay_score) ∈ [0,1]
        slow_mass = (probs_for_penalty * (1.0 - delay_score_all)).sum(dim=-1)
        rewards = rewards - (slow_mass * 2.0)
        # -----------------------------------------------

        done = torch.tensor([next_res.get("is_done", False)]*num_sample, dtype=torch.float32, device=trainer.device)
        flat_agent_indices = torch.arange(self.num_edges, device=trainer.device).unsqueeze(1).expand(-1, trainer.num_services).reshape(-1)
        flat_state= state_3d.view(-1, self.lower_state_dim)
        next_flat_state= next_state_3d.view(-1, self.lower_state_dim)
        avg_mf_loss = self.lower_agent.store_transition_train_mf_batch(
            flat_state, mf_term, ns_mf_term, all_probs, rewards, next_flat_state, done, agent_ids=flat_agent_indices
        )
        trainer.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=state_3d.view(self.num_edges, -1)[0] if len(state_3d) > 0 else None)
        return avg_mf_loss

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

        # Actions for upper agent are placement vectors. For DQN, we need indices.
        # If upper agent is scalar SAC/DQN, we map placement matrix row to an index.
        # But Rainbow agents usually use discrete actions.
        a_ids = acts_matrix[trainer.edge_node_ids].sum(dim=-1).long() # Placeholder logic
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)

        loss = self.upper_agent.store_transition_train_mf_batch(
            s_all, edge_mfs, edge_ns_mfs, a_ids, rewards, ns_all, dones, instance_indices
        )
        return loss

    def run_training(self, trainer_obj: Trainer):
        num_eps = trainer_obj.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = trainer_obj.env.time_manager.max_steps
        print(colored("="*50, "blue", attrs=["bold"]))
        print(colored(f"RB-SAC-CEN Training ({num_eps} frames)", "blue", attrs=["bold"]))
        print(colored("="*50, "blue", attrs=["bold"]))

        for ep in range(num_eps):
            res = trainer_obj.env.reset()
            obs_upper, prev_lower_res = res["upper"], res["lower"]
            prev_lower_res["mean_field"]= torch.zeros((self.num_edges*trainer_obj.num_services, trainer_obj.num_nodes), device=trainer_obj.device)
            current_upper_state = self.build_upper_state(trainer_obj, obs_upper)
            
            epoch_lower_steps = 0
            for slot in range(max_slots):
                if trainer_obj.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer_obj, current_upper_state, obs_upper)
                    trainer_obj.env.step_upper(u_acts_matrix)
                    # Get tasks
                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines  = trainer_obj.workload_gen.generate_step()

                if len(t_idx) > 0:
                    n_idx, m_idx, task_agent_ids, all_probs = self.get_lower_actions(trainer_obj, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    next_res = trainer_obj.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    next_res["mean_field"] = self.compute_next_mean_field(trainer_obj, task_agent_ids, s_idx, n_idx, m_idx, batch_sizes)

                    next_res["is_done"] = (slot == max_slots - 1)
                    lower_train_metrics = self.store_lower_transitions(trainer_obj, prev_lower_res, next_res, t_idx, s_idx, n_idx, m_idx, tasks_min_accuracy, task_deadlines, batch_sizes, all_probs)
                    epoch_lower_steps += 1
                    prev_lower_res = next_res

                    learn_res = self.lower_agent.learn()
                    trainer_obj.total_lower_steps += 1
                    if learn_res and isinstance(learn_res, dict):
                        trainer_obj.aggregator.record_q_stats("Lower", learn_res.get("q_min", 0.0), learn_res.get("q_max", 0.0), learn_res.get("q_mean", 0.0))
                        trainer_obj.aggregator.record_td_losses(lower_losses=learn_res.get("loss", 0.0))
                else:
                    trainer_obj.env.time_manager.tick()

                if trainer_obj.env.time_manager.is_new_frame():
                    next_res_upper = trainer_obj.env.collect_upper_metrics()
                    ns_upper_state = self.build_upper_state(trainer_obj, next_res_upper)

                    is_ep_done = (slot == max_slots - 1)
                    mf_upper_loss = 0.0
                    if trainer_obj.total_lower_steps >= trainer_obj.lower_stable_threshold:
                        mf_upper_loss = self.store_upper_transitions(trainer_obj, current_upper_state, ns_upper_state, obs_upper,
                                                     next_res_upper, u_acts_matrix, is_ep_done)
                        trainer_obj.total_upper_steps += 1

                    trainer_obj.aggregator.add_upper(next_res_upper, mf_loss=mf_upper_loss)

                    # train upper
                    res = self.upper_agent.learn(torch.arange(trainer_obj.num_edge_agents, device=trainer_obj.device))
                    if res is not None:
                        if isinstance(res, dict):
                            loss = res["loss"]
                            trainer_obj.aggregator.record_q_stats("Edge_Group", res["q_min"], res["q_max"], res["q_mean"])
                        else:
                            loss = res
                        trainer_obj.aggregator.record_td_losses(upper_losses=loss)

            # End of Frame - Upper Train
            upper_train_metrics = self.upper_agent.learn()
            if isinstance(upper_train_metrics, dict):
                trainer_obj.aggregator.record_q_stats(
                    "Upper",
                    upper_train_metrics.get("q_min", 0.0),
                    upper_train_metrics.get("q_max", 0.0),
                    upper_train_metrics.get("q_mean", 0.0)
                )

            trainer_obj.aggregator.store_history()
            trainer_obj.update_rates(ep)
            trainer_obj.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {trainer_obj.total_lower_steps} | Upper Samples: {trainer_obj.total_upper_steps}")
            print(f"Zeta Lower: {trainer_obj.zeta_lower:.4f} | Zeta Upper: {trainer_obj.zeta_upper:.4f}")
            print(f"Current Epsilon (Edge N0): {trainer_obj.epsilons[0]:.4f}")


    def heuristic(
        self,
        trainer: Trainer,
        agent_ids,
        service_ids,
        probs,
        accuracies,
        deadlines,
        data_sizes
    ):
        # 3:net 1:cloud [0,2]:edge
        # trainer.env.engine.placement_matrix
        """
        :param agent_ids: agent id need to transfer to edge_id : num_task
        :param service_ids:
        :param probs: num_agent_service x num_node_model
        :param accuracies: num_task
        :param deadlines: num_task
        :return:
        """
        num_nodes = trainer.num_nodes
        max_models = trainer.max_models
        num_tasks= len(service_ids)
        # group index
        indices = (
            agent_ids * trainer.num_services +
            service_ids
        )
        # (num_tasks, num_nodes*max_models)
        selected_probs = probs[indices]
        if hasattr(trainer, "strategy") and getattr(trainer.strategy, "lower_agent", None):
            if trainer.strategy.lower_agent.dist_type == "gaussian":
                # Convert raw (-1 to 1) tanh outputs into proper probability distribution
                # Removed / 0.1 to avoid vanishing gradient from near one-hot distribution
                selected_probs = torch.softmax(selected_probs, dim=-1)

        # --------- placement process (crucial) -----------
        # Only assign tasks to nodes where the Upper Agent ACTUALLY placed the service
        # placement_matrix is (num_nodes, num_services) -> 0 or 1
        placement = trainer.env.engine.placement_matrix
        placed_mask = placement.unsqueeze(-1).expand(-1, -1, max_models) # (N, S, max_models)
        placed_mask = placed_mask.permute(1, 0, 2).reshape(trainer.num_services, -1) # (S, N*max_models)
        selected_placed_mask = placed_mask[service_ids] # (num_tasks, N*max_models)

        # --------- deadline process---------------------
        edge_ids = torch.tensor([self.agent_id_to_edge_idx.get(eid.item()) for eid in agent_ids], device=trainer.device)
        # num_tasks x num_computing_node
        transmissions_delay = self.distance_matrix[edge_ids]*data_sizes.unsqueeze(1)
        # num_tasks x num_computing_node_num_model
        delay= (deadlines.unsqueeze(1) - transmissions_delay)
        delay = torch.clamp(delay, min=0)
        delay = delay.unsqueeze(-1).expand(
            -1,
            -1,
            trainer.max_models
        ).reshape(num_tasks, -1)
        delay_score = delay / (deadlines.unsqueeze(1) + 1e-8)

        # num_tasks x num_computing_node_num_model
        masked_probs= norm_prob(selected_probs * delay_score * selected_placed_mask)
        selected_idx= torch.multinomial(masked_probs, num_samples=1).squeeze(1)
        assigned_nodes= selected_idx//max_models
        assigned_models= selected_idx%max_models

        # Expand masked_probs back to full (E*S, N*M) for buffer storage
        full_masked_probs = torch.zeros((self.num_edges * trainer.num_services, trainer.num_nodes * trainer.max_models), device=trainer.device)
        full_masked_probs.index_add_(0, indices, masked_probs)

        return assigned_nodes, assigned_models, full_masked_probs


def norm_prob(weights):
    weights = weights.float()
    weights = torch.clamp(weights, min=0)

    row_sum = weights.sum(dim=1, keepdim=True)

    probs = torch.where(
        row_sum > 1e-8,
        weights / row_sum,
        torch.ones_like(weights) / weights.size(1)
    )

    return probs