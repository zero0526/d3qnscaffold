import torch
import time
from termcolor import colored
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.agents.rainbow_reduce import RainbowMultiAgent
from matrix_source.agents.masac import MFSACAgent
from matrix_source.trainers.train import Trainer
from matrix_source.utils.math_utils import to_binary

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
        self.edge_id_to_idx = {eid: i for i, eid in enumerate(self.edge_ids)}
        self.edge_adj_matrix = trainer.env.static_matrices["edge_adj_matrix"].to(trainer.device)
        
        self.upper_agent = RainbowMultiAgent(
            node_id="shared_upper",
            node_type="EDGE",
            state_dim=self.upper_state_dim,
            action_dim=self.upper_action_dim,
            u_action_dim=self.upper_action_dim,
            mf_hidden_sizes=trainer.config.hyper_neural["MF_HIDDEN_LAYER"],
            mf_lr=float(trainer.config.hyper_neural["MF_LR"]),
            buffer_min_size=trainer.config.hyper_neural["BUFFER_MIN_SIZE"][0],
            hidden_sizes=trainer.config.hyper_neural["AGENT_HIDDEN_LAYER"],
            lr=float(trainer.config.hyper_neural["UPPER_LR"]),
            gamma=trainer.config.hyper_neural["DISCOUNT_FACTOR"],
            buffer_size=trainer.config.hyper_neural["MEMORY_SIZE"],
            batch_size=trainer.config.hyper_neural["BATCH_SIZE"],
            num_instances=self.num_edges,
            device=trainer.device,
            logs_q=True
        )

        self.lower_agent = MFSACAgent(
            node_id="shared_lower",
            node_type="Offload_Group",
            state_dim=self.lower_state_dim,
            action_dim=self.lower_action_dim,
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
            dist_type="dirichlet",
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
            edge_states, edge_mfs, agent_indices=instance_indices
        )

        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)

        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def _build_lower_state_for_tasks(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
        task_edge_ids = T_E_map[t_idx].argmax(dim=1)
        
        num_tasks = len(t_idx)
        num_nodes = trainer.num_nodes
        num_services = trainer.num_services
        max_models = trainer.max_models
        
        obs_dict = res_lower['obs']
        meta = trainer.env.metadata
        unit_sizes = meta['service_input_size']
        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        
        task_agent_ids = torch.tensor([self.edge_id_to_idx.get(eid.item(), 0) for eid in task_edge_ids], device=trainer.device)
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
        
        # Broadcast environment info to groups
        # Edge Node info: (num_edges, num_services, num_nodes)
        edge_cpus = s_cpus[self.edge_ids].unsqueeze(1).repeat(1, num_services, 1)
        edge_backlogs = s_backlogs[self.edge_ids].unsqueeze(1).repeat(1, num_services, 1)
        
        full_omegas = omegas.unsqueeze(0).repeat(self.num_edges, 1).view(num_groups, 1)
        full_data_sums = data_sums.view(num_groups, 1)
        full_cpus = edge_cpus.view(num_groups, num_nodes)
        full_backlogs = edge_backlogs.view(num_groups, num_nodes)
        
        full_state_tensor = torch.cat([
            task_acc_hists,
            task_dl_hists, 
            full_omegas,
            full_data_sums,
            full_cpus,
            full_backlogs
        ], dim=-1)
        
        state_3d = full_state_tensor.view(self.num_edges, num_services, -1)
        return state_3d, task_agent_ids

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        state_3d, task_agent_ids = self._build_lower_state_for_tasks(trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        
        # Aggregate terminal Mean Field to Edge Level
        mf_terminals = res_lower['mean_field'] # (num_terminals, action_dim)
        T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
        terminal_to_edge_count = T_E_map.sum(dim=0).clamp(min=1.0)
        mf_global_nodes = (T_E_map.T @ mf_terminals) / terminal_to_edge_count.unsqueeze(-1)
        mf_edges = mf_global_nodes[self.edge_ids] # (num_edges, action_dim)
        
        num_edges = self.num_edges
        num_services = trainer.num_services
        
        flat_states = state_3d.view(num_edges * num_services, -1)
        # Repeat edge MF across all services
        flat_mfs = mf_edges.unsqueeze(1).expand(-1, num_services, -1).reshape(num_edges * num_services, -1)
        flat_agent_indices = torch.arange(num_edges, device=trainer.device).unsqueeze(1).expand(-1, num_services).reshape(-1)
        
        all_probs = self.lower_agent.choose_action_batch(
            flat_states,
            flat_mfs,
            agent_indices=flat_agent_indices
        )
        all_probs = torch.as_tensor(all_probs, device=trainer.device, dtype=torch.float32)
        all_probs_3d = all_probs.view(num_edges, num_services, -1)
        
        task_actions_prob = all_probs_3d[task_agent_ids, s_idx]
        a_ids = torch.argmax(task_actions_prob, dim=-1)
        return a_ids // trainer.max_models, a_ids % trainer.max_models, task_agent_ids

    def compute_next_mean_field(self, trainer, source_edge_ids, s_idx, target_edge_ids, target_model_ids=None):
        num_edges = self.num_edges
        num_services = trainer.num_services
        action_dim = self.num_nodes + trainer.max_models if target_model_ids is not None else self.num_nodes
        
        if target_model_ids is not None:
             # Lower MF: (num_edges, num_services, action_dim)
             # One-hot actions for each task
             task_node_oh = torch.nn.functional.one_hot(target_edge_ids, num_classes=self.num_nodes)
             task_model_oh = torch.nn.functional.one_hot(target_model_ids, num_classes=trainer.max_models)
             task_actions = torch.cat([task_node_oh, task_model_oh], dim=-1).float()
             
             # Group tasks by (edge, service)
             num_groups = num_edges * num_services
             combined_indices = source_edge_ids * num_services + s_idx
             group_actions = torch.zeros((num_groups, action_dim), device=trainer.device)
             group_counts = torch.zeros(num_groups, device=trainer.device)
             
             group_actions.index_add_(0, combined_indices, task_actions)
             group_counts.index_add_(0, combined_indices, torch.ones_like(source_edge_ids, dtype=torch.float))
             
             group_mf_internal = group_actions / group_counts.view(-1, 1).clamp(min=1.0)
             group_mf_internal = group_mf_internal.view(num_edges, num_services, action_dim)
             
             # Propagation: Neighbor average
             neighbor_sums = self.edge_adj_matrix @ group_mf_internal.view(num_edges, -1)
             neighbor_counts = self.edge_adj_matrix.sum(dim=1).clamp(min=1.0)
             next_mf = (neighbor_sums / neighbor_counts.unsqueeze(-1)).view(num_edges, num_services, action_dim)
        else:
            # Upper MF: (num_edges, action_dim)
            # target_edge_ids is the placement matrix (E, S) or actions. 
            # If it's a matrix:
            node_actions = target_edge_ids.float()
            neighbor_sums = self.edge_adj_matrix @ node_actions
            neighbor_counts = self.edge_adj_matrix.sum(dim=1).clamp(min=1.0)
            next_mf = neighbor_sums / neighbor_counts.unsqueeze(-1)
            
        return next_mf

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        state_3d, task_agent_ids = self._build_lower_state_for_tasks(trainer, current_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        next_state_3d, _ = self._build_lower_state_for_tasks(trainer, next_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        
        # Aggregate MF (Terminal -> Edge)
        T_E_map = trainer.env.static_matrices["terminal_to_comp_node_map"]
        terminal_to_edge_count = T_E_map.sum(dim=0).clamp(min=1.0)
        
        mf_term = current_res['mean_field']
        mf_edges = (T_E_map.T @ mf_term) / terminal_to_edge_count.unsqueeze(-1)
        ns_mf_term = next_res['mean_field']
        ns_mf_edges = (T_E_map.T @ ns_mf_term) / terminal_to_edge_count.unsqueeze(-1)
        
        num_services = trainer.num_services
        task_group_indices = task_agent_ids * num_services + s_idx 
        unique_group_indices = torch.unique(task_group_indices)
        
        group_agent_instances = unique_group_indices // num_services
        group_service_indices = unique_group_indices % num_services
        
        s_obs = state_3d[group_agent_instances, group_service_indices]
        ns_obs = next_state_3d[group_agent_instances, group_service_indices]
        
        # Important: group_agent_instances are edge indices (0..num_edges-1)
        # We need the MF for those edge nodes
        edge_ids_global = [self.edge_ids[idx.item()] for idx in group_agent_instances]
        edge_ids_tensor = torch.tensor(edge_ids_global, device=trainer.device)
        
        s_mf = mf_edges[edge_ids_tensor]
        ns_mf = ns_mf_edges[edge_ids_tensor]

        # Actions chosen for groups
        representative_task_indices = []
        for g_idx in unique_group_indices:
            representative_task_indices.append((task_group_indices == g_idx).nonzero()[0].item())
        rep_indices = torch.tensor(representative_task_indices, device=trainer.device)
        
        a_ids = n_idx[rep_indices] * trainer.max_models + m_idx[rep_indices]
        acts = torch.nn.functional.one_hot(a_ids, num_classes=self.lower_agent.action_dim).float()
        
        # Scalar reward broadcasting
        reward_val = float(current_res['reward'])
        group_rewards = torch.full((len(unique_group_indices),), reward_val, device=trainer.device)
        dones = torch.full((len(unique_group_indices),), float(next_res['new_frame']), device=trainer.device)
        
        loss = self.lower_agent.store_transition_train_mf_batch(
            s_obs, s_mf, ns_mf, acts, group_rewards, ns_obs, dones, group_agent_instances
        )
        return loss

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        # res_upper keys: 'actions', 'phi_prob', 'mean_fields', 'resources'
        sc_mfs = current_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        sc_ns_mfs = next_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        
        edge_mfs = sc_mfs[trainer.edge_node_ids]
        edge_ns_mfs = sc_ns_mfs[trainer.edge_node_ids]

        reward_val = float(current_res['reward'])
        rewards = torch.full((self.num_edges,), reward_val, device=trainer.device)
        dones = torch.full((self.num_edges,), float(is_done), device=trainer.device)
        agent_ids = torch.arange(self.num_edges, device=trainer.device)
        
        # Actions for upper agent are placement vectors. For DQN, we need indices.
        # If upper agent is scalar SAC/DQN, we map placement matrix row to an index.
        # But Rainbow agents usually use discrete actions.
        a_ids = acts_matrix[trainer.edge_node_ids].sum(dim=-1).long() # Placeholder logic
        
        loss = self.upper_agent.store_transition_train_mf_batch(
            s_all, edge_mfs, edge_ns_mfs, a_ids, rewards, ns_all, dones, agent_ids
        )
        return loss

    def run_training(self, trainer):
        start_time = time.time()
        print(colored("="*50, "blue", attrs=["bold"]))
        print(colored(f"RB-SAC-CEN Training ({trainer.max_epochs} frames)", "blue", attrs=["bold"]))
        print(colored("="*50, "blue", attrs=["bold"]))

        for epoch in range(trainer.max_epochs):
            res = trainer.env.reset()
            res_upper, res_lower = res["upper"], res["lower"]
            current_upper_state = self.build_upper_state(trainer, res_upper)
            
            epoch_lower_loss = 0.0
            epoch_lower_steps = 0
            epoch_upper_mf_loss = 0.0
            is_done = False

            while not is_done:
                # 1. Upper Level Action
                pm = self.get_upper_actions(trainer, current_upper_state, res_upper)
                trainer.env.step_upper(pm)

                # 2. Lower Level Slot Loop
                while not trainer.env.time_manager.is_new_frame() and not is_done:
                    # Get tasks
                    terminal_indices, svc_indices, task_batch_sizes, task_deadlines, tasks_min_accuracy, res_lower = trainer.get_task_samples()

                    if len(terminal_indices) > 0:
                        n_idx, m_idx, task_agent_ids = self.get_lower_actions(trainer, res_lower, terminal_indices, svc_indices, tasks_min_accuracy, task_deadlines, task_batch_sizes)
                        next_res = trainer.env.step_lower(terminal_indices, svc_indices, task_batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    
                        next_res["mean_field"] = self.compute_next_mean_field(trainer, task_agent_ids, svc_indices, n_idx, m_idx)
                        
                        lower_train_metrics = self.store_lower_transitions(trainer, res_lower, next_res, terminal_indices, svc_indices, n_idx, m_idx, tasks_min_accuracy, task_deadlines, task_batch_sizes)
                        if isinstance(lower_train_metrics, dict):
                            epoch_lower_loss += lower_train_metrics.get("loss", lower_train_metrics.get("critic_loss", 0.0))
                            trainer.aggregator.record_q_stats(
                                "Lower", 
                                lower_train_metrics.get("q_min", 0.0), 
                                lower_train_metrics.get("q_max", 0.0), 
                                lower_train_metrics.get("q_mean", 0.0)
                            )
                        else:
                            epoch_lower_loss += lower_train_metrics
                            
                        epoch_lower_steps += 1
                        trainer.aggregator.record_step(next_res)
                        res_lower = next_res

                        if self.lower_agent.memory.total_adds % trainer.config.hyper_neural["LOWER_TRAIN_FREQ"] == 0:
                            learn_res = self.lower_agent.learn()
                            if learn_res and isinstance(learn_res, dict):
                                trainer.aggregator.record_q_stats("Lower", learn_res.get("q_min", 0.0), learn_res.get("q_max", 0.0), learn_res.get("q_mean", 0.0))

                    else:
                        break # No tasks

                # 3. Upper Metrics & Transition
                next_res_upper = trainer.env.collect_upper_metrics()
                is_done = next_res_upper["is_done"]
                ns_upper_state = self.build_upper_state(trainer, next_res_upper['obs'])
                
                loss_uf = self.store_upper_transitions(trainer, current_upper_state, ns_upper_state, res_upper, next_res_upper, pm, is_done)
                epoch_upper_mf_loss += loss_uf
                current_upper_state = ns_upper_state
                res_upper = next_res_upper

            # End of Frame - Upper Train
            upper_train_metrics = self.upper_agent.learn()
            if isinstance(upper_train_metrics, dict):
                trainer.aggregator.record_q_stats(
                    "Upper", 
                    upper_train_metrics.get("q_min", 0.0), 
                    upper_train_metrics.get("q_max", 0.0), 
                    upper_train_metrics.get("q_mean", 0.0)
                )
            
            trainer.aggregator.store_history()
            trainer.update_rates(epoch)
            
            # Print epoch summary
            if (epoch + 1) % 5 == 0:
                dt = time.time() - start_time
                trainer.aggregator.print_epoch_summary(epoch, trainer.max_epochs, dt)

        # Plot metrics
        trainer.aggregator.plot_history(trainer.config.plot_dir)
        print(f"\nTraining completed! Artifacts saved to: {trainer.config.plot_dir}")