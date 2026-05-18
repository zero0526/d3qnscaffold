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
            edge_states, edge_mfs, zeta=trainer.zeta_upper, agent_indices=instance_indices
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
        num_services = trainer.num_services
        max_models = trainer.max_models
        
        obs_dict = res_lower['obs']
        meta = trainer.env.metadata
        unit_sizes = meta['service_input_size']
        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        
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
        
        state_3d = full_state_tensor.view(self.num_edges, num_services, -1)
        return state_3d, task_agent_ids, task_edge_ids

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        state_3d, task_agent_ids, task_edge_ids = self._build_lower_state_for_tasks(trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        
        # Aggregate terminal Mean Field to Edge Level
        mf = res_lower['mean_field'] # (num_terminals, action_dim)

        num_edges = self.num_edges
        num_services = trainer.num_services
        
        flat_states = state_3d.view(num_edges * num_services, -1)
        # Repeat edge MF across all services
        flat_agent_indices = torch.arange(num_edges, device=trainer.device).unsqueeze(1).expand(-1, num_services).reshape(-1)
        
        all_probs = self.lower_agent.choose_action_batch(
            flat_states,
            mf,
            agent_indices=flat_agent_indices
        )
        all_probs = torch.as_tensor(all_probs, device=trainer.device, dtype=torch.float32)

        node_ids, model_ids= self.heuristic(trainer, task_agent_ids, s_idx, all_probs, tasks_min_accuracy)
        return node_ids, model_ids, task_agent_ids, all_probs

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
        action_mf = torch.einsum('ij,jnk->ink', normalized_adj, actions)

        action_mf = action_mf.permute(0, 2, 1)
        mean = action_mf.mean(dim=-1, keepdim=True)
        std = action_mf.std(dim=-1, keepdim=True)

        normalized_action_mf = (action_mf - mean) / (std + 1e-6)

        normalized_action_mf = normalized_action_mf.reshape(
            num_edges,
            -1
        )
        return normalized_action_mf

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx, tasks_min_accuracy, task_deadlines, batch_sizes, all_probs):
        state_3d, task_agent_ids, _ = self._build_lower_state_for_tasks(trainer, current_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        next_state_3d, _, _ = self._build_lower_state_for_tasks(trainer, next_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
        
        mf_term = current_res['mean_field']
        ns_mf_term = next_res['mean_field']

        reward = next_res['reward']
        rew_divisor = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        num_sample= state_3d.shape[0]*state_3d.shape[1]
        rewards = torch.full((num_sample,), norm_rew, dtype=torch.float32, device=trainer.device)

        done = torch.tensor([next_res["new_frame"]]*num_sample, dtype=torch.float32, device=trainer.device)
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
        rewards = torch.full((self.num_edges,), reward, device=trainer.device)
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

    def run_training(self, trainer: Trainer):
        num_eps = trainer.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = trainer.env.time_manager.max_steps
        print(colored("="*50, "blue", attrs=["bold"]))
        print(colored(f"RB-SAC-CEN Training ({num_eps} frames)", "blue", attrs=["bold"]))
        print(colored("="*50, "blue", attrs=["bold"]))

        for ep in range(num_eps):
            res = trainer.env.reset()
            obs_upper, prev_lower_res = res["upper"], res["lower"]
            current_upper_state = self.build_upper_state(trainer, obs_upper)
            
            epoch_lower_steps = 0
            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)
                    # Get tasks
                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines  = trainer.workload_gen.generate_step()

                if len(t_idx) > 0:
                    n_idx, m_idx, task_agent_ids, all_probs = self.get_lower_actions(trainer, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    next_res = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    next_res["mean_field"] = self.compute_next_mean_field(trainer, task_agent_ids, s_idx, n_idx, m_idx, batch_sizes)

                    lower_train_metrics = self.store_lower_transitions(trainer, prev_lower_res, next_res, t_idx, s_idx, n_idx, m_idx, tasks_min_accuracy, task_deadlines, batch_sizes, all_probs)
                    epoch_lower_steps += 1
                    trainer.aggregator.record_step(next_res)
                    prev_lower_res = next_res

                    if self.lower_agent.memory.total_adds % trainer.config.hyper_neural["LOWER_TRAIN_FREQ"] == 0:
                        learn_res = self.lower_agent.learn()
                        if learn_res and isinstance(learn_res, dict):
                            trainer.aggregator.record_q_stats("Lower", learn_res.get("q_min", 0.0), learn_res.get("q_max", 0.0), learn_res.get("q_mean", 0.0))
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    next_res_upper = trainer.env.collect_upper_metrics()
                    ns_upper_state = self.build_upper_state(trainer, next_res_upper)

                    trainer.aggregator.add_upper(next_res_upper)
                    is_ep_done = (slot == max_slots - 1)
                    if trainer.total_lower_steps >= trainer.lower_stable_threshold / 10:
                        self.store_upper_transitions(trainer, current_upper_state, ns_upper_state, obs_upper,
                                                     next_res_upper, u_acts_matrix, is_ep_done)
                        trainer.total_upper_steps += 1

                        # train upper
                    res = trainer.shared_upper_agent.learn(torch.arange(trainer.num_edge_agents, device=trainer.device))
                    if res is not None:
                        if isinstance(res, dict):
                            loss = res["loss"]
                            trainer.aggregator.record_q_stats("Edge_Group", res["q_min"], res["q_max"], res["q_mean"])
                        else:
                            loss = res
                        trainer.aggregator.record_td_losses(upper_losses=loss)

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
            trainer.update_rates(ep)
            trainer.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {trainer.total_lower_steps} | Upper Samples: {trainer.total_upper_steps}")
            print(f"Zeta Lower: {trainer.zeta_lower:.4f} | Zeta Upper: {trainer.zeta_upper:.4f}")
            print(f"Current Epsilon (Edge N0): {trainer.epsilons[0]:.4f}")


    def heuristic(
        self,
        trainer: Trainer,
        agent_ids,
        service_ids,
        probs,
        accuracies,
        deadlines=None
    ):
        num_nodes = trainer.num_nodes
        max_models = trainer.max_models

        # (num_services, max_models)
        model_acc = trainer.env.metadata["model_accuracies"]

        # valid models by accuracy
        # (B, max_models)
        model_mask = (
            model_acc[service_ids] >= accuracies.unsqueeze(-1)
        ).float()

        # expand to nodes
        # (B, num_nodes, max_models)
        model_mask = (
            model_mask.unsqueeze(1)
            .expand(-1, num_nodes, -1)
        )

        # flatten
        # (B, num_nodes*max_models)
        model_mask = model_mask.reshape(
            len(service_ids),
            -1
        )

        # group index
        indices = (
            agent_ids * trainer.num_services +
            service_ids
        )

        # (B, num_nodes*max_models)
        selected_probs = probs[indices]

        # apply mask
        selected_probs = selected_probs * model_mask

        # avoid all-zero rows
        row_sum = selected_probs.sum(dim=-1, keepdim=True)

        fallback_mask = (row_sum <= 1e-8)

        if fallback_mask.any():
            selected_probs[fallback_mask.squeeze(-1)] = (
                model_mask[fallback_mask.squeeze(-1)]
            )

            row_sum = selected_probs.sum(dim=-1, keepdim=True)

        selected_probs = (
            selected_probs /
            (row_sum + 1e-8)
        )

        action_ids = torch.multinomial(
            selected_probs,
            num_samples=1
        ).squeeze(-1)

        target_nodes = action_ids // max_models
        target_models = action_ids % max_models

        return target_nodes, target_models