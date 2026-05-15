import torch
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.trainers.strategies import AlgorithmStrategy
from matrix_source.utils.math_utils import to_binary
from tqdm import tqdm

class D3QNStrategy(AlgorithmStrategy):
    def initialize_agents(self, trainer):
        # Upper Agent
        trainer.shared_upper_agent = D3QNAgent(
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

        # Lower Agent
        trainer.shared_lower_agent = D3QNAgent(
            node_id=-1, node_type="Terminal_Group",
            state_dim=trainer.lower_state_dim,
            action_dim=trainer.lower_action_dim,
            u_action_dim=trainer.lower_u_action_dim,
            mf_hidden_sizes=tuple(trainer.config.hyper_neural["MF_HIDDEN_LAYER"]),
            mf_lr=float(trainer.config.hyper_neural['MF_LR']),
            buffer_min_size=float(trainer.config.hyper_neural["BUFFER_MIN_SIZE"][1]),
            hidden_sizes=tuple(trainer.config.hyper_neural['AGENT_HIDDEN_LAYER']),
            lr=float(trainer.config.hyper_neural['LOWER_LR']),
            gamma=trainer.config.hyper_neural['DISCOUNT_FACTOR'],
            alpha=float(trainer.config.hyper_neural['UPDATE_TARGET_COEF']),
            buffer_size=trainer.config.hyper_neural['MEMORY_SIZE'],
            batch_size=trainer.config.hyper_neural['BATCH_SIZE'],
            num_instances=trainer.num_terminals,
            device=trainer.device,
            logs_q=False
        )

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        act_matrix = torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        edge_states = current_upper_state[trainer.edge_node_ids]
        edge_mfs = mf_global[trainer.edge_node_ids]
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)
        
        batch_a_ids = trainer.shared_upper_agent.choose_action_batch(
            edge_states, edge_mfs, trainer.zeta_upper, agent_indices=instance_indices
        )
        
        for i, nid in enumerate(trainer.edge_node_ids):
            act_matrix[nid] = torch.tensor(to_binary(batch_a_ids[i], trainer.num_services), device=trainer.device)
        
        for nid in trainer.env.static_matrices.get("cloud_ids", []):
            act_matrix[nid] = torch.ones(trainer.num_services, device=trainer.device)
        return act_matrix

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        meta = trainer.env.metadata
        unit_sizes = meta['service_input_size']
        
        data_sizes = batch_sizes * unit_sizes[s_idx].squeeze(-1)
        s_tasks = torch.stack([data_sizes, tasks_min_accuracy, task_deadlines, meta['service_omega'][s_idx].squeeze(-1)], dim=1).float()
        s_backlogs = obs_dict['backlog'][:, s_idx].T
        s_cpus = obs_dict['cpu_alloc'][:, s_idx].T
        states = torch.cat([s_tasks, s_backlogs, s_cpus], dim=1)
        
        states[:, 0] /= trainer.config.norm_data_size
        states[:, 1] /= 100.0
        if states.shape[1] > 4:
            states[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            
        masks = torch.ones((len(t_idx), trainer.num_nodes * trainer.max_models), device=trainer.device)
        mfs = mf_terminals[t_idx]
        
        batch_actions = trainer.shared_lower_agent.choose_action_batch(
            states, mfs, trainer.zeta_lower, masks_batch=masks, agent_indices=torch.arange(trainer.num_terminals, device=trainer.device)
        )
        
        a_ids = torch.tensor(batch_actions, device=trainer.device)
        return a_ids // trainer.max_models, a_ids % trainer.max_models

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        from matrix_source.trainers.train import log_transform
        reward = next_res['reward']
        done = torch.tensor([next_res["new_frame"]]*len(t_idx), dtype=torch.float32, device=trainer.device)
        c_obs, n_obs = current_res['obs'], next_res['obs']
        c_mf, n_mf = current_res['mean_field'], next_res['mean_field']
        
        def build_state(obs, tidx, sidx):
            st = torch.cat([obs['task_reqs'][tidx], obs['backlog'][:, sidx].T, obs['cpu_alloc'][:, sidx].T], dim=1)
            st[:, 0] /= trainer.config.norm_data_size
            st[:, 1] /= 100.0
            if st.shape[1] > 4: st[:, 4:4+2*trainer.num_nodes] /= trainer.config.norm_gflop
            return st

        states = build_state(c_obs, t_idx, s_idx)
        next_states = build_state(n_obs, t_idx, s_idx)
        
        rew_divisor = trainer.config.norm_lower_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((len(t_idx),), norm_rew, dtype=torch.float32, device=trainer.device)
        a_ids = (n_idx * trainer.max_models + m_idx).long()
        
        avg_mf_loss = trainer.shared_lower_agent.store_transition_train_mf_batch(
            states, c_mf[t_idx], n_mf[t_idx], a_ids, rewards, next_states, done, agent_ids=t_idx
        )
        trainer.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=states[0] if len(states) > 0 else None)

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        from matrix_source.trainers.train import log_transform
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((trainer.num_nodes, trainer.num_services), device=trainer.device))
        n_mf = next_res['mean_fields']
        dones = torch.full((trainer.num_edge_agents,), 1.0 if is_done else 0.0, dtype=torch.float32, device=trainer.device)
        
        edge_states = s_all[trainer.edge_node_ids]
        edge_next_states = ns_all[trainer.edge_node_ids]
        edge_c_mfs = c_mf[trainer.edge_node_ids]
        edge_n_mfs = n_mf[trainer.edge_node_ids]
        edge_acts = acts_matrix[trainer.edge_node_ids]
        
        pw2 = 2 ** torch.arange(trainer.num_services - 1, -1, -1, device=trainer.device).float()
        edge_a_ids = (edge_acts * pw2).sum(dim=1).long()
        
        rew_divisor = trainer.config.norm_upper_rw
        norm_rew = log_transform(reward / (rew_divisor if rew_divisor != 0 else 1.0))
        rewards = torch.full((trainer.num_edge_agents,), norm_rew, dtype=torch.float32, device=trainer.device)
        instance_indices = torch.tensor([trainer.node_to_instance[nid] for nid in trainer.edge_node_ids], device=trainer.device)

        avg_mf_loss = trainer.shared_upper_agent.store_transition_train_mf_batch(
            edge_states, edge_c_mfs, edge_n_mfs, edge_a_ids, rewards, edge_next_states, dones, agent_ids=instance_indices
        )
        trainer.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=edge_states[0] if len(edge_states) > 0 else None)

    def run_training(self, trainer):
        num_eps = trainer.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = trainer.env.time_manager.max_steps

        for ep in tqdm(range(num_eps), desc="Training"):
            obs = trainer.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            current_upper_state = self.build_upper_state(trainer, obs_upper) 
            
            for slot in range(max_slots):
                if trainer.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(trainer, current_upper_state, obs_upper)
                    trainer.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = trainer.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(trainer, prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = trainer.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    self.store_lower_transitions(trainer, prev_lower_res, results, t_idx, s_idx, n_idx, m_idx)
                    
                    # Accumulate Metrics
                    trainer.aggregator.add_step_matrices(
                        f_alloc=trainer.env.engine.cpu_alloc_matrix,
                        arrivals=results['info']['arrival_matrix'],
                        backlog=trainer.env.engine.backlog_queue.sum(dim=-1)
                    )
                    
                    prev_lower_res = results
                    trainer.total_lower_steps += 1 
                    
                    # train lower
                    res = trainer.shared_lower_agent.learn(torch.arange(trainer.num_terminals, device=trainer.device))
                    if res is not None:
                        if isinstance(res, dict):
                            loss = res["loss"]
                            trainer.aggregator.record_q_stats("Terminal_Group", res["q_min"], res["q_max"], res["q_mean"])
                        else:
                            loss = res
                        trainer.aggregator.record_td_losses(lower_losses=loss)
                else:
                    trainer.env.time_manager.tick()

                if trainer.env.time_manager.is_new_frame():
                    res_upper = trainer.env.collect_upper_metrics()
                    next_upper_state = self.build_upper_state(trainer, res_upper)
                    
                    trainer.aggregator.add_upper(res_upper)

                    is_ep_done = (slot == max_slots - 1)
                    
                    if trainer.total_lower_steps >= trainer.lower_stable_threshold/10:
                        self.store_upper_transitions(trainer, current_upper_state, next_upper_state, obs_upper, res_upper, u_acts_matrix, is_ep_done)
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

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            # update ep and history
            trainer.update_rates(ep)
            trainer.aggregator.store_history()
            trainer.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {trainer.total_lower_steps} | Upper Samples: {trainer.total_upper_steps}")
            print(f"Zeta Lower: {trainer.zeta_lower:.4f} | Zeta Upper: {trainer.zeta_upper:.4f}")
            print(f"Current Epsilon (Edge N0): {trainer.epsilons[0]:.4f}")

