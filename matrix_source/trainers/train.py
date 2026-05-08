import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.configs.configs import cfg
from matrix_source.utils.math_utils import to_binary, one_hot
from matrix_source.visualize.aggregator import MetricsAggregator

class Trainer:
    def __init__(self):
        self.config = cfg
        self.device = cfg.hyper_neural.get('DEVICE', 'cpu')
        
        # 1. Initialize Environment & Workload
        self.env = MatrixSixGEnvironment(config=cfg, device=self.device)
        self.workload_gen = MatrixWorkloadGenerator(cfg, self.env.metadata, device=self.device)

        self.num_services = self.env.engine.num_services
        self.num_nodes = self.env.engine.num_nodes
        self.num_terminals = self.workload_gen.num_terminals
        self.max_models = self.env.metadata.get("max_models", 5)

        # State Dims
        self.upper_state_dim = self.num_services * 2
        self.lower_state_dim = 4 + (self.num_nodes * 2)

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes + self.max_models
        self.lower_u_action_dim = self.num_nodes * self.max_models

        # --- Hyperparams ---
        self.min_epsilon = cfg.hyper_neural.get("EPSILON", 0.05)
        self.epsilon_decay = cfg.hyper_neural.get("EPSILON_DECAY", 0.9985)
        self.epsilons = {nid: 1.0 for nid in range(self.num_nodes)}
        self.lower_epsilons = {tid: 1.0 for tid in range(self.num_terminals)}
        self.zeta_initial = cfg.hyper_neural.get("ZETA", 1.0)
        self.zeta_max = cfg.hyper_neural.get("ZETA_MAX", 10.0)
        self.zeta_upper = self.zeta_initial
        self.zeta_lower = self.zeta_initial

        # Training control variables
        self.total_lower_steps = 0
        self.total_upper_steps = 0
        self.lower_stable_threshold = 50000
        
        self.aggregator = MetricsAggregator()
        self.upper_agents: Dict[int, D3QNAgent] = {}
        self.lower_agents: Dict[int, D3QNAgent] = {}
        self.__init_agents()

    def __init_agents(self):
        for nid in range(self.num_nodes):
            if nid in self.env.static_matrices["cloud_ids"]:
                continue
            self.upper_agents[nid] = D3QNAgent(
                node_id=nid, node_type="edge",
                state_dim=self.upper_state_dim,
                action_dim=self.upper_action_dim,
                u_action_dim=self.upper_u_action_dim,
                mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
                mf_lr=float(self.config.hyper_neural['MF_LR']),
                hidden_sizes=self.config.hyper_neural['AGENT_HIDDEN_LAYER'],
                lr=float(self.config.hyper_neural['UPPER_LR']),
                gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
                alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
                buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
                batch_size=self.config.hyper_neural['BATCH_SIZE']
            )

        for tid in range(self.num_terminals):
            self.lower_agents[tid] = D3QNAgent(
                node_id=tid, node_type="Terminal",
                state_dim=self.lower_state_dim,
                action_dim=self.lower_action_dim,
                u_action_dim=self.lower_u_action_dim,
                mf_hidden_sizes=(32, 32),
                mf_lr=float(self.config.hyper_neural['MF_LR']),
                hidden_sizes=tuple(self.config.hyper_neural['AGENT_HIDDEN_LAYER']),
                lr=float(self.config.hyper_neural['LOWER_LR']),
                gamma=self.config.hyper_neural['DISCOUNT_FACTOR'],
                alpha=float(self.config.hyper_neural['UPDATE_TARGET_COEF']),
                buffer_size=self.config.hyper_neural['MEMORY_SIZE'],
                batch_size=self.config.hyper_neural['BATCH_SIZE']
            )

    def train(self):
        num_eps = self.config.hyper_neural['NUMOF_TRAIN_EP']
        max_slots = self.env.time_manager.max_steps

        for ep in tqdm(range(num_eps), desc="Training"):
            obs = self.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            current_upper_state = self.get_upper_state(obs_upper) 
            
            for slot in range(max_slots):
                if self.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(current_upper_state, obs_upper)
                    self.env.step_upper(u_acts_matrix)

                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = self.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    results = self.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    self.store_lower_transitions(prev_lower_res, results, t_idx, s_idx, n_idx, m_idx)
                    prev_lower_res = results
                    self.total_lower_steps += 1 # Count samples collected

                    #     train lower
                    lower_td_losses = []
                    for agent in self.lower_agents.values():
                        loss = agent.learn() # D3QNAgent checks min_batch_size internally
                        if loss is not None:
                            lower_td_losses.append(loss)
                    if lower_td_losses:
                        self.aggregator.record_td_losses(lower_losses=lower_td_losses)
                else:
                    self.env.time_manager.tick()

                if self.env.time_manager.is_new_frame():
                    res_upper = self.env.collect_upper_metrics()
                    next_upper_state = self.get_upper_state(res_upper)

                    is_ep_done = (slot == max_slots - 1)
                    
                    # Phased Curriculum: Only collect and train Upper after Lower is stable
                    if self.total_lower_steps >= self.lower_stable_threshold:
                        self.store_upper_transitions(current_upper_state, next_upper_state, obs_upper, res_upper, u_acts_matrix, is_ep_done)
                        self.total_upper_steps += 1 # Approximate upper samples

                        # train upper
                        upper_td_losses = []
                        for agent in self.upper_agents.values():
                            loss = agent.learn()
                            if loss is not None:
                                upper_td_losses.append(loss)
                        if upper_td_losses:
                            self.aggregator.record_td_losses(upper_losses=upper_td_losses)

                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            # update ep and history
            self.update_rates(ep)
            self.aggregator.store_history()
            self.aggregator.report_episode(ep)
            print(f"--- Global Metrics ---")
            print(f"Lower Samples: {self.total_lower_steps} | Upper Samples: {self.total_upper_steps}")
            print(f"Zeta Lower: {self.zeta_lower:.4f} | Zeta Upper: {self.zeta_upper:.4f}")
            print(f"Current Epsilon (Edge N0): {self.epsilons[0]:.4f}")

    def get_upper_state(self, obs_upper):
        state = torch.cat([obs_upper['actions'], obs_upper['phi_prob']], dim=-1)
        
        # Apply Normalization
        norm_factors = self.config.normalization.get("upper_state", {}).get("features", {})
        if norm_factors:
            for idx, divisor in norm_factors.items():
                idx_int = int(idx)
                if idx_int < state.shape[-1]:
                    state[..., idx_int] /= (divisor if divisor != 0 else 1.0)
        return state

    def get_upper_actions(self, current_upper_state, obs_upper):
        act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        
        for nid, agent in self.upper_agents.items():
            s = current_upper_state[nid]
            mf = mf_global[nid]
            a_id = agent.choose_action(s, mf, self.epsilons[nid], self.zeta_upper)
            act_matrix[nid] = torch.tensor(to_binary(a_id, self.num_services), device=self.device)
        
        for nid in self.env.static_matrices["cloud_ids"]:
            act_matrix[nid] = torch.ones(self.num_services, device=self.device)
        return act_matrix

    def get_lower_actions(self, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field']
        
        node_indices = torch.zeros_like(t_idx)
        model_indices = torch.zeros_like(t_idx)
        
        meta = self.env.metadata
        unit_sizes = meta['service_input_size']
        service_omega = meta['service_omega']
        placement_matrix = self.env.engine.placement_matrix
        model_accs = meta['model_accuracies']

        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            sid = int(s_idx[i])
            
            data_size = (batch_sizes[i] * unit_sizes[sid]).item()
            s_task = torch.tensor([
                data_size, tasks_min_accuracy[i], task_deadlines[i], service_omega[sid]
            ], device=self.device)
            
            s = torch.cat([s_task, obs_dict['backlog'][:, sid], obs_dict['cpu_alloc'][:, sid]])
            
            # Apply Normalization
            norm_factors = self.config.normalization.get("lower_state", {}).get("features", {})
            if norm_factors:
                for idx, divisor in norm_factors.items():
                    idx_int = int(idx)
                    if idx_int < s.shape[-1]:
                        s[idx_int] /= (divisor if divisor != 0 else 1.0)
            
            p_mask = placement_matrix[:, sid] > 0
            a_mask = model_accs[sid, :] >= tasks_min_accuracy[i]
            mask = torch.outer(p_mask.float(), a_mask.float()).flatten()
            
            if not mask.any(): mask = torch.ones_like(mask)

            a_id = self.lower_agents[tid].choose_action(s, mf_terminals[tid], self.lower_epsilons[tid], self.zeta_lower, mask=mask)
            
            node_indices[i] = a_id // self.max_models
            model_indices[i] = a_id % self.max_models
            
        return node_indices, model_indices

    def store_lower_transitions(self, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        reward = next_res['reward']
        done = next_res["new_frame"]
        
        c_obs, n_obs = current_res['obs'], next_res['obs']
        c_mf, n_mf = current_res['mean_field'], next_res['mean_field']

        mf_losses = []
        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            sid = int(s_idx[i])
            
            s = torch.cat([c_obs['task_reqs'][tid], c_obs['backlog'][:, sid], c_obs['cpu_alloc'][:, sid]])
            ns = torch.cat([n_obs['task_reqs'][tid], n_obs['backlog'][:, sid], n_obs['cpu_alloc'][:, sid]])
            
            # Apply Normalization to States
            norm_factors = self.config.normalization.get("lower_state", {}).get("features", {})
            if norm_factors:
                for idx, divisor in norm_factors.items():
                    idx_int = int(idx)
                    d = (divisor if divisor != 0 else 1.0)
                    if idx_int < s.shape[-1]: s[idx_int] /= d
                    if idx_int < ns.shape[-1]: ns[idx_int] /= d

            # Normalize Reward
            rew_divisor = self.config.normalization.get("rewards", {}).get("lower_divisor", 1.0)
            normalized_reward = reward / (rew_divisor if rew_divisor != 0 else 1.0)
            
            a_id = int(n_idx[i] * self.max_models + m_idx[i])
            loss = self.lower_agents[tid].store_transition_train_mf(s, c_mf[tid], n_mf[tid], a_id, normalized_reward, ns, done)
            if loss is not None:
                mf_losses.append(loss)
        
        avg_mf_loss = np.mean(mf_losses) if mf_losses else None
        
        # Pass a sample state for feature distribution analysis
        sample_state = None
        if t_idx.tolist():
            tid = int(t_idx[0])
            sid = int(s_idx[0])
            sample_state = torch.cat([c_obs['task_reqs'][tid], c_obs['backlog'][:, sid], c_obs['cpu_alloc'][:, sid]])
            
        self.aggregator.add_lower(next_res, mf_loss=avg_mf_loss, state=sample_state)

    def store_upper_transitions(self, s_all, ns_all, current_res, next_res, acts_matrix, done):
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        n_mf = next_res['mean_fields']

        mf_losses = []
        for nid in self.upper_agents:
            s, ns = s_all[nid], ns_all[nid]
            a_binary = acts_matrix[nid].cpu().numpy().astype(int)
            a_id = 0
            for bit in a_binary: a_id = (a_id << 1) | bit

            # Normalize Reward
            rew_divisor = self.config.normalization.get("rewards", {}).get("upper_divisor", 1.0)
            normalized_reward = reward / (rew_divisor if rew_divisor != 0 else 1.0)

            loss = self.upper_agents[nid].store_transition_train_mf(s, c_mf[nid], n_mf[nid], a_id, normalized_reward, ns, done)
            if loss is not None:
                mf_losses.append(loss)
        
        avg_mf_loss = np.mean(mf_losses) if mf_losses else None
        
        # Pass a sample state (first node)
        sample_state = s_all[0] if len(s_all) > 0 else None
            
        self.aggregator.add_upper(next_res, mf_loss=avg_mf_loss, state=sample_state)

    def update_rates(self, ep):
        # 1. Update Epsilons
        for nid in self.epsilons: 
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons: 
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)
            
        # 2. Phased Zeta Annealing
        num_eps = self.config.hyper_neural.get('NUMOF_TRAIN_EP', 3000)
        
        # Lower Zeta: Increases from 20k to 50k samples
        fraction = min(1.0, ep / num_eps)
        if self.total_lower_steps < 20000:
            self.zeta_lower = self.zeta_initial
        elif self.total_lower_steps < self.lower_stable_threshold:
            # Fast increase while lower is stabilizing (20k to 50k)
            bump_factor = min(1.0, (self.total_lower_steps - 20000) / (self.lower_stable_threshold - 20000))
            target = self.zeta_initial + (self.zeta_max * 0.5 - self.zeta_initial) * bump_factor
            self.zeta_lower = max(self.zeta_lower, target)
        else:
            # Slow increase afterwards
            self.zeta_lower = self.zeta_initial + (self.zeta_max - self.zeta_initial) * fraction
            
        # Upper Zeta: Only increases AFTER lower is stable and upper has enough valid samples
        if self.total_lower_steps >= self.lower_stable_threshold and self.total_upper_steps > 5000:
            upper_fraction = min(1.0, (ep) / num_eps) # Simplify scaling
            self.zeta_upper = self.zeta_initial + (self.zeta_max - self.zeta_initial) * upper_fraction
        else:
            self.zeta_upper = self.zeta_initial # Remains low (exploration mode)

if __name__ == "__main__":
    Trainer().train()