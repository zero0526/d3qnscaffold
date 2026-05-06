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
        self.zeta = cfg.hyper_neural.get("ZETA", 1.0)

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
                else:
                    self.env.time_manager.tick()

                if self.env.time_manager.is_new_frame():
                    res_upper = self.env.collect_upper_metrics()
                    next_upper_state = self.get_upper_state(res_upper)
                    
                    is_ep_done = (slot == max_slots - 1)
                    self.store_upper_transitions(current_upper_state, next_upper_state, obs_upper, res_upper, u_acts_matrix, is_ep_done)
                    
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

            self.update_rates(ep)
            for agent in list(self.upper_agents.values()) + list(self.lower_agents.values()):
                agent.learn()

    def get_upper_state(self, obs_upper):
        return torch.cat([obs_upper['actions'], obs_upper['phi_prob']], dim=-1)

    def get_upper_actions(self, current_upper_state, obs_upper):
        act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        
        for nid, agent in self.upper_agents.items():
            s = current_upper_state[nid]
            mf = mf_global[nid]
            a_id = agent.choose_action(s, mf, self.epsilons[nid], self.zeta)
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
            
            p_mask = placement_matrix[:, sid] > 0
            a_mask = model_accs[sid, :] >= tasks_min_accuracy[i]
            mask = torch.outer(p_mask.float(), a_mask.float()).flatten()
            
            if not mask.any(): mask = torch.ones_like(mask)

            a_id = self.lower_agents[tid].choose_action(s, mf_terminals[tid], self.lower_epsilons[tid], self.zeta, mask=mask)
            
            node_indices[i] = a_id // self.max_models
            model_indices[i] = a_id % self.max_models
            
        return node_indices, model_indices

    def store_lower_transitions(self, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        reward = next_res['reward']
        done = next_res["new_frame"]
        
        c_obs, n_obs = current_res['obs'], next_res['obs']
        c_mf, n_mf = current_res['mean_field'], next_res['mean_field']

        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            sid = int(s_idx[i])
            
            s = torch.cat([c_obs['task_reqs'][tid], c_obs['backlog'][:, sid], c_obs['cpu_alloc'][:, sid]])
            ns = torch.cat([n_obs['task_reqs'][tid], n_obs['backlog'][:, sid], n_obs['cpu_alloc'][:, sid]])
            
            a_id = int(n_idx[i] * self.max_models + m_idx[i])
            self.lower_agents[tid].store_transition_train_mf(s, c_mf[tid], n_mf[tid], a_id, reward, ns, done)

    def store_upper_transitions(self, s_all, ns_all, current_res, next_res, acts_matrix, done):
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        n_mf = next_res['mean_fields']

        for nid in self.upper_agents:
            s, ns = s_all[nid], ns_all[nid]
            a_binary = acts_matrix[nid].cpu().numpy().astype(int)
            a_id = 0
            for bit in a_binary: a_id = (a_id << 1) | bit

            self.upper_agents[nid].store_transition_train_mf(s, c_mf[nid], n_mf[nid], a_id, reward, ns, done)

    def update_rates(self, ep):
        for nid in self.epsilons: self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons: self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)

if __name__ == "__main__":
    Trainer().train()