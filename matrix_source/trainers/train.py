import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.agents import D3QNAgent
from matrix_source.configs import cfg
from matrix_source.utils.math_utils import to_binary, one_hot
from matrix_source.visualize.aggregator import MetricsAggregator

class Trainer:
    def __init__(self):
        self.config = cfg
        self.device = cfg.hyper_neural.get('DEVICE', 'cpu')
        
        # 1. Initialize Environment & Workload
        self.env = MatrixSixGEnvironment(config=cfg, device=self.device)
        self.workload_gen = MatrixWorkloadGenerator(cfg, self.env.metadata)

        self.num_services = self.env.engine.num_services
        self.num_nodes = self.env.engine.num_nodes
        self.num_terminals = self.workload_gen.num_terminals
        self.max_models = self.env.metadata.get("max_models", 5)

        # Upper State: Node-level view of services (S x 7 channels)
        self.upper_state_dim = self.num_services * 7
        
        # Lower State Dim: task_req(4) + global_backlog(M*S) + global_alloc(M*S)
        self.lower_state_dim = 4 + (self.num_nodes * self.num_services * 2)

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes * self.max_models

        # --- Training Hyperparams ---
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
                u_action_dim=self.lower_action_dim,
                mf_hidden_sizes=(32, 32), # MF for terminals
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
        max_slots = self.config.get('slots_per_episode', 100)

        for ep in tqdm(range(num_eps), desc="Training"):
            # Initial resets
            res_upper = self.env.reset()
            res_lower = self.env.engine.reset_lower()
            
            for slot in range(max_slots):
                obs_v = self.env.get_observation()
                
                # 1. Upper Level Decision (Timeframe start)
                if self.env.time_manager.is_new_frame():
                    u_acts_matrix = self.get_upper_actions(obs_v, res_upper)
                    res_upper = self.env.step_upper(u_acts_matrix) # Summary of PREV frame
                
                # 2. Lower Level Decision
                t_idx, s_idx, batch_sizes = self.workload_gen.generate_step()
                n_idx, m_idx = self.get_lower_actions(res_lower, t_idx)
                
                # 3. Environment Step
                results = self.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx)
                
                # 4. Learning & Storage (Lower)
                self.store_lower_transitions(res_lower, results, t_idx, n_idx, m_idx)
                
                res_lower = results
                if slot == max_slots - 1:
                    break

            self.update_rates(ep)
            # Parallel learn
            for agent in list(self.upper_agents.values()) + list(self.lower_agents.values()):
                agent.learn()

    def get_upper_actions(self, obs_v, res_upper):
        act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        mf_global = res_upper['mean_fields'].cpu().numpy()
        
        for nid, agent in self.upper_agents.items():
            s = obs_v[nid].flatten().cpu().numpy()
            mf = mf_global[nid]
            a_id = agent.choose_action(s, mf, self.epsilons[nid], self.zeta)
            act_matrix[nid] = torch.tensor(to_binary(a_id, self.num_services), device=self.device)
        return act_matrix

    def get_lower_actions(self, res_lower, t_idx):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field'].cpu().numpy()
        
        node_indices = torch.zeros_like(t_idx)
        model_indices = torch.zeros_like(t_idx)

        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            s_task = obs_dict['task_reqs'][tid].cpu().numpy()
            s_back = obs_dict['backlog_prev'].flatten().cpu().numpy()
            s_allo = obs_dict['cpu_alloc_prev'].flatten().cpu().numpy()
            s = np.concatenate([s_task, s_back, s_allo])
            
            mf = mf_terminals[tid]
            a_id = self.lower_agents[tid].choose_action(s, mf, self.lower_epsilons[tid], self.zeta)
            
            node_indices[i] = a_id // self.max_models
            model_indices[i] = a_id % self.max_models
            
        return node_indices, model_indices

    def store_lower_transitions(self, current_res, next_res, t_idx, n_idx, m_idx):
        reward = next_res['reward']
        done = False
        
        c_obs = current_res['obs']
        n_obs = next_res['obs']
        c_mf = current_res['mean_field'].cpu().numpy()
        n_mf = next_res['mean_field'].cpu().numpy()
        
        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            s = np.concatenate([
                c_obs['task_reqs'][tid].cpu().numpy(),
                c_obs['backlog_prev'].flatten().cpu().numpy(),
                c_obs['cpu_alloc_prev'].flatten().cpu().numpy()
            ])
            ns = np.concatenate([
                n_obs['task_reqs'][tid].cpu().numpy(),
                n_obs['backlog_prev'].flatten().cpu().numpy(),
                n_obs['cpu_alloc_prev'].flatten().cpu().numpy()
            ])
            
            a_id = int(n_idx[i] * self.max_models + m_idx[i])
            self.lower_agents[tid].store_transition(s, c_mf[tid], n_mf[tid], a_id, reward, ns, done)

    def update_rates(self, ep):
        for nid in self.epsilons:
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons:
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()