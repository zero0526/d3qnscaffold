import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.agents import D3QNAgent
from matrix_source.configs import cfg
from matrix_source.utils.math_utils import to_binary, from_binary, one_hot
from matrix_source.visualize.aggregator import MetricsAggregator

class Trainer:
    def __init__(self):
        self.config = cfg
        self.device = cfg.hyper_neural.get('DEVICE', 'cpu')
        
        # 1. Initialize Environment & Workload
        self.env = MatrixSixGEnvironment(config=cfg, device=self.device)
        self.workload_gen = MatrixWorkloadGenerator(cfg, self.env.metadata)

        self.upper_agents: Dict[int, D3QNAgent] = {}
        self.lower_agents: Dict[int, D3QNAgent] = {}

        self.num_services = self.env.engine.num_services
        self.num_nodes = self.env.engine.num_nodes
        self.max_models = self.config.get('max_models_per_service', 5)

        # Upper State: Own node services status (S x 7 channels)
        self.upper_state_dim = self.num_services * 7
        
        # Lower State: Task (S) + All Nodes resources (M x 7)
        self.lower_state_dim = self.num_services + (self.num_nodes * 7)

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes * self.max_models

        # --- Training Hyperparams ---
        self.min_epsilon = cfg.hyper_neural.get("EPSILON", 0.05)
        self.epsilon_decay = cfg.hyper_neural.get("EPSILON_DECAY", 0.9985)
        self.epsilons = {nid: 1.0 for nid in range(self.num_nodes)}
        self.lower_epsilons = {tid: 1.0 for tid in range(self.workload_gen.num_terminals)}
        self.zeta = cfg.hyper_neural.get("ZETA", 1.0)

        self.aggregator = MetricsAggregator()
        self.__init_agents()

    def __init_agents(self):
        for nid in range(self.num_nodes):
            self.upper_agents[nid] = D3QNAgent(
                node_id=nid, node_type="Edge",
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

        for tid in range(self.workload_gen.num_terminals):
            self.lower_agents[tid] = D3QNAgent(
                node_id=tid, node_type="Terminal",
                state_dim=self.lower_state_dim,
                action_dim=self.lower_action_dim,
                u_action_dim=self.lower_action_dim,
                mf_hidden_sizes=tuple(self.config.hyper_neural["MF_HIDDEN_LAYER"]),
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
            self.env.reset()
            obs = self.env.get_observation()
            
            for slot in range(max_slots):
                # 1. Upper Level
                if self.env.time_manager.is_new_frame():
                    u_acts = self.get_upper_actions(obs)
                    u_act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
                    for nid, a in u_acts.items():
                        u_act_matrix[nid] = torch.tensor(a, device=self.device)
                    self.env.step_upper(u_act_matrix)

                # 2. Lower Level: Scheduling
                t_idx, s_idx = self.workload_gen.generate_step()
                sched_acts, lower_states = self.get_lower_actions_with_states(t_idx, s_idx, obs)
                
                n_idx = torch.tensor([a[0] for a in sched_acts], device=self.device)
                m_idx = torch.tensor([a[1] for a in sched_acts], device=self.device)
                
                results = self.env.step_lower(t_idx, s_idx, n_idx, m_idx)
                next_obs = self.env.get_observation()
                
                # 3. Learning
                self.store_batch_transitions(obs, next_obs, results, t_idx, s_idx, n_idx, m_idx, lower_states)
                
                obs = next_obs
                if slot == max_slots - 1: # Fake done for now
                    break

            self.update_rates(ep)
            # Optional: learn after episode
            for agent in list(self.upper_agents.values()) + list(self.lower_agents.values()):
                agent.learn()

    def get_upper_actions(self, obs):
        actions = {}
        for nid, agent in self.upper_agents.items():
            s = obs[nid].flatten().cpu().numpy()
            mf = np.zeros(1)
            a_id = agent.choose_action(s, mf, self.epsilons[nid], self.zeta)
            actions[nid] = to_binary(a_id, self.num_services)
        return actions

    def get_lower_actions_with_states(self, t_idx, s_idx, obs):
        res_summary = obs.mean(dim=1).flatten().cpu().numpy()
        actions = []
        states = {}
        for tid_val, sid_val in zip(t_idx.tolist(), s_idx.tolist()):
            tid = int(tid_val)
            s_task = one_hot(sid_val, self.num_services)
            s = np.concatenate([s_task, res_summary])
            states[tid] = s
            mf = np.zeros(1)
            a_id = self.lower_agents[tid].choose_action(s, mf, self.lower_epsilons[tid], self.zeta)
            actions.append((a_id // self.max_models, a_id % self.max_models))
        return actions, states

    def store_batch_transitions(self, obs, next_obs, results, t_idx, s_idx, n_idx, m_idx, lower_states):
        # Global reward shared for simplicity in this refactor
        reward = results['reward'].item()
        done = False
        
        # Upper Storage
        for nid, agent in self.upper_agents.items():
            s = obs[nid].flatten().cpu().numpy()
            ns = next_obs[nid].flatten().cpu().numpy()
            # Recover action ID from env state? (Or track it)
            # For brevity, use a dummy or track from u_acts
            # self.upper_agents[nid].store_transition(s, np.zeros(1), np.zeros(1), 0, reward, ns, done)
            pass

        # Lower Storage
        next_res_summary = next_obs.mean(dim=1).flatten().cpu().numpy()
        for idx, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            # Reconstruct next state for terminal tid
            ns_task = one_hot(int(s_idx[idx]), self.num_services) # Same service for simplicity
            ns = np.concatenate([ns_task, next_res_summary])
            
            a_id = int(n_idx[idx]) * self.max_models + int(m_idx[idx])
            self.lower_agents[tid].store_transition(
                lower_states[tid], np.zeros(1), np.zeros(1), a_id, reward, ns, done
            )

    def update_rates(self, ep):
        for nid in self.epsilons:
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons:
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()