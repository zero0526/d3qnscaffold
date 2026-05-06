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

        # Upper State: Node-level view of services (S x 2 channels)
        self.upper_state_dim = self.num_services * 2
        
        # Lower State Dim: task_req(4) + backlog(M) + alloc(M) across all nodes for the specific service
        self.lower_state_dim = 4 + (self.num_nodes * 2)

        self.upper_action_dim = self.num_services
        self.upper_u_action_dim = 1 << self.num_services
        self.lower_action_dim = self.num_nodes + self.max_models
        self.lower_u_action_dim = self.num_nodes * self.max_models


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
        max_slots = self.env.time_manager.max_steps

        for ep in tqdm(range(num_eps), desc="Training"):
            # 1. Unified Reset
            obs = self.env.reset()
            obs_upper = obs['upper']
            prev_lower_res = obs['lower']
            
            # Initial state for Upper Level
            current_upper_state = self.get_upper_state(obs_upper) 
            
            for slot in range(max_slots):
                # 1. Upper Level Decision (Timeframe start)
                if self.env.time_manager.is_new_frame():
                    # Select and execute actions for THIS timeframe
                    u_acts_matrix = self.get_upper_actions(current_upper_state, obs_upper)
                    self.env.step_upper(u_acts_matrix)

                # 2. Lower Level Decision
                t_idx, s_idx, batch_sizes, tasks_min_accuracy, task_deadlines = self.workload_gen.generate_step()
                if len(t_idx) > 0:
                    n_idx, m_idx = self.get_lower_actions(prev_lower_res, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes)
                    
                    # 3. Environment Step
                    results = self.env.step_lower(t_idx, s_idx, batch_sizes, n_idx, m_idx, task_deadlines, tasks_min_accuracy)
                    
                    # 4. Storage (Lower)
                    self.store_lower_transitions(prev_lower_res, results, t_idx, s_idx, n_idx, m_idx)
                    prev_lower_res = results
                else:
                    self.env.time_manager.tick()

                # 5. End of Timeframe Handling
                if self.env.time_manager.is_new_frame():
                    # Collect metrics for the timeframe that just finished
                    res_upper = self.env.collect_upper_metrics()
                    next_upper_state = self.get_upper_state(res_upper)
                    
                    # Store timeframe transition
                    is_ep_done = (slot == max_slots - 1)
                    self.store_upper_transitions(
                        current_upper_state, next_upper_state, 
                        obs_upper, res_upper, u_acts_matrix, is_ep_done
                    )
                    
                    # Prepare for next frame
                    current_upper_state = next_upper_state
                    obs_upper = res_upper

                if slot == max_slots - 1:
                    break

            self.update_rates(ep)
            # Parallel learn
            for agent in list(self.upper_agents.values()) + list(self.lower_agents.values()):
                agent.learn()

    def get_upper_state(self, obs_upper):
        """Construct state as [Previous Actions, Phi Probability]"""
        prev_acts = obs_upper['actions'] # (M, S)
        phi_prob = obs_upper['phi_prob'] # (M, S)
        return torch.cat([prev_acts, phi_prob], dim=-1)

    def get_upper_actions(self, current_upper_state, obs_upper):
        act_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        mf_global = obs_upper.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        if isinstance(mf_global, torch.Tensor):
            mf_global = mf_global.cpu().numpy()
        
        for nid, agent in self.upper_agents.items():
            s = current_upper_state[nid].cpu().numpy()
            mf = mf_global[nid]
            a_id = agent.choose_action(s, mf, self.epsilons[nid], self.zeta)
            act_matrix[nid] = torch.tensor(to_binary(a_id, self.num_services), device=self.device)
        # cloud always run all services
        for nid in self.env.static_matrices["cloud_ids"]:
            act_matrix[nid] = torch.ones(self.num_services, device=self.device)
        return act_matrix

    def get_lower_actions(self, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        obs_dict = res_lower['obs']
        mf_terminals = res_lower['mean_field'].cpu().numpy()
        
        node_indices = torch.zeros_like(t_idx)
        model_indices = torch.zeros_like(t_idx)
        
        # Get metadata and current placement for masking
        meta = self.env.metadata
        unit_sizes = meta['service_input_size']
        service_omega = meta['service_omega']
        placement_matrix = self.env.engine.placement_matrix
        model_accs = meta['model_accuracies']

        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            sid = int(s_idx[i])
            
            # Construct s_task: [data_size, dl, acc, type]
            data_size = (batch_sizes[i] * unit_sizes[sid]).item()
            s_task = np.array([
                data_size, 
                tasks_min_accuracy[i].item(), 
                task_deadlines[i].item(), 
                service_omega[sid].item()
            ])
            s_back = obs_dict['backlog'][:, sid].cpu().numpy()
            s_allo = obs_dict['cpu_alloc'][:, sid].cpu().numpy()
            s = np.concatenate([s_task, s_back, s_allo])
            
            # --- ACTION MASKING ---
            # 1. Placement Mask: Only nodes where sid is deployed
            p_mask = placement_matrix[:, sid] > 0
            
            # 2. Accuracy Mask: Only models meeting min requirement
            target_acc = tasks_min_accuracy[i]
            a_mask = model_accs[sid, :] >= target_acc
            
            # 3. Combine to form a (Node x Model) mask
            # mask_2d[n, m] = 1.0 if (node n is valid AND model m is valid)
            mask_2d = torch.outer(p_mask.float(), a_mask.float())
            mask_flat = mask_2d.flatten().cpu().numpy()
            
            # Fallback: if no valid actions (all zero), allow all to avoid NaN in softmax
            if not np.any(mask_flat):
                mask_flat = np.ones_like(mask_flat)

            mf = mf_terminals[tid]
            a_id = self.lower_agents[tid].choose_action(s, mf, self.lower_epsilons[tid], self.zeta, mask=mask_flat)
            
            node_indices[i] = a_id // self.max_models
            model_indices[i] = a_id % self.max_models
            
        return node_indices, model_indices

    def store_lower_transitions(self, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        reward = next_res['reward']
        done = next_res["new_frame"]
        
        c_obs = current_res['obs']
        n_obs = next_res['obs']
        c_mf = current_res['mean_field'].cpu().numpy()
        n_mf = next_res['mean_field'].cpu().numpy()

        for i, tid_val in enumerate(t_idx.tolist()):
            tid = int(tid_val)
            sid = int(s_idx[i])
            
            s = np.concatenate([
                c_obs['task_reqs'][tid].cpu().numpy(),
                c_obs['backlog'][:, sid].cpu().numpy(),
                c_obs['cpu_alloc'][:, sid].cpu().numpy()
            ])
            ns = np.concatenate([
                n_obs['task_reqs'][tid].cpu().numpy(),
                n_obs['backlog'][:, sid].cpu().numpy(),
                n_obs['cpu_alloc'][:, sid].cpu().numpy()
            ])
            
            a_id = int(n_idx[i] * self.max_models + m_idx[i])
            self.lower_agents[tid].store_transition(s, c_mf[tid], n_mf[tid], a_id, reward, ns, done)

    def store_upper_transitions(self, s_all, ns_all, current_res, next_res, acts_matrix, done):
        reward = next_res['reward_global']
        c_mf = current_res.get('mean_fields', torch.zeros((self.num_nodes, self.num_services), device=self.device))
        if isinstance(c_mf, torch.Tensor): c_mf = c_mf.cpu().numpy()
        
        n_mf = next_res['mean_fields']
        if isinstance(n_mf, torch.Tensor): n_mf = n_mf.cpu().numpy()

        for nid in range(self.num_nodes):
            if nid not in self.env.static_matrices["cloud_ids"]:
                s = s_all[nid].cpu().numpy()
                ns = ns_all[nid].cpu().numpy()

                # Action decoding
                a_binary = acts_matrix[nid].cpu().numpy().astype(int)
                a_id = 0
                for bit in a_binary:
                    a_id = (a_id << 1) | bit

                self.upper_agents[nid].store_transition(s, c_mf[nid], n_mf[nid], a_id, reward, ns, done)

    def update_rates(self, ep):
        for nid in self.epsilons:
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons:
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()