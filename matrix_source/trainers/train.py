import torch
import numpy as np
from tqdm import tqdm
from typing import Dict

from matrix_source.envs.matrix_env import MatrixSixGEnvironment
from matrix_source.envs.workload_generator import MatrixWorkloadGenerator
from matrix_source.configs.configs import cfg
from matrix_source.visualize.aggregator import MetricsAggregator
from matrix_source.trainers.ppo_stategy import PPOStrategy

class Trainer:
    def __init__(self, strategy=None):
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
        self.upper_state_dim = (self.num_services * 2)
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
        self.lower_stable_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][0]*10
        self.lower_start_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][1]
        
        self.aggregator = MetricsAggregator()
        self.shared_upper_agent = None
        self.shared_lower_agent = None
        
        # Track which nodes are edge agents (to map to weight indices)
        self.edge_node_ids = [nid for nid in range(self.num_nodes) 
                             if nid not in self.env.static_matrices.get("cloud_ids", [])]
        self.node_to_instance = {nid: i for i, nid in enumerate(self.edge_node_ids)}
        self.num_edge_agents = len(self.edge_node_ids)

        # 2. Strategy Injection
        self.strategy = strategy if strategy is not None else PPOStrategy()
        self.strategy.initialize_agents(self)

    def train(self):
        self.strategy.run_training(self)

    def update_rates(self, ep):
        # 1. Update Epsilons
        for nid in self.epsilons: 
            self.epsilons[nid] = max(self.min_epsilon, self.epsilons[nid] * self.epsilon_decay)
        for tid in self.lower_epsilons: 
            self.lower_epsilons[tid] = max(self.min_epsilon, self.lower_epsilons[tid] * self.epsilon_decay)
            
        # 2. Phased Zeta Annealing
        fraction = min(1.0, ep / self.config.hyper_neural["ANNEALING_LENGTH"])
        if self.total_lower_steps < self.lower_start_threshold:
            self.zeta_lower = self.zeta_initial
        elif self.total_lower_steps < self.lower_stable_threshold:
            bump_factor = min(1.0, (self.total_lower_steps - self.lower_start_threshold) / (self.lower_stable_threshold - self.lower_start_threshold))
            target = self.zeta_initial + (self.zeta_max * 0.5 - self.zeta_initial) * bump_factor
            self.zeta_lower = max(self.zeta_lower, target)
        else:
            self.zeta_lower = self.zeta_initial + (self.zeta_max - self.zeta_initial) * fraction
            
        if self.total_lower_steps >= self.lower_stable_threshold:
            self.zeta_upper = self.zeta_initial + (self.zeta_max - self.zeta_initial) * fraction
        else:
            self.zeta_upper = self.zeta_initial

def log_transform(reward: float) -> float:
    return reward

if __name__ == "__main__":
    Trainer().train()