import torch
import numpy as np
from matrix_source.agents.d3qn import D3QNAgent
from matrix_source.agents.ppo import PPOAgent
from matrix_source.utils.math_utils import to_binary

class AlgorithmStrategy:
    def log_transform_reward(self, reward):
        # Match original train.py logic (identity)
        return reward

    def initialize_agents(self, trainer):
        raise NotImplementedError

    def get_upper_actions(self, trainer, current_upper_state, obs_upper):
        raise NotImplementedError

    def get_lower_actions(self, trainer, res_lower, t_idx, s_idx, tasks_min_accuracy, task_deadlines, batch_sizes):
        raise NotImplementedError

    def store_lower_transitions(self, trainer, current_res, next_res, t_idx, s_idx, n_idx, m_idx):
        raise NotImplementedError

    def store_upper_transitions(self, trainer, s_all, ns_all, current_res, next_res, acts_matrix, is_done):
        raise NotImplementedError

    def build_upper_state(self, trainer, obs_upper):
        actions = obs_upper['actions'] # (N, S)
        phi = obs_upper['phi_prob']    # (N, S)
        return torch.cat([actions, phi], dim=-1)

    def run_training(self, trainer):
        raise NotImplementedError


