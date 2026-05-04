import random
import torch
import matrix_source.utils.tensor_ops as ops
from matrix_source.models.resource_solver import KKTSolverADMM

class MatrixPhysicalEngine:
    def __init__(self, config, static_matrices, metadata, device):
        self.config = config
        self.device = device
        
        # Static Parameters
        self.resource_specs = static_matrices['resource_matrix']
        self.delay_matrix = static_matrices['transmission_delay_matrix']
        self.terminal_to_node_map = static_matrices['terminal_to_comp_node_map']
        self.adj_matrix = static_matrices['adj_matrix'].to(device)
        self.terminal_adj_matrix = static_matrices['terminal_adj_matrix'].to(device)
        
        self.service_omega = metadata['service_omega']
        self.service_deadlines = metadata['service_deadlines']
        self.service_input_size = metadata['service_input_size']
        self.model_workloads = metadata['model_workloads']
        self.model_accuracies = metadata['model_accuracies']
        self.max_queue_delay = static_matrices['max_queue_delay'] 
        self.service_size = metadata['service_size'] / 1024.0 # MB -> GB
        
        # Dynamics Configuration
        self.slot_duration = config.get('slot_duration', 0.1)
        self.lypa_coef = config.get('lypa_coef', 10.0)
        self.energy_coef = config.get('energy_coef', 1.0)
        self.cold_start_delay_min = config.get('cold_start_delay_min', 0.5)
        self.cold_start_delay_max = config.get('cold_start_delay_max', 0.85)
        self.energy_cold_start = config.get('energy_cold_start', 10.0)
        
        # Reward Weights
        self.omega_1 = config.get('omega_1', 10.0)
        self.omega_2 = config.get('omega_2', 0.1)
        
        # State Tensors
        self.num_nodes = self.resource_specs.shape[0]
        self.num_services = self.service_omega.shape[0]
        self.num_terminals = self.terminal_to_node_map.shape[0]
        self.max_K = config.get('max_queue_size', 100) 
        
        self.backlog_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        self.deadline_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        self.q_deadline_queue = torch.zeros((self.num_nodes, self.num_services, self.max_K), device=self.device)
        
        self.cpu_alloc_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.prev_placement_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.newly_placed_mask = torch.zeros((self.num_nodes, self.num_services), dtype=torch.bool, device=self.device)
        self.used_resources = torch.zeros((self.num_nodes, 4), device=self.device)
        
        # Lower Level Action Tracking (for MARL)
        self.prev_node_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.prev_model_indices = torch.zeros(self.num_terminals, dtype=torch.long, device=self.device)
        self.current_task_reqs = torch.zeros((self.num_terminals, 4), device=self.device) # [data_size, dl, acc, type]
        
        # Accumulators for Timeframe Metrics
        self.phi_accumulator = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        self.reward_global_accumulator = 0.0
        
        # Solver
        self.solver = KKTSolverADMM(config, self.resource_specs, self.service_omega)
        self.immediate_fails = 0
        self.placement_violations = 0

    def reset(self):
        self.backlog_queue.zero_()
        self.deadline_queue.zero_()
        self.q_deadline_queue.zero_()
        self.prev_placement_matrix.zero_()
        self.newly_placed_mask.zero_()
        self.reward_global_accumulator = 0.0
        self.phi_accumulator.zero_()
        self.prev_node_indices.zero_()
        self.prev_model_indices.zero_()
        self.current_task_reqs.zero_()
        return self.reset_upper()
    
    def reset_upper(self):
        # Calculate Mean Fields (Average actions of nodes)
        neighbor_count = self.adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_fields = (self.adj_matrix @ self.placement_matrix) / neighbor_count
        
        # Convert phi statistics to probability distribution
        phi_prob = ops.transform2prob(self.phi_accumulator)
        
        res = {
            "actions": self.placement_matrix.clone(),
            "phi": phi_prob,
            "mean_fields": mean_fields,
            "reward_global": self.reward_global_accumulator
        }
        
        # Reset accumulators for next timeframe
        self.phi_accumulator.zero_()
        self.reward_global_accumulator = 0.0
        self.reset_lower()
        return res

    def reset_lower(self):
        # Initial lower observation format
        mean_field_terminals = self._calc_terminal_mean_field()
        
        obs = {
            "task_reqs": self.current_task_reqs.clone(),
            "backlog_prev": self.backlog_queue.sum(dim=-1).clone(),
            "cpu_alloc_prev": self.cpu_alloc_matrix.clone()
        }
        
        self.immediate_fails = 0
        return {
            "obs": obs,
            "mean_field": mean_field_terminals,
            "prev_actions": {
                "node_selection": self.prev_node_indices.clone(),
                "model_selection": self.prev_model_indices.clone()
            }
        }

    def _calc_terminal_mean_field(self):
        # neighboring_count: (Num_Terminals, 1)
        neighbor_count = self.terminal_adj_matrix.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Calculate mean actions of neighbors
        mf_node = (self.terminal_adj_matrix @ self.prev_node_indices.float().unsqueeze(1)) / neighbor_count
        mf_model = (self.terminal_adj_matrix @ self.prev_model_indices.float().unsqueeze(1)) / neighbor_count
        return torch.cat([mf_node, mf_model], dim=-1)

    def update_placement(self, new_placement):
        # Before updating, get current summary (Step Upper logic)
        res = self.reset_upper()
        
        self.prev_placement_matrix = self.placement_matrix.clone()
        valid_placement = new_placement.clone()
        omega_1_mask = (self.service_omega.squeeze() == 1)
        omega_0_mask = (self.service_omega.squeeze() == 0)
        
        ram_reqs = (valid_placement * omega_1_mask) @ self.service_size
        ram_over = ram_reqs > self.resource_specs[:, 1].to(self.device)
        hdd_reqs = (valid_placement * omega_0_mask) @ self.service_size
        hdd_over = hdd_reqs > self.resource_specs[:, 2].to(self.device)
        
        over_mask = ram_over | hdd_over
        valid_placement[over_mask] = self.placement_matrix[over_mask]
        self.placement_violations = over_mask.float().sum().item()
        
        self.placement_matrix = valid_placement
        self.newly_placed_mask = (self.placement_matrix > 0) & (self.prev_placement_matrix == 0)
        self.cpu_alloc_matrix *= self.placement_matrix
        
        return res

    def process_arrivals(self, terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes):
        # Store current actions for next slot's prev_actions
        self.prev_node_indices.index_copy_(0, terminal_indices, node_indices)
        self.prev_model_indices.index_copy_(0, terminal_indices, model_indices)

        num_tasks = len(svc_indices)
        trans_energy_total = 0.0
        node_arrival_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        cold_delays = torch.tensor([], device=self.device)
        self.immediate_fails = 0
        
        # Reset current task reqs
        self.current_task_reqs.zero_()

        if num_tasks == 0:
            return node_arrival_matrix, trans_energy_total, cold_delays

        src_node_indices = torch.argmax(self.terminal_to_node_map[terminal_indices], dim=1)
        base_input_sizes = self.service_input_size[svc_indices].squeeze()
        task_data_sizes = base_input_sizes * task_batch_sizes
        
        trans_delays, trans_energy_tasks = ops.compute_transmission_metrics(
            src_node_indices, node_indices, self.delay_matrix, task_data_sizes, 
            beta=self.config.get('trans_energy_beta', 1e-6)
        )
        trans_energy_total = trans_energy_tasks.sum()

        base_workloads = self.model_workloads[svc_indices, model_indices]
        task_workloads = base_workloads * task_batch_sizes
        
        task_mean_deadlines = self.service_deadlines[svc_indices]
        task_max_queue = self.max_queue_delay[node_indices, svc_indices]
        task_accuracies = self.model_accuracies[svc_indices, model_indices]
        task_types = self.service_omega[svc_indices].squeeze()
        
        # Populate current_task_reqs: [data_size, dl, acc, type]
        req_features = torch.stack([
            task_data_sizes, task_mean_deadlines, task_accuracies, task_types
        ], dim=-1)
        self.current_task_reqs.index_copy_(0, terminal_indices, req_features)

        task_cold_start = self.newly_placed_mask[node_indices, svc_indices] & (task_types == 0)
        cold_delays = task_cold_start.float() * random.uniform(self.cold_start_delay_min, self.cold_start_delay_max)
        
        t_rem_raw = task_mean_deadlines - trans_delays - cold_delays
        t_q_rem = t_rem_raw - task_max_queue
        
        valid_mask = t_rem_raw >= 1e-4
        self.immediate_fails = (~valid_mask).sum().item()
        
        if valid_mask.any():
            vn, vs, vw, vt, vq, vb = (
                node_indices[valid_mask], svc_indices[valid_mask], task_workloads[valid_mask], 
                t_rem_raw[valid_mask], t_q_rem[valid_mask], task_batch_sizes[valid_mask]
            )
            self.phi_accumulator.index_put_((vn, vs), vb, accumulate=True)
            
            for n, s, w, t, q in zip(vn, vs, vw, vt, vq):
                num_active = (self.backlog_queue[n, s, :] > 0).sum().item()
                if num_active < self.max_K:
                    idx = int(num_active)
                    self.backlog_queue[n, s, idx] = w
                    self.deadline_queue[n, s, idx] = t
                    self.q_deadline_queue[n, s, idx] = q
                else:
                    self.immediate_fails += 1
            
            node_arrival_matrix.index_put_((vn, vs), vw, accumulate=True)
            
        return node_arrival_matrix, trans_energy_total, cold_delays

    def get_f_min_matrix(self):
        f_min_matrix = torch.zeros((self.num_nodes, self.num_services), device=self.device)
        valid_tasks = self.backlog_queue > 0
        if not valid_tasks.any():
            return f_min_matrix

        cum_backlog = torch.cumsum(self.backlog_queue, dim=-1)
        req_matrix = cum_backlog / (self.q_deadline_queue + 1e-9)
        req_matrix = torch.where(valid_tasks, req_matrix, torch.zeros_like(req_matrix))
        f_min_matrix, _ = torch.max(req_matrix, dim=-1)
        return f_min_matrix

    def optimize_allocation(self, node_arrival_matrix, f_min_matrix):
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        G = current_backlog_total * self.placement_matrix
        Z = self.lypa_coef * self.energy_coef * self.placement_matrix * node_arrival_matrix

        f_max = (self.resource_specs[:, 0:1] * self.placement_matrix).to(self.device)
        f_min = f_min_matrix.clamp(max=f_max)
        self.cpu_alloc_matrix = self.solver.solve(G, Z, f_min, f_max)

    def execute_and_collect_metrics(self, node_arrival_matrix, trans_energy_total, cold_delays):
        current_backlog_total = self.backlog_queue.sum(dim=-1)
        prev_cpu_alloc = self.cpu_alloc_matrix.clone()
        
        self.backlog_queue, actual_processed, in_slot_violation_mask = ops.deplete_float_queue(
            self.backlog_queue, self.deadline_queue, self.cpu_alloc_matrix, self.slot_duration
        )
        
        self.backlog_queue, self.deadline_queue, self.q_deadline_queue, expired_count = ops.age_and_clean_dual_queue(
            self.backlog_queue, self.deadline_queue, self.q_deadline_queue, in_slot_violation_mask, self.slot_duration
        )
        
        num_violations = expired_count + self.immediate_fails
        total_drift = ops.calculate_lyapunov_drift(current_backlog_total, node_arrival_matrix, self.cpu_alloc_matrix*self.slot_duration)
        comp_energy = ops.compute_batch_energy(
            self.cpu_alloc_matrix, actual_processed, self.energy_coef, 
            cold_delays, epsilon_cold=self.energy_cold_start
        )
        total_energy = comp_energy + trans_energy_total
        
        f1 = total_drift + self.lypa_coef * total_energy
        self.reward_global_accumulator += f1.item()
        
        qos_penalty = self.omega_1 * torch.exp(torch.tensor(self.omega_2 * num_violations, device=self.device))
        reward = -(f1 + qos_penalty + self.placement_violations * 50.0)
        
        # MARL observations
        mean_field_terminals = self._calc_terminal_mean_field()
        obs = {
            "task_reqs": self.current_task_reqs.clone(),
            "backlog_prev": self.backlog_queue.sum(dim=-1).clone(),
            "cpu_alloc_prev": prev_cpu_alloc
        }
        
        self.used_resources[:, 0] = self.cpu_alloc_matrix.sum(dim=1) 
        self.used_resources[:, 1] = (self.placement_matrix * (self.service_omega.T.to(self.device) == 1)) @ self.service_size.to(self.device)
        self.used_resources[:, 2] = (self.placement_matrix * (self.service_omega.T.to(self.device) == 0)) @ self.service_size.to(self.device)
        self.used_resources[:, 3] = comp_energy / (self.resource_specs[:, 0].to(self.device).sum() + 1e-9)
        
        self.placement_violations = 0
        
        return {
            "reward": reward,
            "backlog": current_backlog_total,
            "energy": total_energy,
            "violations": num_violations,
            "obs": obs,
            "mean_field": mean_field_terminals,
            "prev_actions": {
                "node_selection": self.prev_node_indices.clone(),
                "model_selection": self.prev_model_indices.clone()
            }
        }
