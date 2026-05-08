import torch
from matrix_source.envs.time_manager import TimeManager
from matrix_source.envs.matrix_physical_engine import MatrixPhysicalEngine
from matrix_source.envs.init_matrices import init_static_matrices, init_metadata_tensors

class MatrixSixGEnvironment:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        
        # 1. Initialize Matrices and Metadata (Automated)
        init_data = init_static_matrices(config, device=device)
        self.static_matrices = init_data
        self.terminals = init_data['terminals']
        
        metadata = init_metadata_tensors(config, device=device)
        self.metadata = metadata
        
        # 2. Initialize Engine
        self.engine = MatrixPhysicalEngine(config, self.static_matrices, metadata, device)
        
        # 3. Time Management
        self.time_manager = TimeManager(
            slot_duration=self.engine.slot_duration,
            timeframe_size=config.hyper_neural["TIME_SLOT_PER_TIMEFRAME"] ,
            max_steps=config.hyper_neural["NUMOF_TF_EP"]*config.hyper_neural["TIME_SLOT_PER_TIMEFRAME"]
        )

    def reset(self):
        self.time_manager.reset()
        return self.engine.reset()

    def step_upper(self, placement_matrix):
        """
        Upper-level decision: Update service placement internally.
        """
        self.engine.set_upper_action(placement_matrix)
        
    def collect_upper_metrics(self):
        """
        Collect results for the timeframe that just finished.
        """
        metrics = self.engine.collect_upper_metrics()
        metrics["is_done"] = self.time_manager.current_step >= self.time_manager.max_steps
        return metrics

    def step_lower(self, terminal_indices, svc_indices, task_batch_sizes, node_indices, model_indices, task_deadlines, tasks_min_accuracy):
        """
        Execute the lower-level step via the physical engine.
        """
        # 1. Process Arrivals
        node_arrival_matrix, trans_energy_total, cold_delays, f_min_matrix = self.engine.process_arrivals(
            terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, tasks_min_accuracy
        )
        
        # 2. Solver Optimization
        # f_min_matrix = self.engine.get_f_min_matrix()
        self.engine.optimize_allocation(node_arrival_matrix, f_min_matrix)
        
        # 3. Execution & Metrics
        results = self.engine.execute_and_collect_metrics(node_arrival_matrix, trans_energy_total, cold_delays)
        
        # 4. Finalize Slot
        self.time_manager.tick()

        return {
            "reward": results['reward'].item(),
            "energy": results['energy'].item(),
            "violations": results['violations'],
            "obs": results['obs'],
            "info": results["info"],
            "mean_field": results['mean_field'],
            "prev_actions": results['prev_actions'],
            "new_frame": self.time_manager.is_new_frame()
        }
