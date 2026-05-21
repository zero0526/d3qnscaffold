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
        self.eps_upper = 1.0
        self.eps_lower = 1.0
        self.lower_epsilons = {tid: 1.0 for tid in range(self.num_terminals)}
        self.zeta_initial = cfg.hyper_neural.get("ZETA", 1.0)
        self.zeta_max = cfg.hyper_neural.get("ZETA_MAX", 20.0)
        self.zeta_upper = self.zeta_initial
        self.zeta_lower = self.zeta_initial
        self.epsilon_start = 1.0
        self.epsilon_tau = cfg.hyper_neural.get("ANNEALING_LENGTH", 5000)
        # --- Lower-level zeta exploration schedule (independent from upper) ---
        # Lower zeta stays frozen during warmup, then increases slowly per episode
        self.zeta_lower_warmup = int(cfg.hyper_neural.get("ZETA_LOWER_WARMUP", 200))   # episodes to keep zeta_lower = initial
        self.zeta_lower_step  = float(cfg.hyper_neural.get("ZETA_LOWER_STEP", 0.005))  # additive increment per episode after warmup
        self.zeta_lower_max   = float(cfg.hyper_neural.get("ZETA_LOWER_MAX", 5.0))     # separate cap for lower (keep < upper cap)

        # Training control variables
        self.total_lower_steps = 0
        self.total_upper_steps = 0
        self.lower_stable_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][0]*10
        self.lower_start_threshold = self.config.hyper_neural["BUFFER_MIN_SIZE"][1]
        
        self.aggregator = MetricsAggregator()
        self.shared_upper_agent = None
        self.shared_lower_agent = None

        self.edge_ids = self.env.static_matrices["edge_ids"]
        self.edge_node_ids = [nid for nid in range(self.num_nodes)
                              if nid not in self.env.static_matrices.get("cloud_ids", [])]
        self.node_to_instance = {nid: i for i, nid in enumerate(self.edge_node_ids)}
        self.num_edge_agents = len(self.edge_node_ids)
        self.max_epochs= 3000
        # 2. Strategy Injection
        self.strategy = strategy if strategy is not None else PPOStrategy()
        self.strategy.initialize_agents(self)

    def train(self):
        self.strategy.run_training(self)

    def update_rates(self, ep):
        # 1. Update Epsilons using exponential decay: eps = eps_end + (eps_start - eps_end) * exp(-t / tau)
        import math
        self.eps_upper = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                         math.exp(-self.total_upper_steps / 10 / self.epsilon_tau)

        self.eps_lower = self.min_epsilon + (self.epsilon_start - self.min_epsilon) * \
                         math.exp(-self.total_lower_steps / 100 / self.epsilon_tau)

        # 2. Delayed Linear Zeta Annealing
        # Zeta only starts increasing after Epsilon has reached its "mature" phase
        lower_zeta_threshold = self.epsilon_tau * 100
        upper_zeta_threshold = self.epsilon_tau * 10

        if self.total_lower_steps > lower_zeta_threshold:
            zeta_steps_l = self.total_lower_steps - lower_zeta_threshold
            self.zeta_lower = min(self.zeta_max, self.zeta_initial + zeta_steps_l * self.config.zeta_lower_step)

        if self.total_upper_steps > upper_zeta_threshold:
            zeta_steps_u = self.total_upper_steps - upper_zeta_threshold
            self.zeta_upper = min(self.zeta_max, self.zeta_initial + zeta_steps_u * self.config.zeta_upper_step)
def log_transform(reward: float) -> float:
    return reward

if __name__ == "__main__":
    Trainer().train()