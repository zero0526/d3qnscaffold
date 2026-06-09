from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy
from matrix_source.trainers.semi_distribute_task import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.pposcaffold import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.residual_routing_ppo import ResidualRoutingPPOStrategy
from matrix_source.configs.configs import cfg


cfg.hyper_neural["NUM_LOWER_AGENTS"] =60
if cfg.hyper_neural["NUM_LOWER_AGENTS"]==60:
    cfg.norm_upper_rw*=2
    cfg.norm_lower_rw*=2
elif cfg.hyper_neural["NUM_LOWER_AGENTS"]==40:
    cfg.norm_upper_rw *= 1
    cfg.norm_lower_rw *= 1
if __name__ == '__main__':
    strategy = D3QNScaffoldStrategy()
    trainer = Trainer(strategy=strategy)
    trainer.train()