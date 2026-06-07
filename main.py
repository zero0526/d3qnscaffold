from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy
# from matrix_source.trainers.semi_distribute_task import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.pposcaffold import PPOSCAFFOLDREPStrategy
from matrix_source.trainers.residual_routing_ppo import ResidualRoutingPPOStrategy
if __name__ == '__main__':
    strategy = PPOSCAFFOLDREPStrategy()
    trainer = Trainer(strategy=strategy)
    trainer.train()