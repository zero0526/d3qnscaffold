from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy
from matrix_source.trainers.d3qn_scaffold_strategy_v2 import D3QNScaffoldStrategy

from matrix_source.trainers.rb_sac_cen_stra import RB_SAC_CEN_STRA

if __name__ == '__main__':
    strategy = D3QNScaffoldStrategy()
    trainer = Trainer(strategy=strategy)
    trainer.train()