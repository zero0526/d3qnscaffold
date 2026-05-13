from matrix_source.trainers.train import Trainer
from matrix_source.trainers.ppo_stategy import PPOStrategy
from matrix_source.trainers.d3qn_strategy import D3QNStrategy

if __name__ == '__main__':
    # Pick the algorithm strategy: PPOStrategy() or D3QNStrategy()
    strategy = PPOStrategy() 
    
    trainer = Trainer(strategy=strategy)
    trainer.train()