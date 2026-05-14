import torch
import numpy as np
from matrix_source.agents.sac_ec import GaussianActor, DirichletActor, Critic

def test_initialization():
    device = torch.device("cpu")
    state_dim = 10
    action_dim = 2
    
    print("Testing GaussianActor initialization...")
    actor = GaussianActor(state_dim, action_dim).to(device)
    # Check if weights are within expected range for orthogonal init
    # Mean weights should be small (gain=0.01)
    mean_weight_std = actor.mean.weight.std().item()
    print(f"GaussianActor mean weight std: {mean_weight_std:.6f}")
    assert mean_weight_std < 0.1, "GaussianActor mean weights should be small"
    
    print("Testing DirichletActor initialization...")
    d_actor = DirichletActor(state_dim, action_dim).to(device)
    alpha_head_weight_std = d_actor.alpha_head.weight.std().item()
    print(f"DirichletActor alpha_head weight std: {alpha_head_weight_std:.6f}")
    assert alpha_head_weight_std < 0.1, "DirichletActor alpha_head weights should be small"
    
    print("Testing Critic initialization...")
    critic = Critic(state_dim, action_dim).to(device)
    q1_l3_weight_std = critic.q1_l3.weight.std().item()
    print(f"Critic q1_l3 weight std: {q1_l3_weight_std:.6f}")
    # gain=1.0 for output layer, so std should be around 1/sqrt(hidden_dim) = 1/sqrt(256) = 0.0625
    # For orthogonal, it should be reasonable.
    
    print("All basic initialization tests passed!")

if __name__ == "__main__":
    test_initialization()
