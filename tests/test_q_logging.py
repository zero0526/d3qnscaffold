import torch
import torch.nn as nn
from matrix_source.agents.d3qn import D3QNAgent

def test_q_logging():
    # Setup parameters
    node_id = 0
    node_type = "Terminal_Group"
    state_dim = 10
    action_dim = 5
    u_action_dim = 5
    mf_hidden_sizes = (32,)
    mf_lr = 1e-3
    buffer_min_size = 10
    
    # Initialize agent with logs_q=True
    agent = D3QNAgent(
        node_id=node_id, 
        node_type=node_type, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        u_action_dim=u_action_dim, 
        mf_hidden_sizes=mf_hidden_sizes, 
        mf_lr=mf_lr, 
        buffer_min_size=buffer_min_size,
        logs_q=True
    )
    
    # Mock data to fill buffer
    for _ in range(20):
        agent.memory.add_batch(
            torch.randn(1, state_dim),
            torch.randn(1, action_dim),
            torch.randn(1, action_dim),
            torch.randint(0, u_action_dim, (1, 1)),
            torch.randn(1, 1),
            torch.randn(1, state_dim),
            torch.zeros(1, 1),
            torch.tensor([0])
        )
    
    # Run learn
    res = agent.learn(agents_ids=torch.tensor([0]))
    
    print(f"Result type: {type(res)}")
    if isinstance(res, dict):
        print("Result keys:", res.keys())
        print(f"Q-Min: {res['q_min']:.4f}")
        print(f"Q-Max: {res['q_max']:.4f}")
        print(f"Q-Mean: {res['q_mean']:.4f}")
        
        assert "q_min" in res
        assert "q_max" in res
        assert "q_mean" in res
        assert "loss" in res
        print("Verification SUCCESS: Dictionary format is correct.")
    else:
        print("Verification FAILED: Expected dict, got", type(res))

if __name__ == "__main__":
    test_q_logging()
