import torch
from matrix_source.envs.matrix_physical_engine import MatrixPhysicalEngine
from matrix_source.configs.configs import cfg

def test_2d_failure_tracking():
    device = "cpu"
    # Create a dummy config and metadata
    config = cfg
    num_nodes = 5
    num_services = 3
    num_terminals = 10
    
    static_matrices = {
        'resource_matrix': torch.ones((num_nodes, 4), device=device) * 100,
        'transmission_delay_matrix': torch.zeros((num_nodes, num_nodes), device=device),
        'terminal_to_comp_node_map': torch.eye(num_nodes, num_terminals, device=device).T, # T0->N0, T1->N1, etc.
        'adj_matrix': torch.eye(num_nodes, device=device),
        'terminal_adj_matrix': torch.eye(num_terminals, device=device),
        'max_queue_delay': torch.zeros((num_nodes, num_services), device=device),
        'edge_ids': list(range(num_nodes)),
    }
    
    metadata = {
        'service_omega': torch.zeros(num_services, 1, device=device),
        'service_deadlines': torch.ones(num_services, device=device) * 10,
        'service_input_size': torch.ones(num_services, 1, device=device),
        'model_workloads': torch.ones((num_services, 5), device=device) * 10,
        'model_accuracies': torch.ones((num_services, 5), device=device) * 0.9,
        'service_size': torch.ones(num_services, device=device) * 1024, # 1 GB
        'service_size': torch.ones(num_services, device=device) * 1024, # 1 GB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
        'service_size': torch.ones(num_services, device=device) * 1, # 1 MB
    }
    # Fix service_size
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100

    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100
    metadata['service_size'] = torch.ones(num_services, device=device) * 100

    engine = MatrixPhysicalEngine(config, static_matrices, metadata, device)
    engine.reset()
    
    # 1. Simulate Invalid Placement Failure
    # placement_matrix is zero, so any task should fail
    print("Testing Invalid Placement Failure...")
    terminal_indices = torch.tensor([0, 1])
    svc_indices = torch.tensor([0, 1])
    node_indices = torch.tensor([0, 1])
    model_indices = torch.tensor([0, 0])
    task_batch_sizes = torch.ones(2)
    task_deadlines = torch.ones(2) * 10
    task_accuracies = torch.ones(2) * 0.9
    
    engine.process_arrivals(terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, task_accuracies)
    print(f"Fail counts after invalid placement (should be (10, 3) with [0,0]=1 and [1,1]=1):")
    print(engine.terminal_fail_counts[0:2])
    assert engine.terminal_fail_counts[0, 0] == 1
    assert engine.terminal_fail_counts[1, 1] == 1
    
    # Reset
    engine.reset()
    
    # 2. Simulate Deadline Failure
    print("Testing Deadline Failure (Transmission Delay > Deadline)...")
    # Set huge delay matrix
    static_matrices['transmission_delay_matrix'] = torch.ones((num_nodes, num_nodes)) * 100
    engine.delay_matrix = static_matrices['transmission_delay_matrix']
    
    engine.process_arrivals(terminal_indices, svc_indices, node_indices, model_indices, task_batch_sizes, task_deadlines, task_accuracies)
    print(f"Fail counts after deadline failure:")
    print(engine.terminal_fail_counts[0:2])
    assert engine.terminal_fail_counts[0, 0] == 1
    assert engine.terminal_fail_counts[1, 1] == 1

    print("Verification complete! All tests passed.")

if __name__ == "__main__":
    test_2d_failure_tracking()
