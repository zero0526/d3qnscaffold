import torch
import time
from matrix_source.models.resource_solver import KKTSolverADMM

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Simulation setup
    num_nodes = 5
    num_services = 24
    
    # Mock data
    f_max_all = torch.tensor([50.0, 40.0, 60.0, 30.0, 80.0], device=device).unsqueeze(1)
    G = torch.rand((num_nodes, num_services), device=device) * 100
    Z = torch.rand((num_nodes, num_services), device=device) * 0.1
    f_min = torch.rand((num_nodes, num_services), device=device) * 0.5
    f_max = torch.ones((num_nodes, num_services), device=device) * 10.0
    
    iter_counts = [15, 30, 60, 100]
    results = {}

    for max_it in iter_counts:
        print(f"\n{'='*50}")
        print(f"Testing with max_iter = {max_it}")
        print(f"{'='*50}")
        
        solver = KKTSolverADMM(f_max_node=f_max_all, max_iter=max_it, tol=1e-4)
        
        start_time = time.perf_counter()
        # Ensure we use the debug flag we just added
        allocation = solver.solve(G, Z, f_min, f_max, debug=True)
        end_time = time.perf_counter()
        
        elapsed = (end_time - start_time) * 1000
        results[max_it] = elapsed
        
        # Calculate constraint satisfaction
        total_f = allocation.sum(dim=1)
        overage = (total_f - f_max_all.squeeze()).clamp(min=0).sum().item()
        
        print(f"\nResult for max_iter={max_it}:")
        print(f"  Time taken: {elapsed:.2f} ms")
        print(f"  Resource Overage: {overage:.6f}")
        print(f"  Mean Allocation: {allocation.mean().item():.4f}")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for it, t in results.items():
        print(f"max_iter {it:3d}: {t:6.2f} ms")

if __name__ == "__main__":
    run_benchmark()
