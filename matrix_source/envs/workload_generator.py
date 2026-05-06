import torch

class MatrixWorkloadGenerator:
    """
    NHIỆM VỤ CỦA THÀNH PHẦN (WORKLOAD GENERATOR):
    1. Sinh ra các Task mới dưới dạng Ma Trận (Arrival Matrix).
       - Kích thước: (Num_Terminals x Num_Services).
    """
    def __init__(self, config, metadata):
        self.num_terminals = config.hyper_neural.get("NUM_LOWER_AGENTS", 1)
        self.num_services = metadata['service_size'].shape[0]
        self.zipf_probs = metadata['zipf_probs']
        self.accuracies = metadata['model_accuracies']
        self.deadlines = metadata['service_deadlines']
        self.device = config.device
        self.min_batch_size = config.task_min_batch
        self.max_batch_size = config.task_max_batch

    def generate_step(self):
        """
        Mỗi terminal sinh đúng 1 task tại mỗi step.
        """
        device = self.device
        N = self.num_terminals

        # 1. Terminal indices
        terminal_indices = torch.arange(N, device=device)

        # 2. Sample service (Zipf)
        svc_indices = torch.multinomial(self.zipf_probs, N, replacement=True)

        # 3. Batch size
        task_batch_sizes = torch.randint(
            self.min_batch_size,
            self.max_batch_size + 1,
            (N,),
            device=device,
            dtype=torch.float32
        )

        # 4. Accuracy sampling (dùng gather nhanh hơn)
        acc_pool = self.accuracies[svc_indices]   # (N, num_models)

        valid_mask = (acc_pool > 0).float()
        zero_rows = valid_mask.sum(dim=1) == 0
        if zero_rows.any():
            valid_mask[zero_rows] = 1.0
        rand_acc_cols = torch.multinomial(valid_mask, 1).squeeze(1)
        tasks_min_accuracy = acc_pool.gather(1, rand_acc_cols.unsqueeze(1)).squeeze(1) - 1e0

        # 5. Deadline sampling (mask + multinomial an toàn)
        dl_pool = self.deadlines[svc_indices]     # (N, num_deadlines)
        rand_dl_cols = torch.randint(
            0, dl_pool.shape[1], (N,), device=device
        )
        task_deadlines = dl_pool.gather(1, rand_dl_cols.unsqueeze(1)).squeeze(1)


        return (
            terminal_indices,
            svc_indices,
            task_batch_sizes,
            tasks_min_accuracy,
            task_deadlines
        )
