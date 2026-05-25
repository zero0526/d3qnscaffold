import torch

# ==========================================
# 1. TRANSMISSION METRICS
# ==========================================

def compute_transmission_metrics(src_nodes, dst_nodes, delay_matrix, data_sizes, beta=1e-6):
    """
    src_nodes, dst_nodes: (Task_Count,) indices
    delay_matrix: (M, M)
    data_sizes: (Task_Count,)
    """
    trans_delays = delay_matrix[src_nodes, dst_nodes] * data_sizes
    trans_energy = beta * trans_delays
    return trans_delays, trans_energy

# ==========================================
# 2. HIGH-PRECISION FLOAT QUEUE OPERATIONS (3D)
# ==========================================

def deplete_float_queue(backlog, deadline, terminal_queue, src_node_mapping, cpu_alloc, slot_duration):
    """
    Xử lý hàng đợi, tách biệt lượng xử lý nội bộ (local) và tổng thể.
    terminal_queue: (M, S, K) ID của terminal gửi task
    src_node_mapping: (Num_Terminals,) Map từ terminal ID sang Node ID nguồn
    """
    f = cpu_alloc.unsqueeze(-1) + 1e-9 # (M, S, 1)
    num_nodes = backlog.shape[0]
    
    # 1. Tính thời điểm hoàn thành dự kiến của từng task
    cum_backlog = torch.cumsum(backlog, dim=-1)
    time_to_finish = cum_backlog / f # (M, S, K)
    
    # 2. Phân loại Task thành công/thất bại trong slot
    success_mask = (backlog > 0) & (time_to_finish <= slot_duration) & (time_to_finish <= deadline)
    violation_mask = (backlog > 0) & (deadline <= slot_duration) & (deadline < time_to_finish)
    processed_mask = success_mask | violation_mask
    
    # 3. Cập nhật backlog
    new_backlog = backlog.clone()
    new_backlog[processed_mask] = 0
    
    # Xử lý task đang dở dang (pending) ở cuối slot
    pending_mask = (backlog > 0) & (~processed_mask)
    before_backlog = cum_backlog - backlog
    available_for_pending = (f * slot_duration - before_backlog).clamp(min=0)
    actual_processed_pending = torch.min(available_for_pending, new_backlog)
    new_backlog = new_backlog - actual_processed_pending
    
    # 4. Tính toán lượng xử lý (Workload Delta)
    workload_delta = backlog - new_backlog # (M, S, K) lượng GFLOPs thực sự bị trừ đi
    
    # 5. Phân tách Local vs External
    # Xác định các slot trong hàng đợi là local: Node ID hiện tại == Node ID nguồn của Terminal
    node_ids = torch.arange(num_nodes, device=backlog.device).view(num_nodes, 1, 1)
    
    # Handle terminal_queue -1 (empty slots)
    valid_terminal_mask = (terminal_queue >= 0)
    src_nodes = torch.zeros_like(terminal_queue)
    src_nodes[valid_terminal_mask] = src_node_mapping[terminal_queue[valid_terminal_mask]]
    
    is_local_mask = (src_nodes == node_ids) & valid_terminal_mask
    
    local_processed_total = (workload_delta * is_local_mask.float()).sum(dim=-1)
    actual_processed_total = workload_delta.sum(dim=-1)
    
    return new_backlog, actual_processed_total, local_processed_total, violation_mask

def age_and_clean_dual_queue(backlog, deadline, in_slot_violation_mask, slot_duration, *aux_queues):
    """
    Trừ deadline và dọn dẹp hàng đợi.
    in_slot_violation_mask: Mask các task đã fail ngay trong bước deplete
    *aux_queues: Các queue phụ trợ (ví dụ: f_min_queue, terminal_queue)
    """
    # 1. Giảm deadline tuyệt đối
    deadline = deadline - slot_duration
    
    # 2. Xác định tổng số vi phạm
    # Vi phạm cũ (từ bước deplete) + Vi phạm mới (do vừa trừ slot_duration xong bị âm)
    total_violation_mask = in_slot_violation_mask | ((deadline <= 0) & (backlog > 0))
    violation_counts = total_violation_mask.sum(dim=-1) # (M, S)
    
    # NEW: Cần lấy thông tin các task bị fail TRƯỚC KHI XÓA
    # Giả sử queue cuối cùng trong aux_queues là terminal_queue
    failed_terminal_ids = None
    failed_svc_ids = None
    if len(aux_queues) > 0:
        # Lấy queue cuối cùng (Terminal Queue)
        term_queue = aux_queues[-1]
        failed_terminal_ids = term_queue[total_violation_mask]
        
        # Lấy Service ID từ mask
        # total_violation_mask: (N, S, K)
        violation_indices = torch.nonzero(total_violation_mask) # (V, 3) -> [node, svc, k]
        failed_svc_ids = violation_indices[:, 1]

        # Skip -1 values (empty slots)
        valid_mask = (failed_terminal_ids >= 0)
        failed_terminal_ids = failed_terminal_ids[valid_mask]
        failed_svc_ids = failed_svc_ids[valid_mask]

    # 3. Xóa Task vi phạm và làm sạch dữ liệu cũ
    backlog[total_violation_mask] = 0
    empty_mask = (backlog <= 1e-7)
    backlog[empty_mask] = 0
    deadline[empty_mask] = 0
    
    processed_aux = []
    for q in aux_queues:
        if q is not None:
            q[empty_mask] = 0
    
    # 4. DỒN HÀNG (Compaction)
    mask = (backlog > 0).float()
    _, indices = torch.sort(mask, dim=-1, descending=True, stable=True)
    
    backlog = torch.gather(backlog, dim=-1, index=indices)
    deadline = torch.gather(deadline, dim=-1, index=indices)
    
    for q in aux_queues:
        if q is not None:
            q = torch.gather(q, dim=-1, index=indices)
            processed_aux.append(q)
    
    return backlog, deadline, processed_aux, violation_counts, failed_terminal_ids, failed_svc_ids


# ==========================================
# 3. LYAPUNOV & ENERGY
# ==========================================

def calculate_lyapunov_drift(current_backlog, arrivals, processed):
    drift = current_backlog * (arrivals - processed)
    return drift.sum()

def transform2prob(phi: torch.Tensor)-> torch.Tensor:
    # phi: num_node x num_service 
    sum_workload_per_node= phi.sum(dim=-1, keepdim=True) + 1e-6
    return phi/sum_workload_per_node


def compute_batch_energy(f_alloc, processed, epsilon_comp, cold_delays, epsilon_cold=10.0):
    comp_energy = epsilon_comp * (f_alloc ** 2) * processed
    cold_energy = cold_delays*epsilon_cold
    return comp_energy.sum() + cold_energy.sum()
