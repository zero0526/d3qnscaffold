import numpy as np
import torch.nn.functional as F

def to_binary(action_id, dim):
    """
    Converts an integer action ID into a binary vector of size dim.
    Used for multi-binary action spaces like placement.
    """
    binary_str = bin(int(action_id))[2:].zfill(dim)
    return np.array([int(char) for char in reversed(binary_str)])

def from_binary(binary_array):
    """
    Converts a binary vector back into an integer action ID.
    """
    res = 0
    for i, val in enumerate(binary_array):
        if val > 0:
            res += (1 << i)
    return res

def one_hot(idx, dim):
    """
    Returns a one-hot vector of size dim.
    """
    vec = np.zeros(dim)
    if 0 <= idx < dim:
        vec[idx] = 1
    return vec

def compute_kl(prop_logits, delta_logits, mask=None):
    """KL(proposal ‖ final) — cả hai đều có gradient."""
    final = prop_logits + delta_logits
    if mask is not None:
        prop_logits = prop_logits.masked_fill(mask == 0, -1e9)
        final = final.masked_fill(mask == 0, -1e9)

    log_p_prop = F.log_softmax(prop_logits, dim=-1)
    log_p_final = F.log_softmax(final, dim=-1)
    p_prop = F.softmax(prop_logits, dim=-1)

    # KL = Σ p_prop * (log_p_prop - log_p_final)
    kl = (p_prop * (log_p_prop - log_p_final)).sum(dim=-1).mean()
    return kl
