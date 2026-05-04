import numpy as np

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
