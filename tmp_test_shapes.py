import torch
from torch.distributions import Categorical

def test_categorical():
    logits = torch.randn(10, 5)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    print(f"Actions shape: {actions.shape}")
    print(f"Actions tolist: {actions.cpu().tolist()}")
    
    a_ids = torch.tensor(actions.cpu().tolist())
    print(f"a_ids shape: {a_ids.shape}")

if __name__ == "__main__":
    test_categorical()
