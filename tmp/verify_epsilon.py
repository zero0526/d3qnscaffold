import math

# Mocking the constants for calculation
epsilon_start = 1.0
epsilon_min = 0.01
decay_rate = 0.993442

eps = epsilon_start
for i in range(700):
    eps = max(epsilon_min, eps * decay_rate)

print(f"Epsilon after 700 steps: {eps:.6f}")
assert abs(eps - 0.01) < 0.001, f"Expected ~0.01, got {eps}"

eps = max(epsilon_min, eps * decay_rate)
print(f"Epsilon after 701 steps: {eps:.6f}")
assert eps >= epsilon_min, "Epsilon dropped below min"
print("Verification successful!")
