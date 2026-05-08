import re
import os
import numpy as np
from collections import defaultdict

def parse_metrics_log(log_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    upper_rewards = []
    lower_rewards = []
    upper_feature_maxs = defaultdict(list)
    lower_feature_maxs = defaultdict(list)

    # Regex for rewards
    reward_re = re.compile(r"Avg (Upper|Lower) Reward: ([-+]?\d*\.\d+|\d+)")
    # Regex for state stats
    state_start_re = re.compile(r"\[(Upper|Lower) Agents\] Feature distribution")
    # Regex for feature line: Feat | Mean ± Std | Range [Min, Max]
    # Example: 0     |    0.123 ±    0.045 | [    0.000,     1.000]
    feature_re = re.compile(r"(\d+)\s+\|\s+[-+]?\d*\.\d+\s+±\s+\d*\.\d+\s+\|\s+\[\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\s*\]")

    current_agent_type = None
    collecting_features = False

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Parse Rewards
            reward_match = reward_re.search(line)
            if reward_match:
                reward_type = reward_match.group(1)
                reward_val = float(reward_match.group(2))
                if reward_type == "Upper":
                    upper_rewards.append(reward_val)
                else:
                    lower_rewards.append(reward_val)
                continue

            # Parse State Stats Section start
            state_match = state_start_re.search(line)
            if state_match:
                current_agent_type = state_match.group(1)
                collecting_features = True
                continue

            # Parse Feature line
            if collecting_features:
                feat_match = feature_re.search(line)
                if feat_match:
                    feat_idx = int(feat_match.group(1))
                    f_min = float(feat_match.group(2))
                    f_max = float(feat_match.group(3))
                    
                    # We store the absolute max to use as a divisor
                    abs_max = max(abs(f_min), abs(f_max))
                    if current_agent_type == "Upper":
                        upper_feature_maxs[feat_idx].append(abs_max)
                    else:
                        lower_feature_maxs[feat_idx].append(abs_max)
                elif "---" in line or line.strip() == "":
                    # Table end
                    pass
                elif "Average Delay" in line or "Successful Tasks" in line:
                    collecting_features = False

    # Reporting
    print("\n" + "="*60)
    print("         LOG ANALYSIS FOR NORMALIZATION PARAMETERS")
    print("="*60)

    def suggest_divisor(max_val):
        """Simple heuristic to suggest a power of 10 or a clean number."""
        if max_val == 0: return 1.0
        # Round up to nearest power of 10 or 5*10^n
        import math
        order = 10**math.floor(math.log10(max_val))
        if max_val / order <= 2: return 2 * order
        if max_val / order <= 5: return 5 * order
        return 10 * order

    print("\n--- Reward Normalization Suggestions ---")
    if upper_rewards:
        max_u_rew = max([abs(r) for r in upper_rewards])
        print(f"Upper Reward: Abs Max observed = {max_u_rew:.4f} -> Suggested Divisor: {suggest_divisor(max_u_rew)}")
    if lower_rewards:
        max_l_rew = max([abs(r) for r in lower_rewards])
        print(f"Lower Reward: Abs Max observed = {max_l_rew:.4f} -> Suggested Divisor: {suggest_divisor(max_l_rew)}")

    def print_agent_suggestions(label, feature_maxs):
        print(f"\n--- {label} State Feature Suggestions ---")
        if not feature_maxs:
            print("No data found.")
            return
        
        header = f"{'Feat':<5} | {'Observed Abs Max':<20} | {'Suggested Divisor':<20}"
        print(header)
        print("-" * len(header))
        for idx in sorted(feature_maxs.keys()):
            m = max(feature_maxs[idx])
            print(f"{idx:<5} | {m:19.4f} | {suggest_divisor(m):20.1f}")

    print_agent_suggestions("Upper Agents", upper_feature_maxs)
    print_agent_suggestions("Lower Agents", lower_feature_maxs)

    # Save to YAML functionality
    def save_to_yaml(target_path):
        import yaml
        norm_data = {
            "upper_state": {
                "features": {idx: float(suggest_divisor(max(vals))) for idx, vals in upper_feature_maxs.items()}
            },
            "lower_state": {
                "features": {idx: float(suggest_divisor(max(vals))) for idx, vals in lower_feature_maxs.items()}
            },
            "rewards": {
                "upper_divisor": float(suggest_divisor(max([abs(r) for r in upper_rewards]))) if upper_rewards else 1.0,
                "lower_divisor": float(suggest_divisor(max([abs(r) for r in lower_rewards]))) if lower_rewards else 1.0
            }
        }
        
        with open(target_path, 'w', encoding='utf-8') as f:
            yaml.dump(norm_data, f, default_flow_style=False)
        print(f"\n[SUCCESS] Normalization parameters saved to {target_path}")

    norm_yaml = cfg.normalization_path
    save_to_yaml(norm_yaml)

    print("\n" + "="*60)
    print("TIP: Use these divisors in your env.step() or agent.observe() functions.")
    print("     Division by these constants will bring values closer to [-1, 1].")
    print("="*60 + "\n")

if __name__ == "__main__":
    from matrix_source.configs.configs import cfg
    log_file = os.path.join(cfg.logs, "metrics.log")
    parse_metrics_log(log_file)
