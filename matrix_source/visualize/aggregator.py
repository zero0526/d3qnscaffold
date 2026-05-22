import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os
import csv
import logging
from matrix_source.configs.configs import cfg
import torch

class MetricsAggregator:
    def __init__(self):
        self.history = defaultdict(list)
        self.episode_count = 0
        self._setup_logger()
        self.reset_episode()

    def _setup_logger(self):
        """Sets up a logger that writes to both terminal and a file."""
        log_dir = cfg.logs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, "metrics.log")
        
        self.logger = logging.getLogger("MetricsAggregator")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(ch)

    def log(self, message):
        self.logger.info(message)

    def reset_episode(self):
        """Resets the accumulators for a new episode."""
        # Performance: Use lists to collect tensors lazily
        self.episode_upper_rewards = []
        self.episode_lower_rewards = []
        self.episode_energy = []
        self.episode_violations = []
        self.episode_success_qos = []
        self.episode_violate_qos = []
        self.episode_backlog_drift = []
        self.episode_remaining_tasks = []
        self.episode_assigned_list = []
        self.episode_failed_list = []
        
        # Training/State tracking
        self.episode_upper_mf_losses = []
        self.episode_lower_mf_losses = []
        self.episode_upper_td_losses = []
        self.episode_lower_td_losses = []
        self.episode_upper_states = []
        self.episode_lower_states = []
        
        # Q-Stats
        self.episode_upper_q_min = []
        self.episode_upper_q_max = []
        self.episode_upper_q_mean = []
        self.episode_lower_q_min = []
        self.episode_lower_q_max = []
        self.episode_lower_q_mean = []
        
        self.episode_step_fail_reasons = [] # Dicts of tensors
        self.curr_zeta_lower = 1.0
        self.curr_zeta_upper = 1.0
        
        # Matrix tracking
        self.eps_f_alloc = []
        self.eps_arrivals = []
        self.eps_backlog = []
        
        # Derived stats (cleared each episode)
        self.eps_hw_deficit = None
        self.eps_hw_fail_count = None
        self.episode_fail_reasons = {'deadline': 0, 'hardware': 0, 'queue_full': 0, 'invalid_placement': 0}

    def add_upper(self, step_output, mf_loss=0, state=None):
        self.episode_upper_rewards.append(step_output.get("reward_global", 0))
        self.episode_upper_mf_losses.append(mf_loss)
        if state is not None:
            s = state.detach().cpu().numpy() if hasattr(state, "detach") else np.array(state)
            self.episode_upper_states.append(s.flatten())

    def add_lower(self, step_output, mf_loss=0, state=None):
        self.episode_lower_rewards.append(step_output.get("reward", 0.0))
        self.episode_lower_mf_losses.append(mf_loss)
        if state is not None:
            s = state.detach().cpu().numpy() if hasattr(state, "detach") else np.array(state)
            self.episode_lower_states.append(s.flatten())
            
        info = step_output.get("info", {})
        obs = step_output.get("obs", {})
        if obs: self.episode_backlog_drift.append(obs.get("total_drift", 0))
        
        self.episode_energy.append(step_output.get("energy", 0))
        self.episode_violations.append(step_output.get("violations", 0))

        # Lazy summation of QoS vectors
        def _sum(v): return v.float().sum() if isinstance(v, torch.Tensor) else v
        self.episode_success_qos.append(_sum(info.get("success_qos", 0)))
        self.episode_violate_qos.append(_sum(info.get("violate_qos", 0)))
        
        # Task counters
        self.episode_remaining_tasks.append(info.get("remaining", 0))
        self.episode_assigned_list.append(info.get("num_tasks", 0))
        self.episode_failed_list.append(info.get("immediate_fails", 0) + info.get("expired_count", 0))

        # Failure reasons
        reasons = info.get("fail_reasons")
        if reasons: self.episode_step_fail_reasons.append(reasons)

    def add_step_matrices(self, f_alloc, arrivals, backlog):
        self.eps_f_alloc.append(f_alloc)
        self.eps_arrivals.append(arrivals)
        self.eps_backlog.append(backlog)

    def record_td_losses(self, upper_losses=None, lower_losses=None):
        def _f(x):
            if isinstance(x, dict): return float(x.get("loss", 0.0))
            if hasattr(x, "item"): return x.item()
            return float(x)
        if upper_losses is not None:
            val = np.mean([_f(l) for l in upper_losses]) if isinstance(upper_losses, list) else _f(upper_losses)
            self.episode_upper_td_losses.append(val)
        if lower_losses is not None:
            val = np.mean([_f(l) for l in lower_losses]) if isinstance(lower_losses, list) else _f(lower_losses)
            self.episode_lower_td_losses.append(val)

    def record_zeta(self, lower, upper):
        self.curr_zeta_lower = lower
        self.curr_zeta_upper = upper
        
    def record_q_stats(self, node_type, q_min, q_max, q_mean):
        def _f(x): return x.item() if hasattr(x, "item") else float(x)
        if node_type == "Edge_Group":
            self.episode_upper_q_min.append(_f(q_min))
            self.episode_upper_q_max.append(_f(q_max))
            self.episode_upper_q_mean.append(_f(q_mean))
        else:
            self.episode_lower_q_min.append(_f(q_min))
            self.episode_lower_q_max.append(_f(q_max))
            self.episode_lower_q_mean.append(_f(q_mean))

    def store_history(self):
        """Processes collected episode metrics and stores them in history. Reset for next episode."""
        def to_float_list(arr):
            if not arr: return []
            res = []
            for v in arr:
                if isinstance(v, torch.Tensor):
                    res.append(float(v.detach().cpu().sum().item()))
                elif isinstance(v, dict):
                    res.append(sum(float(val.item() if hasattr(val, 'item') else val) for val in v.values()))
                else: res.append(float(v))
            return res

        # Rewards and Energy
        u_rw = to_float_list(self.episode_upper_rewards)
        l_rw = to_float_list(self.episode_lower_rewards)
        self.history["total_reward"].append(float(np.sum(u_rw) + np.sum(l_rw)))
        
        energy_vals = to_float_list(self.episode_energy)
        total_energy = float(np.sum(energy_vals))
        self.history["total_energy"].append(total_energy)

        # Process step matrices (Batch average)
        if self.eps_f_alloc:
            # Stack all tensors and mean across time (dim 0)
            self.eps_f_alloc = [x.detach().cpu() if hasattr(x, "detach") else torch.tensor(x) for x in self.eps_f_alloc]
            self.eps_f_all = torch.stack(self.eps_f_alloc).mean(dim=0).numpy()
            self.eps_arrivals = [x.detach().cpu() if hasattr(x, "detach") else torch.tensor(x) for x in self.eps_arrivals]
            self.eps_arr_all = torch.stack(self.eps_arrivals).mean(dim=0).numpy()
            self.eps_backlog = [x.detach().cpu() if hasattr(x, "detach") else torch.tensor(x) for x in self.eps_backlog]
            self.eps_back_all = torch.stack(self.eps_backlog).mean(dim=0).numpy()

        # Process failure reasons
        for sr in self.episode_step_fail_reasons:
            for k in ['deadline', 'hardware', 'queue_full', 'invalid_placement']:
                v = sr.get(k, 0)
                self.episode_fail_reasons[k] += float(v.item() if hasattr(v, "item") else v)
            
            # Service-specific hardware stats
            hd, hc = sr.get("hw_deficit_per_svc"), sr.get("hw_fail_count_per_svc")
            if hd is not None:
                val = hd.detach().cpu().numpy() if hasattr(hd, "detach") else hd
                if self.eps_hw_deficit is None: self.eps_hw_deficit = val.copy()
                else: self.eps_hw_deficit += val
            if hc is not None:
                val = hc.detach().cpu().numpy() if hasattr(hc, "detach") else hc
                if self.eps_hw_fail_count is None: self.eps_hw_fail_count = val.copy()
                else: self.eps_hw_fail_count += val

        # Training Averages
        def _avg(arr, key):
            if not arr: return self.history[key][-1] if self.history[key] else 0
            return float(np.mean(arr))

        self.history["avg_upper_mf_loss"].append(_avg(self.episode_upper_mf_losses, "avg_upper_mf_loss"))
        self.history["avg_lower_mf_loss"].append(_avg(self.episode_lower_mf_losses, "avg_lower_mf_loss"))
        self.history["avg_upper_td_loss"].append(_avg(self.episode_upper_td_losses, "avg_upper_td_loss"))
        self.history["avg_lower_td_loss"].append(_avg(self.episode_lower_td_losses, "avg_lower_td_loss"))
        
        self.history["zeta_lower"].append(self.curr_zeta_lower)
        self.history["zeta_upper"].append(self.curr_zeta_upper)

        for k in ["upper_q_min", "upper_q_max", "upper_q_mean", "lower_q_min", "lower_q_max", "lower_q_mean"]:
            self.history[k].append(_avg(getattr(self, f"episode_{k}"), k))

        # QoS and Completion
        s_list, v_list = to_float_list(self.episode_success_qos), to_float_list(self.episode_violate_qos)
        s_sum, v_sum = np.sum(s_list), np.sum(v_list)
        self.history["qos_success_rate"].append(s_sum / (s_sum + v_sum) if (s_sum + v_sum) > 0 else 0)
        
        assigned = np.sum(to_float_list(self.episode_assigned_list))
        failed = np.sum(to_float_list(self.episode_failed_list))
        rem = self.episode_remaining_tasks[-1]
        rem_val = float(rem.item() if hasattr(rem, "item") else rem)
        self.history["completion_rate"].append((assigned - failed - rem_val) / assigned if assigned > 0 else 0)
        
        self.history["avg_backlog_drift"].append(_avg(to_float_list(self.episode_backlog_drift), "avg_backlog_drift"))
        self.history["avg_remaining_tasks"].append(_avg(to_float_list(self.episode_remaining_tasks), "avg_remaining_tasks"))
        self.history["qos_rate"].append(s_sum / (v_sum if v_sum > 0 else 1.0))
        self.history["total_violations"].append(float(v_sum))
        
        self.episode_count += 1
        if self.episode_count % 50 == 0:
            self.plot_history(ep=self.episode_count)
            self.save_history_csv()
        
        self.reset_episode()

    def report_episode(self, ep):
        """Prints summary. history[-1] contains processed results for current ep."""
        tr = self.history["total_reward"][-1] if self.history["total_reward"] else 0
        en = self.history["total_energy"][-1] if self.history["total_energy"] else 0
        qos = self.history["qos_success_rate"][-1] if self.history["qos_success_rate"] else 0
        cr = self.history["completion_rate"][-1] if self.history["completion_rate"] else 0
        self.log(f"EP {ep:4d} | Rew: {tr:8.2f} | Energy: {en:8.2f} | QoS: {qos:6.2%} | Comp: {cr:6.2%}")

    def plot_history(self, ep=None):
        if not self.history["total_reward"]: return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history["total_reward"], label="Total Reward")
        axes[0, 0].set_title("Reward Convergence")
        axes[0, 1].plot(self.history["qos_success_rate"], label="QoS Success Rate", color='green')
        axes[0, 1].set_title("QoS Stability")
        axes[1, 0].plot(self.history["total_energy"], label="Total Energy", color='red')
        axes[1, 0].set_title("Energy Consumption")
        axes[1, 1].plot(self.history["completion_rate"], label="Completion Rate", color='blue')
        axes[1, 1].set_title("Task Throughput")
        for ax in axes.flat: ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.plot_dir, f"training_progress_{ep if ep else 'latest'}.png"))
        plt.close()

    def save_history_csv(self):
        path = os.path.join(cfg.results, "training_history.csv")
        if not os.path.exists(cfg.results): os.makedirs(cfg.results)
        keys = sorted(self.history.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for i in range(len(self.history[keys[0]])):
                writer.writerow([self.history[k][i] for k in keys])
