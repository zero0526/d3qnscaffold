"""
Residual Routing PPO Agent
==========================
Multi-agent PPO với kiến trúc Proposal → Refinement (Residual).

Architecture:
  - MFNetwork     : Predict mean field (shared, MultiInstance)
  - ProposalActor : Base routing decision  (shared, MultiInstance)
  - RefineActor   : Residual correction δz  (shared, MultiInstance, init ≈ 0)
  - Critic        : State value V(s)          (shared, MultiInstance)

Interface tương thích hoàn toàn với PPOAgent:
  choose_action_batch(states, mfs, masks_batch, agent_indices, deterministic, zeta)
  store_transition_train_mf_batch(...)
  learn(agents_ids, zeta)
  save(path) / load(path)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from matrix_source.utils import math_utils

from matrix_source.agents.base import MultiInstanceLinear, MultiInstanceRMSNorm
from matrix_source.agents.buffer.rollout_buffer import MultiAgentRolloutBuffer

def compute_overload(h, weights):
    h_weighted_mean = (h * weights).sum()
    overload = (h - h_weighted_mean) / (h_weighted_mean + 1e-8)
    return overload

class MFNetwork(nn.Module):
    """Predict current mean field từ (state || prev_mf).
    Output: sigmoid -> [0,1]^mf_dim.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes, num_instances: int = 1):
        super().__init__()
        h1, h2 = hidden_sizes
        self.fc1  = MultiInstanceLinear(num_instances, input_dim, h1)
        self.norm = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2  = MultiInstanceLinear(num_instances, h1, h2)
        self.out  = MultiInstanceLinear(num_instances, h2, output_dim)

    def forward(self, x, indices=None):
        x = F.silu(self.norm(self.fc1(x, indices), indices))
        x = F.silu(self.fc2(x, indices))
        return torch.sigmoid(self.out(x, indices))


class ProposalActor(nn.Module):
    """(task || svc || mf) → logits"""
    def __init__(self, task_state, service_state, mf_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        in_dim = task_state + service_state + mf_dim
        self.fc1    = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1  = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2    = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2  = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

    def forward(self, task, svc, mf, indices=None):
        x = torch.cat([task, svc, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)

    def evaluate(self, task, svc, mf, action, masks=None,
                 indices=None, residual_logits=None, exclude_zero=False):
        logits = self.forward(task, svc, mf, indices)
        if residual_logits is not None:
            logits = logits + residual_logits
        if masks is not None:
            logits = logits.masked_fill(masks == 0, -1e9)
        if exclude_zero and logits.shape[-1] > 1:
            logits[:, 0] = -1e9
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class RefineActor(nn.Module):
    """(task || svc || mf || proposal || hist || overload) → δlogits"""
    def __init__(self, task_state, service_state, mf_dim,
                 proposal_dim, action_dim,
                 hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        M = service_state // 2  # service_state = 2*M
        hist_dim = 2 * M        # histogram + overload
        in_dim = task_state + service_state + mf_dim + proposal_dim + hist_dim

        self.fc1    = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1  = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2    = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2  = MultiInstanceRMSNorm(num_instances, h2)
        self.logits = MultiInstanceLinear(num_instances, h2, action_dim)

        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, task, svc, mf, proposal_logits,
                histogram, overload, indices=None):
        x = torch.cat([
            task, svc, mf,
            proposal_logits.detach(),
            histogram, overload,
        ], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.logits(x, indices)


class ResidualCritic(nn.Module):
    """(general_task || svc || mf) → V(s)"""
    def __init__(self, general_task_states, service_states,
                 mf_dim, hidden_sizes, num_instances=1):
        super().__init__()
        h1, h2 = hidden_sizes
        in_dim = general_task_states + service_states + mf_dim
        self.fc1   = MultiInstanceLinear(num_instances, in_dim, h1)
        self.norm1 = MultiInstanceRMSNorm(num_instances, h1)
        self.fc2   = MultiInstanceLinear(num_instances, h1, h2)
        self.norm2 = MultiInstanceRMSNorm(num_instances, h2)
        self.v     = MultiInstanceLinear(num_instances, h2, 1)

    def forward(self, general_task, svc, mf, indices=None):
        x = torch.cat([general_task, svc, mf], dim=-1)
        x = F.silu(self.norm1(self.fc1(x, indices), indices))
        x = F.silu(self.norm2(self.fc2(x, indices), indices))
        return self.v(x, indices).squeeze(-1)


# ============================================================
# AGENT
# ============================================================

class ResidualRoutingAgent:
    def __init__(self, agent_id, node_type,
                 service_state_dim, mf_dim, proposal_dim,
                 action_dim, u_action_dim,
                 mf_hidden_sizes, mf_lr, buffer_min_size,
                 hidden_sizes=(128, 64), lr=3e-4,
                 gamma=0.99, alpha=0.005,
                 buffer_size=100_000, batch_size=128,
                 lam=0.95, clip_eps=0.2, k_epochs=5,
                 entropy_coef=0.05, exclude_zero=False,
                 num_instances=1, device=None):

        self.agent_id   = agent_id
        self.node_type = node_type

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_instances = num_instances
        self.action_dim    = action_dim
        self.u_action_dim  = u_action_dim
        self.exclude_zero  = exclude_zero

        # ═══ THÊM: M và max_models ═══
        self.M          = service_state_dim // 2
        self.max_models = u_action_dim // self.M

        self.gamma      = gamma
        self.lmbda      = lam
        self.eps_clip   = clip_eps
        self.k_epochs   = k_epochs
        self.batch_size = batch_size
        self.min_batch_size = buffer_min_size
        self.alpha      = alpha

        self.initial_entropy_coef = entropy_coef
        self.entropy_coef         = entropy_coef
        self.entropy_decay_rate   = 0.99
        self.min_entropy_coef     = 0.001

        TASK_DIM         = 4
        GENERAL_TASK_DIM = 7

        # ── Networks ──
        self.mf_net = MFNetwork(
            input_dim    = GENERAL_TASK_DIM + service_state_dim + mf_dim,
            output_dim   = mf_dim,
            hidden_sizes = mf_hidden_sizes,
            num_instances= num_instances,
        ).to(self.device)

        self.proposal = ProposalActor(
            task_state    = TASK_DIM,
            service_state = service_state_dim,
            mf_dim        = mf_dim,
            action_dim    = u_action_dim,
            hidden_sizes  = hidden_sizes,
            num_instances = num_instances,
        ).to(self.device)

        self.refine = RefineActor(
            task_state    = TASK_DIM,
            service_state = service_state_dim,
            mf_dim        = mf_dim,
            proposal_dim  = proposal_dim,
            action_dim    = u_action_dim,
            hidden_sizes  = hidden_sizes,
            num_instances = num_instances,
        ).to(self.device)

        self.critic = ResidualCritic(
            general_task_states = GENERAL_TASK_DIM,
            service_states      = service_state_dim,
            mf_dim              = mf_dim,
            hidden_sizes        = hidden_sizes,
            num_instances       = num_instances,
        ).to(self.device)

        # ── Optimizers ──
        self.optimizer_proposal = optim.Adam(self.proposal.parameters(), lr=lr)
        self.optimizer_refine   = optim.Adam(self.refine.parameters(),   lr=lr)
        self.optimizer_critic   = optim.Adam(self.critic.parameters(),   lr=lr)
        self.mf_optimizer       = optim.Adam(self.mf_net.parameters(),  lr=mf_lr)
        self.loss_fn            = nn.SmoothL1Loss()

        self.memory = MultiAgentRolloutBuffer(
            num_agents         = num_instances,
            node_type          = node_type,
            max_size_per_agent = buffer_size,
            service_state_dim  = service_state_dim,
            action_dim         = action_dim,
            device             = self.device,
        )

        self.learn_step_counter = 0

    def choose_action(self, state, prev_mf, mask=None, agent_idx=0,
                      task_state=None, deterministic=False, zeta=1.0):
        """
        Args:
            state:      (2*M,) service state
            prev_mf:    (mf_dim,) previous mean field
            mask:       (u_action_dim,) hoặc None
            task_state: (N, 4) task tensors cho agent này
        Returns:
            actions:   (N,) LongTensor
            log_probs: (N,) FloatTensor
            value:     scalar
        """
        idx_t = torch.tensor([agent_idx], device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if prev_mf.dim() == 1:
            prev_mf = prev_mf.unsqueeze(0)
        if task_state is None:
            raise ValueError("task_state required")

        if not isinstance(task_state, list):
            task_state = [task_state]
        if mask is not None and not isinstance(mask, list):
            mask = [mask]

        actions, log_probs, values = self.choose_action_batch(
            service_states=state,
            prev_mfs=prev_mf,
            task_states=task_state,
            masks_batch=mask,
            agent_indices=idx_t,
            deterministic=deterministic,
            zeta=zeta,
        )

        return actions[0], log_probs[0], values[0]

    def tasks_to_general(self, task_states):
        """
        Args:
            task_states: List[(N_i, 4)] hoặc (N_i, 4) Tensor
        Returns:
            Nếu list: (B, 7) — mỗi hàng là summary cho 1 agent
            Nếu tensor: (7,) — summary cho 1 group
        """
        # Nếu là list (batch)
        if isinstance(task_states, (list, tuple)):
            results = []
            for t in task_states:
                results.append(self._general_single(t))
            return torch.stack(results)  # (B, 7)

        # Nếu là tensor (single group)
        return self._general_single(task_states)  # (7,)

    def _general_single(self, tasks):
        """tasks: (N_i, 4) → (7,)"""
        if tasks.shape[0] == 0:
            return torch.zeros(7, device=self.device)

        t = tasks.float()
        mean = t.mean(dim=0)
        std = t.std(dim=0, correction=0) if t.shape[0] > 1 else torch.zeros_like(mean)

        return torch.tensor([
            mean[0],  # mean omega
            mean[1],  # mean data_size
            float(t.shape[0]),  # num_tasks
            mean[2],  # mean deadline
            std[2],  # std deadline
            mean[3],  # mean acc
            std[3],  # std acc
        ], dtype=torch.float32, device=self.device)

    def choose_action_batch(self, service_states, prev_mfs, task_states,
                            masks_batch=None, agent_indices=None,
                            deterministic=False, zeta=1.0):
        """
        Returns:
            all_actions:   List[(N_i,) LongTensor]
            all_log_probs: List[(N_i,) FloatTensor]
            all_values:    (B,) FloatTensor
        """
        B = service_states.shape[0]
        device = self.device

        if agent_indices is None:
            agent_indices = torch.zeros(B, dtype=torch.long, device=device)
        else:
            agent_indices = agent_indices.to(device).view(-1)

        service_states = service_states.to(device).float()
        prev_mfs = prev_mfs.to(device).float()

        # General task: (B, 7)
        general_task = self.tasks_to_general(task_states)

        all_actions = []
        all_log_probs = []
        all_values = []

        with torch.no_grad():
            # ═══ MF: once per agent ═══
            mf_input = torch.cat([general_task, service_states, prev_mfs], dim=-1)
            pred_mfs = self.mf_net(mf_input, indices=agent_indices)  # (B, mf_dim)

            for i in range(B):
                n_i = task_states[i].shape[0]
                tasks_i = task_states[i].to(device).float()  # (N_i, 4)
                aid = agent_indices[i]

                svc_i = service_states[i].unsqueeze(0).expand(n_i, -1)  # (N_i, 2M)
                mf_i = pred_mfs[i].unsqueeze(0).expand(n_i, -1)  # (N_i, mf_dim)
                idx_i = aid.unsqueeze(0).expand(n_i)  # (N_i,)

                mask_i = (masks_batch[i].to(device)
                          if masks_batch is not None else None)

                # ── 1. Proposal: (task, svc, mf) → logits ──
                prop_logits = self.proposal(tasks_i, svc_i, mf_i, indices=idx_i)
                # (N_i, u_action_dim)

                # ── 2. Histogram ──
                prop_for_hist = prop_logits.clone()
                if mask_i is not None:
                    prop_for_hist = prop_for_hist.masked_fill(mask_i == 0, -1e9)

                safe_logits = self._sanitize_logits(prop_for_hist)
                prop_probs = F.softmax(safe_logits, dim=-1)  # (N_i, A)

                prop_3d = prop_probs.view(n_i, self.M, self.max_models)
                h_node = prop_3d.sum(dim=(0, 2))  # (M,)

                # ── 3. Overload ──
                f_v = service_states[i, :self.M]  # (M,)
                overload = compute_overload(h_node, f_v)  # (M,)

                # ── 4. Expand ──
                hist_exp = h_node.unsqueeze(0).expand(n_i, -1)  # (N_i, M)
                over_exp = overload.unsqueeze(0).expand(n_i, -1)  # (N_i, M)

                # ── 5. Refinement: (task, svc, mf, prop, hist, over) → δ ──
                delta_logits = self.refine(
                    tasks_i, svc_i, mf_i,
                    prop_logits, hist_exp, over_exp,
                    indices=idx_i,
                )  # (N_i, u_action_dim)

                # ── 6. Fusion ──
                final_logits = prop_logits + delta_logits

                if mask_i is not None:
                    final_logits = final_logits.masked_fill(mask_i == 0, -1e9)
                if self.exclude_zero and self.u_action_dim > 1:
                    final_logits[:, 0] = -1e9

                final_logits = self._sanitize_logits(final_logits)

                # ── 7. Sample ──
                if deterministic:
                    actions_i = final_logits.argmax(dim=-1)
                    log_probs_i = torch.zeros(n_i, device=device)
                else:
                    dist = Categorical(logits=final_logits)
                    actions_i = dist.sample()
                    log_probs_i = dist.log_prob(actions_i)

                all_actions.append(actions_i)
                all_log_probs.append(log_probs_i.mean())  # ← MEAN aggregation

            # ── 8. Critic: (general_task, svc, mf) → V ──
            all_values = self.critic(
                general_task, service_states, pred_mfs,
                indices=agent_indices,
            )  # (B,)

        return all_actions, all_log_probs, all_values

    @staticmethod
    def _sanitize_logits(z):
        """Đảm bảo logits hợp lệ: không NaN, không toàn -inf."""
        z = torch.where(
            torch.isnan(z),
            torch.tensor(-1e6, device=z.device),
            z,
        )
        z = torch.where(
            torch.isinf(z),
            torch.tensor(-1e6, device=z.device),
            z,
        )
        # Nếu TẤT CẢ logits trong 1 hàng đều <= -1e5 → uniform
        all_bad = (z <= -1e5).all(dim=-1, keepdim=True)
        z = torch.where(all_bad, torch.zeros_like(z), z)
        return z

    # ----------------------------------------------------------
    # ② STORE TRANSITION + TRAIN MF
    # ----------------------------------------------------------

    def store_transition_train_mf_batch(
            self, service_states, task_states, prev_mfs, curr_mfs,
            actions, rewards, next_service_states, dones,
            agent_ids, log_probs, values, masks=None,
    ):
        """
        1. Train MF (supervised) với ground truth curr_mfs
        2. Store vào buffer
        """
        # ── Train MF ──
        general_tasks = self.tasks_to_general(task_states)  # (B, 7)
        loss_mf = self.learn_mf_batch(
            general_tasks, service_states, prev_mfs, curr_mfs, agent_ids
        )

        # ── Store ──
        self.memory.add_batch(
            service_states, task_states, prev_mfs, curr_mfs,
            actions, rewards, next_service_states, dones,
            log_probs, values, agent_ids, masks=masks,
        )

        return loss_mf

    def learn_mf_batch(self, general_tasks, service_states,
                       prev_mfs, ground_truth_mfs, agent_ids):
        """
        Train MF net supervised với ground truth MF.

        Args:
            general_tasks:       (B, 7)
            service_states:      (B, 2*M)
            prev_mfs:            (B, mf_dim)
            ground_truth_mfs:    (B, mf_dim)
            agent_ids:           (B,)
        """
        gt = torch.as_tensor(general_tasks, dtype=torch.float32, device=self.device)
        s = torch.as_tensor(service_states, dtype=torch.float32, device=self.device)
        pm = torch.as_tensor(prev_mfs, dtype=torch.float32, device=self.device)
        gf = torch.as_tensor(ground_truth_mfs, dtype=torch.float32, device=self.device)

        # MF input: general_task || service_state || prev_mf
        mf_input = torch.cat([gt, s, pm], dim=-1)
        pred_mf = self.mf_net(mf_input, indices=agent_ids)
        loss = self.loss_fn(pred_mf, gf)

        self.mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mf_net.parameters(), max_norm=5.0)
        self.mf_optimizer.step()
        return loss.item()

    # ----------------------------------------------------------
    # ③ PPO LEARN
    # ----------------------------------------------------------
    def _unpack_task_batch(self, task_batch_cat, task_lens):
        """
        Args:
            task_batch_cat: (total_tasks, task_dim) — concatenated
            task_lens:      (dataset_size,)          — số task mỗi sample
        Returns:
            task_states:    List[(N_i, task_dim)]
        """
        task_states = []
        offset = 0
        for n_i in task_lens:
            n_i = int(n_i.item())
            task_states.append(task_batch_cat[offset:offset + n_i])
            offset += n_i
        return task_states

    def _forward_single_with_action(self, tasks_i, svc_i, mf_i,
                                    mask_i, aid, actions_i):
        """
        Args:
            tasks_i:   (N_i, 4)
            svc_i:     (N_i, 2M)  — detached (từ b_pred_mfs)
            mf_i:      (N_i, mf_dim) — detached
            mask_i:    (N_i, u_action_dim) hoặc None
            aid:       scalar
            actions_i: (N_i,) — action RIÊNG cho mỗi task

        Returns:
            log_probs:    (N_i,)
            entropy:      (N_i,)
            delta_logits: (N_i, u_action_dim)
            prop_logits:  (N_i, u_action_dim)
        """
        n_i = tasks_i.shape[0]
        idx_i = aid.unsqueeze(0).expand(n_i)

        # 1. Proposal
        prop_logits = self.proposal(tasks_i, svc_i, mf_i, indices=idx_i)

        # 2. Histogram
        prop_for_hist = prop_logits.clone()
        if mask_i is not None:
            prop_for_hist = prop_for_hist.masked_fill(mask_i == 0, -1e9)

        safe_logits = self._sanitize_logits(prop_for_hist.detach())
        prop_probs = F.softmax(safe_logits, dim=-1)

        prop_3d = prop_probs.view(n_i, self.M, self.max_models)
        h_node = prop_3d.sum(dim=(0, 2))

        # 3. Overload
        f_v = svc_i[0, :self.M]
        overload = compute_overload(h_node, f_v)

        # 4. Expand
        hist_exp = h_node.unsqueeze(0).expand(n_i, -1)
        over_exp = overload.unsqueeze(0).expand(n_i, -1)

        # 5. Refinement
        delta_logits = self.refine(
            tasks_i, svc_i, mf_i,
            prop_logits, hist_exp, over_exp,
            indices=idx_i,
        )

        # 6. Final logits + log_prob
        final_logits = prop_logits + delta_logits

        if mask_i is not None:
            final_logits = final_logits.masked_fill(mask_i == 0, -1e9)
        if self.exclude_zero and self.u_action_dim > 1:
            final_logits[:, 0] = -1e9

        final_logits = self._sanitize_logits(final_logits)

        dist = Categorical(logits=final_logits)
        log_probs = dist.log_prob(actions_i)  # (N_i,)
        entropy = dist.entropy()  # (N_i,)

        return log_probs, entropy, delta_logits, prop_logits

    def learn(self, agents_ids=None, zeta=1.0):
        from matrix_source.trainers.ppo_stategy import compute_gae

        if agents_ids is not None:
            agents_ids = agents_ids.to(self.device).view(-1)

        data = self.memory.get_all_ready(
            min_size=self.min_batch_size, agent_ids_pool=agents_ids,
        )
        if data is None:
            return None

        self.entropy_coef = max(
            self.entropy_coef * self.entropy_decay_rate,
            self.min_entropy_coef,
        )

        # ═══ UNPACK 14 FIELDS ═══
        (service_states,  # 0  (D, svc_dim)
         task_batch_cat,  # 1  (total_tasks, 4)
         task_lens,  # 2  (D,)
         actions_cat,  # 3  (total_tasks,)
         action_lens,  # 4  (D,)
         prev_mfs,  # 5  (D, mf_dim)
         curr_mfs,  # 6  (D, mf_dim)
         rewards,  # 7  (D, 1)
         next_service_states,  # 8 (D, svc_dim)
         dones,  # 9  (D, 1)
         old_log_probs,  # 10 (D, 1)
         old_values,  # 11 (D, 1)
         masks,  # 12 List[D]
         agent_ids,  # 13 (D,)
         ) = data

        old_log_probs = old_log_probs.squeeze(-1)
        old_values = old_values.squeeze(-1)
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        dataset_size = service_states.shape[0]

        # Reconstruct variable-length
        task_states = self._unpack_task_batch(task_batch_cat, task_lens)
        action_list = self._unpack_task_batch(actions_cat, action_lens)

        general_tasks = self.tasks_to_general(task_states)

        # ═══ GAE ═══
        with torch.no_grad():
            mf_in = torch.cat([general_tasks, next_service_states, curr_mfs], dim=-1)
            next_mf = self.mf_net(mf_in, indices=agent_ids)
            next_val = self.critic(general_tasks, next_service_states,
                                   next_mf, indices=agent_ids)

            advantages = compute_gae(
                rewards, next_val, old_values, dones, agent_ids,
                self.gamma, self.lmbda,
            )
            returns = advantages + old_values

            if advantages.numel() > 1:
                advantages = (
                        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                )

        # ═══ PPO ═══
        epoch_metrics = {'v': 0.0, 'p': 0.0, 'ent': 0.0, 'l2': 0.0, 'kl': 0.0}
        total_batches = 0

        for _ in range(self.k_epochs):
            perm = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                idx = perm[start:start + self.batch_size]

                b_svc = service_states[idx]
                b_prev = prev_mfs[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]
                b_aids = agent_ids[idx]
                b_gen = general_tasks[idx]
                b_masks = [masks[int(i)] for i in idx] if masks else None

                # MF (detach)
                mf_in = torch.cat([b_gen, b_svc, b_prev], dim=-1)
                b_mf = self.mf_net(mf_in, indices=b_aids).detach()

                # Per-sample forward
                # Per-sample forward
                all_lp, all_ent, all_delta, all_prop, all_masks_j = [], [], [], [], []

                for j in range(len(idx)):
                    orig = int(idx[j])
                    n_j = task_states[orig].shape[0]
                    t_j = task_states[orig].to(self.device).float()
                    s_j = b_svc[j].unsqueeze(0).expand(n_j, -1)
                    m_j = b_mf[j].unsqueeze(0).expand(n_j, -1)
                    mk_j = b_masks[j] if b_masks else None

                    act_j = action_list[orig].to(self.device)

                    lp, ent, delta, prop = self._forward_single_with_action(
                        t_j, s_j, m_j, mk_j, b_aids[j],
                        actions_i=act_j,
                    )

                    all_lp.append(lp)
                    all_ent.append(ent)
                    all_delta.append(delta)
                    all_prop.append(prop)
                    all_masks_j.append(mk_j)  # ← THÊM DÒNG NÀY

                # Aggregate
                new_lp = torch.stack([lp.mean() for lp in all_lp])
                ent_m = torch.stack([e.mean() for e in all_ent])

                # PPO
                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip,
                                    1 + self.eps_clip) * b_adv

                # L2
                l2 = torch.stack([d.norm(dim=-1).mean()
                                  for d in all_delta]).mean()

                # ═══ FIX: KL đúng — proposal vs final ═══
                kl = torch.stack([
                    math_utils.compute_kl(p, d, mk)
                    for p, d, mk in zip(all_prop, all_delta, all_masks_j)
                ]).mean()

                actor_loss = (
                        -torch.min(surr1, surr2).mean()
                        - self.entropy_coef * ent_m.mean()
                        + 0.01 * l2
                        + 0.05 * kl
                )

                self.optimizer_proposal.zero_grad()
                self.optimizer_refine.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.proposal.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.refine.parameters(), 0.5)
                self.optimizer_proposal.step()
                self.optimizer_refine.step()

                # Critic
                c_vals = self.critic(b_gen, b_svc, b_mf, indices=b_aids)
                c_loss = F.mse_loss(c_vals, b_ret)

                self.optimizer_critic.zero_grad()
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                epoch_metrics['v'] += c_loss.item()
                epoch_metrics['p'] += actor_loss.item()
                epoch_metrics['ent'] += ent_m.mean().item()
                epoch_metrics['l2'] += l2.item()
                epoch_metrics['kl'] += kl.item()
                total_batches += 1

        self.learn_step_counter += 1

        log_freq = 10 if self.node_type == "Edge_Group" else 100
        if self.learn_step_counter % log_freq == 0 and total_batches > 0:
            n = total_batches
            print(
                f"[{self.node_type}] Step {self.learn_step_counter:5d} | "
                f"V: {epoch_metrics['v'] / n:.5f} | "
                f"P: {epoch_metrics['p'] / n:.5f} | "
                f"Ent: {epoch_metrics['ent'] / n:.4f} | "
                f"L2: {epoch_metrics['l2'] / n:.6f} | "
                f"KL: {epoch_metrics['kl'] / n:.6f}"
            )

        self.memory.clear()
        return epoch_metrics['v'] / total_batches if total_batches > 0 else 0.0

    # ----------------------------------------------------------
    # ④ Checkpoint
    # ----------------------------------------------------------

    def save(self, path: str):
        checkpoint = {
            'proposal'    : self.proposal.state_dict(),
            'refine'      : self.refine.state_dict(),
            'critic'      : self.critic.state_dict(),
            'mf_net'      : self.mf_net.state_dict(),
            'proposal_opt': self.optimizer_proposal.state_dict(),
            'refine_opt'  : self.optimizer_refine.state_dict(),
            'critic_opt'  : self.optimizer_critic.state_dict(),
            'mf_opt'      : self.mf_optimizer.state_dict(),
            'learn_step'  : self.learn_step_counter,
            'entropy_coef': self.entropy_coef,
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.proposal.load_state_dict(ckpt['proposal'])
        self.refine.load_state_dict(ckpt['refine'])
        self.critic.load_state_dict(ckpt['critic'])
        self.mf_net.load_state_dict(ckpt['mf_net'])
        self.optimizer_proposal.load_state_dict(ckpt['proposal_opt'])
        self.optimizer_refine.load_state_dict(ckpt['refine_opt'])
        self.optimizer_critic.load_state_dict(ckpt['critic_opt'])
        self.mf_optimizer.load_state_dict(ckpt['mf_opt'])
        self.learn_step_counter = ckpt.get('learn_step', 0)
        self.entropy_coef       = ckpt.get('entropy_coef', self.initial_entropy_coef)

    def set_lr_factor(self, factor: float):
        opts = [self.optimizer_proposal, self.optimizer_refine,
                self.optimizer_critic, self.mf_optimizer]
        for opt in opts:
            for pg in opt.param_groups:
                pg['lr'] *= factor
        print(
            f"[{self.node_type}] LR scaled ×{factor}. "
            f"Proposal LR: {self.optimizer_proposal.param_groups[0]['lr']:.6f}"
        )


