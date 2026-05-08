import torch

class KKTSolverADMM:
    def __init__(self, f_max_node, rho=1.0, max_iter=100, tol=1e-4):
        self.f_max_node = f_max_node
        self.rho_base = rho
        self.max_iter = max_iter
        self.tol = tol

    def project_simplex(self, v, budgets):
        """
        Projection of multiple vectors onto their respective simplexes: sum(v_i) <= budget_i, v_i >= 0.
        v: (Batch, N)
        budgets: (Batch, 1)
        """
        v_clipped = torch.clamp(v, min=0.0)
        sums = v_clipped.sum(dim=1, keepdim=True)

        # Nodes that already satisfy the budget constraint
        within_budget = sums <= budgets
        if within_budget.all():
            return v_clipped

        # For nodes exceeding budget, project onto the sum(z) = budget plane
        # Algorithm: Duchi et al. (2008) "Efficient Projections onto the L1-Ball for Learning in High Dimensions"
        mu, _ = torch.sort(v, dim=1, descending=True)
        cum_sum = torch.cumsum(mu, dim=1)

        # (Batch, N)
        idx = torch.arange(1, v.shape[1] + 1, device=v.device).float()

        # Condition: mu_i - (cum_sum_i - budget) / i > 0
        theta = (cum_sum - budgets) / idx
        valid = mu > theta

        # Get the largest index i that satisfies the condition
        # Clamp to 0 to avoid index -1 when budgets are 0 or precision causes no matches
        rho_idx = (torch.sum(valid.float(), dim=1).long() - 1).clamp(min=0)

        # Selected thresholds
        chosen_theta = torch.gather(theta, 1, rho_idx.unsqueeze(1))

        return torch.clamp(v - chosen_theta, min=0.0)

    def solve(self, G, Z, f_min, f_max):
        """
        Vectorized ADMM solver for all nodes simultaneously.
        G: (M, S) - Backlog weights
        Z: (M, S) - Energy weights
        f_min, f_max: (M, S) - Box constraints
        """
        M, S = G.shape
        device = G.device

        # Initialize variables
        f = torch.zeros((M, S), device=device)
        z = torch.zeros((M, S), device=device)
        u = torch.zeros((M, S), device=device)

        # Adaptive rho initialization
        rho = 2 * Z.mean(dim=1, keepdim=True).clamp(min=1e-4) # (M, 1)

        # Pre-calculate budgets (Constant within solve call)
        budgets = (self.f_max_node - f_min.sum(dim=1, keepdim=True)).clamp(min=0.0)

        # Pre-calculate denominator for f-update
        denom = rho + 2 * Z

        for _ in range(self.max_iter):
            z_prev = z.clone()

            # 1. f-update
            f = (G + rho * (z - u)) / denom

            # 2. z-update: Simplex projection
            z_proj = self.project_simplex(f + u - f_min, budgets)
            z = torch.clamp(z_proj + f_min, max=f_max)

            # 3. u-update: Dual update
            u = u + (f - z)

            # 4. Residual check for convergence
            res_r = torch.norm(f - z, dim=1)
            res_s = torch.norm(rho * (z - z_prev), dim=1)

            if res_r.max() < self.tol and res_s.max() < self.tol:
                break

        return z
# import torch
#
#
# class KKTSolverADMM:
#     def __init__(self, f_max_node, rho=1.0, max_iter=100, tol=1e-4):
#         self.f_max_node = f_max_node
#         self.rho_base = rho
#         self.max_iter = max_iter
#         self.tol = tol
#
#     def project_simplex(self, v: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
#         """
#         Project rows of v onto:
#             x >= 0, sum(x) <= budget
#         v: (B, S)
#         budgets: (B, 1)
#         """
#         v = torch.clamp(v, min=0.0)
#         row_sum = v.sum(dim=1, keepdim=True)
#
#         # Rows already feasible: return directly
#         feasible = row_sum <= budgets
#         if feasible.all():
#             return v
#
#         out = v.clone()
#         idx = (~feasible).squeeze(1)
#         v_bad = v[idx]
#         b_bad = budgets[idx]
#
#         # Duchi projection onto simplex / L1 ball
#         u, _ = torch.sort(v_bad, dim=1, descending=True)
#         cssv = torch.cumsum(u, dim=1) - b_bad
#
#         j = torch.arange(
#             1,
#             u.shape[1] + 1,
#             device=v.device,
#             dtype=v.dtype
#         ).view(1, -1)
#
#         cond = u - cssv / j > 0
#
#         rho = cond.sum(dim=1).sub(1).clamp(min=0)
#
#         theta_num = cssv.gather(1, rho.unsqueeze(1))
#
#         theta_den = (rho + 1).to(v.dtype).unsqueeze(1)
#
#         theta = theta_num / theta_den
#         out[idx] = torch.clamp(v_bad - theta, min=0.0)
#
#         return out
#
#     @torch.no_grad()
#     def solve(self, G, Z, f_min, f_max):
#         """
#         G: (M, S)
#         Z: (M, S)
#         f_min, f_max: (M, S)
#         """
#         M, S = G.shape
#         device = G.device
#         dtype = G.dtype
#
#         f = torch.empty_like(G)
#         z = torch.zeros_like(G)
#         u = torch.zeros_like(G)
#
#         z_prev = torch.empty_like(G)
#         tmp = torch.empty_like(G)
#
#         rho = torch.full((M, 1), self.rho_base, device=device, dtype=dtype)
#         denom = rho + 2.0 * Z
#
#         budgets = (self.f_max_node - f_min.sum(dim=1, keepdim=True)).clamp_min(0.0)
#
#         tol_sq = self.tol * self.tol
#         rho_sq = rho.square()
#
#         for _ in range(self.max_iter):
#             z_prev.copy_(z)
#
#             # f-update
#             tmp.copy_(z)
#             tmp.sub_(u)
#             tmp.mul_(rho)
#             tmp.add_(G)
#             f.copy_(tmp)
#             f.div_(denom)
#
#             # z-update
#             tmp.copy_(f)
#             tmp.add_(u)
#             tmp.sub_(f_min)
#
#             z_proj = self.project_simplex(tmp, budgets)
#             z.copy_(z_proj)
#             z.add_(f_min)
#             z.clamp_(max=f_max)
#
#             # dual update
#             u.add_(f - z)
#
#             # residuals (squared norm, no sqrt)
#             res_r = (f - z).square().sum(dim=1)
#             res_s = ((z - z_prev) * rho_sq.sqrt()).square().sum(dim=1)
#
#             if res_r.max() <= tol_sq and res_s.max() <= tol_sq:
#                 break
#
#         return z