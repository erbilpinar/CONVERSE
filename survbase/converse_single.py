from __future__ import annotations

from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted

from survbase.base import SurvivalNNBase
from survbase.utils import build_mlp, make_interpolator
from survbase.ml_based_models import _deephit_nll

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering

# =====================================================================
# CONVERSE with single autoencoder
# =====================================================================

class _CONVERSE_single(nn.Module):

    def __init__(
        self,
        in_dim: int,
        *,
        latent_dim: int = 8,
        num_durations: int = 50,    
        n_clusters: int = 3,
        # enc/dec
        enc_layers: int = 2,
        enc_hidden: Union[int, Sequence[int]] = 64,
        dec_layers: int = 1,
        dec_hidden: Union[int, Sequence[int]] = 16,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        student_t_df: float = 1.0,      # α in paper (degrees of freedom)
        surv_head: nn.Module|nn.ModuleList|Any = None,
        variational: bool = False,
        random_state: int = 42,
    ):
        super().__init__()
        L = latent_dim
        self.L = L
        self.in_dim = in_dim
        self.K = n_clusters
        self.df = float(student_t_df)
        self.random_state = random_state

        def make_enc():
            if variational:
                output_dim = L * 2  # mean and log_var
            else:
                output_dim = L

            return build_mlp(
                input_dim=in_dim, output_dim=output_dim,
                num_layers=enc_layers, hidden_units=enc_hidden,
                activation=activation, dropout=dropout,
                batchnorm=batchnorm, residual=residual,
                use_attention=False, bias=bias, out_activation="identity",
            )

        def make_dec():
            return build_mlp(
                input_dim=L, output_dim=in_dim,
                num_layers=dec_layers, hidden_units=dec_hidden,
                activation=activation, dropout=0.0,
                batchnorm=False, residual=False,
                use_attention=False, bias=True, out_activation="identity",
            )
        # Siamese autoencoders
        self.enc = make_enc()
        self.dec = make_dec()

        # Cluster centers per view (seeded for reproducibility)
        torch.manual_seed(self.random_state)
        self.register_buffer("centers", torch.randn(self.K, L)*0.05)

        # survival backbone: use provided DeepHit head
        self.num_durations = int(num_durations)
        fuse_dim = latent_dim + in_dim
        self.fuse_dim = fuse_dim
        # Cluster-specific heads (ModuleList of K heads) or single shared head
        if isinstance(surv_head, nn.ModuleList):
            self.surv_heads = surv_head
            self.use_cluster_heads = True
        else:
            self.surv_head = surv_head
            self.use_cluster_heads = False
        self.variational = variational

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample z.
        Use mu directly during evaluation.
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        torch.manual_seed(self.random_state)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _student_t_sim(z: torch.Tensor, centers: torch.Tensor, df: float) -> torch.Tensor:
        """
        Student's t-kernel similarities per paper (Eq. 11 style).
        Returns soft labels q_{i,k} (rows sum to 1).
        """
        # z: (B,L), centers: (K,L)
        diff = z.unsqueeze(1) - centers.unsqueeze(0)      # (B,K,L)
        dist2 = (diff ** 2).sum(-1)                       # (B,K)
        # q ∝ (1 + dist2/α)^(-(α+1)/2)
        logits = -0.5 * (df + 1) * torch.log1p(dist2 / df)
        # Numeric safety
        logits = logits - logits.max(dim=1, keepdim=True).values
        q = torch.softmax(logits, dim=1)                  # (B,K)
        return q

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_params = self.enc(x)   # (B, 2*L)
        
        if self.variational:
            mu, log_var = z_params[:, :self.L], z_params[:, self.L:]

            # Sample z using reparameterization trick
            z = self._reparameterize(mu, log_var)  # (B, L)

            # Soft labels via Student-t similarities for the loss function, eq. (11)
            # use mu instead of z for stability
            q = self._student_t_sim(mu, self.centers, self.df)   # (B,K) #type: ignore

            # Fuse representation for survival head
            mean = mu

        else:
            z = z_params

            mu = z.new_zeros(z.shape)  # (B, L)
            log_var = z.new_zeros(z.shape)  # (B, L)

            # Soft labels via Student-t similarities for the loss function, eq. (11)
            # use mu instead of z for stability
            q = self._student_t_sim(z, self.centers, self.df)   # (B,K)

            # Fuse representation for survival head
            mean = z

        h = torch.cat([mean, x], dim=1)                     # (B, L + D)
        
        # Use cluster-specific heads if enabled
        if self.use_cluster_heads:
            # Get hard cluster assignments from averaged q
            cluster_idx = q.argmax(dim=1)  # (B,)
            
            # Apply appropriate head for each sample based on its cluster
            surv_logits = torch.zeros(h.size(0), self.num_durations, device=h.device)
            for k in range(self.K):
                mask = (cluster_idx == k)
                if mask.any():
                    surv_logits[mask] = self.surv_heads[k](h[mask])
        else:
            surv_logits = self.surv_head(h)                       # (B, T)
        
        # Required for the loss
        x_hat = self.dec(z)
        rec_ps = F.mse_loss(x_hat, x, reduction="none").mean(dim=1)  # (B,)

        if self.variational:
            # KLD(N(mu, var) || N(0, I)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        else:
            kld = z.new_zeros(z.size(0))
        
        return {
        "z": z,
        "mu": mu, "log_var": log_var, "kld_ps": kld, # VAE terms
        "x_hat": x_hat,
        "rec_ps": rec_ps,
        "q": q,
        "surv_logits": surv_logits,
        "centers": self.centers,
        }


class CONVERSE_single(SurvivalNNBase):
    def __init__(
        self,
        *,
        latent_dim: int = 8,
        # For Deephit
        num_durations: int = 50,
        num_layers: int = 2,
        hidden_units: Union[int, Sequence[int]] = 16,
        # enc/dec
        enc_layers: int = 1,
        enc_hidden: Union[int, Sequence[int]] = 8,
        dec_layers: int = 2,
        dec_hidden: Union[int, Sequence[int]] = 16,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = False,
        student_t_df: float = 1,
        variational: bool = True,
        # loss weights
        alpha_REC: float = 1e-1,      # reconstruction loss
        alpha_KLD: float = 1e-2,      # VAE KL divergence
        alpha_DVC: float = 1e-2,      # clustering pull-to-center (per view)
        alpha_IVCG: float = 1e-2,     # intra-view cluster-guided contrastive
        alpha_NLL: float = 1e-1,      # survival NLL
        alpha_RANK: float = 1e-2,     # survival ranking loss
        temperature: float = 0.1,
        # SPL (self-paced) — use batchwise soft gating by percentile
        spl_enable: bool = True,
        # training
        epochs: int = 1000,
        batch_size: int = 128,
        lr: float = 1e-3,
        lr_scheduler: str | None = None,
        weight_decay: float =  1e-5,
        patience: int = 30,
        val_fraction: float = 0.15,
        interpolation: str = "step",
        random_state: int = 42,
        device: str | None = None,
        pretraining: bool = False,
        clustering_method: str = "kmeans",
        n_clusters: int = 4,
        linkage: str = "complete",
        eps: float = 0.5,
        min_samples: int = 5,
        covariance_type: str = "full",
        affinity: str = "rbf",
        use_cluster_heads: bool = False,
        reg_covar: float = 1e-6,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            patience=patience,
            val_fraction=val_fraction,
            interpolation=interpolation,
            random_state=random_state,
            device=device,
        )
        self.num_durations = num_durations
        self.latent_dim = latent_dim

        # For deephit
        self.num_layers = num_layers
        self.hidden_units = hidden_units

        # arch
        self.enc_layers = enc_layers
        self.enc_hidden = enc_hidden
        self.dec_layers = dec_layers
        self.dec_hidden = dec_hidden
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.bias = bias
        self.student_t_df = student_t_df

        # losses
        self.alpha_REC = alpha_REC
        self.alpha_KLD = alpha_KLD
        self.alpha_DVC = alpha_DVC
        self.alpha_IVCG = alpha_IVCG
        self.alpha_NLL  = alpha_NLL
        self.alpha_RANK = alpha_RANK
        self.temperature = temperature

        # SPL
        self.spl_enable = spl_enable

        self.device = device
        self.pretraining = pretraining
        self.variational = variational

        # clustering parameters
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.eps = eps
        self.min_samples = min_samples
        self.covariance_type = covariance_type
        self.affinity = affinity
        self.algorithm = clustering_method

        self.use_cluster_heads = use_cluster_heads
        self.reg_covar = reg_covar

    # ---------- network ----------
    def _init_net(self, n_features: int) -> torch.nn.Module:
        fuse_dim = self.latent_dim + n_features

        # REUSE THE DEEPHIT HEAD ARCH
        # Create K separate survival heads (one per cluster)
        if self.use_cluster_heads:
            dh_heads = nn.ModuleList([
                build_mlp(
                    input_dim=fuse_dim,
                    output_dim=self.num_durations,
                    bias=self.bias,
                    num_layers=self.num_layers if hasattr(self, "num_layers") else 2,
                    hidden_units=self.hidden_units if hasattr(self, "hidden_units") else 128,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    activation=self.activation,
                    out_activation="identity",
                    residual=self.residual,
                    use_attention=getattr(self, "use_attention", False),
                )
                for _ in range(self.n_clusters)
            ])
        else:
            dh_heads = build_mlp(
                input_dim=fuse_dim,
                output_dim=self.num_durations,
                bias=self.bias,
                num_layers=self.num_layers if hasattr(self, "num_layers") else 2,
                hidden_units=self.hidden_units if hasattr(self, "hidden_units") else 128,
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                activation=self.activation,
                out_activation="identity",
                residual=self.residual,
                use_attention=getattr(self, "use_attention", False),
            )

        self.net_ = _CONVERSE_single(
            in_dim=n_features,
            latent_dim=self.latent_dim,
            num_durations=self.num_durations,
            n_clusters=self.n_clusters,
            enc_layers=self.enc_layers,
            enc_hidden=self.enc_hidden,
            dec_layers=self.dec_layers,
            dec_hidden=self.dec_hidden,
            activation=self.activation,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            residual=self.residual,
            bias=self.bias,
            student_t_df=self.student_t_df,
            surv_head=dh_heads,
            variational=self.variational,
            random_state=self.random_state,
        )
        return self.net_

    def initialize_centers(self, X: np.ndarray, batch_size: int = 256, random_state: int | None = None):
        # Check fo pretraining
        check_is_fitted(self, "net_")
        self.net_.eval()

        # encode in mini-batches to avoid OOM
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        n = X.shape[0]
        B = batch_size

        # collect all the latent representations from both of the autoencoders
        zs = []
        with torch.no_grad():
            for i in range(0, n, B):
                xb = torch.tensor(X[i:i+B], dtype=torch.float32, device=self.device)
                if self.variational:
                    z_params = self.net_.enc(xb)
                    z = z_params[:, :self.net_.L] #mu
                else:
                    z = self.net_.enc(xb)
          
                zs.append(z.cpu().numpy())
        Z = np.concatenate(zs, axis=0)    # (n, L)

        # clustering
        if self.algorithm == "kmeans":
            km = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=(random_state if random_state is not None else self.random_state))
            km.fit(Z)
            C   = km.cluster_centers_                        # (K, 2L)
            L   = self.latent_dim
        elif self.algorithm == "agglomerative":
            agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
            labels = agg.fit_predict(Z)

            # Compute centers directly
            C = np.vstack([
                Z[labels == k].mean(axis=0)
                for k in range(self.n_clusters)
            ]).astype(np.float32)

            L = self.latent_dim
        elif self.algorithm == "GaussianMixture":
            # Convert to float64 for numerical stability
            n_components = min(self.n_clusters, Z.shape[0])
            Z_float64 = Z.astype(np.float64)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,  # Add regularization
                random_state=(random_state if random_state is not None else self.random_state)
            )
            gmm.fit(Z_float64)
            C = gmm.means_.astype(np.float32)  # Convert back to float32
            L   = self.latent_dim

            # If reduced, pad back (optional)
            if n_components < self.n_clusters:
                idx = np.arange(self.n_clusters - n_components) % n_components
                C = np.vstack([C, C[idx]])

        elif self.algorithm == "SpectralClustering":
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
            sc = SpectralClustering(n_clusters=self.n_clusters, n_neighbors=min(10, Z.shape[0] - 1), affinity=self.affinity, random_state=(random_state if random_state is not None else self.random_state), assign_labels='discretize')
            labels = sc.fit_predict(Z)
            
            # DO NOT collapse K
            C = np.zeros((self.n_clusters, Z.shape[1]), dtype=np.float32)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    C[k] = Z[mask].mean(axis=0)
                else:
                    # fallback: random or small noise
                    C[k] = Z[np.random.randint(0, Z.shape[0])]
            
            L   = self.latent_dim
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

        # write into model buffers
        with torch.no_grad():
            self.net_.centers.copy_(torch.tensor(C, dtype=torch.float32, device=self.device))
    
    # ---------- targets ----------
    def _prepare_targets(self, y) -> Dict[str, Any]:
        # Same with DeepHit
        times = y["time"].astype(np.float32)
        events = y["event"].astype(np.float32)
        # Discrete grid (store upper edges)
        self.time_grid_ = np.linspace(0.0, times.max(), self.num_durations + 1, dtype=np.float32)[1:]
        bin_idx = np.searchsorted(self.time_grid_, times, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.num_durations - 1).astype(np.int64)
        return {"bin_idx": bin_idx, "event": events, "time": times}

    # ---------- loss pieces ----------
    @staticmethod
    def _recon_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # From the siamese autoencoder eq (3)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=1)  # per-sample

    @staticmethod
    def _cluster_pull_loss(z: torch.Tensor, q: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        hard = q.argmax(dim=1)
        c_exp = centers[hard]
        return ((z - c_exp) ** 2).sum(dim=1)

    @staticmethod
    def _info_nce(anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        # cosine sims
        anc = F.normalize(anchor, dim=-1)
        pos = F.normalize(pos, dim=-1)
        neg = F.normalize(neg, dim=-1)
        # For implementation simplicity, treat each row's positives as dot with its own pair in 'pos'
        # and negatives are row-wise in 'neg'.
        # Score: exp(sim/τ)
        s_pos = (anc * pos).sum(-1, keepdim=True) / temperature            # (N,1) numerator
        s_neg = (anc.unsqueeze(1) * neg).sum(-1) / temperature             # (N,M) denominator terms
        denom = torch.cat([s_pos, s_neg], dim=1)                           # (N, 1+M)
        log_prob = s_pos - torch.logsumexp(denom, dim=1, keepdim=True)     # (N,1)
        return -log_prob.mean()
    
    def _intra_view_cluster_guided(self, z: torch.Tensor, q: torch.Tensor, event: torch.Tensor, hard_override: torch.Tensor | None = None) -> torch.Tensor:
        
        B, K = q.shape
        #hard = q.argmax(dim=1)   # <--- hard labels here (Update 4)
        hard = hard_override if hard_override is not None else q.argmax(dim=1)
        cens = (event == 0)
        unc  = (event == 1)

        if cens.sum() == 0 or unc.sum() == 0:
            return z.new_tensor(0.0)

        z_c = z[cens]                  # embeddings of censored anchors (N_c, L)
        c_c = hard[cens]               # their cluster labels
        z_u = z[unc]                   # embeddings of uncensored candidates (N_u, L)
        c_u = hard[unc]                # their cluster labels
        
        pos_list, neg_list = [], []

        for i, c_idx in enumerate(c_c):
            same = (c_u == c_idx) # uncensored in the SAME cluster as this anchor
            if same.any():
                pos = z_u[same].mean(dim=0, keepdim=True)  # robust positive: mean of uncensored in same cluster
            else:
                # If no uncensored in same cluster in this batch, use the anchor itself (no-op)
                pos = z_c[i:i+1]  # fallback no-op
            pos_list.append(pos)

            neg = z_u[~same]
            if neg.shape[0] == 0:
                neg = z_c[i:i+1]  # fallback avoid empty
            neg_list.append(neg.unsqueeze(0))  # (1,M,L)

        pos = torch.cat(pos_list, dim=0)  # (Nc, L)
        # pad negatives in the batch to same M
        M = max(t.shape[1] for t in neg_list)
        neg_padded = []
        for t in neg_list:
            if t.shape[1] < M:
                pad = t.repeat(1, M // t.shape[1] + 1, 1)[:, :M, :]
                neg_padded.append(pad)
            else:
                neg_padded.append(t[:, :M, :])
        neg = torch.cat(neg_padded, dim=0)  # (Nc, M, L)

        return self._info_nce(z_c, pos, neg, self.temperature)
    
    @staticmethod
    def _deephit_rank_loss(pmf: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """
        Pairwise ranking (single-risk variant):
        For events i,j with t_i < t_j, encourage higher cumulative incidence for i at t_i.
        Loss = -log(sigmoid((F_i(t_i) - F_j(t_i)) / sigma))
        log-sigmoid variant that’s mathematically equivalent up to a monotonic transformation, and more stable for gradient descent.
        """
        # cumulative incidence F(t) over discrete bins
        F = torch.cumsum(pmf, dim=1)

        # keep only uncensored (event==1)
        uncensored_mask = (event == 1)
        if uncensored_mask.sum() < 2:
            return pmf.new_tensor(0.0)

        F_uncensored = F[uncensored_mask]
        # each person’s discrete event time index
        bi = bin_idx[uncensored_mask].clamp(0, pmf.size(1) - 1)

        num_uncensored = F_uncensored.size(0)

        # Get each person’s own CDF value at their event time
        Fi_ti = F_uncensored[torch.arange(num_uncensored, device=F_uncensored.device), bi]     # (Ne,)

        # Compare every subject against everyone else at t_i
        Fj_ti = F_uncensored[:, bi]

        # Build pairwise mask for t_i < t_j  <=>  bin_i < bin_j
        bi_col = bi.view(num_uncensored, 1)
        bj_row = bi.view(1, num_uncensored)
        comp_mask = (bi_col < bj_row)

        # Expand Fi(t_i) to matrix shape to compare against Fj(t_i)
        Fi_mat = Fi_ti.view(num_uncensored, 1).expand(num_uncensored, num_uncensored)              # (Ne,Ne)

        # Differences for valid pairs (i,j) with t_i < t_j
        diff = (Fi_mat - Fj_ti)[comp_mask]                     # (M,)

        # Logistic ranking loss
        return -torch.nn.functional.logsigmoid(diff / sigma).mean()

    def _assign_S_batch(self,
                    z: torch.Tensor,
                    centers: torch.Tensor) -> torch.Tensor:
        """
        Hard cluster assignments S for the current batch, with fixed centers M:
        k* = argmin_k ||z - c_k||^2
        Returns: hard indices (B,)
        """
        # (B,K) squared distances to each view's centers
        d = ((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        return d.argmin(dim=1)  # (B,)

    # ---------- master loss ----------
    def _loss_fn(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
        # Optional raw input (present when _forward/_forward_batch inject "_input")
        z = outputs["z"]
        mu = outputs["mu"]              # deterministic latents
        q = outputs["q"]
        logits = outputs["surv_logits"]             # (B, T)
        bin_idx = targets["bin_idx"]
        event = targets["event"]
        centers = outputs["centers"]

        # --- per-sample pieces ---
        # reconstruction terms (only if we have x)rec_sum = outputs["rec_ps"]
        rec_sum = outputs["rec_ps"]

        # Get KLD loss terms, if not variational just zeros
        kld_sum = outputs["kld_ps"]

        if self.pretraining:
            base_per_item = self.alpha_REC * rec_sum + self.alpha_KLD * kld_sum
            dvc = 0.0
            w = torch.ones_like(base_per_item)
            ivcg = 0.0

        else:
            # batch-level hard assignments S (fixed centers) for both views
            if self.variational:
                hard = self._assign_S_batch(mu, centers)
            else:
                hard = self._assign_S_batch(z, centers)

            c = centers[hard]; 
            if self.variational:
                dvc = ((mu - c) ** 2).sum(dim=1)
            else:
                dvc = ((z - c) ** 2).sum(dim=1)
            base_per_item = self.alpha_REC * rec_sum + self.alpha_KLD * kld_sum + self.alpha_DVC * dvc

            # SPL
            if not self.spl_enable:
                w = torch.ones_like(base_per_item)
            else:
                # Equation (7)
                Li = self.alpha_REC * rec_sum + self.alpha_KLD * kld_sum + self.alpha_DVC * dvc
                spl_mean, spl_std = Li.detach().mean(), Li.detach().std(unbiased=False)
                prog = getattr(self, "_epoch", 1) / max(getattr(self, "epochs", 1), 1)
                # Equation (8) Set the threshold λ = μ + progress·σ
                lam = spl_mean + prog * spl_std
                # Equation (6)
                w = (Li.detach() <= lam).float()

            if self.variational:
                tensor = mu
            else:
                tensor = z

            ivcg = self._intra_view_cluster_guided(tensor, q, event,  hard_override=hard)
            
        # Equation 5
        base = (w * base_per_item).mean() 

        # survival term
        nll = _deephit_nll(logits, bin_idx, event)
        probs = F.softmax(logits, dim=1)
        rank = self._deephit_rank_loss(probs, bin_idx, event, sigma=0.1)

        # Take the all mean here
        loss = base + self.alpha_NLL * nll + self.alpha_RANK * rank + self.alpha_IVCG * ivcg
        
        return loss

    # ---------- prediction ----------
    @torch.no_grad()
    def predict(self, X, num_times: int = 1000):
        # Same as DeepHit
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            out = self.net_(xb)
            logits = out["surv_logits"].cpu().numpy()     # (n, T_d)
        
        # Handle potential numerical issues
        # Replace any non-finite logits with a large negative value
        if not np.all(np.isfinite(logits)):
            logits = np.where(np.isfinite(logits), logits, -1e10)
        
        # pmf and survival on discrete grid edges
        logits_max = logits.max(axis=1, keepdims=True)
        # Ensure max is finite
        logits_max = np.where(np.isfinite(logits_max), logits_max, 0.0)
        logits = logits - logits_max
        
        pmf = np.exp(logits)
        pmf_sum = pmf.sum(axis=1, keepdims=True)
        # Protect against division by zero or near-zero
        pmf_sum = np.maximum(pmf_sum, 1e-10)
        pmf = pmf / pmf_sum
        
        surv_d = 1.0 - np.cumsum(pmf, axis=1)
        surv_d = np.clip(surv_d, 0.0, 1.0)
        
        # Ensure survival curves are finite
        if not np.all(np.isfinite(surv_d)):
            surv_d = np.where(np.isfinite(surv_d), surv_d, 0.0)

        times = np.linspace(0.0, self.max_time_, num_times).astype(np.float32)
        fn = make_interpolator(surv_d, self.time_grid_, mode=self.interpolation)
        result = fn(times)
        
        # Final check: ensure output is finite
        if not np.all(np.isfinite(result)):
            result = np.where(np.isfinite(result), result, 0.0)
        
        return result

    # ---------- prediction for saved models and notebooks ----------
    @torch.no_grad()
    def encode(self, X):
        """
        Extract latent representations from the CONVERSE_single model.
        Returns the latent representation.
        
        For variational models, returns mu (mean) instead of sampled z.
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            out = self.net_(xb)
            
            if self.variational:
                z = out["mu"].cpu().numpy()
            else:
                z = out["z"].cpu().numpy()
            
            return z
    
    def predict_clusters(self, X):
        """
        Extract cluster assignments from the CONVERSE_single model.
        Returns hard cluster labels based on soft assignment.
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            out = self.net_(xb)
            q = out["q"].cpu().numpy()
            
            # Average soft assignments and get hard labels
            return q.argmax(axis=1)
