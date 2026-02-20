from __future__ import annotations
from typing import Any, Dict, Sequence, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted

from survbase.base import SurvivalNNBase
from survbase.nonparametric import breslow_estimator
from survbase.utils import build_mlp, make_interpolator

import torch.nn as nn
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans

from survbase.statistical_based_models import cox_partial_log_likelihood
from survbase.ml_based_models import _deephit_nll

# =====================================================================
# SCA (Survival Cluster Analysis) — non-federated
# =====================================================================
class _SCANet(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        *,
        latent_dim: int = 16,
        K: int = 25,
        nu: float = 1.0,  # Student-t dof
        # Noise parameters
        noise_dist: str = "uniform",   # "uniform" or "gaussian"
        noise_loc: float = 0.0,        # shift
        noise_scale: float = 1.0,      # scale
        standardize_uniform: bool = False,  # map U(0,1) → mean 0, var 1
        # encoder
        enc_layers: int = 2,
        enc_hidden: Union[int, Sequence[int]] = 128,
        enc_activation: str = "relu",
        enc_dropout: float = 0.1,
        enc_batchnorm: bool = False,
        enc_residual: bool = False,
        enc_bias: bool = True,
        # generator
        gen_layers: int = 2,
        gen_hidden: Union[int, Sequence[int]] = 128,
        gen_activation: str = "relu",
        gen_dropout: float = 0.1,
        gen_batchnorm: bool = False,
        gen_residual: bool = False,
        gen_bias: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.K = K
        self.nu = float(nu)
        # Noise parameters
        self.noise_dist = noise_dist.lower()
        self.noise_loc = float(noise_loc)
        self.noise_scale = float(noise_scale)
        self.standardize_uniform = bool(standardize_uniform)
        
        self.encoder = build_mlp(
            input_dim=n_features,
            output_dim=latent_dim,
            num_layers=enc_layers,
            hidden_units=enc_hidden,
            activation=enc_activation,
            dropout=enc_dropout,
            batchnorm=enc_batchnorm,
            residual=enc_residual,
            use_attention=False,
            bias=enc_bias,
            out_activation="identity",
        )
        # generator g_θ takes [z, ε] and outputs a positive time via softplus
        self.generator = build_mlp(
            input_dim=latent_dim + 1,
            output_dim=1,
            num_layers=gen_layers,
            hidden_units=gen_hidden,
            activation=gen_activation,
            dropout=gen_dropout,
            batchnorm=gen_batchnorm,
            residual=gen_residual,
            use_attention=False,
            bias=gen_bias,
            out_activation="identity",
        )

        # global mixture: unconstrained logits for π and K centers c_k in latent space
        self.pi_logits = torch.nn.Parameter(torch.zeros(K))               # π ~ softmax(pi_logits)
        self.centers = torch.nn.Parameter(torch.randn(K, latent_dim) * 0.1)


    # Centralized noise sampler
    def _sample_eps(self, B: int, S: int, device) -> torch.Tensor:
        if self.noise_dist == "uniform":
            e = torch.rand(B, S, device=device)  # U(0,1)
            if self.standardize_uniform:
                # map to mean=0, var=1: (u - 0.5) * sqrt(12)
                e = (e - 0.5) * 3.4641016151377544
            e = e * self.noise_scale + self.noise_loc
        elif self.noise_dist == "gaussian":
            e = torch.randn(B, S, device=device) * self.noise_scale + self.noise_loc
        else:
            raise ValueError("noise_dist must be 'uniform' or 'gaussian'")
        return e

    @staticmethod
    def _student_t_log_kernel(z: torch.Tensor, centers: torch.Tensor, nu: float) -> torch.Tensor:
        """
        Return log kernel (up to additive const) of Student-t_ν(z | c_k, I) for each k.

        z:        (B, L)
        centers:  (K, L)
        returns:  (B, K) with log t_ν(z_n | c_k) up to a constant.
        """
        B, L = z.shape
        # (B,1,L) - (1,K,L) -> (B,K,L)
        diff = z.unsqueeze(1) - centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(-1)  # (B, K)
        # log t up to constant:  -0.5*(ν+L)*log(1 + dist2/ν)
        return -0.5 * (nu + L) * torch.log1p(dist2 / nu)

    def forward(self, x: torch.Tensor, eps: torch.Tensor | None = None, mc_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        x: (B, p), eps: (B, S) ~ U(0,1) if provided. If None, sampled internally.
        returns dict with:
          z: (B, L), encoded latent
          pi: (K,), mixture weights for each cluster
          log_k: (B, K) = logp(z|c_k,v), Score how well each cluster explains z, we do student-t log kernel because of the assumption
          alpha: (B, K)  responsibilities ∝ π_k t_v(z | c_k)
          t_hat: (B, S)  sampled times from g_θ
        """
        B = x.shape[0]
        S = mc_samples
        z = self.encoder(x)  # (B, L)

        # How likely z is a student-t distribution k
        log_k = self._student_t_log_kernel(z, self.centers, self.nu)  # (B, K)
        
        # Soft assignments, eq (5)
        log_pi = torch.log_softmax(self.pi_logits, dim=0)       # (K,)
        log_alpha = log_k + log_pi.unsqueeze(0)                 # (B, K)
        alpha = torch.softmax(log_alpha, dim=1)                 # (B, K)

        # Noise: gaussian or uniform
        if eps is None:
            eps = self._sample_eps(B, S, device=x.device)

        # Get the preeicted time
        z_rep = z.unsqueeze(1).expand(B, S, z.shape[1]).reshape(B * S, -1)
        g_in = torch.cat([z_rep, eps.reshape(B * S, 1)], dim=1)
        t_hat = torch.nn.functional.softplus(self.generator(g_in)).reshape(B, S)
        
        return {
            "z": z,                                         # latent embeddings r_ψ(x)
            "pi": torch.softmax(self.pi_logits, dim=0),     # global mixture weights π
            "log_k": log_k,                                 # per-cluster log t-kernel
            "alpha": alpha,                                 # responsibilities q(u=k|x)
            "t_hat": t_hat,                                 # MC time samples from g_θ
        }


class SCA(SurvivalNNBase):
    def __init__(
        self,
        *,
        K: int = 25,
        gamma0: float = 4.0,
        nu: float = 1.0,                 # Student-t dof in latent (ν=1 used in paper for heavy tails)
        latent_dim: int = 16,
        mc_samples: int = 4,             # Monte-Carlo time samples per item for training
        # Noise parameters
        noise_dist: str = "uniform",
        noise_loc: float = 0.0,
        noise_scale: float = 1.0,
        standardize_uniform: bool = False,
        # encoder/gen architecture
        enc_layers: int = 2,
        enc_hidden: Union[int, Sequence[int]] = 128,
        gen_layers: int = 2,
        gen_hidden: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        # loss weights
        lambda_acc: float = 1.0, #these are all set 1 in the paper
        lambda_cal: float = 1.0,
        lambda_dp: float = 1.0,
        # training
        epochs: int = 120,
        batch_size: int | None = None,
        lr: float = 1e-3,
        lr_scheduler: str | None = None,
        weight_decay: float = 0.0,
        patience: int = 10,
        val_fraction: float = 0.15,
        interpolation: str = "step",
        random_state: int = 42,
        device: str | None = None,
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
        self.K = K
        self.gamma0 = float(gamma0)
        self.nu = float(nu)

        # Noise parameters
        self.noise_dist = noise_dist
        self.noise_loc = noise_loc
        self.noise_scale = noise_scale
        self.standardize_uniform = standardize_uniform
        
        self.latent_dim = latent_dim
        self.mc_samples = int(mc_samples)

        self.enc_layers = enc_layers
        self.enc_hidden = enc_hidden
        self.gen_layers = gen_layers
        self.gen_hidden = gen_hidden
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.bias = bias

        self.lambda_acc = lambda_acc
        self.lambda_cal = lambda_cal
        self.lambda_dp = lambda_dp
        self.device = device

    # ---------- network ----------
    def _init_net(self, n_features: int) -> torch.nn.Module:
        return _SCANet(
            n_features,
            latent_dim=self.latent_dim,
            K=self.K,
            nu=self.nu,
            noise_dist=self.noise_dist,
            noise_loc=self.noise_loc,
            noise_scale=self.noise_scale,
            standardize_uniform=self.standardize_uniform,
            enc_layers=self.enc_layers,
            enc_hidden=self.enc_hidden,
            enc_activation=self.activation,
            enc_dropout=self.dropout,
            enc_batchnorm=self.batchnorm,
            enc_residual=self.residual,
            enc_bias=self.bias,
            gen_layers=self.gen_layers,
            gen_hidden=self.gen_hidden,
            gen_activation=self.activation,
            gen_dropout=self.dropout,
            gen_batchnorm=self.batchnorm,
            gen_residual=self.residual,
            gen_bias=self.bias,
        )

    # ---------- targets ----------
    def _prepare_targets(self, y) -> Dict[str, Any]:
        return {
            "time": y["time"].astype(np.float32),
            "event": y["event"].astype(np.float32),
        }

    # ---------- Helpers for the loss function ----------
    @staticmethod
    def _accuracy_loss(t_hat: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Eq. (10):
        censored   (event=0):  E_ε [ max(0, t - gθ(rψ(x), ε)) ]
        uncensored (event=1):  E_ε [ | t - gθ(rψ(x), ε) | ]
        t_hat: (B,S) samples from gθ  (or (B,) which we treat as S=1)
        time : (B,)
        event: (B,)  in {0,1}
        """
        if t_hat.ndim == 1:                       # make it (B,S)
            t_hat = t_hat.unsqueeze(1)
        # broadcast time to (B,S)
        time_ = time.unsqueeze(1)

        # per-sample losses, then MC-average over S
        l1_mc    = (time_ - t_hat).abs().mean(dim=1)          # (B,)
        hinge_mc = torch.relu(time_ - t_hat).mean(dim=1)      # (B,)

        # group-wise means (the two expectations in Eq. 10)
        n_evt = torch.clamp(event.sum(), min=1.0)
        n_cen = torch.clamp((1.0 - event).sum(), min=1.0)

        loss_evt = (l1_mc * event).sum() / n_evt              # E over non-censored
        loss_cen = (hinge_mc * (1.0 - event)).sum() / n_cen   # E over censored
        return loss_evt + loss_cen


    @staticmethod
    def _SPKM(times_sorted: torch.Tensor, T_hat: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        S_PKM in equation (12) of the paper. Used in calibration loss
        times_sorted: (M_t,) strictly increasing distinct observed times (both censored & uncensored)
        T_hat: (B,) per-subject times (either observed or model-sampled)
        event: (B,) 1 if uncensored, 0 if censored (ground truth)
        returns: (M_t,) estimated Ŝ(t_i)
        """
        # Heaviside H(b) = 0.5 * (sign(b) + 1)
        def H(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * (torch.sign(x) + 1.0)

        M_t = times_sorted.shape[0]
        M = T_hat.shape[0]

        # Prepend t0=0 for the recursion; handle piecewise via loops (M_t is small per batch)
        t_prev = torch.tensor(0.0, device=T_hat.device)
        s_prev = torch.tensor(1.0, device=T_hat.device)
        out = []
        for i in range(M_t):
            t_i = times_sorted[i]
            # numerator: sum_{n: event=1} [H(T̂n - t_{i-1}) - H(T̂n - t_i)]
            num = (((H(T_hat - t_prev) - H(T_hat - t_i)) * event)).sum()
            # denominator: B - sum_n H(t_{i-1} - T̂n)
            denom = (M - H(t_prev - T_hat).sum()).clamp_min(1.0)
            s_i = (1.0 - num / denom) * s_prev
            out.append(s_i)
            s_prev = s_i
            t_prev = t_i
        return torch.stack(out, dim=0)

    def _calibration_loss(self, t_hat: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        """
        Eq. (11)–(12): average |S_emp(t_i) - S_model(t_i)|.
        Use a single MC collapse (or the mean) for T̂ to form Ŝ.
        """
        # collapse MC samples to a representative draw (mean works well and is smooth)
        if t_hat.ndim == 2:
            t_hat = t_hat.mean(dim=1)  # (B,)

        # distinct observed times in the minibatch (keep numerical stability)
        T = torch.unique(time.detach(), sorted=True)
        S_emp = self._SPKM(T, time, event)            # use observed times
        S_model = self._SPKM(T, t_hat, event)          # plug in model-implied times
        return (S_emp - S_model).abs().mean()

    
    @staticmethod
    def _kl_dirichlet(q_alpha: torch.Tensor, p_alpha: torch.Tensor) -> torch.Tensor:
        """
        This is the function (8) in the paper.
        KL( Dir(q_alpha) || Dir(p_alpha) ), vectorized over batch dim if provided.

        q_alpha, p_alpha: (..., K)
        returns: (...) scalar KL per leading batch entry
        """
        q_sum = q_alpha.sum(-1)
        p_sum = p_alpha.sum(-1)
        lgamma = torch.lgamma
        digamma = torch.digamma

        t1 = lgamma(q_sum) - lgamma(p_sum)
        t2 = (lgamma(p_alpha).sum(-1) - lgamma(q_alpha).sum(-1)) 
        t3 = ((q_alpha - p_alpha) * (digamma(q_alpha) - digamma(q_sum).unsqueeze(-1))).sum(-1)
        return t1 + t2 + t3 

    def _expected_stick_breaking(self, K: int, gamma0: float, device, dtype):
        """
        Expected stick-breaking weights under V_k ~ Beta(1, gamma0):
        E[V_k] = 1 / (1 + gamma0)
        π_k = E[V_k] * ∏_{l<k} (1 - E[V_l])
        """
        ev = 1.0 / (1.0 + gamma0)
        pi = []
        running = 1.0
        for _ in range(K):
            pi.append(ev * running)
            running *= (1.0 - ev)
        return torch.tensor(pi, device=device, dtype=dtype)  # (K,)

    def _mixture_dp_loss(self, log_k: torch.Tensor, pi_mix: torch.Tensor, alpha_mix: torch.Tensor | None) -> torch.Tensor:
        """
        Implements Eq. (5)–(8) from the paper:
        - Mixture view:   q(u|x) ∝ π_mix * t_ν(z|c_k) → α_mix, ξ = 1/K + Σ_n α_mix
        - DP view:        p(u|x) ∝ π_dp  * t_ν(z|c_k) → α_dp,  γ = γ0  + Σ_n α_dp
        - Loss: KL( Dir(ξ) || Dir(γ) ) + (optional) mixture NLL
        """
        B, K = log_k.shape
        # mixture-view responsibilities (if not provided)
        if alpha_mix is None:
            log_pi_mix = torch.log(pi_mix + 1e-12)                    # (K,)
            alpha_mix = torch.softmax(log_k + log_pi_mix.unsqueeze(0), dim=1)  # (B,K)

        # DP-view: expected stick-breaking weights π_dp from γ0 (Eq. (6) text; E[V_k] trick)
        pi_dp = self._expected_stick_breaking(K, self.gamma0, device=log_k.device, dtype=log_k.dtype)  # (K,)
        log_pi_dp = torch.log(pi_dp + 1e-12)
        alpha_dp  = torch.softmax(log_k + log_pi_dp.unsqueeze(0), dim=1)  # (B,K)

        # Dirichlet params (Eq. (5) and (6))
        counts_q = alpha_mix.sum(dim=0)               # mixture view counts → ξ
        counts_p = alpha_dp.sum(dim=0)                # DP view counts     → γ
        ksi = counts_q + (1.0 / K)                    # ξ_k = 1/K + Σ α_mix
        gam = counts_p + self.gamma0                  # γ_k = γ0  + Σ α_dp

        # KL( Dir(ξ) || Dir(γ) ) (Eq. (8))
        kl_dp = self._kl_dirichlet(ksi, gam)          # closed form Dirichlet–Dirichlet KL

        # Optional stabilizer: mixture negative log-likelihood (not in paper, but helpful)
        log_pi_mix = torch.log(pi_mix + 1e-12)
        log_mix = torch.logsumexp(log_k + log_pi_mix.unsqueeze(0), dim=1)  # (B,)
        nll_mix = -log_mix.mean()

        return nll_mix + kl_dp

    def _loss_fn(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
        z, log_k, alpha, pi, t_hat = outputs["z"], outputs["log_k"], outputs["alpha"], outputs["pi"], outputs["t_hat"]
        time = targets["time"]
        event = targets["event"]

        # Accuracy (Eq. 10)
        loss_acc = self._accuracy_loss(t_hat, time, event)

        # Calibration (Eq. 11–12)
        loss_cal = self._calibration_loss(t_hat, time, event)

        # Mixture vs DP (Eq. 5–8): single correct call, with (log_k, pi_mix, alpha_mix)
        loss_mixdp = self._mixture_dp_loss(log_k, pi, alpha)

        return self.lambda_acc * loss_acc + self.lambda_cal * loss_cal + self.lambda_dp * loss_mixdp


    # ---------- predictions ----------
    @torch.no_grad()
    def predict(self, X, num_times: int = 1000, n_mc: int = 1000) -> np.ndarray:
        """
        Return per-sample survival curves via MC:
          S_n(t) ≈ P(T̂_n > t) = mean_s [ 1{ T̂_n^{(s)} > t } ], t on a uniform grid [0, max_time_].
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        x = torch.tensor(X, dtype=torch.float32, device=self.device)
        B = x.shape[0]

        # Eq. (4) sample x n_mc times to get T̂
        eps = torch.rand(B, n_mc, device=self.device)
        out = self.net_(x, eps=eps, mc_samples=n_mc)
        t_hat = out["t_hat"].detach().cpu().numpy()  # (B, n_mc)

        times = np.linspace(0.0, self.max_time_, num_times, dtype=np.float32)
        # S(t) ≈ fraction of samples greater than t
        S = (t_hat[:, :, None] > times[None, None, :]).mean(axis=1)  # (B, T)
        return S.astype(np.float32)


# =====================================================================
# VaDeSC — Variational Deep Survival Clustering — non-federated
# =====================================================================
def _weibull_loglik(t, d, lam, k, eps: float = 1e-8):
    t = t.clamp_min(eps)
    lam = lam.clamp_min(eps)
    k = k.clamp_min(eps)

    # Simplified Weibull log-likelihood
    log_t = torch.log(t)
    log_lam = torch.log(lam)
    # This term is common to both the PDF and Survival function logs
    xk = ((t / lam).clamp_min(eps)).pow(k)

    # k/lambda * (t/lambda)^(k-1) exp(-(t/lambda)^k)
    log_f_t = torch.log(k) - log_lam + (k - 1.0) * (log_t - log_lam) - xk

    # log S(t|lam, k)
    log_S_t = -xk

    # d is the event indicator (1 for uncensored, 0 for censored)
    # The total log-likelihood is the sum of log f(t) for uncensored
    # and log S(t) for censored data.
    return d * log_f_t + (1.0 - d) * log_S_t


class _VaDeSCNet(nn.Module):
    """
    VAE with Gaussian Mixture prior (K components) + cluster-specific Weibull survival head.

    - Encoder q(z|x) = N(mu(x), diag(sigma^2(x)))
    - Decoder p(x|z): Gaussian with learnable per-feature log-variance (stable for tabular)
      (set 'recon_distribution="bernoulli"' if your X is in [0,1] and binary-ish)
    - Prior p(z|c) = N(mu_k, diag(sigma_k^2)), p(c)=Cat(pi)
    - Survival p(t|z,c): Weibull with scale lambda = softplus(z^T beta_c), shape k>0 (global)
    """
    def __init__(
        self,
        n_features: int,
        *,
        latent_dim: int = 16,
        K: int = 4,
        # encoder/decoder
        enc_layers: int = 2,
        enc_hidden: Union[int, Sequence[int]] = 128,
        dec_layers: int = 2,
        dec_hidden: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        recon_distribution: str = "gaussian",   # "gaussian" or "bernoulli"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_features = n_features
        self.K = K
        self.recon_distribution = recon_distribution.lower()

        # Encoder -> (mu, logvar) of q(z|x)
        self.encoder = build_mlp(
            input_dim=n_features,
            output_dim=2 * latent_dim,
            num_layers=enc_layers,
            hidden_units=enc_hidden,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            residual=residual,
            use_attention=False,
            bias=bias,
            out_activation="identity",
        )

        # Decoder mean  E[x|z]
        self.decoder_mean = build_mlp(
            input_dim=latent_dim,
            output_dim=n_features,
            num_layers=dec_layers,
            hidden_units=dec_hidden,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            residual=residual,
            use_attention=False,
            bias=bias,
            out_activation="identity" if self.recon_distribution == "gaussian" else "sigmoid",
        )
        # For Gaussian recon: per-feature log-variance (free parameter, stable)
        if self.recon_distribution == "gaussian":
            self.dec_logvar = nn.Parameter(torch.zeros(n_features))

        # GMM prior params (trainable)
        self.pi_logits = nn.Parameter(torch.zeros(K))                          # (K,)
        self.prior_mu = nn.Parameter(torch.randn(K, latent_dim) * 0.05)        # (K,L)
        self.prior_logvar = nn.Parameter(torch.zeros(K, latent_dim))           # (K,L)

        # Survival head: beta per cluster (L,) and global shape k
        self.beta = nn.Parameter(torch.randn(K, latent_dim) * 0.05)            # (K,L)
        
        # λ controls the time scale of events. 
        # If λ starts tiny,  S(t)=exp⁡(−(t/λ)k) crashes to 0; if huge, it’s flat.
        self.beta0 = nn.Parameter(torch.full((K,), np.log(np.expm1(1.0)).astype(np.float32)))
        self._k_raw = nn.Parameter(torch.tensor(np.log(np.expm1(1.0)).astype(np.float32)))
                
    # ----- helpers -----
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        L = self.latent_dim
        mu = h[:, :L] #mean
        logvar = h[:, L:] #variance
        return {"mu": mu, "logvar": logvar}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Sample z ~ N(mu, diag(var)) via reparameterization
        # z = μ + σ ⋅ ϵ, with ϵ ∼ N(0, I).
        std = torch.exp(0.5 * logvar) # σ = exp(0.5 log σ²)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = self.decoder_mean(z)
        if self.recon_distribution == "gaussian":
            logvar = self.dec_logvar  # (D,)
            return {"x_mean": mean, "x_logvar": logvar}
        else:
            # Bernoulli: mean is in (0,1)
            return {"x_mean": mean}

    def prior_log_prob_z_given_c(self, z: torch.Tensor) -> torch.Tensor:
        # log Gaussian density formula
        # Log p(z|c) ~ N(z; μ_c, Σ_c)
        B, L = z.shape
        z_exp = z.unsqueeze(1)                # (B,1,L)
        mu = self.prior_mu.unsqueeze(0)       # (1,K,L)
        logvar = self.prior_logvar.unsqueeze(0) # (1,K,L)
        var = torch.exp(logvar)               # (1,K,L)

        # z_j − μ_cj
        diff2 = (z_exp - mu).pow(2)           # (B,K,L)
        # (z_j−μ_cj)/σ_cj
        quad = (diff2 / var).sum(dim=2)       # (B,K)
        # log(σ_cj^2)
        logdet = logvar.sum(dim=2)            # (1,K)
        # L*log(2π)
        const = L * np.log(2.0 * np.pi)       # scalar

        #logN(z;μ,Σ) = −1/2​[Llog(2π) + log∣Σ∣ + Σ(z−μ)^2/σ^2]
        return -0.5 * (const + logdet + quad) # (B,K)

    def k_shape(self) -> torch.Tensor:
        k = F.softplus(self._k_raw) + 1e-6
        return k

    def lambda_scale(self, z: torch.Tensor) -> torch.Tensor:
        lam_raw = z @ self.beta.t() + self.beta0[None, :]   # (B,K)
        lam = F.softplus(lam_raw) + 1e-6                   
        return lam

    def forward(self, x: torch.Tensor, *, sample_z: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.encode(x)
        z = self.reparameterize(enc["mu"], enc["logvar"]) if sample_z else enc["mu"]
        dec = self.decode(z)

        # --- quantities needed by the ELBO, precomputed here ---
        # log p(z|c) for all c  -> (B, K)
        log_pz_c = self.prior_log_prob_z_given_c(z)

        # log p(c) -> (1, K) broadcastable
        log_pc = torch.log_softmax(self.pi_logits, dim=0).unsqueeze(0)

        # lambda(z,c) -> (B, K)  and global k (scalar tensor)
        lam = self.lambda_scale(z)
        k = self.k_shape()  # scalar > 0

        out = {
            "mu": enc["mu"],            # (B, L), mean of q(z|x)
            "logvar": enc["logvar"],    # (B, L), logvar of q(z|x)
            "z": z,                     # (B, L), latent
            **dec,                      # x_mean (+ x_logvar if gaussian)
            "_input": x,                # echo minibatch for recon loss
            # survival priors for the ELBO (loss function)
            "log_pz_c": log_pz_c,       # (B, K), log prior densities logp(z|c)
            "log_pc": log_pc,           # (1, K), log prior cluster probs logp(c)
            "lam": lam,                 # (B, K), weibull scale per cluster
            "k": k,                     # scalar tensor, global weibull shape
        }
        return out


class VaDeSC(SurvivalNNBase):
    """
    Variational Deep Survival Clustering (VaDeSC).

    - Fully PyTorch; same API as other models in this codebase.
    - ELBO = E_q(z|x)[ log p(x|z) + E_{p(c|z,t)} log p(t|z,c) + E_{p(c|z,t)}(log p(z|c)+log p(c)-log p(c|z,t)) ] - E_q log q(z|x)
      (approximated with a single z sample per item)
    - p(t|z,c) is Weibull with cluster-specific scale and global shape.
    """

    def __init__(
        self,
        *,
        K: int = 4,
        latent_dim: int = 16,
        # encoder/decoder
        enc_layers: int = 2,
        enc_hidden: Union[int, Sequence[int]] = 128,
        dec_layers: int = 2,
        dec_hidden: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        recon_distribution: str = "gaussian",   # "gaussian" or "bernoulli"
        # training
        epochs: int = 200,
        batch_size: int | None = None,
        lr: float = 1e-3,
        lr_scheduler: str | None = None,
        weight_decay: float = 0.0,
        patience: int = 15,
        val_fraction: float = 0.15,
        interpolation: str = "step",
        random_state: int = 42,
        device: str | None = None,
        # loss weights (you can tune these if needed)
        lambda_recon: float = 1.0,
        lambda_surv: float = 1.0,
        lambda_kl: float = 1.0,
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
        # arch / priors
        self.K = K
        self.latent_dim = latent_dim
        self.enc_layers = enc_layers
        self.enc_hidden = enc_hidden
        self.dec_layers = dec_layers
        self.dec_hidden = dec_hidden
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.bias = bias
        self.recon_distribution = recon_distribution

        # weights
        self.lambda_recon = lambda_recon
        self.lambda_surv = lambda_surv
        self.lambda_kl = lambda_kl
        self.device = device

    # ---------- network ----------
    def _init_net(self, n_features: int) -> nn.Module:
        return _VaDeSCNet(
            n_features=n_features,
            latent_dim=self.latent_dim,
            K=self.K,
            enc_layers=self.enc_layers,
            enc_hidden=self.enc_hidden,
            dec_layers=self.dec_layers,
            dec_hidden=self.dec_hidden,
            activation=self.activation,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            residual=self.residual,
            bias=self.bias,
            recon_distribution=self.recon_distribution,
        )

    # ---------- targets ----------
    def _prepare_targets(self, y) -> Dict[str, Any]:
        # --- NEW: learn a time scaling factor once ---
        t = y["time"].astype(np.float32)
        if not hasattr(self, "_time_scale_"):
            # choose a robust scale; 75th percentile or median both work
            s = np.nanpercentile(t[t > 0], 75).astype(np.float32)
            if not np.isfinite(s) or s <= 0:
                s = np.float32(1.0)
            self._time_scale_ = float(s)

        t_scaled = t / self._time_scale_
        return {"time": t_scaled, "event": y["event"].astype(np.float32)}


    # ---------- ELBO components ----------
    def _recon_loglik(self, x: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Equation (7) in the paper: log p(x|z)
        if self.recon_distribution == "gaussian":
            x_mean = out["x_mean"]
            x_logvar = out["x_logvar"].unsqueeze(0).expand_as(x_mean)

            # Formula for Gaussian log-likelihood
            # -0.5 * (log(2pi) + log(var) + (x - mean)^2 / var)
            log_px = -0.5 * (
                np.log(2.0 * np.pi) + x_logvar + (x - x_mean).pow(2) / torch.exp(x_logvar)
            )
            return log_px.sum(dim=-1)
        else:
            # Bernoulli
            x_mean = out["x_mean"].clamp(1e-6, 1.0 - 1e-6)
            return (x * torch.log(x_mean) + (1.0 - x) * torch.log(1.0 - x_mean)).sum(dim=-1)
    
    def _posterior_c_given_zt(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        d: torch.Tensor,
        *,
        log_pz_c: torch.Tensor,
        log_pc: torch.Tensor,
        lam: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        p(c|z,t) ∝ p(t|z,c) p(z|c) p(c) eq. (4) in the paper.
        All inputs are provided directly (no self.net_ usage).
        Shapes:
        z: (B, L), t: (B,), d: (B,)
        log_pz_c: (B, K), log_pc: (1, K), lam: (B, K), k: scalar tensor
        """       
        log_pt = _weibull_loglik(t.unsqueeze(1), d.unsqueeze(1), lam, k)  # (B,K)
        logits = log_pt + log_pz_c + log_pc      # (B, K)
        logits = logits - logits.max(dim=1, keepdim=True).values #avoid numerical issues, overflow
        post = torch.softmax(logits, dim=1) # divides by sum_c exp(logits_c)
        return post.clamp_min(1e-12)


    # ---------- loss ----------
    def _loss_fn(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
        """
        Negative ELBO (mean over batch).
        Equation (6) in the paper.
        All quantities come from `outputs` (no self.net_ access).
        """
        x = outputs["_input"]                  # (B, D)
        t = targets["time"]                    # (B,)
        d = targets["event"]                   # (B,)

        z = outputs["z"]                       # (B, L)
        mu = outputs["mu"]                     # (B, L)
        logvar = outputs["logvar"]             # (B, L)

        # reconstruction log-likelihood, log p(x|z), (term 1)
        log_px = self._recon_loglik(x, outputs)  # (B,)

        # precomputed in forward
        log_pz_c = outputs["log_pz_c"]         # (B, K)
        log_pc   = outputs["log_pc"]           # (1, K)
        lam      = outputs["lam"]              # (B, K)
        k        = outputs["k"]                # scalar tensor

        # p(c|z,t), (required for term 2)
        p_c = self._posterior_c_given_zt(z, t, d, log_pz_c=log_pz_c, log_pc=log_pc, lam=lam, k=k)  # (B,K)

        # survival term log p(t|z,c), (term 2), eq(8)
        log_pt = _weibull_loglik(t.unsqueeze(1), d.unsqueeze(1), lam, k)  # (B,K)
        surv_term = (p_c * log_pt).sum(dim=1)                              # (B,)

        # categorical prior/regularizer: E_{p(c|z,t)}[ log p(z|c) + log p(c) - log p(c|z,t) ]
        # Eq. (9)+(10): clustering/prior piece (term 3 and term 4)
        # E_{p(c|z,t)}[log p(z|c) + log p(c) - log p(c|z,t)]
        # cat_term = (p_c * (log_pz_c + log_pc - p_c.clamp_min(1e-12).log())).sum(dim=1)  # (B,)
        cat_term = (p_c * (log_pz_c + log_pc - torch.log(p_c))).sum(dim=1)


        # entropy of q(z|x)
        # Eq. (11)–(12): entropy of q(z|x) -> entropy  (diagonal Gaussian) (term 5)
        # H[q] = 0.5 * sum_j [1 + log(2π) + log σ_j^2]
        H_q = 0.5 * (np.log(2.0 * np.pi) + 1.0 + logvar).sum(dim=1)  # (B,)

        #elbo = self.lambda_recon * log_px + self.lambda_surv * surv_term + self.lambda_kl * cat_term + H_q
        kl_block = cat_term + H_q
        elbo = self.lambda_recon * log_px + self.lambda_surv * surv_term + self.lambda_kl * kl_block

        # negative ELBO to minimize
        return -elbo.mean()

    def _forward_batch(self, Xb: np.ndarray, train: bool = True) -> Dict[str, torch.Tensor]:
        xb = torch.tensor(Xb, dtype=torch.float32, device=self.device)
        # Sample z from q(z|x) if training, else use mu
        out = self.net_(xb, sample_z=True) if train else self.net_(xb, sample_z=False)
        # include original input for recon term without roundtrips
        out["_input"] = xb
        return out

    @torch.no_grad()
    def _encode_and_prior(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        # p(c∣z) ∝ p(z∣c)p(c)
        check_is_fitted(self, "net_")
        self.net_.eval()
        x = torch.tensor(X, dtype=torch.float32, device=self.device)

        # forward with sample_z=False -> uses μ and returns all needed tensors
        out = self.net_(x, sample_z=False)
        z = out["mu"]                # (n, L)
        log_pz_c = out["log_pz_c"]   # (n, K)
        log_pc = out["log_pc"]       # (1, K)
        lam = out["lam"]             # (n, K)
        k_t = out["k"]               # scalar tensor

        logits = log_pz_c + log_pc                  # (n, K)
        logits = logits - logits.max(dim=1, keepdim=True).values
        pc_z = torch.softmax(logits, dim=1)         # (n, K)

        return {
            "z": z.cpu().numpy(),
            "pc_z": pc_z.cpu().numpy(),
            "lam": lam.cpu().numpy(),
            "k": np.array([k_t.item()], dtype=np.float32),
        }

    @torch.no_grad()
    def predict(self, X, num_times: int = 1000):
        """
        Returns S(t) on a dense grid [0, max_time_], shape (n, T).
        If training used time scaling, we evaluate S at scaled times internally
        but still return a curve over the original [0, max_time_] grid.
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        enc = self._encode_and_prior(X)
        lam = enc["lam"]                    # (n,K)
        pc  = enc["pc_z"]                   # (n,K)
        k   = enc["k"]                      # scalar or (K,)

        # --- NEW: respect training time scaling ---
        s = getattr(self, "_time_scale_", 1.0)  # 1.0 if no scaling
        times_orig = np.linspace(0.0, self.max_time_, num_times, dtype=np.float32)  # (T,)
        times_scaled = times_orig / float(s)    # <-- use scaled time for Weibull calc

        lam_exp = lam[:, :, None] + 1e-8        # (n,K,1)
        x = (times_scaled[None, None, :] / lam_exp) ** float(k)
        S_comp = np.exp(-x)                     # (n,K,T)

        # S(t∣x)≈ ∑​p(c∣z)S_c​(t).
        S_mix = (pc[:, :, None] * S_comp).sum(axis=1)  # (n,T)
        return np.clip(S_mix, 0.0, 1.0)
    

# =====================================================================
# DCM (Cox with Mixtures) — non-federated
# =====================================================================
def _vae_loss(x, x_recon, mu, logvar, recon_weight=1.0):
    # Recon term: MSE (change to feature-wise likelihoods if needed)
    recon = torch.mean((x_recon - x)**2) * recon_weight
    # KL(q(z|x) || N(0, I)) per sample, then mean
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl


class _DCMNetVAE(nn.Module):
    def __init__(self, in_dim, K, *, enc_output=64,  # latent dim (z)
                 enc_layers=2, enc_hidden=128,
                 activation="relu", dropout=0.1, batchnorm=False,
                 residual=False, use_attention=False, bias=True):
        super().__init__()

        self.enc_feat = build_mlp(
                input_dim=in_dim,
                output_dim=enc_hidden,
                num_layers=enc_layers,
                hidden_units=enc_hidden,
                activation=activation,
                dropout=dropout,
                batchnorm=batchnorm,
                residual=residual,
                use_attention=use_attention,
                bias=bias,
                out_activation="identity",
            )
        hdim = enc_hidden

        # Figure hidden width
        if enc_layers == 0:
            self.enc_feat = nn.Identity()
            hdim = in_dim

        self.mu     = nn.Linear(hdim, enc_output)
        self.logvar = nn.Linear(hdim, enc_output)

        # Decoder from z
        self.dec = build_mlp(
            input_dim=enc_output, output_dim=in_dim,
            num_layers=2, hidden_units=enc_hidden,
            activation=activation, dropout=0.0, batchnorm=False,
            residual=False, use_attention=False, bias=True,
            out_activation="identity",
        )

        # Heads operate on z
        self.expert = nn.Linear(enc_output, K)
        self.gate   = nn.Linear(enc_output, K)

    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc_feat(x)                        # (B, hdim)
        mu, logvar = self.mu(h), self.logvar(h)     # (B, enc_output)
        z = self.reparameterize(mu, logvar)
        x_recon = self.dec(z)
        return {"risk": self.expert(z), "gate": self.gate(z),
                "mu": mu, "logvar": logvar, "x_recon": x_recon, "x_in": x}


class DeepCoxMixtures(SurvivalNNBase):
    """
    Deep Cox Mixtures (single-event).

    - K mixture components.
    - Each component k has its own Cox risk head (η_k(x)) and its own baseline H0_k(t).
    - A gate outputs π_k(x) = softmax(g(x)).

    Training loss (per minibatch, hard-EM approximation):
      L = - [ sum_k PLL_k(assigned_to_k) + λ * sum_i log π_{z_i}(x_i) ] / (#events in batch)
    where z_i = argmax_k π_k(x_i).
    """

    def __init__(
        self,
        *,
        K: int = 3,
        # encoder + heads
        enc_layers: int = 2,
        enc_output: int = 32,
        enc_hidden: int = 128,
        activation: str = "relu",
        vae_alpha: float = 1.0,
        vae_recon_weight: float = 1.0,
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        use_attention: bool = False,
        bias: bool = True,
        # training
        gate_ce_weight: float = 0.1,      # λ for gating cross-entropy term
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float = 1e-3,
        lr_scheduler: str | None = None,
        weight_decay: float = 0.0,
        patience: int = 10,
        val_fraction: float = 0.15,
        interpolation: str = "step",
        random_state: int = 42,
        device: str | None = None,
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
        self.K = K
        self.enc_layers = enc_layers
        self.enc_output = enc_output
        self.enc_hidden = enc_hidden
        self.activation = activation
        self.vae_alpha = vae_alpha
        self.vae_recon_weight = vae_recon_weight
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.use_attention = use_attention
        self.bias = bias
        self.gate_ce_weight = gate_ce_weight
        self.device_ = device

    def _init_net(self, n_features: int) -> torch.nn.Module:
        return _DCMNetVAE(
            in_dim=n_features,
            K=self.K,
            enc_layers=self.enc_layers,
            enc_output=self.enc_output,
            enc_hidden=self.enc_hidden,
            activation=self.activation,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            residual=self.residual,
            use_attention=self.use_attention,
            bias=self.bias,
        )
    
    def _prepare_targets(self, y) -> Dict[str, Any]:
        return {
            "time": y["time"].astype(np.float32),
            "event": y["event"].astype(np.float32),
        }

    def _loss_fn(self, logits: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
        """
        logits["risk"]: (B,K) risk scores per component (η_k)
        logits["gate"]: (B,K) gate logits
        """
        event = targets["event"]          # (B,)
        time = targets["time"]            # (B,)

        risk = logits["risk"]             # (B,K)
        gate_logits = logits["gate"]      # (B,K)
        
        # Hard assignment
        with torch.no_grad():
            pi = torch.softmax(gate_logits, dim=-1)     # soft counts
            z_hard = torch.argmax(pi, dim=-1)           # categorical(pi)

        # 1) Sum Cox partial log-likelihood per component on its hard-assigned subset
        pll_sum = torch.tensor(0.0, dtype=risk.dtype, device=risk.device)
        for k in range(self.K):
            idx = (z_hard == k)
            if idx.any():
                eta_k = risk[idx, k]                    # (n_k,)
                t_k = time[idx]
                e_k = event[idx]
                pll_k = cox_partial_log_likelihood(e_k, eta_k, t_k)
                pll_sum = pll_sum + pll_k # Σ_k ln PL(𝓓_b^k; θ)

        # 2) Gating cross-entropy on hard assignments to stabilize gate
        #    CE = - sum_i log softmax(gate)_z_i
        log_pi = gate_logits.log_softmax(dim=-1)
        ce = -log_pi[torch.arange(log_pi.size(0), device=log_pi.device), z_hard].sum()

        # Normalize by number of events in batch (avoid 0 with clamp)
        num_events = torch.clamp(event.sum(), min=1.0)

        loss = -(pll_sum) / num_events + self.gate_ce_weight * (ce / num_events)

         # ---- VAE term (robust) ----
        vae = 0.0
        x_batch = logits.get("x_in")
        # Detach x so gradients only flow through the decoder (x_recon) side
        if x_batch is None: 
            print("I shouldn't really go here...")
            mu, logvar = logits["mu"], logits["logvar"]
            vae = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            x_batch = x_batch.detach()
            vae = _vae_loss(x_batch, logits["x_recon"], logits["mu"], logits["logvar"], self.vae_recon_weight)

        return loss + self.vae_alpha * vae
        #return loss

    # ------------------------------------------------------------------
    # After fit: estimate per-component baselines via Breslow on full train set
    # ------------------------------------------------------------------
    def _post_fit(self, X, y):
        """
        - Compute hard assignments z on the full training set using the current gate.
        - For each component k: estimate baseline H0_k(t) with Breslow on subset z=k.
        """
        if hasattr(X, "to_numpy"):
            Xn = X.to_numpy()
        else:
            Xn = np.asarray(X)
        Xn = Xn.astype(np.float32)

        # Forward pass
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(Xn, dtype=torch.float32, device=self.device_))
            gate = out["gate"].softmax(dim=-1).cpu().numpy()  # (N,K)
            risk = out["risk"].cpu().numpy()                  # (N,K)
        z_hard = gate.argmax(axis=1)                          # (N,)

        # Store component-wise baselines
        self.baseline_times_ = []
        self.baseline_cumhaz_ = []
        self.train_assignments_ = z_hard

        events_full = y["event"]

        for k in range(self.K):
            idx = (z_hard == k)
            if idx.sum() == 0 or events_full[idx].sum() == 0:
                # No data or no events in that cluster → empty baseline
                self.baseline_times_.append(np.asarray([], dtype=np.float32))
                self.baseline_cumhaz_.append(np.asarray([], dtype=np.float32))
                continue

            # Cox within component k uses η_k
            xbeta_k = risk[idx, k].reshape(-1)
            y_k = y[idx]  # preserves dtype with named fields
            t_unique, cum_haz = breslow_estimator(y_k, xbeta_k)
            self.baseline_times_.append(t_unique.astype(np.float32))
            self.baseline_cumhaz_.append(cum_haz.astype(np.float32))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _forward_numpy(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (risk, gate_prob) as numpy arrays."""
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(X, dtype=torch.float32, device=self.device_))
            risk = out["risk"].cpu().numpy()              # (N,K)
            gate_prob = out["gate"].softmax(dim=-1).cpu().numpy()  # (N,K)
        return risk, gate_prob

    def predict(self, X, num_times: int = 1000) -> np.ndarray:
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        risk, gate = self._forward_numpy(X)                # (N,K), (N,K)
        N = X.shape[0]
        dense_times = np.linspace(0.0, self.max_time_, num_times).astype(np.float32)
        S_mix = np.zeros((N, num_times), dtype=np.float32)

        for k in range(self.K):
            t_k  = self.baseline_times_[k]        # (T_k,)
            H0_k = self.baseline_cumhaz_[k]       # (T_k,)
            if t_k.size == 0:
                continue

            # Ensure the curve starts at (0, 0) for hazard (good for spline)
            if t_k[0] > 0.0:
                t_k  = np.concatenate([[0.0], t_k.astype(np.float32)])
                H0_k = np.concatenate([[0.0], H0_k.astype(np.float32)])

            # 1) Monotone spline of baseline cumulative hazard
            H0_dense = PchipInterpolator(t_k, H0_k, extrapolate=True)(dense_times)  # (num_times,)

            # 2) Scale by exp(η_k) and convert to survival
            rr_k = np.exp(risk[:, k]).astype(np.float32)                            # (N,)
            expo = - rr_k[:, None] * H0_dense[None, :]
            expo = np.clip(expo, -1e3, 0)  # safe for float32 exp
            S_k = np.exp(expo).astype(np.float32)
        
            # 3) Mix with gate probabilities
            S_mix += gate[:, k:k+1].astype(np.float32) * S_k

        return np.clip(S_mix, 0.0, 1.0)

    def predict_risk(self, X, num_times: int = 1000) -> np.ndarray:
        """
        Natural DCM risk proxy: expected exp(η_k) under gate:
          r(x) = sum_k π_k(x) * exp(η_k(x))
        This preserves a monotone relation with individual hazards for ranking.
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        risk, gate = self._forward_numpy(X)          # (N,K), (N,K)
        return (gate * np.exp(risk)).sum(axis=1)     # (N,)


# =====================================================================
# DVC-Surv — Deep Contrastive Survival Analysis with Dual-View Clustering — non-federated
# =====================================================================
class _DVCNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        *,
        latent_dim: int = 16,
        num_durations: int = 50,    
        K_clusters: int = 3,
        # enc/dec
        enc_layers: int = 1,
        enc_hidden: Union[int, Sequence[int]] = 128,
        dec_layers: int = 3,
        dec_hidden: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        student_t_df: float = 1.0,      # α in paper (degrees of freedom)
        surv_head: nn.Module = None,
    ):
        super().__init__()
        L = latent_dim
        self.L = L
        self.in_dim = in_dim
        self.K = K_clusters
        self.df = float(student_t_df)

        def make_enc():
            return build_mlp(
                input_dim=in_dim, output_dim=L,
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
        self.enc1 = make_enc()
        self.enc2 = make_enc()
        self.dec1 = make_dec()
        self.dec2 = make_dec()

        # Cluster centers per view
        self.register_buffer("centers1", torch.randn(self.K, L)*0.05)
        self.register_buffer("centers2", torch.randn(self.K, L)*0.05)

        # survival backbone: use provided DeepHit head
        self.num_durations = int(num_durations)
        fuse_dim = latent_dim + in_dim
        self.fuse_dim = fuse_dim
        self.surv_head = surv_head

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
        # Siamese encode
        z1 = self.enc1(x)   # (B,L)
        z2 = self.enc2(x)   # (B,L)
        
        # Soft labels via Student-t similarities for the loss function, eq. (11)
        q1 = self._student_t_sim(z1, self.centers1, self.df)   # (B,K)
        q2 = self._student_t_sim(z2, self.centers2, self.df)   # (B,K)

        # Fuse representation for survival head
        z_mean = 0.5 * (z1 + z2)
        h = torch.cat([z_mean, x], dim=1)                     # (B, L + D)
        surv_logits = self.surv_head(h)                       # (B, T)

        # Required for the loss
        x1_hat = self.dec1(z1)
        x2_hat = self.dec2(z2)
        rec1_ps = F.mse_loss(x1_hat, x, reduction="none").mean(dim=1)  # (B,)
        rec2_ps = F.mse_loss(x2_hat, x, reduction="none").mean(dim=1)  # (B,)

        return {
        "z1": z1, "z2": z2,
        "q1": q1, "q2": q2,
        "x1_hat": x1_hat, "x2_hat": x2_hat,
        "rec1_ps": rec1_ps, "rec2_ps": rec2_ps, 
        "surv_logits": surv_logits,
        "centers1": self.centers1, "centers2": self.centers2,
        }


class DVCSurv(SurvivalNNBase):
    """
    DVC-Surv: Deep Contrastive Survival Analysis with Dual-View Clustering
    (Cui, Tang, Zhang, 2024). This version:
      - learns two views (siamese AEs),
      - maintains dual-view clusters with Student-t soft labels
      - uses triple contrastive losses,
      - predicts discrete-time mass (DeepHit-like) from fused representation.
    Reference: Deep Contrastive Survival Analysis with Dual-View Clustering (Electronics 2024).
    """

    def __init__(
        self,
        *,
        num_durations: int = 50,
        K_clusters: int = 4,
        latent_dim: int = 8,
        # For Deephit
        num_layers: int = 2,
        hidden_units: Union[int, Sequence[int]] = 16,
        # enc/dec
        enc_layers: int = 1,
        enc_hidden: Union[int, Sequence[int]] = 8,
        dec_layers: int = 1,
        dec_hidden: Union[int, Sequence[int]] = 8,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        bias: bool = True,
        student_t_df: float = 1.0,
        # loss weights
        alpha_REC: float = 1e-1,   # good is 1e-5
        alpha_DVC: float = 1e-2,      # clustering pull-to-center (per view)
        alpha_IVCG: float = 1e-2,     # intra-view cluster-guided contrastive
        alpha_IVIW: float = 1e-2,     # inter-view instance-wise contrastive
        alpha_IVCW: float = 1e-2,     # inter-view cluster-wise contrastive
        alpha_NLL: float = 1e-1,      # survival NLL
        alpha_RANK: float = 1e-2,      # survival ranking loss
        temperature: float = 0.1,
        # SPL (self-paced) — use batchwise soft gating by percentile
        spl_enable: bool = True,
        spl_percentile: float = 0.5,  # keep samples with per-sample (REC+DVC) loss below this quantile
        # training
        epochs: int = 500,
        batch_size: int = 128,
        lr: float = 1e-3,
        lr_scheduler: str | None = None,
        weight_decay: float = 1e-5,
        patience: int = 30,
        val_fraction: float = 0.15,
        interpolation: str = "step",
        random_state: int = 42,
        device: str | None = None,
        pretraining: bool = False,
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
        self.K_clusters = K_clusters
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
        self.alpha_DVC = alpha_DVC
        self.alpha_IVCG = alpha_IVCG
        self.alpha_IVIW = alpha_IVIW
        self.alpha_IVCW = alpha_IVCW
        self.alpha_NLL  = alpha_NLL
        self.alpha_RANK = alpha_RANK
        self.temperature = temperature

        # SPL
        self.spl_enable = spl_enable
        self.spl_percentile = float(spl_percentile)

        self.device = device
        self.pretraining = pretraining

    # ---------- network ----------
    def _init_net(self, n_features: int) -> torch.nn.Module:
        fuse_dim = self.latent_dim + n_features

        # REUSE THE DEEPHIT HEAD ARCH
        dh_head = build_mlp(
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
        return _DVCNet(
            in_dim=n_features,
            latent_dim=self.latent_dim,
            num_durations=self.num_durations,
            K_clusters=self.K_clusters,
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
            surv_head=dh_head,
        )

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
        zs1, zs2 = [], []
        with torch.no_grad():
            for i in range(0, n, B):
                xb = torch.tensor(X[i:i+B], dtype=torch.float32, device=self.device)
                z1 = self.net_.enc1(xb)
                z2 = self.net_.enc2(xb)
                zs1.append(z1.cpu().numpy())
                zs2.append(z2.cpu().numpy())

        Z1 = np.concatenate(zs1, axis=0)    # (n, L)
        Z2 = np.concatenate(zs2, axis=0)    # (n, L)
        Z  = np.concatenate([Z1, Z2], axis=1)  # (n, 2L)

        km = KMeans(n_clusters=self.K_clusters, n_init=10, random_state=random_state if random_state is not None else self.random_state)
        km.fit(Z)
        C   = km.cluster_centers_                        # (K, 2L)
        L   = self.latent_dim
        # Row k of C1 and row k of C2 correspond to the same cluster, just in different views.
        C1, C2 = C[:, :L], C[:, L:]                      # split centers for two views

        # write into model buffers
        with torch.no_grad():
            self.net_.centers1.copy_(torch.tensor(C1, dtype=torch.float32, device=self.device))
            self.net_.centers2.copy_(torch.tensor(C2, dtype=torch.float32, device=self.device))

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

    def _inter_view_instance_wise(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        
        z1n = F.normalize(z1, dim=-1)
        z2n = F.normalize(z2, dim=-1)
        # row i = similarities between patient i’s view 1 and all patients’ view 2.
        sim12 = z1n @ z2n.t()
        sim21 = z2n @ z1n.t()
        logits12 = sim12 / self.temperature
        logits21 = sim21 / self.temperature
        labels = torch.arange(z1.shape[0], device=z1.device)
        return 0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits21, labels))

    def _inter_view_cluster_wise(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        eq (12): Cluster k in view 1 should “look like” cluster k in view 2, not like other cluster
        Align cluster distributions across views. Treat each cluster k's confidence vector across batch
        as an "instance", pull q1[:,k] toward q2[:,k], push toward q2[:,k'] (k'!=k).
        """
        #q1, q2: soft cluster assignments from each view, shape (B, K)
        B, K = q1.shape
        # form cluster "features" as normalized column vectors
        Q1 = F.normalize(q1, dim=0)   # (B,K)
        Q2 = F.normalize(q2, dim=0)   # (B,K)
        # Row i = similarities between cluster i of view 1 and all clusters of view 2.
        sim12 = Q1.t() @ Q2             # (K,K)
        sim21 = Q2.t() @ Q1             # (K,K)
        logits21 = sim21 / self.temperature
        logits12 = sim12 / self.temperature
        labels = torch.arange(K, device=q1.device)
        return 0.5 * (F.cross_entropy(logits12, labels) + F.cross_entropy(logits21, labels))

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

        # For each uncensored i, pick its own CDF at its own time bin t_i
        Fi_ti = F_uncensored[torch.arange(num_uncensored, device=F_uncensored.device), bi]     # (Ne,)

        # Where column i contains every subject j evaluated at i’s time t_i.
        Fj_ti = F_uncensored[:, bi]

        # Build pairwise mask for t_i < t_j  <=>  bin_i < bin_j
        bi_col = bi.view(num_uncensored, 1)
        bj_row = bi.view(1, num_uncensored)
        comp_mask = (bi_col < bj_row)

        if not comp_mask.any():
            return pmf.new_tensor(0.0)

        # Expand Fi(t_i) to matrix shape to compare against Fj(t_i)
        Fi_mat = Fi_ti.view(num_uncensored, 1).expand(num_uncensored, num_uncensored)              # (Ne,Ne)

        # Differences for valid pairs (i,j) with t_i < t_j
        diff = (Fi_mat - Fj_ti)[comp_mask]                     # (M,)

        # Logistic ranking loss
        return -torch.nn.functional.logsigmoid(diff / sigma).mean()

    def _assign_S_batch(self,
                    z1: torch.Tensor, z2: torch.Tensor,
                    centers1: torch.Tensor, centers2: torch.Tensor) -> torch.Tensor:
        """
        Hard cluster assignments S for the current batch, with fixed centers M:
        k* = argmin_k ||z1 - c1_k||^2 + ||z2 - c2_k||^2
        Returns: hard indices (B,)
        """
        # (B,K) squared distances to each view's centers
        d1 = ((z1.unsqueeze(1) - centers1.unsqueeze(0)) ** 2).sum(-1)
        d2 = ((z2.unsqueeze(1) - centers2.unsqueeze(0)) ** 2).sum(-1)
        return (d1 + d2).argmin(dim=1)  # (B,)

    def _loss_fn(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, Any]) -> torch.Tensor:
        z1, z2 = outputs["z1"], outputs["z2"]
        q1, q2 = outputs["q1"], outputs["q2"]
        logits = outputs["surv_logits"]             # (B, T)
        bin_idx = targets["bin_idx"]
        event = targets["event"]
        centers1, centers2 = outputs["centers1"], outputs["centers2"]

        # --- per-sample pieces ---
        # reconstruction terms (only if we have x)
        rec_sum = outputs["rec1_ps"] + outputs["rec2_ps"]

        # batch-level hard assignments S (fixed centers) for both views
        hard = self._assign_S_batch(z1, z2, centers1, centers2)
        c1 = centers1[hard]; 
        c2 = centers2[hard]
        dvc1 = ((z1 - c1) ** 2).sum(dim=1)
        dvc2 = ((z2 - c2) ** 2).sum(dim=1) 

        if self.pretraining:
            base_per_item = self.alpha_REC * rec_sum
            dvc1 = dvc1.new_tensor(0.0)
            dvc2 = dvc2.new_tensor(0.0)
        else:
            base_per_item = self.alpha_DVC * (dvc1 + dvc2) + self.alpha_REC * rec_sum 

        # SPL
        if self.pretraining or (not self.spl_enable):
            w = torch.ones_like(base_per_item)
        else:
            # Equation (7)
            Li = self.alpha_REC * rec_sum + self.alpha_DVC * (dvc1 + dvc2) 
            mu, std = Li.detach().mean(), Li.detach().std(unbiased=False)
            prog = getattr(self, "_epoch", 1) / max(getattr(self, "epochs", 1), 1)
            # Equation (8) Set the threshold λ = μ + progress·σ
            lam = mu + prog * std
            # Equation (6)
            w = (Li.detach() <= lam).float()

        # Equation 5
        base = (w * base_per_item).mean() 

        # survival term
        nll = _deephit_nll(logits, bin_idx, event)
        probs = F.softmax(logits, dim=1)
        rank = self._deephit_rank_loss(probs, bin_idx, event, sigma=0.1)

        # contrastive terms
        if self.pretraining:
            ivcg_1 = z1.new_tensor(0.0)
            ivcg_2 = z2.new_tensor(0.0)
            iviw   = z1.new_tensor(0.0)
            ivcw   = z1.new_tensor(0.0)
        else:
            ivcg_1 = self._intra_view_cluster_guided(z1, q1, event,  hard_override=hard)
            ivcg_2 = self._intra_view_cluster_guided(z2, q2, event,  hard_override=hard)
            iviw   = self._inter_view_instance_wise(z1, z2)
            ivcw   = self._inter_view_cluster_wise(q1, q2)

        # Take the all mean here
        loss = base + self.alpha_NLL * nll + self.alpha_RANK * rank \
                + self.alpha_IVCG * (ivcg_1 + ivcg_2) + self.alpha_IVIW * iviw + self.alpha_IVCW * ivcw
        
        # ---- NEW: stash a parts dict you can log per batch
        self._last_parts = {
            "base": float(base.detach().item()),
            "nll": float(self.alpha_NLL * nll.detach().item()),
            "rank": float(self.alpha_RANK * rank.detach().item()),
            "rec": float(self.alpha_REC * rec_sum.detach().mean().item()),
            "dvc": float(self.alpha_DVC * ((dvc1 + dvc2).detach().mean().item())),
            "tcl": float(self.alpha_IVCG * ((ivcg_1 + ivcg_2).detach().item()) + self.alpha_IVIW * iviw.detach().item() + float(self.alpha_IVCW * ivcw.detach().item())),
        }
        return loss

    def _forward(self, Xb: np.ndarray, train: bool = True) -> Dict[str, torch.Tensor]:
        xb = torch.tensor(Xb, dtype=torch.float32, device=self.device)
        out = self.net_(xb)
        out["_input"] = xb
        return out

    def _forward_batch(self, Xb: np.ndarray, train: bool = True) -> Dict[str, torch.Tensor]:
        xb = torch.tensor(Xb, dtype=torch.float32, device=self.device)
        out = self.net_(xb)
        out["_input"] = xb
        return out

    # ---------- prediction ----------
    @torch.no_grad()
    def predict(self, X, num_times: int = 1000):
        """
        Predict S(t) on dense grid. Same as DeepHit
        """
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)
        self.net_.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            out = self.net_(xb)
            logits = out["surv_logits"].cpu().numpy()     # (n, T_d)
        # pmf and survival on discrete grid edges
        logits = logits - logits.max(axis=1, keepdims=True)
        pmf = np.exp(logits); pmf /= pmf.sum(axis=1, keepdims=True)
        surv_d = 1.0 - np.cumsum(pmf, axis=1)
        surv_d = np.clip(surv_d, 0.0, 1.0)

        times = np.linspace(0.0, self.max_time_, num_times).astype(np.float32)
        fn = make_interpolator(surv_d, self.time_grid_, mode=self.interpolation)
        return fn(times)
    