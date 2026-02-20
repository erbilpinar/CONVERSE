from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted

from survbase.base import SurvivalNNBase
from survbase.nonparametric import breslow_estimator
from survbase.utils import build_mlp, make_interpolator


# ---------------------------------------------------------------------
# Cox partial log-likelihood (Breslow ties)
# ---------------------------------------------------------------------
def cox_partial_log_likelihood(
    event: torch.Tensor, risk_scores: torch.Tensor, time: torch.Tensor
) -> torch.Tensor:
    """
    Cox partial log-likelihood (Breslow) with stable log-sum-exp denominator.

    event      : (n,)   bool/float tensor (1=event, 0=censored)
    risk_scores: (n,)   tensor (η = xβ)
    time       : (n,)   tensor (observed time)
    returns    : scalar log-likelihood
    """
    # Ensure 1-D shapes
    event = event.reshape(-1).to(risk_scores.dtype)
    risk_scores = risk_scores.reshape(-1)
    time = time.reshape(-1)

    # Sort by ascending time so each risk set R_i is everyone at/after i
    order = torch.argsort(time)
    event_sorted = event[order]
    risk_sorted = risk_scores[order]

    # log sum_{j in R_i} exp(η_j) using log-cumsum-exp on reversed vector
    denom_log = torch.flip(
        torch.logcumsumexp(torch.flip(risk_sorted, dims=[0]), dim=0),
        dims=[0],
    )
    pll = (event_sorted * (risk_sorted - denom_log)).sum()
    return pll


# =====================================================================
# CoxPH (linear)  — non-federated
# =====================================================================
class CoxPH(SurvivalNNBase):
    """Linear Cox proportional hazards model."""

    def __init__(
        self,
        *,
        epochs: int = 100,
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
            lr=lr,
            lr_scheduler=lr_scheduler,
            weight_decay=weight_decay,
            patience=patience,
            val_fraction=val_fraction,
            interpolation=interpolation,
            random_state=random_state,
            device=device,
        )

    # Tiny one-layer “network” (no bias ⇒ no intercept)
    def _init_net(self, n_features: int) -> torch.nn.Module:
        return build_mlp(
            input_dim=n_features,
            output_dim=1,
            num_layers=0,
            bias=False,
            out_activation="identity",
        )

    def _prepare_targets(self, y) -> Dict[str, Any]:
        return {
            "time": y["time"].astype(np.float32),
            "event": y["event"].astype(np.float32),
        }

    def _loss_fn(self, logits: torch.Tensor, targets: Dict[str, Any]) -> torch.Tensor:
        # logits: (B,1) -> (B,)
        eta = logits.reshape(-1)
        pll = cox_partial_log_likelihood(targets["event"], eta, targets["time"])
        # Minimize negative log-likelihood, normalize by #events (common)
        return -pll / torch.clamp(targets["event"].sum(), min=1.0)

    def _post_fit(self, X, y, targets_train):
        # Breslow baseline cumulative hazard on training set
        xbeta = self._predict_risk_scores(X)
        t_unique, cum_haz = breslow_estimator(y, xbeta)
        self.baseline_times_ = t_unique
        self.baseline_cumhaz_ = cum_haz

    def _predict_risk_scores(self, X) -> np.ndarray:
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        self.net_.eval()
        with torch.no_grad():
            rs = self.net_(torch.tensor(X.astype(np.float32))).cpu().numpy()
        return rs.reshape(-1)  # η = xβ

    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            risk = np.exp(self._predict_risk_scores(X))  # exp(η)
        H = self.baseline_cumhaz_.reshape((1, -1)) * risk[:, None]
        fn = make_interpolator(H, self.baseline_times_, mode=self.interpolation)
        times = np.linspace(0.0, self.max_time_, num_times)
        return np.exp(-fn(times))  # S(t)

    # Cox models have a natural risk score (η). Override base default.
    def predict_risk(self, X, num_times: int = 64) -> np.ndarray:
        return self._predict_risk_scores(X)


# =====================================================================
# Accelerated Failure Time (AFT) — non-federated
# =====================================================================
def _aft_nll(
    log_scale: torch.Tensor,
    log_shape: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    distribution: str = "weibull",
) -> torch.Tensor:
    """
    AFT negative log-likelihood (Weibull or log-normal).

    log_scale : (B,)   log(λ) — network output for scale
    log_shape : scalar log(k) — learnable shape parameter
    time      : (B,)   observed time
    event     : (B,)   1 if event, 0 if censored
    """
    scale = torch.exp(log_scale)  # λ > 0
    shape = torch.exp(log_shape)  # k > 0

    eps = 1e-8
    t_clamp = torch.clamp(time, min=eps)

    if distribution == "weibull":
        # S(t) = exp(-(t/λ)^k), f(t) = (k/λ)(t/λ)^(k-1) exp(-(t/λ)^k)
        z = t_clamp / scale
        log_pdf = torch.log(shape + eps) - log_scale + (shape - 1.0) * torch.log(z + eps) - torch.pow(z, shape)
        log_surv = -torch.pow(z, shape)
    else:  # lognormal
        # log(T) ~ N(log(λ), σ²), σ = 1/k
        mu = log_scale
        sigma = 1.0 / (shape + eps)
        log_t = torch.log(t_clamp)
        z_norm = (log_t - mu) / (sigma + eps)

        log_pdf = -0.5 * torch.log(torch.tensor(2.0 * np.pi)) - torch.log(sigma + eps) - torch.log(t_clamp + eps) - 0.5 * z_norm**2
        log_surv = torch.log(torch.clamp(1.0 - torch.special.ndtr(z_norm), min=eps))

    nll = -(event * log_pdf + (1.0 - event) * log_surv)
    return nll.mean()


class AFT(SurvivalNNBase):
    """
    Accelerated Failure Time model (Weibull or log-normal).

    - Network outputs log(λ_i) per sample; global shape k is learned
    - AFT form: T_i = λ_i ε, where ε follows the chosen distribution
    - Loss: negative log-likelihood of parametric survival model
    - Prediction: S(t|x) = exp(-(t/λ)^k) for Weibull or normal-CDF-based
      for log-normal, interpolated to a uniform grid in [0, max_time_]
    """

    def __init__(
        self,
        *,
        distribution: str = "weibull",
        num_layers: int = 2,
        hidden_units: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        use_attention: bool = False,
        bias: bool = True,
        # training
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
        if distribution not in ("weibull", "lognormal"):
            raise ValueError(f"distribution must be 'weibull' or 'lognormal', got {distribution}")
        self.distribution = distribution
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.use_attention = use_attention
        self.bias = bias

    def _init_net(self, n_features: int) -> torch.nn.Module:
        # Network outputs log(scale) per sample, plus a global log(shape)
        class AFTNet(torch.nn.Module):
            def __init__(self, base_net, parent):
                super().__init__()
                self.base_net = base_net
                # Global shape parameter (log space for unconstrained optimization)
                self.log_shape = torch.nn.Parameter(torch.tensor(0.0))
                # Keep reference to parent AFT instance for _loss_fn access
                self.parent = parent

            def forward(self, x):
                return self.base_net(x)

        base = build_mlp(
            input_dim=n_features,
            output_dim=1,
            bias=self.bias,
            num_layers=self.num_layers,
            hidden_units=self.hidden_units,
            batchnorm=self.batchnorm,
            dropout=self.dropout,
            activation=self.activation,
            out_activation="identity",
            residual=self.residual,
            use_attention=self.use_attention,
        )
        net = AFTNet(base, self)
        # Store reference to the network so _loss_fn can access log_shape
        self._net = net
        return net

    def _prepare_targets(self, y) -> Dict[str, Any]:
        return {
            "time": y["time"].astype(np.float32),
            "event": y["event"].astype(np.float32),
        }

    def _loss_fn(self, logits: torch.Tensor, targets: Dict[str, Any]) -> torch.Tensor:
        log_scale = logits.reshape(-1)
        # Access log_shape from the network stored during _init_net
        log_shape = self._net.log_shape
        return _aft_nll(
            log_scale=log_scale,
            log_shape=log_shape,
            time=targets["time"],
            event=targets["event"],
            distribution=self.distribution,
        )

    def _post_fit(self, X, y, targets_train):
        # Store the learned shape parameter for prediction
        self.shape_ = float(torch.exp(self.net_.log_shape).detach().cpu().numpy())

    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        self.net_.eval()
        with torch.no_grad():
            log_scale = self.net_(torch.tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)

        scale = np.exp(log_scale)  # λ
        shape = self.shape_  # k

        times = np.linspace(0.0, self.max_time_, num_times)
        t_grid = times.reshape(1, -1)  # (1, m)
        scale_col = scale[:, None]  # (n, 1)

        eps = 1e-8
        if self.distribution == "weibull":
            # S(t) = exp(-(t/λ)^k)
            z = np.maximum(t_grid / scale_col, eps)
            surv = np.exp(-np.power(z, shape))
        else:  # lognormal
            # S(t) = 1 - Φ((log(t) - log(λ)) / σ), where σ = 1/k
            sigma = 1.0 / (shape + eps)
            log_t = np.log(np.maximum(t_grid, eps))
            log_scale_col = log_scale[:, None]
            z_norm = (log_t - log_scale_col) / sigma
            from scipy.stats import norm
            surv = 1.0 - norm.cdf(z_norm)

        return np.clip(surv, 0.0, 1.0)


# =====================================================================
# (Optional) Piecewise-Constant Hazard model — non-federated
# =====================================================================
def _pc_hazard_nll(
    logits: torch.Tensor,
    time: torch.Tensor,
    bin_idx: torch.Tensor,
    event: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """
    logits : (B, K)  raw network output (log-hazards)
    time   : (B,)    observed time
    bin_idx: (B,)    interval index of event/censoring
    event  : (B,)    1 if event, 0 if censored
    delta  : (K,)    interval widths (equi-spaced grid assumed)
    """
    haz = torch.exp(logits)  # λ_k ≥ 0

    B, K = haz.shape
    grid = torch.arange(K, device=haz.device).expand(B, K)
    delta_row = delta.expand(B, K)

    # full exposure for intervals strictly before the last observed one
    exp_full = torch.where(grid < bin_idx.unsqueeze(1), delta_row, torch.zeros_like(delta_row))

    # partial exposure in the last observed interval
    t_left = (bin_idx.to(delta.dtype) * delta[0]).unsqueeze(1)  # equi-spaced assumption
    part = (time.unsqueeze(1) - t_left).clamp_min(0.0)
    exp_last = torch.zeros_like(exp_full)
    exp_last.scatter_(1, bin_idx.unsqueeze(1), part)

    exposure = exp_full + exp_last  # (B, K)

    cum_haz = (haz * exposure).sum(1)  # H(t)
    haz_j = haz.gather(1, bin_idx.unsqueeze(1)).squeeze(1)
    nll = cum_haz - event * torch.log(haz_j + 1e-8)  # −log L
    return nll.mean()


class PCHazard(SurvivalNNBase):
    """
    Piecewise-Constant Hazard model on an equi-spaced time grid.

    - Network outputs K log-hazards (λ_k) for K intervals.
    - Loss: negative log-likelihood under piecewise-constant hazard.
    - Prediction: S(t) = exp(- sum_j λ_j * Δ_j up to t), then interpolated
      to a uniform grid in [0, max_time_].

    Parameters
    ----------
    num_durations : int
        Number of piecewise-constant intervals (K).
    num_layers, hidden_units, activation, dropout, batchnorm, residual, use_attention, bias
        Architecture knobs for the underlying MLP.
    epochs, lr, lr_scheduler, weight_decay, patience, val_fraction, interpolation, random_state, device
        Training & utility parameters shared with SurvivalNNBase.
    """

    def __init__(
        self,
        *,
        num_durations: int = 50,
        num_layers: int = 2,
        hidden_units: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        use_attention: bool = False,
        bias: bool = True,
        # training
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
        self.num_durations = num_durations
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.use_attention = use_attention
        self.bias = bias

        # learned/derived at fit-time
        self.time_grid_: np.ndarray | None = None  # (K,) upper edges
        self.delta_: np.ndarray | None = None  # (K,) interval widths (equi-spaced)

    # ---------- network ----------
    def _init_net(self, n_features: int) -> torch.nn.Module:
        # Outputs K log-hazards (unconstrained); we exponentiate inside loss/predict.
        return build_mlp(
            input_dim=n_features,
            output_dim=self.num_durations,
            bias=self.bias,
            num_layers=self.num_layers,
            hidden_units=self.hidden_units,
            batchnorm=self.batchnorm,
            dropout=self.dropout,
            activation=self.activation,
            out_activation="identity",
            residual=self.residual,
            use_attention=self.use_attention,
        )

    # ---------- targets ----------
    def _prepare_targets(self, y) -> Dict[str, Any]:
        times = y["time"].astype(np.float32)
        events = y["event"].astype(np.float32)

        # Equi-spaced grid over [0, max_time], store upper edges (K,)
        t_max = times.max()
        self.time_grid_ = np.linspace(0.0, t_max, self.num_durations + 1, dtype=np.float32)[1:]
        # Constant width Δ for each interval
        delta = (
            (t_max / self.num_durations).astype(np.float32)
            if np.isscalar(t_max)
            else np.float32(t_max / self.num_durations)
        )
        self.delta_ = np.full(self.num_durations, delta, dtype=np.float32)

        # Interval index of the observed time (right-closed)
        bin_idx = np.searchsorted(self.time_grid_, times, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.num_durations - 1).astype(np.int64)

        return {"time": times, "bin_idx": bin_idx, "event": events}

    # ---------- loss ----------
    def _loss_fn(self, logits: torch.Tensor, targets: Dict[str, Any]) -> torch.Tensor:
        # Make sure delta is a tensor on the same device
        assert self.delta_ is not None
        delta_t = torch.tensor(self.delta_, dtype=logits.dtype, device=logits.device)
        return _pc_hazard_nll(
            logits=logits,
            time=targets["time"],
            bin_idx=targets["bin_idx"],
            event=targets["event"],
            delta=delta_t,
        )

    # ---------- prediction ----------
    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        assert self.time_grid_ is not None and self.delta_ is not None
        delta = self.delta_

        # Network outputs log-hazards; convert to hazards λ_k
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        hazards = np.exp(logits)  # (n, K)

        # Cumulative hazard at bin edges: H_k = sum_{j<=k} λ_j * Δ
        cum_haz_d = np.cumsum(hazards * delta.reshape(1, -1), axis=1)  # (n, K)
        surv_d = np.exp(-cum_haz_d)  # S at upper edges

        # Interpolate discrete S_k to a dense, regular grid
        times = np.linspace(0.0, self.max_time_, num_times)
        fn = make_interpolator(surv_d, self.time_grid_, mode=self.interpolation)
        return fn(times)


# =====================================================================
# Logistic-Hazard (Nnet-survival, discrete time) — non-federated
# =====================================================================
def _logistic_hazard_loss(
    logits: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """
    logits : (B, K)  – raw network output per interval
    bin_idx: (B,)    – last observed interval index
    event  : (B,)    – 1 if event occurred, else 0
    """
    B, K = logits.shape
    grid = torch.arange(K, device=logits.device).expand(B, K)

    obs_mask = grid <= bin_idx.unsqueeze(1)
    event_mask = grid == bin_idx.unsqueeze(1)

    targets = torch.zeros_like(logits)
    targets[event_mask] = event

    bce = F.binary_cross_entropy_with_logits(logits[obs_mask], targets[obs_mask], reduction="sum")
    # Normalize by #events (standard trick for stable scale)
    return bce / torch.clamp(event.sum(), min=1.0)


class LogisticHazard(SurvivalNNBase):
    """Discrete-time logistic-hazard model (a.k.a. Nnet-Survival)."""

    def __init__(
        self,
        *,
        num_durations: int = 50,
        num_layers: int = 2,
        hidden_units: Union[int, Sequence[int]] = 128,
        activation: str = "relu",
        dropout: float = 0.1,
        batchnorm: bool = False,
        residual: bool = False,
        use_attention: bool = False,
        bias: bool = True,
        # training
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
        self.num_durations = num_durations
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.use_attention = use_attention
        self.bias = bias

    def _init_net(self, n_features: int) -> torch.nn.Module:
        return build_mlp(
            input_dim=n_features,
            output_dim=self.num_durations,
            bias=self.bias,
            num_layers=self.num_layers,
            hidden_units=self.hidden_units,
            batchnorm=self.batchnorm,
            dropout=self.dropout,
            activation=self.activation,
            out_activation="identity",
            residual=self.residual,
            use_attention=self.use_attention,
        )

    def _prepare_targets(self, y) -> Dict[str, Any]:
        # Global, equi-spaced time grid; store upper edges (K,)
        times = y["time"].astype(np.float32)
        events = y["event"].astype(np.float32)
        self.time_grid_ = np.linspace(0.0, times.max(), self.num_durations + 1)[1:]
        bin_idx = np.searchsorted(self.time_grid_, times, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.num_durations - 1).astype(np.int64)
        return {"bin_idx": bin_idx, "event": events}

    def _loss_fn(self, logits, targets):
        return _logistic_hazard_loss(logits, targets["bin_idx"], targets["event"])

    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)


        eps = 1e-6  # good for float32; use ~1e-12 for float64

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.as_tensor(X, dtype=torch.float32))
            hazards = torch.sigmoid(logits).clamp(min=eps, max=1 - eps).cpu().numpy()

        surv_d = np.cumprod(1.0 - hazards, axis=1)  # discrete S_k at bin edges

        dense_times = np.linspace(0.0, self.max_time_, num_times)
        fn = make_interpolator(surv_d, self.time_grid_, mode=self.interpolation)
        return fn(dense_times)

