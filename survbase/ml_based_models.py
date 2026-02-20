from __future__ import annotations

from typing import Optional, Any, Dict, Sequence, Union
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.base import check_is_fitted, check_array

from survbase.base import SurvivalAnalysisMixin, SurvivalNNBase
from survbase.nonparametric import breslow_estimator, kaplan_meier
from survbase.utils import make_interpolator, build_mlp
from survbase.statistical_based_models import CoxPH

import torch
import torch.nn.functional as F

@dataclass
class _LeafSurvival:
    leaf_id: int
    times: np.ndarray
    surv: np.ndarray


# =====================================================================
# Survival Forest  — non-federated
# =====================================================================
class SurvivalForest(SurvivalAnalysisMixin):
    """
    Simplified, efficient Survival Forest.

    Uses sklearn's RandomForestRegressor for fast partitioning.
    Each leaf stores a Kaplan-Meier survival curve of its samples.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 3,
        max_features: str | int | float = "sqrt",
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = 42,
        max_samples: Optional[int | float] = None,
        interpolation: str = "linear",
        bootstrap: bool = True,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.interpolation = interpolation

    # ----------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Random Survival Forest.
        y: structured array with fields ('event', bool) and ('time', float)
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=float, copy=False)
        if y.dtype.names is None or len(y.dtype.names) < 2:
            raise ValueError("y must be structured array with ('event','time') fields")

        self.event_ = y[y.dtype.names[0]].astype(np.bool_)
        self.time_ = y[y.dtype.names[1]].astype(np.float64)
        self.y_ = y
        self.max_time_ = float(np.max(self.time_))

        # Regression target = time (rough proxy to survival ranking)
        y_reg = np.where(self.event_, self.time_, self.time_ * 1.1)

        X = check_array(X, ensure_2d=True, dtype=np.float64)

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
        )
        self._forest.fit(X, y_reg)

        self._leaf_models_: list[list[_LeafSurvival]] = []
        for est in self._forest.estimators_:
            leaf_ids = est.apply(X)
            uniq = np.unique(leaf_ids)
            leaf_survs = []
            for lid in uniq:
                idx = np.where(leaf_ids == lid)[0]
                if len(idx) < self.min_samples_leaf:
                    continue
                t, s = kaplan_meier(self.y_[idx])
                leaf_survs.append(_LeafSurvival(lid, t, s))
            self._leaf_models_.append(leaf_survs)
        return self

    def _predict_tree_surv(self, est_idx: int, X: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Predict survival matrix for one tree on given grid.
        """
        est = self._forest.estimators_[est_idx]
        leaf_models = self._leaf_models_[est_idx]
        leaf_map = {lm.leaf_id: lm for lm in leaf_models}

        leaf_ids = est.apply(X)
        n, m = X.shape[0], grid.shape[0]
        out = np.ones((n, m), dtype=np.float64)

        for i in range(n):
            lid = leaf_ids[i]
            model = leaf_map.get(lid, None)
            if model is not None:
                interp = make_interpolator(model.surv, model.times, mode=self.interpolation)
                out[i, :] = interp(grid)
        return out

    def predict(self, X: np.ndarray, num_times: int = 64) -> np.ndarray:
        """
        Predict survival probabilities S(t) over a uniform grid [0, max_time_].
        Returns array shape (n_samples, num_times).
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy(dtype=float, copy=False)
        if not hasattr(self, "_forest"):
            raise RuntimeError("Model not fitted yet.")

        grid = np.linspace(0.0, self.max_time_, num_times, dtype=np.float64)
        n = X.shape[0]
        S_all = np.zeros((n, num_times), dtype=np.float64)

        for t_idx in range(len(self._forest.estimators_)):
            S_all += self._predict_tree_surv(t_idx, X, grid)

        S_all /= len(self._forest.estimators_)
        return S_all


# =====================================================================
# XGBoostCox  — non-federated
# =====================================================================
class XGBoostCox(SurvivalAnalysisMixin):
    """
    Standard single-client XGBoost Cox model.

    Notes
    -----
    - Uses the common “sign trick” label encoding:
        label = +time for events, -time for censored.
      (Matches the original code you had; keep this unless you switch encodings.)
    - After training, Breslow baseline hazard is estimated on the *training* data.
      Prediction uses S(t) = exp(- H0(t) * exp(xβ)).
    """

    def __init__(
        self,
        *,
        num_boost_round: int = 200,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = 42,
        interpolation: str = "step",
    ):
        super().__init__()
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.interpolation = interpolation

    def fit(self, X, y):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32, copy=False)

        # Encode labels for XGBoost Cox
        label = y["time"].astype(np.float32).copy()
        label[y["event"] == 0] *= -1.0  # negative time for censored

        self.max_time_ = float(y["time"].max())

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "seed": self.random_state,
        }

        dtrain = xgb.DMatrix(X, label=label)
        self.booster_ = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            verbose_eval=False,
        )

        # ---- Breslow baseline on training set
        xbeta = self._predict_risk_scores(X)  # log-risk
        t_u, H0 = breslow_estimator(y, xbeta)
        self.baseline_times_ = t_u.astype(float, copy=False)
        self.baseline_cumhaz_ = H0.astype(float, copy=False)

        self.is_fitted_ = True
        return self

    def _predict_risk_scores(self, X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        dmat = xgb.DMatrix(X.astype(np.float32, copy=False))
        return self.booster_.predict(dmat, output_margin=True)

    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "booster_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32, copy=False)

        times = np.linspace(0.0, self.max_time_, num_times)
        log_risk = self._predict_risk_scores(X)
        risk = np.exp(log_risk)  # exp(xβ)

        # Individual cumulative hazard: H_i(t) = H0(t) * exp(xβ_i)
        H_matrix = self.baseline_cumhaz_.reshape(1, -1) * risk[:, None]  # (n, m0)
        fn = make_interpolator(H_matrix, self.baseline_times_, mode=self.interpolation)
        H_reg = fn(times)  # (n, num_times)
        return np.exp(-H_reg)


# =====================================================================
# DeepSurv (nonlinear Cox) — non-federated
# =====================================================================
class DeepSurv(CoxPH):
    """Nonlinear Cox network (same loss as CoxPH; different architecture)."""

    def __init__(
        self,
        *,
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
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.residual = residual
        self.use_attention = use_attention
        self.bias = bias

    def _init_net(self, n_features: int):
        return build_mlp(
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


# =====================================================================
# DeepHit (single event, discrete time) — non-federated
# =====================================================================
def _deephit_nll(logits: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    logits : (B, K) – raw network output
    bin_idx: (B,)   – interval of event/censor
    event  : (B,)   – 1 if event occurred, 0 if right-censored
    """
    pmf = F.softmax(logits, dim=1)  # P(T == t_k)
    log_pmf = F.log_softmax(logits, dim=1)

    surv = 1.0 - torch.cumsum(pmf, dim=1)  # S_k = P(T > t_k)
    idx = bin_idx.unsqueeze(1)
    log_pmf_k = log_pmf.gather(1, idx).squeeze(1)
    surv_k = surv.gather(1, idx).squeeze(1)

    loss_evt = -log_pmf_k * event
    loss_cen = -torch.log(torch.clamp(surv_k, min=1e-8)) * (1.0 - event)
    return (loss_evt + loss_cen).mean()


class DeepHit(SurvivalNNBase):
    """DeepHit (single-event) with only the likelihood term for brevity."""

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
        times = y["time"].astype(np.float32)
        events = y["event"].astype(np.float32)
        self.time_grid_ = np.linspace(0.0, times.max(), self.num_durations + 1)[1:]
        bin_idx = np.searchsorted(self.time_grid_, times, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, self.num_durations - 1).astype(np.int64)
        return {"bin_idx": bin_idx, "event": events}

    def _loss_fn(self, logits, targets):
        return _deephit_nll(logits, targets["bin_idx"], targets["event"])

    def predict(self, X, num_times: int = 64):
        check_is_fitted(self, "net_")
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = X.astype(np.float32)

        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        # Stable softmax in numpy
        logits = logits - logits.max(axis=1, keepdims=True)
        pmf = np.exp(logits)
        pmf /= pmf.sum(axis=1, keepdims=True)
        surv_d = 1.0 - np.cumsum(pmf, axis=1)  # S_k

        surv_d = np.clip(surv_d, 0.0, 1.0)
        dense_times = np.linspace(0.0, self.max_time_, num_times)
        fn = make_interpolator(surv_d, self.time_grid_, mode=self.interpolation)
        return fn(dense_times)

