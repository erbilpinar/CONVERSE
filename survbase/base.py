from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import torch
from sklearn import set_config
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from survbase.metrics import concordance_index, integrated_brier_score

set_config(enable_metadata_routing=True)


class SurvivalAnalysisMixin(BaseEstimator):
    """
    Base mixin for survival estimators.

    Requires subclasses to implement:
      - fit(X, y)
      - predict(X, num_times=...)
        (returns survival probabilities S(t) with shape [n_samples, n_times])

    Provides:
      - predict_risk(X): default risk = -∫ S(t) dt (negative mean survival time)
      - score(X_test, y_test, y_train): C-index minus 2 * IBS
    """

    def __init__(self) -> None:
        # opt in to sklearn metadata routing for scoring
        self.set_score_request(y_test=True, y_train=True)
        super().__init__()

    # ---------------- Abstract API ----------------
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X, num_times: int = 64):
        raise NotImplementedError

    # ---------------- Provided defaults ----------------
    def predict_risk(self, X, num_times: int = 64) -> np.ndarray:
        """
        Default risk = - ∫ S(t) dt  (negative mean survival time).

        Assumes `predict` returns an array S with shape (n_samples, n_times)
        evaluated on an *evenly-spaced* grid from 0 to self.max_time_ (set in fit).
        Subclasses (e.g., partial/semi-parametric models) can override this method.
        """
        S = self.predict(X, num_times=num_times)  # (n, m)
        if S.ndim != 2:
            raise ValueError("predict(X) must return a 2-D array [n_samples, n_times].")
        if not hasattr(self, "max_time_"):
            raise AttributeError("self.max_time_ not set. Set it in fit().")

        m = S.shape[1]
        t = np.linspace(0.0, float(self.max_time_), m, dtype=float)
        # Negative area under the survival curve (mean survival time with a minus sign)
        risk = -np.trapezoid(S, t, axis=1)
        return risk

    def score(self, X_test, y_test, y_train):
        """
        Score = C-index(risk) - 2 * IBS, where risk is by default -∫S(t)dt.
        """
        preds = self.predict(X_test)  # S(t) matrix
        risks = self.predict_risk(X_test)  # default negative-MST
        return concordance_index(y_test, risks) - 2.0 * integrated_brier_score(
            y_train, y_test, preds
        )


class SurvivalNNBase(SurvivalAnalysisMixin, ABC):
    """
    Base class for PyTorch survival models with a simple, correct training loop.
    Subclasses must implement:
      - _init_net(n_features) -> torch.nn.Module
      - _prepare_targets(y) -> Dict[str, Any]  (numpy arrays)
      - _loss_fn(logits, targets) -> torch.Tensor
    Optionally:
      - _post_fit(X_train, y_train, targets_train)  (e.g., compute baseline hazards)

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    lr_scheduler : {"plateau","step",None}
        Optional LR scheduler.
    weight_decay : float
        Adam weight decay.
    patience : int
        Early stopping patience on validation loss.
    val_fraction : float
        Fraction of data used for validation split.
    interpolation : {"step","linear"}
        Interpolation mode used downstream for predict().
    random_state : int
        Seed for split and torch RNG.
    device : {"cpu","cuda",None}
        Training device (auto if None).
    batch_size : Optional[int]
        If None (default), trains full-batch. If an int, uses shuffled mini-batches
        of that size each epoch.
    """

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
        batch_size: int | None = None,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.interpolation = interpolation
        self.random_state = random_state
        self._device_str = device  # "cpu" | "cuda" | None -> auto
        self.batch_size = batch_size  # None -> full-batch; int -> mini-batch size
        self.warm_start = False  # for sklearn compatibility

    # ---------------------- abstract hooks -----------------------------
    @abstractmethod
    def _init_net(self, n_features: int) -> torch.nn.Module: ...

    @abstractmethod
    def _prepare_targets(self, y) -> Dict[str, Any]: ...

    @abstractmethod
    def _loss_fn(self, logits: torch.Tensor, targets: Dict[str, Any]) -> torch.Tensor: ...

    def _post_fit(self, X, y, targets_train):
        """Optional post-processing (e.g., baseline estimation)."""
        pass

    # ---------------------- training loop -----------------------------
    def fit(self, X, y):
        # to numpy float32
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float32)

        # preprocess labels/targets (numpy arrays)
        targets = self._prepare_targets(y)
        self.max_time_ = float(np.asarray(y["time"]).max())

        # validation split (stratified on event)
        idx_train, idx_val = train_test_split(
            np.arange(len(X)),
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=np.asarray(y["event"], dtype=np.bool_),
        )

        # device
        device = torch.device(self._device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

        # tensors
        X_train_t = torch.tensor(X[idx_train], dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X[idx_val], dtype=torch.float32, device=device)

        train_targets_t = {k: torch.tensor(v[idx_train]).to(device) for k, v in targets.items()}
        val_targets_t = {k: torch.tensor(v[idx_val]).to(device) for k, v in targets.items()}

        # model & optimizer
        # Allow pretraining and warm start for DVCSurv base models
        reuse = bool(self.warm_start) and hasattr(self, "net_")
        if reuse:
            # keep the existing network and move to the current device
            net = self.net_.to(device)
        else:
            net = self._init_net(X.shape[1]).to(device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # scheduler (optional)
        sch = None
        if self.lr_scheduler == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=max(1, self.patience // 2)
            )
        elif self.lr_scheduler == "step":
            sch = torch.optim.lr_scheduler.StepLR(
                opt, step_size=max(1, self.epochs // 3), gamma=0.5
            )

        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        wait = 0

        self.train_loss_history_ = []
        self.val_loss_history_ = []

        torch.manual_seed(self.random_state)
        net.train()
        n_train = X_train_t.shape[0]
        bsz = self.batch_size

        for epoch in range(self.epochs):
            # ---- train (full batch or mini-batch) ----
            if bsz is None or bsz >= n_train:
                # full batch (original behavior)
                opt.zero_grad(set_to_none=True)
                logits_tr = net(X_train_t)
                loss_tr = self._loss_fn(logits_tr, train_targets_t)
                loss_tr.backward()
                opt.step()
                epoch_train_loss = float(loss_tr.item())
            else:
                # mini-batch
                perm = torch.randperm(n_train, device=device)
                running = 0.0
                total = 0
                for start in range(0, n_train, bsz):
                    idxb = perm[start : start + bsz]
                    xb = X_train_t.index_select(0, idxb)
                    tb = {k: v.index_select(0, idxb) for k, v in train_targets_t.items()}

                    opt.zero_grad(set_to_none=True)
                    logits_b = net(xb)
                    loss_b = self._loss_fn(logits_b, tb)
                    loss_b.backward()
                    opt.step()

                    nb = int(idxb.numel())
                    running += float(loss_b.item()) * nb
                    total += nb
                epoch_train_loss = running / max(1, total)

            self.train_loss_history_.append(epoch_train_loss)

            # ---- val ----
            net.eval()
            with torch.no_grad():
                logits_val = net(X_val_t)
                loss_val = self._loss_fn(logits_val, val_targets_t)
            self.val_loss_history_.append(loss_val.item())

            # scheduler step
            if sch is not None:
                if self.lr_scheduler == "plateau":
                    sch.step(loss_val.item())
                else:
                    sch.step()

            # ---- early stopping ----
            if loss_val.item() < best_val - 1e-6:
                best_val = loss_val.item()
                wait = 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
            net.train()

        # restore best
        if best_state is not None:
            net.load_state_dict(best_state)

        # keep CPU copy for inference
        self.net_ = net.to("cpu")
        self.is_fitted_ = True

        # optional post-fit on train portion
        self._post_fit(X[idx_train], y[idx_train], {k: v[idx_train] for k, v in targets.items()})
        return self
