import random
from typing import Callable, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_interpolator(
    values: np.ndarray | Sequence[float],
    times: np.ndarray | Sequence[float],
    *,
    mode: str = "step",
) -> Callable[[np.ndarray | Sequence[float]], np.ndarray]:
    """
    Build a left-continuous *step* interpolator or a *linear* interpolator.

    Parameters
    ----------
    values : array_like, shape (n_samples, n_times) or (n_times,)
        Function values for each sample at the grid ``times``.
    times  : array_like, shape (n_times,)
        Strictly increasing grid shared by all samples.
    mode   : {"step", "linear"}, default="step"
        Interpolation mode.
        * "step"   - left-continuous, piece-wise constant (previous behaviour)
        * "linear" - piece-wise linear between grid points

    Returns
    -------
    interp : callable
        A function `interp(eval_times)` that returns an array with shape
        (n_samples, len(eval_times)).  For points outside ``times`` the nearest
        boundary value is returned (no extrapolation).
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("`times` must be 1-D and strictly increasing.")
    if values.ndim == 1:  # promote to 2-D
        values = values[np.newaxis, :]
    if values.shape[1] != times.size:
        raise ValueError("`values` second dimension must equal `times` length.")
    if mode not in {"step", "linear"}:
        raise ValueError("`mode` must be 'step' or 'linear'.")

    # ------------------------------------------------------------------ helpers
    def _step(et: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(times, et, side="right") - 1  # left index
        idx = np.clip(idx, 0, times.size - 1)
        return values[:, idx]

    def _linear(et: np.ndarray) -> np.ndarray:
        out = np.empty((values.shape[0], et.size), dtype=float)
        for i in range(values.shape[0]):
            out[i] = np.interp(
                et,
                times,
                values[i],
                left=values[i, 0],
                right=values[i, -1],
            )
        return out

    _impl = _step if mode == "step" else _linear

    def interp(eval_times: np.ndarray | Sequence[float]) -> np.ndarray:
        et = np.asarray(eval_times, dtype=float)
        if et.ndim != 1:
            raise ValueError("`eval_times` must be 1-D.")
        return _impl(et)

    return interp


# ――― mapping of activation names → callables ――― #
_ACTS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "identity": nn.Identity,
}


class FeatureAttention(nn.Module):
    """
    Light-weight, feature-wise attention gate for tabular tensors.
    Scales each feature by a learned importance weight in (0, 1).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, F]
        attn = torch.softmax(self.score(x), dim=-1)  # B × F
        return x * attn


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, act, batchnorm, dropout, *, bias: bool):
        super().__init__()
        layers = []
        if batchnorm:
            layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.Linear(in_dim, out_dim, bias=bias))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out_dim, out_dim, bias=bias))
        self.main = nn.Sequential(*layers)
        self.proj = nn.Linear(in_dim, out_dim, bias=bias) if in_dim != out_dim else None

    def forward(self, x):
        out = self.main(x)
        skip = self.proj(x) if self.proj is not None else x
        return out + skip


def build_mlp(
    input_dim: int,
    output_dim: int,
    bias: bool = True,
    num_layers: int = 3,
    hidden_units: Union[int, Sequence[int]] = 128,
    batchnorm: bool = False,
    dropout: float = 0.0,
    activation: str = "relu",
    out_activation: Optional[str] = None,
    use_attention: bool = False,
    residual: bool = False,
) -> nn.Module:
    """
    Returns an nn.Module ready for training.

    Parameters
    ----------
    input_dim, output_dim : int
        Sizes of input feature vector and required output.
    bias : bool
        Use bias in all Linear layers.
    num_layers : int
        Total fully-connected blocks (not counting output head).
    hidden_units : int | Sequence[int]
        If int → every hidden layer gets this many units;
        if sequence → explicit width per layer.
    batchnorm : bool
        Insert BatchNorm1d before each Linear layer.
    dropout : float
        Drop probability after activation in each hidden layer.
    activation, out_activation : str | None
        Names from `_ACTS`.  `None` → nn.Identity().
    use_attention : bool
        Adds a simple feature-wise attention gate just after the input.
    residual : bool
        Use ResidualBlock instead of plain Linear+Act.
    """
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units] * num_layers
    else:
        if len(hidden_units) != num_layers:
            raise ValueError("hidden_units length must equal num_layers")

    if activation not in _ACTS:
        raise ValueError(f"Unknown activation: {activation!r}")
    act_cls = _ACTS[activation]
    if out_activation is None:
        out_act_cls = nn.Identity
    else:
        if out_activation not in _ACTS:
            raise ValueError(f"Unknown out_activation: {out_activation!r}")
        out_act_cls = _ACTS[out_activation]

    layers: List[nn.Module] = []

    # optional feature-attention gate
    if use_attention:
        layers.append(FeatureAttention(input_dim))

    # hidden blocks
    in_dim = input_dim
    for h in hidden_units:
        if residual:
            block = ResidualBlock(in_dim, h, act_cls, batchnorm, dropout, bias=bias)
            layers.append(block)
        else:
            if batchnorm:
                layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, h, bias=bias))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        in_dim = h

    # output head
    layers.append(nn.Linear(in_dim, output_dim, bias=bias))
    if out_activation is not None and out_activation.lower() != "identity":
        layers.append(out_act_cls())

    return nn.Sequential(*layers)
