from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from survbase.data import build_survival_y


def _winsorize_df(
    X_num: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    """
    Clip numerical columns to given quantiles (winsorization). Operates column-wise.
    Assumes X_num is a DataFrame of only numeric columns.
    """
    if X_num.shape[1] == 0 or (lower_q <= 0.0 and upper_q >= 1.0):
        return X_num

    q_low = X_num.quantile(lower_q)
    q_high = X_num.quantile(upper_q)
    return X_num.clip(lower=q_low, upper=q_high, axis="columns")


class IdentityDf(BaseEstimator, TransformerMixin):
    """
    Pass-through transformer that preserves DataFrame index/columns
    (useful to keep consistent types inside ColumnTransformer branches).
    """

    def fit(self, X, y=None):  # noqa: D401
        return self

    def transform(self, X):
        return X


@dataclass
class TabularPreprocessorConfig:
    # column discovery
    event_col: str = "event"
    time_col: str = "time"
    include_cols: Optional[Sequence[str]] = None  # if provided, only use these as features
    drop_cols: Optional[Sequence[str]] = None  # columns to always drop (besides event/time)

    # numeric
    numeric_imputation: str = "median"  # "median" or "mean" or "most_frequent"
    scale_numeric: bool = True
    winsorize_quantiles: Tuple[float, float] = (0.0, 1.0)  # e.g., (0.01, 0.99) to clip outliers
    treat_inf_as_na: bool = True

    # categorical
    categorical_imputation: str = "most_frequent"
    rare_threshold: float = 0.01  # <1% → infrequent category
    drop_first_dummy: bool = False  # k-1 encoding; keep False by default
    handle_unknown: str = "ignore"  # for OneHotEncoder

    # general
    drop_zero_variance: bool = True  # remove constant columns after encoding
    output_dataframe: bool = True  # return pandas.DataFrame instead of ndarray


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor for tabular survival features.

    - Detects numeric vs categorical features (excludes event/time columns).
    - Imputes NaNs (median/mean for numeric; most_frequent for categorical).
    - Optional winsorization of numerics to reduce outlier impact.
    - Standard scales numeric columns (zero mean / unit variance).
    - Groups infrequent categorical levels (< rare_threshold) via OneHotEncoder(min_frequency=...).
    - One-hot encodes categoricals (with handle_unknown='ignore' to safely transform val/test).
    - Drops post-encoding zero-variance columns (safe for train/val/test).
    - Preserves feature names and offers get_feature_names_out().

    Usage:
        prep = TabularPreprocessor(TabularPreprocessorConfig(...)).fit(df_train)
        X_train = prep.transform(df_train)
        X_val   = prep.transform(df_val)
        y_train = build_survival_y(df_train, event_col=..., time_col=...)
    """

    def __init__(self, config: Optional[TabularPreprocessorConfig] = None):
        self.config = config or TabularPreprocessorConfig()
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._ct: Optional[ColumnTransformer] = None
        self._final_selector: Optional[VarianceThreshold] = None
        self._feature_names_out: Optional[np.ndarray] = None

    # ---------- column discovery ----------
    def _discover_columns(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cfg = self.config
        feats = list(df.columns)

        # remove survival label columns
        for col in (cfg.event_col, cfg.time_col):
            if col in feats:
                feats.remove(col)

        # include/exclude controls
        if cfg.include_cols is not None:
            feats = [c for c in feats if c in set(cfg.include_cols)]
        if cfg.drop_cols is not None:
            feats = [c for c in feats if c not in set(cfg.drop_cols)]

        # split numeric vs categorical
        num_cols = [c for c in feats if is_numeric_dtype(df[c])]
        cat_cols = [c for c in feats if c not in num_cols]

        return num_cols, cat_cols

    # ---------- sklearn API ----------
    def fit(self, df: pd.DataFrame, y=None):  # noqa: D401
        cfg = self.config
        df = df.copy()

        # Handle inf values up-front (optional)
        if cfg.treat_inf_as_na:
            df = df.replace([np.inf, -np.inf], np.nan)

        self._num_cols, self._cat_cols = self._discover_columns(df)

        # Build numeric pipeline
        num_steps = []
        if cfg.winsorize_quantiles != (0.0, 1.0):
            lower, upper = cfg.winsorize_quantiles
            # Apply winsorization in DataFrame space within the numeric branch
            num_steps.append(
                (
                    "winsorize",
                    FunctionTransformer(
                        lambda X: _winsorize_df(
                            pd.DataFrame(X, columns=self._num_cols), lower, upper
                        ),
                        feature_names_out="one-to-one",
                    ),
                )
            )
        num_steps.append(("impute", SimpleImputer(strategy=cfg.numeric_imputation)))
        if cfg.scale_numeric:
            num_steps.append(("scale", StandardScaler()))

        num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

        # Build categorical pipeline
        cat_pipe = Pipeline(
            steps=[
                (
                    "impute",
                    SimpleImputer(strategy=cfg.categorical_imputation, fill_value="__MISSING__"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown=cfg.handle_unknown,
                        drop="first" if cfg.drop_first_dummy else None,
                        min_frequency=cfg.rare_threshold if cfg.rare_threshold > 0 else None,
                        sparse_output=False,
                    ),
                ),
            ]
        )

        # ColumnTransformer
        self._ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self._num_cols),
                ("cat", cat_pipe, self._cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # Fit ColumnTransformer
        self._ct.fit(df)

        # Optional final zero-variance filter (after encoding)
        if self.config.drop_zero_variance:
            self._final_selector = VarianceThreshold(threshold=0.0)
            self._final_selector.fit(self._ct.transform(df))

            # feature names after CT
            feat_names = self._get_ct_feature_names()
            mask = self._final_selector.get_support()
            self._feature_names_out = np.asarray(feat_names)[mask]
        else:
            self._final_selector = None
            self._feature_names_out = np.asarray(self._get_ct_feature_names())

        return self

    def transform(self, df: pd.DataFrame):
        if self._ct is None:
            raise RuntimeError("Call fit() before transform().")

        X = df.copy()
        if self.config.treat_inf_as_na:
            X = X.replace([np.inf, -np.inf], np.nan)

        Xt = self._ct.transform(X)
        if self._final_selector is not None:
            Xt = self._final_selector.transform(Xt)

        if self.config.output_dataframe:
            return pd.DataFrame(Xt, index=df.index, columns=self.get_feature_names_out())
        return Xt

    # ---------- helpers ----------
    def _get_ct_feature_names(self) -> List[str]:
        """
        Names emitted by the ColumnTransformer before the final selector.
        """
        assert self._ct is not None
        names: List[str] = []

        # numeric names
        if self._num_cols:
            # after impute/scale, still one column per numeric input
            names.extend(self._num_cols)

        # categorical expanded names
        if self._cat_cols:
            ohe: OneHotEncoder = self._ct.named_transformers_["cat"].named_steps["onehot"]
            cat_out = ohe.get_feature_names_out(self._cat_cols)
            names.extend(cat_out.tolist())

        return names

    def get_feature_names_out(self) -> np.ndarray:
        if self._feature_names_out is None:
            # Fall back to CT names if selector hasn't run
            return np.asarray(self._get_ct_feature_names())
        return self._feature_names_out

    # Convenience splitter if you want X/y separated in one call
    def split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        X = self.transform(df)
        y = build_survival_y(df, event_col=self.config.event_col, time_col=self.config.time_col)
        return X, y


def _extract_event_time(
    y: np.ndarray, event_name: Optional[str] = None, time_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    """
    Validate and extract (event, time) from a structured ndarray with ≥2 fields.
    Uses the first two fields by position unless explicit names are provided.
    Returns (event_bool, time_float, (event_field_name, time_field_name))
    """
    if y.dtype.names is None or len(y.dtype.names) < 2:
        raise ValueError("`y` must be a structured array with at least two fields (event, time).")
    names = y.dtype.names
    e_name = event_name or names[0]
    t_name = time_name or names[1]
    event = y[e_name].astype(np.bool_)
    time = y[t_name].astype(np.float64)
    return event, time, (e_name, t_name)


@dataclass
class SurvivalTargetConfig:
    # Scaling
    scale_time_to_unit: bool = True
    clip_outside_train_range: bool = True  # clamp val/test times to [t_min_, t_max_] before scaling

    # Discretization
    discretize: bool = False
    strategy: Literal["uniform", "quantile"] = "uniform"
    n_bins: Optional[int] = None  # required if bin_edges is None and discretize=True
    bin_edges: Optional[np.ndarray] = None  # optional explicit edges (length m+1)
    right_inclusive: bool = True  # whether the right edge is inclusive in binning

    # Outputs
    return_bins: bool = True  # return integer bin indices as a separate array
    return_bins_onehot: bool = False  # also return one-hot matrix for bins

    # Column names (optional; if None, uses first two fields by position)
    event_field: Optional[str] = None
    time_field: Optional[str] = None


class SurvivalTargetPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocess survival targets y = structured array [(event,bool),(time,float)].

    - fit(y_train): learns min/max time for scaling; and, if requested,
      bin edges from train (uniform or quantile) unless explicit edges given.
    - transform(y): returns a dict with:
        {
          "y_scaled": structured array like y but with time scaled to [0,1] (if enabled),
          "t_bins": np.ndarray[int] or None,
          "t_bins_onehot": np.ndarray[float] or None,
        }
      If scaling is disabled, "y_scaled" is just y (copied).

    Supports inverse_transform_time() to map scaled times back to the original scale,
    and get_bin_edges().
    """

    def __init__(self, config: Optional[SurvivalTargetConfig] = None):
        self.config = config or SurvivalTargetConfig()
        # learned during fit
        self.t_min_: float = 0.0
        self.t_max_: float = 1.0
        self._scale_den_: float = 1.0
        self.bin_edges_: Optional[np.ndarray] = None
        self.n_bins_: Optional[int] = None
        self._fitted_names_: Optional[Tuple[str, str]] = None

    # ---------- sklearn API ----------
    def fit(self, y: np.ndarray, X: Any = None):
        cfg = self.config
        event, time, names = _extract_event_time(y, cfg.event_field, cfg.time_field)
        self._fitted_names_ = names

        # learn scaling
        if cfg.scale_time_to_unit:
            finite = np.isfinite(time)
            if not finite.any():
                raise ValueError("All times are non-finite; cannot compute scaling.")
            self.t_min_ = float(np.nanmin(time[finite]))
            self.t_max_ = float(np.nanmax(time[finite]))
            self._scale_den_ = max(self.t_max_ - self.t_min_, 0.0)
        else:
            # defaults to avoid zero-div on transform if user calls inverse
            self.t_min_, self.t_max_, self._scale_den_ = 0.0, 1.0, 1.0

        # learn binning
        if cfg.discretize:
            if cfg.bin_edges is not None:
                edges = np.asarray(cfg.bin_edges, dtype=float)
                if edges.ndim != 1 or edges.size < 2:
                    raise ValueError("`bin_edges` must be a 1-D array with at least two values.")
                # ensure strictly increasing
                if not np.all(np.diff(edges) > 0):
                    raise ValueError("`bin_edges` must be strictly increasing.")
                self.bin_edges_ = edges
            else:
                if cfg.n_bins is None or cfg.n_bins < 1:
                    raise ValueError("When discretize=True and bin_edges is None, set n_bins >= 1.")
                if cfg.strategy == "uniform":
                    # Use (train) observed min/max (not clipped).
                    self.bin_edges_ = np.linspace(self.t_min_, self.t_max_, cfg.n_bins + 1)
                elif cfg.strategy == "quantile":
                    qs = np.linspace(0.0, 1.0, cfg.n_bins + 1)
                    self.bin_edges_ = np.quantile(time, qs)
                    # de-duplicate (possible with repeated quantiles), ensure strictly increasing
                    self.bin_edges_ = np.unique(self.bin_edges_)
                    if self.bin_edges_.size < 2:
                        raise ValueError(
                            "Quantile binning produced <2 unique edges. Increase variability or n_bins."
                        )
                else:
                    raise ValueError("`strategy` must be 'uniform' or 'quantile'.")

            self.n_bins_ = self.bin_edges_.size - 1
        else:
            self.bin_edges_ = None
            self.n_bins_ = None

        return self

    def transform(self, y: np.ndarray) -> Dict[str, Any]:
        if self._fitted_names_ is None:
            raise RuntimeError("Call fit() before transform().")

        cfg = self.config
        e_name, t_name = self._fitted_names_
        event, time, _ = _extract_event_time(y, e_name, t_name)

        time_in = time.copy()

        # Optionally clip to train range before scaling to keep val/test in-range
        if cfg.scale_time_to_unit and cfg.clip_outside_train_range:
            time_in = np.clip(time_in, self.t_min_, self.t_max_)

        # Scale to [0,1] (safe when range==0)
        if cfg.scale_time_to_unit:
            if self._scale_den_ == 0.0:
                time_scaled = np.zeros_like(time_in, dtype=float)
            else:
                time_scaled = (time_in - self.t_min_) / self._scale_den_
            time_out = time_scaled
        else:
            time_out = time_in.astype(float, copy=False)

        # Build structured y with the same field names
        y_scaled = np.zeros(len(y), dtype=[(e_name, bool), (t_name, float)])
        y_scaled[e_name] = event
        y_scaled[t_name] = time_out

        t_bins = None
        t_bins_onehot = None
        if cfg.discretize and self.bin_edges_ is not None:
            # numpy digitize: control right-inclusiveness
            # bins are 0..n_bins-1; values below first edge -> -1; above last edge -> n_bins
            right = bool(cfg.right_inclusive)
            idx = np.digitize(time, self.bin_edges_, right=right) - 1  # shift to 0-based
            # clamp to [0, n_bins_-1]
            if self.n_bins_ and self.n_bins_ > 0:
                idx = np.clip(idx, 0, self.n_bins_ - 1)
            t_bins = idx.astype(np.int64, copy=False)

            if cfg.return_bins_onehot:
                n = len(time)
                k = int(self.n_bins_ or 0)
                oh = np.zeros((n, k), dtype=float)
                if k > 0:
                    oh[np.arange(n), t_bins] = 1.0
                t_bins_onehot = oh

        return {
            "y_scaled": y_scaled,
            "t_bins": t_bins,
            "t_bins_onehot": t_bins_onehot,
        }

    # ---------- utilities ----------
    def inverse_transform_time(self, time_scaled: np.ndarray) -> np.ndarray:
        """
        Map times from [0,1] back to the original scale learned on train.
        If scaling was disabled, this returns the input.
        """
        if not self.config.scale_time_to_unit:
            return np.asarray(time_scaled, dtype=float)
        return np.asarray(time_scaled, dtype=float) * self._scale_den_ + self.t_min_

    def get_bin_edges(self) -> Optional[np.ndarray]:
        return None if self.bin_edges_ is None else self.bin_edges_.copy()
