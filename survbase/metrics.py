import math
from typing import Literal, Sequence

import numpy as np
from numba import njit

from survbase.nonparametric import kaplan_meier
from survbase.utils import make_interpolator


@njit(cache=True)
def _concordance_index_numba(events: np.ndarray, times: np.ndarray, preds: np.ndarray) -> float:
    """
    Low-level Concordance-index computed with numba.

    Parameters
    ----------
    events : 1-d bool array
        `True` if the subject experienced the event, `False` if censored.
    times  : 1-d float array
        Observed times (same length as `events`).
    preds  : 1-d float array
        Risk scores or predicted hazards (same length as `events`).

    Returns
    -------
    float
        Concordance index in the interval [0, 1].  `np.nan` if no
        comparable pair is found.
    """
    n = events.shape[0]
    concordant = 0.0  # weighted count of concordant pairs
    comparable = 0.0  # count of comparable pairs

    for i in range(n - 1):
        ti = times[i]
        ei = events[i]
        pi = preds[i]

        for j in range(i + 1, n):
            tj = times[j]
            ej = events[j]
            pj = preds[j]

            if ti == tj:  # tied follow-up ⇒ not comparable
                continue

            # Determine the “winner” of the pair (earlier event)
            if ei and ti < tj:
                comparable += 1.0
                if pi > pj:
                    concordant += 1.0
                elif pi == pj:
                    concordant += 0.5

            elif ej and tj < ti:
                comparable += 1.0
                if pj > pi:
                    concordant += 1.0
                elif pj == pi:
                    concordant += 0.5

    return math.nan if comparable == 0 else concordant / comparable


def concordance_index(y_true: np.ndarray, preds: np.ndarray) -> float:
    """
    Python-friendly wrapper around `_concordance_index_numba`.

    Parameters
    ----------
    y_true : structured ndarray
        Must have two fields *in this order*:
          1. event indicator (bool)
          2. time (float)
        The field names themselves may vary.
    preds  : 1-d array-like
        Risk scores, hazards, or any monotone transformation thereof.

    Returns
    -------
    float
        Concordance index, or `np.nan` when no permissible pairs exist.

    Raises
    ------
    ValueError
        If inputs are malformed.
    """
    if len(y_true) != len(preds):
        raise ValueError("`y_true` and `preds` must have the same length")

    if y_true.dtype.names is None or len(y_true.dtype.names) < 2:
        raise ValueError(
            "`y_true` must be a structured array with at least two fields "
            "(event indicator first, time second)"
        )

    # Extract by position, not by name, to honour arbitrary field names
    events = y_true[y_true.dtype.names[0]].astype(np.bool_)
    times = y_true[y_true.dtype.names[1]].astype(np.float64)
    preds = np.asarray(preds, dtype=np.float64)

    # Substitute non-finite predictions with the average of finite ones
    if not np.all(np.isfinite(preds)):
        finite_mask = np.isfinite(preds)
        if np.any(finite_mask):
            mean_finite = np.mean(preds[finite_mask])
        else:
            mean_finite = 0.5
        preds = np.where(finite_mask, preds, mean_finite)

    return _concordance_index_numba(events, times, preds)


# --------------------------------------------------------------------------- #
# numba core
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _ibs_numba(event, time, preds_eval, t_grid, km_t, km_s, w_max):
    """
    event      : bool array, shape (n,)
    time       : float array, shape (n,)
    preds_eval : float array, shape (n, m)
                 – predicted S(t) for every sample and grid time
    t_grid     : float array, shape (m,)
                 – evaluation time grid (monotone increasing)
    km_t/km_s  : Kaplan-Meier grid of censoring survival Ĝ(t)
    w_max      : clipping bound for IPCW weights
    """
    n = event.shape[0]
    m = t_grid.shape[0]

    # helper: left-continuous step function lookup for Ĝ
    def _G_hat(t):
        idx = np.searchsorted(km_t, t, side="right") - 1
        if idx < 0:
            return 1.0
        return km_s[idx]

    # ------------------------------------------------------------------ BS(t)
    bs = np.empty(m, dtype=np.float64)

    for j in range(m):
        t = t_grid[j]
        G_t = _G_hat(t)

        acc = 0.0
        for i in range(n):
            if time[i] >= t:  # still under observation at t
                if G_t == 0.0:
                    continue
                w = 1.0 / G_t
                y_it = 1.0  # survived to t
            elif event[i]:  # event before t
                G_tau = _G_hat(time[i])
                if G_tau == 0.0:
                    continue
                w = 1.0 / G_tau
                y_it = 0.0
            else:  # censored before t ⇒ skip
                continue

            # clip IPCW weight
            if w > w_max:
                w = w_max

            d = y_it - preds_eval[i, j]
            acc += w * d * d

        bs[j] = acc / n  # denominator = n, as in Graf et al.

    # ---------------------------------------------------------------- integral
    ibs = 0.0
    for j in range(m - 1):
        dt = t_grid[j + 1] - t_grid[j]
        ibs += 0.5 * (bs[j] + bs[j + 1]) * dt
    ibs /= t_grid[-1] - t_grid[0]

    return ibs


@njit(cache=True)
def _replace_nonfinite(a: np.ndarray) -> np.ndarray:
    """
    Replace non-finite values with the previous value along the last axis,
    initialized to 1.0 if the first value(s) are non-finite.
    """
    n, m = a.shape
    out = np.empty_like(a)
    for i in range(n):
        last = 1.0
        for j in range(m):
            v = a[i, j]
            if math.isfinite(v):
                out[i, j] = v
                last = v
            else:
                out[i, j] = last
    return out



# --------------------------------------------------------------------------- #


def integrated_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    preds: np.ndarray,
    *,
    pred_times: Sequence[float] | np.ndarray | None = None,
    mode: Literal["step", "linear"] = "step",
    weight_cap: float = 10.0,
) -> float:
    """
    Compute the Integrated Brier Score (IBS) on a 10-90 % time window.

    Parameters
    ----------
    y_train, y_test : structured arrays with first field = event (bool),
                      second field = time (float)
    preds           : ndarray, shape (n_test, n_pred_times)
                      Predicted survival probabilities Ŝ(t) for each subject.
    pred_times      : 1-D array of times corresponding to ``preds`` columns.
                      If *None*, it is assumed to be equally spaced in
                      [0, max(y_train['time'])].
    mode            : Interpolation mode passed to ``make_interpolator``.
    weight_cap      : maximum IPCW weight for the Brier score calculation.

    Returns
    -------
    ibs : float
        Integrated Brier Score.
    """
    # ----------------------------- field access (works for arbitrary names)
    ev_tr, tt_tr = y_train[y_train.dtype.names[0]], y_train[y_train.dtype.names[1]]
    ev_te, tt_te = y_test[y_test.dtype.names[0]], y_test[y_test.dtype.names[1]]

    if len(ev_te) != preds.shape[0]:
        raise ValueError("`preds` rows must equal `y_test` length.")

    # ----------------------------- KM of censoring on the *training* set
    km_t, km_s = kaplan_meier(y_train, reverse=True)  # censoring survival Ĝ

    # ----------------------------- evaluation window & grid
    t_lo, t_hi = np.quantile(tt_tr, [0.10, 0.90])
    if t_hi <= t_lo:
        raise ValueError("Degenerate 10-90 % interval. Check the training times.")

    if pred_times is None:
        pred_times = np.linspace(0.0, tt_tr.max(), preds.shape[1], dtype=float)
    else:
        pred_times = np.asarray(pred_times, dtype=float)
        if pred_times.ndim != 1 or pred_times.size != preds.shape[1]:
            raise ValueError("`pred_times` must be 1-D and match `preds` columns.")

    mask = (pred_times >= t_lo) & (pred_times <= t_hi)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two prediction points inside 10-90 % window.")
    t_grid = pred_times[mask]

    # ----------------------------- interpolate predictions on t_grid
    preds = _replace_nonfinite(preds)
    interp = make_interpolator(preds, pred_times, mode=mode)
    preds_eval = interp(t_grid).astype(np.float64)  # shape (n_test, m)

    # ----------------------------- IBS via numba core
    ibs = _ibs_numba(
        ev_te.astype(np.bool_),
        tt_te.astype(np.float64),
        preds_eval,
        t_grid.astype(np.float64),
        km_t.astype(np.float64),
        km_s.astype(np.float64),
        weight_cap,
    )
    return float(ibs)


@njit(cache=True)
def _one_calibration_numba(
    group_ids: np.ndarray,
    preds_eval: np.ndarray,
    t_grid: np.ndarray,
    km_curves_t: np.ndarray,
    km_curves_s: np.ndarray,
    km_lengths: np.ndarray,
    n_groups: int,
) -> float:
    """
    Numba core for 1-calibration computation.

    Parameters
    ----------
    group_ids   : (n_test,) int array - risk group assignment for each sample
    preds_eval  : (n_test, m) float array - predicted S(t) at grid points
    t_grid      : (m,) float array - evaluation time grid
    km_curves_t : (n_groups, max_len) float array - KM time points per group
    km_curves_s : (n_groups, max_len) float array - KM survival per group
    km_lengths  : (n_groups,) int array - actual length of each KM curve
    n_groups    : int - number of risk groups

    Returns
    -------
    float
        1-calibration (D-calibration) score
    """
    n = preds_eval.shape[0]
    m = t_grid.shape[0]

    # Count samples per group
    group_counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(n):
        group_counts[group_ids[i]] += 1

    # Calibration error at each time point
    cal_errors = np.zeros(m, dtype=np.float64)

    for j in range(m):
        t = t_grid[j]

        for g in range(n_groups):
            if group_counts[g] == 0:
                continue

            # Get KM estimate at time t (left-continuous step function)
            km_length = km_lengths[g]
            km_at_t = 1.0  # default if t < first event time
            for k in range(km_length):
                if km_curves_t[g, k] <= t:
                    km_at_t = km_curves_s[g, k]
                else:
                    break

            # Compute mean predicted survival for this group at time t
            mean_pred = 0.0
            for i in range(n):
                if group_ids[i] == g:
                    mean_pred += preds_eval[i, j]
            mean_pred /= group_counts[g]

            # Accumulate absolute calibration error
            cal_errors[j] += np.abs(km_at_t - mean_pred)

    # Average over groups (divide by number of non-empty groups)
    active_groups = 0
    for g in range(n_groups):
        if group_counts[g] > 0:
            active_groups += 1

    if active_groups > 0:
        cal_errors /= active_groups

    # Integrate using trapezoidal rule
    ici = 0.0
    for j in range(m - 1):
        dt = t_grid[j + 1] - t_grid[j]
        ici += 0.5 * (cal_errors[j] + cal_errors[j + 1]) * dt

    if t_grid[-1] > t_grid[0]:
        ici /= t_grid[-1] - t_grid[0]

    return ici


def one_calibration(
    y_test: np.ndarray,
    preds: np.ndarray,
    *,
    pred_times: Sequence[float] | np.ndarray | None = None,
    mode: Literal["step", "linear"] = "step",
    n_groups: int = 10,
) -> float:
    """
    Compute the 1-calibration (D-calibration) metric on a 10-90 % time window.

    This metric measures calibration by stratifying patients into risk groups
    and comparing predicted survival probabilities to observed Kaplan-Meier
    estimates within each group.

    Parameters
    ----------
    y_test      : structured array with first field = event (bool),
                  second field = time (float)
    preds       : ndarray, shape (n_test, n_pred_times)
                  Predicted survival probabilities Ŝ(t) for each subject.
    pred_times  : 1-D array of times corresponding to ``preds`` columns.
                  If *None*, assumed equally spaced in [0, max(y_test['time'])].
    mode        : Interpolation mode passed to ``make_interpolator``.
    n_groups    : Number of risk groups (default 10 for deciles).

    Returns
    -------
    one_cal : float
        1-calibration score (lower is better, 0 = perfect calibration).

    References
    ----------
    Haider et al. (2020). Effective Ways to Build and Evaluate Individual
    Survival Distributions. JMLR.
    """
    # Extract fields (arbitrary names supported)
    ev_te = y_test[y_test.dtype.names[0]].astype(np.bool_)
    tt_te = y_test[y_test.dtype.names[1]].astype(np.float64)

    if len(ev_te) != preds.shape[0]:
        raise ValueError("`preds` rows must equal `y_test` length.")

    # Evaluation window: 10-90 percentiles
    t_lo, t_hi = np.quantile(tt_te, [0.10, 0.90])
    if t_hi <= t_lo:
        raise ValueError("Degenerate 10-90 % interval. Check the test times.")

    # Establish prediction times grid
    if pred_times is None:
        pred_times = np.linspace(0.0, tt_te.max(), preds.shape[1], dtype=float)
    else:
        pred_times = np.asarray(pred_times, dtype=float)
        if pred_times.ndim != 1 or pred_times.size != preds.shape[1]:
            raise ValueError("`pred_times` must be 1-D and match `preds` columns.")

    mask = (pred_times >= t_lo) & (pred_times <= t_hi)
    if np.sum(mask) < 2:
        raise ValueError("Need at least two prediction points inside 10-90 % window.")
    t_grid = pred_times[mask]

    # Interpolate predictions on t_grid
    if not np.all(np.isfinite(preds)):
        raise ValueError("`preds` contains non-finite values.")
    interp = make_interpolator(preds, pred_times, mode=mode)
    preds_eval = interp(t_grid).astype(np.float64)  # shape (n_test, m)

    # Assign risk groups based on mean predicted survival
    mean_surv_pred = preds_eval.mean(axis=1)
    # Higher mean survival = lower risk, so we invert for grouping
    # Group 0 = highest risk (lowest predicted survival)
    group_ids = np.zeros(len(ev_te), dtype=np.int64)
    sorted_idx = np.argsort(mean_surv_pred)  # ascending order
    
    samples_per_group = len(ev_te) // n_groups
    remainder = len(ev_te) % n_groups
    
    start_idx = 0
    for g in range(n_groups):
        # Distribute remainder across first groups
        group_size = samples_per_group + (1 if g < remainder else 0)
        end_idx = start_idx + group_size
        group_ids[sorted_idx[start_idx:end_idx]] = g
        start_idx = end_idx

    # Compute KM curve for each group
    max_km_length = 0
    group_km_t_list = []
    group_km_s_list = []

    for g in range(n_groups):
        mask_g = group_ids == g
        if not np.any(mask_g):
            group_km_t_list.append(np.array([0.0]))
            group_km_s_list.append(np.array([1.0]))
            max_km_length = max(max_km_length, 1)
            continue

        # Create structured array for this group
        y_group = np.empty(
            np.sum(mask_g),
            dtype=[(y_test.dtype.names[0], np.bool_), (y_test.dtype.names[1], np.float64)],
        )
        y_group[y_test.dtype.names[0]] = ev_te[mask_g]
        y_group[y_test.dtype.names[1]] = tt_te[mask_g]

        km_t, km_s = kaplan_meier(y_group)
        group_km_t_list.append(km_t)
        group_km_s_list.append(km_s)
        max_km_length = max(max_km_length, len(km_t))

    # Pack KM curves into fixed-size arrays for numba
    km_curves_t = np.zeros((n_groups, max_km_length), dtype=np.float64)
    km_curves_s = np.zeros((n_groups, max_km_length), dtype=np.float64)
    km_lengths = np.zeros(n_groups, dtype=np.int64)

    for g in range(n_groups):
        length = len(group_km_t_list[g])
        km_curves_t[g, :length] = group_km_t_list[g]
        km_curves_s[g, :length] = group_km_s_list[g]
        km_lengths[g] = length

    # Compute 1-calibration via numba core
    ici = _one_calibration_numba(
        group_ids,
        preds_eval,
        t_grid,
        km_curves_t,
        km_curves_s,
        km_lengths,
        n_groups,
    )

    return float(ici)
