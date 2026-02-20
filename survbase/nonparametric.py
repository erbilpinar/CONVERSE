import numpy as np
from numba import njit


@njit(cache=True)
def _kaplan_meier_numba(event: np.ndarray, time: np.ndarray):
    order = np.argsort(time)
    ev = event[order]
    tt = time[order]

    n = len(tt)
    surv = 1.0

    out_t = []  # start empty to avoid double 0.0
    out_s = []

    i = 0
    while i < n:
        t = tt[i]
        d_i = 0
        j = i
        while j < n and tt[j] == t:
            if ev[j]:
                d_i += 1
            j += 1

        if d_i > 0:
            n_i = n - i
            if n_i > 0:
                surv *= 1.0 - d_i / n_i
            # prepend baseline if first event time > 0
            if len(out_t) == 0 and t > 0.0:
                out_t.append(0.0)
                out_s.append(1.0)
            out_t.append(t)
            out_s.append(surv)

        i = j

    if len(out_t) == 0:  # no events safeguard (should be handled by wrapper)
        return np.asarray([0.0], np.float64), np.asarray([1.0], np.float64)

    return np.asarray(out_t, np.float64), np.asarray(out_s, np.float64)


def kaplan_meier(y: np.ndarray, *, reverse: bool = False):
    """Python wrapper with dtype-agnostic field access."""
    if y.dtype.names is None or len(y.dtype.names) < 2:
        raise ValueError("`y` must be a structured array with ≥2 fields")

    event = y[y.dtype.names[0]].astype(np.bool_)
    time = y[y.dtype.names[1]].astype(np.float64)

    if reverse:
        event = ~event

    # Decide policy: return trivial curve instead of raising.
    if not np.any(event):
        return np.asarray([0.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)

    return _kaplan_meier_numba(event, time)


@njit(cache=True)
def _breslow_cumhaz_numba(event, time, linpred):
    """
    Breslow cumulative baseline hazard Λ̂₀(t).
    Parameters
    ----------
    event   : 1-d bool array  (δᵢ)
    time    : 1-d float array (Tᵢ)
    linpred : 1-d float array (ηᵢ = Xᵢ·β)
    Returns
    -------
    t_unique : (m,) float array of distinct event times
    cumhaz   : (m,) float array, Breslow Λ̂₀ evaluated at t_unique
    """
    # sort by time ascending
    order = np.argsort(time)
    tt = time[order]
    ev = event[order]
    eta = linpred[order]

    # pre-compute exponentials once (O(n))
    riskexp = np.exp(eta)

    n = len(tt)
    at_risk_sum = riskexp.sum()  # initial denominator
    cumhaz = 0.0

    out_t = []
    out_ch = []

    i = 0
    while i < n:
        t = tt[i]

        # all individuals with this exact time
        j = i
        d_i = 0
        while j < n and tt[j] == t:
            if ev[j]:
                d_i += 1
            j += 1

        if d_i:  # only update at event times
            delta = d_i / at_risk_sum if at_risk_sum > 0 else 0.0
            cumhaz += delta
            out_t.append(t)
            out_ch.append(cumhaz)

        # remove those who just failed or were censored at t from risk set
        for k in range(i, j):
            at_risk_sum -= riskexp[k]

        i = j  # jump to next distinct time

    return np.asarray(out_t), np.asarray(out_ch)


def breslow_estimator(y, linpred):
    """
    Wrapper around the Numba kernel.
    Parameters
    ----------
    y        : structured array with fields ('event','time')
    linpred  : 1-d array of pre-computed linear predictors (same length as y)
    Returns
    -------
    t_unique, cumhaz : 1-d arrays
    """
    if y.dtype.names is None or len(y.dtype.names) < 2:
        raise ValueError("`y` must be a structured array with ≥2 fields")
    event = y[y.dtype.names[0]].astype(np.bool_)
    time = y[y.dtype.names[1]].astype(np.float64)

    if len(linpred) != len(time):
        raise ValueError("`linpred` must have same length as y")

    t, ch = _breslow_cumhaz_numba(event, time, linpred.astype(np.float64))
    return t, ch
