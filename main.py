"""
WJRC Forecasting Service — Box-Jenkins SARIMA via statsmodels.

A small HTTP service the Laravel app calls out to for the actual model
identification, estimation, and forecasting step. Laravel still owns pulling
the raw job-order / product-sales history from the database and all of the
descriptive analytics around the forecast (seasonal profile, top services,
etc.) — this service's only job is: given a numeric series, fit a SARIMA
model (auto-selected by AICc, the standard Box-Jenkins model-selection
criterion), fit a non-seasonal ARIMA baseline on the same data, and return
the forecast, its confidence interval, an accuracy backtest for each (MSE,
MAE, RMSE, MAPE on the same 12-month hold-out), and a couple of diagnostic
checks — per FR21.

Using statsmodels here (instead of a hand-rolled optimizer) means the actual
MLE fitting, forecast-interval math, and statistical tests (ADF, Ljung-Box)
are the same well-tested implementations any textbook Box-Jenkins analysis
in Python or R would use.
"""

import os
import time
import warnings
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

app = FastAPI(title="WJRC Forecasting Service")

# Shared-secret check so this service (public on the internet once deployed)
# only answers requests from the Laravel app, not anyone who finds the URL.
# Left unset locally so it's easy to hit the API during development.
API_KEY = os.environ.get("FORECAST_API_KEY")


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

Z_SCORES = {90: 1.645, 95: 1.96, 99: 2.576}


class ForecastRequest(BaseModel):
    values: list[float] = Field(..., min_length=1)
    seasonal_period: int = 12
    steps: int = 12
    confidence: int = 95


class ForecastPoint(BaseModel):
    value: float
    lower: float
    upper: float


class Accuracy(BaseModel):
    available: bool
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    holdout: int = 0
    actual: list[float] = Field(default_factory=list)
    predicted: list[float] = Field(default_factory=list)


class Stationarity(BaseModel):
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None
    ljung_box_pvalue: Optional[float] = None
    is_stationary: Optional[bool] = None


class Baseline(BaseModel):
    """Non-seasonal ARIMA fit on the same series, scored on the same
    hold-out window as the SARIMA model — FR21's comparison baseline."""
    fitted: bool
    order_label: str
    order: list[int] = Field(default_factory=list)
    accuracy: Accuracy


class ForecastResponse(BaseModel):
    fitted: bool
    order: list[int]
    seasonal_order: list[int]
    order_label: str
    aicc: Optional[float] = None
    forecast: list[ForecastPoint]
    accuracy: Accuracy
    stationarity: Stationarity
    baseline: Optional[Baseline] = None


def order_label(order: tuple, seasonal_order: tuple) -> str:
    p, d, q = order
    P, D, Q, s = seasonal_order
    if s > 1 and (P + D + Q) > 0:
        return f"SARIMA({p},{d},{q})({P},{D},{Q}){s}"
    return f"ARIMA({p},{d},{q})"


def identify_d(y: list[float], max_d: int = 2) -> int:
    """Box-Jenkins identification step for the regular differencing order:
    KPSS stationarity test on the series, differencing and retesting while it
    rejects the stationarity null (p < 0.05) or until max_d is reached — the
    unit-root test R's forecast::auto.arima() / ndiffs() use by default. (ADF
    has low power on short series and tends to over-difference: on this
    system's ~60-point seasonally-differenced series it asked for d=2.)
    Testing d directly instead of grid-searching it alongside every other
    order is also why it's cheap: one test per candidate d, not a model fit.
    """
    w = list(y)
    for d in range(max_d + 1):
        if len(w) < 8:
            return d
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_value = kpss(w, regression="c", nlags="auto")[1]
            if p_value >= 0.05:
                return d
        except Exception:
            return d
        w = np.diff(w).tolist()
    return max_d


def seasonal_diff(y, s: int) -> list[float]:
    """Lag-s seasonal difference y[t] - y[t-s] (not np.diff(y, s), which is the
    s-th order regular difference)."""
    arr = np.asarray(y, dtype=float)
    return (arr[s:] - arr[:-s]).tolist()


def identify_seasonal_d(y: list[float], d: int, s: int) -> int:
    """Seasonal-differencing identification: regular-difference the series d
    times, then check its autocorrelation at the seasonal lag s. A strong
    positive autocorrelation there (same threshold the PHP version used)
    means the seasonal pattern isn't settling back to a stable level on its
    own and needs a seasonal difference."""
    w = y
    for _ in range(d):
        w = np.diff(w).tolist()
    n = len(w)
    if n < 2 * s:
        return 0
    arr = np.array(w)
    mean = arr.mean()
    num = np.sum((arr[s:] - mean) * (arr[:-s] - mean))
    den = np.sum((arr - mean) ** 2)
    acf_s = num / den if den > 0 else 0.0
    return 1 if acf_s > 0.3 else 0


# Wall-clock budget per search call (SARIMA and the ARIMA baseline each get
# their own), leaving headroom within Laravel's ~24s timeout for the
# identification step, two hold-out backtest fits, and response/network
# overhead. A free, fractional-CPU host means a fixed candidate count can't
# guarantee a time budget — a time budget can.
GRID_SEARCH_BUDGET_SECONDS = 4.0


def trend_for(d: int, D: int) -> str:
    """With exactly one difference in the model (typically just the seasonal
    one), a constant term is what carries a steady growth trend forward —
    without it the forecast would flatten out. With two or more differences a
    constant would instead imply a runaway quadratic trend, so it's left out
    (the usual auto.arima convention)."""
    return "c" if d + D == 1 else "n"


def search_arima(y: list[float], d: int, p_max: int, q_max: int, D: int = 0, P_max: int = 0, Q_max: int = 0, s: int = 0):
    """Hyndman-Khandakar search (Hyndman & Khandakar, 2008, "Automatic Time
    Series Forecasting: The forecast Package for R" — the same algorithm
    behind R's forecast::auto.arima() and Python's pmdarima.auto_arima()):
    fit a small set of structurally different seed models, then greedily
    step through neighbors (p, q, P, Q each +/-1) while AICc keeps
    improving. Passing P_max=Q_max=0 (D=0, s=0) runs this as a plain
    non-seasonal ARIMA search — used for FR21's baseline comparison.

    Reaches a good model in a handful of fits instead of an exhaustive grid
    — important on a free, fractional-CPU host where every fit is expensive
    relative to the request's time budget.
    """
    n = len(y)
    deadline = time.monotonic() + GRID_SEARCH_BUDGET_SECONDS
    tried: dict[tuple[int, int, int, int], object] = {}

    def try_fit(p: int, q: int, P: int, Q: int):
        p, q = min(max(p, 0), p_max), min(max(q, 0), q_max)
        P, Q = min(max(P, 0), P_max), min(max(Q, 0), Q_max)
        key = (p, q, P, Q)
        if key in tried:
            return tried[key]

        order = (p, d, q)
        seasonal_order = (P, D, Q, s)
        k = p + q + P + Q
        n_eff = n - d - D * (s or 1)
        if n_eff < 10 or n_eff < (k + 2) * 3:
            tried[key] = None
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend_for(d, D),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    # Solve the noise variance analytically instead of searching
                    # for it: one fewer parameter, so each fit converges in far
                    # fewer iterations — keeps the slow free host inside the
                    # website's ~27s request budget.
                    concentrate_scale=True,
                ).fit(disp=False, maxiter=35)
            aicc = res.aicc
            result = (order, seasonal_order, res, aicc) if np.isfinite(aicc) else None
        except Exception:
            result = None

        tried[key] = result
        return result

    # Seed models: a rich default, white noise, AR-only, and MA-only — the
    # same four Hyndman-Khandakar starts with, covering structurally
    # different shapes so the stepwise search below starts from whichever
    # shape this series actually favors, not just "simplest first".
    seeds = [
        (min(2, p_max), min(2, q_max), min(1, P_max), min(1, Q_max)),
        (0, 0, 0, 0),
        (min(1, p_max), 0, min(1, P_max), 0),
        (0, min(1, q_max), 0, min(1, Q_max)),
    ]
    best = None
    for p, q, P, Q in seeds:
        r = try_fit(p, q, P, Q)
        if r and (best is None or r[3] < best[3]):
            best = r
        if time.monotonic() >= deadline:
            break

    # Greedy stepwise refinement from the best seed: try each neighbor one
    # step away, accept the first improving move, then repeat, continue
    # until nothing nearby improves AICc or time runs out.
    if best is not None:
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            p, _, q = best[0]
            P, _, Q, _ = best[1]
            for dp, dq, dP, dQ in [(1, 0, 0, 0), (-1, 0, 0, 0), (0, 1, 0, 0), (0, -1, 0, 0),
                                    (0, 0, 1, 0), (0, 0, -1, 0), (0, 0, 0, 1), (0, 0, 0, -1)]:
                np_, nq, nP, nQ = p + dp, q + dq, P + dP, Q + dQ
                if np_ < 0 or nq < 0 or nP < 0 or nQ < 0:
                    continue
                r = try_fit(np_, nq, nP, nQ)
                if r and r[3] < best[3] - 1e-6:
                    best = r
                    improved = True
                    break
                if time.monotonic() >= deadline:
                    break

    return (best[0], best[1], best[2]) if best else None


def fit_naive(y: list[float]):
    """Last-resort fallback for series too short/degenerate for any order
    search to converge: a naive-drift ARIMA(0,1,0)."""
    try:
        order, seasonal_order = (0, 1, 0), (0, 0, 0, 0)
        res = SARIMAX(y, order=order, seasonal_order=seasonal_order).fit(disp=False)
        return order, seasonal_order, res
    except Exception:
        return None


def backtest_accuracy(y: list[float], order, seasonal_order) -> Accuracy:
    """Hold-out backtest: refit the already-selected order on all but the most
    recent periods, forecast that held-out window, and score against what
    actually happened. 12-month hold-out per FR21 (20% of history, clamped
    to 3-12 periods — the same convention the PHP version used, so results
    stay comparable with history shorter than 60 months)."""
    n = len(y)
    holdout = min(12, max(3, int(n * 0.2)))
    train_size = n - holdout
    if train_size < 4:
        return Accuracy(available=False)

    train, actual = y[:train_size], y[train_size:]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend_for(order[1], seasonal_order[1]),
                enforce_stationarity=False,
                enforce_invertibility=False,
                    concentrate_scale=True,
            ).fit(disp=False, maxiter=35)
        predicted = res.get_forecast(holdout).predicted_mean
        predicted = np.maximum(predicted, 0)
    except Exception:
        return Accuracy(available=False)

    actual = np.array(actual)
    abs_err = np.abs(actual - predicted)
    sq_err = (actual - predicted) ** 2
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None

    return Accuracy(
        available=True,
        mae=round(float(np.mean(abs_err)), 2),
        mse=round(float(np.mean(sq_err)), 2),
        rmse=round(float(np.sqrt(np.mean(sq_err))), 2),
        mape=round(mape, 1) if mape is not None else None,
        holdout=holdout,
        actual=[round(float(v), 1) for v in actual],
        predicted=[round(float(v), 1) for v in predicted],
    )


def stationarity_check(y: list[float], residuals: Optional[np.ndarray]) -> Stationarity:
    """Augmented Dickey-Fuller on the raw series (unit-root test — the
    standard way to check whether differencing was needed) plus a Ljung-Box
    test on the fitted model's residuals (checks the residuals are white
    noise, i.e. the model captured the autocorrelation structure) — the
    textbook Box-Jenkins diagnostic-checking step."""
    adf_stat = adf_p = None
    if len(y) >= 8:
        try:
            adf_stat, adf_p = adfuller(y, autolag="AIC")[:2]
        except Exception:
            pass

    lb_p = None
    if residuals is not None and len(residuals) >= 8:
        try:
            lb = acorr_ljungbox(residuals, lags=[min(10, len(residuals) // 2)], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            pass

    is_stationary = (adf_p is not None and adf_p < 0.05) if adf_p is not None else None

    return Stationarity(
        adf_statistic=round(float(adf_stat), 4) if adf_stat is not None else None,
        adf_pvalue=round(float(adf_p), 4) if adf_p is not None else None,
        ljung_box_pvalue=round(lb_p, 4) if lb_p is not None else None,
        is_stationary=is_stationary,
    )


def select_models(y: list[float], s: int):
    """Box-Jenkins identification + AICc order search for the SARIMA model and
    its non-seasonal ARIMA baseline (FR21). Returns (sarima, baseline), each a
    (order, seasonal_order, fitted results) tuple or None."""
    n = len(y)
    seasonal_ok = s > 1 and n >= (2 * s + 6)
    # Seasonal difference first, then test whether a regular difference is
    # still needed on the seasonally-differenced series (Box-Jenkins; Hyndman &
    # Athanasopoulos, "Forecasting: Principles and Practice", ch. 9). Testing
    # the raw series first lets strong seasonality fool the ADF test into
    # always picking d=1, and d=1 on top of D=1 over-differences: the model
    # then projects one unusual month's year-over-year change into the future.
    D = identify_seasonal_d(y, 0, s) if seasonal_ok else 0
    d = identify_d(seasonal_diff(y, s) if D else y)
    p_max = 2 if n >= 30 else (1 if n >= 16 else 0)
    q_max = p_max
    P_max = 1 if seasonal_ok else 0
    Q_max = P_max
    sp = s if seasonal_ok else 0

    # FR21: evaluate SARIMA against a non-seasonal ARIMA baseline on the same
    # hold-out — same d, same data, baseline just has no seasonal component.
    sarima = search_arima(y, d, p_max, q_max, D, P_max, Q_max, sp) or fit_naive(y)
    baseline = search_arima(y, d, p_max, q_max) or fit_naive(y)

    return sarima, baseline


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(require_api_key)])
def forecast(req: ForecastRequest):
    y = req.values
    n = len(y)
    alpha = 1 - req.confidence / 100

    not_fitted = ForecastResponse(
        fitted=False,
        order=[0, 0, 0],
        seasonal_order=[0, 0, 0, 0],
        order_label="—",
        forecast=[],
        accuracy=Accuracy(available=False),
        stationarity=Stationarity(),
    )

    if n < 4:
        return not_fitted

    sarima, baseline_fit = select_models(y, req.seasonal_period)

    if sarima is None:
        return not_fitted

    order, seasonal_order, res = sarima

    fc = res.get_forecast(req.steps)
    frame = fc.summary_frame(alpha=alpha)
    points = [
        ForecastPoint(
            value=round(max(0.0, float(row["mean"])), 1),
            lower=round(max(0.0, float(row["mean_ci_lower"])), 1),
            upper=round(max(float(row["mean_ci_upper"]), float(row["mean"])), 1),
        )
        for _, row in frame.iterrows()
    ]

    accuracy = backtest_accuracy(y, order, seasonal_order)
    stationarity = stationarity_check(y, res.resid if hasattr(res, "resid") else None)

    baseline = None
    if baseline_fit is not None:
        b_order, b_seasonal_order, _ = baseline_fit
        baseline = Baseline(
            fitted=True,
            order_label=order_label(b_order, b_seasonal_order),
            order=list(b_order),
            accuracy=backtest_accuracy(y, b_order, b_seasonal_order),
        )

    return ForecastResponse(
        fitted=True,
        order=list(order),
        seasonal_order=list(seasonal_order),
        order_label=order_label(order, seasonal_order),
        aicc=round(float(res.aicc), 2) if np.isfinite(res.aicc) else None,
        forecast=points,
        accuracy=accuracy,
        stationarity=stationarity,
        baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Model validation — the Box-Jenkins diagnostic-checking step plus an
# out-of-sample test, reported in full (series, not just summary numbers) so
# the Laravel app can chart every stage: differencing, stationarity tests,
# residual diagnostics, and hold-out errors / accuracy (MAE, RMSE, MAPE, MASE).
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    values: list[float] = Field(..., min_length=1)
    seasonal_period: int = 12
    confidence: int = 95
    # The exact orders the live forecast is using, so the validation describes
    # that model rather than re-selecting one. Searched afresh if omitted.
    order: Optional[list[int]] = None
    seasonal_order: Optional[list[int]] = None
    baseline_order: Optional[list[int]] = None


def num(x, digits: int = 4):
    """JSON-safe rounded float (NaN/inf → None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, digits) if np.isfinite(x) else None


def nums(xs, digits: int = 4):
    return [num(x, digits) for x in xs]


def fit_order(y, order, seasonal_order):
    """Fit one fixed order with the same estimator settings the search and the
    /forecast backtest use, so validation scores the same model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            y,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend=trend_for(order[1], seasonal_order[1]),
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        ).fit(disp=False, maxiter=35)


def adf_test(w):
    """Augmented Dickey-Fuller. H0: unit root (non-stationary); p < 0.05 rejects it."""
    if len(w) < 8:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lags, nobs, crit, _ = adfuller(w, autolag="AIC")
    except Exception:
        return None
    return {
        "statistic": num(stat),
        "p_value": num(p),
        "lags": int(lags),
        "nobs": int(nobs),
        "critical": {k: num(v) for k, v in crit.items()},
        "stationary": bool(p < 0.05),
    }


def kpss_test(w):
    """KPSS — the complementary test. H0: stationary; p < 0.05 rejects it. Its
    p-value is interpolated from a table, so statsmodels caps it to [0.01, 0.10]."""
    if len(w) < 8:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, lags, crit = kpss(w, regression="c", nlags="auto")
    except Exception:
        return None
    return {
        "statistic": num(stat),
        "p_value": num(p),
        "lags": int(lags),
        "critical": {k: num(crit[k]) for k in ("10%", "5%", "1%") if k in crit},
        "stationary": bool(p >= 0.05),
        "p_bounded": bool(p <= 0.01 or p >= 0.1),
    }


def correlogram(w, max_lag: int):
    n = len(w)
    lags = min(max_lag, n // 2 - 1)
    if lags < 1:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = acf(w, nlags=lags, fft=False)
            pa = pacf(w, nlags=lags, method="ywm")
    except Exception:
        return None
    return {
        "lags": list(range(1, lags + 1)),
        "acf": nums(a[1:]),
        "pacf": nums(pa[1:]),
        # Approximate 95% significance bound for a white-noise series.
        "bound": num(1.96 / np.sqrt(n)),
    }


def rolling_stats(w, window: int):
    arr = np.asarray(w, dtype=float)
    means, stds = [], []
    for i in range(len(arr)):
        if i + 1 < window:
            means.append(None)
            stds.append(None)
            continue
        seg = arr[i + 1 - window:i + 1]
        means.append(num(seg.mean(), 2))
        stds.append(num(seg.std(ddof=1), 2))
    return {"window": window, "mean": means, "std": stds}


def describe_stage(key, label, offset, w, s):
    window = s if s > 1 and len(w) >= 2 * s else max(3, len(w) // 6)
    return {
        "key": key,
        "label": label,
        "offset": offset,
        "values": nums(w, 2),
        "adf": adf_test(w),
        "kpss": kpss_test(w),
        "rolling": rolling_stats(w, window),
        "correlogram": correlogram(w, max(2 * s, 12) if s > 1 else 12),
    }


def error_metrics(actual, predicted, train, m: int):
    """MAE, MSE, RMSE, MAPE and MASE on the hold-out. MASE scales MAE by the
    in-sample MAE of the (seasonal) naive forecast on the training data
    (Hyndman & Koehler, 2006): below 1 means better than naive."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    e = a - p
    nonzero = a != 0
    tr = np.asarray(train, dtype=float)
    scale = float(np.mean(np.abs(tr[m:] - tr[:-m]))) if len(tr) > m else None
    mae = float(np.mean(np.abs(e)))
    return {
        "mae": num(mae, 2),
        "mse": num(np.mean(e ** 2), 2),
        "rmse": num(np.sqrt(np.mean(e ** 2)), 2),
        "mape": num(np.mean(np.abs(e[nonzero] / a[nonzero])) * 100, 2) if nonzero.any() else None,
        "mape_excluded": int((~nonzero).sum()),
        "mase": num(mae / scale, 3) if scale else None,
        "mean_error": num(np.mean(e), 2),
    }


@app.post("/validate", dependencies=[Depends(require_api_key)])
def validate(req: ValidateRequest):
    y = [float(v) for v in req.values]
    n = len(y)
    if n < 8:
        return {"fitted": False, "reason": "Not enough history to validate (need at least 8 periods)."}

    alpha = 1 - req.confidence / 100

    if req.order and req.seasonal_order:
        order, seasonal_order = tuple(req.order), tuple(req.seasonal_order)
        b_order = tuple(req.baseline_order) if req.baseline_order else None
    else:
        sarima, baseline_fit = select_models(y, req.seasonal_period)
        if sarima is None:
            return {"fitted": False, "reason": "No model could be fitted to this series."}
        order, seasonal_order = sarima[0], sarima[1]
        b_order = baseline_fit[0] if baseline_fit else None
    if b_order is None:
        baseline_fit = search_arima(y, order[1], 2, 2) or fit_naive(y)
        b_order = baseline_fit[0] if baseline_fit else (0, 1, 0)

    d, D, s = order[1], seasonal_order[1], seasonal_order[3]
    s_data = s if s > 1 else req.seasonal_period

    # 1. Differencing — every transformation the model applies, in order.
    stages = [describe_stage("original", "Original series", 0, y, s_data)]
    w, offset = np.asarray(y, dtype=float), 0
    if D and s > 1:
        w = w[s:] - w[:-s]
        offset += s
        stages.append(describe_stage("seasonal", f"Seasonal difference (lag {s})", offset, w, s_data))
    for i in range(d):
        w = np.diff(w)
        offset += 1
        stages.append(describe_stage(f"diff{i + 1}", "First difference" if i == 0 else "Second difference", offset, w, s_data))

    # 2. Residual diagnostics on the full-history fit.
    try:
        res_full = fit_order(y, order, seasonal_order)
    except Exception:
        return {"fitted": False, "reason": "The selected model failed to fit the full series."}

    burn = d + D * s  # the first residuals are dominated by the differencing start-up
    resid = np.asarray(res_full.resid, dtype=float)[burn:]
    fitted_vals = np.asarray(y, dtype=float)[burn:] - resid
    k = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    max_lb = min(2 * s if s > 1 else 10, len(resid) // 2)
    ljung_box = []
    if max_lb > k:
        try:
            lb = acorr_ljungbox(resid, lags=list(range(k + 1, max_lb + 1)), model_df=k, return_df=True)
            ljung_box = [{"lag": int(lag), "statistic": num(row["lb_stat"]), "p_value": num(row["lb_pvalue"])} for lag, row in lb.iterrows()]
        except Exception:
            ljung_box = []
    jb_stat, jb_p = sps.jarque_bera(resid) if len(resid) >= 8 else (None, None)

    residuals = {
        "offset": burn,
        "values": nums(resid, 2),
        "fitted": nums(fitted_vals, 2),
        "mean": num(np.mean(resid), 3),
        "std": num(np.std(resid, ddof=1), 3),
        "skewness": num(sps.skew(resid), 3),
        "excess_kurtosis": num(sps.kurtosis(resid), 3),
        "jarque_bera": {"statistic": num(jb_stat), "p_value": num(jb_p)},
        "ljung_box": ljung_box,
        "ljung_box_model_df": k,
        "white_noise": bool(ljung_box[-1]["p_value"] > 0.05) if ljung_box and ljung_box[-1]["p_value"] is not None else None,
        "correlogram": correlogram(resid, max(2 * s, 12) if s > 1 else 12),
    }

    # 3. Out-of-sample test: refit on the training window, forecast the hold-out
    #    (same 20%-clamped-to-3..12 window as the /forecast backtest).
    holdout = min(12, max(3, int(n * 0.2)))
    train, actual = y[:n - holdout], y[n - holdout:]
    test = {"available": False, "holdout": holdout, "train_size": n - holdout}
    if len(train) >= 4:
        try:
            frame = fit_order(train, order, seasonal_order).get_forecast(holdout).summary_frame(alpha=alpha)
            pred = np.maximum(frame["mean"].to_numpy(), 0)
            lower = np.maximum(frame["mean_ci_lower"].to_numpy(), 0)
            upper = np.maximum(frame["mean_ci_upper"].to_numpy(), pred)

            try:
                b_pred = np.maximum(fit_order(train, b_order, (0, 0, 0, 0)).get_forecast(holdout).predicted_mean, 0)
            except Exception:
                b_pred = None

            m = s_data if s_data > 1 and len(train) > s_data else 1
            naive = np.array([train[len(train) - m + (h % m)] for h in range(holdout)], dtype=float)

            test.update({
                "available": True,
                "mase_period": m,
                "points": [{
                    "actual": num(a, 2),
                    "predicted": num(p, 2),
                    "lower": num(lo, 2),
                    "upper": num(hi, 2),
                    "error": num(a - p, 2),
                    "pct_error": num((a - p) / a * 100, 2) if a != 0 else None,
                    "baseline": num(b_pred[i], 2) if b_pred is not None else None,
                    "naive": num(naive[i], 2),
                } for i, (a, p, lo, hi) in enumerate(zip(actual, pred, lower, upper))],
                "metrics": {
                    "sarima": error_metrics(actual, pred, train, m),
                    "baseline": error_metrics(actual, b_pred, train, m) if b_pred is not None else None,
                    "naive": error_metrics(actual, naive, train, m),
                },
            })
        except Exception:
            pass

    return {
        "fitted": True,
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "order_label": order_label(order, seasonal_order),
        "baseline_order_label": order_label(b_order, (0, 0, 0, 0)),
        "aicc": num(res_full.aicc, 2),
        "n": n,
        "seasonal_period": s_data,
        "confidence": req.confidence,
        "differencing": {"d": d, "D": D, "s": s, "stages": stages},
        "residuals": residuals,
        "test": test,
    }
