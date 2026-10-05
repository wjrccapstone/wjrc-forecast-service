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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
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
    Augmented Dickey-Fuller unit-root test on the series, differencing and
    retesting until it rejects the unit-root null (p < 0.05) or max_d is
    reached. This is the textbook way to pick d — testing it directly,
    instead of grid-searching it alongside every other order — which is
    also why it's cheap: one ADF test per candidate d, not a full model fit.
    """
    w = list(y)
    for d in range(max_d + 1):
        if len(w) < 8:
            return d
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_value = adfuller(w, autolag="AIC")[:2]
            if p_value < 0.05:
                return d
        except Exception:
            return d
        w = np.diff(w).tolist()
    return max_d


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

    s = req.seasonal_period
    seasonal_ok = s > 1 and n >= (2 * s + 6)
    # Seasonal difference first, then test whether a regular difference is
    # still needed on the seasonally-differenced series (Box-Jenkins; Hyndman &
    # Athanasopoulos, "Forecasting: Principles and Practice", ch. 9). Testing
    # the raw series first lets strong seasonality fool the ADF test into
    # always picking d=1, and d=1 on top of D=1 over-differences: the model
    # then projects one unusual month's year-over-year change into the future.
    D = identify_seasonal_d(y, 0, s) if seasonal_ok else 0
    d = identify_d(np.diff(y, s).tolist() if D else y)
    p_max = 2 if n >= 30 else (1 if n >= 16 else 0)
    q_max = p_max
    P_max = 1 if seasonal_ok else 0
    Q_max = P_max
    sp = s if seasonal_ok else 0

    # FR21: evaluate SARIMA against a non-seasonal ARIMA baseline on the same
    # hold-out — same d, same data, baseline just has no seasonal component.
    sarima = search_arima(y, d, p_max, q_max, D, P_max, Q_max, sp) or fit_naive(y)
    baseline_fit = search_arima(y, d, p_max, q_max) or fit_naive(y)

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
