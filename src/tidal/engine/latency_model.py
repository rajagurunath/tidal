"""Self-calibrating step-latency model (spec §5.2).

ConServe's context-aware interference form, fitted online from live traffic:

    T(P, C) = k1*P + k2*P*(P + C) + k4*(P + C) + k5

where ``P`` is the number of tokens computed in a step and ``C`` the resident
KV context (in tokens). Feature vector ``[P, P*(P+C), (P+C), 1]``, coefficients
``[k1, k2, k4, k5]`` recovered by ordinary least squares over a ring buffer of
the last ``window`` observations, clamped to be non-negative (latency is
monotone non-decreasing in both work and context; a negative coefficient is
always noise, never physics).

Everything here is deterministic and clock-free: the caller supplies the
measured step time. Refits happen on an observation counter, never on a timer.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["NotReadyError", "StepLatencyModel"]

#: Observations required before the first least-squares fit is attempted.
MIN_SAMPLES = 512
#: Refit cadence, in observations (a 4-feature lstsq is microseconds).
REFIT_INTERVAL = 256
#: Returned by :meth:`StepLatencyModel.max_batch_tokens` when the fit is
#: degenerate (every coefficient except the intercept clamped to zero), i.e.
#: the model believes latency does not depend on the batch at all. The caller
#: always takes ``min(X, slack)``, so this reads as "no model-imposed cap".
UNBOUNDED_TOKENS = 1 << 30


class NotReadyError(RuntimeError):
    """Raised when a prediction is requested before the fit has converged.

    Callers handle this as the cold-start path (static ``cold_start_batch_frac``
    cap) rather than trusting an unconverged model.
    """


class StepLatencyModel:
    """Rolling least-squares fit of step latency against (P, C).

    Args:
        window: ring-buffer size; the fit only ever sees the last ``window``
            observations, so regime changes are forgotten within one window.
        r2_gate: minimum in-window R^2 for :meth:`ready` to return True.
    """

    def __init__(self, window: int, r2_gate: float) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 0.0 <= r2_gate <= 1.0:
            raise ValueError(f"r2_gate must be in [0, 1], got {r2_gate}")
        self.window = int(window)
        self.r2_gate = float(r2_gate)

        self._x = np.zeros((self.window, 4), dtype=np.float64)
        self._y = np.zeros(self.window, dtype=np.float64)
        self._head = 0  # next slot to write
        self._count = 0  # observations currently in the buffer
        self._total = 0  # observations ever seen
        self._since_fit = 0  # observations since the last refit
        self._coefs: np.ndarray | None = None

    # -- ingestion ---------------------------------------------------------

    def observe(self, p: int, c: int, step_ms: float) -> None:
        """Record one step: ``p`` tokens computed against ``c`` resident context."""
        if p < 0 or c < 0:
            raise ValueError(f"p and c must be >= 0, got p={p}, c={c}")
        step_ms = float(step_ms)
        if not math.isfinite(step_ms):
            raise ValueError(f"step_ms must be finite, got {step_ms}")

        i = self._head
        total = float(p) + float(c)
        self._x[i, 0] = float(p)
        self._x[i, 1] = float(p) * total
        self._x[i, 2] = total
        self._x[i, 3] = 1.0
        self._y[i] = step_ms

        self._head = (i + 1) % self.window
        self._count = min(self._count + 1, self.window)
        self._total += 1
        self._since_fit += 1

        if self._total >= MIN_SAMPLES and self._since_fit >= REFIT_INTERVAL:
            self._fit()

    # -- fit state ---------------------------------------------------------

    def ready(self) -> bool:
        """True once a fit exists and explains the window at R^2 >= gate."""
        if self._coefs is None:
            if self._total < MIN_SAMPLES:
                return False
            self._fit()  # first check after enough samples forces a fit
            if self._coefs is None:
                return False
        return self.r2() >= self.r2_gate

    def coefficients(self) -> tuple[float, float, float, float]:
        """``(k1, k2, k4, k5)`` of the current fit; all zeros before any fit."""
        if self._coefs is None:
            return (0.0, 0.0, 0.0, 0.0)
        k1, k2, k4, k5 = (float(v) for v in self._coefs)
        return (k1, k2, k4, k5)

    def r2(self) -> float:
        """Coefficient of determination of the current fit over the window."""
        if self._coefs is None or self._count == 0:
            return 0.0
        y = self._y[: self._count]
        resid = y - self._design() @ self._coefs
        ss_res = float(resid @ resid)
        centred = y - y.mean()
        ss_tot = float(centred @ centred)
        if ss_tot <= 1e-12:
            return 1.0 if ss_res <= 1e-12 else 0.0
        return 1.0 - ss_res / ss_tot

    def mape(self) -> float:
        """Rolling mean absolute percentage error, as a fraction (0.05 == 5%).

        NaN until a fit exists.
        """
        if self._coefs is None or self._count == 0:
            return float("nan")
        y = self._y[: self._count]
        pred = self._design() @ self._coefs
        denom = np.where(np.abs(y) < 1e-9, 1e-9, np.abs(y))
        return float(np.mean(np.abs(pred - y) / denom))

    # -- prediction --------------------------------------------------------

    def predict_ms(self, p: int, c: int) -> float:
        """T̂(p, c) in milliseconds. Requires a fit (not necessarily the gate)."""
        if self._coefs is None:
            raise NotReadyError(f"no fit yet ({self._total}/{MIN_SAMPLES} observations)")
        k1, k2, k4, k5 = self.coefficients()
        total = float(p) + float(c)
        return k1 * float(p) + k2 * float(p) * total + k4 * total + k5

    def max_batch_tokens(self, p_on: int, c_total: int, budget_ms: float) -> int:
        """Largest ``x >= 0`` with ``predict_ms(p_on + x, c_total) <= budget_ms``.

        Closed form. Writing ``s = p_on + c_total`` and substituting
        ``P = p_on + x``, ``C = c_total`` into T̂ gives a quadratic in x::

            T(x) = k2*x^2 + B*x + T0
              B  = k1 + k2*(p_on + s) + k4
              T0 = predict_ms(p_on, c_total)          (= T(0))

        so the constraint is ``k2*x^2 + B*x - D <= 0`` with ``D = budget - T0``.
        With all coefficients clamped >= 0 the parabola opens upwards and the
        admissible set is ``[0, x*]`` where x* is the positive root::

            x* = (-B + sqrt(B^2 + 4*k2*D)) / (2*k2)
               = 2*D / (B + sqrt(B^2 + 4*k2*D))       (numerically stable form)

        The second form is used: it is exact as ``k2 -> 0`` (degenerating to the
        linear solution ``D / B``) and avoids catastrophic cancellation for the
        small ``k2`` values a real fit produces. Result is floored to an int and
        clamped at 0.

        Raises:
            NotReadyError: if :meth:`ready` is False (caller applies the
                cold-start cap instead).
        """
        if p_on < 0 or c_total < 0:
            raise ValueError(f"p_on and c_total must be >= 0, got {p_on}, {c_total}")
        if not self.ready():
            raise NotReadyError(
                f"fit not converged (R^2={self.r2():.3f} < gate={self.r2_gate}, "
                f"{self._total} observations)"
            )

        k1, k2, k4, _ = self.coefficients()
        t0 = self.predict_ms(p_on, c_total)
        d = float(budget_ms) - t0
        if d <= 0.0:
            return 0  # online alone already blows the budget
        s = float(p_on) + float(c_total)
        b = k1 + k2 * (float(p_on) + s) + k4
        if b <= 0.0 and k2 <= 0.0:
            return UNBOUNDED_TOKENS  # degenerate fit: latency independent of x
        x = 2.0 * d / (b + math.sqrt(b * b + 4.0 * k2 * d))
        if not math.isfinite(x) or x <= 0.0:
            return 0
        return int(min(x, float(UNBOUNDED_TOKENS)))

    # -- internals ---------------------------------------------------------

    def _design(self) -> np.ndarray:
        """The populated rows of the ring buffer (order is irrelevant to OLS)."""
        return self._x[: self._count]

    def _fit(self) -> None:
        self._since_fit = 0
        if self._count < 4:
            return
        try:
            coefs, *_ = np.linalg.lstsq(self._design(), self._y[: self._count], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - lstsq is very robust
            return
        if not np.all(np.isfinite(coefs)):  # pragma: no cover
            return
        self._coefs = np.maximum(coefs, 0.0)  # latency is monotone in P and C
