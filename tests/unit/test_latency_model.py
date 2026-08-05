"""Self-calibrating step-latency model: pure numpy, fully deterministic.

The synthetic ground truth is the ConServe context-aware form used in the spec
(§5.2): T(P, C) = k1*P + k2*P*(P+C) + k4*(P+C) + k5.
"""

from __future__ import annotations

import numpy as np
import pytest

from tidal.engine.latency_model import NotReadyError, StepLatencyModel

# Ground-truth coefficients used to synthesise observations.
K1, K2, K4, K5 = 0.01, 1e-6, 2e-4, 5.0


def true_t(p: int, c: int) -> float:
    """The exact formula used to synthesise every observation below."""
    return K1 * p + K2 * p * (p + c) + K4 * (p + c) + K5


def _train(
    m: StepLatencyModel,
    n: int = 2000,
    seed: int = 0,
    sigma: float = 0.1,
) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        m.observe(p, c, true_t(p, c) + rng.normal(0, sigma))


def _fitted_model(window: int = 4000, r2_gate: float = 0.85, **kw) -> StepLatencyModel:
    m = StepLatencyModel(window=window, r2_gate=r2_gate)
    _train(m, **kw)
    assert m.ready()
    return m


# --------------------------------------------------------------------------
# Plan-named tests
# --------------------------------------------------------------------------


def test_recovers_synthetic_coefficients():
    m = StepLatencyModel(window=4000, r2_gate=0.85)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        t = 0.01 * p + 1e-6 * p * (p + c) + 2e-4 * (p + c) + 5 + rng.normal(0, 0.1)
        m.observe(p, c, t)
    assert m.ready()
    truth = true_t(512, 1024)
    assert abs(m.predict_ms(512, 1024) - truth) / truth < 0.05


def test_not_ready_before_min_samples_or_bad_r2():
    # (a) too few samples -> no fit at all
    cold = StepLatencyModel(window=4000, r2_gate=0.85)
    assert not cold.ready()
    _train(cold, n=100)
    assert not cold.ready()

    # (b) plenty of samples but the signal is pure noise -> R^2 below the gate
    noisy = StepLatencyModel(window=4000, r2_gate=0.85)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        noisy.observe(p, c, float(abs(rng.normal(50, 20))))
    assert not noisy.ready()


def test_max_batch_tokens_inverse_of_predict():
    m = _fitted_model()
    p_on, c_total = 256, 4096
    budget = m.predict_ms(p_on, c_total) + 12.0
    x = m.max_batch_tokens(p_on, c_total, budget)
    assert x > 0
    assert m.predict_ms(p_on + x, c_total) <= budget + 1e-9
    assert m.predict_ms(p_on + x + 1, c_total) > budget
    assert m.predict_ms(p_on + x + 64, c_total) > budget


def test_max_batch_tokens_zero_when_online_alone_exceeds_budget():
    m = _fitted_model()
    p_on, c_total = 2048, 8192
    budget = m.predict_ms(p_on, c_total) * 0.5
    assert m.max_batch_tokens(p_on, c_total, budget) == 0


def test_mape_reflects_noise():
    clean = _fitted_model(seed=2, sigma=0.05)
    dirty = StepLatencyModel(window=4000, r2_gate=0.85)
    _train(dirty, seed=2, sigma=5.0)  # too noisy to pass the gate; mape still defined
    assert 0.0 <= clean.mape() < 0.02
    assert dirty.mape() > clean.mape() * 3


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_predict_and_max_batch_tokens_raise_before_convergence():
    m = StepLatencyModel(window=4000, r2_gate=0.85)
    with pytest.raises(NotReadyError):
        m.predict_ms(128, 512)
    with pytest.raises(NotReadyError):
        m.max_batch_tokens(128, 512, 100.0)
    _train(m, n=100)
    with pytest.raises(NotReadyError):
        m.max_batch_tokens(128, 512, 100.0)
    assert np.isnan(m.mape())


def test_max_batch_tokens_raises_when_fit_below_r2_gate():
    """A fit exists (predict works) but the gate is not met -> still NotReady."""
    m = StepLatencyModel(window=4000, r2_gate=0.999999)
    _train(m, sigma=8.0)
    assert not m.ready()
    m.predict_ms(128, 512)  # a fit exists, prediction is allowed
    with pytest.raises(NotReadyError):
        m.max_batch_tokens(128, 512, 100.0)


def test_coefficients_clamped_non_negative_under_adversarial_noise():
    """Data with a negative p-slope must not yield negative coefficients."""
    m = StepLatencyModel(window=4000, r2_gate=0.85)
    rng = np.random.default_rng(3)
    for _ in range(1500):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        # deliberately decreasing in p and in (p + c)
        m.observe(p, c, 100.0 - 0.02 * p - 1e-3 * (p + c) + rng.normal(0, 0.5))
    coefs = m.coefficients()
    assert len(coefs) == 4
    assert all(k >= 0.0 for k in coefs)
    assert m.predict_ms(512, 1024) >= 0.0


def test_max_batch_tokens_monotone_decreasing_in_p_on_and_c_total():
    m = _fitted_model()
    budget = 60.0
    by_p = [m.max_batch_tokens(p_on, 4096, budget) for p_on in (0, 128, 256, 512, 1024)]
    assert by_p == sorted(by_p, reverse=True)
    assert by_p[0] > by_p[-1]
    by_c = [m.max_batch_tokens(256, c, budget) for c in (0, 1024, 8192, 32768, 131072)]
    assert by_c == sorted(by_c, reverse=True)
    assert by_c[0] > by_c[-1]


def test_max_batch_tokens_never_negative_and_is_int():
    m = _fitted_model()
    x = m.max_batch_tokens(0, 0, m.predict_ms(0, 0) - 1.0)
    assert isinstance(x, int)
    assert x == 0


def test_ring_buffer_forgets_old_regime():
    """Window-sized eviction: a regime change is fully absorbed."""
    m = StepLatencyModel(window=600, r2_gate=0.85)
    rng = np.random.default_rng(4)
    for _ in range(600):  # regime 1: 4x slower
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        m.observe(p, c, 4.0 * true_t(p, c))
    # regime 2 overwrites the whole buffer, then runs on to the next refit
    for _ in range(900):
        p, c = int(rng.integers(1, 2048)), int(rng.integers(0, 8192))
        m.observe(p, c, true_t(p, c))
    assert m.ready()
    truth = true_t(512, 1024)
    assert abs(m.predict_ms(512, 1024) - truth) / truth < 0.05


def test_observe_is_deterministic_and_order_independent_of_wallclock():
    a = StepLatencyModel(window=4000, r2_gate=0.85)
    b = StepLatencyModel(window=4000, r2_gate=0.85)
    _train(a, seed=7)
    _train(b, seed=7)
    assert a.coefficients() == b.coefficients()
    assert a.predict_ms(333, 777) == b.predict_ms(333, 777)
    assert a.max_batch_tokens(64, 512, 50.0) == b.max_batch_tokens(64, 512, 50.0)


def test_invalid_construction_and_observations_rejected():
    with pytest.raises(ValueError):
        StepLatencyModel(window=0, r2_gate=0.85)
    with pytest.raises(ValueError):
        StepLatencyModel(window=100, r2_gate=1.5)
    m = StepLatencyModel(window=100, r2_gate=0.85)
    with pytest.raises(ValueError):
        m.observe(-1, 0, 10.0)
    with pytest.raises(ValueError):
        m.observe(1, -5, 10.0)
    with pytest.raises(ValueError):
        m.observe(1, 0, float("nan"))
    with pytest.raises(ValueError):
        m.max_batch_tokens(-1, 0, 10.0)
