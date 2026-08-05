"""All Tidal configuration in one place. Every field is overridable via
environment variables prefixed TIDAL_ (pydantic-settings style, hand-rolled
to avoid the extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class TidalConfig:
    # -- wiring --
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_metrics_url: str = "http://127.0.0.1:8000/metrics"
    dsn: str = "sqlite:///tidal.db"
    blob_dir: str = "tidal_blobs"
    api_key: str = "tidal-dev-key"          # static bearer token for the Batch API
    host: str = "127.0.0.1"
    port: int = 8080

    # -- batch semantics --
    completion_window_hours: int = 24
    max_item_attempts: int = 3
    served_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # -- dispatcher (technique A: external control) --
    poll_interval_s: float = 1.0
    ewma_alpha: float = 0.5
    kv_high: float = 0.85
    kv_low: float = 0.70
    max_inflight: int = 4                    # 64 on GPU; 4 for Mac CPU dev
    online_waiting_tolerance: int = 0

    # -- laxity escalation (both techniques) --
    batch_priority_max: int = 100            # fully safe
    batch_priority_min: int = 1              # most urgent, still below online (0)
    sla_strict: bool = False                 # True → urgency 1.0 may reach priority 0
    floor_urgency: float = 0.9               # token-bucket floor kicks in above this
    floor_items_per_s: float = 0.2

    # -- engine (technique B: TidalScheduler) --
    tbt_slo_ms: float = 200.0
    interference_guard: float = 0.2          # fraction of TBT SLO held back
    cold_start_batch_frac: float = 0.25      # static cap until T̂ fit converges
    fit_window_steps: int = 4000
    fit_r2_gate: float = 0.85
    kv_guardband_min: float = 0.05
    kv_guardband_max: float = 0.30

    # -- metering --
    batch_discount: float = 0.5

    @classmethod
    def from_env(cls) -> "TidalConfig":
        kwargs = {}
        for f in fields(cls):
            env = os.environ.get(f"TIDAL_{f.name.upper()}")
            if env is not None:
                typ = type(getattr(cls, f.name, f.default))
                kwargs[f.name] = (env.lower() in ("1", "true", "yes")) if typ is bool else typ(env)
        return cls(**kwargs)
