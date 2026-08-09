| condition | node | online p50 | online p99 | vs baseline (cross-node) | batch otok/s | % of steady-state ceiling | salvaged |
|---|---|---|---|---|---|---|---|
| online_only | gpu-f0251bd73e | 0.753s | 1.570s | 1.00x / 1.00x | 0 | — | False |
| offline_only | — | — | — | — | — | — | missing |
| naive | — | — | — | — | — | — | missing |
| technique_a | gpu-076fa96ed5 | 0.886s | 1.927s | 1.18x / 1.23x | 1408 | 69.3% | False |
| technique_b | — | — | — | — | — | — | missing |

Steady-state ceilings by node (last 10s of the probe burst, otok/s): {"gpu-f0251bd73ed2-20260809T031215Z": 2049.6, "gpu-076fa96ed554-20260809T050514Z": 2031.5, "gpu-18c425a48ee1-20260809T081901Z": 2057.0}

Whole-burst probe averages by node (RAMP-DOMINATED -- reported, not used as a denominator): {"gpu-f0251bd73ed2-20260809T031215Z": 1438.3, "gpu-076fa96ed554-20260809T050514Z": 1426.7, "gpu-18c425a48ee1-20260809T081901Z": 1467.6}

Latency `vs baseline` ratios are CROSS-NODE (no within-node online_only/technique_a pair exists); throughput is normalized per-node.

Cross-node steady-state band: +-0.62%

Cross-node whole-burst band: +-1.41%