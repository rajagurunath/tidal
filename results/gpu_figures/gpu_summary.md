| condition | node | online p50 | online p99 | vs baseline | batch otok/s | % of ceiling | salvaged |
|---|---|---|---|---|---|---|---|
| online_only | gpu-f0251bd73e | 0.753s | 1.570s | 1.00x / 1.00x | 0 | — | False |
| offline_only | — | — | — | — | — | — | missing |
| naive | — | — | — | — | — | — | missing |
| technique_a | gpu-076fa96ed5 | 0.886s | 1.927s | 1.18x / 1.23x | 1408 | 98.7% | False |
| technique_b | — | — | — | — | — | — | missing |

Probe ceilings by node: {"gpu-18c425a48ee1-20260809T081901Z": 1467.6, "gpu-076fa96ed554-20260809T050514Z": 1426.7, "gpu-f0251bd73ed2-20260809T031215Z": 1438.3}