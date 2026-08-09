#!/usr/bin/env python3
"""Aggregate GPU evidence across hires into the paper's comparison table + charts.

Usage: .venv/bin/python scripts/gpu_analysis.py evidence/gpu-*/  -o results/gpu_figures

Collects every matrix/*.json and probe/*.json under the given evidence dirs
(extracted tarballs), dedupes by (condition, started_at) keeping the newest,
flags salvaged runs (drain_timed_out), normalizes cross-node comparisons by
each run's own probe ceiling, and emits:
  - gpu_summary.md          (the paper table, markdown)
  - gpu_latency_cdf.png     (online request latency CDFs per condition)
  - gpu_throughput.png      (batch otok/s as % of same-node ceiling)
  - gpu_diurnal_tide.png    (per-minute batch completions vs online arrivals,
                             if a diurnal run is present)

Two methodology corrections (independent audit, 2026-08-09):

1. The ceiling is the probe's STEADY-STATE TAIL, not its whole-burst average.
   The probe drains a 600-item pool at concurrency 512 in an ~18.5 s makespan,
   so roughly half of that window is pipeline fill: the whole-burst average
   (1438.3 / 1426.7 / 1467.6 otok/s) is ramp-dominated and is not a ceiling.
   The tell was that a co-serving run "exceeded" it -- any figure above 100%
   means the denominator is not a ceiling. We therefore take the last
   TAIL_S = 10 s of the burst (tokens completed in that window / TAIL_S),
   which yields 2049.6 / 2031.5 / 2057.0 otok/s: a tighter cross-node band
   (+-0.62% vs +-1.4%), i.e. a better instrument as well as a correct one.

2. The diurnal correlation excludes bin 0. Minute 0 is pipeline fill (771
   batch completions versus 1667-1894 in every later minute); including it
   drags corr(online/min, batch/min) to -0.38, which is a startup artifact,
   not a weak tide. Excluded, the correlation is -0.85 -- the GPU tide agrees
   with the CPU tide (-0.95). Both figures are reported.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAIL_S = 10.0  # steady-state window at the end of the probe burst

ACCENT = "#1B7878"
BLUE = "#3B6FB6"
ORANGE = "#D9822B"
PINK = "#C86A87"
GREEN = "#2E8B57"

COND_STYLE = {
    "online_only": (BLUE, "online only (baseline)"),
    "naive": (GREEN, "naive (priority field only)"),
    "technique_a": (ORANGE, "technique A (gateway)"),
    "technique_b": (PINK, "technique B (in-engine)"),
}


def steady_state_ceiling(d: dict) -> float | None:
    """Output tok/s over the last TAIL_S seconds of the probe's burst.

    The whole-burst average is ramp-dominated (see module docstring); the tail
    is the rate the node sustains once the pipeline is full, which is what a
    co-serving run is actually competing against.
    """
    makespan = (d.get("batch") or {}).get("makespan_s")
    comps = [c for c in d.get("batch_completions") or [] if c.get("error") is None]
    if not makespan or not comps:
        return None
    lo = makespan - TAIL_S
    tail = [c for c in comps if lo <= (c.get("finished_at") or -1) <= makespan]
    if not tail:
        return None
    return sum(c.get("completion_tokens") or 0 for c in tail) / TAIL_S


def load_runs(evidence_dirs: list[str]) -> dict:
    runs: dict[str, dict] = {}
    probes: dict[str, float] = {}  # run-dir prefix -> steady-state ceiling otok/s
    burst: dict[str, float] = {}  # run-dir prefix -> whole-burst avg (NOT a ceiling)
    for ev in evidence_dirs:
        for f in glob.glob(f"{ev}/**/probe/*.json", recursive=True):
            d = json.load(open(f))
            node = Path(f).parts[-3]
            otps = (d.get("batch") or {}).get("output_tokens_per_s")
            if otps:
                burst[node] = otps
            tail = steady_state_ceiling(d)
            if tail:
                probes[node] = tail
        for f in glob.glob(f"{ev}/**/matrix/*.json", recursive=True):
            d = json.load(open(f))
            cond = d.get("condition") or Path(f).stem
            node = Path(f).parts[-3]
            d["_node"] = node
            d["_ceiling"] = probes.get(node)
            prev = runs.get(cond)
            if prev is None or (d.get("started_at") or "") > (prev.get("started_at") or ""):
                runs[cond] = d
    return {"runs": runs, "probes": probes, "burst": burst}


def summarize(data: dict, out: Path) -> str:
    runs, probes, burst = data["runs"], data["probes"], data["burst"]
    base = runs.get("online_only", {})
    bsum = (base.get("online") or {}).get("summary") or {}
    lines = [
        "| condition | node | online p50 | online p99 | vs baseline (cross-node) | "
        "batch otok/s | % of steady-state ceiling | salvaged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond in ("online_only", "offline_only", "naive", "technique_a", "technique_b"):
        d = runs.get(cond)
        if not d:
            lines.append(f"| {cond} | — | — | — | — | — | — | missing |")
            continue
        on = (d.get("online") or {}).get("summary") or {}
        bt = d.get("batch") or {}
        ceil = d.get("_ceiling")
        pc = f"{100 * bt['output_tokens_per_s'] / ceil:.1f}%" if bt.get("output_tokens_per_s") and ceil else "—"
        vs = (
            f"{on['p50'] / bsum['p50']:.2f}x / {on['p99'] / bsum['p99']:.2f}x"
            if on.get("p50") and bsum.get("p50")
            else "—"
        )
        lines.append(
            f"| {cond} | {d['_node'][:14]} | "
            f"{on.get('p50', float('nan')):.3f}s | {on.get('p99', float('nan')):.3f}s | {vs} | "
            f"{bt.get('output_tokens_per_s') or 0:.0f} | {pc} | {bt.get('drain_timed_out', False)} |"
        )
    lines.append("")
    lines.append(
        f"Steady-state ceilings by node (last {TAIL_S:.0f}s of the probe burst, otok/s): "
        f"{json.dumps({k: round(v, 1) for k, v in probes.items()})}"
    )
    lines.append("")
    lines.append(
        "Whole-burst probe averages by node (RAMP-DOMINATED -- reported, not used as a "
        f"denominator): {json.dumps({k: round(v, 1) for k, v in burst.items()})}"
    )
    lines.append("")
    lines.append(
        "Latency `vs baseline` ratios are CROSS-NODE (no within-node online_only/"
        "technique_a pair exists); throughput is normalized per-node."
    )
    for label, vals in (("steady-state", probes), ("whole-burst", burst)):
        v = list(vals.values())
        if len(v) > 1:
            mid = (max(v) + min(v)) / 2
            lines.append("")
            lines.append(f"Cross-node {label} band: +-{100 * (max(v) - min(v)) / 2 / mid:.2f}%")
    md = "\n".join(lines)
    (out / "gpu_summary.md").write_text(md)
    return md


def latency_cdf(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for cond, (color, label) in COND_STYLE.items():
        d = data["runs"].get(cond)
        if not d:
            continue
        lats = sorted(
            r["latency_s"] for r in (d.get("online") or {}).get("requests", []) if r.get("latency_s")
        )
        if not lats:
            continue
        ax.plot(lats, np.linspace(0, 1, len(lats)), color=color, lw=1.8, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("online request latency (s, log)")
    ax.set_ylabel("fraction of requests")
    ax.set_title("GPU (A6000, Qwen2.5-7B): online latency by condition", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "gpu_latency_cdf.png", dpi=150)


def throughput(data: dict, out: Path) -> None:
    conds, vals, colors = [], [], []
    for cond in ("offline_only", "naive", "technique_a", "technique_b"):
        d = data["runs"].get(cond)
        if not d or not d.get("_ceiling"):
            continue
        bt = d.get("batch") or {}
        if not bt.get("output_tokens_per_s"):
            continue
        conds.append(cond)
        vals.append(100 * bt["output_tokens_per_s"] / d["_ceiling"])
        colors.append(COND_STYLE.get(cond, (ACCENT, cond))[0])
    if not conds:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(conds, vals, color=colors, width=min(0.6, 0.2 + 0.2 * len(conds)))
    ax.axhline(100, ls="--", c="#999", lw=1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("batch throughput\n(% of same-node steady-state offline ceiling)", fontsize=9)
    ax.set_title("GPU: work conservation by condition", fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.text(
        len(conds) - 0.5,
        101.5,
        "steady-state ceiling (last 10 s of probe burst)",
        ha="right",
        fontsize=7.5,
        color="#777",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "gpu_throughput.png", dpi=150)


def diurnal(data: dict, out: Path, evidence_dirs: list[str]) -> None:
    files = []
    for ev in evidence_dirs:
        files += glob.glob(f"{ev}/**/matrix/diurnal_technique_*.json", recursive=True)
        files += glob.glob(f"{ev}/**/diurnal/diurnal_technique_*.json", recursive=True)
    if not files:
        return
    d = json.load(open(sorted(files)[-1]))
    W = int(d.get("window_s") or 1200)
    B = 60
    bins = np.arange(0, W + B, B)
    on_t = [r["scheduled_at"] for r in (d.get("online") or {}).get("requests", []) if r.get("scheduled_at") is not None]
    bc_t = [
        c["finished_at"]
        for c in d.get("batch_completions", [])
        if c.get("error") is None and c.get("finished_at", 1e9) <= W
    ]
    if not on_t or not bc_t:
        return
    on_h, _ = np.histogram(on_t, bins)
    bc_h, _ = np.histogram(bc_t, bins)
    # Bin 0 is pipeline fill, not tide: it completes a fraction of every later
    # minute's batch work while online load is already at its first peak, which
    # single-handedly halves the correlation. Report both, correlate on [1:].
    r_raw = np.corrcoef(on_h, bc_h)[0, 1]
    r = np.corrcoef(on_h[1:], bc_h[1:])[0, 1]
    hi = bc_h[1:][on_h[1:] > on_h[1:].mean()].mean()
    lo = bc_h[1:][on_h[1:] <= on_h[1:].mean()].mean()
    print(
        f"diurnal: corr = {r:.4f} (bin 0 excluded), raw = {r_raw:.4f}; "
        f"batch/min {hi:.0f} high-load vs {lo:.0f} low-load ({100 * (hi - lo) / lo:+.1f}%); "
        f"bin 0 = {bc_h[0]} completions vs {bc_h[1:].min()}-{bc_h[1:].max()} thereafter"
    )
    mids = bins[:-1] / 60 + 0.5
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.fill_between(mids, on_h, color=BLUE, alpha=0.25)
    ax.plot(mids, on_h, color=BLUE, lw=1.8, label="online arrivals / min")
    ax.bar(mids, bc_h, width=0.82, color=ACCENT, alpha=0.85, label="batch completed / min")
    ax.axvspan(0, 1, color="#B0B0B0", alpha=0.35, zorder=0)
    ax.annotate(
        "pipeline fill\n(excluded)",
        xy=(0.5, bc_h[1:].max() * 0.55),
        ha="center",
        fontsize=7.5,
        color="#555",
    )
    ax.annotate(
        f"corr = {r:.2f}\nfirst-minute fill bin excluded\n(raw r = {r_raw:.2f} with it)",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#999"),
    )
    ax.set_xlabel("minute of run")
    ax.set_ylabel("count per minute")
    ax.set_title(f"GPU tide-filling ({d.get('condition')})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "gpu_diurnal_tide.png", dpi=150)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-o"]
    out = Path(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else Path("results/gpu_figures")
    dirs = [a for a in args if not str(out).startswith(a)] or ["evidence"]
    out.mkdir(parents=True, exist_ok=True)
    data = load_runs(dirs)
    print(summarize(data, out))
    latency_cdf(data, out)
    throughput(data, out)
    diurnal(data, out, dirs)
    print(f"figures -> {out}/")


if __name__ == "__main__":
    main()
