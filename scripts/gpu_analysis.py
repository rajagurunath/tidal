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


def load_runs(evidence_dirs: list[str]) -> dict:
    runs: dict[str, dict] = {}
    probes: dict[str, float] = {}  # run-dir prefix -> ceiling otok/s
    for ev in evidence_dirs:
        for f in glob.glob(f"{ev}/**/probe/*.json", recursive=True):
            d = json.load(open(f))
            node = Path(f).parts[-3]
            otps = (d.get("batch") or {}).get("output_tokens_per_s")
            if otps:
                probes[node] = otps
        for f in glob.glob(f"{ev}/**/matrix/*.json", recursive=True):
            d = json.load(open(f))
            cond = d.get("condition") or Path(f).stem
            node = Path(f).parts[-3]
            d["_node"] = node
            d["_ceiling"] = probes.get(node)
            prev = runs.get(cond)
            if prev is None or (d.get("started_at") or "") > (prev.get("started_at") or ""):
                runs[cond] = d
    return {"runs": runs, "probes": probes}


def summarize(data: dict, out: Path) -> str:
    runs, probes = data["runs"], data["probes"]
    base = runs.get("online_only", {})
    bsum = (base.get("online") or {}).get("summary") or {}
    lines = [
        "| condition | node | online p50 | online p99 | vs baseline | batch otok/s | % of ceiling | salvaged |",
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
    lines.append(f"Probe ceilings by node: {json.dumps({k: round(v, 1) for k, v in probes.items()})}")
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
    ax.bar(conds, vals, color=colors)
    ax.axhline(100, ls="--", c="#999", lw=1)
    ax.set_ylabel("batch throughput (% of same-node offline ceiling)")
    ax.set_title("GPU: work conservation by condition", fontsize=11, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
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
    r = np.corrcoef(on_h, bc_h)[0, 1]
    mids = bins[:-1] / 60 + 0.5
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.fill_between(mids, on_h, color=BLUE, alpha=0.25)
    ax.plot(mids, on_h, color=BLUE, lw=1.8, label="online arrivals / min")
    ax.bar(mids, bc_h, width=0.82, color=ACCENT, alpha=0.85, label="batch completed / min")
    ax.annotate(f"corr = {r:.2f}", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="#999"))
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
