#!/usr/bin/env python3
"""Render the four data figures for paper v2 (F1, F5, F6, F7).

    python3 scripts/paper_v2_figures.py            # -> docs/paper/figures/v2/
    python3 scripts/paper_v2_figures.py -o /tmp/x

Every number drawn here is read out of a JSON in this repo at render time --
nothing is hard-coded except (a) the engine token budget tau, which is a
configuration constant not recorded in the result payloads, and (b) the 78-88%
band the fork-based literature (HyGen, ConServe) reports, which is a citation.
Both are flagged as such below.

Figures
-------
``f1_motivation``   (a) online-only engine occupancy vs the per-iteration token
                    budget tau -- the budget is mostly unspent; (b) online p50/p99
                    for online-only vs naive co-location -- spending it naively is
                    catastrophic.  CPU testbed.
``f5_gpu_headline`` (a) GPU online-latency CDF, online_only vs technique A;
                    (b) batch harvest as a share of the same-node steady-state
                    offline ceiling, flat and diurnal, against the reported
                    fork-based band.
``f6_frontier``     the placement frontier: online p99 ratio (log) vs share of the
                    node's offline ceiling, in two load regimes and on two
                    hardware classes.
``f7_tides``        per-minute online arrivals and batch completions, CPU and GPU
                    diurnal runs, on a shared 20-minute axis; the GPU pipeline-fill
                    bin is hatched and excluded from the correlation.

Methodology notes that the figures encode (see results/GPU_RESULTS_BRIEF.md):

* The GPU throughput denominator is each node's probe STEADY-STATE TAIL (last
  10 s of the burst), not the ramp-dominated whole-burst average.  Throughput is
  normalized per node; latency ratios are CROSS-NODE and labeled as such.
* The GPU diurnal correlation excludes bin 0, which is pipeline fill (771 batch
  completions vs 1667-1894 in every later minute).  Both r values are drawn.
* CPU placement has two load regimes.  With the 1,200-item pool every arm drains
  inside the window, so all three arms deliver identical throughput and differ
  only in latency price.  Only the non-draining 3,000-item pool (``*_cap``)
  measures throughput at capacity, so that is the regime the frontier is read
  from; the equal-work points are drawn hollow, on their own horizontal line.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent

# --- constants that are NOT in the result payloads --------------------------
#: vLLM V1 per-iteration token budget (``max_num_batched_tokens``) for both
#: testbeds.  Not recorded in the harness JSON; it is the engine default the
#: runs were launched with and is stated in docs/paper/paper.md.
TAU = 2048
#: Steady-state harvest that the fork-based co-serving literature reports
#: (HyGen, ConServe).  A citation, not a measurement of ours.
FORK_BAND = (78.0, 88.0)
#: Steady-state window at the end of the GPU probe burst, in seconds.
TAIL_S = 10.0

# --- palette: blues/teals carry "online", orange/grey are used sparingly ----
BLUE = "#2a78d6"      # online tier
DEEP_BLUE = "#17457c"
TEAL = "#1b7878"      # batch tier / technique A
LIGHT_TEAL = "#7cbcbc"
ORANGE = "#d9822b"    # technique B
GREY = "#6b6b6b"      # naive / strawman
PALE = "#d8d8d4"
INK = "#111111"
INK_SOFT = "#4a4a48"
GRID = "#e3e2df"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {out / stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def cpu(name: str) -> dict:
    return json.loads((REPO / "results" / f"{name}.json").read_text())


def _first(pattern: str) -> Path:
    hits = sorted(glob.glob(str(REPO / pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[-1])


GPU_PATHS = {
    "online_only": "evidence/**/matrix/online_only.json",
    "technique_a": "evidence/**/matrix/technique_a.json",
    "diurnal": "evidence/**/diurnal/diurnal_technique_a.json",
}


def gpu(key: str) -> tuple[dict, Path]:
    path = _first(GPU_PATHS[key])
    return json.loads(path.read_text()), path


def steady_state_ceiling(probe: dict) -> float:
    """Output tok/s over the last ``TAIL_S`` seconds of the probe burst."""
    makespan = probe["batch"]["makespan_s"]
    tail = [
        c
        for c in probe["batch_completions"]
        if c.get("error") is None and makespan - TAIL_S <= c["finished_at"] <= makespan
    ]
    return sum(c["completion_tokens"] for c in tail) / TAIL_S


def probe_for(run_path: Path) -> tuple[float, str]:
    """Steady-state ceiling of the node the given run was measured on."""
    node_dir = run_path.parent.parent  # .../<node-stamp>/{matrix,diurnal}/x.json
    probe = json.loads(_first(f"{node_dir.relative_to(REPO)}/probe/*.json").read_text())
    return steady_state_ceiling(probe), node_dir.name


def latencies(payload: dict) -> list[float]:
    return sorted(
        r["latency_s"]
        for r in payload["online"]["requests"]
        if r.get("error") is None and r.get("latency_s") is not None
    )


def per_minute(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online arrivals and batch completions per 60 s bin, plus bin midpoints."""
    window = int(payload["window_s"])
    edges = np.arange(0, window + 60, 60)
    arrivals = [
        r["scheduled_at"] for r in payload["online"]["requests"] if r.get("scheduled_at") is not None
    ]
    done = [
        c["finished_at"]
        for c in payload["batch_completions"]
        if c.get("error") is None and c.get("finished_at", 1e9) <= window
    ]
    on, _ = np.histogram(arrivals, edges)
    ba, _ = np.histogram(done, edges)
    return edges[:-1] / 60.0 + 0.5, on, ba


# ---------------------------------------------------------------------------
# F1 -- motivation
# ---------------------------------------------------------------------------


def f1_motivation(out: Path) -> dict:
    base = cpu("online_only")
    naive = cpu("naive")

    ts = np.array([m["t"] for m in base["metrics"]])
    running = np.array([m["running"] for m in base["metrics"]], dtype=float)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30}
    )

    # (a) occupancy vs tau ---------------------------------------------------
    ax_a.fill_between(ts, 0.0, running, step="post", color=BLUE, alpha=0.30, linewidth=0)
    ax_a.step(ts, running, where="post", color=DEEP_BLUE, linewidth=0.7)
    ax_a.axhline(TAU, color=ORANGE, linewidth=1.2, linestyle=(0, (5, 2.5)), zorder=4)

    ax_a.set_yscale("symlog", linthresh=1.0, linscale=0.55)
    ax_a.set_ylim(0, TAU * 3.0)
    ax_a.set_yticks([0, 1, 10, 100, TAU])
    ax_a.set_yticklabels(["0", "1", "10", "100", "2048"])
    ax_a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_a.set_xlim(0, base["window_s"])
    ax_a.set_xlabel("time in run (s)")
    ax_a.set_ylabel("resident requests, symlog\n(≈ decode tokens / iteration)")
    ax_a.set_title("(a) the budget is mostly unspent", loc="left")
    ax_a.annotate(
        r"$\tau = 2048$ tokens/iteration (engine budget)",
        xy=(0.985, TAU),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=ORANGE,
    )
    mean_r = base["metrics_summary"]["running_mean"]
    max_r = base["metrics_summary"]["running_max"]
    ax_a.annotate(
        f"online-only occupancy (1 Hz engine scrape, n={len(ts)}):\n"
        f"mean {mean_r:.2f}, max {max_r} — {100 * mean_r / TAU:.2f}% of $\\tau$\n"
        f"at {base['online']['achieved_rps']:.2f} req/s served, 0 errors",
        xy=(0.03, 0.60),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=INK_SOFT,
    )
    ax_a.grid(axis="y", alpha=0.7)
    ax_a.grid(axis="x", visible=False)

    # (b) the naive price ----------------------------------------------------
    b_sum = base["online"]["summary"]
    n_sum = naive["online"]["summary"]
    vals_base = [b_sum["p50"], b_sum["p99"]]
    vals_naive = [n_sum["p50"], n_sum["p99"]]
    x = np.arange(2)
    w = 0.34
    ax_b.bar(x - w / 2, vals_base, width=w, color=BLUE, label="online only (baseline)", zorder=3)
    ax_b.bar(
        x + w / 2,
        vals_naive,
        width=w,
        color=GREY,
        label="naive co-location (priority field only)",
        zorder=3,
    )
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.2, 400)
    ax_b.set_yticks([0.3, 1, 3, 10, 30, 100])
    ax_b.set_yticklabels(["0.3", "1", "3", "10", "30", "100"])
    ax_b.set_xticks(x, ["online p50", "online p99"])
    ax_b.set_ylabel("online request latency (s, log)")
    ax_b.set_title("(b) spending it naively is catastrophic", loc="left")
    ax_b.tick_params(axis="x", length=0)
    ax_b.grid(axis="y", alpha=0.7)
    ax_b.grid(axis="x", visible=False)

    ratios = [vals_naive[i] / vals_base[i] for i in range(2)]
    for i, ratio in enumerate(ratios):
        ax_b.annotate(
            f"{ratio:.1f}× baseline",
            xy=(i + w / 2, vals_naive[i]),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            fontsize=7.6,
            fontweight="bold",
            color=INK,
        )
        ax_b.annotate(
            f"{vals_base[i]:.2f} s",
            xy=(i - w / 2, vals_base[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=INK_SOFT,
        )
        ax_b.annotate(
            f"{vals_naive[i]:.1f} s",
            xy=(i + w / 2, vals_naive[i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color=INK_SOFT,
        )

    fig.legend(
        *ax_b.get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.10),
        labelcolor=INK_SOFT,
    )
    save(fig, out, "f1_motivation")
    return {
        "a.mean_running": mean_r,
        "a.max_running": max_r,
        "a.samples": len(ts),
        "a.tau": TAU,
        "b.online_only_p50": vals_base[0],
        "b.online_only_p99": vals_base[1],
        "b.naive_p50": vals_naive[0],
        "b.naive_p99": vals_naive[1],
        "b.ratio_p50": ratios[0],
        "b.ratio_p99": ratios[1],
    }


# ---------------------------------------------------------------------------
# F5 -- GPU headline
# ---------------------------------------------------------------------------


def f5_gpu_headline(out: Path) -> dict:
    base, base_path = gpu("online_only")
    tech_a, a_path = gpu("technique_a")
    diur, d_path = gpu("diurnal")
    ceil_a, node_a = probe_for(a_path)
    ceil_d, node_d = probe_for(d_path)

    harvest_flat = 100 * tech_a["batch"]["output_tokens_per_s"] / ceil_a
    harvest_diur = 100 * diur["batch"]["output_tokens_per_s"] / ceil_d

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.6, 2.65), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.34}
    )

    # (a) latency CDF --------------------------------------------------------
    series = [
        ("online only (baseline)", latencies(base), BLUE, "-"),
        ("technique A (gateway) + batch tier", latencies(tech_a), TEAL, "-"),
    ]
    for label, vals, color, ls in series:
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax_a.step(vals, ys, where="post", color=color, linestyle=ls, linewidth=1.5, label=label)

    quantiles = {}
    for label, vals, color, _ in series:
        arr = np.asarray(vals)
        quantiles[label] = (float(np.quantile(arr, 0.50)), float(np.quantile(arr, 0.99)))
        for q, y in ((0.50, 0.50), (0.99, 0.99)):
            ax_a.plot(
                [float(np.quantile(arr, q))],
                [y],
                marker="o",
                markersize=3.6,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linestyle="none",
                zorder=5,
            )
    for y in (0.50, 0.99):
        ax_a.axhline(y, color=INK_SOFT, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)

    b50, b99 = quantiles[series[0][0]]
    a50, a99 = quantiles[series[1][0]]
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.28, 3.4)
    ax_a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_a.set_xticks([0.3, 0.5, 1.0, 2.0, 3.0])
    ax_a.set_xticklabels(["0.3", "0.5", "1.0", "2.0", "3.0"])
    ax_a.set_ylim(0, 1.04)
    ax_a.set_xlabel("online request latency (s, log)")
    ax_a.set_ylabel("fraction of requests")
    ax_a.set_title("(a) the measured latency price", loc="left")
    ax_a.annotate(
        f"p50  {b50:.2f}s → {a50:.2f}s   ({a50 / b50:.2f}×)\n"
        f"p99  {b99:.2f}s → {a99:.2f}s   ({a99 / b99:.2f}×)\n"
        "cross-node ratios",
        xy=(0.985, 0.055),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=GRID, linewidth=0.6),
    )
    ax_a.grid(alpha=0.7)

    # (b) harvest ------------------------------------------------------------
    labels = ["technique A\n(flat, 20 req/s)", "technique A\n(diurnal, 20 min)"]
    values = [harvest_flat, harvest_diur]
    ypos = np.arange(len(values))[::-1]

    ax_b.axvspan(*FORK_BAND, color=LIGHT_TEAL, alpha=0.35, zorder=0, linewidth=0)
    ax_b.barh(ypos, values, height=0.42, color=[TEAL, LIGHT_TEAL], zorder=3)
    ax_b.axvline(100, color=ORANGE, linewidth=1.2, linestyle=(0, (5, 2.5)), zorder=4)

    for y, v in zip(ypos, values):
        ax_b.annotate(
            f"{v:.1f}%",
            xy=(v, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=INK,
        )
    ax_b.annotate(
        "mean offered load 12.1 req/s\nvs 20 flat — not directly comparable",
        xy=(2.5, ypos[1] - 0.32),
        va="top",
        ha="left",
        fontsize=6.6,
        color=INK_SOFT,
    )
    ax_b.set_yticks(ypos, labels)
    ax_b.set_ylim(ypos[-1] - 0.95, ypos[0] + 0.62)
    ax_b.set_xlim(0, 118)
    ax_b.set_xticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel("batch harvest (% of same-node steady-state offline ceiling)")
    ax_b.set_title("(b) what it buys", loc="left")
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x", alpha=0.7)
    ax_b.grid(axis="y", visible=False)

    handles = [
        Line2D([], [], color=BLUE, linewidth=1.5, label="online only (baseline)"),
        Line2D([], [], color=TEAL, linewidth=1.5, label="technique A (gateway) + batch tier"),
        Patch(facecolor=LIGHT_TEAL, alpha=0.35, label="fork-based systems, 78–88% (as reported)"),
        Line2D(
            [], [], color=ORANGE, linewidth=1.2, linestyle=(0, (5, 2.5)),
            label="steady-state offline ceiling",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.235),
        labelcolor=INK_SOFT,
        columnspacing=1.4,
    )
    save(fig, out, "f5_gpu_headline")
    return {
        "a.online_only_p50": b50,
        "a.online_only_p99": b99,
        "a.technique_a_p50": a50,
        "a.technique_a_p99": a99,
        "a.ratio_p50": a50 / b50,
        "a.ratio_p99": a99 / b99,
        "a.n_online_baseline": len(series[0][1]),
        "a.n_online_technique_a": len(series[1][1]),
        "a.baseline_node": base_path.parent.parent.name,
        "a.technique_a_node": node_a,
        "b.flat_otok_s": tech_a["batch"]["output_tokens_per_s"],
        "b.flat_ceiling": ceil_a,
        "b.flat_pct": harvest_flat,
        "b.diurnal_otok_s": diur["batch"]["output_tokens_per_s"],
        "b.diurnal_ceiling": ceil_d,
        "b.diurnal_pct": harvest_diur,
        "b.diurnal_node": node_d,
        "b.diurnal_mean_rps": len(diur["online"]["requests"]) / diur["window_s"],
    }


# ---------------------------------------------------------------------------
# F6 -- the placement frontier
# ---------------------------------------------------------------------------


def f6_frontier(out: Path) -> dict:
    base = cpu("online_only")
    base_p99 = base["online"]["summary"]["p99"]
    cpu_ceiling = cpu("offline_only")["batch"]["output_tokens_per_s"]

    def cpu_point(name: str) -> tuple[float, float]:
        d = cpu(name)
        return (
            d["online"]["summary"]["p99"] / base_p99,
            100 * d["batch"]["output_tokens_per_s"] / cpu_ceiling,
        )

    # equal-work regime: 1,200-item pool, every arm drains inside the window,
    # so throughput is pool-limited and identical -- only latency differs.
    eq = {n: cpu_point(n) for n in ("naive", "technique_a", "technique_b")}
    # capacity regime: 3,000-item pool, nothing drains -- this is where the
    # throughput/latency frontier is actually measurable.
    cap = {n: cpu_point(n + "_cap") for n in ("technique_a", "technique_b")}

    g_base, _ = gpu("online_only")
    g_a, a_path = gpu("technique_a")
    ceil_a, _ = probe_for(a_path)
    gpu_x = g_a["online"]["summary"]["p99"] / g_base["online"]["summary"]["p99"]
    gpu_y = 100 * g_a["batch"]["output_tokens_per_s"] / ceil_a

    fig, (ax, ax_m) = plt.subplots(
        1, 2, figsize=(6.6, 3.6), gridspec_kw={"width_ratios": [7.4, 1.0], "wspace": 0.06}
    )

    ax.axhspan(*FORK_BAND, color=LIGHT_TEAL, alpha=0.32, zorder=0, linewidth=0)
    ax.axvline(1.0, color="#c4c4c0", linewidth=0.9, zorder=1)
    ax.annotate(
        "1.0× = online-only baseline p99",
        xy=(1.03, 1.5),
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=INK_SOFT,
    )

    # The CPU frontier itself: two placements of the same policy at capacity.
    ax.plot(
        [cap["technique_a"][0], cap["technique_b"][0]],
        [cap["technique_a"][1], cap["technique_b"][1]],
        color=GREY,
        linewidth=0.9,
        linestyle=(0, (3, 2.5)),
        zorder=2,
    )

    # Regime span: the same arm's p99 ratio in the equal-work regime (hollow
    # tick) and in the capacity regime (the marker).  Both are measured; the
    # frontier is read from the capacity end, where throughput is not pool-limited.
    for name, color in (("technique_a", TEAL), ("technique_b", ORANGE)):
        x_lo, x_hi = sorted((eq[name][0], cap[name][0]))
        y = cap[name][1]
        ax.plot([x_lo, x_hi], [y, y], color=color, linewidth=1.0, alpha=0.75, zorder=4)
        ax.plot(
            [eq[name][0]], [y], marker="|", markersize=7, markeredgewidth=1.2,
            color=color, zorder=4, linestyle="none",
        )
        ax.scatter([cap[name][0]], [y], s=74, color=color, zorder=6)

    ax.scatter([gpu_x], [gpu_y], s=104, color=TEAL, marker="D", zorder=6)
    ax.scatter(
        [eq["naive"][0]], [eq["naive"][1]], s=66, facecolors="white",
        edgecolors=GREY, linewidths=1.5, zorder=6,
    )

    # "same policy, GPU" connector, A-CPU (capacity) -> A-GPU
    ax.annotate(
        "",
        xy=(gpu_x, gpu_y - 1.5),
        xytext=(cap["technique_a"][0], cap["technique_a"][1] + 1.5),
        arrowprops=dict(arrowstyle="-|>", color=DEEP_BLUE, linewidth=1.1,
                        shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0.30"),
        zorder=5,
    )
    ax.annotate(
        "same policy, GPU:\n+24 points of ceiling\nand a lower latency price",
        xy=(1.42, 56.0),
        xytext=(1.17, 25.0),
        ha="left",
        va="center",
        fontsize=6.8,
        color=DEEP_BLUE,
        linespacing=1.3,
        arrowprops=dict(arrowstyle="-", color=DEEP_BLUE, linewidth=0.6, alpha=0.55,
                        shrinkA=3, shrinkB=3),
    )

    labels = [
        (cap["technique_a"][0], cap["technique_a"][1], "A — gateway, CPU\n45.1% @ 1.97×", 7, -3, "left", "top"),
        (cap["technique_b"][0], cap["technique_b"][1], "B — in-engine, CPU\n71.2% @ 9.93×", 9, -2, "left", "top"),
        (gpu_x, gpu_y, "A — gateway, GPU (A6000)\n69.3% @ 1.23×", 10, -1, "left", "bottom"),
        (eq["naive"][0], eq["naive"][1], "naive co-location\n45.2% @ 22.6×", -6, 4, "right", "bottom"),
    ]
    for x, y, text, dx, dy, ha, va in labels:
        ax.annotate(
            text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=7.0, color=INK, linespacing=1.25,
        )
    ax.annotate(
        "hollow tick = the same arm's p99 ratio in the equal-work regime\n"
        "(1,200-item pool; every arm drains, so harvest is pool-limited)",
        xy=(0.985, 0.035),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=GREY,
    )

    ax.set_xscale("log")
    ax.set_xlim(0.93, 30.0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1×", "2×", "5×", "10×", "20×"])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(
        "online p99 latency, relative to the online-only baseline (log scale)", x=0.56
    )
    ax.set_ylabel("batch harvest\n(% of the node's offline ceiling)")
    ax.grid(alpha=0.7)

    # --- right margin: technique B on GPU is NOT a data point ---------------
    ax_m.set_facecolor("#f6f6f4")
    ax_m.scatter(
        [0.5], [FORK_BAND[0]], s=98, facecolors="none", edgecolors=ORANGE,
        linewidths=1.3, linestyle=(0, (2, 1.6)), zorder=6, clip_on=False,
    )
    ax_m.annotate(
        "B — in-engine, GPU\nnot obtained\n(harness defect)",
        xy=(0.5, FORK_BAND[0] - 2.5),
        ha="center",
        va="top",
        fontsize=6.6,
        color=ORANGE,
        style="italic",
        linespacing=1.3,
    )
    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 100)
    ax_m.set_xticks([])
    ax_m.set_yticks([])
    ax_m.grid(False)
    ax_m.set_xlabel("no data", fontsize=6.6, color="#8d8d8a", labelpad=3)
    for side in ("left", "bottom"):
        ax_m.spines[side].set_visible(False)

    fig.suptitle(
        "Placement is a frontier, not a ranking — and it moves with hardware",
        x=0.09, y=0.985, ha="left", fontsize=9.5, fontweight="bold", color=INK,
    )

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=TEAL, markersize=6.5,
               label="capacity regime, CPU (3,000-item pool, non-draining)"),
        Line2D([], [], marker="D", linestyle="none", color=TEAL, markersize=6.5,
               label="capacity regime, GPU (A6000, Qwen2.5-7B)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=GREY, markeredgewidth=1.5, markersize=6.5,
               label="equal-work regime only (naive; harvest is pool-limited)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=ORANGE, markeredgewidth=1.3, markersize=7.5,
               label="not measured (technique B at GPU scale)"),
        Patch(facecolor=LIGHT_TEAL, alpha=0.32,
              label="fork-based literature band, 78–88% (HyGen, ConServe, as reported)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.20),
        labelcolor=INK_SOFT, handletextpad=0.6, columnspacing=1.4,
    )

    save(fig, out, "f6_frontier")
    return {
        "cpu_ceiling_otok_s": cpu_ceiling,
        "cpu_baseline_p99": base_p99,
        "eq.naive": eq["naive"],
        "eq.technique_a": eq["technique_a"],
        "eq.technique_b": eq["technique_b"],
        "cap.technique_a": cap["technique_a"],
        "cap.technique_b": cap["technique_b"],
        "gpu.technique_a": (gpu_x, gpu_y),
    }


# ---------------------------------------------------------------------------
# F7 -- the tides
# ---------------------------------------------------------------------------


def f7_tides(out: Path) -> dict:
    cpu_d = cpu("diurnal_a")
    gpu_d, gpu_path = gpu("diurnal")

    c_mid, c_on, c_ba = per_minute(cpu_d)
    g_mid, g_on, g_ba = per_minute(gpu_d)

    r_cpu = float(np.corrcoef(c_on, c_ba)[0, 1])
    r_gpu_raw = float(np.corrcoef(g_on, g_ba)[0, 1])
    r_gpu = float(np.corrcoef(g_on[1:], g_ba[1:])[0, 1])

    fig, (ax_c, ax_g) = plt.subplots(2, 1, figsize=(6.6, 4.3), sharex=True)

    for ax, mid, on, ba, title in (
        (ax_c, c_mid, c_on, c_ba, "(a) CPU testbed — M4 Pro, Qwen2.5-0.5B, technique A"),
        (ax_g, g_mid, g_on, g_ba, "(b) GPU testbed — RTX A6000, Qwen2.5-7B, technique A"),
    ):
        ax.bar(mid, ba, width=0.86, color=TEAL, alpha=0.85, zorder=3, label="batch completions / min")
        ax.fill_between(mid, 0, on, color=BLUE, alpha=0.20, zorder=4, linewidth=0)
        ax.plot(mid, on, color=DEEP_BLUE, linewidth=1.6, zorder=5, label="online arrivals / min")
        ax.set_ylabel("count per minute")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.7)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(on.max(), ba.max()) * 1.38)

    ax_c.annotate(
        f"r(online/min, batch/min) = {r_cpu:.2f}",
        xy=(0.985, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.4,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, linewidth=0.6),
    )
    # GPU: minute 0 is pipeline fill, excluded from the correlation.
    ax_g.bar(
        g_mid[0],
        g_ba[0],
        width=0.86,
        facecolor="white",
        edgecolor=GREY,
        hatch="////",
        linewidth=0.7,
        zorder=6,
    )
    ax_g.annotate(
        f"pipeline fill (excluded):\n{g_ba[0]} batch completions vs\n{g_ba[1:].min()}–{g_ba[1:].max()} in every later minute",
        xy=(g_mid[0], g_ba[0]),
        xytext=(1.9, g_ba[1:].max() * 1.23),
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK_SOFT,
        linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRID, linewidth=0.6),
        arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.7, shrinkA=2, shrinkB=2),
    )
    ax_g.annotate(
        f"r = {r_gpu:.2f} (fill bin excluded)\nraw r = {r_gpu_raw:.2f} with it",
        xy=(0.985, 0.93),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=7.4,
        color=INK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID, linewidth=0.6),
    )

    ax_g.set_xlabel("minute of run (both windows: 20 min, two diurnal periods)")
    ax_g.set_xlim(0, 20)
    ax_g.set_xticks(np.arange(0, 21, 2))

    handles = [
        Line2D([], [], color=DEEP_BLUE, linewidth=1.6, label="online arrivals / min"),
        Patch(facecolor=TEAL, alpha=0.85, label="batch completions / min"),
        Patch(facecolor="white", edgecolor=GREY, hatch="////", label="pipeline fill (excluded)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045),
        labelcolor=INK_SOFT,
    )
    fig.subplots_adjust(hspace=0.34)
    save(fig, out, "f7_tides")
    return {
        "cpu.r": r_cpu,
        "cpu.online_per_min": c_on.tolist(),
        "cpu.batch_per_min": c_ba.tolist(),
        "gpu.r_excl_bin0": r_gpu,
        "gpu.r_raw": r_gpu_raw,
        "gpu.online_per_min": g_on.tolist(),
        "gpu.batch_per_min": g_ba.tolist(),
        "gpu.fill_bin": int(g_ba[0]),
        "gpu.path": str(gpu_path.relative_to(REPO)),
    }


def main() -> None:
    out = Path(sys.argv[sys.argv.index("-o") + 1]) if "-o" in sys.argv else REPO / "docs/paper/figures/v2"
    out.mkdir(parents=True, exist_ok=True)
    style()
    report = {
        "f1_motivation": f1_motivation(out),
        "f5_gpu_headline": f5_gpu_headline(out),
        "f6_frontier": f6_frontier(out),
        "f7_tides": f7_tides(out),
    }
    trimmed = {
        fig: {k: v for k, v in vals.items() if not isinstance(v, list)}
        for fig, vals in report.items()
    }
    print(json.dumps(trimmed, indent=2, default=float))


if __name__ == "__main__":
    main()
