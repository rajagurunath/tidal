"""Charts for the A/B evaluation (plan B5).

    python -m tidal.eval.plots --results-dir results --out-dir results/figures

Reads every ``results/<condition>.json`` written by
:mod:`tidal.eval.harness` and renders three figures, which together are the
whole argument of the paper:

``latency_cdf.png``
    Online request-latency CDF, one line per condition. The claim "batch work
    is invisible to online traffic" is a statement about where these lines sit
    relative to ``online_only`` — a CDF shows the whole distribution, so it
    cannot hide a fat tail the way a bar of means can.
``batch_throughput.png``
    Batch throughput as a percentage of the ``offline_only`` ceiling. The
    ceiling is the engine with nothing else running; a co-serving scheme is
    interesting exactly to the extent that it approaches it.
``technique_b_timeline.png``
    For technique B only: online vs batch tokens per scheduling step over the
    run, straight from the ``TidalScheduler`` telemetry. This is the
    work-conserving claim made visible — batch fills the trough and yields
    when online arrives.

Design rules (the ``dataviz`` skill): fixed categorical order — a condition
keeps its colour in every figure, never re-assigned by rank; one axis per
chart, never two scales; a legend for every multi-series figure plus direct
end-labels; recessive grid, no chartjunk, no 3-D, no gradients. The palette is
the skill's validated colourblind-safe categorical set, taken in slot order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import typer

from tidal.eval.harness import CONDITIONS, percent_of_ceiling
from tidal.eval.loadgen import percentile

__all__ = ["CONDITION_COLORS", "app", "cdf", "load_results", "make_figures"]

#: dataviz reference categorical palette, light mode, slots 1..5 in fixed
#: order. Colour follows the *condition*, never its rank, so a figure that
#: drops a condition never repaints the survivors.
CONDITION_COLORS: dict[str, str] = {
    "online_only": "#2a78d6",  # blue
    "offline_only": "#eb6834",  # orange
    "naive": "#1baf7a",  # aqua
    "technique_a": "#eda100",  # yellow
    "technique_b": "#e87ba4",  # magenta
}
CONDITION_LABELS: dict[str, str] = {
    "online_only": "online only",
    "offline_only": "offline only (ceiling)",
    "naive": "naive batch",
    "technique_a": "technique A (gateway)",
    "technique_b": "technique B (scheduler)",
}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"

app = typer.Typer(add_completion=False, help="Render the Tidal A/B evaluation figures.")


def _style() -> None:
    """Recessive chrome, ink-coloured text, no chartjunk."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "legend.frameon": False,
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )


def load_results(results_dir: str | Path) -> dict[str, dict]:
    """Load ``<results_dir>/*.json`` keyed by condition, in canonical order."""
    found: dict[str, dict] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        condition = payload.get("condition") or path.stem
        found[condition] = payload
    ordered = {c: found[c] for c in CONDITIONS if c in found}
    ordered.update({k: v for k, v in found.items() if k not in ordered})
    return ordered


def cdf(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """Empirical CDF: sorted values and their cumulative probabilities.

    Probabilities are ``i/n`` for the i-th of n sorted values, so the curve
    reaches 1.0 at the maximum — the tail is drawn, not implied.
    """
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return [], []
    return ordered, [(i + 1) / n for i in range(n)]


def _label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _color(condition: str) -> str:
    return CONDITION_COLORS.get(condition, "#4a3aa7")


def _online_latencies(payload: dict) -> list[float]:
    return [
        r["latency_s"]
        for r in payload.get("online", {}).get("requests", [])
        if r.get("error") is None and r.get("status") == 200
    ]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_latency_cdf(results: dict[str, dict], out_path: Path) -> Path | None:
    """Online latency CDF, one line per condition (log-x: tails are the story)."""
    import matplotlib.pyplot as plt

    series = {c: _online_latencies(p) for c, p in results.items()}
    series = {c: v for c, v in series.items() if v}
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition, values in series.items():
        xs, ys = cdf(values)
        ax.step(
            xs,
            ys,
            where="post",
            color=_color(condition),
            linewidth=2.0,
            label=_label(condition),
            solid_capstyle="round",
        )
        # A dot on the p99 line, not a text label: curves converge at the top
        # of a CDF, so five end-labels collide into an unreadable stack. The
        # legend carries identity; the dot carries the number the SLO is about.
        ax.plot(
            [percentile(values, 0.99)],
            [0.99],
            marker="o",
            markersize=6,
            color=_color(condition),
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=4,
            linestyle="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("online request latency (s, log scale)")
    ax.set_ylabel("fraction of requests")
    ax.set_ylim(0, 1.02)
    ax.set_title("Online latency distribution by condition")
    ax.grid(axis="both", alpha=0.6)
    ax.axhline(0.99, color=INK_SOFT, linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(
        "p99",
        xy=(1.0, 0.99),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_SOFT,
    )
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK_SOFT)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_batch_throughput(results: dict[str, dict], out_path: Path) -> Path | None:
    """Batch output-token throughput as a percentage of the offline ceiling."""
    import matplotlib.pyplot as plt

    shares = percent_of_ceiling(results, key="output_tokens_per_s")
    shares = {c: v for c, v in shares.items() if c != "online_only"}
    if not shares:
        return None

    conditions = [c for c in CONDITIONS if c in shares]
    values = [shares[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(
        range(len(conditions)),
        values,
        color=[_color(c) for c in conditions],
        width=0.62,
        zorder=3,
    )
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.0f}%",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.axhline(100, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "offline-only ceiling",
        xy=(1.0, 100),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_SOFT,
    )
    ax.set_xticks(range(len(conditions)), [_label(c) for c in conditions], fontsize=9)
    ax.set_ylabel("batch output tokens/s\n(% of offline-only)")
    ax.set_ylim(0, max(110, max(values) * 1.15))
    ax.set_title("Batch throughput recovered while serving online traffic")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_technique_b_timeline(results: dict[str, dict], out_path: Path) -> Path | None:
    """Online vs batch tokens per step, from the in-engine telemetry.

    One axis, one unit (tokens per scheduling step) — the two series are
    directly comparable, which is the entire point: the batch line rises into
    the space the online line leaves free.
    """
    import matplotlib.pyplot as plt

    payload = results.get("technique_b") or {}
    stats = [s for s in payload.get("tidal_stats") or [] if "t" in s]
    if not stats:
        return None
    stats.sort(key=lambda s: s["t"])
    ts = [s["t"] for s in stats]
    online = [s.get("online_tokens", 0) for s in stats]
    batch = [s.get("batch_tokens", 0) for s in stats]
    held = [s.get("held", 0) for s in stats]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.0), sharex=True, height_ratios=[2.2, 1.0])
    # A short run yields only a handful of telemetry lines (one per 100 steps);
    # a polyline through 1-2 points draws nothing at all, so mark the points.
    sparse = {"marker": "o", "markersize": 4} if len(ts) < 30 else {}
    ax.plot(
        ts,
        online,
        color=CONDITION_COLORS["online_only"],
        linewidth=2.0,
        label="online tokens",
        **sparse,
    )
    ax.plot(
        ts,
        batch,
        color=CONDITION_COLORS["technique_b"],
        linewidth=2.0,
        label="batch tokens",
        **sparse,
    )
    ax.set_ylabel("tokens scheduled\nper step")
    ax.set_title("Technique B: batch work fills the slack, step by step")
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK_SOFT)
    ax.grid(axis="y", alpha=0.6)

    ax2.plot(ts, held, color=CONDITION_COLORS["naive"], linewidth=2.0, label="batch held", **sparse)
    ax2.set_ylabel("batch requests\nheld back")
    ax2.set_xlabel("time since run start (s)")
    ax2.grid(axis="y", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9, labelcolor=INK_SOFT)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def make_figures(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Render every figure the available results support."""
    _style()
    results = load_results(results_dir)
    if not results:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = [
        plot_latency_cdf(results, out / "latency_cdf.png"),
        plot_batch_throughput(results, out / "batch_throughput.png"),
        plot_technique_b_timeline(results, out / "technique_b_timeline.png"),
    ]
    return [p for p in made if p is not None]


@app.command()
def render(
    results_dir: str = typer.Option("results", help="Directory of harness JSON results."),
    out_dir: str = typer.Option("results/figures", help="Where to write the PNGs."),
) -> None:
    """Render the comparison figures from a directory of condition results."""
    made = make_figures(results_dir, out_dir)
    if not made:
        typer.echo(f"no usable results in {results_dir}")
        raise typer.Exit(code=1)
    for path in made:
        typer.echo(f"wrote {path}")


@app.callback()
def _main() -> None:  # pragma: no cover - typer plumbing
    """Render the Tidal A/B evaluation figures."""


if __name__ == "__main__":  # pragma: no cover
    app()
