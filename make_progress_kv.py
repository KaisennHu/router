"""Plot Progress's incremental effect on KV-aware routing without affinity."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
with (ROOT / "data" / "repeats-metrics.csv").open(newline="") as f:
    ROWS = list(csv.DictReader(f))

CAPS = ("4", "8", "16", "32")
PANELS = (
    ("P queue: change in mean", "prefill_queue_s_mean_ms", "ms / request", False, False),
    ("P prefill: change in mean", "prefill_prefill_s_mean_ms", "% vs Progress off", True, False),
    ("Client TTFT: change in mean", "ttft_ms_mean", "% vs Progress off", True, False),
    ("Output throughput: change", "output_tok_s", "% vs Progress off", True, True),
)


def mean(cap, arm, metric):
    rows = [r for r in ROWS if r["concurrency"] == cap and r["arm"] == arm]
    assert len(rows) == 2 and {r["repeat"] for r in rows} == {"1", "2"}
    assert all(r["requests"] == "730" and r["sessions"] == "32" for r in rows)
    return sum(float(r[metric]) for r in rows) / 2


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5, "axes.titlesize": 16,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})
fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), layout="constrained")
fig.suptitle("Prefill progress: KV → KV + Progress (no D affinity)\n"
             "Mean of two runs per arm (32 sessions / 730 requests)",
             fontsize=19, fontweight="bold")
x = np.arange(len(CAPS))
for ax, (title, metric, unit, percent, higher_is_better) in zip(axes.flat, PANELS):
    before = [mean(c, "kv_aware", metric) for c in CAPS]
    after = [mean(c, "kv_aware_progress", metric) for c in CAPS]
    delta = [(b / a - 1) * 100 if percent else b - a for a, b in zip(before, after)]
    colors = ["#047857" if ((v > 0) if higher_is_better else (v < 0)) else "#b91c1c" for v in delta]
    bars = ax.bar(x, delta, width=0.46, color=colors, zorder=3)
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in delta], padding=3, fontsize=11)
    ax.axhline(0, color="#334155", linewidth=1)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.set_title(title, loc="left")
    ax.set_ylabel(unit)
    ax.set_xticks(x, [f"C{c}" for c in CAPS])
    ax.text(0.99, 0.97, "Higher is better" if higher_is_better else "Lower is better",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#475569")
    low, high = ax.get_ylim()
    ax.set_ylim(low - (high - low) * 0.12, high + (high - low) * 0.17)

out = ROOT / "figures" / "progress-kv-no-affinity.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(out)
