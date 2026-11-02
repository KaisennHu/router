"""Generate the Full-versus-Cache figure from the published run data."""

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
    ("TTFT mean", "ttft_ms_mean", "ms / request", False),
    ("TPOT mean", "tpot_ms_mean", "ms / output token", False),
    ("Task duration (JCT) mean", "jct_task_duration_s_mean", "s / task", False),
    ("Total throughput", "total_tok_s", "logical tokens / s", True),
)


def mean(cap, arm, metric):
    rows = [row for row in ROWS if row["concurrency"] == cap and row["arm"] == arm]
    expected_repeats = {"1"} if arm == "cache_aware" else {"1", "2"}
    assert len(rows) == len(expected_repeats)
    assert {row["repeat"] for row in rows} == expected_repeats
    assert all(row["requests"] == "730" and row["sessions"] == "32" for row in rows)
    return sum(float(row[metric]) for row in rows) / len(rows)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4), layout="constrained")
fig.suptitle("Full KV-aware vs Cache-aware P: end-to-end results\n"
             "Cache: one run; Full: mean of two runs (32 sessions / 730 requests)",
             fontsize=18, fontweight="bold")
x = np.arange(len(CAPS))
width = 0.31
for ax, (title, metric, unit, higher_is_better) in zip(axes.flat, PANELS):
    cache = np.array([mean(c, "cache_aware", metric) for c in CAPS])
    full = np.array([mean(c, "kv_aware_aff_progress", metric) for c in CAPS])
    ax.bar(x - width / 2, cache, width, color="#64748b", label="Cache-aware P", zorder=3)
    ax.bar(x + width / 2, full, width, color="#2563eb",
           label="KV-aware + affinity + progress", zorder=3)
    height = max(cache.max(), full.max())
    for index, (baseline, treatment) in enumerate(zip(cache, full)):
        change = (treatment / baseline - 1) * 100
        better = change > 0 if higher_is_better else change < 0
        sign = "+" if change >= 0 else "−"
        ax.text(index, max(baseline, treatment) + height * 0.025,
                f"{sign}{abs(change):.2f}%", ha="center", va="bottom",
                color="#047857" if better else "#b91c1c", fontweight="bold", fontsize=11)
    ax.set_ylim(0, height * 1.23)
    ax.set_xticks(x, [f"C{c}" for c in CAPS])
    ax.set_title(title, loc="left")
    ax.set_ylabel(unit)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.text(0.99, 0.97, "Higher is better" if higher_is_better else "Lower is better",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#475569")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, fontsize=12)
out = ROOT / "figures" / "full-vs-cache.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(out)
