"""Plot solve-rate learning curves from runs/*/log.csv.

Two views: solve rate vs training examples seen (sample efficiency, immune to
machine speed) and vs wall-clock seconds (how long you actually wait).

Wall-clock note: this is a CPU project on a laptop that aggressively throttles
when idle/screen-off (one transformer run stalled ~108 min mid-epoch while I was
away from it). So for the wall-clock axis we rebuild time from per-epoch deltas
with any single epoch capped at 4x the run's median, which strips one-time idle
stalls without touching the raw logs. Capped runs get a * in the legend.
"""

import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(path):
    ep, ex, total_s, solve = [], [], [], []
    with open(Path(path) / "log.csv") as f:
        for row in csv.DictReader(f):
            ep.append(int(row["epoch"]))
            ex.append(int(row["examples"]) / 1000)
            total_s.append(float(row["total_s"]))
            solve.append(float(row["solve"]) * 100 if row["solve"] else None)
    # per-epoch wall-clock deltas, then cap idle-throttle outliers
    deltas = [total_s[0]] + [total_s[i] - total_s[i - 1] for i in range(1, len(total_s))]
    med = median(deltas)
    capped = [min(d, 4 * med) for d in deltas]
    clean_s, run = [], 0.0
    for d in capped:
        run += d
        clean_s.append(run)
    was_capped = any(c < d - 1 for c, d in zip(capped, deltas))
    return ex, clean_s, solve, was_capped


def main():
    runs = sys.argv[1:] or sorted(str(p.parent) for p in Path("runs").glob("*/log.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        ex, clean_s, solve, capped = read_run(r)
        # only plot points that have a solve-rate measurement
        xs1 = [s for s, y in zip(clean_s, solve) if y is not None]
        xs2 = [e for e, y in zip(ex, solve) if y is not None]
        ys = [y for y in solve if y is not None]
        label = Path(r).name + (" *" if capped else "")
        ax1.plot(xs1, ys, marker="o", ms=3, label=label)
        ax2.plot(xs2, ys, marker="o", ms=3, label=label)
    ax1.set_xlabel("wall clock (s, idle stalls removed)")
    ax1.set_ylabel("held-out solve rate (%)")
    ax2.set_xlabel("training examples seen (k)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
    ax1.set_title("how fast does it learn to solve ladders?")
    ax2.set_title("sample efficiency")
    fig.tight_layout()
    out = Path("results/learning_curves.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
