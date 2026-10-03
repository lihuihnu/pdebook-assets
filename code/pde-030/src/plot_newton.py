"""Pure quantitative figures from saved Newton trajectories and trials."""
import argparse
import csv
import io
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"full": ("#0072B2", "-", "o", "Full Newton"),
         "damped": ("#D55E00", "--", "s", "Damped Newton")}


def read(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def selected(rows, N, method):
    return [r for r in rows if r["mu"] == "1000" and r["N"] == str(N)
            and r["initial"] == "0" and r["method"] == method]


def save(fig, stem):
    for extension, metadata in (("png", {"Software": "pdebook"}),
                               ("pdf", {"Creator": "pdebook",
                                        "CreationDate": None, "ModDate": None})):
        stream = io.BytesIO()
        fig.savefig(stream, format=extension, dpi=200, metadata=metadata)
        target = stem.with_suffix("." + extension)
        pending = target.with_suffix(target.suffix + ".tmp")
        pending.write_bytes(stream.getvalue())
        pending.replace(target)
    plt.close(fig)


def plot(results, output):
    history, trials = read(results / "history.csv"), read(results / "trials.csv")
    plt.rcParams.update({"font.size": 19, "axes.labelsize": 22, "legend.fontsize": 18,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 10), constrained_layout=True)
    for ax, N in zip(axes, (2, 32)):
        for method, (color, line, marker, label) in STYLE.items():
            rows = [r for r in selected(history, N, method)
                    if float(r["relative_residual"]) > 0]
            ax.plot([int(r["k"]) for r in rows],
                    [float(r["relative_residual"]) for r in rows],
                    color=color, linestyle=line, marker=marker, label=label,
                    linewidth=2.2, markersize=6)
        ax.axhline(1e-10, color="#555555", linestyle=":", linewidth=1.5,
                   label=r"Tolerance $10^{-10}$")
        ax.set(xlabel=rf"Iteration $k$, $N={N}$, $\mu=1000$",
               ylabel="Relative residual", yscale="log", ylim=(1e-16, 1e7),
               xlim=(-0.4, 17.4), xticks=(0, 4, 8, 12, 16))
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                  fontsize=16, handlelength=1.6, columnspacing=1.2)
        ax.grid(True, which="major", alpha=0.22)
    save(fig, output / "fig-001-nonlinear-residuals")

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 10), constrained_layout=True)
    rows = [r for r in selected(trials, 2, "damped") if r["k"] == "0"]
    initial_norm = float(selected(history, 2, "damped")[0]["residual_norm"])
    j = [int(r["trial"]) for r in rows]
    ratios = [float(r["residual_norm"]) / initial_norm for r in rows]
    limits = [math.sqrt(2 * float(r["armijo_bound"])) / initial_norm for r in rows]
    axes[0].plot(j, ratios, "o-", color="#0072B2", linewidth=2.2,
                 markersize=6, label="Trial residual ratio")
    axes[0].plot(j, limits, ":", color="#555555", linewidth=1.7,
                 label="Armijo ratio limit")
    accepted = [r for r in rows if r["accepted"] == "1"]
    axes[0].scatter([int(r["trial"]) for r in accepted],
                    [float(r["residual_norm"]) / initial_norm for r in accepted],
                    color="#D55E00", marker="s", s=80, zorder=4, label="Accepted")
    axes[0].set(xlabel=r"Trial $j$, $\lambda=2^{-j}$, $N=2$",
                ylabel="Candidate / initial residual", yscale="log", xticks=j)
    axes[0].legend(loc="upper right")
    for method, (color, line, marker, label) in STYLE.items():
        rows = selected(history, 2, method)
        pairs = [(float(a["error_max"]), float(b["error_max"])) for a, b in zip(rows, rows[1:])
                 if float(b["lambda"]) == 1.0 and 0 < float(a["error_max"]) <= 0.5
                 and float(b["error_max"]) > 0]
        axes[1].plot([a for a, b in pairs], [b for a, b in pairs],
                      color=color, linestyle=line, marker=marker,
                      linewidth=1.5, markersize=7, label=label)
    reference_x = [10 ** (-8 + 8 * i / 100) for i in range(101)]
    axes[1].plot(reference_x, [(750 / 752) * x * x for x in reference_x],
                  ":", color="#555555", linewidth=1.7, label=r"$(750/752)e^2$")
    axes[1].set(xlabel=r"Current error $|e^k|$, $N=2$",
                ylabel=r"Next error $|e^{k+1}|$", xscale="log", yscale="log",
                xlim=(1e-8, 0.5), ylim=(1e-16, 1))
    axes[1].legend(loc="upper left")
    for ax in axes:
        ax.grid(True, which="major", alpha=0.22)
    save(fig, output / "fig-002-backtracking-and-local-errors")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "experiments/pde-030/results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "exports")
    args = parser.parse_args()
    plot(args.results, args.output)
