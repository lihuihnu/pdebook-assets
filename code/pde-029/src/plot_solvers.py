"""Independent quantitative plots reading only actual saved CSV data."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {
    "jacobi": ("#333333", "-", "o", "Jacobi"),
    "cg": ("#0072B2", "--", "s", "CG"),
    "gmres": ("#D55E00", ":", "^", "GMRES"),
    "gmres_restart": ("#009E73", "-.", "D", "GMRES(12)")
}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def plot(results, output):
    summary, history = read(results / "summary.csv"), read(results / "history.csv")
    plt.rcParams.update({"font.size": 19, "axes.labelsize": 22, "legend.fontsize": 18,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 10), constrained_layout=True)
    constant = [r for r in summary if r["problem"] == "constant" and r["method"] == "cg"]
    n = [int(r["n"]) for r in constant]
    axes[0].plot(n, [q * q for q in n], "o-", color="#333333", label=r"Dense values: $n^2$")
    axes[0].plot(n, [int(r["nnz"]) for r in constant], "s--", color="#0072B2",
                 label=r"CSR values: $\mathrm{nnz}(A)$")
    axes[0].set(xlabel=r"Internal unknowns $n$", ylabel="Number of stored values",
                yscale="log")
    axes[0].legend(loc="upper left")
    for method, (color, line, marker, label) in STYLE.items():
        data = [r for r in summary if r["problem"] == "constant" and r["method"] == method]
        axes[1].plot([int(r["N"]) for r in data], [int(r["iterations"]) for r in data],
                     color=color, linestyle=line, marker=marker, label=label,
                     markersize=7, linewidth=2)
    axes[1].set(xlabel=r"Elements $N$, $f=1$, $c=0$", ylabel="Iterations to tolerance",
                yscale="log")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    for ax in axes:
        ax.grid(True, which="major", alpha=0.22)
    save(fig, output / "fig-001-storage-and-iterations")
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 10), constrained_layout=True)
    for ax, problem, methods, label in (
        (axes[0], "constant", list(STYLE), r"Iteration $k+1$, $N=32$, $f=1$, $c=0$"),
        (axes[1], "convection", ["gmres", "gmres_restart"],
         r"Iteration $k+1$, $N=32$, $c=8$")):
        for method in methods:
            color, line, marker, name = STYLE[method]
            data = [r for r in history if r["problem"] == problem and r["N"] == "32"
                    and r["method"] == method and float(r["relative_residual"]) > 0]
            ax.plot([int(r["k"]) + 1 for r in data],
                    [float(r["relative_residual"]) for r in data],
                    color=color, linestyle=line, label=name, linewidth=2.2)
        ax.axhline(1e-8, color="#777777", linestyle="--", linewidth=1.2,
                   label=r"$\varepsilon_{\mathrm{rel}}=10^{-8}$")
        ax.set(xlabel=label, ylabel=r"$\|b-AU^{(k)}\|_2/\|b\|_2$",
               xscale="log", yscale="log", ylim=(1e-16, 10))
        ax.legend(loc="lower left")
        ax.grid(True, which="major", alpha=0.22)
    save(fig, output / "fig-002-true-residuals")


def save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=200, metadata={"Software": "pdebook"})
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "pdebook",
                                                   "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=root / "experiments/pde-029/results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "exports")
    args = parser.parse_args()
    plot(args.results, args.output)
