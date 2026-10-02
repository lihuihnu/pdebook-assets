"""Plot only stored experiment CSVs; no PDE solving or invented data."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"fd": "Finite difference", "fv": "Nodal finite volume", "fe": "Linear finite element"}
COLORS = {"fd": "#304D72", "fv": "#A15728", "fe": "#497453"}
STYLES = {"fd": "-", "fv": "--", "fe": ":"}
MARKERS = {"fd": "o", "fv": "s", "fe": "^"}


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def save(figure, output, name):
    figure.savefig(output / (name + ".png"), dpi=200, facecolor="white",
                   metadata={"Software": "Matplotlib"})
    figure.savefig(output / (name + ".pdf"),
                   metadata={"Creator": "Matplotlib", "CreationDate": None,
                             "ModDate": None})
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "experiments/pde-028/results")
    parser.add_argument("--output", type=Path, default=root / "figures/pde-028/exports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 21, "legend.fontsize": 17,
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    profiles = read_csv(args.results / "profiles.csv")
    summary = read_csv(args.results / "summary.csv")
    figure, axes = plt.subplots(2, 1, figsize=(9.6, 10.0))
    for method in LABELS:
        data = [r for r in profiles if r["source"] == "quadratic"
                and r["mesh"] == "uniform-4" and r["method"] == method]
        axes[0].plot([float(r["x"]) for r in data], [abs(float(r["error"])) for r in data],
                     color=COLORS[method], linestyle=STYLES[method], linewidth=2.8,
                     label=LABELS[method])
        data = [r for r in summary if r["family"] == "uniform"
                and r["source"] == "quadratic" and r["method"] == method]
        data.sort(key=lambda row: int(row["N"]))
        axes[1].loglog([float(r["hmax"]) for r in data],
                       [float(r["l2_error"]) for r in data],
                       color=COLORS[method], linestyle=STYLES[method],
                       marker=MARKERS[method], markersize=8, linewidth=2.6,
                       label=LABELS[method])
    axes[0].set(xlabel=r"$x$", ylabel=r"$|u_h-u|$", xlim=(0, 1), ylim=(0, 0.0075))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].set(xlabel=r"$h$", ylabel=r"$E_{L^2}$")
    axes[1].set_xticks([1/64, 1/32, 1/16, 1/8, 1/4],
                       labels=[r"$1/64$", r"$1/32$", r"$1/16$", r"$1/8$", r"$1/4$"])
    axes[1].minorticks_off()
    axes[1].legend(loc="upper left", frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.7)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.08, top=0.96, hspace=0.36)
    save(figure, args.output, "fig-002-reconstruction-errors")
    balances = read_csv(args.results / "balances.csv")
    figure, axes = plt.subplots(2, 1, figsize=(9.6, 9.0))
    for method in LABELS:
        data = [r for r in balances if r["mesh"] == "graded-4"
                and r["source"] == "quadratic" and r["method"] == method]
        data.sort(key=lambda row: int(row["i"]))
        x = [float(r["x"]) for r in data]
        axes[0].plot(x, [float(r["difference"]) for r in data], color=COLORS[method],
                     linestyle=STYLES[method], marker=MARKERS[method], linewidth=2.6,
                     markersize=9, label=LABELS[method])
        axes[1].plot(x, [float(r["cv_residual"]) for r in data], color=COLORS[method],
                     linestyle=STYLES[method], marker=MARKERS[method], linewidth=2.6,
                     markersize=9, label=LABELS[method])
    data = [r for r in balances if r["mesh"] == "graded-4"
            and r["source"] == "quadratic" and r["method"] == "fv"]
    data.sort(key=lambda row: int(row["i"]))
    axes[0].plot([float(r["x"]) for r in data], [float(r["cv_source"]) for r in data],
                 color="#222222", linestyle="none", marker="o", markerfacecolor="none",
                 markeredgewidth=1.7, markersize=17, label="Exact CV source")
    axes[1].axhline(0, color="#777777", linewidth=0.9)
    axes[0].set(xlabel=r"$x_i$", ylabel=r"$Q_{i+1/2}-Q_{i-1/2}$")
    axes[1].set(xlabel=r"$x_i$", ylabel=r"$R_i$")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].legend(loc="lower left", frameon=False)
    for axis in axes:
        axis.set_xlim(0, 0.63)
        axis.grid(alpha=0.22, linewidth=0.7)
    figure.subplots_adjust(left=0.18, right=0.97, bottom=0.09, top=0.96, hspace=0.37)
    save(figure, args.output, "fig-001-control-volume-balances")


if __name__ == "__main__":
    main()
