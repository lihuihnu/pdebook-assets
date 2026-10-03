"""Two quantitative figures, plotted only from recorded computation files."""
from pathlib import Path
from io import BytesIO
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
if HERE.parents[2].name == "figures":
    DATA = HERE.parents[3]/"experiments/pde-032/results"
    OUT = HERE.parents[1]/"exports"
else:
    DATA = HERE.parents[1]/"results"
    OUT = HERE.parents[1]/"figures"
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":20,
    "axes.labelsize":21,"xtick.labelsize":18,"ytick.labelsize":18,
    "legend.fontsize":17,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":1.1,"lines.linewidth":2.4,"lines.markersize":6})
BLUE, RED, GREEN, GRAY = "#246078", "#9c4b35", "#49664a", "#6a6a6a"


def read(name):
    with (DATA/(name+".csv")).open() as f:return list(csv.DictReader(f))


def save(fig, stem):
    for ext in ["png","pdf"]:
        metadata = {"Software":"pdebook"} if ext == "png" else {"Creator":"pdebook","CreationDate":None,"ModDate":None}
        buf = BytesIO()
        fig.savefig(buf,format=ext,dpi=240,metadata=metadata,facecolor="white")
        target = OUT/(stem+"."+ext)
        temp = target.with_suffix("."+ext+".tmp")
        temp.write_bytes(buf.getvalue());os.replace(temp,target)
    plt.close(fig)


def run():
    heat = [r for r in read("heat_summary") if r["group"] == "sweep"]
    fig, axs = plt.subplots(2,1,figsize=(8,9.6))
    fig.subplots_adjust(left=.17,right=.965,bottom=.075,top=.89,hspace=.68)
    for letter,ax in zip(["(a)","(b)"],axs):
        ax.text(0,1.23,letter,transform=ax.transAxes)
        ax.grid(True,alpha=.18,linewidth=.8)
    for N,color,marker,style in [(16,GRAY,"o","-"),(32,RED,"s","--"),(64,BLUE,"^","-."),(128,GREEN,"D",":")]:
        rows = sorted([r for r in heat if int(r["N"]) == N],key=lambda r:float(r["dt"]))
        axs[0].loglog([float(r["dt"]) for r in rows],[float(r["error_inf"]) for r in rows],
            color=color,marker=marker,linestyle=style,label=f"$N={N}$")
    axs[0].set_xlabel("$\\Delta t$");axs[0].set_ylabel("$E(T)$")
    axs[0].legend(loc="lower center",bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)
    axs[1].loglog([int(r["work_units"]) for r in heat],[float(r["error_inf"]) for r in heat],
        linestyle="none",marker="o",color=GRAY,alpha=.55,markersize=4,label="All candidates")
    frontier, best_error = [], float("inf")
    for r in sorted(heat,key=lambda r:(int(r["work_units"]),float(r["error_inf"]))):
        error = float(r["error_inf"])
        if error < best_error:
            frontier.append(r);best_error = error
    axs[1].loglog([int(r["work_units"]) for r in frontier],[float(r["error_inf"]) for r in frontier],
        color=BLUE,marker="o",label="Lower envelope")
    target = .0005
    best = min([r for r in heat if float(r["error_inf"]) <= target],key=lambda r:int(r["work_units"]))
    axs[1].loglog([int(best["work_units"])],[float(best["error_inf"])],
        marker="D",color=RED,markersize=10,linestyle="none",label="Least $W$, $E\\leq\\epsilon$")
    axs[1].axhline(target,color="black",linestyle="--",linewidth=1.2,label="$\\epsilon=5\\cdot10^{-4}$")
    axs[1].set_xlabel("$W=(N-1)M$");axs[1].set_ylabel("$E(T)$")
    axs[1].legend(loc="lower center",bbox_to_anchor=(.5,1.015),ncol=2,frameon=False,
        columnspacing=.8,handlelength=1.4)
    save(fig,"fig-001-error-and-work")
    profiles = [r for r in read("mesh_profiles") if abs(float(r["target"])-.005) < 1e-15]
    summaries = [r for r in read("space_summary") if abs(float(r["target"])-.005) < 1e-15]
    final_steps = {r["variant"]:int(r["iteration"]) for r in summaries}
    nodes = [r for r in read("mesh_nodes") if abs(float(r["target"])-.005) < 1e-15
        and int(r["iteration"]) == final_steps[r["variant"]]]
    fig, axs = plt.subplots(2,1,figsize=(8,8),gridspec_kw={"height_ratios":[3,1.25]})
    fig.subplots_adjust(left=.18,right=.965,bottom=.09,top=.88,hspace=.62)
    for letter,ax in zip(["(a)","(b)"],axs):
        ax.text(0,1.18,letter,transform=ax.transAxes)
        ax.set_xlim(-.02,1.02);ax.set_xlabel("$x$")
    for variant,color,style,marker,label,y in [
        ("adaptive",BLUE,"-","o","Adaptive",1),
        ("uniform_same_count",RED,"--","s","Uniform",0)]:
        rows = [r for r in profiles if r["variant"] == variant]
        axs[0].plot([float(r["x"]) for r in rows],[float(r["error"]) for r in rows],
            color=color,linestyle=style,label=label)
        xn = [float(r["x"]) for r in nodes if r["variant"] == variant]
        axs[1].plot(xn,[y]*len(xn),linestyle="none",marker=marker,color=color,
            markersize=7,markerfacecolor=color if variant=="adaptive" else "white")
        axs[1].hlines(y,0,1,color=color,linewidth=1)
    axs[0].axhline(.005,color="black",linestyle=":",linewidth=1.3,label="$\\epsilon_s$")
    axs[0].set_ylabel("$u_h(x)-x^4$")
    axs[0].ticklabel_format(axis="y",style="sci",scilimits=(0,0),useMathText=True)
    axs[0].grid(True,alpha=.18)
    axs[0].legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3,frameon=False,
        columnspacing=.75,handlelength=1.35)
    axs[1].set_ylim(-.6,1.6);axs[1].set_yticks([0,1],["Uniform","Adaptive"])
    axs[1].spines["left"].set_visible(False);axs[1].tick_params(axis="y",length=0)
    save(fig,"fig-002-local-refinement")
    print("Two computed CSV figures saved: 1920x2304 and 1920x1920 PNG, complete vector PDFs.")


if __name__ == "__main__":run()
