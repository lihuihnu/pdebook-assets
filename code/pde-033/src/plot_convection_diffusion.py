"""Quantitative figures from recorded nodes and analytic reference samples."""
from pathlib import Path
from io import BytesIO
import csv
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve()
if HERE.parents[2].name == "figures":
    DATA = HERE.parents[3]/"experiments/pde-033/results"
    OUT = HERE.parents[1]/"exports"
else:
    DATA = HERE.parents[1]/"results"
    OUT = HERE.parents[1]/"figures"
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":20,
    "axes.labelsize":22,"xtick.labelsize":19,"ytick.labelsize":19,
    "legend.fontsize":18,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":1.1,"lines.linewidth":2.6,"lines.markersize":7})


def read(name):
    with (DATA/(name+".csv")).open() as file:
        return list(csv.DictReader(file))


def save(fig, stem):
    for ext in ["png","pdf"]:
        metadata = {"Software":"pdebook"} if ext == "png" else {
            "Creator":"pdebook","CreationDate":None,"ModDate":None}
        buffer = BytesIO()
        fig.savefig(buffer,format=ext,dpi=240,metadata=metadata,facecolor="white")
        path = OUT/(stem+"."+ext)
        temporary = path.with_suffix("."+ext+".tmp")
        temporary.write_bytes(buffer.getvalue());os.replace(temporary,path)
    plt.close(fig)


def run():
    rows = read("nodes")
    field = [r for r in rows if r["case"] == "smooth-64"]
    n = max(int(r["i"]) for r in field)
    grids = {key:[[0.0]*(n+1) for j in range(n+1)] for key in ["value","error"]}
    for row in field:
        for key in grids:
            grids[key][int(row["j"])][int(row["i"])] = float(row[key])
    fig,axes = plt.subplots(2,1,figsize=(8,9.6))
    fig.subplots_adjust(left=.16,right=.81,bottom=.075,top=.94,hspace=.50)
    limit = math.ceil(100*max(abs(v) for row in grids["error"] for v in row))/100
    for ax,key,letter,label in zip(axes,["value","error"],["(a)","(b)"],["$U$","$U-u_{\\mathrm{exact}}$"]):
        options = {"cmap":"cividis","vmin":0,"vmax":2.1} if key == "value" else {
            "cmap":"RdBu_r","norm":TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)}
        im = ax.imshow(grids[key],origin="lower",interpolation="nearest",
                       extent=[-1/(2*n),1+1/(2*n),-1/(2*n),1+1/(2*n)],
                       aspect="equal",**options)
        ax.set_xlim(0,1);ax.set_ylim(0,1)
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks([0,.25,.5,.75,1])
        ax.set_xlabel("$x$");ax.set_ylabel("$y$")
        levels = [.5,1,1.5,2] if key == "value" else [-.04,-.02]
        lines = ax.contour([i/n for i in range(n+1)],[j/n for j in range(n+1)],
                          grids[key],levels=levels,colors="#303030",linewidths=.65)
        ax.clabel(lines,inline=True,fontsize=15,fmt="%g")
        ax.text(0,1.08,letter,transform=ax.transAxes)
        cbar = fig.colorbar(im,ax=ax,fraction=.055,pad=.045)
        cbar.set_label(label,labelpad=10)
        if key == "error":
            cbar.set_ticks([-.05,-.025,0,.025,.05])
    save(fig,"fig-001-solution-and-error")

    fig,ax = plt.subplots(figsize=(8,5.2))
    fig.subplots_adjust(left=.22,right=.96,bottom=.18,top=.82)
    exact = read("layer_reference")
    ax.plot([float(r["x"]) for r in exact],[float(r["exact"]) for r in exact],
            color="#343434",label="Exact",zorder=3)
    for case,color,marker,linestyle,label in [
        ("layer-upwind","#246078","o","-","Upwind"),
        ("layer-central","#9c4b35","s","--","Central")]:
        subset = [r for r in rows if r["case"]==case and r["j"]=="5"]
        subset.sort(key=lambda r:float(r["x"]))
        ax.plot([float(r["x"]) for r in subset],[float(r["value"]) for r in subset],
                color=color,marker=marker,linestyle=linestyle,label=label,
                markerfacecolor="white",markeredgewidth=1.5,zorder=4)
    ax.axhline(0,color="#b2b2b2",linewidth=1,zorder=1)
    ax.set_xlim(-.025,1.025);ax.set_ylim(-.82,1.12)
    ax.set_xticks([0,.2,.4,.6,.8,1]);ax.set_yticks([-.75,-.5,-.25,0,.25,.5,.75,1])
    ax.set_xlabel("$x$");ax.set_ylabel("$u(x,0.5)$")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.07),ncol=3,frameon=False)
    save(fig,"fig-002-layer-profile")


if __name__ == "__main__":
    run()
