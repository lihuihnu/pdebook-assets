"""Computed evidence only; all plotted points come from recorded CSV files.

Private: python figures/pde-031/src/plot_verification.py
Public:  python code/pde-031/src/plot_verification.py
"""
from pathlib import Path
from io import BytesIO
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE=Path(__file__).resolve()
if HERE.parents[2].name=='figures':
    DATA=HERE.parents[3]/'experiments/pde-031/results'
    OUT=HERE.parents[1]/'exports'
else:
    DATA=HERE.parents[1]/'results'
    OUT=HERE.parents[1]/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':20,'axes.labelsize':21,
    'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.1,
    'lines.linewidth':2.4,'lines.markersize':7,'pdf.compression':6})
BLUE='#246078';RED='#9c4b35';GRAY='#4f4f4f'


def read(name):
    with (DATA/(name+'.csv')).open() as f:return list(csv.DictReader(f))


def select(rows,model,variant,N=None):
    out=[r for r in rows if r['model']==model and r['variant']==variant
         and (N is None or int(r['N'])==N)]
    return out


def save(fig,stem):
    for ext in ['png','pdf']:
        buf=BytesIO()
        metadata={'Software':'pdebook'} if ext=='png' else {'Creator':'pdebook','CreationDate':None,'ModDate':None}
        fig.savefig(buf,format=ext,dpi=240,metadata=metadata,facecolor='white')
        target=OUT/(stem+'.'+ext);temp=target.with_suffix('.'+ext+'.tmp')
        temp.write_bytes(buf.getvalue());os.replace(temp,target)
    plt.close(fig)


def axes_pair():
    fig,axs=plt.subplots(2,1,figsize=(8,25/3))
    fig.subplots_adjust(left=.17,right=.965,bottom=.075,top=.90,hspace=.64)
    for letter,ax in zip(['(a)','(b)'],axs):
        ax.text(0,1.17,letter,transform=ax.transAxes,fontsize=20)
        ax.grid(True,alpha=.18,linewidth=.8)
    return fig,axs


def mesh_axis(ax):
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    ax.set_xticks([4,8,16,32,64,128]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('$N$');ax.set_xlim(3.6,145)


def run():
    summary,refinement,nodes,balances=[read(n) for n in ['summary','refinement','nodes','balances']]
    fig,axs=axes_pair()
    for model,color,marker,label in [('sine',BLUE,'o','Sine'),('quartic',RED,'s','Quartic')]:
        r=select(summary,model,'correct')
        axs[0].plot([int(t['N']) for t in r],[float(t['error_inf']) for t in r],color=color,marker=marker,label=label)
    mesh_axis(axs[0]);axs[0].set_ylabel('$E_h$')
    axs[0].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    for variant,color,marker,style,label in [('correct',BLUE,'o','-','Correct $E_h$'),('biased_source',RED,'s','-','Biased $E_h$')]:
        r=select(summary,'quartic',variant)
        axs[1].plot([int(t['N']) for t in r],[float(t['error_inf']) for t in r],color=color,marker=marker,linestyle=style,label=label)
    for variant,color,marker,label in [('correct',GRAY,'o','Correct $D_h$'),('biased_source',RED,'x','Biased $D_h$')]:
        r=select(refinement,'quartic',variant)
        axs[1].plot([int(t['N']) for t in r],[float(t['difference_inf']) for t in r],color=color,marker=marker,
            markerfacecolor='white',linestyle='--',label=label)
    mesh_axis(axs[1]);axs[1].set_ylabel('$E_h$ or $D_h$')
    axs[1].legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,columnspacing=.7,handlelength=1.3)
    save(fig,'fig-001-error-and-grid-difference')
    fig,axs=axes_pair()
    for variant,color,marker,style,label in [('correct',BLUE,'o','-','Correct'),('balanced_perturbation',RED,'s','--','Perturbed')]:
        r=select(nodes,'quartic',variant,32)
        axs[0].plot([float(t['x']) for t in r],[float(t['error']) for t in r],color=color,marker=marker,
            markersize=5,linestyle=style,label=label)
        r=select(balances,'quartic',variant,32)
        axs[1].plot([float(t['x']) for t in r],[float(t['local_pde_defect']) for t in r],color=color,marker=marker,
            markersize=5,linestyle=style,label=label)
    for ax in axs:
        ax.set_xlabel('$x$');ax.set_xlim(0,1)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    axs[0].set_ylabel('Nodal error');axs[1].set_ylabel('Local PDE defect')
    save(fig,'fig-002-local-and-global-balance')
    print('Two actual-data figures saved as 1920x2000 PNG and complete vector PDF.')


if __name__=='__main__':run()
