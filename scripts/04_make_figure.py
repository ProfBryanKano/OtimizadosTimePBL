"""Figure 1 of the paper (labels in Portuguese). Writes figures/figura1.png and figures/figura1.pdf."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if 'Carlito' in f: fm.fontManager.addfont(f)
_fams=[f.name for f in fm.fontManager.ttflist]
_font=next((x for x in ('Calibri','Carlito') if x in _fams),'DejaVu Sans')
plt.rcParams.update({'font.family':_font,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':9.5,'axes.titleweight':'bold','axes.titlelocation':'left'})
ROOT=Path(__file__).resolve().parents[1]
M=pd.read_csv(ROOT/'results'/'raw'/'all_modules.csv');A=pd.read_csv(ROOT/'results'/'summary'/'cohort_summary.csv')
M=M[M.m==4];A4=A[A.m==4]
col={'RND':'#9a9a9a','SNK':'#e08a2c','SA':'#1f5fa8','SA-E':'#7b5ea7','SA-EM':'#1a9e77','SA-EM-ALT':'#c9a227','SA-EM-ADAPT':'#d1495b'}
lab={'RND':'Sorteio','SNK':'Serpentina','SA':'Annealing (equilíbrio)','SA-E':'Annealing + equidade','SA-EM':'Annealing + equidade + memória','SA-EM-ALT':'Memória, equidade alternada','SA-EM-ADAPT':'Memória, equidade adaptativa'}
fmt=FuncFormatter(lambda v,p:('%g'%v).replace('.',','))
fig,ax=plt.subplots(2,2,figsize=(6.7,4.8))
# (a) one module, n=120
a=ax[0,0];m1=M[(M.module==0)&(M.n==120)]
rnd=m1[m1.policy=='RND'].set_index('rep').sd_skill
for p in ['RND','SNK','SA','SA-E']:
    s=m1[m1.policy==p].set_index('rep');g=(1-s.sd_skill/rnd).mean()*100;i=s.iso_any.mean()*100
    a.plot(i,g,'o',ms=8,color=col[p],mec='k',mew=.4)
    off={'RND':(6,3),'SNK':(6,-10),'SA':(-6,6),'SA-E':(7,-3)}[p]
    a.annotate(lab[p],(i,g),textcoords='offset points',xytext=off,fontsize=7.2,ha='right' if off[0]<0 else 'left')
a.set_xlim(-5,75);a.set_ylim(-8,105);a.set_xlabel('Sub-representados isolados no time (%)');a.set_ylabel('Desequilíbrio eliminado (%)')
a.set_title('(a) Um módulo: equidade custa pouco')
# (b) repeats by module n=40
a=ax[0,1];r=M[(M.n==40)&(M.module>0)]
for p in ['SNK','SA-E','SA','RND','SA-EM']:
    s=r[r.policy==p].groupby('module').repeat_share.mean()*100
    a.plot(s.index+1,s.values,'-o',ms=3,lw=1.5,color=col[p],label=lab[p])
a.set_xlabel('Módulo (turma de 40)');a.set_ylabel('Duplas que já trabalharam juntas (%)');a.set_ylim(-3,135);a.set_yticks([0,20,40,60,80,100])
a.set_title('(b) Reotimizar sem memória repete duplas');a.legend(fontsize=6.6,frameon=False,loc='upper left',ncol=2,columnspacing=0.8,handlelength=1.4)
# (c) distinct teammates by class size
a=ax[1,0];w=0.26
for j,p in enumerate(['RND','SA-E','SA-EM']):
    for i,n in enumerate((40,120,400)):
        s=A4[(A4.n==n)&(A4.policy==p)];x=i+(j-1)*w
        a.bar(x,s.distinct_mean.mean(),w,color=col[p],label=lab[p] if i==0 else None);a.plot([x-w/2,x+w/2],[s.distinct_min.mean()]*2,color='k',lw=1.3)
a.axhline(24,color='k',lw=.8,ls=(0,(3,2)));a.text(-0.4,22.6,'Máximo (24)',fontsize=7,ha='left')
a.plot([],[],color='k',lw=1.3,label='Estudante menos conectado')
a.set_xticks(range(3));a.set_xticklabels(['Turma de 40','Turma de 120','Turma de 400']);a.set_ylim(0,34);a.set_yticks([0,5,10,15,20,25]);a.set_ylabel('Colegas distintos após 8 módulos')
a.set_title('(c) Alcance da rede de cada estudante');a.legend(fontsize=6.6,frameon=False,loc='upper left',ncol=2,columnspacing=0.8,handlelength=1.4)
# (d) protected students tradeoff n=40
a=ax[1,1];s40=A4[A4.n==40]
for p in ['RND','SA-E','SA-EM','SA-EM-ADAPT','SA-EM-ALT']:
    s=s40[s40.policy==p];x=s.ever_iso.mean()*100;y=s.deficit_protected.mean()
    a.plot(x,y,'o',ms=8,color=col[p],mec='k',mew=.4)
    off={'RND':(-6,6),'SA-E':(7,-3),'SA-EM':(7,0),'SA-EM-ADAPT':(7,2),'SA-EM-ALT':(-6,6)}[p]
    a.annotate(lab[p].replace('Memória, ','Memória,\n'),(x,y),textcoords='offset points',xytext=off,fontsize=7,ha='right' if off[0]<0 else 'left',va='center')
a.set_xlim(-8,108);a.set_ylim(-0.4,4.2);a.axhline(0,color='k',lw=.5)
a.set_xlabel('Protegidos isolados em algum módulo (%)');a.set_ylabel('Déficit de colegas dos protegidos')
a.set_title('(d) Tensão residual em turmas de 40')
for x in ax.ravel(): x.yaxis.set_major_formatter(fmt)
fig.tight_layout(h_pad=1.2,w_pad=1.4);(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures'/'figura1.png',dpi=300);fig.savefig(ROOT/'figures'/'figura1.pdf')
