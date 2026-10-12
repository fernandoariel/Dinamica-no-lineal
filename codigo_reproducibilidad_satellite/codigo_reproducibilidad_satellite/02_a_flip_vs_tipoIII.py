"""Reproduce el análisis para a: flip supercrítico vs intermitencia Tipo III."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import poincare_sequence, classify_period

OUT=Path(__file__).resolve().parent/"resultados_a"
OUT.mkdir(exist_ok=True)
B=0.175; C=0.4

# 1) Barrido de firmas de período
a_scan=np.array([0.3676,0.3677,0.3678,0.3680,0.3690,0.3710,0.3730,0.3740,0.3750,0.3760])
rows=[]
for a in a_scan:
    _,pts=poincare_sequence(a=a,b=B,c=C,tmax=1000,transient=400,rtol=2e-8,atol=2e-10,max_step=.16)
    x=pts[-160:,0]
    p,sig=classify_period(x,tol=1e-4)
    rows.append([a,p,sig[1],sig[2],sig[4],sig[8],np.std(x)])

df=pd.DataFrame(rows,columns=["a","period","R1","R2","R4","R8","std_x"])
df.to_csv(OUT/"a_tipoIII_vs_perioddoubling.csv",index=False)

fig=plt.figure(figsize=(7.0,4.8))
for col,mark in [("R1","o"),("R2","s"),("R4","^"),("R8","d")]:
    plt.semilogy(df.a,df[col],mark+"-",label=col)
plt.xlabel(r"$a$"); plt.ylabel("RMS de diferencia de retorno")
plt.title("Firmas de período cerca de la transición en a")
plt.legend(); fig.tight_layout(); fig.savefig(OUT/"a_firmas_periodo.png",dpi=600,bbox_inches="tight")
plt.close(fig)

# 2) Mapa local alrededor de la P2 inmediatamente después del primer flip
a0=0.3680
_,pts=poincare_sequence(a=a0,b=B,c=C,tmax=1200,transient=400,rtol=2e-8,atol=2e-10,max_step=.14)
x=pts[:,0]

# 2-means en 1D para localizar los dos puntos P2
c1,c2=np.percentile(x,[25,75])
for _ in range(60):
    lab=np.abs(x-c1)<=np.abs(x-c2)
    n1=x[lab].mean() if lab.any() else c1
    n2=x[~lab].mean() if (~lab).any() else c2
    if abs(n1-c1)+abs(n2-c2)<1e-13: break
    c1,c2=n1,n2
centers=np.sort([c1,c2])
d=np.minimum(np.abs(x-centers[0]),np.abs(x-centers[1]))
mad=np.median(np.abs(d-np.median(d)))
delta=max(5*1.4826*mad,2e-5)
lam=d<=delta

rein=[]; burst_blocks=[]; lam_runs=[]
i=0
while i<len(lam):
    if lam[i]:
        j=i
        while j<len(lam) and lam[j]: j+=1
        lam_runs.append(j-i); i=j
    else:
        j=i
        while j<len(lam) and not lam[j]: j+=1
        burst_blocks.append(j-i)
        if i>0 and j<len(lam): rein.append((x[j-1],x[j]))
        i=j
rein=np.asarray(rein) if rein else np.empty((0,2))

summary=pd.DataFrame([{
    "a":a0,"P2_center_1":centers[0],"P2_center_2":centers[1],
    "laminar_halfwidth":delta,"laminar_fraction":lam.mean(),
    "n_reinjections":len(rein),"n_burst_blocks":len(burst_blocks),
    "mean_laminar_run":np.mean(lam_runs) if lam_runs else np.nan,
}])
summary.to_csv(OUT/"a_intervalo_laminar.csv",index=False)

fig=plt.figure(figsize=(7.0,4.8))
plt.plot(x[:-1],x[1:],".",ms=3,label="retornos")
if len(rein): plt.plot(rein[:,0],rein[:,1],"x",ms=6,label="reinyecciones")
xx=np.linspace(x.min(),x.max(),200)
plt.plot(xx,xx,lw=1,label="diagonal")
for cc in centers: plt.axvspan(cc-delta,cc+delta,alpha=.15)
plt.xlabel(r"$\omega_{x,n}$"); plt.ylabel(r"$\omega_{x,n+1}$")
plt.title(r"Mapa local alrededor de la órbita P2, $a=0.3680$")
plt.legend(); fig.tight_layout(); fig.savefig(OUT/"a_mapa_local_P2.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"a_mapa_local_P2.pdf",bbox_inches="tight"); plt.close(fig)

fig=plt.figure(figsize=(7.0,4.6))
N=min(500,len(x)); plt.plot(np.arange(N),x[:N],lw=.8)
for cc in centers: plt.axhline(cc,lw=1)
plt.xlabel("Iteración de Poincaré"); plt.ylabel(r"$\omega_x$")
plt.title("Secuencia post-flip: órbita P2 estable")
fig.tight_layout(); fig.savefig(OUT/"a_secuencia_P2.png",dpi=600,bbox_inches="tight"); plt.close(fig)

print(df)
print(summary)
print("Interpretación: si la fracción laminar ~1 y hay muy pocas reinyecciones, la dinámica es P2 estable, no Tipo III desarrollada.")
