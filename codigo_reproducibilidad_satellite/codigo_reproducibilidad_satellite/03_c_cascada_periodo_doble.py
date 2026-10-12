"""Reproduce la cascada de duplicación de período al variar c."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import poincare_sequence, classify_period, map_jacobian

OUT=Path(__file__).resolve().parent/"resultados_c"
OUT.mkdir(exist_ok=True)
A=0.4; B=0.175

# 1) Barrido de firmas de período
c_scan=np.arange(0.096,0.1531,0.002)
rows=[]
store={}
for c in c_scan:
    _,pts=poincare_sequence(a=A,b=B,c=c,tmax=850,transient=350,rtol=2e-8,atol=2e-10,max_step=.18)
    x=pts[-160:,0]
    p,sig=classify_period(x,tol=2e-4)
    rows.append([c,p,sig[1],sig[2],sig[4],sig[8],sig[16]])
    store[round(float(c),6)]=pts

df=pd.DataFrame(rows,columns=["c","period","R1","R2","R4","R8","R16"])
df.to_csv(OUT/"c_cascada_periodos.csv",index=False)

fig=plt.figure(figsize=(7.0,4.8))
for col,mark in [("R1","o"),("R2","s"),("R4","^"),("R8","d")]:
    plt.semilogy(df.c,df[col],mark+"-",ms=3,label=col)
plt.xlabel(r"$c$"); plt.ylabel("RMS de diferencia de retorno")
plt.title("Cascada de duplicación de período al variar c")
plt.legend(); fig.tight_layout(); fig.savefig(OUT/"c_cascada_periodos.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"c_cascada_periodos.pdf",bbox_inches="tight"); plt.close(fig)

# 2) Umbrales de aparición de P2/P4/P8 en esta resolución
def first_onset(period):
    q=df[df.period==period]
    return float(q.c.iloc[0]) if len(q) else np.nan
c2=first_onset(2); c4=first_onset(4); c8=first_onset(8)
delta=(c4-c2)/(c8-c4) if np.all(np.isfinite([c2,c4,c8])) and c8!=c4 else np.nan
summary=pd.DataFrame([{"c_P2":c2,"c_P4":c4,"c_P8":c8,"delta_est":delta}])
summary.to_csv(OUT/"c_feigenbaum_estimacion.csv",index=False)

# 3) Multiplicadores aproximados del mapa P^k sobre ramas atractoras
#    Se usan pocos puntos cercanos al extremo estable para mantener el cálculo razonable.
branch_ranges={
    1:(0.1040,0.1090),
    2:(0.1320,0.1370),
    4:(0.1400,0.1450),
}
fl=[]
for k,(lo,hi) in branch_ranges.items():
    for c in np.linspace(lo,hi,3):
        _,pts=poincare_sequence(a=A,b=B,c=float(c),tmax=800,transient=400,rtol=2e-8,atol=2e-10,max_step=.16)
        orbit=pts[-k:,:2]
        P=np.eye(2)
        for q in orbit:
            P=map_jacobian(q,k=1,a=A,b=B,c=float(c),h=2e-5)@P
        eig=np.linalg.eigvals(P)
        mu=eig[np.argmin(np.abs(eig+1))]
        fl.append([f"P{k}",c,mu.real,mu.imag])
fl=pd.DataFrame(fl,columns=["branch","c","mu_real","mu_imag"])
fl.to_csv(OUT/"c_floquet_sparse.csv",index=False)

crit=[]
for br in fl.branch.unique():
    q=fl[fl.branch==br].sort_values("c")
    if len(q)>=2:
        m,p=np.polyfit(q.c,q.mu_real,1)
        cc=(-1-p)/m if abs(m)>1e-12 else np.nan
        crit.append([br,cc])
crit=pd.DataFrame(crit,columns=["branch","c_at_mu_minus1"])
crit.to_csv(OUT/"c_floquet_flip_estimates.csv",index=False)

fig=plt.figure(figsize=(7.0,4.8))
for br,mk in [("P1","o"),("P2","s"),("P4","^")]:
    q=fl[fl.branch==br]
    if len(q): plt.plot(q.c,q.mu_real,mk+"-",label=br)
plt.axhline(-1,lw=1)
plt.xlabel(r"$c$"); plt.ylabel(r"Multiplicador próximo a $-1$")
plt.title("Multiplicadores sobre ramas estables")
plt.legend(); fig.tight_layout(); fig.savefig(OUT/"c_floquet_sparse.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"c_floquet_sparse.pdf",bbox_inches="tight"); plt.close(fig)

print(df)
print(summary)
print(fl)
print(crit)
