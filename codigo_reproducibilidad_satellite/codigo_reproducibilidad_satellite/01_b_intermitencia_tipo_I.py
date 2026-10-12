"""Reproduce los resultados principales para b: intermitencia Tipo I.

Resultados objetivo:
- aparición de bursts cerca de b~0.15934-0.15940
- órbita P3 estable antes del umbral
- multiplicador dominante de P^3 -> +1
- estimación de b_c por condición det(DP^3-I)=0
- mapa de retorno clasificado y métricas laminares
- comparación del escalamiento con pendiente teórica -1/2 (sin forzar el ajuste)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root

from common import (poincare_sequence, find_periodic_orbit, floquet_multipliers_map,
                    iterate_map, map_jacobian, robust_mad)

OUT = Path(__file__).resolve().parent / "resultados_b"
OUT.mkdir(exist_ok=True)
A = C = 0.4
IC = np.array([0.15,0.15,0.15])

# ------------------------------------------------------------
# 1) Métricas de intermitencia a partir de tiempos de retorno
# ------------------------------------------------------------
# Se toma una referencia periódica por debajo del umbral.
_, pts_ref = poincare_sequence(a=A,b=0.15930,c=C,tmax=2600,transient=700)
te_ref, _ = poincare_sequence(a=A,b=0.15930,c=C,tmax=2600,transient=700)
Tref = np.diff(te_ref)
T0 = np.median(Tref)
sigma_T = robust_mad(Tref)
# Umbral robusto: suficientemente amplio para no confundir el ciclo P3 con bursts.
threshold = max(3.0*sigma_T, 0.02)

b_scan = np.array([0.159200,0.159250,0.159300,0.159325,0.159350,
                   0.159375,0.159400,0.159425,0.159450])
rows=[]
for b in b_scan:
    te, pts = poincare_sequence(a=A,b=b,c=C,tmax=2700,transient=700)
    T = np.diff(te)
    burst = np.abs(T-T0) > threshold
    lam = ~burst
    runs=[]; r=0
    for flag in lam:
        if flag:
            r += 1
        elif r:
            runs.append(r); r=0
    if r: runs.append(r)
    runs=np.asarray(runs,dtype=float)
    rows.append([b, np.std(T)/np.mean(T), burst.mean(),
                 runs.mean() if len(runs) else np.nan,
                 np.median(runs) if len(runs) else np.nan,
                 runs.max() if len(runs) else np.nan,
                 np.median(T), len(T)])

metrics = pd.DataFrame(rows,columns=["b","CV_intervals","burst_fraction",
    "mean_laminar_cycles","median_laminar_cycles","max_laminar_cycles",
    "median_period","n_intervals"])
metrics.to_csv(OUT/"intermitencia_metricas_b.csv",index=False)

# ------------------------------------------------------------
# 2) Mapa de Poincaré clasificado en b=0.1594
# ------------------------------------------------------------
b_map = 0.1594
te, pts = poincare_sequence(a=A,b=b_map,c=C,tmax=4500,transient=900)
x = pts[:,0]
T = np.diff(te)
burst_T = np.abs(T-T0) > threshold
# Asignar clasificación al punto n por su intervalo de salida T_n
is_burst = np.zeros(len(x)-1,dtype=bool)
is_burst[:] = burst_T

fig=plt.figure(figsize=(7.2,5.5))
plt.plot(x[:-1][~is_burst],x[1:][~is_burst],"o",ms=3,mfc="none",label="Fase laminar")
plt.plot(x[:-1][is_burst],x[1:][is_burst],"x",ms=5,label="Burst caótico")
lo=min(x.min(),x[1:].min()); hi=max(x.max(),x[1:].max())
plt.plot([lo,hi],[lo,hi],"--",lw=1,label=r"$x_{n+1}=x_n$")
plt.xlabel(r"$\omega_{x,n}$"); plt.ylabel(r"$\omega_{x,n+1}$")
plt.title(r"Mapa de retorno de Poincaré: $b=0.1594$")
plt.legend(); fig.tight_layout()
fig.savefig(OUT/"poincare_intermitencia_clasificado.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"poincare_intermitencia_clasificado.pdf",bbox_inches="tight")
plt.close(fig)

# Secuencia de intervalos de retorno
fig=plt.figure(figsize=(7.2,4.6))
plt.plot(np.arange(len(T)),T,".",ms=3)
plt.axhline(T0+threshold,lw=1)
plt.axhline(T0-threshold,lw=1)
plt.xlabel("Iteración de Poincaré n"); plt.ylabel(r"$T_n=t_{n+1}-t_n$")
plt.title(r"Secuencia de retornos y bursts para $b=0.1594$")
fig.tight_layout(); fig.savefig(OUT/"secuencia_intermitencia_b.png",dpi=600,bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# 3) Continuación local de la órbita P3 y multiplicadores
# ------------------------------------------------------------
# Semilla obtenida del atractor periódico en b=0.15930.
_, pts_seed = poincare_sequence(a=A,b=0.15930,c=C,tmax=1600,transient=900)
q = find_periodic_orbit(pts_seed[-1,:2],3,a=A,b=0.15930,c=C)

b_vals = np.array([0.15933000,0.15933100,0.15933200,0.15933300,0.15933400,
                   0.15933450,0.15933500,0.15933520,0.15933540,0.15933542,
                   0.15933544,0.15933546,0.15933548,0.15933550,0.15933552])
fl_rows=[]
for b in b_vals:
    q = find_periodic_orbit(q,3,a=A,b=b,c=C)
    eig,M = floquet_multipliers_map(q,3,a=A,b=b,c=C,h=1e-6)
    mu = eig[np.argmax(np.real(eig))]
    fl_rows.append([b,q[0],q[1],np.real(mu),np.imag(mu),np.linalg.det(M),np.trace(M)])

floq = pd.DataFrame(fl_rows,columns=["b","wx_section","wy_section","mu_dom_real",
                                    "mu_dom_imag","det_DP3","trace_DP3"])
floq.to_csv(OUT/"multiplicador_periodo3.csv",index=False)

fig=plt.figure(figsize=(7.0,4.8))
plt.plot(floq.b,floq.mu_dom_real,"o-")
plt.axhline(1,lw=1)
plt.xlabel(r"$b$"); plt.ylabel(r"Multiplicador dominante de $DP^3$")
plt.title("Aproximación del multiplicador a +1")
fig.tight_layout(); fig.savefig(OUT/"multiplicador_periodo3.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"multiplicador_periodo3.pdf",bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------
# 4) Resolver el punto crítico: P^3(q)-q=0 y det(DP^3-I)=0
# ------------------------------------------------------------
# Variables desconocidas: qx,qy,b
q0 = floq[["wx_section","wy_section"]].iloc[-1].to_numpy()
b0 = floq.b.iloc[-1]

def critical_system(z):
    qx,qy,b = z
    q=np.array([qx,qy])
    F = iterate_map(q,3,a=A,b=b,c=C)-q
    M = map_jacobian(q,3,a=A,b=b,c=C,h=7e-7)
    g = np.linalg.det(M-np.eye(2))
    return np.r_[F,g]

crit = root(critical_system,np.r_[q0,b0],tol=1e-9)
if not crit.success:
    print("ADVERTENCIA: solve crítico no convergió completamente:",crit.message)
qx,qy,bc = crit.x
Mc = map_jacobian([qx,qy],3,a=A,b=bc,c=C,h=7e-7)
eigc=np.linalg.eigvals(Mc)
np.savetxt(OUT/"critical_DP3_matrix.txt",Mc)
with open(OUT/"critical_point.txt","w",encoding="utf-8") as fh:
    fh.write(f"bc = {bc:.12f}\n")
    fh.write(f"q_c = ({qx:.12f}, {qy:.12f})\n")
    fh.write(f"eig(DP3) = {eigc}\n")
    fh.write(f"det(DP3-I) = {np.linalg.det(Mc-np.eye(2)):.6e}\n")

print(f"b_c estimado = {bc:.12f}")
print("q_c =",qx,qy)
print("multiplicadores críticos =",eigc)

# ------------------------------------------------------------
# 5) Tangencia local usando el autovector asociado a mu~+1
# ------------------------------------------------------------
w,v=np.linalg.eig(Mc)
j=np.argmin(np.abs(w-1))
vec=np.real(v[:,j]); vec/=np.linalg.norm(vec)
ss=np.linspace(-0.004,0.004,121)
Pproj=[]
for s in ss:
    q=np.array([qx,qy])+s*vec
    qp=iterate_map(q,3,a=A,b=bc,c=C)
    Pproj.append(np.dot(qp-np.array([qx,qy]),vec))
Pproj=np.asarray(Pproj)
coef=np.polyfit(ss,Pproj-ss,2)

fig=plt.figure(figsize=(7.0,4.8))
plt.plot(ss,Pproj,label=r"$P^3(s)$")
plt.plot(ss,ss,label=r"$s$")
plt.xlabel(r"$s$"); plt.ylabel(r"$P^3(s)$")
plt.title("Tangencia local del tercer retorno en el umbral")
plt.legend(); fig.tight_layout()
fig.savefig(OUT/"P3_tangencia_local.png",dpi=600,bbox_inches="tight")
fig.savefig(OUT/"P3_tangencia_local.pdf",bbox_inches="tight"); plt.close(fig)

with open(OUT/"normal_form_fit.txt","w",encoding="utf-8") as fh:
    fh.write("Ajuste de P3(s)-s = c2*s^2+c1*s+c0\n")
    fh.write(str(coef)+"\n")

# ------------------------------------------------------------
# 6) Escalamiento: prueba de consistencia con gamma=1/2
# ------------------------------------------------------------
# Usar sólo puntos con bursts observados para evitar datos censurados.
ms = metrics[(metrics.b>bc)&(metrics.burst_fraction>0)&np.isfinite(metrics.mean_laminar_cycles)]
if len(ms)>=2:
    eps = ms.b.to_numpy()-bc
    L=ms.mean_laminar_cycles.to_numpy()
    # Referencia -1/2 anclada al primer punto: no es un ajuste forzado.
    ref=L[0]*(eps/eps[0])**(-0.5)
    fig=plt.figure(figsize=(7.0,4.8))
    plt.loglog(eps,L,"o",label="Datos numéricos")
    plt.loglog(eps,ref,"-",label=r"Referencia Tipo I: $\gamma=1/2$")
    plt.xlabel(r"$b-b_c$"); plt.ylabel(r"$\langle L\rangle$")
    plt.title("Escalamiento laminar: prueba de consistencia")
    plt.legend(); fig.tight_layout()
    fig.savefig(OUT/"escalamiento_laminar_referencia.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUT/"escalamiento_laminar_referencia.pdf",bbox_inches="tight")
    plt.close(fig)

print("Resultados guardados en",OUT)
