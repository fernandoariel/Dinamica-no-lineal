"""Ejecuta los tres análisis principales en secuencia."""
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
scripts=["01_b_intermitencia_tipo_I.py","02_a_flip_vs_tipoIII.py","03_c_cascada_periodo_doble.py","04_generar_figuras_alta_resolucion.py"]
for s in scripts:
    print("\n=== Ejecutando",s,"===")
    subprocess.run([sys.executable,str(ROOT/s)],cwd=ROOT,check=True)
print("\nTodos los análisis finalizados.")
