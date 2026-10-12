# Reproducibilidad de los resultados de dinámica caótica del satélite

Este paquete reproduce los resultados principales añadidos al documento ampliado.

## Sistema

Se integra la Ec. (9):

- `dwx/dt = wz*wy/3 - a*wx + wz/sqrt(6)`
- `dwy/dt = -wz*wx + b*wy`
- `dwz/dt = wx*wy - sqrt(6)*wx - c*wz`

Condición inicial utilizada en los barridos: `(0.15, 0.15, 0.15)`.

Sección de Poincaré:

`wz = 0`, con cruce ascendente `dwz/dt > 0`.

## Archivos

- `common.py`: integración, mapa de Poincaré, órbitas periódicas, Jacobianos y firmas de período.
- `01_b_intermitencia_tipo_I.py`: reproduce la evidencia de Tipo I para `b`, incluida la órbita P3, el multiplicador `mu -> +1`, el punto crítico `b_c`, clasificación laminar/burst y prueba de escalamiento.
- `02_a_flip_vs_tipoIII.py`: reproduce el análisis que distingue flip supercrítico de intermitencia Tipo III al variar `a`.
- `03_c_cascada_periodo_doble.py`: reproduce la cascada `P1 -> P2 -> P4 -> P8` al variar `c`, los multiplicadores aproximados y el primer cociente tipo-Feigenbaum.
- `04_generar_figuras_alta_resolucion.py`: rasteriza a 1200 dpi los PDF vectoriales generados.
- `00_run_all.py`: ejecuta todo en secuencia.

## Instalación

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

```bash
python 00_run_all.py
```

Los resultados se guardan en:

- `resultados_b/`
- `resultados_a/`
- `resultados_c/`

## Notas numéricas importantes

1. La estimación de `b_c` se obtiene resolviendo simultáneamente `P^3(q)-q=0` y `det(DP^3-I)=0`. Con tolerancias suficientemente estrictas debe reproducirse un valor próximo a `0.1593355`.
2. La ley `⟨L⟩ ~ (b-b_c)^(-1/2)` se muestra como **prueba de consistencia**. El código no fuerza un ajuste con exponente 1/2, porque los datos finitos y la definición del umbral laminar introducen sensibilidad.
3. Para `a`, un multiplicador cercano a `-1` no se interpreta automáticamente como Tipo III. El código comprueba si existe realmente una secuencia repetitiva laminar-burst-reinyección. El resultado principal es una órbita P2 estable, compatible con flip supercrítico.
4. Para `c`, el primer cociente tipo-Feigenbaum es preasintótico. Para demostrar convergencia a `4.669...` haría falta resolver transiciones adicionales `P8->P16`, `P16->P32`, etc.
5. Los tiempos de ejecución dependen de las tolerancias de `solve_ivp` y de la CPU. El análisis de `b_c` es la parte más costosa.
