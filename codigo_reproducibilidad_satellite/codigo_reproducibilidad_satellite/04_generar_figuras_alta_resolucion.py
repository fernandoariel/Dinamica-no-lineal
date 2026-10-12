"""Convierte PDFs vectoriales de resultados a PNG de alta resolución (1200 dpi)."""
from pathlib import Path
import fitz

ROOT=Path(__file__).resolve().parent
for folder in [ROOT/"resultados_b",ROOT/"resultados_a",ROOT/"resultados_c"]:
    if not folder.exists():
        continue
    for pdf in folder.glob("*.pdf"):
        doc=fitz.open(pdf)
        page=doc[0]
        pix=page.get_pixmap(matrix=fitz.Matrix(1200/72,1200/72),alpha=False)
        out=pdf.with_name(pdf.stem+"_1200dpi.png")
        pix.save(out)
        print(out)
