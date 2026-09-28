# Manuscript

El documento vigente es [main.pdf](main.pdf), cuya fuente es
[main.tex](main.tex). Incluye las cifras de `numbers.tex`, seis figuras y los
suplementos generados desde `results/egb-extremality.json`,
`results/egb-rotating-profiles.json` y
`results/egb-rotating-resolution-study.json`.

La compilación de referencia en Linux usa Tectonic:

```bash
PYTHONPATH=src python3 experiments/egb_rotating_resolution_study.py  # solo si falta el estudio de resolucion
PYTHONPATH=src python3 manuscript/make_figures.py
python3 manuscript/check_manuscript.py
tectonic -X compile manuscript/main.tex
```

Desde la raíz de BHs:

```powershell
python experiments/egb_rotating_profile_atlas.py   # solo si falta el atlas de perfiles
python manuscript/make_figures.py
python manuscript/check_manuscript.py
powershell -NoProfile -ExecutionPolicy Bypass -File manuscript/build-local.ps1
```

Usar el entorno de la [guía de reproducción](../docs/reproduccion.md).
La compilación requiere MiKTeX y sus paquetes instalados; no los descarga.
El correo está vacío y `jhepstyle.sty` es un estilo local.

La [explicación física en español](explicacion-en-espanol.md) describe el objetivo
y el método. La [guía de resultados](../docs/resultados.md) distingue los valores
de producción de la envolvente de sensibilidad usada en el paper.
El checker estático verifica macros y citas, no la corrección física por sí solo.

Manuscrito, figuras originales y textos: CC BY 4.0. Código: MIT.
