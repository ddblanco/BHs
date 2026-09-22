# Manuscript

El documento vigente es [main.pdf](main.pdf), cuya fuente es
[main.tex](main.tex). Incluye las cifras de `numbers.tex`, tres figuras y los
suplementos generados desde `results/egb-extremality.json`.

Desde la raíz de BHs:

```powershell
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
