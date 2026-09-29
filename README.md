# Rotating black holes at finite Gauss–Bonnet coupling

Código y evidencia del proyecto **Extremality shift of rotating black holes
at finite Gauss–Bonnet coupling**.
La pregunta es cómo cambia la masa extrema a momento angular fijo en gravedad
de Einstein–Gauss–Bonnet, en cinco dimensiones y con dos momentos angulares iguales.

El [manuscrito](manuscript/main.pdf), su [fuente LaTeX](manuscript/main.tex), la [explicación en español](manuscript/explicacion-en-espanol.md) y el [resumen en inglés](https://github.com/ddblanco/BHs/blob/main/short-summary/summary.pdf) son los puntos de entrada.
Las estimaciones del límite extremo son extrapolaciones numéricas; las barras del
paper describen sensibilidad a los ajustes, no intervalos de confianza.

## Contenido

- `src/rotating_bh/`: biblioteca científica y expresiones simbólicas generadas.
- `experiments/`: derivaciones, solvers, benchmarks y corridas de producción.
- `tests/`: controles científicos, negativos, de integridad y del manuscrito vigente.
- `results/`, `artifacts/`: resultados guardados, figuras y procedencia.
- `manuscript/`: paper, figuras, cifras generadas y suplementos.
- `docs/`, `references/`: convenciones, método y referencias bibliográficas.
- `reports/`: dos conjuntos de validación científica, su script y la fuente de una corrida rechazada.
- `prompts/`: registro cronológico de los prompts humanos y el uso de tiempo y de tokens del proyecto.

## Instalación y verificación

Desde esta carpeta, con Python 3.11 o posterior, en PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r environment/requirements-lock.txt
.venv/Scripts/python -m pip install --no-build-isolation --no-deps -e .
.venv/Scripts/python checks/verify_distribution.py
.venv/Scripts/python checks/verify_recorded_digests.py
.venv/Scripts/python -m pytest
.venv/Scripts/python manuscript/check_manuscript.py
```

En Linux/macOS, sustituir `.venv/Scripts/python` por `.venv/bin/python`.
La suite incluye cálculos numéricos: no es una comprobación instantánea.
No se necesita inicializar un repositorio ni ejecutar Git.

La [guía de reproducción](docs/reproduccion.md) separa verificación de archivos
guardados, regeneración del paper y corridas largas. La [guía de resultados](docs/resultados.md)
explica qué partes del JSON original usa el paper vigente y conserva las limitaciones.
El [registro de tests](docs/cambios-tests.md) detalla cada control adaptado, eliminado o nuevo.

## Licencias

El código está bajo **MIT**, copyright © **2026 David Blanco**; véase [LICENSE](LICENSE).
El manuscrito, las figuras originales, los textos y los datos originales de esta
distribución se ofrecen bajo **[Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.
Para atribuirlos, indicar el título del trabajo, 2026, el enlace a la
licencia y las modificaciones realizadas, si las hay.

La licencia propia no cubre obras de terceros. Los artículos citados y la imagen
de la figura 1b de arXiv:1010.0860 no se redistribuyen aquí. La ficha de arXiv enlaza
una licencia de distribución concedida a arXiv, sin autorización general para
redistribuirla por terceros. Véanse las [instrucciones de obtención](references/data/README.md).
Los dos tests que requieren esa imagen se saltan cuando falta; los resultados
históricos de la comparación permanecen disponibles.
