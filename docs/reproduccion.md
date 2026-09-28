# Reproducción autónoma

Ejecutar desde la raíz de BHs. Se necesita Python >=3.11; las dependencias de
referencia están fijadas en `environment/requirements-lock.txt`. El registro
`dependency-downloads.json` identifica las distribuciones usadas originalmente;
no contiene los paquetes instalados. Para el PDF se requiere además MiKTeX con
los paquetes LaTeX necesarios ya instalados. Ningún paso requiere Git.

Las notas técnicas `convenciones.md`, `myers-perry.md`, `static-egb.md`,
`vacuum-bvp.md` y `egb-rotating.md` conservan sus bytes históricos porque forman
parte de la procedencia de las corridas. Algunas remiten a antiguos informes o
scripts `checks/hito-*.ps1` que se excluyen de esta distribución. Para ejecutar
o reproducir el proyecto, usar los comandos de esta guía; esos enlaces antiguos
no son requisitos de ejecución.

## Entorno

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r environment/requirements-lock.txt
.venv/Scripts/python -m pip install --no-build-isolation --no-deps -e .
.venv/Scripts/python -m rotating_bh.cli environment --json
```

En Linux/macOS usar `.venv/bin/python`. En los ejemplos siguientes `python`
significa el intérprete de ese entorno (activar el entorno o usar su ruta).

## Comprobaciones de archivos guardados

```powershell
python checks/verify_distribution.py
python checks/verify_recorded_digests.py
python -m rotating_bh.cli validate-provenance artifacts/manifest.json
python manuscript/check_manuscript.py
python experiments/egb_rotating_convergence.py --verify-only
python experiments/egb_rotating_family_artifacts.py --verify-only
python -m pytest
```

Los primeros cuatro comandos son rápidos. La suite completa resuelve problemas
numéricos y puede tardar varios minutos. Dos tests de imagen se saltan en la
distribución sin el insumo externo; ver [cómo obtenerlo](../references/data/README.md).
Los demás tests conservan tolerancias y controles negativos.

El inventario de distribución comprueba la entrega tal como fue publicada y
no se actualiza automáticamente. Una regeneración puede cambiarlo legítimamente,
por ejemplo por fechas embebidas en los PDF; eso no autoriza a reemplazar hashes
históricos. Hacer las corridas que sobrescriben resultados en una copia de trabajo.

## Reconstrucción del paper con resultados guardados

```powershell
python manuscript/make_figures.py
python manuscript/check_manuscript.py
powershell -NoProfile -ExecutionPolicy Bypass -File manuscript/build-local.ps1
```

El primer comando regenera `numbers.tex`, cuatro figuras PDF, el CSV de soluciones
y la auditoría de sensibilidad. Lee `results/egb-extremality.json` y
`results/egb-rotating-profiles.json`, y no modifica ninguno de los dos. El atlas
de perfiles se regenera, cuando hace falta, con
`python experiments/egb_rotating_profile_atlas.py`; esa corrida sí resuelve el
problema no lineal en las trece constantes de acoplamiento y tarda.
El script de compilación ejecuta tres pasadas de LaTeX y produce
`manuscript/main.pdf`, con instalación automática desactivada. No se
incluye la versión anterior del paper. `jhepstyle.sty` es el estilo local.

## Corridas largas

Los módulos simbólicos generados ya están incluidos. Sólo para volver a derivarlos:

```powershell
python experiments/derive_vacuum.py
python experiments/derive_static_egb.py
python experiments/derive_egb_rotating.py
python experiments/derive_near_horizon.py
```

Benchmarks y validación escalonada:

```powershell
python experiments/bvp_solver_spike.py
python experiments/myers_perry_benchmark.py
python experiments/myers_perry_bvp.py
python experiments/static_egb_bvp.py
python experiments/egb_rotating_bvp.py
python experiments/egb_rotating_cross_check.py
python experiments/egb_rotating_convergence.py
```

Familia rotante y contrastes derivados, respetando este orden:

```powershell
python experiments/egb_rotating_family.py
python experiments/egb_rotating_family_artifacts.py
python experiments/egb_rotating_external_routes.py
python experiments/egb_rotating_external_contrasts.py
# Sólo después de obtener la imagen externa:
python experiments/egb_rotating_paper_profiles.py
```

Estudios de semillas, respuesta al acoplamiento y extremalidad:

```powershell
python experiments/egb_rotating_neural_seed.py
python experiments/egb_rotating_gb_potential.py
python experiments/egb_extremality.py
python manuscript/make_figures.py
python manuscript/check_manuscript.py
```

La última corrida es la producción central del paper; puede ser costosa.
El estudio de semillas es una validación complementaria y no una condición
para leer los resultados guardados. Las comprobaciones complementarias son:

```powershell
python reports/revision-manuscript-checks.py --saved-only
python reports/revision-manuscript-checks.py
```

La primera reproduce ajustes guardados y una especialización simbólica; la
segunda vuelve a resolver estados a tres resoluciones. Ambos escriben sus
propios JSON de validación.

Algunos experimentos terminan con código 1 cuando registran un rechazo científico.
Leer sus comprobaciones y distinguir ese caso de un traceback o fallo de ejecución.
Las [limitaciones conservadas](resultados.md) explican los rechazos de referencia.
La fuente histórica de la continuación fallida se conserva como evidencia,
no se ejecuta como parte de esta secuencia.

Si se modifica un resultado, deben regenerarse los productos que lo toman como
entrada antes de verificar los hashes. Una comprobación de integridad no prueba
la exactitud física; los tests de ecuaciones, límites, convergencia y controles
negativos aportan evidencia independiente.
