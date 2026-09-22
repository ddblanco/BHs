# Resultados y alcance del paper vigente

El insumo central es `results/egb-extremality.json`. Se conserva sin modificar
sus números, booleanos, hashes ni análisis históricos. El paper es
`manuscript/main.pdf`, generado desde `manuscript/main.tex`.

`manuscript/make_figures.py` usa las soluciones de `walks`, los valores centrales
de `extremals`, los controles estáticos y de vacío, `route_comparison`, el
contraste near-horizon y el lugar de cambio de signo. Exporta las soluciones a
CSV y todas las cantidades citadas a `numbers.tex`.

Los valores centrales extremos conservan el ajuste cúbico en temperatura de
los seis estados más fríos. `manuscript/reanalysis.py` calcula la sensibilidad
a formas alternativas, ventanas de seis a doce puntos y coordenadas `T_H` y
`tau`, incluyendo términos no analíticos. El resultado se guarda en
`manuscript/supplementary/extrapolation-sensitivity.json`. Su envolvente,
combinada con la dispersión original, es la usada en tablas y barras del paper.
No debe sustituirse por el antiguo campo `psi_uncertainty` del JSON de producción.

El mínimo se acota por puntos vecinos; no se presenta la precisión de un
interpolante como localización medida. Las discrepancias de extracción de masa
en el sector estático se muestran como controles numéricos, sin reutilizar una
interpretación descartada del borrador. Las fórmulas perturbativas y el sector
extremo near-horizon publicados son anclas externas, no resultados originales.

Los archivos `reports/revision-manuscript-checks.json` y
`reports/revision-manuscript-fits.json` conservan comprobaciones numéricas y
simbólicas complementarias. Sus nombres se mantienen para conservar sus rutas;
el segundo compara ajustes cúbicos y no sustituye la envolvente del suplemento.

## Resultados negativos conservados

`artifacts/limitations.json` declara las cuatro comprobaciones fallidas del
inventario publicado. Dos pertenecen al resultado de la familia y su figura:
el radio de ergosuperficie calculado `1.102101` no concuerda con el valor
aproximado publicado `1.104` a `alpha_paper=2` bajo la tolerancia fijada.
El contraste externo independiente conserva el mismo rechazo. La cuarta
comprobación es una continuación que no convergió, preservada junto con
`reports/hito-4c-failed-continuation-source.txt`.

Estos rechazos no se borran al pasar la suite: los tests verifican que estén
correctamente registrados. La semilla neuronal y los intentos fallidos siguen
en los resultados, con su fiabilidad medida, sin presentarlos como éxitos.

## Procedencia e integridad

`artifacts/manifest.json` retiene sólo los registros cuyos artefactos se publican,
con comandos, parámetros, comprobaciones y hashes originales. `prompt_refs`
queda vacío porque los registros internos no se publican. La imagen excluida
queda declarada como `external_inputs`; su digest original permanece en
`parameters.source_sha256` y en el resultado de perfiles.

`checks/verify_recorded_digests.py` verifica los hashes históricos disponibles.
Mantiene la excepción preexistente de `docs/egb-rotating.md`: ese texto fue
modificado después de una corrida y sus bytes anteriores no están disponibles.
No se ha recalculado el hash para ocultarlo. La segunda ausencia explícita es
la imagen de arXiv, opcional por licencia; si se obtiene, su hash sí se verifica.

`artifacts/distribution-sha256.json` es otro inventario: registra los bytes de
la distribución final, incluyendo documentación y tests adaptados. Lo verifica
`checks/verify_distribution.py`, que también controla el esquema y las entradas
del manifiesto, la cobertura de resultados y figuras y los rechazos declarados.
No reemplaza la procedencia científica ni es una firma criptográfica.

Los módulos científicos y los experimentos se distribuyen con sus bytes
originales. Por ello algunos comentarios y metadatos mencionan etapas antiguas
o prompts. No son dependencias para ejecutar el código. Los experimentos pueden
volver a escribir esos metadatos históricos al regenerar: conservar la salida
en una copia de trabajo y no confundirla con el inventario de esta entrega.
