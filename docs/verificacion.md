# Verificación de esta distribución — 22 de septiembre de 2026

Se ejecutó la suite completa sobre la copia autónoma con el código de `src/`
de esa copia: **490 tests aprobados, 2 saltados**, en 581,31 segundos.
Los dos saltos corresponden exclusivamente a la imagen externa no redistribuida.
Ambos tests también pasaron con la imagen original en otra copia temporal,
sin incorporarla a la distribución.

Comprobaciones adicionales realizadas:

- Auditoría de distribución, esquema del manifiesto, cobertura de resultados
  y figuras, entradas presentes y cuatro rechazos científicos declarados.
- Verificación de 238 registros de hashes históricos, con las excepciones
  explícitas de la imagen ausente y del documento modificado históricamente.
- Checker del manuscrito: 78 cantidades generadas, 20 referencias.
- Verificaciones de resolución y artefactos de la familia. Esta última sigue
  informando `Scientific closure: False`: se conserva el rechazo científico.
- Regeneración de cifras, CSV de soluciones y sensibilidad en los tests del paper.
- Reproducción de los ajustes complementarios guardados y de la identidad
  simbólica de Wu–Lü; los ajustes coinciden con el JSON publicado.
- Compilación de `manuscript/main.pdf` en tres pasadas: 15 páginas. Se comprobó
  la ausencia de la leyenda de envío a JHEP en el texto extraído del PDF.
- Comparación byte por byte: los 46 módulos científicos, los 19 experimentos
  y los 15 resultados JSON permanecen iguales a las fuentes originales.

Después del cambio final de nombre del PDF y de estilo se repitieron los
controles del manuscrito, distribución, hashes, imagen y repetibilidad del
benchmark Myers–Perry. La compilación usó una caché temporal de MiKTeX; la
primera preparación de fuentes emitió avisos de bloqueo de su base de paquetes,
pero la compilación final de `main.pdf` terminó con código 0.

No se repitieron todas las corridas largas de producción ni se instaló un
entorno nuevo descargando dependencias. Se usó Python 3.11 y el entorno local
disponible, fijando explícitamente la ruta de importación a esta distribución.
Pasar los tests no convierte las extrapolaciones en cotas rigurosas ni elimina
las [limitaciones científicas](resultados.md).

Una revisión independiente de la selección y sus adaptaciones no encontró
problemas bloqueantes. No se ejecutó ningún comando de Git.
