# Uso de tiempo y de tokens

Medición del costo total del proyecto, tomado como un todo: desde el primer prompt
del 4 de septiembre de 2026 hasta la última sesión recuperable del 28 de septiembre,
cubriendo sus tres directorios (`GW-AI-course/project` con Codex y Claude Code,
el worktree `GW-AI-course/project-codex`, y este repositorio `BHs`).

El registro cronológico seleccionado de esta edición está en
[`prompt-log-cronologico.md`](prompt-log-cronologico.md). Las métricas aquí presentadas
corresponden a la actividad del proyecto completo. El registro cronológico documenta 79 prompts.

---

## 1. Totales

| Magnitud | Valor |
|---|---|
| Ventana calendario | 04/09/2026 → 28/09/2026 (25 días; 13 con actividad) |
| **Tiempo activo con agentes** | **36,8 h** |
| **Tokens totales procesados** | **1.415.508.752** (1,42 mil millones) |
| **Tokens de salida generados** | **5.942.462** (≈ 24 MB de texto) |
| Prompts documentados en el registro cronológico | 79 |
| Sesiones | 16 de Claude Code + 11 hilos de Codex (28 archivos de rollout, porque cada subagente abre el suyo) |
| Mensajes de asistente (Claude Code) | 3.871, con 1.215 bloques de razonamiento |
| Llamadas a herramientas | 2.111 en Claude Code (1.556 de ellas a Bash) + 1.064 en Codex (894 `exec`, 17 `spawn_agent`) |

Reparto por herramienta:

| | Tokens procesados | Tokens de salida |
|---|---:|---:|
| Claude Code | 1.319.141.751 (93 %) | 5.353.961 (90 %) |
| Codex (GPT-5) | 96.367.001 (7 %) | 588.501 (10 %) |

Dentro de Claude Code, 1.301.219.499 tokens — el 99 % de su total — son lecturas de
caché: el contexto se reenvía entero en cada llamada a herramienta. La escritura de
caché suma 12.560.581 y la entrada no cacheada, 7.710. Por modelo: 879.709.449 con
Opus 5 y 439.432.302 con Sonnet 5.

**Los dos sistemas de contabilidad no son comparables entre sí.** Codex trabajó con
una ventana de 258 k y turnos más largos; Claude Code, con ventana de 1 M y una
llamada a herramienta por paso. Por eso Codex aparece con 10–17 M de tokens por hora
y Claude Code con 35–62 M. Las dos magnitudes que sí se comparan entre agentes son el
**tiempo de reloj** y los **tokens de salida**.

### Cómo se midió

- Fuente: los transcriptos locales. Claude Code informa `usage` por mensaje
  (`~/.claude/projects/*/*.jsonl`); Codex informa `total_token_usage` acumulado por
  sesión (`~/.codex/sessions/**/rollout-*.jsonl`).
- El «tiempo activo» suma los intervalos entre eventos consecutivos de cualquiera de
  los dos agentes, cortando cuando hay más de 20 minutos sin eventos. El criterio es
  arbitrario y la cifra es sensible a él: 30,2 h con umbral de 10 min, 34,8 h con
  15 min, 36,8 h con 20 min, 39,3 h con 30 min, 42,7 h con 45 min. Todo lo que sigue
  usa 20 min.
- Una sesión de Claude Code (`d663fab2`) quedó contenida dentro de otra por
  bifurcación; se dedujo por identificador de mensaje para no contarla dos veces.

### Qué no está contado

- El tiempo humano de lectura, de decisión y de escritura de los prompts.
- El tiempo de cómputo de las corridas numéricas largas que no pasan por el modelo:
  los barridos de los Hitos 4C, 6 y 8 corrieron en segundo plano durante horas, y la
  suite de aceptación del Hito 4C sola tarda 236 s.
- Cualquier sesión borrada del disco. El registro de los agentes menciona al menos
  una que se cerró sin aviso (el informe de árbitro del Hito 8 hubo que recuperarlo
  del transcripto porque no se había guardado como archivo).

---

## 2. Reparto por hito

| Bloque | Horas | % | Tokens procesados | % | Tokens de salida | % |
|---|---:|---:|---:|---:|---:|---:|
| Hitos 0–3 — reproducción de la literatura conocida | 4,4 | 12 % | 101.558.821 | 7 % | 747.126 | 13 % |
| **Hito 4 (4A + 4B + 4C) — agujero negro rotante en EGB** | **11,2** | **31 %** | **507.954.588** | **36 %** | **2.271.301** | **38 %** |
| Hito 5 — semilla neuronal | 3,7 | 10 % | 118.533.656 | 8 % | 328.661 | 6 % |
| Hito 6 — potencial conjugado `Psi_GB` | 4,5 | 12 % | 168.757.196 | 12 % | 305.427 | 5 % |
| Hito 7 — entrega reproducible | 1,9 | 5 % | 68.134.005 | 5 % | 346.788 | 6 % |
| Presentaciones y auditorías intermedias | 0,4 | 1 % | 18.708.750 | 1 % | 208.010 | 4 % |
| **Hito 8 — medición de `Psi_GB`, manuscrito y arbitraje** | **8,7** | **24 %** | **408.181.207** | **29 %** | **1.587.772** | **27 %** |
| Publicación en GitHub y revisiones finales | 1,8 | 5 % | 23.680.529 | 2 % | 147.377 | 2 % |
| **Total** | **36,6** | | **1.415.508.752** | | **5.942.462** | |

Desglose fino del Hito 4, que el ROADMAP partió en tres a mitad de camino
precisamente porque no entraba en uno:

| Sub-bloque | Horas | Tokens | Salida |
|---|---:|---:|---:|
| Revisión AstraCheck + 4A/4B: ecuaciones EGB rotantes y solver espectral | 5,9 | 369.177.618 | 1.431.711 |
| 4B: contraste adaptativo en el worktree de Codex | 2,0 | 27.332.188 | 171.283 |
| 4C: familia, observables, primera ley, contraste externo | 3,3 | 111.444.782 | 668.307 |

Por día (hora local, UTC−3):

| Día | Horas activas |
|---|---:|
| 04/09 | 1,9 |
| 05/09 | 1,5 |
| 06/09 | 0,0 |
| 08/09 | 0,1 |
| 09/09 | 6,9 |
| 10/09 | 2,3 |
| 11/09 | 9,2 |
| 12/09 | 4,4 |
| 14/09 | 6,1 |
| 15/09 | 2,5 |
| 18/09 | 0,0 |
| 22/09 | 1,6 |
| 28/09 | 0,2 |

Los dos picos de consumo tocaron los límites de las cuentas: el 11/09 Claude Code
agotó el límite de uso dos veces durante el Hito 4C y el Hito 5, y Codex llegó al
72 % de su cuota semanal el 05/09 y al 76 % el 22/09.

---

## 3. Los tres bloques que más consumieron

El orden es el mismo con las tres métricas —reloj, tokens procesados y tokens de
salida—, así que no depende de cuál se elija.

### 1.º — Hito 4: el agujero negro rotante en EGB · 11,2 h · 508 M tokens · 2,27 M de salida

Derivar las ecuaciones del ansatz rotante con Gauss-Bonnet, resolverlas, y recorrer
la familia de soluciones con sus observables. Lo que lo hizo caro está documentado:

- La derivación tensorial directa resultó **intratable simbólicamente**: Christoffel
  y Riemann de la métrica no diagonal no terminaban. Hubo que pasar a la acción
  reducida y verificar la equivalencia con el tensor completo por vía numérica sobre
  perfiles genéricos, no simbólicamente.
- **Cinco rutas de solver descartadas** antes de la que funcionó: ajustar un solo
  parámetro de horizonte; disparo multi-paramétrico con `least_squares` + `solve_ivp`;
  `solve_bvp` con semilla de Myers-Perry y jacobiano analítico exacto (jacobiano
  singular, parámetros divergiendo a 10⁴–10²²); extraer las condiciones de horizonte
  de la forma compactificada (`sp.limit` no terminaba); y traducir las relaciones
  crudas a variables compactas (dio expresiones correctas pero irreconocibles, que no
  reproducían ni la forma de vacío). La que funcionó exigió rehacer las expansiones
  de horizonte e infinito **directamente** en variables compactas.
- Tres errores propios detectados y corregidos en el camino, cada uno con su costo:
  `z` y `x` tratados como símbolos independientes; el numerador extraído antes de la
  serie, que produjo una relación espuria; y `f''`,`w''` omitidos en los jets
  físicos, que daba residuo tensorial de orden 1 incluso sobre Myers-Perry exacto.
- El contraste externo de 4C con dos radios nunca cerró en uno de ellos. Se registró
  como fallo en vez de ensanchar la tolerancia, y eso quedó en el manuscrito.

### 2.º — Hito 8: medición de `Psi_GB`, manuscrito y arbitraje · 8,7 h · 408 M tokens · 1,59 M de salida

Es donde se produjo el único resultado nuevo del proyecto: `Psi_GB` exacto sobre 356
estados aceptados en trece acoplamientos, con las dos anclas externas reproducidas.

- Dos cambios de método fueron lo que lo hizo posible: continuación sembrada por
  respuesta lineal, y derivada en el acoplamiento tomada exacta de una sola
  resolución lineal. El segundo, de paso, **corrigió la tabla del Hito 6**, cuyos
  valores eran la misma cantidad con truncamiento de hasta 1,5·10⁻².
- La revisión bibliográfica **recortó el hito**: dos de las piezas que iba a afirmar
  ya estaban publicadas (2010 y 2023). Pasaron a ser anclas externas.
- Cuatro errores propios corregidos a mitad de camino: la medida de distancia a
  extremalidad estaba mal elegida y se anulaba en los dos extremos de la familia; la
  escala del residuo relativo se anulaba en vacío; la ruta de masas se estimó primero
  con un ajuste polinómico global cuya derivada erraba un 5 %; y el barrido moría con
  un solo fallo de portón.
- Después vino una ronda de arbitraje completa: nueve objeciones convertidas en
  puertas recalculadas, una verificación pedida que **falló al hacerla** (las barras
  de la figura 2 no cubrían las secantes), y un error de masa real que la cota exacta
  delató y que los tres diagnósticos internos no veían porque compartían la misma
  ventana de ajuste.

### 3.º — Hito 6: potencial conjugado `Psi_GB` · 4,5 h · 169 M tokens · 0,31 M de salida

Calcular la carga conjugada al acoplamiento, que la primera ley del Hito 4C había
dejado pendiente.

- Bloqueo numérico real: el solver no converge al vacío desde su semilla trivial para
  `q ≥ 0,55`, lo que trababa la escalera en su primer peldaño.
- Las corridas se interrumpían repetidamente perdiendo cerca de una hora de trabajo
  cada vez, hasta que se construyó persistencia incremental por punto y por peldaño.
- Cinco errores propios corregidos: un `arange` que devolvía el acoplamiento
  equivocado **en silencio**; una puerta escrita como desviación relativa, mal
  condicionada justo donde el oráculo cambia de signo; una contención con desigualdad
  estricta que rechazaba una raíz legítima; un bloque reescrito con el cuerpo vacío; y
  `Cache` usando `Path` sin importarlo.
- Y una predicción propia que resultó falsa: el cero de `Psi_GB` se desplaza hacia
  giros mayores, no menores.

---

## 4. ¿Alguno fue exageradamente más caro que los otros?

**Entre los tres, no.** Los dos primeros están prácticamente empatados: el Hito 4
supera al Hito 8 por un factor 1,3 en tiempo y 1,2 en tokens. El salto aparece
recién al tercero: el Hito 8 duplica al Hito 6 en tiempo y lo multiplica por 2,4 en
tokens.

**El contraste exagerado está en otro lado: entre estos bloques y el resto del
proyecto.**

| Comparación | Tiempo | Tokens | Salida |
|---|---:|---:|---:|
| Hito 4 ÷ Hitos 0–3 | **2,5×** | **5,0×** | **3,0×** |
| Hitos 4 + 8 sobre el total | 54 % | 65 % | 65 % |

Los Hitos 0 a 3 son la reproducción completa de lo ya conocido: el tensor de
Einstein, Myers-Perry cerrado, el BVP de vacío por dos métodos independientes y la
rama estática de Einstein-Gauss-Bonnet. Todo eso costó 4,4 h y el 7 % de los tokens.
**El primer objeto genuinamente nuevo —el agujero negro rotante en EGB— costó cinco
veces más tokens que todo ese benchmark junto.**

Dos observaciones más, por si sirven para la presentación:

- **El Hito 6 quemó contexto sin producir texto.** Es el 12 % de los tokens
  procesados pero sólo el 5 % de los tokens de salida: 68 k de salida por hora contra
  203 k del Hito 4 y 183 k del Hito 8. Es la firma de las corridas que se
  interrumpían y se reintentaban.
- **El gasto no está en escribir código.** Los 5,94 M de tokens de salida equivalen a
  unos 24 MB de texto generado, contra 1,73 MB de producto final —código, tests,
  checks, experimentos, documentos, informes, planes y manuscrito de la etapa
  `project`—: una relación de 14 a 1. Lo que consumió el presupuesto fue
  verificar, fallar, detectar errores propios y rehacer — que es, a la vez, lo que
  hizo que el resultado sea defendible.
