# EGB rotante D=5 — sistema radial, compactificación y primer solver (Hitos 4A-4B)

## Estado

**Hito 4A cerrado; Hito 4B con un primer punto aceptado.** Este documento
cubre el plan de `docs/2026-09-09-hito-4-diseno.md` (ajustado por
`reports/hito-4-review.md`): el sistema radial `b,f,h,w` genérico con
`alpha_GB` y rotación simultáneos, derivado y verificado; su
compactificación (horizonte e infinito, en las variables listas para un
solver espectral); y un solver espectral real, verificado contra Myers–Perry
y contra el tensor completo, que produjo el primer punto EGB rotante
aceptado (`omega_H=0.3`, `alpha_GB` hasta 0.1, `results/egb-rotating.json`).
**No** cubre todavía: el puente de rotación lenta contra la ec. (2.21) del
paper, un método adaptativo independiente para contrastar contra el
espectral, ni la familia/comparación externa de Hito 4C.

## Método

Un intento de derivación tensorial directa (Riemann+GB contraídos sobre la
métrica rotante completa, extendiendo `experiments/derive_static_egb.py`)
resultó computacionalmente intratable en este entorno: sólo construir
Christoffel/Riemann simbólicos para esta métrica no diagonal 5×5 no terminó
en varios minutos. Es la misma razón, presumiblemente, por la que el Hito 2
usó la acción reducida en vez del tensor directo para el caso rotante.

Se usó en cambio la acción reducida (ecs. 2.10–2.12 de 1010.0860v1, leídas
del PDF cacheado y confirmadas independientemente contra el renderizado
HTML/MathML de arXiv, término a término):

\[
L_E=\sqrt{\frac{fh}b}\Big(b'g'+\frac g{2h}b'h'+\frac b{2g}g'^2+\frac bh g'h'
+\frac12ghw'^2+\frac{2b}f\Big(4-\frac hg\Big)\Big),
\]

\[
L_{GB}=\sqrt{\frac{fh}b}\frac1g\Big(\frac{4h}gb'g'+2(4g-3h)\Big(\frac{b'h'}h
+hw'^2\Big)-\frac f{2h}b'h'g'^2-\frac12fhg'^2w'^2\Big),
\]

con `L_eff=L_E+alpha_GB*L_GB` (la normalización interna `alpha_GB` ya
absorbe el factor 1/4 de la ec. 2.10 del paper, `docs/convenciones.md`).

**Equivalencia acción/tensor (ajuste AstraCheck, 2026-09-09):** la puerta no
es que las densidades lagrangianas coincidan símbolo a símbolo, sino que la
variación Euler–Lagrange reproduzca la contracción tensorial completa sobre
perfiles genéricos que no son solución. Se comprueba numéricamente (no
simbólicamente, por la razón de arriba) mediante la identidad ya usada y
probada en el Hito 2:

\[
E_v=-r^2\,(G+\alpha_{\rm GB}H)^{ij}\frac{\partial g_{ij}}{\partial v},
\qquad
E_f=r^2\,\frac{(G+\alpha_{\rm GB}H)^r{}_r}{f},
\]

evaluada con el evaluador tensorial numérico de `einstein.py`
(`curvature`+`gauss_bonnet`, NumPy puro — rápido incluso para una métrica no
diagonal, a diferencia de la derivación simbólica completa) sobre jets
locales de orden 2 para `b,f,h,w`, en seis perfiles genéricos fijos
(`experiments/derive_egb_rotating.py::PROFILES`) que cubren signos y
magnitudes distintas y `alpha_GB` entre 0 y 0.6.

`L_E` se comprueba además, por separado, contra la Lagrangiana de vacío ya
derivada y probada en el Hito 2 (`docs/vacuum-bvp.md`): ambas dan
exactamente las mismas ecuaciones de Euler–Lagrange (no sólo proporcionales)
en los perfiles de prueba — es decir, difieren a lo sumo en una derivada
total, que resultó no hacer falta invocar en este caso.

## Resultado de la derivación

Variando `L_eff` respecto de `b,g,h,w` **antes** de fijar el gauge (como en
Hito 2), luego sustituyendo `g=r^2,g'=2r,g''=2`, se obtiene un sistema de
cuatro ecuaciones (`E_b=E_g=E_h=E_w=0`), lineal y conjuntamente resoluble
para `[f',b'',h'',w'']` — **las mismas cuatro incógnitas que en vacío**, no
un conjunto mayor. La matriz de coeficientes es genéricamente invertible
(determinante no nulo verificado en los seis perfiles de prueba; forma
cerrada disponible pero no ilustrativa por sí sola).

**Esto refuta la anticipación "7→9" del diseño original** (ver
`reports/hito-4-review.md`, hallazgo 4): el orden total del sistema
permanece en 7 (`f'` orden 1, `b'',h'',w''` orden 2 cada una), igual que en
el Hito 2. La curvatura cuadrática de Gauss–Bonnet no aumentó el orden de
este sistema reducido, consistente con que Lovelock da ecuaciones de campo
de segundo orden en la métrica.

`E_f` (variación respecto de `f`, sin derivada temporal ni de `f`,
algebraica en `b,f,h,w,b',h',w'`) es la constraint reservada, exactamente
como en Hito 2 (`docs/vacuum-bvp.md`, `C=E_f f/r^2=G^r_r`), y la identidad
se comprobó válida también con el tensor GB incluido (ver arriba).

**Propagación de la constraint (cerrado).** `r^2\sqrt{bh}\,C=\text{const}`
cuando `E_b=E_g=E_h=E_w=0` — **la misma forma funcional que en vacío**, sin
corrección explícita de `alpha_GB` (el acoplamiento sólo entra
implícitamente, a través de qué `b,f,h,w` resuelven las cuatro ecuaciones).
No se derivó por la identidad de Bianchi contraída sobre el tensor completo
(la misma razón de intratabilidad de arriba lo impide para este ansatz);
se encontró en cambio por la identidad de Beltrami/Hamiltoniano de la
acción reducida, que no requiere el tensor:

`L_eff` no tiene `r` explícito (sólo a través de `b,f,g,h,w` y sus
derivadas), así que `Hcal=\sum_v v'\,\partial L_{\rm eff}/\partial v'-L_{\rm
eff}` (`v` en `b,g,h,w`) cumple idénticamente
`Hcal'=-\sum_v v' E_v-f'E_f` (cálculo de variaciones puro; el término `f'E_f`
es fácil de pasar por alto porque `f` no tiene término cinético propio, pero
`Hcal` igual depende de `f` a través de `L`). Por separado, `Hcal=2fE_f`
exactamente (identidad algebraica). Combinando ambas *on-shell*
(`E_b=E_g=E_h=E_w=0`) da `E_f\propto f^{-3/2}$ antes de normalizar/fijar
gauge, que se traduce en `r^2\sqrt{bh}\,C=\text{const}` después. Comprobado
con `sp.simplify` sobre `b,f,h,w` genéricos (no sólo en puntos numéricos):
da exactamente cero. Test de regresión numérico (integración RK4 del
sistema resuelto) en `tests/test_egb_rotating_equations.py`.

A `alpha_GB=0`, tanto las cuatro ecuaciones resueltas como el sistema
completo de cinco variaciones (`equations()`) coinciden con
`vacuum_equations.radial_equations` a precisión de máquina (~1e-15) en un
muestreo de diez puntos genéricos — la comprobación de consistencia más
fuerte disponible con el código ya probado del Hito 2.

## Módulos

- `experiments/derive_egb_rotating.py`: deriva, comprueba las cuatro puertas
  científicas (transcripción de `L_E` contra vacío; equivalencia
  acción/tensor; orden/rango del sistema; propagación de la constraint) y
  genera el código si las cuatro pasan; aborta con error si alguna falla.
- `src/rotating_bh/_egb_rotating_generated.py`: `rhs(r,b,f,h,w,bp,hp,wp,
  alpha_gb)` → `(fp,bpp,hpp,wpp)`, en la coordenada `r` **sin compactificar**
  (no es todavía el sistema listo para un BVP en `[0,1]`).
- `src/rotating_bh/_egb_rotating_equations_generated.py`: `equations(r,
  values,first,second,alpha_gb)` → `(E_b,E_f,E_g,E_h,E_w)`, para validación;
  `E_f` es la constraint reservada.
- `tests/test_egb_rotating_equations.py`: alpha_GB=0 contra
  `vacuum_equations.py`; `rhs()` anula las cuatro ecuaciones resueltas en
  cualquier `alpha_GB`; equivalencia con el tensor completo (incluida
  `E_f`) en los mismos perfiles genéricos, para varios `alpha_GB`;
  propagación de la constraint por integración numérica del sistema.

## Expansión de horizonte (parcial, verificada)

**Método:** en vez de extraer condiciones de regularidad de las fórmulas
compactificadas (`Bxx,Fx,Hxx,Wxx`, singulares en `z=1` de forma no trivial
para datos genéricos — ningún atajo evidente hasta ahora), se expandieron
directamente las ecuaciones sin compactificar alrededor de `r=r_H=1`:
`b=b_1\epsilon+b_2\epsilon^2/2`, análogamente `f,h,w` (`h_0,w_0` finitos,
`\epsilon=r-1`), sustituidas en los numeradores ya despejados de
denominadores de `E_b,E_g,E_h,E_w,E_f` y expandidas con `sp.series`
(truncar antes de expandir, no expandir-y-truncar después, fue la diferencia
entre minutos y segundos de cómputo).

**Resultado:** `E_b` y `E_f` tienen su primer término no nulo en orden
`\epsilon^1`, y son exactamente proporcionales entre sí
(`E_f|_{O(\epsilon)}=-E_b|_{O(\epsilon)}/h_0`) — la misma redundancia que
predice el argumento de `docs/vacuum-bvp.md` ("horizonte regular no extremal
tiene `C` finito y `b\to0`, por lo que esa constante debe ser cero"),
reproducida aquí por cálculo directo, no supuesta. `E_g,E_h,E_w` tienen su
primer término no nulo en orden `\epsilon^2`. En total, **cuatro relaciones
independientes** (una de `E_b`/`E_f`, tres de `E_g,E_h,E_w`) entre
`b_1,f_1,h_0,h_1,w_1` (orden 1) y `f_2,b_2,w_2` (orden 2, con `h_2`
apareciendo también). Fórmulas completas en
`experiments/derive_egb_rotating.py::horizon_expansion`.

**Verificación:** las cuatro relaciones se anulan a precisión de máquina
(`<1e-8`, en la práctica `~1e-14`) al evaluarlas en la solución cerrada de
Myers–Perry (`alpha_GB=0`, `rotating_bh.myers_perry.MyersPerry`, derivando
sus coeficientes de Taylor en el horizonte simbólicamente, no por
diferencias finitas) — código y datos independientes de esta derivación.
Puerta científica automatizada en
`check_horizon_expansion_against_myers_perry`.

**No cerrado:** no se tradujeron estas cuatro relaciones a la forma
compactificada de Hito 2 (`F+H=1` etc.) ni se determinó cuántos parámetros
libres quedan en el horizonte una vez fijados `r_H,\Omega_H,\alpha_{\rm
GB}` (ver más abajo — un experimento numérico sugiere que es más de uno).

## Expansión de infinito (cerrada)

**Método:** igual que el horizonte, pero con un error de proceso detectado
y corregido en el camino: extraer el numerador (`sp.fraction(sp.together(
...))`) antes de tomar la serie en `s=1/r` desplazó el orden aparente de
`E_w` (el denominador introducía su propia dependencia en `s`), lo que dio
una relación espuria (`V=2\alpha_{\rm GB}U`, que anularía `V` en el límite
de vacío) contradicha de inmediato por Myers–Perry. La corrección fue tomar
`sp.series` directamente sobre `gauged[name]` completo, sin separar
numerador y denominador.

**Resultado** (ansatz `b=1+Us^2+u_4s^4+u_6s^6`, análogamente `f` con `V,v_6`;
`h=1/s^2+H_2+H_4s^2+H_6s^4`; `w=Ws^4+w_6s^6+w_8s^8`, `s=1/r`):

- Orden `s^2` (sólo `E_f`): `H_2=0`.
- Orden `s^4`: `H_4=V`, `u_4=0`, `w_6=0`.
- Orden `s^6`/`s^8` (cuatro ecuaciones independientes —`E_b,E_f` en `s^6`;
  `E_g,E_h` recién en `s^8`— para tres incógnitas, consistentes entre sí):

\[
H_6=-UV-W^2,\quad
u_6=2\alpha_{\rm GB}U^2-\tfrac{UV}3+\tfrac{2W^2}3,\quad
v_6=2\alpha_{\rm GB}U^2-UV-W^2,\quad
w_8=W(2\alpha_{\rm GB}U-V).
\]

`U,V,W` (las cargas asintóticas, como en Myers–Perry) **quedan libres**;
`alpha_GB` sólo aparece a partir de `H_6,u_6,v_6,w_8` (orden `1/r^6`), no en
`H_4=V` ni en los órdenes anteriores — consistente con que la corrección de
Gauss–Bonnet es subdominante a grandes `r` y no cambia los momentos
multipolares principales.

**Verificación:** las diez relaciones (incluida la orden 8 de `E_g,E_h`) se
anulan a precisión de máquina contra Myers–Perry (`alpha_GB=0`, extrayendo
`U,V,v_6,u_4,u_6,H_2,H_4,H_6,W,w_6,w_8` por serie simbólica de la solución
cerrada — no por diferencias finitas). Puerta científica automatizada en
`check_infinity_expansion_against_myers_perry`.

## Primer intento numérico (fallido, informativo)

Con horizonte e infinito derivados, se intentó una integración numérica
directa: partir de los datos de horizonte de Myers–Perry, ajustar sólo
`f_1` mediante `L1` para restaurar consistencia a `alpha_GB` pequeño
(dejando `b_1,h_0,h_1,w_1` en sus valores de vacío) e integrar con
`solve_ivp` usando `rhs()`. **Resultado: la integración falla**
("required step size is less than spacing between numbers") para
`alpha_GB=0.001,0.01,0.05` — el sistema desarrolla una singularidad
espuria en el exterior. Esto confirma que `L1` por sí sola es necesaria
pero muy lejos de suficiente: hay más de un modo inestable en el horizonte
que debe controlarse simultáneamente (no basta perturbar un solo
parámetro a mano). Construir el disparo/colocación correcto —análogo al
solver espectral+Newton de los Hitos 2–3, no una perturbación ad hoc—
es trabajo pendiente, no completado en esta sesión.

Como control positivo de todo el resto de la maquinaria: la misma
integración a `alpha_GB=0` (con `h_1,w_1` exactos de Myers–Perry, sin
ajustar nada) sí reproduce Myers–Perry con precisión creciente hacia
`r` grande, confirmando que `rhs()` y los datos de horizonte usados son
correctos — el problema es específicamente la falta de un disparo
multi-parámetro (o solver de colocación) para `alpha_GB\neq0`.

## Segundo intento: disparo multi-parámetro y `solve_bvp` (también fallidos, informativos)

Con las cuatro relaciones de horizonte y los datos de Myers–Perry, se
probaron tres estrategias adicionales, las tres con **el mismo síntoma
final** — inestabilidad numérica genuina, no un problema de la semilla:

1. **Disparo multi-parámetro real** (`scipy.optimize.least_squares` sobre
   `(b_1,h_0,h_1,w_1)`, con `f_1` fijado por `L_1`, integrando con
   `solve_ivp` hasta `r=25`): falla ya en el primer paso, porque la
   integración **sin perturbar** (con los datos exactos de Myers–Perry pero
   `alpha_GB\neq0`) explota antes de `r\sim10$ para `alpha_GB=0.001`, y antes
   de `r\sim3.2` para `alpha_GB=0.01` — el radio de explosión se acorta al
   crecer `alpha_GB`, no depende de cuánto se perturbe. Esto descarta que el
   problema sea "falta de ajuste fino"; es una inestabilidad exponencial
   genuina en la dirección de integración saliente, del tipo que motiva a
   la relatividad numérica a preferir relajación/colocación sobre disparo
   puro para estos problemas de contorno.
2. **`scipy.integrate.solve_bvp`** (colocación adaptativa con parámetros
   libres, debería ser inmune a la inestancia de disparo por resolver todo
   el dominio a la vez): con datos iniciales de Myers–Perry razonables,
   diverge igual — el refinamiento de malla interno empuja los parámetros
   a valores sin sentido físico (`\sim10^{4}$–$10^{22}`) mientras el residuo
   de frontera converge engañosamente a un valor chico. Probado también con
   `\epsilon` más lejos del horizonte y datos de Taylor de orden 2: mismo
   resultado ("Singular Jacobian").
3. **Límite de regularidad directo en variables compactas** (expandir
   `Bxx,Fx,Hxx,Wxx` — ya verificadas, ver arriba — cerca de `x=0` con
   `B,F,H,W` como serie de Taylor local, buscando qué relación cancela el
   polo, en vez de traducir las relaciones de `r` cruda): computacionalmente
   intratable en este entorno (`sp.limit`/`sp.expand` no terminan en tiempo
   razonable para estas expresiones); un intento de traducir directamente
   `L_1$–$L_4` (crudas) a `B_H,F_H,H_H,Q_H,V_H,W_H` mediante el mapa de
   coordenadas (`b_1=2B_H$, etc.) dio expresiones correctas pero que no se
   redujeron a la forma simple de vacío (`F+H=1`) ni siquiera en
   `alpha_GB=0` — indicio de que Hito 2 usa una combinación distinta de las
   cuatro ecuaciones crudas (o una convención de índices distinta) para
   llegar a esa forma particular, no reproducible por simple traducción de
   variables sin rehacer la derivación en las variables compactas desde el
   principio.

**Conclusión, con evidencia concreta en las tres direcciones probadas:**
este no es un problema resoluble por disparo directo ni por aplicar
`solve_bvp` de SciPy con una semilla razonable. Requiere la misma
infraestructura hecha a mano que usan los Hitos 2–3 (matrices de
diferenciación de Chebyshev, Newton amortiguado con búsqueda de línea,
Jacobiano por paso complejo, control explícito de positividad durante la
iteración) — no una adaptación rápida, sino un desarrollo del mismo orden
de esfuerzo que esos hitos completos. No se fuerza ni se simula ese
resultado aquí.

Ambos intentos fallidos fueron decisivos para lo que sigue: mostraron que
había que trabajar en variables compactas desde la propia derivación (no
traducir después) y que la colocación global (no el disparo) es la única
vía viable — exactamente lo que se hizo a continuación, pidiéndolo el
usuario explícitamente ("hagamos ese esfuerzo analítico explícito").

## Horizonte e infinito en variables compactas (cerrado)

**El error del tercer intento (arriba) era real pero acotado:** al
evaluar las ecuaciones crudas (numerador, sin denominador) exactamente en
`b=f=0` (el horizonte), el resultado es **idénticamente cero para
cualquier valor de las demás variables** — cada término de `E_b,E_g,E_h,E_w`
tiene un factor explícito de `b` o `f` (comprobado por sustitución directa,
no supuesto). Por eso evaluar justo en el horizonte da `0=0` trivial; el
contenido real está un orden más allá — el mismo patrón que
`horizon_expansion`, ahora repetido en las variables `B,F,H,W` mismas, con
sympy verificando cada paso:

1. Ansatz local `B(z)=B_H-P(z-1)+B_{xx}(z-1)^2/2` (análogamente `F` con
   sólo `F_x`, y `H,W` con `Q,V,H_{xx},W_{xx}`), sustituido directamente en
   `b=(1-z^2)B(z)`, etc., y de ahí en las ecuaciones crudas `E_b,E_g,E_h,E_w`
   — no en el sistema ya despejado por Cramer (que es el que se vuelve
   singular en el horizonte; las ecuaciones crudas, polinómicas en
   `b,f,h,w`, no lo son).
2. `E_b` da cero hasta orden `\epsilon^1` inclusive y una relación real en
   `\epsilon^2`; `E_g,E_h,E_w` dan cero en `\epsilon^0,\epsilon^1` y una
   relación real en `\epsilon^2` cada una. Cuatro relaciones en total, ahora
   **directamente en `B_H,F_H,H_H,W_H,P,Q,V,F_x,W_{xx}`** — sin traducir
   nada desde `r` cruda.
3. Verificadas contra Myers–Perry calculando `B(z),F(z),H(z),W(z)`
   directamente de la solución cerrada (`b/(1-z^2)`, etc.) y su serie de
   Taylor en `z=1`: las cuatro se anulan a precisión de máquina
   (`check_compact_horizon_against_myers_perry`).
4. Las condiciones de infinito (`z=0`) se obtuvieron más simple: haciendo
   coincidir, término a término y con `sp.solve` (no a mano), la serie de
   Taylor de la ansatz compacta contra la ya verificada `infinity_expansion`
   en `s=1/r`. Da exactamente `B=1,F=1,H_x=0,W_x=0` en `z=0` — **las mismas
   cuatro condiciones que usa el propio código de `vacuum_bvp.py::_spectral`
   en su extremo `x=0`** (la prosa de `docs/vacuum-bvp.md` sólo menciona
   tres; el código impone las cuatro). `alpha_GB` no aparece en estas
   cuatro condiciones — coherente con que sólo corrige el infinito recién
   en `1/r^6`.

`experiments/derive_egb_rotating.py::compact_horizon_conditions` y
`compact_interior_system` (este último: convertir el sistema ya resuelto a
`Bxx,Fx,Hxx,Wxx` en variables compactas, para los nodos interiores — **no**
el horizonte, que sigue singular ahí; verificado exactamente contra
`rotating_bh._vacuum_generated.rhs`, el código ya probado de Hito 2, en
`alpha_GB=0`) generan el código numérico (`_egb_rotating_horizon_generated.py`,
`_egb_rotating_compact_generated.py`).

## Solver espectral (Hito 4B): primer punto rotante EGB aceptado

Con horizonte, infinito e interior ya en variables compactas, se construyó
`src/rotating_bh/egb_rotating_bvp.py`, calcado de la arquitectura de
`vacuum_bvp.py::_spectral` (Chebyshev–Lobatto, Newton amortiguado con
Jacobiano por paso complejo y búsqueda de línea) — **no** el disparo que
había fallado antes.

**Un problema abierto que se resolvió empíricamente:** las cuatro
relaciones de horizonte no incluyen ninguna que fije `W_H=\Omega_H`
(el valor de `w` en el horizonte es, por definición, `\Omega_H`, pero eso
no aparece en `E_b,E_g,E_h,E_w`). Se comprobó que la relación de `E_w` por
sí sola determina genuinamente `W_{xx}$ (no es redundante: sustituyendo
`E_b=E_g=E_h=0` en `E_w` con valores numéricos genéricos se obtiene una
relación lineal no trivial en `W_{xx}`, no `0=0`). La solución adoptada
—que reproduce Myers–Perry exactamente, ver abajo— fue usar `E_b,E_g,E_h`
más `W_H-\Omega_H=0` como las cuatro filas de horizonte, dejando que
`W_{xx}` en el horizonte emerja de la representación espectral global (como
`F=1` "emerge" en vacío) en vez de imponerse ahí. Esto no está demostrado
analíticamente — es una elección validada por resultado, documentada como
tal en el código.

**Resultado, verificado en cascada:**

1. **`alpha_GB=0` reproduce Myers–Perry** partiendo de una semilla trivial
   (`B=F=1,H=0,W=\Omega_H`, sin ninguna referencia a Myers–Perry): Newton
   converge en pocas iteraciones y el perfil coincide con la solución
   cerrada a 8 cifras en varios radios.
2. **Continuación en `alpha_GB`** (usando la solución anterior como
   semilla) alcanza `alpha_GB=0.1$–$0.325` sin dificultad con pasos
   moderados (`0.02`–`0.1`); pasos más grandes o `alpha_GB` mayor
   necesitan pasos más finos (comportamiento esperable de continuación, no
   una falla).
3. **Verificación física independiente, la más fuerte disponible:** el
   perfil convergido en `alpha_GB=0.1$–$0.2` se evaluó con el tensor
   completo (`einstein.py::curvature`+`gauss_bonnet`, construyendo jets
   físicos `b,b',b'',f,f',f'',h,h',h'',w,w',w''` desde la representación
   compacta — con `f'',w''` incluidos: un intento previo los omitió por
   asumir que la física no los necesita como incógnitas independientes,
   pero el tensor métrico sí los necesita vía la regla del producto/cociente
   para `1/f` y `h w^2`, y omitirlos daba un residuo espurio de `O(1)` cerca
   del horizonte incluso para Myers–Perry exacto — detectado y corregido
   antes de confiar en el resultado). Residuo `G+\alpha_{\rm GB}H`:
   `<10^{-12}` en todo el dominio muestreado, para `omega_H=0.3,
   alpha_GB=0.2`.
4. Control negativo (`alpha_GB` incorrecto en la verificación): residuo
   `\sim5\times10^{-3}`, muy por encima del umbral — el chequeo detecta el
   error deliberado.

**Artefactos:** `results/egb-rotating.json`, `artifacts/egb-rotating.png`
(`experiments/egb_rotating_bvp.py`, reproducible por hash — corrido dos
veces, mismo hash ambas). `src/rotating_bh/egb_rotating_validation.py`
(jets físicos + residuo tensorial, análogo a `static_egb_validation.py`).
`tests/test_egb_rotating_bvp.py` (8 tests; cada `solve()` cuesta varios
segundos por el tamaño de las ecuaciones EGB, no es un error).

## Qué falta

1. ~~Propagación tipo Bianchi del constraint.~~ ~~Horizonte.~~
   ~~Infinito.~~ ~~Compactificación.~~ ~~Un solver que converja.~~ Todo
   cerrado, arriba.
2. Puente de rotación lenta contra la ec. (2.21) del paper — no iniciado;
   no bloquea el punto ya aceptado, sólo sirve como verificación adicional.
3. **Método adaptativo independiente** (análogo al `solve_bvp` con
   transporte de Taylor de `vacuum_bvp.py`/`static_egb_bvp.py`) para
   contrastar contra el espectral (`methods_agree`, como en Hitos 2-3) — no
   existe todavía; por ahora la única verificación independiente del
   espectral es el tensor completo (fuerte, pero no reemplaza tener un
   segundo método numérico).
4. Barrido de resolución más amplio y en la tolerancia (Hito 4B sólo probó
   dos resoluciones en un caso); termodinámica (`E,J,T_H,S`, primera ley);
   family en `\Omega_H,\alpha_{\rm GB}` y comparación externa cuantitativa
   — eso es Hito 4C, no 4B.
5. Reservas explícitas del primer pase: un solo `\Omega_H` (0.3), sin
   estudio de qué tan cerca de extremalidad o de `\alpha_{\rm GB}` grande
   sigue funcionando este mismo par de rutas (horizonte/infinito); no se
   investigó si hace falta refinar el `W_H=\Omega_H` vs. `E_w` elegido
   empíricamente en el punto 3 de la sección anterior para regímenes más
   exigentes.

## Contraste adaptativo del punto aceptado (2026-09-10)

**Resultado:** un segundo método numérico converge en
`r_H=1, omega_H=0.3, alpha_GB=0.1`. La comparación nominal da
`max|delta(B,F,H,W)|=9.8894e-9` en 151 puntos de `x∈[0.02,0.98]`;
el tensor completo fuera de nodos da `2.4179e-7` en 51 puntos. Esto
aporta la evidencia solicitada para el ítem de contraste del Hito 4B;
la decisión de integración/cierre queda pendiente. No se avanzó a 4C
ni se modificó el solver espectral o la fuente derivada de 4A.

### Método, condiciones y alcance de independencia

`src/rotating_bh/egb_rotating_adaptive.py` usa la colocación adaptativa
de cuarto orden de `scipy.integrate.solve_bvp`, con 81 nodos iniciales,
máximo 4000 y Jacobiano interior por paso complejo. Comparte las ecuaciones
generadas de 4A, pero no las matrices de Chebyshev, la discretización,
la búsqueda de línea ni la representación espectral. La semilla se obtiene
por `EGBRotatingSolution.evaluate()`, incluida su primera derivada;
**ningún valor espectral se introduce en las condiciones de frontera**.
Se verifica así un segundo método dado un punto de partida cercano,
no un descubrimiento de la rama independiente de la semilla.

La coordenada utilizada por el código es `x=1-1/r`, `z=1-x`: **horizonte
x=0, infinito x=1**. Algunas descripciones previas invierten estos extremos;
no se alteran aquí ni los archivos generados ni las secciones históricas.
Las siete variables son `y=(B,F,H,W,Bx,Hx,Wx)` y el intervalo resuelto es
`[epsilon,1-epsilon]`. Se transporta a los extremos verdaderos mediante

`y_end = y ± epsilon*y' + epsilon²*y''/2`,

con `y''=(∂x+f·∂y)f` evaluado por paso complejo. Esto se usa exclusivamente
para las fronteras. En el horizonte se imponen las tres relaciones generadas
`E_b,E_g,E_h` y `W-Omega_H`; `F_x` se transporta como
`F_x(0)≈F_x(epsilon)-epsilon*F_xx(epsilon)`. En infinito se imponen
`B=1,Hx=Wx=0`. Son siete condiciones para siete variables, como en la ruta
adaptativa del Hito 2; **F(infinito)=1 se reserva como comprobación posterior**.
El transporte no se declara exacto: el estudio de epsilon mide su efecto.

La representación final integra los splines cúbicos de `Bx,Hx,Wx` para
obtener `B,H,W` de clase C2, restando una deriva constante que conserva los
valores en las dos fronteras; `F` conserva el spline de SciPy. Los jets
usados en el tensor son derivadas de esa representación, nunca reemplazos
por el RHS. Se comprueba coherencia por diferencias finitas y continuidad
de las segundas derivadas de `B,H,W` en los nodos. `evaluate()` rechaza
extrapolar fuera del intervalo resuelto.

### Criterios y controles independientes

Antes de mirar el acuerdo EGB se fijaron dos tolerancias absolutas de
`1e-5`: para las cuatro amplitudes adimensionales en 151 puntos de
`[0.02,0.98]`, y para el máximo tensorial en 51 puntos fuera de nodos,
con la normalización de `egb_rotating_validation.py`. El intervalo
corresponde a `r≈1.0204…50`, no incluye los extremos singulares. Son
criterios conservadores para un contraste de un punto; no prometen
precisión uniforme hasta el horizonte o infinito. Las fronteras de la
representación deben además tener residuo `<1e-8`, y el residuo RMS de
colocación debe satisfacer la tolerancia solicitada. El indicador
`success` de SciPy por sí solo no acepta una corrida.

Puertas comprobadas:

- El oráculo compacto se compara con `myers_perry.MyersPerry`; sus derivadas
  se obtienen con SymPy. Las siete ODE reproducen esas derivadas y el error
  de transporte de frontera decrece al reducir epsilon. En vacío, la
  solución adaptativa tiene diferencia máxima de amplitudes `3.74e-10`
  contra el oráculo cerrado y residuo tensorial `7.90e-9` (31 puntos).
  Otro test parte del oráculo multiplicado por `1.001` y recupera la solución.
- Cada paso espectral `alpha_GB=0,0.02,0.05,0.1` se verifica con el tensor
  completo antes de usarse; en el último paso, con 24 nodos y tolerancia
  `1e-11`, el residuo muestreado es `7.69e-12`.
- En el caso nominal EGB, el residuo de frontera representada es `3.04e-12`
  y el de colocación `9.11e-8`, tras seis iteraciones y 177 nodos.
  La condición no impuesta de infinito tiene defecto `|F∞-1|=4.65e-11`,
  estimado por Taylor del spline representado, sin sustituir derivadas por
  la ODE.
- Usar deliberadamente `alpha_GB=0.15` en el tensor del perfil de `0.1`
  produce residuo `5.3673e-3` en `x=0.5`: el control detecta el error.
- Multiplicar toda la semilla espectral y sus derivadas por `1.001` lleva
  a un perfil que difiere del nominal sólo `2.66e-15`. El acuerdo no resulta
  de devolver la interpolación de la semilla sin resolver.

### Sensibilidad al corte y a la tolerancia

Todas las filas usan la misma semilla EGB convergida y el mismo límite de
4000 nodos. Se comparan las amplitudes representadas, no sólo observables
integrados ni valores en los nodos de colocación.

| epsilon | tol de SciPy | nodos finales | máximo delta de amplitudes | máximo tensorial |
| --- | --- | --- | --- | --- |
| 0.006 | 1e-7 | 179 | 9.910e-8 | 2.774e-6 |
| 0.003 | 1e-6 | 88 | 1.016e-8 | 2.456e-7 |
| 0.003 | 1e-7 | 177 | 9.889e-9 | 2.418e-7 |
| 0.0015 | 1e-7 | 173 | 1.004e-9 | 1.641e-8 |
| 0.003 | 1e-8 | 374 | 9.872e-9 | 2.461e-7 |

Reducir epsilon mejora tanto perfiles como tensor en los tres cortes
probados. Reducir sólo la tolerancia de SciPy a epsilon fijo apenas
mejora el error de perfil y no mejora monótonamente el tensor muestreado:
en este régimen domina el efecto de frontera/corte, no el RMS interior.
Estos datos no constituyen una prueba de orden asintótico ni un estudio
de otros acoplamientos o de extremalidad.

### Intento fallido y controles de recursos

El primer ensayo para construir la semilla usó un salto espectral directo
`alpha_GB: 0→0.1`, con `omega_H=0.3`, 24 nodos, `tol=1e-11` y máximo
50 iteraciones. Agotó las 50 iteraciones con residuo `5.61795`. El fallo
ocurrió **antes de llamar al adaptativo**; no es evidencia de que éste
diverja. Manteniendo nodos y tolerancia, usar los pasos intermedios
`0.02,0.05,0.1` converge en 6, 6 y 8 iteraciones respectivamente. Esto
identifica sensibilidad de ese salto a la continuación, sin demostrar
una causa física o una inestabilidad exponencial. El test usa esa ruta
de continuación y conserva explícitamente este intento en la documentación.

Como control reproducible de fallo real del adaptativo, se lo ejecutó con
la semilla EGB aceptada, sólo ocho nodos permitidos, `epsilon=0.003` y
`tol=1e-10`. Devuelve `status=1`, "The maximum number of mesh nodes is
exceeded", RMS `0.0385608`, aun cuando el residuo de frontera es
`2.92346e-11`. Se rechaza y se registra: una frontera pequeña no implica
resolver el interior.

La convergencia actual acota las conclusiones generales de los intentos
fallidos históricos: `solve_bvp` sí funciona con esta formulación de
fronteras y una semilla ya convergida. No se rehicieron aquellos ensayos
controlando una sola diferencia por vez, por lo que no se atribuye su
fallo exclusivamente a la semilla ni se confirma retrospectivamente una
inestabilidad exponencial como causa única.

### Evidencia reproducible y preparación del checkout

El experimento nuevo `experiments/egb_rotating_cross_check.py` genera
`results/egb-rotating-cross-check.json` y su entrada nueva en
`artifacts/manifest.json`; guarda parámetros, versiones, hashes de fuentes,
diagnósticos y controles fallidos. Los tests nuevos están en
`tests/test_egb_rotating_adaptive.py`, `tests/test_egb_rotating_cross_check.py`
y `tests/test_egb_rotating_angular.py`. No se regeneraron los resultados
aceptados de 4A/4B.

El worktree entregado tenía HEAD `84880dc`, con Hito 4 sin seguimiento en
la raíz y un `project/` antiguo. Con autorización del usuario se copiaron
130 archivos dentro de `project/`, verificados por SHA-256. Los 11 archivos
anteriores que diferían quedaron preservados en
`reports/baseline-before-hito-4b/`, junto con el manifiesto de importación.
La suite completa de esa base pasó 240 tests. No se instaló ninguna
dependencia. La raíz provista no se sustituyó.

La restricción posterior del usuario prohíbe escrituras fuera de
`project/`, incluidos los metadatos Git del worktree: por eso la entrega
queda sin commit ni merge. La preparación y los resultados son archivos
revisables dentro de `project/`; la integración debe conservar el trabajo
previo de main y distinguir la base importada de estos cambios nuevos.

## Relación angular de horizonte: argumento y verificación (2026-09-10)

Hay un argumento analítico para el **sector angular regular**, aunque no
una prueba de existencia/unicidad del sistema EGB acoplado. La acción
reducida existente no depende de `w`, sólo de `w'`. Por tanto

\[
E_w=-\frac{d\mathcal J}{dr},\qquad
\mathcal J=\frac{\partial L_{\rm eff}}{\partial w'}
=h\sqrt{\frac{fh}{b}}
\left[r^2+\alpha_{\rm GB}\left(16-\frac{12h}{r^2}-4f\right)\right]w'.
\]

No se rederivó ni editó el sistema radial: SymPy diferencia la Lagrangiana
ya existente y comprueba esta identidad exactamente. `mathcal J` es el
momento conjugado de la acción reducida, sin asignarle una normalización
ADM. En variables compactas, definiendo

\[
A=1+z^4H,\qquad
C=1+4\alpha_{\rm GB}z^2[1-(1-z^2)F-3z^4H],
\]

se obtiene `mathcal J=sqrt(F/B) A^(3/2) C (z Wx-4W)`.
`egb_rotating_angular.py::first_integral` evalúa esa expresión sólo como
diagnóstico. SymPy verifica también el cambio de variables y la identidad
con la cuarta relación generada de horizonte:

\[
R_{w,H}=-\frac{8 B_H F_H}{\sqrt{F_H/B_H}\sqrt{1+H_H}}
\left.\frac{d\mathcal J}{dx}\right|_{x=0}.
\]

La igualdad se comprueba sobre símbolos genéricos, incluyendo `F_x` y
`W_xx`, sin imponer las otras tres relaciones de horizonte. En particular,
el coeficiente de `W_xx` en la relación cruda es
`-8 B_H F_H(1+H_H)[1+4 alpha_GB(1-3H_H)]`. Si ese coeficiente no se anula,
la relación determina una derivada de segundo orden; **no es una identidad
algebraica redundante**. Para una solución C2 regular que satisface la
ecuación interior hasta el horizonte, su límite ya impone `R_w,H=0`.

Para perfiles `b,f,h` dados, regulares y con el coeficiente `K(r)` de
`w'` finito/no nulo en el horizonte y sin ceros exteriores, la solución
angular se escribe como `w'=mathcal J/K`. La condición `w(infinito)=0`
elimina la constante aditiva (fija el marco asintótico), y

\[
\Omega_H=-\mathcal J\int_1^\infty\frac{dr}{K(r)}
\]

fija la constante restante cuando la integral existe y no es cero. Así,
`W_H=Omega_H` especifica un dato físico de frontera, mientras que la
relación cruda de `E_w` expresa el límite de una ecuación que ya se resuelve
en el interior. No hace falta imponer ambas como datos independientes de
la ecuación angular. Este argumento no justifica por sí solo todas las
filas del sistema espectral discreto ni excluye degeneraciones del sistema
acoplado; no se extrapola a ramas donde `C` pueda anularse.

Verificación: el primer integral es constante en Myers–Perry cerrado
(`mathcal J=-4 omega_H/(1-omega_H²)` para `r_H=1`). En el espectral EGB
de 24 nodos su variación muestreada es `3.33e-13` y la relación cruda
`R_w,H=-2.40e-10`, aun sin imponerse como fila. El experimento registra
además la conservación en cada perfil adaptativo, el mínimo muestreado
de `C` y una cuadratura independiente de la integral que reconstruye
`Omega_H`. El control tensorial sigue siendo la comprobación independiente
de la acción; el primer integral y la identidad simbólica son verificaciones
complementarias, no un reemplazo del tensor.

## Barrido ampliado y puerta de cierre científico de 4B (2026-09-10)

La continuación del programa de hitos identificó una puerta todavía
pendiente después del contraste adaptativo: ampliar el estudio espectral
de resolución y tolerancia. Se implementó en
`src/rotating_bh/egb_rotating_convergence.py` y
`experiments/egb_rotating_convergence.py`, sin modificar ninguno de los
dos solvers ni las fuentes derivadas de 4A.

**Diseño previo al barrido:** `omega_H=0.3,alpha_GB=0.1,r_H=1`, cinco
resoluciones `N=12,16,24,32,40` y tres tolerancias de Newton
`1e-5,1e-8,1e-11` (15 celdas). Todas parten de la misma solución numérica
gruesa de 16 nodos, multiplicada por `1.01`. La semilla se construye
mediante continuación `alpha_GB=0,0.02,0.05,0.1`; cada paso pasa el
tensor completo y el primero también Myers–Perry cerrado. La semilla no
se reemplaza con cada solución refinada: se evita que una tolerancia
menor simplemente devuelva un perfil que ya la satisfacía. Cada celda
dispone de un máximo de 15 iteraciones; se guardan todos los resultados,
incluidos los convergidos que no pasan las puertas físicas.

El contraste es el adaptativo independiente con `epsilon=0.0015` y
`tol=1e-7`: tensor máximo `1.64e-8`, frontera `4.76e-11` y RMS de
colocación `8.22e-8`. Se verifican las cuatro amplitudes en 151 puntos
de `[0.02,0.98]`, el tensor completo en 51 puntos fuera de nodos,
positividad en 501 puntos del intervalo compacto completo, horizonte
simple (`b'_H=2B_H>0,f'_H=2F_H>0`) y las condiciones asintóticas
representadas. Las derivadas que alimentan el tensor son las del perfil
retornado, no el RHS.

**Criterios fijados antes de mirar el barrido:** cada perfil aceptado
requiere tensor `<1e-6`, diferencia adaptativa `<1e-5`, frontera e
infinito `<1e-8`, exterior positivo y horizonte simple. Para cerrar la
puerta de convergencia se requiere la grilla completa, los tres `N>=24`
nominales (`tol=1e-8`) aceptados, al menos dos de esos N con `tol=1e-11`
aceptados, mejora tensorial de al menos 100 entre N=12 y N=40, y
diferencia entre perfiles N=32/40 `<1e-8`. La semilla y el adaptativo
tienen sus propios controles independientes, reconstruidos desde los
datos guardados al verificar el artefacto.

### Resultados medidos

| N, tol=1e-8 | Máximo tensorial | Diferencia con adaptativo | Frontera | Aceptado |
| --- | --- | --- | --- | --- |
| 12 | 1.4574e-5 | 2.6116e-7 | 1.39e-13 | no |
| 16 | 7.6225e-8 | 2.2209e-9 | 2.16e-14 | sí |
| 24 | 7.7444e-12 | 1.0039e-9 | 4.79e-13 | sí |
| 32 | 1.6166e-13 | 1.0039e-9 | 1.47e-12 | sí |
| 40 | 2.9842e-13 | 1.0039e-9 | 5.21e-14 | sí |

Las 15 celdas convergen; **11 se aceptan y cuatro se rechazan**.
N=12 se rechaza en las tres tolerancias por tensor insuficiente, incluso
cuando la frontera es muy pequeña. N=16 con tol=1e-5 se rechaza por
frontera `9.08e-6`, aunque su tensor sea `1.34e-7` y sus amplitudes
coincidan bien. No se relajaron las puertas para aceptar esas filas.

La tolerancia laxa necesita dos iteraciones en N=12/16; las demás
celdas requieren tres. En los tres N finos, el tercer paso de Newton
ya cae por debajo de todas las tolerancias ensayadas, de modo que
comparten el mismo resultado; no es una devolución de la semilla sin
iterar. La diferencia entre N=32 y N=40 es `2.2538e-14`.

El tensor deja de mejorar entre 32 y 40 nodos. Se registra esa meseta
sin exigir monotonía indefinida ni adjudicar automáticamente su causa
a un fallo físico. La diferencia con el adaptativo se estabiliza cerca
de `1e-9`, coherente con el estudio de corte anterior: mide también
el error de la referencia adaptativa, no sólo el error espectral.
No se deduce de este punto una garantía para toda la familia, regiones
extremales o acoplamientos mayores.

### Entrega y alcance del cierre

Los siete controles del barrido pasan. Los artefactos nuevos son
`results/egb-rotating-resolution.json` y
`artifacts/egb-rotating-resolution.png`, con registros separados en
el manifiesto. La figura marca explícitamente las corridas no aceptadas;
Newton convergente no se confunde con aceptación física.

`checks/hito-4b.ps1` ejecuta la suite, recalcula los controles del barrido
y verifica hashes y procedencia de éste y del contraste adaptativo
anterior. `-Reproduce` repite el estudio nuevo dos veces y
compara sus hashes. No vuelve a generar las fuentes protegidas ni
reescribe los resultados aceptados anteriores.

El informe `reports/hito-4b-closure.md` registra la reproducción final y
la aceptación desde una copia limpia dentro de `project/.cache/`.
El cierre es **científico y local**: no implica commit ni integración
Git, prohibidos por la restricción vigente de escrituras. Queda como
siguiente etapa obligatoria **4C**, con familia a dos acoplamientos,
observables, primera ley y comparación externa cuantitativa. El puente
de rotación lenta sigue como control adicional no bloqueante de 4A.


## 2026-09-10 — Hito 4C: familia y termodinámica verificadas; contraste externo parcial

Tras autorización explícita para avanzar al próximo hito se añadió un
módulo de observables, sin modificar solvers ni fuentes generadas de 4A.
Diez puntos de familia en alpha_GB=0.05,0.1 y cinco giros pasan los controles
tensoriales. Se midieron E,J,T,S, resolución con semilla perturbada,
ajustes de cola, corte adaptativo y ambas rutas de continuación.
La primera ley a acoplamiento fijo presenta convergencia de segundo orden.

El contraste explícito del radio de ergosuperficie concuerda con el texto
para alpha_paper=1, pero da 1.1021009314 frente a 1.104 en alpha_paper=2.
N40, colocación adaptativa y continuación desde estático confirman el valor
calculado; se conserva la discrepancia sin afirmar que el paper sea erróneo.
Diez lecturas cuantitativas del perfil alpha_paper=3 de la figura1b son
compatibles dentro de la incertidumbre gráfica declarada.

Se documentan dos fallos de Newton: el salto alpha_GB=0.5→0.6 agotó
50 iteraciones (residuo0.0284411), y la corrida final con presupuesto15
agotó el paso0.05→0.02 (residuo0.0928415). Reducir el primer salto a
0.025 permitió alcanzar0.75; el segundo queda para retomar. No representan
fronteras físicas. El ajuste cuadrático de cola tampoco pasó el umbral en
acoplamientos altos; elevar el orden a cúbico/cuártico sí lo resolvió,
manteniendo el umbral y registrando las estimaciones.

Se cierra este bloque de familia/observables; 4C permanece abierto conforme
al protocolo completo. Informe, tablas, límites y reproducción:
[reports/hito-4c.md](../reports/hito-4c.md). No hay commit ni merge:
los metadatos externos del worktree permanecen fuera del ámbito permitido.
