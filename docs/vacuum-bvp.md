# Problema de contorno de Einstein en D=5

## Derivación y ecuación reservada

Se conservan el ansatz y los signos de `convenciones.md`. La fuente primaria
es [Brihaye et al., 1010.0860v1](https://arxiv.org/html/1010.0860v1),
ecuaciones (2.8), (2.11), (2.15)–(2.17). La parte de Einstein de la acción
reducida, escrita de una forma algebraicamente equivalente, es

\[
L=g\sqrt{bh/f}\left\{f\left[
\frac{g'^2}{2g^2}+\frac{b'g'}{bg}+\frac{b'h'}{2bh}
+\frac{g'h'}{gh}+\frac{h w'^2}{2b}\right]+\frac8g-\frac{2h}{g^2}\right\}.
\]

Variar **antes** de fijar `g=r²`. Definir
`E_v=(∂L/∂v-d/dr(∂L/∂v'))/sqrt(bh/f)`.
`experiments/derive_vacuum.py` deriva E_b,E_g,E_h,E_w y despeja
`f',b'',h'',w''`; luego transforma al dominio compacto y genera
`src/rotating_bh/_vacuum_generated.py`. La generación no importa Myers–Perry.
`vacuum_equations.py` conserva las cinco variaciones explícitas para comprobar
jets radiales, incluida E_f, que no participa en la resolución.

El sistema tiene orden total siete. El despeje es regular en el exterior
con b,f,h positivos; no permite evaluar directamente los extremos singulares.
El test con perfiles genéricos, que no son solución, comprueba para las cinco
variaciones la identidad

\[
E_v=-r^2G^{ij}\frac{\partial g_{ij}}{\partial v}.
\]

La contracción usa el evaluador tensorial del Hito 1. Además se comprueban
vacío, inversión del giro y perturbación de w. Estas comprobaciones detectan
errores de coeficientes y signos que un test exclusivo sobre la solución
exacta podría no detectar.

La ecuación reservada es `C=E_f f/r²=G^r_r`. Cuando las otras cuatro
variaciones se anulan, la identidad de Bianchi da

\[
C'+\left(\frac{b'}{2b}+\frac2r+\frac{h'}{2h}\right)C=0,
\qquad r^2\sqrt{bh}\,C=\text{constante}.
\]

Un horizonte regular no extremal tiene C finito y b→0, por lo que esa
constante debe ser cero. Esto justifica la propagación del constraint; no
se impone C en los nodos para hacer que el control numérico sea nulo.

## Compactificación y condiciones

Usar `r_H=1`, `x=1-1/r`, `z=1-x`, `A=1-z²` y

\[
b=AB,\quad f=AF,\quad h=z^{-2}+z^2H,\quad w=z^4W.
\]

Para restaurar unidades, h se multiplica por r_H² y w se divide por r_H.
Las incógnitas son B,F,H,W y, en el sistema de primer orden,
`P=B_x,Q=H_x,V=W_x`. Nunca se fijan H o W a sus amplitudes analíticas.

La expansión regular en el horizonte proporciona

\[
F+H=1,\qquad
Q-\frac{8H}{1-H}+\frac{(1+H)^2(V-4W)^2}{2B}=0,
\]
\[
F_x+\frac{3FP}{B}+4H-\frac{3F(1+H)(V-4W)^2}{2B}=0,
\qquad W=q.
\]

La tercera relación es necesaria: omitirla y fijar F en el extremo exterior
de un intervalo cortado dio un BVP que sólo recuperaba el caso estático.
Puede verificarse expandiendo `b=b1(r-1)+b2(r-1)²+…`, y análogamente f,
h y w, en las variaciones. Las relaciones iniciales incluyen
`f1=4-2h0`,
`h1=2h0²/(2-h0)-h0² w1²/b1` y
`-8b1 f1-b1 f2-12b1 h0+16b1-3b2 f1+3f1 h0 w1²=0`.
Estas relaciones provienen de E_b,E_g,E_h; no de insertar Myers–Perry.

En infinito se exige normalización temporal B=1 y regularidad H_x=W_x=0.
El límite regular de la ecuación de F exige además F=1. En el adaptativo
son cuatro condiciones interiores y tres exteriores para siete variables.
En el espectral se sustituyen las ocho filas de extremos por las cuatro
relaciones del horizonte y B=F=1,H_x=W_x=0 en infinito. La fila F=1
representa el límite regular de su ODE, no una carga prescrita adicional.

Se comprueban B_H,F_H positivos y finitos, por lo que b1=2B_H y f1=2F_H
son positivos: ceros simples. El área requiere también 1+H_H>0.
Se excluye la extremalidad; no se extrapola la aceptación hasta ella.

## Solvers y representaciones

El espectral usa n nodos Chebyshev–Lobatto en [0,1], matrices D y D²,
Newton con Jacobiano por paso complejo y reducción de paso hasta 16 intentos.
El máximo predeterminado es 30 iteraciones. DCT-I construye los polinomios
interpolantes; sus derivadas son las que se verifican.

El adaptativo usa solve_bvp sobre `[epsilon,1-epsilon]`, con epsilon=0.003.
El corte es explícito: no se afirma que SciPy resuelva directamente los
extremos singulares. Las condiciones se transportan por Taylor de orden dos:
`y(0)≈y(e)-e y'(e)+e² y''(e)/2`, y análogamente en infinito.
Para definir esas condiciones, `y'=R(x,y)` y
`y''=∂xR+(∂yR)R`, obtenida mediante diferenciación direccional compleja.
Se estudian también epsilon=0.01 y 0.001, manteniendo tolerancia 1e-8.

La representación devuelta de B,H,W integra los splines cúbicos calculados
de P,Q,V y corrige linealmente la deriva para conservar sus dos valores de
borde. Es C2 y cuártica por tramos, como en el spike previo. F usa el spline
cúbico original, pues sólo se requiere F_x en las ecuaciones de Einstein.
Las derivadas usadas en diagnósticos provienen de estas representaciones;
no se sustituyen por R. El residuo de borde se recalcula después de reconstruir.

Las condiciones transportadas son aproximaciones locales, no certificación
del horizonte adaptativo exacto. La comparación entre cortes y con el solver
espectral permite medir su efecto en el conjunto estudiado.

## Continuación, fallos y normas

La semilla estática lleva una perturbación suave de amplitud 0.002. Cada
solución rotante se inicializa con la representación numérica anterior.
Se recorren `0,0.1,0.2,0.33,0.5,0.6` y, desde otro inicio estático,
`0,-0.1,-0.2,-0.33`. Un fallo reduce el paso a la mitad, con mínimo 1e-3
y hasta 20 fallos por destino. Se conservan todos los intentos.

Convergencia numérica y aceptación científica son estados distintos. En el
estudio de convergencia se pueden continuar candidatos de baja resolución;
cada fila conserva sus diagnósticos y su aceptación individual. Nunca se
declaran verificados sólo por el retorno exitoso del solver. No hay
interpretación de fallos como límites físicos.

En 401 puntos medios de `[0.005,0.995]`, restringidos al intervalo resuelto,
se desplazan coincidencias con los nodos. Las normas son máximos muestreados:

- Error absoluto en B,F,H,W contra el oráculo analítico, menor que 1e-6.
- Defecto de `B_xx,F_x,H_xx,W_xx` respecto a R, multiplicado respectivamente
  por `x²z²,xz,xz²,xz²`, menor que 1e-6. Son escalas fijas que regularizan
  factores singulares; no equivalen a un control uniforme en los extremos.
- Constraint `max|E_f f/r²|`, menor que 1e-6 en r_H=1.
- Einstein: norma coordenada del Hito 1, menor que 1e-6 en cinco radios
  `r/r_H=1.05,1.3,2,5,20` y tres ángulos `0.2,0.7,1.2`.
- Residuo máximo absoluto de condiciones regularizadas, menor que 1e-8.

El tensor se construye con jets numéricos b,f,h,w: un polinomio local de
orden dos reproduce cada jet y SymPy diferencia la métrica, mientras NumPy
contrae Einstein sin usar las ODE. Se comprueba restauración de escala y
un control negativo que modifica w y sus derivadas en un 1 %.
Falta de cobertura de la muestra tensorial rechaza el candidato.

El experimento registra diferencias entre métodos y entre resoluciones o
tolerancias sucesivas, además del error contra la referencia. Los casos
gruesos pueden fallar: son evidencia de convergencia, no una razón para
relajar umbrales. No se exige mejora monótona al alcanzar redondeo.

## Reproducción y alcance

`experiments/derive_vacuum.py` regenera código simbólico; la aceptación compara
su hash con el archivo presente. `experiments/myers_perry_bvp.py` produce
JSON, figura y procedencia. `checks/hito-2.ps1` incluye los hitos anteriores.
Todas las cachés y temporales se alojan en project/. No se añaden dependencias.

Se valida GR no extremal en una muestra exterior. No se certifican cotas
globales, singularidades de carta, interior ni EGB. Los cortes adaptativos y
la saturación espectral permanecen visibles en los resultados.
