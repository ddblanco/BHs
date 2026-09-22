# Problema estático de EGB en D=5

## Alcance y ansatz

Rama estática (sin rotación) de EGB: `w=0`, `h=r^2` en el ansatz de
`convenciones.md`. La métrica se vuelve diagonal,

\[
ds^2=-b(r)dt^2+\frac{dr^2}{f(r)}+r^2\left(d\theta^2+\sin^2\theta\,d\varphi_1^2
+\cos^2\theta\,d\varphi_2^2\right),
\]

el producto warped 2D × S³ estándar. El oráculo cerrado es
[Brihaye et al., 1010.0860v1](https://arxiv.org/html/1010.0860v1), ec. (2.20);
el límite `alpha_GB->0` coincide con el `q=0` de Myers–Perry ya verificado.

## Derivación directa desde el tensor

A diferencia del sistema rotante de Hito 2, aquí no se reconstruye una acción
reducida con el término de Gauss–Bonnet: 1010.0860v1 no publica esa acción
reducida en forma explícita (sólo ecs. (2.10)–(2.12), sin las ODE finales),
y reconstruirla para un ansatz con términos cruzados añadiría riesgo sin
necesidad, porque el caso estático es completamente diagonal.

`experiments/derive_static_egb.py` extiende el evaluador tensorial del
Hito 1 (`einstein.py`) para materializar el tensor de Riemann mixto completo
(antes sólo se contraía hasta Ricci) y agrega `gauss_bonnet`, que calcula
`H_{\mu\nu}` y `L_{GB}=R^2-4R_{\mu\nu}R^{\mu\nu}+R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}`
con la fórmula de `convenciones.md`. Ambas extensiones se validan contra
curvatura constante en D=5: `L_{GB}=120K^2`,
`H_{\mu\nu}=-12K^2g_{\mu\nu}=-\tfrac12(D-1)(D-2)(D-3)(D-4)K^2g_{\mu\nu}`.
También se comprueba que `H_{\mu\nu}` es no nulo sobre Schwarzschild–Tangherlini
D=5 (Einstein=0 pero EGB no se resuelve trivialmente con ese perfil GR: si
`H` fuera cero ahí, `alpha_GB` no deformaría nada).

Con b(r), f(r) simbólicos (SymPy `Function`), se construye la métrica
diagonal, se calcula `G_{\mu\nu}+\alpha_{\rm GB}H_{\mu\nu}` componente a
componente y se factoriza cada entrada diagonal. La componente tt, dividida
por el factor común `3b/(2r^3)`, da una ecuación que **no contiene a b ni a
b'**, sólo a f y f':

\[
4\alpha_{\rm GB}(f-1)f'-r\bigl(rf'+2f-2\bigr)=0
\quad\Longrightarrow\quad
f'=\frac{2r(1-f)}{r^2+4\alpha_{\rm GB}(1-f)}.
\]

Con `q=1-f`, esta ODE de primer orden es equivalente a
`r^2q+2\alpha_{\rm GB}q^2=\text{const}`; con `f(r_H)=0` la constante es
`r_H^2+2\alpha_{\rm GB}`, reproduciendo exactamente la solución cerrada
(y, en `alpha_GB=0`, Schwarzschild–Tangherlini `f=1-r_H^2/r^2`).

La componente rr, evaluada con la f' anterior, fuerza `b'/b=f'/f`; junto con
`b,f\to1` en infinito, esto fija `b=f` exactamente (no sólo proporcional). La
componente θθ (y, por la simetría de la S³ redonda, φ1φ1/φ2φ2) se anula
**idénticamente** al sustituir `b=f` y la f' derivada — no sólo en `f=1` —
confirmando la consistencia de Bianchi sin imponerla. Por último, la
solución cerrada de la ec. (2.20), sustituida en la f' derivada, da residuo
simbólico exactamente nulo. Estas cuatro comprobaciones (`verify_and_generate`
en `experiments/derive_static_egb.py`) son la puerta científica del hito: si
alguna falla, el script aborta con un error explícito en vez de generar
código.

## Sistema resultante: una sola ODE de primer orden, regular en r_H

A diferencia del sistema rotante (orden 7, singular en ambos extremos de la
coordenada compacta), el sistema estático se reduce a una única incógnita
`f=b` con una ODE de **primer orden**, y esa ODE es regular en el horizonte:
no hace falta ningún corte cerca de `x=0`. La única singularidad removible
está en `x=1` (infinito espacial, `z=1-x=1/r=0`), donde `f\to1` cancela el
numerador al mismo orden que el denominador. Por eso `cutoff` en
`static_egb_bvp.py` sólo actúa cerca de `x=1`.

Con `x=1-r_H/r`, `z=1-x=r_H/r` (r_H=1 en el solver) y `alpha_hat=\alpha_{\rm
GB}/r_H^2`, la ODE compacta generada es

\[
F_x=\frac{2(1-F)}{z\left(1+4\,\widehat\alpha\,z^2(1-F)\right)}.
\]

Al ser de primer orden, sólo se impone `F(0)=0` (definición del horizonte
como cero simple); `F(1)\approx1` (planitud asintótica) **no se impone**: se
verifica después, como diagnóstico independiente, siguiendo la misma lógica
que reservó `E_f` en Hito 2. Esto hace que ambos solvers queden exactamente
determinados (un número de ecuaciones igual al de incógnitas), sin la
sobre-determinación aparente de pedir dos condiciones a una ODE de orden 1.

## Solvers y verificación

`static_egb_bvp.py` implementa dos métodos independientes sobre `[0,1-cutoff]`:
Chebyshev–Lobatto con Newton amortiguado (Jacobiano por paso complejo,
igual que en Hito 2), y `solve_ivp` (RK45) integrando desde el horizonte.
La derivada representada para diagnósticos es la del polinomio de Chebyshev
(espectral) o una diferencia finita central de la salida densa de
`solve_ivp` (adaptativo) — nunca la ODE recalculada en el mismo punto, para
que comparar contra `rhs` sea una prueba real y no una tautología.

`static_egb_validation.py` reevalúa, fuera de nodos: error de perfil contra
el oráculo cerrado, residuo de la ODE con la derivada representada, y el
tensor completo `G_{\mu\nu}+\alpha_{\rm GB}H_{\mu\nu}` (las cinco componentes
diagonales, no sólo tt) sobre saltos locales de segundo orden de la solución
numérica, en los mismos radios/ángulos que Hito 1–2
(`r/r_H=1.05,1.3,2,5,20`, `\theta=0.2,0.7,1.2`). Sólo tt fue colocado por
cualquiera de los dos solvers: que rr y θθ también se satisfagan sobre la
solución numérica es una prueba independiente, no una repetición de la
derivación simbólica.

## Resultados observados y límites

`static_egb_benchmark.py` barre `alpha_hat=0,0.01,0.02,0.05,0.1,0.2,0.5`
(espectral n=32, adaptativo tol=1e-13, elegidos tras observar que la
amplificación de error de las matrices de diferenciación espectral y de la
diferencia finita central del adaptativo dominan por debajo de esas
resoluciones/tolerancias — ver `reports/hito-3.md` para las cifras), un
barrido de resolución espectral (n=16..48) y de tolerancia adaptativa
(1e-8..1e-13) en el caso más exigente (`alpha_hat=0.5`), y la distancia al
límite GR cerrado (sin solver) para cada `alpha_hat`, medida y no supuesta
a priori con un orden fijo.

No se cubre extremalidad, el interior, ni `alpha_GB` fuera del barrido
pequeño declarado. La rama de signo opuesto en la raíz cuadrada (sin límite
GR suave) no se persigue. Los residuos son máximos muestreados, no cotas
globales.
