# Banco analítico Myers–Perry en D=5

## Fuente y parámetros

Se usa el ansatz de `docs/convenciones.md`, con coordenadas
`(t,r,theta,phi1,phi2)` y `g=r²`. Fuente primaria: Brihaye et al.,
[1010.0860v1](https://arxiv.org/html/1010.0860v1), ecuaciones (2.17)–(2.19)
(pp. 5–6), (3.6)–(3.7) (p. 8), (3.10) (p. 9) y (4.3) (p. 14).
El PDF local está inventariado por SHA-256 en `references/sources.json`.
En esta implementación se cotejaron las fórmulas con la versión HTML.

Los parámetros de entrada son `r_h`, `omega_h` y `G5`, con escala y constante
de Newton positivas. Definimos `q=r_h*omega_h`, `z=r_h/r` y `d=1-q²`.
Se exige `|q|<1/sqrt(2)`: así `r_h` es el horizonte exterior no extremal.
Un mismo signo común de los dos spins puede ser positivo o negativo.
No se incluyen configuraciones con spins de signos opuestos en este ansatz.

## Funciones y cantidades

La implementación radial reorganiza las expresiones publicadas como

\[
f=(1-z^2)(1-q^2z^2/d),\quad h=r^2(1+q^2z^4/d),\quad
b=fr^2/h,\quad w=\Omega_Hz^4/(d+q^2z^4).
\]

La factorización conserva el cero de `f` en el horizonte sin restar tres
términos casi iguales. Allí

\[
b_1=2(1-2q^2)/r_H,\quad f_1=b_1/d,\quad h_H=r_H^2/d.
\]

Las colas son `U=-r_h²/d`, `V=r_h⁴ q²/d`, `W=r_h³ q/d`.
En unidades `c=hbar=1` y con `G5` explícito,

\[
M=-3\pi U/(8G_5),\quad J_1=J_2=\pi W/(4G_5),\quad
T_H=\sqrt{b_1 f_1}/(4\pi),\quad A_H=2\pi^2r_H^2\sqrt{h_H},\quad
S=A_H/(4G_5).
\]

La identidad de Smarr se comprueba como
`2M/3 = T_H*S + Omega_H*(J1+J2)`. La primera ley se evalúa con diferencias
centradas en las direcciones independientes `(r_h,q)`, manteniendo `G5`
fijo, a dos pasos relativos `2e-5` y `1e-5`. No se confunde cada `J` con
el momento angular total.

## Evaluación independiente de Einstein

La métrica simbólica se construye en parámetros `mu=-U`, `a=r_h*q`:
`f=1-mu/r²+mu*a²/r⁴`, `h=r²+mu*a²/r²`, `b=r²*f/h`,
`w=mu*a/(r⁴+mu*a²)`. `mu` es dos veces el parámetro de masa de la nota
al pie 3 del paper; no es la masa física `M`. En el límite estático,
`mu=r_h²` y se recupera Schwarzschild–Tangherlini.

SymPy diferencia cada componente métrica hasta segundo orden y compila
expresiones numéricas; NumPy invierte la métrica y contrae conexión, Ricci
y Einstein con la convención interna. No se insertan las ecuaciones de
vacío como simplificaciones. Se contrastan estas funciones simbólicas con
la implementación radial y con `g_tt=-1+mu/r²`.

El control tensorial independiente incluye Minkowski esférico, de Sitter
en D=5 (`R=20K`, `R_ij=4K*g_ij`, `G_ij=-6K*g_ij`) y coordenadas planas
no lineales dependientes del tiempo. Multiplicar solamente `w` por 1.01
debe producir un residuo no nulo.

## Muestra, tolerancias y límites

El experimento usa `r_h=0.7,1,2`, `q=0,-0.33,0.33,0.68`, cinco radios
`r/r_h=1.05,1.3,2,5,20` y tres ángulos `theta=0.2,0.7,1.2`:
180 puntos y 25 componentes por punto. Cada entrada guarda también Ricci
y el escalar. La norma es `r_h² max_ij |G_ij/(s_i*s_j)|`, con
`s=(1,1,r,r,r)`. Es un diagnóstico coordenado adimensional, no un
invariante ni una cota global; no mide precisión uniforme en otros gauges.

La tolerancia de Einstein es `1e-8`, con margen frente a cancelación e
inversión en doble precisión. Smarr y horizonte usan `1e-13`; primera ley
y cargas asintóticas usan `2e-8` por truncamiento de diferencias y extracción
de colas. Las colas se muestrean en radios 100 y 1000 veces el horizonte;
la desviación de planitud debe disminuir más de 90 veces entre ellos.
Estas tolerancias se fijaron antes del primer barrido, no a partir de fallos.

Las funciones radiales admiten el horizonte, pero el tensor se evalúa sólo
en el exterior y fuera de los ejes. No se certifican extremalidad exacta,
interior, singularidades coordenadas, EGB ni un solver de contorno.
Se conserva el entorno del Hito 0, sin nuevas dependencias.
