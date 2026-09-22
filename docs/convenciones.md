# Convenciones físicas internas

Este documento es la autoridad única para toda derivación, implementación y
comparación numérica del proyecto. Si una fuente usa otra normalización, se
traduce aquí antes de copiar una ecuación. Las etiquetas **Comprobado**,
**Decisión interna** e **Inferencia pendiente** tienen el sentido definido en
`references/README.md`.

## Alcance del benchmark

**Comprobado.** El mínimo científico es gravedad de
Einstein--Gauss--Bonnet (EGB), asintóticamente plana, en (D=5), con horizonte
(S^3) y dos momentos angulares de igual magnitud. Es exactamente el sector de
1010.0860v1: resumen y p. 2; ansatz en p. 3, ec. (2.8). El caso AdS de
0801.1021v4 y las teorías cúbicas/cuárticas de las otras fuentes no forman
parte de este mínimo.

## Índices, firma y curvatura

**Decisión interna.** Se usan índices griegos
(mu,\nu,\ldots=0,\ldots,4), firma mostly-plus
((- + + + +)), conexión de Levi-Civita y

\[
R^\rho{}_{\sigma\mu\nu}
=\partial_\mu\Gamma^\rho{}_{\nu\sigma}
-\partial_\nu\Gamma^\rho{}_{\mu\sigma}
+\Gamma^\rho{}_{\mu\lambda}\Gamma^\lambda{}_{\nu\sigma}
-\Gamma^\rho{}_{\nu\lambda}\Gamma^\lambda{}_{\mu\sigma},
\qquad
R_{\sigma\nu}=R^\rho{}_{\sigma\rho\nu}.
\]

Se define (R=g^{\mu\nu}R_{\mu\nu}) y
(G_{\mu\nu}=R_{\mu\nu}-\tfrac12R g_{\mu\nu}). La firma se ve en el ansatz
de 1010.0860v1, p. 3, ec. (2.8), pero el paper no declara la definición de
Riemann/Ricci; por eso esta última es una decisión interna y su compatibilidad
de signos se audita con las identidades del final de este documento.

## Acción, Gauss--Bonnet y ecuaciones

**Decisión interna.** El acoplamiento sin ambigüedad en archivos y código será
`alpha_gb`, escrito (alpha_{\rm GB}):

\[
I_{\rm bulk}=\frac{1}{16\pi G_5}\int d^5x\sqrt{-g}
\left(R+\alpha_{\rm GB}\mathcal L_{\rm GB}\right),
\]

\[
\mathcal L_{\rm GB}
=R^2-4R_{\mu\nu}R^{\mu\nu}
+R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma},
\]

\[
G_{\mu\nu}+\alpha_{\rm GB}H_{\mu\nu}=0,
\]

\[
H_{\mu\nu}=2\left(
R_{\mu\rho\sigma\lambda}R_\nu{}^{\rho\sigma\lambda}
-2R_{\mu\rho\nu\sigma}R^{\rho\sigma}
-2R_{\mu\rho}R_\nu{}^\rho+R R_{\mu\nu}\right)
-\frac12g_{\mu\nu}\mathcal L_{\rm GB}.
\]

La forma de (mathcal L_{\rm GB}) y de (H_{\mu\nu}) está comprobada en
1010.0860v1, pp. 2--3, ecs. (2.2)--(2.4). Los términos de borde de sus
ecs. (2.5)--(2.7) son necesarios para la acción variacional y cargas, pero no
se agregan a las ecuaciones de bulk.

## Normalización de `alpha`

1010.0860v1 escribe (R+\alpha_{1010}\mathcal L_{\rm GB}/4) (p. 2,
ec. (2.1)). Por definición interna,

\[
\boxed{\alpha_{1010}=4\alpha_{\rm GB}},\qquad
\boxed{\alpha_{\rm GB}=\alpha_{1010}/4}.
\]

La misma relación vale para 0801.1021v4, ec. (2.1) (su localizador de página
es provisional; véase la nota de lectura). En la notación
Lovelock de 2607.07418v1, p. 1, ec. (1), el coeficiente relativo del término
cuadrático es (alpha_2/\alpha_1); el paper define en p. 4
(alpha_{2607}=4\alpha_2/\alpha_1). Por lo tanto,

\[
\alpha_{\rm GB}=\frac{\alpha_2}{\alpha_1}
=\frac{\alpha_{2607}}4,
\qquad
\alpha_{2607}=\alpha_{1010}.
\]

2606.07070v2 estudia una EFT en (D=4) con acoplamientos
(\lambda_3,\lambda_4,\widetilde\lambda_4) (p. 3, ecs. (1.1)--(1.2)). No
existe una traducción de esos símbolos a (alpha_{\rm GB}); inventarla queda
prohibido.

## Constante de Newton y dimensiones

**Decisión interna.** `G5` denota (G_5), la constante de Newton en cinco
dimensiones, y nunca el (G_4) de 2606.07070v2. Se conserva explícita en
fórmulas físicas; sólo un experimento que lo declare puede usar (G_5=1).
Con (c=\hbar=1), ([G_5]=L^3) y
([\alpha_{\rm GB}]=[\alpha_{1010}]=L^2).

Las dimensiones de coordenadas y funciones quedan fijadas por

\[
[r]=[t]=L,\qquad
[\theta]=[\varphi_1]=[\varphi_2]=1,
\]

\[
[b]=[f]=1,\qquad [g]=[h]=L^2,\qquad
[w]=[\Omega_H]=L^{-1}.
\]

Los ángulos son adimensionales. Con ([ds^2]=L^2), estas asignaciones hacen
que cada término de (ds^2) tenga dimensión (L^2), y fijan la restauración de
unidades después de trabajar con (r_H=1).

1010.0860v1 llama (G) a (G_5); por ejemplo, p. 8, ec. (3.6), y
(V_3=2\pi^2). 0801.1021v4 también escribe (G), pero sus figuras numéricas
usan (G=1). 2607.07418v1 fija
(alpha_1=(16\pi G)^{-1}), p. 1. 2606.07070v2 usa
(kappa^2=8\pi G_4) y luego (G_4=1), p. 5, ec. (2.1).

## Coordenadas, escala y compactificación

La coordenada física (r) es la de área de la base de la fibración, fijada
por el gauge (g(r)=r^2). El horizonte exterior está en (r=r_H>0).

**Decisión interna.** La escala de referencia es (L_\star=r_H). Las
variables adimensionales canónicas son

\[
\rho=\frac r{r_H},\qquad
\widehat\alpha_{\rm GB}=\frac{\alpha_{\rm GB}}{r_H^2},\qquad
\widehat\Omega_H=r_H\Omega_H,\qquad
\widehat h=\frac h{r_H^2}.
\]

Así, un valor de acoplamiento publicado se convierte mediante
(widehat\alpha_{\rm GB}=\alpha_{1010}/(4r_H^2)). La elección (r_H=1) de
1010.0860v1, p. 10, es una fijación de escala, no una identidad dimensional.

La coordenada compacta interna será

\[
x=1-\frac{r_H}{r}=1-\rho^{-1}\in[0,1],\qquad
r=\frac{r_H}{1-x},\qquad
\frac{d}{dr}=\frac{(1-x)^2}{r_H}\frac{d}{dx}.
\]

Por convención, (x=0) es el horizonte y (x=1) el infinito espacial.
2607.07418v1 usa en cambio (r-r_H=\tan x_{\rm Akribeia}), p. 4; no se
identifican ambas (x) sin aplicar la transformación explícita.

## Ansatz interno y condiciones de borde

Se conserva la forma coordenada del benchmark primario (1010.0860v1, p. 3,
ec. (2.8)):

\[
\begin{aligned}
ds^2={}&\frac{dr^2}{f(r)}+r^2d\theta^2
+h(r)\sin^2\theta\,[d\varphi_1-w(r)dt]^2\\
&+h(r)\cos^2\theta\,[d\varphi_2-w(r)dt]^2
+[r^2-h(r)]\sin^2\theta\cos^2\theta
(d\varphi_1-d\varphi_2)^2-b(r)dt^2,
\end{aligned}
\]

con (0\leq\theta\leq\pi/2) y
(0\leq\varphi_1,\varphi_2<2\pi). Se usan siempre las funciones minúsculas
(b,f,h,w); (g) queda eliminado por (g=r^2).

Para soluciones no extremales, las condiciones son

\[
f(r_H)=b(r_H)=0,\quad w(r_H)=\Omega_H,\quad h(r_H)=h_H>0,
\quad f'(r_H)>0,\quad b'(r_H)>0,
\]

con ceros simples de (f,b), como en 1010.0860v1, pp. 4--5, ec. (2.15). En
infinito asintóticamente plano,

\[
f\to1,\qquad b\to1,\qquad h/r^2\to1,\qquad w\to0,
\]

y los órdenes detallados son los de p. 5, ec. (2.16). Las configuraciones
extremales, donde los ceros dejan de ser simples, constituyen un problema de
borde separado.

## Traducción de símbolos

| Concepto | Interno | 1010.0860v1 | 0801.1021v4 | 2607.07418v1 |
|---|---|---|---|---|
| Dimensión | (D=5) | (d=5) | (d=5) | (D=2n+1), usar (n=2) |
| Newton | (G_5) | (G) | (G) | (G), (alpha_1=(16\pi G)^{-1}) |
| GB en la acción | (alpha_{\rm GB}) | (alpha/4) | (alpha/4) | (alpha_2/\alpha_1) |
| Acoplamiento publicado | (4\alpha_{\rm GB}) | (alpha) | (alpha) | (alpha=4\alpha_2/\alpha_1) |
| Horizonte | (r_H) | (r_H) | (r_h) | (r_H) |
| Giro del horizonte | (Omega_H) | (Omega_H=w(r_H)) | (w_h=\omega_h) | (Omega_H=w(r_H)) |
| Squashing en horizonte | (h_H) | (h_H) | (h_h) | (Theta_H) |
| Ángulos | (\varphi_1,\varphi_2) | (\varphi_1,\varphi_2) | (\varphi,\psi) | fibra (zeta) |
| Cola asintótica | (mathcal U,\mathcal V,\mathcal W) | (mathcal U,\mathcal V,\mathcal W) | (f_2,b_2,w_4) | no tabulada en v1 |
| Compacta | (x=1-r_H/r) | no fija una | no fija una | (r-r_H=\tan x_{\rm Akribeia}) |

La forma con uno-formas de 1010.0860v1, pp. 3--4, ec. (2.9), usa
(2\theta=\bar\theta),
(\varphi_1-\varphi_2=\phi) y
(\varphi_1+\varphi_2=\psi). El signo aparente
((\sigma_3+2w,dt)^2) no autoriza a cambiar el signo de (w) en la forma
coordenada; es consecuencia de esa transformación angular.

## Identidades sensibles a signos para auditoría independiente

1. Variar la acción interna debe producir exactamente
   (G_{\mu\nu}+\alpha_{\rm GB}H_{\mu\nu}=0). Tras usar
   (alpha_{1010}=4\alpha_{\rm GB}), deben coincidir todos los signos de
   1010.0860v1, pp. 2--3, ecs. (2.3)--(2.4), en especial los términos
   (-2R_{\mu\rho\nu\sigma}R^{\rho\sigma}) y
   (-\tfrac12g_{\mu\nu}\mathcal L_{\rm GB}).
2. En el gauge (g=r^2), el corchete de la primera integral de p. 4,
   ec. (2.12), se vuelve
   (r^2-\alpha_{1010}(f-4+3h/r^2)). Es el negativo exacto del corchete
   dentro de la derivada radial de la ec. (2.13). La diferencia sólo puede
   absorberse en la orientación de la constante de integración; no en un
   cambio silencioso de (w) o de (alpha).

Como control adicional, p. 5 da (b=1+\mathcal U/r^2+\cdots), mientras p. 8,
ec. (3.6), da
(E=-3V_3\mathcal U/(16\pi G_5)): masa positiva exige
(\mathcal U<0), tal como en el límite Myers--Perry de p. 6,
ec. (2.19).

## Estado de la auditoría de signos

**Comprobado por revisión independiente.** Las dos identidades anteriores
fueron reconstruidas desde 1010.0860v1. La variación GB pasó el control de
curvatura constante en (D=5), donde
(H_{\mu\nu}=-12K^2g_{\mu\nu}), y la rama estática pasó la identidad
(r^2q+2\alpha_{\rm GB}q^2=r_H^2+2\alpha_{\rm GB}), con (q=1-f).
También se confirmó la equivalencia por signo de las ecs. (2.12)--(2.13),
incluido el control asintótico (w'=-4\mathcal W/r^5). La derivación completa
queda registrada en `reports/sdd/task-1-report.md`.

1010.0860v1 no publica las ODE completas (p. 4, después de la ec. (2.12)).
Por ello, toda implementación futura de componentes aún deberá conservar como
tests de regresión Myers--Perry (pp. 5--6, ec. (2.17)) y la rama estática EGB
(p. 6, ec. (2.20)); esto ya no es una revisión documental pendiente, sino un
requisito de aceptación del solver.
