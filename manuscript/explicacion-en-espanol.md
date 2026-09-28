# Qué intenta medir este trabajo y cómo lo hace

Esta explicación acompaña el manuscrito [main.pdf](main.pdf).

La pregunta física es concreta: **si mantenemos fijo el momento angular de un agujero negro, ¿cómo cambia la masa mínima que necesita para tener horizonte cuando modificamos la teoría de Einstein añadiendo el término de Gauss–Bonnet?** El trabajo intenta responderla numéricamente, más allá del primer orden en el acoplamiento.

## El problema físico, antes del método

En esta familia, un agujero negro rotante tiene una masa mínima para cada momento angular. En el límite que alcanza esa masa se lo llama extremo y su temperatura es cero. La familia considerada vive en cinco dimensiones, donde hay dos planos independientes de rotación. Se toma el mismo momento angular en ambos: \(J_1=J_2=J\). Esa restricción simplifica las ecuaciones: las funciones desconocidas dependen únicamente del radio.

La teoría estudiada tiene acción

\[
I=\frac1{16\pi G}\int \sqrt{-g}\,[R+\alpha\mathcal L_{GB}]\,d^5x.
\]

El número \(\alpha\) controla cuánto pesa la corrección de Gauss–Bonnet. Cambiarlo significa comparar soluciones de teorías con distinto acoplamiento. No se está simulando un agujero negro cuyo acoplamiento varía físicamente con el tiempo.

Para \(\alpha=0\), la solución es Myers–Perry. A primer orden ya se conocía

\[
M_{\rm ext}(J,\alpha)
=\frac32\pi^{1/3}J^{2/3}+\pi\alpha+O(\alpha^2).
\]

Por lo tanto, la pendiente inicial es positiva e igual a \(\pi\): al encender la corrección, la masa extrema aumenta a \(J\) fijo. Eso ya es un resultado publicado, no la novedad del manuscrito. [Ma, Li y Lü](https://arxiv.org/abs/2009.00015).

Lo que se quiere conocer ahora es la pendiente cuando \(\alpha\) ya no es infinitesimal. Puede cambiar mucho aunque conserve el signo.

## Qué significa «potencial conjugado»

Acá «potencial» no significa un campo adicional ni una función del radio, como un potencial electrostático. Es **un número asociado a cada solución**, que mide una respuesta termodinámica:

\[
\boxed{\Psi=\left(\frac{\partial M}{\partial\alpha}\right)_{S,J}.}
\]

Es el cambio de masa por unidad de cambio de acoplamiento cuando se mantienen constantes la entropía y ambos momentos angulares. Es análogo a que la temperatura sea la derivada de la energía respecto de la entropía.

Su definición aparece en la primera ley extendida:

\[
dM=T\,dS+2\Omega_H\,dJ+\Psi\,d\alpha.
\]

El factor 2 suma las contribuciones de los dos planos: \(\Omega_1dJ_1+\Omega_2dJ_2=2\Omega_HdJ\). La extensión de la termodinámica a acoplamientos de Lovelock tiene antecedentes; para el caso estacionario hay un tratamiento explícito en [Neri y Liberati](https://arxiv.org/abs/2404.16981).

## El potencial en el caso rotante

El programa encuentra soluciones fijando el radio del horizonte \(r_H\), la velocidad angular \(\Omega_H\) y el acoplamiento \(\alpha\). Las cantidades \(M,S,J\) salen de esas soluciones. Al variar \(\alpha\) dejando fijos \(r_H,\Omega_H\), **en general cambian tanto \(S\) como \(J\)**.

Eso no impide calcular \(\Psi\). Aplicando la primera ley a ese recorrido se obtiene

\[
\left(\frac{\partial M}{\partial\alpha}\right)_{r_H,\Omega_H}
=T\left(\frac{\partial S}{\partial\alpha}\right)_{r_H,\Omega_H}
+2\Omega_H\left(\frac{\partial J}{\partial\alpha}\right)_{r_H,\Omega_H}
+\Psi.
\]

Despejando:

\[
\boxed{
\Psi=M_\alpha-T S_\alpha-2\Omega_H J_\alpha,
\qquad \text{derivadas a }r_H,\Omega_H\text{ fijos}.
}
\]

El sentido de esta expresión es quitar del cambio total de masa lo que corresponde al cambio de entropía y de momento angular. Lo que queda corresponde al cambio de teoría.

No se están igualando dos derivadas tomadas con restricciones diferentes. La igualdad es entre una derivada a \(S,J\) fijos y **una combinación de tres derivadas** tomadas a \(r_H,\Omega_H\) fijos. Tampoco faltan términos \(S T_\alpha\) o \(J\Omega_\alpha\): se evalúa la primera ley diferencial, en la que \(T\) y \(\Omega_H\) multiplican a \(dS\) y \(dJ\) y no se varían. Esos términos aparecerían al derivar la relación de Smarr \(2M=3TS+6\Omega_H J+2\alpha\Psi\), que es otra cosa: una relación algebraica entre las cargas de **una sola** solución, cuyos enteros son pesos de escala. Por eso el único factor que se repite en \(\Psi\) es el \(2\) que cuenta los dos momentos angulares, y no aparecen ni el \(3\) ni el \(6\).

Más generalmente, para cualquier recorrido de equilibrio parametrizado por \(\lambda\),

\[
\Psi=\frac{dM/d\lambda-T\,dS/d\lambda-2\Omega_H\,dJ/d\lambda}
{d\alpha/d\lambda},\qquad d\alpha/d\lambda\ne0.
\]

Esta expresión se aplica a rotantes estacionarios; no es una fórmula general para una evolución dinámica, como una fusión. Estacionario significa que la geometría no cambia con el tiempo, aunque el horizonte rote.

## Un ejemplo estático que muestra por qué hay que restar \(T S_\alpha\)

En el límite sin rotación se conocen expresiones cerradas:

\[
M=\frac{3\pi}{8}(r_H^2+2\alpha),\quad
T=\frac{r_H}{2\pi(r_H^2+4\alpha)},\quad
S=\frac{\pi^2}{2}r_H^3+6\pi^2\alpha r_H.
\]

A radio fijo, \(M_\alpha=3\pi/4\), que siempre es positivo. Pero ése no es el potencial conjugado, porque a radio fijo también aumenta la entropía: \(S_\alpha=6\pi^2r_H\). Entonces

\[
\Psi_{\rm static}=\frac{3\pi}{4}-T(6\pi^2r_H)
=\frac{3\pi}{4}\frac{4\alpha-3r_H^2}{r_H^2+4\alpha}.
\]

En particular, a \(\alpha=0\), \(\Psi=-9\pi/4\): **la masa aumenta si fijás el radio, pero disminuye si fijás la entropía**. Son comparaciones distintas, sin contradicción. Mantener la entropía requiere ajustar el radio.

En el caso rotante ocurre lo mismo, con el término adicional \(2\Omega_H J_\alpha\).

## Cómo obtiene las derivadas el programa

Primero resuelve numéricamente las ecuaciones gravitatorias con regularidad en el horizonte y planitud en el infinito. La discretización reemplaza las funciones métricas por un vector de valores \(u\), y las ecuaciones por un sistema

\[
R_N(u;\alpha,r_H,\Omega_H)=0.
\]

Una vez encontrada una solución, pregunta cómo debe cambiar \(u\) si cambia ligeramente \(\alpha\). Derivando implícitamente,

\[
\underbrace{\frac{\partial R_N}{\partial u}}_{A}
\underbrace{\frac{\partial u}{\partial\alpha}}_{u_\alpha}
=-\frac{\partial R_N}{\partial\alpha}.
\]

Se resuelve este sistema lineal para \(u_\alpha\). A partir de él se calculan \(M_\alpha,J_\alpha,S_\alpha\), y se insertan en la expresión del potencial. Al derivar la entropía se incluye tanto el cambio de geometría como la dependencia explícita de la fórmula de Wald en \(\alpha\).

La ventaja frente a resolver dos agujeros negros vecinos y restar sus masas es evitar la cancelación de números cercanos y el error de una diferencia finita real en el acoplamiento. El código usa paso complejo para evaluar derivadas locales y regla de la cadena para armar el jacobiano.

Aunque el sistema de respuesta sea lineal, el cálculo puede hacerse alrededor de una solución con cualquier \(\alpha\) del rango estudiado. **Linealizar la respuesta local no equivale a aproximar toda la familia por su expansión a primer orden alrededor de \(\alpha=0\).** Ése es el sentido válido de «no perturbativo en el acoplamiento» en este trabajo.

## Qué es exacto y qué no

| Objeto | Su condición |
|---|---|
| Primera ley, regla de la cadena y relación de escala | Identidades analíticas, bajo sus hipótesis |
| Fórmulas del caso estático | Expresiones analíticas cerradas |
| Ecuación \(A u_\alpha=-R_{N,\alpha}\) | Identidad para la rama diferenciable del problema discretizado |
| Solución computada de esa ecuación | Numérica; conserva errores de discretización, resolución y redondeo |
| Valores rotantes de \(\Psi\) a acoplamiento finito | Numéricos |
| Límite \(T\to0\), posición y profundidad del mínimo | Extrapolaciones y ajustes numéricos |

El procedimiento se describe así: **«Calculamos numéricamente el potencial mediante diferenciación implícita del sistema discretizado, sin diferencias finitas reales entre soluciones de acoplamientos vecinos».** El paso complejo también es un procedimiento numérico; evita cancelación sustractiva, no todos los errores.

Existe además otra evaluación, para \(\alpha\ne0\), a partir de Smarr:

\[
\Psi=\frac{2M-3TS-6\Omega_HJ}{2\alpha}.
\]

El manuscrito usa esta identidad como control. Cerca de \(\alpha=0\), extraer \(\Psi\) así es delicado por la cancelación del numerador y la división por un número pequeño. A \(\alpha=0\) no permite despejarlo.

## Por qué esto informa la masa extrema

En una rama extrema regular, \(T=0\). La primera ley, restringida a esa rama y a \(J\) fijo, queda

\[
dM_{\rm ext}=\Psi_{\rm ext}\,d\alpha.
\]

Luego

\[
\boxed{\left(\frac{\partial M_{\rm ext}}{\partial\alpha}\right)_J
=\Psi_{\rm ext}.}
\]

El programa se acerca al extremo con soluciones de temperatura positiva y extrapola \(\Psi\). La interpretación requiere que esa secuencia llegue a la rama extrema relevante y que la termodinámica tenga el límite regular supuesto. No obtiene una solución extrema exacta.

También se puede escribir, a temperatura positiva,

\[
\left(\frac{\partial S}{\partial\alpha}\right)_{M,J}=-\frac{\Psi}{T}.
\]

Ésta aclara el signo: cerca del extremo, si \(\Psi>0\), la entropía a masa y momento angular fijos **disminuye** al aumentar \(\alpha\). Eso es compatible con que la entropía de los estados extremos aumente: en ese segundo recorrido también aumenta la masa. La relación entre entropía y desplazamiento de extremalidad se discute en [Goon y Penco](https://arxiv.org/abs/1909.05254).

## Qué resultado sugieren los datos

Para comparar estados de distinto tamaño se usa \(y=\alpha/J^{2/3}\). Por escala,

\[
M_{\rm ext}=J^{2/3}\mu(y),\qquad \Psi_{\rm ext}=\mu'(y).
\]

Los valores guardados muestran una pendiente que empieza en \(\pi\simeq3.14\), baja a alrededor de (1.07) y después sube a alrededor de (1.44) en el extremo del rango estudiado, \(y\simeq0.51\). El descenso y la recuperación aparecen en los puntos extrapolados; la localización más precisa del mínimo depende del ajuste.

**La masa extrema sigue aumentando. Lo no monótono es su pendiente respecto de \(\alpha\).** Tampoco el factor de supresión de casi tres debe interpretarse como que la diferencia total de masas sea tres veces menor. La diferencia acumulada es

\[
M_{\rm ext}(J,\alpha)-M_{\rm ext}(J,0)
=J^{2/3}\int_0^{\alpha/J^{2/3}}\Psi_{\rm ext}(y')\,dy'.
\]

La posible contribución original es medir esa dependencia a acoplamiento finito y el desplazamiento del lugar donde \(\Psi\) cambia de signo en la familia no extrema. Los agujeros negros rotantes de esta teoría ya se habían construido: [Brihaye y colaboradores, 2010](https://arxiv.org/abs/1010.0860), y [Kleihaus, Kunz y Radu, 2023](https://arxiv.org/abs/2303.12471). La interpretación se limita a la familia y al rango de acoplamientos estudiados.

## Alcance de la medición

> Estudiamos cómo cambia la masa extrema de agujeros negros rotantes de cinco dimensiones cuando se añade a Einstein un término de Gauss–Bonnet. Consideramos dos momentos angulares iguales y resolvemos numéricamente las ecuaciones a acoplamiento finito. Para cada solución calculamos el potencial conjugado al acoplamiento: la parte de la variación de masa que queda después de descontar los cambios de entropía y momento angular. Obtenemos las derivadas mediante la respuesta lineal del problema discretizado. Al extrapolar hacia temperatura cero, ese potencial da la pendiente de la masa extrema a momento angular fijo. En el rango estudiado la pendiente permanece positiva, pero primero disminuye y luego aumenta: el signo conocido a primer orden persiste, mientras que su magnitud cambia considerablemente. El resultado corresponde a la teoría clásica de Einstein–Gauss–Bonnet y está sujeto a los errores de la solución y de la extrapolación numéricas.
