# Entrada externa opcional: figura 1b

`1010.0860v1-figure-1b.png` **no está incluido** en esta distribución.
Verificación realizada el 22 de septiembre de 2026:

- [Ficha de arXiv:1010.0860v1](https://arxiv.org/abs/1010.0860v1).
- Su enlace «view license» apunta a [arXiv non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html).

Esa licencia concede distribución a arXiv; no concede una licencia general de
redistribución a terceros. La imagen no queda cubierta por MIT ni por CC BY 4.0
de este proyecto. Obtenerla para la reproducción local no autoriza a publicarla.

Para recuperar el mismo insumo, ejecutar desde la raíz en PowerShell:

```powershell
New-Item -ItemType Directory -Force references/data | Out-Null
Invoke-WebRequest -UseBasicParsing 'https://arxiv.org/html/1010.0860v1/profiles-alpha.png' -OutFile references/data/1010.0860v1-figure-1b.png
Get-FileHash -Algorithm SHA256 references/data/1010.0860v1-figure-1b.png
.venv/Scripts/python -m pytest tests/test_egb_rotating_paper_profiles.py
.venv/Scripts/python checks/verify_recorded_digests.py
```

SHA-256 esperado, conservado del resultado original:

```text
a5760ea5b2d2ae6c397560bef6431e55acd37e72fc31ce7b10a1ea0006673543
```

Si el recurso deja de estar disponible, consultar el
[HTML de la versión](https://arxiv.org/html/1010.0860v1) o su
[fuente](https://arxiv.org/src/1010.0860v1). Una rasterización alternativa puede
cambiar las coordenadas de píxeles: no reemplazar el hash ni las calibraciones
para hacer pasar los controles. El procedimiento publicado requiere los bytes
indicados arriba.

Sin la imagen se omiten solamente los dos tests de extracción y rechazo del
perfil de vacío. El verificador informa la entrada no comprobada; no la cuenta
como una reproducción. Si la imagen está presente, exige su hash original.
Para regenerar la comparación, una vez obtenida:

```powershell
.venv/Scripts/python experiments/egb_rotating_paper_profiles.py
```


## Entrada externa opcional: el EPS vectorial de la misma figura

`1010.0860-src/profiles-alpha.eps` **tampoco está incluido**, por la misma razón:
la licencia de arXiv no concede redistribución a terceros. Es la figura 1b tal
como la envió el autor, PostScript de gnuplot, donde cada curva es una polilínea
con sus propios vértices. Leerla evita por completo la digitalización de píxeles.

Para obtenerla desde la raíz del repositorio:

```bash
curl -sSL -o /tmp/1010.0860.tar.gz https://arxiv.org/e-print/1010.0860
mkdir -p references/data/1010.0860-src
tar -xzf /tmp/1010.0860.tar.gz -C references/data/1010.0860-src profiles-alpha.eps
sha256sum references/data/1010.0860-src/profiles-alpha.eps
PYTHONPATH=src python experiments/egb_rotating_paper_eps_profiles.py
```

SHA-256 esperado del EPS, registrado en el resultado:

```text
ed6981a241131041955a1d1ac469621074f900a209e7089e1bd46b61c897e142
```

La calibración de ejes no se escribe a mano: se ajusta con las marcas impresas
del propio archivo y se rechaza si no son colineales dentro del redondeo de un
paso de dispositivo. Sin el EPS, `experiments/egb_rotating_paper_eps_profiles.py`
se detiene con un mensaje; el resultado guardado
`results/egb-rotating-paper-eps-profiles.json` conserva las coordenadas extraídas
y la comparación, que es lo único que se redistribuye.
