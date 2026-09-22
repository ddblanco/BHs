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
