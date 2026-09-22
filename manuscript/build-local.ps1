# Compile using installed MiKTeX; keep configuration, logs and temporary files
# in the project. Do not run installation, maintenance or registry updates.
$ErrorActionPreference = 'Stop'
$paperRoot = Split-Path $PSScriptRoot -Parent
$paperCache = Join-Path $paperRoot '.cache/miktex'
$texExecutable = (Get-Command pdflatex -ErrorAction Stop).Source
$texInstall = Split-Path (Split-Path (Split-Path (Split-Path $texExecutable)))
$settings = @{
    MIKTEX_USERCONFIG = "$paperCache/config"
    MIKTEX_USERDATA = "$paperCache/data"
    MIKTEX_USERINSTALL = $texInstall
    MIKTEX_CORE_USERLOGDIRECTORY = "$paperCache/log"
    MIKTEX_CORE_NOREGISTRY = 'true'
    TEMP = "$paperCache/tmp"
    TMP = "$paperCache/tmp"
}
$previous = @{}
foreach ($folder in @('config', 'data', 'log', 'tmp')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $paperCache $folder) | Out-Null
}
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    Push-Location $PSScriptRoot
    try {
        foreach ($pass in 1..3) {
            & $texExecutable --miktex-disable-maintenance --miktex-disable-diagnose --disable-installer -interaction=nonstopmode -halt-on-error -jobname=main main.tex > "$paperCache/pass-$pass.txt"
            if ($LASTEXITCODE -ne 0) {
                Get-Content "$paperCache/pass-$pass.txt" -Tail 30
                throw "LaTeX pass $pass failed."
            }
        }
        Get-Content "$paperCache/pass-3.txt" -Tail 3
    } finally { Pop-Location }
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
