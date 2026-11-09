$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:MLFLOW_DISABLE_AGENT_HINT = '1'
$pythonTP7 = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
foreach ($script in @('verifier_modele','xai_importance','xai_global','xai_local','xai_lime','interpretables')) {
    Write-Host "Activité : $script"
    & $pythonTP7 "src/$script.py" 2>&1 | Tee-Object -FilePath "reports/$script.txt"
    if ($LASTEXITCODE -ne 0) { throw "Échec de $script ($LASTEXITCODE)" }
}
& $pythonTP7 -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
Write-Host 'Calculs terminés. Exécuter ensuite journaliser.py, rédiger la Model Card puis publier_card.py.'
