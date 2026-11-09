$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Le registre TP2 a été utilisé par MLflow 3.18.0 au TP6.
# Le client TP7 conserve 3.16.1, version de sérialisation du modèle.
$serveurPython = Join-Path (Split-Path $PSScriptRoot -Parent) 'tp6-workflows\.venv\Scripts\python.exe'
& $serveurPython -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
