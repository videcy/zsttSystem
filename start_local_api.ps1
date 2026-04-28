$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".env")) {
    throw "Missing .env. Create it from .env.example first."
}

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    throw "Missing .venv\\Scripts\\python.exe. Create the local virtual environment first."
}

& ".venv\\Scripts\\python.exe" -m uvicorn src.main:app --host 127.0.0.1 --port 8000
