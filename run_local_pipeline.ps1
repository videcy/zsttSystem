$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".env")) {
    throw "Missing .env. Create it from .env.example first."
}

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
    throw "Missing .venv\\Scripts\\python.exe. Create the local virtual environment first."
}

$stage = "all"
if ($args.Length -gt 0 -and $args[0]) {
    $stage = $args[0]
}

& ".venv\\Scripts\\python.exe" run_pipeline.py --stage $stage
