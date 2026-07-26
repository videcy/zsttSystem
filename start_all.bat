@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run: python -m venv .venv
    exit /b 1
)

docker compose up -d --wait chromadb neo4j
if errorlevel 1 (
    echo Failed to start ChromaDB or Neo4j. Check Docker Desktop and .env.
    exit /b 1
)

call .venv\Scripts\Activate.bat

powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; try { $response=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3; if ($response.StatusCode -eq 200) { exit 0 } } catch {}; $listener=Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -First 1; if ($listener) { exit 2 }; exit 1"
set "api_state=%errorlevel%"
if "%api_state%"=="0" (
    echo zsttSystem API is already running at http://127.0.0.1:8000
    goto services
)
if "%api_state%"=="2" (
    echo Port 8000 is already occupied by an unhealthy or unrelated process.
    echo Stop that process, then run start_all.bat again.
    echo Inspect it with: netstat -ano ^| findstr :8000
    exit /b 1
)

echo Starting zsttSystem API...
start "zsttSystem" .venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
powershell -NoProfile -Command "$ready=$false; for ($i=0; $i -lt 45; $i++) { try { $response=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($response.StatusCode -eq 200) { $ready=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if (-not $ready) { exit 1 }"
if errorlevel 1 (
    echo zsttSystem API did not become ready. Check the zsttSystem window for logs.
    exit /b 1
)

:services
echo ChromaDB: http://127.0.0.1:8001
echo Neo4j Browser: http://127.0.0.1:7474
echo zsttSystem: http://127.0.0.1:8000
