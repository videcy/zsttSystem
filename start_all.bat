@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Run: python -m venv .venv
    exit /b 1
)

docker compose up -d chromadb
if errorlevel 1 (
    echo Failed to start ChromaDB. Check Docker Desktop.
    exit /b 1
)

call .venv\Scripts\Activate.bat
echo Starting zsttSystem API...
start "zsttSystem" .venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
echo ChromaDB: http://127.0.0.1:8001
echo zsttSystem: http://127.0.0.1:8000
