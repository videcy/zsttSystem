@echo off
chcp 65001 >nul
REM ===== zsttSystem 全栈启动脚本 =====
REM 在 Windows 中双击运行（需要以管理员身份运行一次来开放端口）
REM 按依赖顺序启动：Embedding -> LightRAG -> zsttSystem，每步轮询健康检查直到就绪

cd /d D:\python\zsttSystem1.1\zsttSystem
call .venv\Scripts\Activate.bat

echo ============================================
echo  Step 1/3: 本地 Embedding 服务 (port 11435)
echo ============================================
start "EmbedServer" .venv\Scripts\python.exe -m src.utils.embedding_server --port 11435
call :wait_health "Embed" http://127.0.0.1:11435/health 30
if errorlevel 1 goto :startup_failed

echo ============================================
echo  Step 2/3: LightRAG 检索引擎 (port 9621)
echo ============================================
SET EMBEDDING_DIM=384
start "LightRAG" .venv\Scripts\python.exe run_lightrag.py --port 9621 --llm-binding openai --key zstt_local_dev_key
call :wait_health "LightRAG" http://127.0.0.1:9621/health 90
if errorlevel 1 goto :startup_failed

echo ============================================
echo  Step 3/3: zsttSystem API (port 8000)
echo ============================================
start "zsttSystem" .venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
call :wait_health "zsttSystem" http://127.0.0.1:8000/health 60
if errorlevel 1 goto :startup_failed

echo.
echo ============================================
echo  All services started!
echo  Open http://127.0.0.1:8000 in your browser.
echo ============================================
pause
exit /b 0

:startup_failed
echo.
echo ============================================
echo  启动失败：上面标记 FAIL 的服务未在超时内就绪。
echo  请查看对应服务窗口的报错信息（通常是依赖/端口/模型加载问题）。
echo ============================================
pause
exit /b 1

REM ---------------------------------------------------------------
REM :wait_health  <显示名>  <健康URL>  <最大等待秒数>
REM 每秒轮询一次，端口开始响应即视为就绪；超时则返回错误码 1
REM ---------------------------------------------------------------
:wait_health
setlocal
set "name=%~1"
set "url=%~2"
set "max=%~3"
set /a count=0
echo|set /p="  正在等待 %name% 就绪 "
:wh_loop
curl -s -o nul "%url%"
if not errorlevel 1 (
    echo.
    echo   %name%: OK
    endlocal & exit /b 0
)
set /a count+=1
if %count% geq %max% (
    echo.
    echo   %name%: FAIL ^(超过 %max% 秒仍未就绪^)
    endlocal & exit /b 1
)
echo|set /p="."
timeout /t 1 /nobreak >nul
goto :wh_loop
