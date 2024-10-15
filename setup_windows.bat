@echo off

REM 
REM 
if not exist "venv" (
    echo Creando el entorno virtual...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r src\backend\requirements.txt
) else (
    call venv\Scripts\activate
    echo El entorno virtual ya existe.
)

REM 
echo Lanzando el backend...
cd src\backend
REM 
start /b python app.py
set BACKEND_PID=%ERRORLEVEL%

REM 
cd ..\

REM 
cd frontend

npm ci
start /b npm run dev
set FRONTEND_PID=%ERRORLEVEL%

REM 
:cleanup
    echo Deteniendo el backend...
    taskkill /PID %BACKEND_PID% /F >nul 2>&1
    echo Deteniendo el frontend...
    taskkill /PID %FRONTEND_PID% /F >nul 2>&1
    exit /B

REM 
trap cleanup EXIT

REM 
wait %BACKEND_PID%
wait %FRONTEND_PID%

