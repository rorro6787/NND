@echo off
setlocal

REM Create the virtual environment
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
) ELSE (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
pip install -r src\backend\requirements.txt

REM Launch the backend
echo Launching backend...
cd src\backend
REM Assuming the backend runs with 'python app.py' or a similar command
start /B python app.py
SET BACKEND_PID=%PROCESS_ID%

REM Go back to the main directory
cd ..\..

REM Launch the frontend
echo Launching frontend...
cd frontend

npm ci
start /B npm run dev
SET FRONTEND_PID=%PROCESS_ID%

REM Function to stop processes when the script ends
:cleanup
    echo Stopping backend...
    taskkill /PID %BACKEND_PID% /F
    echo Stopping frontend...
    taskkill /PID %FRONTEND_PID% /F
    exit /B

REM Trap the interrupt signal (Ctrl+C) to run cleanup
REM The equivalent of trap in batch is not straightforward, so we'll use a simple method to keep the script running
:waitForProcesses
    REM Check if backend and frontend are still running
    timeout /T 1 > NUL
    tasklist | findstr /I "python.exe" > NUL
    if ERRORLEVEL 1 (
        goto cleanup
    )
    tasklist | findstr /I "npm.exe" > NUL
    if ERRORLEVEL 1 (
        goto cleanup
    )
    goto waitForProcesses

REM Start waiting for the processes
goto waitForProcesses