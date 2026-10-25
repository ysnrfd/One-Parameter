@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Parameter Studio needs the project Python environment.
    echo.
    echo Install with Python 3.12 and uv:
    echo   uv venv --python 3.12 .venv
    echo   uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
    echo   uv pip install --python .venv\Scripts\python.exe -e .[export]
    echo.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import json,sys,urllib.request; from single_parameter_observatory import __version__; reply=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/health',timeout=2)); sys.exit((0 if reply.get('version')==__version__ else 2) if reply.get('service')=='single-parameter-studio' else 1)" >nul 2>&1
if errorlevel 2 (
    echo A different Studio version is already running on port 8765.
    echo Close its server window, then launch this copy again.
    pause
    exit /b 1
)
if not errorlevel 1 (
    start "" "http://127.0.0.1:8765/"
    exit /b 0
)
echo Opening Parameter Studio on http://127.0.0.1:8765
echo Keep this window open while using the studio. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m single_parameter_observatory studio --storage "%~dp0.spobs-studio"
if errorlevel 1 pause
