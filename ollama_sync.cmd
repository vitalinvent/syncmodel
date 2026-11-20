@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        py -3 ollama_sync.py %*
    ) else (
        echo Python 3.10+ is required. Install a current Python version.
        set "EXITCODE=1"
        goto finish
    )
) else (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        python ollama_sync.py %*
    ) else (
        echo Python 3.10+ is required. Install Python and add it to PATH.
        set "EXITCODE=1"
        goto finish
    )
)
set "EXITCODE=%errorlevel%"
:finish
echo.
if not "%EXITCODE%"=="0" echo Ollama Model Transfer завершился с кодом %EXITCODE%.
pause
exit /b %EXITCODE%
