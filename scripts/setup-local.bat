@echo off
REM Seismic MLOps Pipeline - Local Setup (Windows)
REM Usage: scripts\setup-local.bat

echo ==============================================
echo Seismic MLOps Pipeline - Local Setup
echo ==============================================

REM Check Python version
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create data directories
echo.
echo Creating data directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\bronze" mkdir data\bronze
if not exist "data\silver" mkdir data\silver
if not exist "data\gold" mkdir data\gold
if not exist "models" mkdir models
if not exist "mlruns" mkdir mlruns
if not exist "feature_store" mkdir feature_store

REM Check Ollama (optional)
echo.
echo Checking Ollama (optional for LLM features)...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo Ollama is installed.
    echo To enable LLM features, run: ollama pull llama3.1:8b
) else (
    echo Ollama not installed. LLM features will be disabled.
    echo To install: https://ollama.ai/download
)

echo.
echo ==============================================
echo Setup complete!
echo ==============================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To run the pipeline:
echo   python run_all_stages.py
echo.
echo To run quick validation:
echo   python src/stage8_cicd.py
echo.
pause
