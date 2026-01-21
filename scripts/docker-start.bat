@echo off
REM Seismic MLOps Pipeline - Docker Quick Start (Windows)
REM Usage: scripts\docker-start.bat

echo ==============================================
echo Seismic MLOps Pipeline - Docker Setup
echo ==============================================

REM Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is running...

REM Build the image
echo.
echo Building Docker image...
docker-compose build
if errorlevel 1 (
    echo ERROR: Docker build failed.
    pause
    exit /b 1
)

REM Start services
echo.
echo Starting services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start services.
    pause
    exit /b 1
)

REM Wait for services to be ready
echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo.
echo Checking service status...
docker-compose ps

echo.
echo ==============================================
echo Services are ready!
echo ==============================================
echo.
echo Access points:
echo   - API Server: http://localhost:8000
echo   - MLflow UI:  http://localhost:5000
echo   - Metrics:    http://localhost:8001/metrics
echo.
echo To run the full pipeline:
echo   docker-compose exec mlops python run_all_stages.py
echo.
echo To run quick validation:
echo   docker-compose exec mlops python src/stage8_cicd.py
echo.
echo To stop services:
echo   docker-compose down
echo.
pause
