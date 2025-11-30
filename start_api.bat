@echo off
REM ============================================================
REM FramePack Studio API Server Startup Script
REM For Windows environments
REM ============================================================

echo.
echo ============================================================
echo     FramePack Studio API Server
echo ============================================================
echo.

REM 1. Navigate to script directory
cd /d "%~dp0"
echo [OK] Working directory: %CD%

REM 2. Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [WARN] No venv found, using system Python
)

REM 3. OPTIMIZATION: Prevent Out-Of-Memory crashes
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo [OK] PyTorch memory optimization enabled

REM 4. API Configuration
REM Set your API secret here or via environment variable
if "%FRAMEPACK_API_SECRET%"=="" (
    set FRAMEPACK_API_SECRET=your_secure_api_key_here
    echo [WARN] Using default API secret - change for production!
) else (
    echo [OK] API secret loaded from environment
)

REM Rate limiting (optional)
if "%RATE_LIMIT_REQUESTS%"=="" set RATE_LIMIT_REQUESTS=20
if "%RATE_LIMIT_WINDOW%"=="" set RATE_LIMIT_WINDOW=60
echo [OK] Rate limit: %RATE_LIMIT_REQUESTS% requests per %RATE_LIMIT_WINDOW% seconds

REM 5. Launch the API
echo.
echo ============================================================
echo    Starting FramePack Studio API Server...
echo ============================================================
echo.
echo  API Endpoints:
echo    * Health Check: http://localhost:8000/health
echo    * API Docs:     http://localhost:8000/docs  
echo    * Generate:     POST http://localhost:8000/generate
echo.
echo  Remember to set X-API-Key header in requests!
echo.
echo  (First run will download ~30GB models)
echo ============================================================
echo.

python api.py

pause
