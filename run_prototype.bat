@echo off
echo ========================================================
echo    VOICE GUARD — AI Real-Time Voice Clone Detection
echo               SIH Hackathon Prototype
echo ========================================================
echo.

echo [1/2] Starting Voice Guard FastAPI ML Backend on http://127.0.0.1:8000 ...
start "Voice Guard Backend" cmd /k "cd backend && python main.py"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Voice Guard React + Vite Frontend on http://localhost:5173 ...
start "Voice Guard Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================================
echo   Voice Guard is running!
echo   Frontend: http://localhost:5173
echo   Backend API: http://127.0.0.1:8000
echo   API Docs: http://127.0.0.1:8000/docs
echo ========================================================
pause
