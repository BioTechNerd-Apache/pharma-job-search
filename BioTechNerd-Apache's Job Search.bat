@echo off
REM ============================================================
REM BioTechNerd-Apache's Job Search — Dashboard Launcher (Windows)
REM Double-click to view your latest job search results in a browser
REM ============================================================

set PORT=8501
set URL=http://localhost:%PORT%

echo ========================================
echo   BioTechNerd-Apache's Job Search Dashboard
echo ========================================
echo.
echo Loading latest data from: %~dp0data\
echo.

REM Kill any existing Streamlit on this port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do taskkill /PID %%a /F >nul 2>&1

REM Start Streamlit dashboard
echo Starting dashboard...
cd /d "%~dp0"
start "" python -m streamlit run src/dashboard.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false

REM Wait for server to be ready
timeout /t 5 /nobreak >nul

REM Open in default browser
start %URL%

echo.
echo Dashboard running at: %URL%
echo.
echo Click the 'Run New Search' button in the sidebar to search for new jobs.
echo Or run from Terminal:  python job_search.py
echo.
echo Close this window to stop the dashboard.
pause
