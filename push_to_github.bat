@echo off
REM ============================================================
REM Pharma Job Search — Push to GitHub (Windows)
REM Double-click to commit and push local changes to GitHub
REM API keys and data files are protected by .gitignore
REM ============================================================

echo ========================================
echo   Push to GitHub
echo ========================================
echo.

REM Navigate to project directory (same folder as this script)
cd /d "%~dp0"

REM Check if we're in a git repo
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo ERROR: Not a git repository!
    pause
    exit /b 1
)

REM Show current branch
for /f "tokens=*" %%b in ('git branch --show-current') do set BRANCH=%%b
echo Branch: %BRANCH%
for /f "tokens=*" %%r in ('git remote get-url origin 2^>nul') do echo Remote: %%r
echo.

REM Check for changes
git diff --quiet HEAD 2>nul
set DIFF_RESULT=%errorlevel%
for /f %%c in ('git ls-files --others --exclude-standard ^| find /c /v ""') do set UNTRACKED=%%c

if %DIFF_RESULT%==0 if %UNTRACKED%==0 (
    echo No changes to commit. Everything is up to date.
    echo.
    pause
    exit /b 0
)

REM Show what's changed
echo --- Changed/modified files ---
git diff --name-status HEAD 2>nul
echo.
echo --- New untracked files ---
git ls-files --others --exclude-standard 2>nul
echo.

REM Safety check: show protected files
echo --- Protected by .gitignore (will NOT be pushed) ---
if exist config.yaml echo   config.yaml
if exist data\pharma_jobs.csv echo   data\pharma_jobs.csv
if exist data\pharma_jobs.xlsx echo   data\pharma_jobs.xlsx
if exist data\evaluations.json echo   data\evaluations.json
if exist data\resume_profile.json echo   data\resume_profile.json
if exist data\reviewed.json echo   data\reviewed.json
echo.

REM Get date for default message
for /f "tokens=2 delims==" %%d in ('wmic os get localdatetime /value') do set DT=%%d
set DEFAULT_MSG=Update pharma-job-search (%DT:~0,4%-%DT:~4,2%-%DT:~6,2% %DT:~8,2%:%DT:~10,2%)

echo Enter a commit message (or press Enter for default):
echo   Default: "%DEFAULT_MSG%"
echo.
set /p CUSTOM_MSG="> "

if "%CUSTOM_MSG%"=="" (
    set "COMMIT_MSG=%DEFAULT_MSG%"
) else (
    set "COMMIT_MSG=%CUSTOM_MSG%"
)

echo.
echo Staging files...
git add -A

echo Committing...
git commit -m "%COMMIT_MSG%"
if errorlevel 1 (
    echo.
    echo ERROR: Commit failed! See message above.
    pause
    exit /b 1
)

echo.
echo Pushing to GitHub...
git push origin %BRANCH%

if errorlevel 1 (
    echo.
    echo ERROR: Push failed! Check your network connection or GitHub credentials.
) else (
    echo.
    echo ========================================
    echo   Successfully pushed to GitHub!
    echo ========================================
    echo.
    git log --oneline -1
)

echo.
pause
