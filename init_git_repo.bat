@echo off
cd /d "%~dp0"
setlocal EnableDelayedExpansion

REM ============================================================
REM  [Git] Init this folder as a Git repository (universal)
REM  Copy this file to any project folder and double-click / run.
REM  - Creates repo on branch main (if not already a repo)
REM  - Writes a generic .gitignore only when missing
REM  Does NOT commit. You commit yourself when ready.
REM ============================================================

echo ========================================
echo   [Git] Init Repository  (universal)
echo ========================================
echo.
echo Folder: %CD%
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found. Install Git first.
    goto END
)

set "ALREADY=0"
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 set "ALREADY=1"

if "!ALREADY!"=="1" (
    for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do cd /d "%%i"
    for /f "delims=" %%i in ('git branch --show-current 2^>nul') do set "BRANCH=%%i"
    if not defined BRANCH set "BRANCH=main"
    echo [INFO] Already a git repo.
    echo Root:   %CD%
    echo Branch: !BRANCH!
    echo.
    echo Skip git init. Will only check .gitignore.
    echo.
) else (
    echo [1/2] git init -b main
    git init -b main
    if errorlevel 1 (
        echo [WARN] git init -b main failed, try plain git init ...
        git init
        if errorlevel 1 (
            echo [ERROR] git init failed.
            goto END
        )
        git branch -M main >nul 2>&1
    )
    set "BRANCH=main"
    echo OK
    echo.
)

REM ---- .gitignore (do not overwrite existing) ----
if exist ".gitignore" (
    echo [2/2] .gitignore already exists - skip
) else (
    echo [2/2] Writing generic .gitignore
    (
        echo # OS / IDE
        echo .DS_Store
        echo Thumbs.db
        echo .idea/
        echo .vscode/
        echo *.iml
        echo *.swp
        echo *~
        echo.
        echo # Logs / env / secrets
        echo *.log
        echo .env
        echo .env.*
        echo !.env.example
        echo *.pem
        echo credentials.json
        echo secrets/
        echo.
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo .venv/
        echo venv/
        echo .pytest_cache/
        echo .mypy_cache/
        echo dist/
        echo build/
        echo *.egg-info/
        echo.
        echo # Node
        echo node_modules/
        echo npm-debug.log*
        echo yarn-error.log*
        echo .pnpm-store/
        echo.
        echo # Java / Gradle / Kotlin
        echo .gradle/
        echo **/build/
        echo **/.kotlin/
        echo *.class
        echo *.jar
        echo !gradle/wrapper/gradle-wrapper.jar
        echo.
        echo # Misc
        echo _tmp/
        echo tmp/
        echo *.exe
        echo *.local
    ) > ".gitignore"
    if errorlevel 1 (
        echo [ERROR] failed to write .gitignore
        goto END
    )
    echo OK
)
echo.

echo ========================================
echo   [Git] Init OK
echo ========================================
git status -sb 2>nul
echo.
echo Next (optional):
echo   git add -A
echo   git commit -m "Initial commit"
echo   git remote add origin ^<url^>
echo   git push -u origin HEAD
echo   Or use: upload_github.bat / upload_gitee.bat
echo.

:END
echo.
pause
endlocal
