@echo off
cd /d "%~dp0"
setlocal EnableDelayedExpansion

REM ============================================================
REM  [GitHub] Universal Upload Script
REM  Copy this file to any Git repo root and run.
REM  Uses remote whose URL contains github.com (prefer name: github)
REM  If missing, you will be asked for the GitHub repo URL once.
REM
REM  Nano / proxy local port - change if your VPN port is different:
set "PROXY_PORT=65532"
REM ============================================================

echo ========================================
echo   [GitHub] Upload Script  (universal)
echo ========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found. Install Git first.
    goto END
)

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a git repository: %CD%
    echo Put this script in the project root.
    goto END
)

for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do cd /d "%%i"
for /f "delims=" %%i in ('git branch --show-current 2^>nul') do set "LOCAL_BRANCH=%%i"
if not defined LOCAL_BRANCH set "LOCAL_BRANCH=HEAD"
set "BRANCH=main"

set "REMOTE="
for /f "tokens=1,2" %%a in ('git remote -v') do (
    echo %%b | findstr /I "github.com" >nul
    if not errorlevel 1 (
        if /I "%%a"=="github" set "REMOTE=github"
        if not defined REMOTE set "REMOTE=%%a"
    )
)

if not defined REMOTE (
    echo No GitHub remote found.
    set /p GHURL=Paste GitHub repo URL: 
    if "!GHURL!"=="" (
        echo [ERROR] URL is empty.
        goto END
    )
    git remote add github "!GHURL!"
    if errorlevel 1 (
        echo [ERROR] failed to add remote github
        goto END
    )
    set "REMOTE=github"
    echo Added remote github -^> !GHURL!
)

for /f "delims=" %%i in ('git remote get-url !REMOTE! 2^>nul') do set "REMOTE_URL=%%i"

REM proxy only for github.com so Gitee is not affected
git config --global http.https://github.com.proxy http://127.0.0.1:!PROXY_PORT!
git config --global https.https://github.com.proxy http://127.0.0.1:!PROXY_PORT!

echo Local:  %CD%
echo Local branch: %LOCAL_BRANCH%
echo Push to:      %BRANCH%
echo Remote: !REMOTE!  -^>  !REMOTE_URL!
echo Proxy:  127.0.0.1:!PROXY_PORT!  (keep Nano connected)
echo.

set "MSG=%*"
if "%MSG%"=="" set /p MSG=Commit message, Enter for update: 
if "%MSG%"=="" set "MSG=update"

echo [1/3] git add -A
git add -A
if errorlevel 1 (
    echo [ERROR] git add failed
    goto END
)

echo [2/3] git commit
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "%MSG%"
    if errorlevel 1 (
        echo [ERROR] git commit failed
        goto END
    )
) else (
    echo No changes to commit
)

echo [3/3] git push !REMOTE! HEAD:%BRANCH%
git push -u !REMOTE! "HEAD:%BRANCH%"
if errorlevel 1 (
    echo.
    echo [WARN] Normal push failed.
    echo Possible cause: remote has commits you do not have locally
    echo          ^(rejected / fetch first / non-fast-forward^).
    echo Also check: Nano VPN / PROXY_PORT=!PROXY_PORT! / login.
    echo.
    echo Force push will OVERWRITE remote branch "%BRANCH%"
    echo with your local files. Remote-only history on that branch is lost.
    set /p FORCE=Overwrite remote with local? [Y/N]: 
    if /I "!FORCE!"=="Y" goto DO_FORCE_GH
    if /I "!FORCE!"=="YES" goto DO_FORCE_GH
    echo Cancelled. No force push.
    goto END
)

:PUSH_OK_GH
echo.
echo ========================================
echo   [GitHub] Upload OK
echo ========================================
goto END

:DO_FORCE_GH
echo.
echo [force] git push --force -u !REMOTE! HEAD:%BRANCH%
git push --force -u !REMOTE! "HEAD:%BRANCH%"
if errorlevel 1 (
    echo [ERROR] Force push failed.
    echo Check Nano VPN / PROXY_PORT=!PROXY_PORT! / login.
    goto END
)
goto PUSH_OK_GH

:END
echo.
pause
endlocal