@echo off
setlocal EnableDelayedExpansion
echo ==========================================
echo   Wafer Defect Project - GitHub Setup
echo ==========================================

:: 1. Add GitHub CLI to PATH (Temporary for this script)
set "PATH=%PATH%;C:\Program Files\GitHub CLI"

:: 2. Check if gh is installed
where gh >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] GitHub CLI `gh` not found.
    echo Please restart your computer to finish the installation I started earlier.
    pause
    exit /b
)

:: 3. Check Authentication
echo [INFO] Checking GitHub Login Status...
gh auth status >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo [ACTION REQUIRED] You are not logged in.
    echo I will now open the browser for you to authenticate.
    echo 1. Select 'GitHub.com' - press Enter
    echo 2. Select 'HTTPS' - press Enter
    echo 3. Select 'Login with a web browser' - press Enter
    echo.
    gh auth login -p https -w
)

:: 4. Create and Push Repo
echo.
echo [INFO] Creating Repository 'wafer-defect-classification'...
gh repo create wafer-defect-classification --public --source=. --remote=origin --push

if !errorlevel! equ 0 (
    echo.
    echo [SUCCESS] Repository created and code pushed!
    echo URL: https://github.com/badrisatyam1-ctrl/wafer-defect-classification
) else (
    echo.
    echo [ERROR] Failed to create repository. It might already exist.
    echo Try running: git push -u origin main
)

echo.
pause
