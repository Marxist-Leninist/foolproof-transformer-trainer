@echo off
rem  Foolproof Transformer Trainer - launcher for Windows 10 and 11.
rem  Double-click this file. The first run sets up a private copy of Python and PyTorch
rem  (a few minutes, needs internet). Nothing else on your computer is changed.
rem  After that it starts in seconds, offline too.
setlocal EnableExtensions DisableDelayedExpansion
title Foolproof Trainer
set "PYTHONUTF8=1"
set "UV_PYTHON_PREFERENCE=only-managed"

set "FP_REPO=Marxist-Leninist/foolproof-transformer-trainer"
if defined FOOLPROOF_REPO set "FP_REPO=%FOOLPROOF_REPO%"
set "FP_BRANCH=main"
if defined FOOLPROOF_BRANCH set "FP_BRANCH=%FOOLPROOF_BRANCH%"
set "FP_DATA=%LOCALAPPDATA%\FoolproofTrainer"
if not exist "%FP_DATA%\bin" mkdir "%FP_DATA%\bin"

echo.
echo   Foolproof Trainer
echo   -----------------
echo.

if /i "%PROCESSOR_ARCHITECTURE%"=="x86" if not defined PROCESSOR_ARCHITEW6432 goto fail_32bit
where curl.exe >nul 2>&1 || goto fail_old
where tar.exe >nul 2>&1 || goto fail_old

rem ---- 1. Find the app itself. If only this file was downloaded (or it was run from inside
rem         the zip without unpacking), fetch the rest. ------------------------------------
set "FP_DOWNLOADED=0"
set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"
if exist "%APP_DIR%\foolproof\__main__.py" goto found_local
set "FP_DOWNLOADED=1"
set "APP_DIR=%FP_DATA%\app"
if "%FOOLPROOF_UPDATE%"=="1" goto get_app
if exist "%APP_DIR%\foolproof\__main__.py" goto have_app
:get_app
echo   Downloading Foolproof Trainer...
curl.exe -fsSL --retry 3 -o "%FP_DATA%\app.zip" "https://github.com/%FP_REPO%/archive/refs/heads/%FP_BRANCH%.zip" || goto fail_net
if exist "%FP_DATA%\app-new" rmdir /s /q "%FP_DATA%\app-new"
mkdir "%FP_DATA%\app-new"
tar.exe -xf "%FP_DATA%\app.zip" -C "%FP_DATA%\app-new" --strip-components=1 || goto fail_unpack
if exist "%APP_DIR%" rmdir /s /q "%APP_DIR%"
move "%FP_DATA%\app-new" "%APP_DIR%" >nul || goto fail_unpack
del "%FP_DATA%\app.zip" >nul 2>&1
goto have_app

:found_local
rem  A developer's git clone runs where it is. A downloaded copy installs itself into a fixed
rem  place, so the Desktop shortcut keeps working after the download is tidied away.
if exist "%APP_DIR%\.git" goto have_app
if /i "%APP_DIR%"=="%FP_DATA%\app" goto have_app
echo   Installing Foolproof Trainer...
robocopy "%APP_DIR%" "%FP_DATA%\app" /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS /NP >nul 2>&1
if errorlevel 8 goto fail_unpack
set "APP_DIR=%FP_DATA%\app"
set "FP_DOWNLOADED=1"
:have_app

rem ---- 2. The installer helper (uv): one small program that fetches Python and packages. --
set "UV=%FP_DATA%\bin\uv.exe"
"%UV%" --version >nul 2>&1 && goto have_uv
echo   Step 1 of 3: fetching the installer helper...
set "UV_ARCH=x86_64"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "UV_ARCH=aarch64"
curl.exe -fsSL --retry 3 -o "%FP_DATA%\uv.zip" "https://github.com/astral-sh/uv/releases/latest/download/uv-%UV_ARCH%-pc-windows-msvc.zip" || goto fail_net
tar.exe -xf "%FP_DATA%\uv.zip" -C "%FP_DATA%\bin" >nul 2>&1
del "%FP_DATA%\uv.zip" >nul 2>&1
"%UV%" --version >nul 2>&1 && goto have_uv
rem  second way in, if the first did not work on this machine: the official installer
set "UV_UNMANAGED_INSTALL=%FP_DATA%\bin"
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" >nul 2>&1
set "UV_UNMANAGED_INSTALL="
"%UV%" --version >nul 2>&1 || goto fail_uv
:have_uv

rem ---- 3. A private Python with PyTorch, kept out of the way in %FP_DATA% -----------------
set "UV_PYTHON_INSTALL_DIR=%FP_DATA%\python"
set "VENV=%FP_DATA%\venv"
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" goto do_install
fc /b "%APP_DIR%\requirements.txt" "%VENV%\fp-installed.txt" >nul 2>&1 && goto run
:do_install
if exist "%PY%" goto have_venv
echo   Step 2 of 3: setting up a private copy of Python...
"%UV%" venv "%VENV%" --python 3.12 --quiet || goto fail_setup
:have_venv
echo   Step 3 of 3: installing PyTorch, the AI engine.
echo   This is a big download (about 3 GB with an NVIDIA graphics card). It happens only once.
"%PY%" "%APP_DIR%\foolproof\install.py" "%UV%" "%APP_DIR%" || goto fail_setup

:run
rem ---- 4. A Desktop shortcut for next time (only when we downloaded the app ourselves). ---
if not "%FP_DOWNLOADED%"=="1" goto launch
if defined CI goto launch
set "FP_LNK_TARGET=%APP_DIR%\START-WINDOWS.bat"
set "FP_LNK_DIR=%APP_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=[Environment]::GetFolderPath('Desktop'); $p=Join-Path $d 'Foolproof Trainer.lnk'; if(-not (Test-Path $p)){ $s=(New-Object -ComObject WScript.Shell).CreateShortcut($p); $s.TargetPath=$env:FP_LNK_TARGET; $s.WorkingDirectory=$env:FP_LNK_DIR; $s.Description='Train your own small AI'; $s.Save(); Write-Host '  Added Foolproof Trainer to your Desktop: double-click it next time.' }" 2>nul

:launch
pushd "%APP_DIR%" || goto fail_unpack
"%PY%" -m foolproof %*
set "RC=%ERRORLEVEL%"
popd
if "%RC%"=="0" goto done
echo.
echo   Foolproof Trainer stopped with a problem (code %RC%).
echo   If you are stuck, ask for help here: https://github.com/%FP_REPO%/issues
goto wait

:fail_32bit
echo   PROBLEM: this is a 32-bit version of Windows, which PyTorch does not support.
set "RC=1"
goto wait
:fail_old
echo   PROBLEM: this needs Windows 10 (2018 or newer) or Windows 11.
set "RC=1"
goto wait
:fail_net
echo.
echo   PROBLEM: could not download. Check your internet connection and try again.
echo   (The first start needs the internet; later starts do not.)
set "RC=1"
goto wait
:fail_unpack
echo.
echo   PROBLEM: could not unpack the app. Is your disk full, or is an antivirus blocking it?
set "RC=1"
goto wait
:fail_uv
echo.
echo   PROBLEM: the installer helper could not be set up. An antivirus may be blocking it.
set "RC=1"
goto wait
:fail_setup
echo.
echo   PROBLEM: setup did not finish. Check your internet connection and that you have
echo   5 GB of free disk space, then double-click START-WINDOWS.bat again: it carries on
echo   from where it stopped.
echo   If you are stuck, ask for help here: https://github.com/%FP_REPO%/issues
set "RC=1"
goto wait

:wait
if defined CI goto done
echo.
pause
:done
endlocal & exit /b %RC%
