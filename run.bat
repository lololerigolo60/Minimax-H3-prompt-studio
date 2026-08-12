@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV_DIR=venv
set PYTHON=python
set MARKER=%VENV_DIR%\.deps_ok

echo ============================================
echo   H3 Prompt Studio - lancement
echo ============================================
echo.

REM --- 1. Verifier que Python est present -----------------------------------
where %PYTHON% >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python introuvable dans le PATH.
    echo Installez Python 3.10+ depuis https://www.python.org/downloads/
    echo puis assurez-vous de cocher "Add python.exe to PATH" a l'installation.
    pause
    exit /b 1
)

REM --- 2. Creer le venv s'il n'existe pas ------------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Environnement virtuel absent, creation dans .\%VENV_DIR% ...
    %PYTHON% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERREUR] Echec de la creation du venv.
        pause
        exit /b 1
    )
    echo [OK] Venv cree.
    REM nouveau venv => on force la reinstallation des dependances
    if exist "%MARKER%" del "%MARKER%" >nul 2>&1
) else (
    echo [OK] Venv deja present.
)

set VENV_PY=%VENV_DIR%\Scripts\python.exe

REM --- 3. Installer les dependances si necessaire -----------------------------
REM Le marqueur %MARKER% n'est ecrit qu'apres une installation reussie et
REM n'est valide que si requirements.txt n'a pas change depuis (comparaison
REM de date de modification).
set NEED_INSTALL=1
if exist "%MARKER%" (
    for %%F in (requirements.txt) do set REQ_TIME=%%~tF
    for %%F in ("%MARKER%") do set MARKER_TIME=%%~tF
    if "!MARKER_TIME!" GEQ "!REQ_TIME!" set NEED_INSTALL=0
)

echo.
if !NEED_INSTALL! EQU 1 (
    echo [INFO] Installation / mise a jour des dependances depuis requirements.txt ...
    "%VENV_PY%" -m pip install --upgrade pip >nul
    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation des dependances.
        pause
        exit /b 1
    )
    echo OK > "%MARKER%"
    echo [OK] Dependances installees.
) else (
    echo [OK] Dependances deja a jour ^(requirements.txt inchange depuis la derniere installation^).
)

REM --- 4. Lancer l'application ------------------------------------------------
echo.
echo [INFO] Lancement de H3 Prompt Studio...
echo.
"%VENV_PY%" main.py

if errorlevel 1 (
    echo.
    echo [ERREUR] L'application s'est terminee avec une erreur.
    pause
)

exit /b 0
