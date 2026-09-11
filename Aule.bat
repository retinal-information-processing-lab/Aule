@echo off
title Aule - patterned OSS stimulation

:: 1. Localisation de Conda
set CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%USERPROFILE%\miniconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%ProgramData%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%ProgramData%\miniconda3\Scripts\activate.bat

:: 2. Navigation vers le dossier du script ("%~dp0" = dossier de ce .bat)
cd /d "%~dp0"

:: 3. Activation de conda (base) puis creation de l'environnement au premier lancement
call "%CONDA_PATH%"
conda env list | findstr /B /C:"aule " >nul
if errorlevel 1 (
    echo Environnement 'aule' introuvable : creation depuis environment.yml (quelques minutes)...
    conda env create -f environment.yml -y
    if errorlevel 1 (
        echo La creation de l'environnement a echoue.
        pause
        exit /b 1
    )
)

:: 4. Activation de l'environnement
call conda activate aule

:: 5. Lancement DETACHE et silencieux (pythonw = pas de console noire)
start "" pythonw -m aule gui

:: 6. Fermeture immediate de la console
exit
