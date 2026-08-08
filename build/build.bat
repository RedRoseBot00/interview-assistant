@echo off
REM ============================================================
REM Script di build per Windows.
REM Da eseguire dalla root del progetto: build\build.bat
REM Richiede: Python 3.10 o 3.11 a 64 bit installato, con "python"
REM disponibile nel PATH.
REM ============================================================

setlocal

echo === 1/4: creazione ambiente virtuale ===
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo === 2/4: installazione dipendenze ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo === 3/4: build eseguibile con PyInstaller ===
pyinstaller build\app.spec --noconfirm

echo === 4/4: build completata ===
echo L'app si trova in dist\InterviewAssistant\InterviewAssistant.exe
echo.
echo Per creare l'installer .exe finale da consegnare al cliente,
echo apri build\installer.iss con Inno Setup e compila (Build ^> Compile).
echo.

endlocal
pause
