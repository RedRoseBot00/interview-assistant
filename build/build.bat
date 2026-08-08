@echo off
REM ============================================================
REM Compilazione su Windows.
REM Da eseguire dalla cartella principale del progetto: build\build.bat
REM Richiede Python 3.11 a 64 bit, con "python" disponibile nel PATH.
REM ============================================================

setlocal

echo === 1/5: creazione ambiente virtuale ===
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat

echo === 2/5: installazione dipendenze ===
python -m pip install --upgrade pip
REM --only-binary per llama-cpp-python: senza, pip proverebbe a compilare
REM llama.cpp da sorgente, operazione che richiede Visual Studio con il
REM compilatore C++ e che sui PC normali fallisce dopo mezz'ora.
pip install -r requirements.txt --only-binary=llama-cpp-python
if errorlevel 1 goto errore

echo === 3/5: compilazione con PyInstaller ===
pyinstaller build\app.spec --noconfirm --clean
if errorlevel 1 goto errore

echo === 4/5: verifica del pacchetto compilato ===
dist\InterviewAssistant\InterviewAssistant.exe --smoke-test
if errorlevel 1 (
    echo.
    echo ATTENZIONE: la verifica ha rilevato librerie mancanti.
    echo Controlla i log in %%APPDATA%%\InterviewAssistant\logs
    goto errore
)

echo === 5/5: compilazione completata ===
echo L'applicazione si trova in dist\InterviewAssistant\InterviewAssistant.exe
echo.
echo Per creare l'installer finale da consegnare al cliente, apri
echo build\installer.iss con Inno Setup e premi Compile.
echo.
goto fine

:errore
echo.
echo *** Compilazione interrotta a causa di un errore. ***
echo.

:fine
endlocal
pause
