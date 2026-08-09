# -*- mode: python ; coding: utf-8 -*-
#
# Configurazione PyInstaller. Da eseguire su Windows, dalla cartella
# principale del progetto:
#
#     pyinstaller build/app.spec
#
# Produce una build "onedir": una cartella con l'eseguibile e le
# librerie. E' piu' affidabile della modalita' a file singolo quando
# sono coinvolti binari nativi pesanti (CTranslate2, llama.cpp,
# onnxruntime). L'utente finale non vede comunque questa cartella:
# riceve un unico file di installazione creato con Inno Setup, che la
# installa come qualunque altro programma Windows.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

project_root = Path(SPECPATH).resolve().parent

binaries = []
datas = []
hiddenimports = []

# Le librerie di calcolo portano con se' DLL native e file di dati
# (modello per il rilevamento della voce, tabelle di tokenizzazione) che
# PyInstaller non individua da solo: senza, il programma si compila ma
# non parte.
PACCHETTI_OBBLIGATORI = (
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "tokenizers",
    "llama_cpp",
    "av",
)
PACCHETTI_FACOLTATIVI = (
    "pyaudiowpatch",
    "docx",
    "huggingface_hub",
    "tqdm",
)

for package in PACCHETTI_OBBLIGATORI + PACCHETTI_FACOLTATIVI:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception as exc:
        # Un pacchetto obbligatorio mancante deve fermare la
        # compilazione: proseguire produrrebbe un installer
        # apparentemente riuscito che pero' non si avvia sul computer
        # del cliente, e nessuno se ne accorgerebbe fino a quel momento.
        if package in PACCHETTI_OBBLIGATORI:
            raise SystemExit(
                f"[app.spec] ERRORE: il pacchetto obbligatorio '{package}' non "
                f"e' utilizzabile ({exc}). Compilazione interrotta."
            )
        print(f"[app.spec] Pacchetto facoltativo '{package}' non incluso: {exc}")
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports += [
    "app",
    "app.main",
    "app.compat",
    "app.diagnostics",
    "app.settings",
    "app.single_instance",
    "app.audio.capture",
    "app.transcription.engine",
    "app.summarization.llm",
    "app.models.download",
    "app.storage.db",
    "app.export.report",
    # Estensione C di PyAudioWPatch: si trova accanto al pacchetto e non
    # al suo interno, quindi la raccolta automatica non la vede.
    "_portaudiowpatch",
    # Dipendenze di faster-whisper che non si individuano leggendo il codice.
    "huggingface_hub",
    "tqdm",
    # Dipendenze di llama_cpp e onnxruntime.
    "diskcache",
    "jinja2",
    "coloredlogs",
    "humanfriendly",
    "flatbuffers",
    "packaging",
    "psutil",
    "requests",
    "certifi",
    # Senza questo, requests avvisa a ogni avvio di non saper
    # riconoscere la codifica dei testi scaricati.
    "charset_normalizer",
    "idna",
    "urllib3",
]

# Verifica finale: se queste librerie native non sono state raccolte, il
# programma si avvia e poi fallisce al primo colloquio.
def _presente(frammento: str) -> bool:
    return any(
        frammento in str(origine).replace("\\", "/") for origine, _ in binaries
    )


for _attesa in ("llama_cpp/lib/llama", "ctranslate2"):
    if not _presente(_attesa):
        raise SystemExit(
            f"[app.spec] ERRORE: libreria nativa '{_attesa}' non raccolta. "
            "Compilazione interrotta."
        )

icon_path = project_root / "app" / "resources" / "icon.ico"

a = Analysis(
    [str(project_root / "run.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Moduli pesanti che non usiamo: alleggeriscono il pacchetto.
        "tkinter",
        "matplotlib",
        "pandas",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.Qt3DCore",
        "PySide6.QtMultimedia",
        "PySide6.QtQuick",
        "PySide6.QtQml",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InterviewAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # La compressione UPX agisce anche sulle DLL native e in passato ha
    # causato chiusure improvvise con onnxruntime e CTranslate2:
    # preferiamo un pacchetto piu' grande ma affidabile.
    upx=False,
    console=False,  # nessuna finestra nera: solo interfaccia grafica
    icon=str(icon_path) if icon_path.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="InterviewAssistant",
)
