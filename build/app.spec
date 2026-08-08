# -*- mode: python ; coding: utf-8 -*-
#
# Spec file per PyInstaller. Da eseguire SOLO su Windows, dalla root del
# progetto, con:
#
#     pyinstaller build/app.spec
#
# Produce una build "onedir" (una cartella con l'exe + tutte le librerie),
# piu' affidabile della modalita' "onefile" con dipendenze pesanti come
# llama-cpp-python e faster-whisper (che contengono binari nativi).

import sys
from pathlib import Path

block_cipher = None

project_root = Path(SPECPATH).resolve().parent

a = Analysis(
    [str(project_root / "run.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "faster_whisper",
        "llama_cpp",
        "pyaudiowpatch",
        "docx",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="InterviewAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # nessuna finestra console, solo GUI
    icon=str(project_root / "app" / "resources" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="InterviewAssistant",
)
