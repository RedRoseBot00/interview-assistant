# -*- mode: python ; coding: utf-8 -*-
#
# Configurazione PyInstaller. Da eseguire su Windows, dalla cartella
# principale del progetto:
#
#     pyinstaller build/app.spec
#
# Produce una build "onedir": una cartella con l'eseguibile e le
# librerie. E' piu' affidabile della modalita' a file singolo quando
# sono coinvolti binari nativi pesanti (CTranslate2, llama.cpp).
# L'utente finale non vede comunque questa cartella:
# riceve un unico file di installazione creato con Inno Setup, che la
# installa come qualunque altro programma Windows.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

project_root = Path(SPECPATH).resolve().parent

# Numero di versione letto da app/config.py: una sola fonte, cosi' la
# risorsa di versione dell'eseguibile non puo' discordare dal programma.
_versione = "0.0.0"
for _riga in (project_root / "app" / "config.py").read_text(encoding="utf-8").splitlines():
    if _riga.startswith("APP_VERSION"):
        _versione = _riga.split('"')[1]
        break
_v = tuple(int(p) for p in (_versione.split(".") + ["0", "0", "0"])[:4])

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
    # onnxruntime NON va incluso. Serviva al rilevatore di voce di
    # faster-whisper, che qui non viene mai usato: il rilevamento e'
    # scritto in casa (app/audio/vad.py) e la trascrizione passa
    # vad_filter=False. Erano diciassette megabyte nell'installer, piu'
    # un paio di centinaia di moduli che tirano dentro torch e
    # transformers e riempiono di falsi allarmi il registro della
    # compilazione, nascondendo quelli veri.
    "tokenizers",
    "llama_cpp",
    "av",
)
# Su Windows la cattura dell'audio di sistema e' la funzione centrale
# del programma: senza PyAudioWPatch la voce del candidato non viene
# trascritta affatto. Non e' quindi un pacchetto facoltativo, e la sua
# assenza deve fermare la compilazione invece di stampare un avviso che
# nessuno legge.
if sys.platform == "win32":
    PACCHETTI_OBBLIGATORI += ("pyaudiowpatch",)

PACCHETTI_FACOLTATIVI = (
    "docx",
    "huggingface_hub",
    "tqdm",
)
if sys.platform != "win32":
    PACCHETTI_FACOLTATIVI += ("pyaudiowpatch",)

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

# Tutti i moduli del programma, elencati automaticamente. Prima erano
# scritti a mano ed erano dodici su ventidue: la lista funzionava per
# caso, perche' i mancanti risultavano comunque raggiungibili leggendo
# il codice. Bastava pero' aggiungere un modulo caricato in modo
# dinamico perche' sparisse dal pacchetto senza alcun errore di
# compilazione, e il guasto si sarebbe visto solo sul computer del
# cliente.
hiddenimports += collect_submodules("app")

hiddenimports += [
    # Estensione C di PyAudioWPatch: si trova accanto al pacchetto e non
    # al suo interno, quindi la raccolta automatica non la vede.
    "_portaudiowpatch",
    # Dipendenze di faster-whisper che non si individuano leggendo il codice.
    "huggingface_hub",
    "tqdm",
    # Dipendenze di llama_cpp.
    "diskcache",
    "jinja2",
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
if icon_path.exists():
    # L'icona serve anche a runtime (barra delle applicazioni, finestre
    # di dialogo): dentro il pacchetto, non solo come risorsa dell'exe.
    datas += [(str(icon_path), "app/resources")]

# Risorsa di versione dell'eseguibile. Senza, la scheda "Dettagli" delle
# proprieta' del file e' vuota: chi assiste il cliente non ha modo di
# sapere quale versione ha in mano partendo dal file, e un binario privo
# di metadati viene giudicato piu' severamente dai filtri antivirus.
_version_file = project_root / "build" / "version_info.txt"
_version_file.write_text(
    f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={_v}, prodvers={_v},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Interview Assistant'),
      StringStruct('FileDescription', 'Assistente AI per colloqui di lavoro'),
      StringStruct('FileVersion', '{_versione}'),
      StringStruct('InternalName', 'InterviewAssistant'),
      StringStruct('OriginalFilename', 'InterviewAssistant.exe'),
      StringStruct('ProductName', 'Interview Assistant'),
      StringStruct('ProductVersion', '{_versione}'),
    ])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""",
    encoding="utf-8",
)

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
    # causato chiusure improvvise con CTranslate2:
    # preferiamo un pacchetto piu' grande ma affidabile.
    upx=False,
    console=False,  # nessuna finestra nera: solo interfaccia grafica
    icon=str(icon_path) if icon_path.exists() else None,
    version=str(_version_file),
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
