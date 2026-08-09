"""
Riquadro che mostra dal vivo la finestra della videochiamata.

Come funziona
-------------
Un programma non puo' "inglobare" la finestra di Teams, Zoom o del
browser: appartiene a un altro processo. Windows offre pero' una
funzione pensata proprio per questo, la stessa che genera le anteprime
animate quando si passa il mouse sulla barra delle applicazioni: le
miniature DWM (Desktop Window Manager).

Registrando una miniatura, il sistema disegna il contenuto aggiornato
di un'altra finestra dentro un rettangolo della nostra. Il risultato e'
una riproduzione dal vivo della videochiamata al centro dell'app, senza
copiare pixel a mano e senza pesare sul processore, perche' a comporre
l'immagine e' la scheda video.

Limiti da conoscere: la miniatura non e' interattiva (per parlare o
condividere lo schermo si usa la finestra originale) e la finestra di
origine non deve essere ridotta a icona.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from app.ui import theme

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# Costanti dell'API DWM
DWM_TNP_RECTDESTINATION = 0x00000001
DWM_TNP_RECTSOURCE = 0x00000002
DWM_TNP_OPACITY = 0x00000004
DWM_TNP_VISIBLE = 0x00000008
DWM_TNP_SOURCECLIENTAREAONLY = 0x00000010

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
DWMWA_CLOAKED = 14


class _ThumbnailProperties(ctypes.Structure):
    _fields_ = [
        ("dwFlags", wintypes.DWORD),
        ("rcDestination", wintypes.RECT),
        ("rcSource", wintypes.RECT),
        ("opacity", ctypes.c_ubyte),
        ("fVisible", wintypes.BOOL),
        ("fSourceClientAreaOnly", wintypes.BOOL),
    ]


@dataclass
class WindowInfo:
    handle: int
    title: str
    process: str

    @property
    def label(self) -> str:
        name = self.process.replace(".exe", "").strip()
        title = self.title if len(self.title) <= 55 else self.title[:52] + "..."
        return f"{title}  —  {name}" if name else title


# Processi che con ogni probabilita' contengono una videochiamata.
_CALL_PROCESSES = {
    "teams.exe", "ms-teams.exe", "zoom.exe", "cpthost.exe",
    "webexmta.exe", "webex.exe", "skype.exe", "gotomeeting.exe",
    "slack.exe", "discord.exe",
}
_BROWSER_PROCESSES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"}
_MEETING_HINTS = ("meet", "zoom", "teams", "webex", "riunione", "meeting", "call")


def _process_name(handle: int) -> str:
    try:
        import psutil

        pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(
            wintypes.HWND(handle), ctypes.byref(pid)
        )
        if not pid.value:
            return ""
        return psutil.Process(pid.value).name()
    except Exception:
        return ""


def _is_cloaked(handle: int) -> bool:
    """
    Le finestre "nascoste dal sistema" esistono ma non sono visibili:
    su Windows 10 ce ne sono diverse e vanno escluse dall'elenco.
    """
    try:
        cloaked = ctypes.c_int(0)
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(handle),
            ctypes.c_uint(DWMWA_CLOAKED),
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
        return bool(cloaked.value)
    except Exception:
        return False


def list_windows(exclude: int = 0) -> list[WindowInfo]:
    """Finestre visibili con un titolo, ordinate mettendo per prime le videochiamate."""
    if not IS_WINDOWS:
        return []

    user32 = ctypes.windll.user32
    results: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _callback(handle, _lparam):
        try:
            if handle == exclude or not user32.IsWindowVisible(handle):
                return True
            if user32.IsIconic(handle):
                return True  # ridotta a icona: la miniatura sarebbe vuota
            length = user32.GetWindowTextLengthW(handle)
            if length <= 0:
                return True
            styles = user32.GetWindowLongW(handle, GWL_EXSTYLE)
            if styles & WS_EX_TOOLWINDOW:
                return True
            if _is_cloaked(handle):
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(handle, buffer, length + 1)
            title = buffer.value.strip()
            if not title:
                return True

            results.append(
                WindowInfo(int(handle), title, _process_name(int(handle)).lower())
            )
        except Exception:
            log.debug("Finestra ignorata durante l'elenco", exc_info=True)
        return True

    try:
        user32.EnumWindows(_callback, 0)
    except Exception:
        log.warning("Elenco delle finestre non riuscito", exc_info=True)
        return []

    def _rank(info: WindowInfo) -> tuple[int, str]:
        if info.process in _CALL_PROCESSES:
            return (0, info.title.lower())
        if info.process in _BROWSER_PROCESSES:
            lowered = info.title.lower()
            if any(hint in lowered for hint in _MEETING_HINTS):
                return (1, lowered)
            return (2, lowered)
        return (3, info.title.lower())

    results.sort(key=_rank)
    return results


def best_call_window(exclude: int = 0) -> WindowInfo | None:
    """Finestra piu' probabile della videochiamata, se ce n'e' una."""
    for info in list_windows(exclude):
        if info.process in _CALL_PROCESSES:
            return info
        if info.process in _BROWSER_PROCESSES and any(
            hint in info.title.lower() for hint in _MEETING_HINTS
        ):
            return info
    return None


class CallView(QWidget):
    """Riquadro che riproduce dal vivo la finestra scelta."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._thumbnail = None
        self._source: WindowInfo | None = None
        self._source_size = (16, 9)
        self.setMinimumSize(320, 200)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

    # ------------------------------------------------------------------
    @property
    def source(self) -> WindowInfo | None:
        return self._source

    @property
    def available(self) -> bool:
        return IS_WINDOWS

    def set_source(self, info: WindowInfo | None) -> bool:
        self.clear()
        if info is None or not IS_WINDOWS:
            self._source = None
            self.update()
            return False

        try:
            destination = int(self.window().winId())
            thumbnail = ctypes.c_void_p()
            result = ctypes.windll.dwmapi.DwmRegisterThumbnail(
                wintypes.HWND(destination),
                wintypes.HWND(info.handle),
                ctypes.byref(thumbnail),
            )
            if result != 0:
                log.warning(
                    "Miniatura non registrabile per '%s' (codice %s)",
                    info.title, result,
                )
                self._source = None
                self.update()
                return False

            self._thumbnail = thumbnail
            self._source = info
            self._read_source_size()
            self.refresh()
            return True
        except Exception:
            log.exception("Impossibile mostrare la finestra della videochiamata")
            self._source = None
            self.update()
            return False

    def clear(self) -> None:
        if self._thumbnail is not None:
            try:
                ctypes.windll.dwmapi.DwmUnregisterThumbnail(self._thumbnail)
            except Exception:
                log.debug("Miniatura non rimossa", exc_info=True)
            self._thumbnail = None
        self._source = None
        self.update()

    # ------------------------------------------------------------------
    def _read_source_size(self) -> None:
        try:
            size = wintypes.SIZE()
            ctypes.windll.dwmapi.DwmQueryThumbnailSourceSize(
                self._thumbnail, ctypes.byref(size)
            )
            if size.cx > 0 and size.cy > 0:
                self._source_size = (size.cx, size.cy)
        except Exception:
            pass

    def _destination_rect(self) -> wintypes.RECT:
        """
        Rettangolo in cui disegnare, nelle coordinate della finestra
        principale: la miniatura viene composta dal sistema, quindi non
        conosce la posizione del nostro riquadro all'interno del layout.
        """
        window = self.window()
        top_left = self.mapTo(window, self.rect().topLeft())
        width, height = self.width(), self.height()

        # Manteniamo le proporzioni della finestra di origine, cosi' il
        # video non risulta deformato.
        source_w, source_h = self._source_size
        if source_w > 0 and source_h > 0:
            scale = min(width / source_w, height / source_h)
            draw_w = max(1, int(source_w * scale))
            draw_h = max(1, int(source_h * scale))
        else:
            draw_w, draw_h = width, height

        offset_x = top_left.x() + (width - draw_w) // 2
        offset_y = top_left.y() + (height - draw_h) // 2

        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        rect = wintypes.RECT()
        rect.left = int(offset_x * ratio)
        rect.top = int(offset_y * ratio)
        rect.right = int((offset_x + draw_w) * ratio)
        rect.bottom = int((offset_y + draw_h) * ratio)
        return rect

    def source_alive(self) -> bool:
        """
        La finestra della videochiamata esiste ancora?

        Serve perche' DWM non avvisa quando la sorgente sparisce: la
        miniatura resta registrata, il riquadro diventa un rettangolo
        nero permanente e il messaggio "Nessuna videochiamata
        selezionata" non ricompare piu'. L'utente vede un buco senza
        alcuna spiegazione.
        """
        if self._thumbnail is None or self._source is None or not IS_WINDOWS:
            return False
        try:
            return bool(ctypes.windll.user32.IsWindow(wintypes.HWND(self._source.handle)))
        except Exception:
            return True     # nel dubbio non buttiamo via l'anteprima

    def refresh(self) -> None:
        """Riallinea la miniatura dopo spostamenti, ridimensionamenti o cambi di scheda."""
        if self._thumbnail is None:
            return
        if not self.source_alive():
            log.info("La finestra della videochiamata e' stata chiusa")
            self.clear()
            self._source = None
            self.update()
            return
        try:
            self._read_source_size()
            properties = _ThumbnailProperties()
            properties.dwFlags = (
                DWM_TNP_RECTDESTINATION
                | DWM_TNP_VISIBLE
                | DWM_TNP_OPACITY
                | DWM_TNP_SOURCECLIENTAREAONLY
            )
            properties.rcDestination = self._destination_rect()
            properties.opacity = 255
            properties.fVisible = bool(self.isVisible())
            properties.fSourceClientAreaOnly = True
            ctypes.windll.dwmapi.DwmUpdateThumbnailProperties(
                self._thumbnail, ctypes.byref(properties)
            )
        except Exception:
            log.debug("Aggiornamento della miniatura non riuscito", exc_info=True)

    def set_visible_thumbnail(self, visible: bool) -> None:
        """Nasconde la miniatura senza perdere la finestra selezionata."""
        if self._thumbnail is None:
            return
        try:
            properties = _ThumbnailProperties()
            properties.dwFlags = DWM_TNP_VISIBLE
            properties.fVisible = bool(visible)
            ctypes.windll.dwmapi.DwmUpdateThumbnailProperties(
                self._thumbnail, ctypes.byref(properties)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    def resizeEvent(self, event):  # noqa: N802 - firma imposta da Qt
        super().resizeEvent(event)
        self.refresh()

    def moveEvent(self, event):  # noqa: N802 - firma imposta da Qt
        super().moveEvent(event)
        self.refresh()

    def showEvent(self, event):  # noqa: N802 - firma imposta da Qt
        super().showEvent(event)
        self.refresh()

    def hideEvent(self, event):  # noqa: N802 - firma imposta da Qt
        self.set_visible_thumbnail(False)
        super().hideEvent(event)

    def paintEvent(self, event):  # noqa: N802 - firma imposta da Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor("#141b26"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

        if self._thumbnail is not None:
            return  # il contenuto lo disegna il sistema

        painter.setPen(QPen(QColor("#8b97a8")))
        if not IS_WINDOWS:
            message = "Anteprima disponibile solo su Windows."
        else:
            message = (
                "Nessuna videochiamata selezionata.\n\n"
                "Apri Teams, Zoom o Google Meet e scegli la finestra\n"
                "dall'elenco qui sopra: comparira' qui dentro, dal vivo."
            )
        painter.drawText(self.rect().adjusted(20, 20, -20, -20), Qt.AlignCenter, message)
