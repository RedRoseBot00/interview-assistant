"""
Rilevamento della piattaforma di videochiamata in uso.

I colloqui si svolgono su Teams, Zoom, Google Meet e simili. L'app non
si integra con quelle piattaforme (non ne servirebbe il permesso): si
limita ad ascoltare l'audio che esce dagli altoparlanti, quindi funziona
con qualunque software. Questo modulo serve solo a mostrare all'utente
un riscontro visivo del tipo "Meet rilevato", utile per capire a colpo
d'occhio che si sta registrando la sorgente giusta.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# nome processo (minuscolo) -> etichetta leggibile
_PROCESS_MAP = {
    "teams.exe": "Microsoft Teams",
    "ms-teams.exe": "Microsoft Teams",
    "zoom.exe": "Zoom",
    "cpthost.exe": "Zoom",
    "webexmta.exe": "Webex",
    "webex.exe": "Webex",
    "slack.exe": "Slack",
    "discord.exe": "Discord",
    "skype.exe": "Skype",
    "gotomeeting.exe": "GoToMeeting",
}

# I browser non indicano da soli se c'e' una call in corso: li
# segnaliamo solo come possibile sorgente di Google Meet.
_BROWSER_PROCESSES = {
    "chrome.exe": "Google Chrome",
    "msedge.exe": "Microsoft Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
}

SUPPORTED_LABELS = ("Microsoft Teams", "Zoom", "Google Meet", "Webex")


@dataclass
class PlatformStatus:
    detected: list[str]          # applicazioni di call rilevate
    browsers: list[str]          # browser aperti (possibile Google Meet)
    available: bool              # psutil disponibile?

    @property
    def summary(self) -> str:
        if not self.available:
            return "Rilevamento non disponibile"
        if self.detected:
            return f"{', '.join(self.detected)}: in esecuzione"
        if self.browsers:
            return f"{self.browsers[0]} aperto (Meet via browser)"
        return "Nessuna videochiamata rilevata"

    @property
    def is_active(self) -> bool:
        return bool(self.detected or self.browsers)


def detect() -> PlatformStatus:
    """Elenca le applicazioni di videoconferenza attualmente in esecuzione."""
    try:
        import psutil
    except Exception:
        return PlatformStatus(detected=[], browsers=[], available=False)

    found: set[str] = set()
    browsers: set[str] = set()

    try:
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower()
            except Exception:
                continue
            if not name:
                continue
            if name in _PROCESS_MAP:
                found.add(_PROCESS_MAP[name])
            elif name in _BROWSER_PROCESSES:
                browsers.add(_BROWSER_PROCESSES[name])
    except Exception:
        log.debug("Scansione processi fallita", exc_info=True)
        return PlatformStatus(detected=[], browsers=[], available=False)

    return PlatformStatus(
        detected=sorted(found), browsers=sorted(browsers), available=True
    )
