"""
Aspetto grafico dell'applicazione: palette, foglio di stile e piccoli
componenti riutilizzabili (schede, etichette-pillola, indicatori di
livello audio).

Tenere lo stile in un unico file rende semplice adattare i colori al
marchio del cliente finale senza toccare la logica delle schermate.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
HEADER_BG = "#1e2836"
PAGE_BG = "#f3f4f6"
CARD_BG = "#ffffff"
BORDER = "#e2e5ea"
TEXT = "#1f2937"
TEXT_MUTED = "#6b7280"
ACCENT = "#2563eb"
SUCCESS = "#16a34a"
DANGER = "#dc2626"
WARNING = "#b45309"

SPEAKER_COLORS = {
    "recruiter": "#b45309",   # chi conduce il colloquio
    "candidate": "#2563eb",   # la persona intervistata
}

FONT_FAMILY = "Segoe UI, Inter, system-ui, sans-serif"

STYLESHEET = f"""
QWidget {{
    font-family: {FONT_FAMILY};
    font-size: 13px;
    color: {TEXT};
}}
QMainWindow, QTabWidget::pane {{
    background: {PAGE_BG};
    border: none;
}}

/* ---- Intestazione ---- */
#HeaderBar {{
    background: {HEADER_BG};
    min-height: 62px;
}}
#HeaderTitle {{
    color: #ffffff;
    font-size: 21px;
    font-weight: 600;
}}
#HeaderVersion {{
    color: #9aa4b2;
    font-size: 12px;
}}

/* ---- Barra piattaforme ---- */
#PlatformBar {{
    background: {CARD_BG};
    border-bottom: 1px solid {BORDER};
    min-height: 46px;
}}
QLabel#PlatformHint {{
    color: {TEXT_MUTED};
}}
QLabel.Chip {{
    background: #eef1f5;
    border: 1px solid {BORDER};
    border-radius: 11px;
    padding: 3px 11px;
    font-weight: 600;
    font-size: 12px;
    color: #374151;
}}
QLabel.SkillChip {{
    background: #e8f0fe;
    border: none;
    border-radius: 10px;
    padding: 3px 10px;
    font-size: 12px;
    color: {ACCENT};
}}

/* ---- Schede ---- */
QFrame.Card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel.CardTitle {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel.Muted {{
    color: {TEXT_MUTED};
}}
QLabel.Metric {{
    font-size: 24px;
    font-weight: 600;
    color: {TEXT};
}}

/* ---- Pulsanti ---- */
QPushButton {{
    background: #eef1f5;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ background: #e4e8ee; }}
QPushButton:disabled {{ color: #a3aab5; background: #f2f4f7; }}

QPushButton#PrimaryStart {{
    background: {SUCCESS};
    border: none;
    color: #ffffff;
}}
QPushButton#PrimaryStart:hover {{ background: #15803d; }}
QPushButton#PrimaryStart:disabled {{ background: #a7d6b8; color: #f0fdf4; }}

QPushButton#StopButton {{
    background: {DANGER};
    border: none;
    color: #ffffff;
}}
QPushButton#StopButton:hover {{ background: #b91c1c; }}

QPushButton#AccentButton {{
    background: {ACCENT};
    border: none;
    color: #ffffff;
}}
QPushButton#AccentButton:hover {{ background: #1d4ed8; }}
QPushButton#AccentButton:disabled {{ background: #b6c8ee; color: #eff4ff; }}

QPushButton#DangerButton {{
    background: #ffffff;
    border: 1px solid #f0c2c2;
    color: {DANGER};
}}
QPushButton#DangerButton:hover {{ background: #fdf2f2; }}
QPushButton#DangerButton:disabled {{
    background: #f7f8fa;
    border-color: {BORDER};
    color: #b9bec7;
}}

QPushButton:checked {{
    background: #dfe7f5;
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* ---- Campi ---- */
/* L'altezza minima e' indispensabile: senza, su Windows il campo si
   adatta al testo con troppo poco margine e le lettere alte o con
   accento risultano tagliate sopra e sotto. */
QLineEdit, QComboBox, QSpinBox {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 9px;
    min-height: 22px;
    selection-background-color: {ACCENT};
}}
/* Il testo non deve toccare gli angoli arrotondati del riquadro. */
QTextEdit, QPlainTextEdit {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QLabel {{ background: transparent; }}

/* Il cronometro va letto con la coda dell'occhio mentre si parla con
   il candidato: la regola generale sui caratteri lo riporterebbe alla
   dimensione di tutto il resto, perche' in un foglio di stile Qt il
   font impostato dal codice viene sempre sovrascritto. */
#TimerLabel {{
    font-size: 17px;
    font-weight: 700;
    color: {TEXT};
}}

/* ---- Liste ---- */
QListWidget {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
}}
QListWidget::item {{ padding: 9px 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: #e8f0fe; color: {TEXT}; }}

/* ---- Schede a tab ---- */
QTabBar::tab {{
    background: transparent;
    padding: 9px 18px;
    margin-right: 4px;
    border: none;
    color: {TEXT_MUTED};
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}

/* ---- Barre di avanzamento ---- */
QProgressBar {{
    background: #eef1f5;
    border: none;
    border-radius: 5px;
    /* In un foglio di stile Qt 'height' vale solo per i sottocontrolli:
       sui widget serve la coppia minima/massima. */
    min-height: 9px;
    max-height: 9px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #cfd5dd; border-radius: 5px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: #b6bec9; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


# --------------------------------------------------------------------------
# Componenti riutilizzabili
# --------------------------------------------------------------------------
class Card(QFrame):
    """Riquadro bianco con bordo arrotondato e titolo opzionale."""

    def __init__(self, title: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        self.setFrameShape(QFrame.NoFrame)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 14)
        self._layout.setSpacing(10)

        self.title_label: QLabel | None = None
        if title:
            self.title_label = QLabel(title)
            self.title_label.setProperty("class", "CardTitle")
            self._layout.addWidget(self.title_label)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self._layout.addWidget(widget, stretch)


class Chip(QLabel):
    """Etichetta a pillola (piattaforme supportate, competenze, ...)."""

    def __init__(self, text: str, kind: str = "Chip", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setProperty("class", kind)
        self.setAlignment(Qt.AlignCenter)


class StatusDot(QLabel):
    """Pallino colorato che riassume uno stato (attivo, in pausa, errore)."""

    def __init__(self, color: str = TEXT_MUTED, diameter: int = 10, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):  # noqa: N802 - firma imposta da Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self._color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self._diameter, self._diameter)


class LevelMeter(QWidget):
    """
    Indicatore del livello audio in ingresso.

    Serve a rispondere alla domanda piu' frequente durante una
    registrazione: "mi sta sentendo?". Se la barra si muove quando si
    parla, la sorgente e' quella giusta.
    """

    def __init__(self, color: str = ACCENT, parent: QWidget | None = None):
        super().__init__(parent)
        self._level = 0.0
        self._color = QColor(color)
        self.setFixedHeight(8)
        self.setMinimumWidth(80)

    def set_level(self, value: float) -> None:
        # Scala logaritmica: la voce umana occupa una porzione ridotta
        # della scala lineare e la barra sembrerebbe sempre ferma.
        import math

        if value <= 0:
            normalised = 0.0
        else:
            db = 20 * math.log10(max(value, 1e-6))
            normalised = (db + 60) / 60  # -60 dB -> 0, 0 dB -> 1
        self._level = max(0.0, min(1.0, normalised))
        self.update()

    def paintEvent(self, event):  # noqa: N802 - firma imposta da Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        painter.setBrush(QColor("#e9edf2"))
        painter.drawRoundedRect(self.rect(), 4, 4)

        if self._level > 0.01:
            filled = self.rect()
            filled.setWidth(int(self.width() * self._level))
            painter.setBrush(self._color)
            painter.drawRoundedRect(filled, 4, 4)


def bold(label: QLabel, size: int | None = None) -> QLabel:
    font: QFont = label.font()
    font.setBold(True)
    if size:
        font.setPointSize(size)
    label.setFont(font)
    return label
