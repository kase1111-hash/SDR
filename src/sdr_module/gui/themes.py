"""
GUI theming: Catppuccin Mocha (dark) and Catppuccin Latte (light).

Every color the GUI shows comes from a :class:`Palette` defined here. Widgets
never hard-code hex colors. Instead they either:

* tag themselves with a ``role`` and/or ``tone`` dynamic property that the
  global stylesheet styles, via :func:`set_role` / :func:`set_tone`::

      set_role(caption_label, "caption")          # small uppercase caption
      set_role(readout, "lcd")                    # monospace LCD readout
      set_tone(status_label, "success")           # green text, re-polished

* or, for custom-painted widgets, read the active palette in ``paintEvent``
  with :func:`get_palette` (cheap, no allocation) and drop any cached pixmaps
  when :func:`theme_notifier` emits ``theme_changed``.

Roles (typography / shape):

    caption       small bold uppercase label (FREQ, SPECTRUM, AVG ...)
    hint          secondary explanatory text, slightly smaller
    muted         secondary text at normal size
    value         emphasized value text (bold)
    display       large numeric readout (e.g. TX power)
    lcd           monospace LCD-style readout (toolbar frequency)
    lcd-small     smaller monospace readout (toolbar level)
    terminal      monospace console look for QTextEdit/QPlainTextEdit
    placeholder   centered empty-state text
    header-strip  compact bar above a plot (QWidget/QFrame)
    card          boxed surface (QFrame)
    callout       boxed note; its colors come from ``tone`` (default info)
    badge         small pill; its colors come from ``tone``
    keycap        keyboard-key label (help dialog, wizard)
    divider       1 px horizontal rule (QFrame); ``vdivider`` is vertical
    primary       (QPushButton/QToolButton) accent-filled call to action
    danger        (QPushButton/QToolButton) destructive / transmit action
    compact       (buttons) dense button, e.g. tuning step buttons

Tones (color): success, warning, danger, info, accent, muted. On QLineEdit,
QSpinBox and QComboBox a tone colors the border (inline validation).

Apply a theme with :func:`apply_theme`, which sets the application palette,
the stylesheet and the generated indicator/arrow icons in one step.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

THEMES: Tuple[str, ...] = ("dark", "light")

# Monospace stack for readouts; the first installed family wins.
MONO_FONT_STACK = (
    "'JetBrains Mono', 'Cascadia Mono', 'Consolas', 'SF Mono', 'Menlo', "
    "'DejaVu Sans Mono', 'Fira Code', monospace"
)


@dataclass(frozen=True)
class Palette:
    """Named color tokens for one theme. All values are ``#rrggbb`` strings."""

    name: str
    is_dark: bool

    # Surfaces, from the deepest background to the most raised control.
    crust: str
    mantle: str
    base: str
    surface0: str
    surface1: str
    surface2: str
    field: str  # background of text inputs, spin boxes and combo boxes
    border_strong: str  # outlines of inputs/indicators (>= 3:1 on base)
    button: str  # resting fill of push/tool buttons
    button_hover: str
    button_pressed: str

    # Text.
    text: str
    subtext: str  # secondary text, still >= 4.3:1 on base
    caption: str  # small caption text
    disabled: str

    # Accent (selection, focus, primary actions).
    accent: str
    accent_hover: str
    accent_muted: str  # low-emphasis accent fill that body text stays legible on
    on_accent: str  # text/icons drawn on an accent (or tone) fill

    # Semantic tones (text-safe on base) and their tinted backgrounds.
    success: str
    warning: str
    danger: str
    info: str
    success_bg: str
    warning_bg: str
    danger_bg: str
    info_bg: str

    # Plots (spectrum, waterfall axes, meters).
    plot_bg: str
    plot_grid: str
    plot_axis: str  # axis tick labels
    plot_trace: str
    plot_peak: str
    plot_marker: str  # tuned-frequency / cursor marker
    plot_text: str

    # LCD-style readouts.
    lcd_bg: str
    lcd_border: str
    lcd_text: str
    lcd_alt: str

    # Extra series colors for charts/meters, in order of preference.
    series: Tuple[str, ...] = ()

    def qcolor(self, token: str, alpha: Optional[int] = None) -> Any:
        """Return a ``QColor`` for a token name (or a literal ``#rrggbb``).

        Args:
            token: Palette field name (``"accent"``) or a hex color.
            alpha: Optional alpha 0-255.
        """
        from PyQt6.QtGui import QColor

        value = token if token.startswith("#") else getattr(self, token)
        color = QColor(value)
        if alpha is not None:
            color.setAlpha(int(alpha))
        return color

    def as_dict(self) -> Dict[str, Any]:
        """All tokens as a plain dict (handy for string templates)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Catppuccin Mocha, with semantic tokens checked for contrast on ``base``.
DARK = Palette(
    name="dark",
    is_dark=True,
    crust="#11111b",
    mantle="#181825",
    base="#1e1e2e",
    surface0="#313244",
    surface1="#45475a",
    surface2="#585b70",
    field="#313244",
    border_strong="#6c7086",
    button="#313244",
    button_hover="#45475a",
    button_pressed="#585b70",
    text="#cdd6f4",
    subtext="#a6adc8",
    caption="#868ba3",
    disabled="#6c7086",
    accent="#89b4fa",
    accent_hover="#b4befe",
    accent_muted="#3a578a",
    on_accent="#1e1e2e",
    success="#a6e3a1",
    warning="#f9e2af",
    danger="#f38ba8",
    info="#89dceb",
    success_bg="#1f2d27",
    warning_bg="#2e2a24",
    danger_bg="#33222c",
    info_bg="#1d2b36",
    plot_bg="#11111b",
    plot_grid="#313244",
    plot_axis="#7f849c",
    plot_trace="#a6e3a1",
    plot_peak="#f38ba8",
    plot_marker="#f9e2af",
    plot_text="#bac2de",
    lcd_bg="#11111b",
    lcd_border="#313244",
    lcd_text="#a6e3a1",
    lcd_alt="#f9e2af",
    series=("#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#94e2d5"),
)

# Catppuccin Latte. Latte's own text tones are too pale on its light base,
# so they are darkened to at least 4.6:1 on base, mantle and their tints.
LIGHT = Palette(
    name="light",
    is_dark=False,
    crust="#dce0e8",
    mantle="#e6e9ef",
    base="#eff1f5",
    surface0="#ccd0da",
    surface1="#bcc0cc",
    surface2="#acb0be",
    field="#ffffff",
    border_strong="#868aa0",
    button="#ffffff",
    button_hover="#e6e9ef",
    button_pressed="#ccd0da",
    text="#4c4f69",
    subtext="#5c5f77",
    caption="#63667a",
    disabled="#9ca0b0",
    accent="#1c5ee1",
    accent_hover="#1a55cc",
    accent_muted="#a9c3fb",
    on_accent="#ffffff",
    success="#2c761e",
    warning="#995700",
    danger="#ca0e37",
    info="#0d727b",
    success_bg="#e3f1de",
    warning_bg="#f8ecd9",
    danger_bg="#f9e1e6",
    info_bg="#dcf0f1",
    plot_bg="#ffffff",
    plot_grid="#dce0e8",
    plot_axis="#6c6f85",
    plot_trace="#1e66f5",
    plot_peak="#d20f39",
    plot_marker="#a15c00",
    plot_text="#4c4f69",
    lcd_bg="#ffffff",
    lcd_border="#bcc0cc",
    lcd_text="#2c761e",
    lcd_alt="#a15c00",
    series=("#1c5ee1", "#2c761e", "#df8e1d", "#ca0e37", "#8839ef", "#179299"),
)

_PALETTES: Dict[str, Palette] = {"dark": DARK, "light": LIGHT}
_current_theme = "dark"
_notifier: Any = None


def normalize_theme(theme: Optional[str]) -> str:
    """Map any theme name to ``"dark"`` or ``"light"`` (default dark)."""
    return "light" if str(theme or "").strip().lower() == "light" else "dark"


def current_theme() -> str:
    """Name of the theme most recently applied with :func:`apply_theme`."""
    return _current_theme


def get_palette(theme: Optional[str] = None) -> Palette:
    """Return the palette for ``theme``, or for the active theme if omitted."""
    return _PALETTES[normalize_theme(theme if theme else _current_theme)]


# ---------------------------------------------------------------------------
# Generated indicator / arrow icons
# ---------------------------------------------------------------------------

_ICON_VERSION = "v2"
_icon_dirs: Dict[str, str] = {}


def _icon_dir(p: Palette) -> str:
    """Directory holding this palette's generated icons (created on demand).

    The folder name hashes the colors the icons are drawn with, so a palette
    change regenerates them instead of reusing stale files.
    """
    key = hashlib.sha1(
        "|".join((_ICON_VERSION, p.on_accent, p.disabled, p.subtext)).encode()
    ).hexdigest()[:10]
    cached = _icon_dirs.get(key)
    if cached and os.path.isdir(cached):
        return cached
    base = ""
    try:
        from PyQt6.QtCore import QStandardPaths

        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation
        )
    except Exception:  # pragma: no cover - PyQt6 missing
        base = ""
    candidates = []
    if base:
        candidates.append(os.path.join(base, "theme-icons"))
    candidates.append(os.path.join(tempfile.gettempdir(), "sdr-module-theme-icons"))
    for root in candidates:
        path = os.path.join(root, f"{p.name}-{key}")
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                _icon_dirs[key] = path
                return path
        except OSError:
            continue
    path = tempfile.mkdtemp(prefix="sdr-theme-")
    _icon_dirs[key] = path
    return path


def _render_icon(out_path: str, kind: str, color: str) -> bool:
    """Draw one 48x48 icon (shown at ~12-16 px, so it stays sharp on HiDPI)."""
    try:
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
    except ImportError:  # pragma: no cover
        return False

    size = 48
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    path = QPainterPath()
    if kind == "check":
        pen.setWidthF(6.0)
        path.moveTo(11, 25)
        path.lineTo(20, 34)
        path.lineTo(37, 14)
        painter.setPen(pen)
        painter.drawPath(path)
    elif kind == "partial":
        pen.setWidthF(6.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(13, 24), QPointF(35, 24))
    elif kind in ("up", "down"):
        pen.setWidthF(5.0)
        if kind == "up":
            path.moveTo(12, 30)
            path.lineTo(24, 18)
            path.lineTo(36, 30)
        else:
            path.moveTo(12, 18)
            path.lineTo(24, 30)
            path.lineTo(36, 18)
        painter.setPen(pen)
        painter.drawPath(path)
    elif kind == "dot":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(24, 24), 10, 10)
    painter.end()
    return img.save(out_path, "PNG")


def _icons(p: Palette) -> Dict[str, str]:
    """Generate (once) and return stylesheet-ready paths for the theme icons."""
    folder = _icon_dir(p)
    specs = {
        "check": ("check", p.on_accent),
        "check_disabled": ("check", p.disabled),
        "partial": ("partial", p.on_accent),
        "dot": ("dot", p.on_accent),
        "dot_disabled": ("dot", p.disabled),
        "up": ("up", p.subtext),
        "down": ("down", p.subtext),
        "up_disabled": ("up", p.disabled),
        "down_disabled": ("down", p.disabled),
    }
    paths: Dict[str, str] = {}
    for name, (kind, color) in specs.items():
        path = os.path.join(folder, f"{name}.png")
        if not os.path.exists(path):
            if not _render_icon(path, kind, color):
                logger.debug("Could not render theme icon %s", path)
        # Stylesheet url() wants forward slashes on every platform.
        paths[name] = path.replace("\\", "/")
    return paths


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


def _build_stylesheet(p: Palette, icons: Dict[str, str]) -> str:
    """Render the application stylesheet for one palette."""
    i = icons
    return f"""
/* ---- Base ---------------------------------------------------------- */
QMainWindow, QDialog {{ background-color: {p.base}; color: {p.text}; }}
QWidget {{ background-color: {p.base}; color: {p.text}; font-size: 12px; }}
QLabel, QCheckBox, QRadioButton {{ background-color: transparent; }}
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {{ color: {p.disabled}; }}
QToolTip {{ background-color: {p.surface0}; color: {p.text};
    border: 1px solid {p.surface1}; padding: 4px 6px; }}

/* ---- Group boxes --------------------------------------------------- */
QGroupBox {{ border: 1px solid {p.surface1}; border-radius: 6px; margin-top: 10px;
    padding: 14px 8px 8px 8px; font-weight: bold; color: {p.accent}; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 4px; }}
QGroupBox:disabled {{ color: {p.disabled}; }}
QGroupBox::indicator {{ width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid {p.surface2}; background-color: {p.surface0}; }}
QGroupBox::indicator:checked {{ background-color: {p.accent};
    border-color: {p.accent}; image: url("{i['check']}"); }}

/* ---- Menus, toolbar, status bar ------------------------------------ */
QMenuBar {{ background-color: {p.mantle}; color: {p.text};
    border-bottom: 1px solid {p.surface0}; }}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background-color: {p.surface0}; border-radius: 4px; }}
QMenu {{ background-color: {p.base}; color: {p.text}; border: 1px solid {p.surface1};
    border-radius: 4px; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 20px; border-radius: 3px; }}
QMenu::item:selected {{ background-color: {p.surface0}; }}
QMenu::item:disabled {{ color: {p.disabled}; }}
QMenu::separator {{ height: 1px; background-color: {p.surface0}; margin: 4px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; border-radius: 3px; }}
QMenu::indicator:non-exclusive:checked {{ background-color: {p.accent};
    image: url("{i['check']}"); }}
QMenu::indicator:non-exclusive:unchecked {{ border: 1px solid {p.border_strong}; }}
QMenu::indicator:exclusive:checked {{ background-color: {p.accent}; border-radius: 7px;
    image: url("{i['dot']}"); }}
QMenu::indicator:exclusive:unchecked {{ border: 1px solid {p.border_strong};
    border-radius: 7px; }}
QToolBar {{ background-color: {p.mantle}; border: none;
    border-bottom: 1px solid {p.surface0}; spacing: 6px; padding: 4px 6px; }}
QToolBar::separator {{ background-color: {p.surface0}; width: 1px; margin: 4px 6px; }}
QToolBar QLabel {{ color: {p.subtext}; padding: 0 2px; }}
QToolBar QToolButton {{ background-color: {p.button}; color: {p.text};
    border: 1px solid {p.surface1}; border-radius: 4px; padding: 4px 14px;
    font-weight: bold; }}
QToolBar QToolButton:hover {{ background-color: {p.button_hover};
    border-color: {p.surface2}; }}
QToolBar QToolButton:pressed {{ background-color: {p.button_pressed}; }}
QToolBar QToolButton:checked {{ background-color: {p.danger}; color: {p.on_accent};
    border-color: {p.danger}; }}
QToolBar QToolButton:disabled {{ color: {p.disabled}; background-color: {p.mantle};
    border-color: {p.surface0}; }}
QStatusBar {{ background-color: {p.mantle}; border-top: 1px solid {p.surface0};
    color: {p.subtext}; font-size: 11px; }}
QStatusBar QLabel {{ color: {p.subtext}; padding: 0 6px; }}
QStatusBar::item {{ border: none; }}

/* ---- Buttons ------------------------------------------------------- */
QPushButton, QToolButton {{ background-color: {p.button}; color: {p.text};
    border: 1px solid {p.surface1}; border-radius: 4px; padding: 5px 14px; }}
QPushButton {{ min-height: 16px; }}
QToolButton {{ padding: 4px 8px; }}
QPushButton:hover, QToolButton:hover {{ background-color: {p.button_hover};
    border-color: {p.surface2}; }}
QPushButton:pressed, QToolButton:pressed {{ background-color: {p.button_pressed}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {p.accent}; }}
QPushButton:checked, QToolButton:checked {{ background-color: {p.accent};
    color: {p.on_accent}; border-color: {p.accent}; }}
QPushButton:disabled, QToolButton:disabled {{ background-color: {p.mantle};
    color: {p.disabled}; border-color: {p.surface0}; }}
QPushButton:default {{ border-color: {p.accent}; }}
QPushButton[role="primary"], QToolButton[role="primary"] {{
    background-color: {p.accent}; color: {p.on_accent};
    border-color: {p.accent}; font-weight: bold; }}
QPushButton[role="primary"]:hover, QToolButton[role="primary"]:hover {{
    background-color: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[role="primary"]:disabled, QToolButton[role="primary"]:disabled {{
    background-color: {p.surface0}; color: {p.disabled};
    border-color: {p.surface0}; }}
QPushButton[role="danger"], QToolButton[role="danger"] {{
    background-color: {p.danger_bg}; color: {p.danger};
    border-color: {p.danger}; font-weight: bold; }}
QPushButton[role="danger"]:hover, QToolButton[role="danger"]:hover {{
    background-color: {p.danger}; color: {p.on_accent}; }}
QPushButton[role="danger"]:checked, QToolButton[role="danger"]:checked {{
    background-color: {p.danger}; color: {p.on_accent}; border-color: {p.danger}; }}
QPushButton[role="danger"]:disabled, QToolButton[role="danger"]:disabled {{
    background-color: {p.mantle}; color: {p.disabled};
    border-color: {p.surface0}; }}
QPushButton[role="compact"] {{ padding: 3px 6px; font-size: 11px; }}
QWidget[role="header-strip"] QPushButton[role="compact"],
QFrame[role="header-strip"] QPushButton[role="compact"] {{ min-height: 20px;
    padding: 3px 8px; font-size: 12px; }}
QDialogButtonBox QPushButton {{ min-width: 72px; }}

/* ---- Inputs -------------------------------------------------------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateTimeEdit {{
    background-color: {p.field}; color: {p.text}; border: 1px solid {p.border_strong};
    placeholder-text-color: {p.subtext};
    border-radius: 4px; padding: 3px 6px; min-height: 20px;
    selection-background-color: {p.accent}; selection-color: {p.on_accent}; }}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {p.subtext}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {p.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{ background-color: {p.mantle}; color: {p.disabled};
    border-color: {p.surface0}; }}
QLineEdit:read-only {{ background-color: {p.mantle}; }}
QLineEdit[tone="success"], QSpinBox[tone="success"], QDoubleSpinBox[tone="success"],
QComboBox[tone="success"] {{ border-color: {p.success}; }}
QLineEdit[tone="warning"], QSpinBox[tone="warning"], QDoubleSpinBox[tone="warning"],
QComboBox[tone="warning"] {{ border-color: {p.warning}; }}
QLineEdit[tone="danger"], QSpinBox[tone="danger"], QDoubleSpinBox[tone="danger"],
QComboBox[tone="danger"] {{ border-color: {p.danger}; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
    width: 20px; border: none; }}
QComboBox::down-arrow {{ image: url("{i['down']}"); width: 12px; height: 12px; }}
QComboBox::down-arrow:disabled {{ image: url("{i['down_disabled']}"); }}
QComboBox QAbstractItemView {{ background-color: {p.base}; color: {p.text};
    border: 1px solid {p.surface1}; outline: none; padding: 2px;
    selection-background-color: {p.surface0}; selection-color: {p.text}; }}
QSpinBox, QDoubleSpinBox {{ padding-top: 2px; padding-bottom: 2px;
    padding-right: 20px; min-height: 19px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border;
    subcontrol-position: top right; width: 18px; border: none;
    border-left: 1px solid {p.surface1}; border-top-right-radius: 4px;
    background-color: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border;
    subcontrol-position: bottom right; width: 18px; border: none;
    border-left: 1px solid {p.surface1}; border-bottom-right-radius: 4px;
    background-color: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {p.surface1}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{i['up']}");
    width: 10px; height: 10px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{i['down']}");
    width: 10px; height: 10px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {{
    image: url("{i['up_disabled']}"); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{
    image: url("{i['down_disabled']}"); }}

/* ---- Check boxes and radio buttons --------------------------------- */
QCheckBox, QRadioButton {{ spacing: 6px; color: {p.text}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid {p.border_strong}; background-color: {p.field}; }}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked {{ background-color: {p.accent}; border-color: {p.accent};
    image: url("{i['check']}"); }}
QCheckBox::indicator:indeterminate {{ background-color: {p.accent};
    border-color: {p.accent}; image: url("{i['partial']}"); }}
QCheckBox::indicator:disabled {{ background-color: {p.mantle};
    border-color: {p.surface0}; }}
QCheckBox::indicator:checked:disabled {{ background-color: {p.surface0};
    image: url("{i['check_disabled']}"); }}
QRadioButton::indicator {{ width: 16px; height: 16px; border-radius: 8px;
    border: 1px solid {p.border_strong}; background-color: {p.field}; }}
QRadioButton::indicator:hover {{ border-color: {p.accent}; }}
QRadioButton::indicator:checked {{ background-color: {p.accent};
    border-color: {p.accent}; image: url("{i['dot']}"); }}
QRadioButton::indicator:disabled {{ background-color: {p.mantle};
    border-color: {p.surface0}; }}
QRadioButton::indicator:checked:disabled {{ background-color: {p.surface0};
    image: url("{i['dot_disabled']}"); }}

/* ---- Sliders and progress ------------------------------------------ */
QSlider {{ background: transparent; }}
QSlider::groove:horizontal {{ height: 6px; background-color: {p.surface0};
    border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background-color: {p.accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background-color: {p.accent}; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px; border: 2px solid {p.base}; }}
QSlider::handle:horizontal:hover {{ background-color: {p.accent_hover}; }}
QSlider::groove:vertical {{ width: 6px; background-color: {p.surface0};
    border-radius: 3px; }}
QSlider::add-page:vertical {{ background-color: {p.accent}; border-radius: 3px; }}
QSlider::handle:vertical {{ background-color: {p.accent}; width: 14px; height: 14px;
    margin: 0 -4px; border-radius: 7px; border: 2px solid {p.base}; }}
QSlider::sub-page:horizontal:disabled, QSlider::add-page:vertical:disabled,
QSlider::handle:disabled {{ background-color: {p.surface1}; }}
QProgressBar {{ background-color: {p.surface0}; border: 1px solid {p.surface1};
    border-radius: 4px; text-align: center; color: {p.text}; font-size: 10px;
    min-height: 12px; }}
QProgressBar::chunk {{ background-color: {p.accent_muted}; border-radius: 3px; }}

/* ---- Tabs ---------------------------------------------------------- */
QTabWidget::pane {{ border: 1px solid {p.surface1}; border-radius: 4px;
    background-color: {p.base}; top: -1px; }}
QTabBar {{ background: transparent; outline: 0; }}
QTabBar::tab {{ background-color: {p.mantle}; color: {p.subtext};
    border: 1px solid {p.surface1}; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 5px 10px; margin-right: 2px; }}
QTabBar::tab:selected {{ background-color: {p.base}; color: {p.accent};
    border-bottom: 2px solid {p.accent}; }}
QTabBar::tab:hover:!selected {{ background-color: {p.surface0}; color: {p.text}; }}
QTabBar::tab:disabled {{ color: {p.disabled}; }}
QTabBar QToolButton {{ padding: 2px; }}
QTabBar[documentMode="true"]::tab {{ background-color: transparent; border: none;
    border-bottom: 2px solid transparent; border-radius: 0; padding: 4px 8px;
    margin-right: 4px; color: {p.subtext}; }}
QTabBar[documentMode="true"]::tab:selected {{ background-color: transparent;
    color: {p.accent}; border-bottom: 2px solid {p.accent}; }}
QTabBar[documentMode="true"]::tab:hover:!selected {{ background-color: transparent;
    color: {p.text}; border-bottom: 2px solid {p.surface1}; }}

/* ---- Item views and text ------------------------------------------- */
QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView {{
    background-color: {p.mantle}; color: {p.text}; border: 1px solid {p.surface1};
    border-radius: 4px; alternate-background-color: {p.base};
    gridline-color: {p.surface0}; outline: none;
    selection-background-color: {p.surface1}; selection-color: {p.text}; }}
QListView::item, QTreeView::item {{ padding: 3px 4px; }}
QTableView::item {{ padding: 0 4px; }}
QListView::item:hover, QTreeView::item:hover, QTableView::item:hover {{
    background-color: {p.surface0}; }}
QListView::item:selected, QTreeView::item:selected, QTableView::item:selected {{
    background-color: {p.surface1}; color: {p.text}; }}
QHeaderView {{ background-color: {p.mantle}; }}
QHeaderView::section {{ background-color: {p.mantle}; color: {p.subtext}; border: none;
    border-bottom: 1px solid {p.surface1}; border-right: 1px solid {p.surface0};
    padding: 4px 8px; font-weight: bold; }}
QTableCornerButton::section {{ background-color: {p.mantle}; border: none; }}
QTextEdit, QPlainTextEdit, QTextBrowser {{ background-color: {p.mantle};
    color: {p.text}; placeholder-text-color: {p.subtext}; border: 1px solid {p.surface1}; border-radius: 4px;
    selection-background-color: {p.accent}; selection-color: {p.on_accent}; }}
QTextEdit[role="terminal"], QPlainTextEdit[role="terminal"] {{
    background-color: {p.lcd_bg}; color: {p.lcd_text};
    font-family: {MONO_FONT_STACK}; }}

/* ---- Scrolling and splitters --------------------------------------- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background-color: {p.base}; }}
QScrollBar:vertical {{ background-color: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background-color: {p.surface1}; border-radius: 4px;
    min-height: 24px; margin: 1px; }}
QScrollBar:horizontal {{ background-color: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background-color: {p.surface1}; border-radius: 4px;
    min-width: 24px; margin: 1px; }}
QScrollBar::handle:hover {{ background-color: {p.surface2}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background-color: {p.surface0}; }}
QSplitter::handle:hover {{ background-color: {p.accent}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical {{ height: 4px; }}

/* ---- Roles --------------------------------------------------------- */
QLabel[role="caption"] {{ color: {p.caption}; font-size: 10px; font-weight: bold;
    letter-spacing: 1px; }}
QLabel[role="hint"] {{ color: {p.subtext}; font-size: 11px; }}
QLabel[role="muted"] {{ color: {p.subtext}; }}
QLabel[role="value"] {{ color: {p.text}; font-weight: bold; }}
QLabel[role="display"] {{ color: {p.text}; font-size: 26px; font-weight: bold; }}
QLabel[role="placeholder"] {{ color: {p.subtext}; font-style: italic; padding: 16px; }}
QLabel[role="lcd"] {{ font-family: {MONO_FONT_STACK}; font-size: 18px;
    font-weight: bold; color: {p.lcd_text}; background-color: {p.lcd_bg};
    border: 1px solid {p.lcd_border}; border-radius: 4px; padding: 2px 10px;
    letter-spacing: 1px; }}
QLabel[role="lcd-small"] {{ font-family: {MONO_FONT_STACK}; font-size: 14px;
    color: {p.lcd_alt}; background-color: {p.lcd_bg};
    border: 1px solid {p.lcd_border}; border-radius: 4px; padding: 2px 8px; }}
QWidget[role="header-strip"], QFrame[role="header-strip"] {{
    background-color: {p.mantle}; border: none;
    border-bottom: 1px solid {p.surface0}; }}
QFrame[role="card"] {{ background-color: {p.mantle}; border: 1px solid {p.surface1};
    border-radius: 6px; }}
QFrame[role="card"] QLabel {{ background: transparent; }}
QLabel[role="callout"], QFrame[role="callout"] {{ color: {p.info};
    background-color: {p.info_bg}; border: 1px solid {p.info}; border-radius: 4px;
    padding: 6px 8px; }}
QLabel[role="badge"] {{ color: {p.on_accent}; background-color: {p.accent};
    border-radius: 3px; padding: 1px 6px; font-weight: bold; }}
QLabel[role="keycap"] {{ color: {p.text}; background-color: {p.surface0};
    border: 1px solid {p.surface2}; border-bottom-width: 2px; border-radius: 4px;
    padding: 0 5px; font-size: 12px; }}
QFrame[role="divider"] {{ background-color: {p.surface1}; border: none;
    min-height: 1px; max-height: 1px; }}
QFrame[role="vdivider"] {{ background-color: {p.surface1}; border: none;
    min-width: 1px; max-width: 1px; }}

/* ---- Tones (after roles so they override role colors) -------------- */
QLabel[tone="success"] {{ color: {p.success}; }}
QLabel[tone="warning"] {{ color: {p.warning}; }}
QLabel[tone="danger"] {{ color: {p.danger}; }}
QLabel[tone="info"] {{ color: {p.info}; }}
QLabel[tone="accent"] {{ color: {p.accent}; }}
QLabel[tone="muted"] {{ color: {p.subtext}; }}
QLabel[role="lcd"][tone="success"], QLabel[role="lcd-small"][tone="success"] {{
    color: {p.success}; }}
QLabel[role="lcd"][tone="warning"], QLabel[role="lcd-small"][tone="warning"] {{
    color: {p.warning}; }}
QLabel[role="lcd"][tone="danger"], QLabel[role="lcd-small"][tone="danger"] {{
    color: {p.danger}; }}
QLabel[role="lcd"][tone="muted"], QLabel[role="lcd-small"][tone="muted"] {{
    color: {p.caption}; }}
QLabel[role="callout"][tone="success"], QFrame[role="callout"][tone="success"] {{
    color: {p.success}; background-color: {p.success_bg}; border-color: {p.success}; }}
QLabel[role="callout"][tone="warning"], QFrame[role="callout"][tone="warning"] {{
    color: {p.warning}; background-color: {p.warning_bg}; border-color: {p.warning}; }}
QLabel[role="callout"][tone="danger"], QFrame[role="callout"][tone="danger"] {{
    color: {p.danger}; background-color: {p.danger_bg}; border-color: {p.danger}; }}
QLabel[role="badge"][tone="success"] {{ background-color: {p.success};
    color: {p.on_accent}; }}
QLabel[role="badge"][tone="warning"] {{ background-color: {p.warning};
    color: {p.on_accent}; }}
QLabel[role="badge"][tone="danger"] {{ background-color: {p.danger};
    color: {p.on_accent}; }}
QLabel[role="badge"][tone="info"] {{ background-color: {p.info};
    color: {p.on_accent}; }}
QLabel[role="badge"][tone="accent"] {{ background-color: {p.accent};
    color: {p.on_accent}; }}
QLabel[role="badge"][tone="muted"] {{ background-color: {p.surface1};
    color: {p.text}; }}

/* ---- Keyboard focus (last, so it wins over roles and hover) --------- */
/* A 2 px ring (padding shrinks by 1 px so the size never changes). The
   accent-filled states get a double ring in the on-accent color so focus
   stays distinct from the accent border of the default button. */
QPushButton:focus {{ border: 2px solid {p.accent}; padding: 4px 13px; }}
QToolButton:focus {{ border: 2px solid {p.accent}; padding: 3px 7px; }}
QToolBar QToolButton:focus {{ padding: 3px 13px; }}
QPushButton[role="compact"]:focus {{ padding: 2px 5px; }}
QWidget[role="header-strip"] QPushButton[role="compact"]:focus,
QFrame[role="header-strip"] QPushButton[role="compact"]:focus {{ padding: 2px 7px; }}
QPushButton[role="primary"]:focus, QToolButton[role="primary"]:focus,
QPushButton:checked:focus, QToolButton:checked:focus {{
    border: 3px double {p.on_accent}; padding: 3px 12px; }}
QPushButton[role="danger"]:focus, QToolButton[role="danger"]:focus {{
    border: 2px solid {p.text}; padding: 4px 13px; }}
QCheckBox::indicator:focus, QRadioButton::indicator:focus {{
    border: 2px solid {p.accent}; width: 14px; height: 14px; }}
QCheckBox::indicator:checked:focus, QCheckBox::indicator:indeterminate:focus,
QRadioButton::indicator:checked:focus {{ border: 2px solid {p.text}; }}
QSlider::handle:horizontal:focus, QSlider::handle:vertical:focus {{
    border: 2px solid {p.text}; }}
QTabBar::tab:selected:focus {{ background-color: {p.surface0};
    border: 1px solid {p.accent}; border-bottom: 2px solid {p.accent}; }}
QTabBar[documentMode="true"]::tab:selected:focus {{ background-color: {p.surface0};
    border: none; border-bottom: 2px solid {p.accent}; }}
QListWidget:focus, QListView:focus, QTreeWidget:focus, QTreeView:focus,
QTableWidget:focus, QTableView:focus, QTextEdit:focus, QPlainTextEdit:focus,
QTextBrowser:focus {{ border-color: {p.accent}; }}
"""


def get_stylesheet(theme: str) -> str:
    """Return the stylesheet text for the given theme name."""
    p = get_palette(normalize_theme(theme))
    return _build_stylesheet(p, _icons(p))


def build_qpalette(theme: Optional[str] = None) -> Any:
    """A ``QPalette`` matching the theme, for Fusion-drawn and native pieces.

    The stylesheet styles most widgets, but message boxes, file dialogs,
    placeholder text, links and custom-painted widgets read the QPalette.
    """
    from PyQt6.QtGui import QColor, QPalette

    p = get_palette(theme)
    pal = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    assignments = {
        role.Window: p.base,
        role.WindowText: p.text,
        role.Base: p.field,
        role.AlternateBase: p.mantle,
        role.ToolTipBase: p.surface0,
        role.ToolTipText: p.text,
        role.PlaceholderText: p.subtext,
        role.Text: p.text,
        role.Button: p.surface0,
        role.ButtonText: p.text,
        role.BrightText: p.danger,
        role.Highlight: p.accent,
        role.HighlightedText: p.on_accent,
        role.Link: p.accent,
        role.LinkVisited: p.accent_hover,
        role.Light: p.surface2,
        role.Midlight: p.surface1,
        role.Mid: p.surface0,
        role.Dark: p.mantle,
        role.Shadow: p.crust,
    }
    for r, value in assignments.items():
        pal.setColor(r, QColor(value))
    for r in (role.WindowText, role.Text, role.ButtonText, role.HighlightedText):
        pal.setColor(group.Disabled, r, QColor(p.disabled))
    pal.setColor(group.Disabled, role.Base, QColor(p.mantle))
    pal.setColor(group.Disabled, role.Button, QColor(p.mantle))
    return pal


def _get_notifier() -> Any:
    global _notifier
    if _notifier is None:
        from PyQt6.QtCore import QObject, pyqtSignal

        class ThemeNotifier(QObject):
            """Emits ``theme_changed(name)`` after :func:`apply_theme`."""

            theme_changed = pyqtSignal(str)

        _notifier = ThemeNotifier()
    return _notifier


def theme_notifier() -> Any:
    """Process-wide object whose ``theme_changed(str)`` fires on theme switch.

    Custom-painted widgets that cache pixmaps connect to it to invalidate
    their caches; widgets that paint straight from :func:`get_palette` only
    need the repaint :func:`apply_theme` already triggers.
    """
    return _get_notifier()


def apply_theme(app: Any, theme: str) -> str:
    """Apply ``theme`` to the whole application and notify listeners.

    Sets the QPalette and the stylesheet, repaints every widget and emits
    :func:`theme_notifier` ``.theme_changed``.

    Args:
        app: The ``QApplication``.
        theme: ``"dark"`` or ``"light"`` (anything else means dark).

    Returns:
        The normalized theme name that was applied.
    """
    global _current_theme
    name = normalize_theme(theme)
    _current_theme = name
    if app is None:
        return name
    app.setPalette(build_qpalette(name))
    app.setStyleSheet(get_stylesheet(name))
    try:
        for widget in app.allWidgets():
            widget.update()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Theme repaint failed: %s", exc)
    try:
        _get_notifier().theme_changed.emit(name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Theme notification failed: %s", exc)
    return name


def repolish(widget: Any) -> None:
    """Re-apply stylesheet rules after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_role(widget: Any, role: Optional[str] = None, tone: Any = ...) -> Any:
    """Tag ``widget`` with a stylesheet ``role`` (and optionally a ``tone``).

    Pass ``role=None`` to clear the role. ``tone`` is left unchanged unless
    given (``None`` clears it). Returns the widget for chaining.
    """
    widget.setProperty("role", role or "")
    if tone is not ...:
        widget.setProperty("tone", tone or "")
    repolish(widget)
    return widget


def set_tone(widget: Any, tone: Optional[str]) -> Any:
    """Set (or clear, with ``None``) the color ``tone`` of ``widget``."""
    if (widget.property("tone") or "") == (tone or ""):
        return widget
    widget.setProperty("tone", tone or "")
    repolish(widget)
    return widget


def mono_font(point_size: float = 10.0, bold: bool = False) -> Any:
    """A monospace ``QFont`` for custom painting (axis labels, readouts)."""
    from PyQt6.QtGui import QFont

    font = QFont()
    font.setFamilies(
        [
            "JetBrains Mono",
            "Cascadia Mono",
            "Consolas",
            "SF Mono",
            "Menlo",
            "DejaVu Sans Mono",
            "monospace",
        ]
    )
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSizeF(point_size)
    font.setBold(bold)
    return font
