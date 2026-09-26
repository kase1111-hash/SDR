"""
Help dialog — keyboard shortcuts reference.

The list comes from one table, :data:`SHORTCUTS`, and is checked against the
shortcuts the parent window has actually installed: its ``QAction`` and
``QShortcut`` key bindings, keys a menu item advertises after a tab
(``"&Start Receiving\tSpace"``, for keys the window handles itself) and the
window's ``_TUNE_STEPS_HZ`` arrow-key tuning table. Entries whose keys are
not bound are left out, and bound menu shortcuts missing from the table are
added under their menu's name, so the dialog stays accurate when the main
window's shortcuts change (and on platforms without, say, a Quit key).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PyQt6.QtCore import QEvent, QObject, Qt
    from PyQt6.QtGui import QAction, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMenu,
        QScrollArea,
        QVBoxLayout,
        QWidget,
    )

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

if HAS_PYQT6:
    from .themes import set_role

# (category, key sequences in QKeySequence portable text, description).
# Several keys in one entry are shown together, e.g. Left / Right.
#
# Categories use the main window's menu names, so the static fallback groups
# entries the same way as the live list (which groups by the real menus).
SHORTCUTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("Radio", ("Space",), "Start or stop the receiver"),
    ("Radio", ("Ctrl+Shift+R",), "Start or stop recording"),
    ("Radio", ("Ctrl+L",), "Type a new center frequency"),
    ("Radio", ("Ctrl+B",), "Bookmark the current frequency"),
    ("Tuning", ("Left", "Right"), "Tune down / up by 10 kHz"),
    ("Tuning", ("Shift+Left", "Shift+Right"), "Tune down / up by 100 kHz"),
    ("Tuning", ("Ctrl+Left", "Ctrl+Right"), "Tune down / up by 1 MHz"),
    ("File", ("Ctrl+O",), "Open a recording"),
    ("File", ("Ctrl+S",), "Save the recording"),
    ("File", ("Ctrl+P",), "Save a screenshot"),
    ("File", ("Ctrl+Q",), "Exit SDR Module"),
    ("Tools", ("Ctrl+F",), "Frequency scanner"),
    ("Tools", ("Ctrl+R",), "AM/FM radio tuner"),
    ("Tools", ("Ctrl+E",), "Error history"),
    ("View", ("Ctrl+T",), "Switch between light and dark theme"),
    ("Help", ("F1",), "Keyboard shortcuts (this window)"),
)

# Submenus whose items are described as "Show the <item> panel".
_PANEL_MENUS = ("Panels",)

# Mouse gestures: (gesture shown as a keycap, description).
MOUSE_ACTIONS: Tuple[Tuple[str, str], ...] = (
    ("Click", "Tune to the clicked frequency on the spectrum or waterfall"),
)

_ARROWS = {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓"}

Entry = Tuple[str, Tuple[str, ...], str]


def _portable(key: str) -> str:
    """Normalize a key sequence string (``"shift+ctrl+r"`` -> ``"Ctrl+Shift+R"``)."""
    if not HAS_PYQT6:
        return key
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.PortableText)


def _clean_action_text(text: str) -> str:
    """``"&Open Recording...\tCtrl+O"`` -> ``"Open Recording"``."""
    text = text.split("\t", 1)[0]
    text = text.replace("&&", "\0").replace("&", "").replace("\0", "&")
    return text.rstrip(".…").strip()


def _advertised_key(text: str) -> str:
    """Key shown after a tab in a menu item's text, if it is a real key."""
    if "\t" not in text:
        return ""
    return _portable(text.split("\t", 1)[1].strip())


def _format_step(step_hz: float) -> str:
    for unit, scale in (("MHz", 1e6), ("kHz", 1e3)):
        if abs(step_hz) >= scale:
            return f"{step_hz / scale:g} {unit}"
    return f"{step_hz:g} Hz"


def tuning_steps(window) -> Optional[List[Tuple[str, float]]]:
    """Arrow-key tuning steps from ``window._TUNE_STEPS_HZ``.

    Returns ``[(modifier prefix, step Hz), ...]`` smallest step first, e.g.
    ``[("", 10e3), ("Shift+", 100e3), ("Ctrl+", 1e6)]``, or ``None`` when the
    window has no such table.
    """
    table = getattr(window, "_TUNE_STEPS_HZ", None) if window is not None else None
    if not isinstance(table, dict) or not table or not HAS_PYQT6:
        return None
    names = (
        (Qt.KeyboardModifier.ControlModifier, "Ctrl+"),
        (Qt.KeyboardModifier.AltModifier, "Alt+"),
        (Qt.KeyboardModifier.ShiftModifier, "Shift+"),
        (Qt.KeyboardModifier.MetaModifier, "Meta+"),
    )
    steps = []
    for mods, step in table.items():
        try:
            prefix = "".join(name for flag, name in names if mods & flag)
            steps.append((prefix, float(step)))
        except (TypeError, ValueError):
            continue
    return sorted(steps, key=lambda item: item[1]) or None


def tuning_entries(steps: Sequence[Tuple[str, float]]) -> List[Entry]:
    """Shortcut entries for arrow-key tuning steps."""
    return [
        (
            "Tuning",
            (f"{prefix}Left", f"{prefix}Right"),
            f"Tune down / up by {_format_step(step)}",
        )
        for prefix, step in steps
    ]


def _menu_path(action) -> List[str]:
    """Titles of the menus holding ``action``, top-level menu first."""
    path: List[str] = []
    seen = set()
    current = action
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        menu = next(
            (
                obj
                for obj in current.associatedObjects()
                if isinstance(obj, QMenu) and obj.title()
            ),
            None,
        )
        if menu is None:
            break
        path.insert(0, _clean_action_text(menu.title()))
        current = menu.menuAction()
    return path


def collect_bindings(window) -> Optional[Dict[str, Tuple[str, str]]]:
    """Shortcuts installed on ``window``: portable key -> (category, text).

    ``category`` is the top-level menu title for menu actions and ``""``
    otherwise; ``text`` is the action's text prefixed with any submenu
    (``"Panels › Decoder"``), or a ``QShortcut``'s What's This text (usually
    ``""``). Returns ``None`` when there is no window to inspect.
    """
    if window is None or not HAS_PYQT6:
        return None
    bindings: Dict[str, Tuple[str, str]] = {}
    for action in window.findChildren(QAction):
        if not action.isVisible():
            continue
        path = _menu_path(action)
        category = path[0] if path else ""
        raw = action.text()
        text = " › ".join(path[1:] + [_clean_action_text(raw)])
        keys = [
            seq.toString(QKeySequence.SequenceFormat.PortableText)
            for seq in action.shortcuts()
        ]
        keys.append(_advertised_key(raw))
        for key in keys:
            if key:
                bindings.setdefault(key, (category, text))
    for shortcut in window.findChildren(QShortcut):
        if not shortcut.isEnabled():
            continue
        for seq in shortcut.keys():
            key = seq.toString(QKeySequence.SequenceFormat.PortableText)
            if key:
                bindings.setdefault(key, ("", shortcut.whatsThis()))
    return bindings


def build_entries(
    bindings: Optional[Dict[str, Tuple[str, str]]],
    table: Sequence[Entry] = SHORTCUTS,
    steps: Optional[Sequence[Tuple[str, float]]] = None,
) -> List[Entry]:
    """The shortcut entries to show, given the window's live ``bindings``.

    ``steps`` (see :func:`tuning_steps`) replaces the table's tuning rows
    with the window's real arrow-key steps. With no bindings to check
    against, the whole table is returned.
    """
    if steps:
        generated = tuning_entries(steps)
        merged: List[Entry] = []
        for entry in table:
            if entry[0] != "Tuning":
                merged.append(entry)
            elif generated:
                merged.extend(generated)
                generated = []
        table = merged + generated
        if bindings is not None:
            bindings = dict(bindings)
            for _cat, keys, _desc in tuning_entries(steps):
                for key in keys:
                    bindings.setdefault(_portable(key), ("", ""))
    if bindings is None:
        return list(table)
    entries: List[Entry] = []
    used = set()
    for category, keys, description in table:
        live = tuple(k for k in keys if _portable(k) in bindings)
        used.update(_portable(k) for k in keys)
        if live:
            # Group under the menu that holds the shortcut, so the dialog
            # mirrors the menus; keys with no menu keep the table's group.
            menu = bindings[_portable(live[0])][0]
            entries.append((menu or category, live, description))
    # Menu shortcuts the table does not know about yet.
    for key, (category, text) in bindings.items():
        if key in used or not text:
            continue
        entries.append((category or "Other", (key,), _describe_action(text)))
    return entries


def _describe_action(text: str) -> str:
    """Readable description for a menu action found on the window.

    ``"Panels › Decoder"`` -> ``"Show the Decoder panel"``; anything else is
    kept as the menu shows it.
    """
    parts = text.split(" › ")
    if len(parts) == 2 and parts[0] in _PANEL_MENUS:
        return f"Show the {parts[1]} panel"
    return text


def _keycap_parts(key: str) -> List[str]:
    """Split a key sequence into keycap labels in the platform's notation."""
    native = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)
    parts: List[str] = []
    for part in native.split("+"):
        if part == "":
            if not parts or parts[-1] != "+":
                parts.append("+")  # the plus key itself ("Ctrl++")
            continue
        parts.append(_ARROWS.get(part, part))
    return parts


def _keycap_tokens(keys: Sequence[str], literal: bool = False) -> List[str]:
    """Keycaps and joiners (``"+"``, ``"/"``) for one entry's keys.

    Keys that share modifiers are compacted: ``Shift+Left`` and
    ``Shift+Right`` become ``Shift + ← / →``.
    """
    if literal:
        return list(keys)
    split = [_keycap_parts(k) for k in keys]
    tokens: List[str] = []
    prefix = split[0][:-1] if split and split[0] else []
    if (
        len(split) > 1
        and all(len(p) > 1 or not prefix for p in split)
        and all(p[:-1] == prefix for p in split)
    ):
        for part in prefix:
            tokens += [part, "+"]
        for i, parts in enumerate(split):
            if i:
                tokens.append("/")
            tokens.append(parts[-1])
        return tokens
    for i, parts in enumerate(split):
        if i:
            tokens.append("/")
        for j, part in enumerate(parts):
            if j:
                tokens.append("+")
            tokens.append(part)
    return tokens


class _SwallowEnter(QObject if HAS_PYQT6 else object):
    """Keeps Enter in the filter box from closing the dialog."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            return True
        return False


class HelpDialog(QDialog if HAS_PYQT6 else object):
    """Shortcut reference dialog."""

    def __init__(self, parent=None):
        if not HAS_PYQT6:
            raise ImportError("PyQt6 is required")
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(560, 620)
        self.setMinimumSize(420, 360)

        window = parent.window() if parent is not None else None
        self._entries = build_entries(
            collect_bindings(window), steps=tuning_steps(window)
        )
        # (row widget, lowercase search text, category section)
        self._rows: List[Tuple[QWidget, str, QWidget]] = []
        self._sections: List[QWidget] = []
        self._caps: List[QWidget] = []  # keycap column of every row

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter shortcuts (e.g. tune, record, Ctrl+S)")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._enter_guard = _SwallowEnter(self)
        self._filter.installEventFilter(self._enter_guard)
        layout.addWidget(self._filter)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(0, 0, 8, 0)
        self._content_layout.setSpacing(14)
        self._populate()
        self._no_match = QLabel()
        self._no_match.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_match.setWordWrap(True)
        set_role(self._no_match, "placeholder")
        self._no_match.hide()
        self._content_layout.addWidget(self._no_match)
        self._content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        note = QLabel(
            "Space and the arrow keys act when the focused control does not "
            "use them itself. If they seem to do nothing, click the spectrum "
            "first."
        )
        note.setWordWrap(True)
        set_role(note, "hint")
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.setDefault(True)
        layout.addWidget(buttons)

        self._filter.setFocus()

    # ------------------------------------------------------------------ #
    def _populate(self) -> None:
        sections: Dict[str, Tuple[QWidget, QVBoxLayout]] = {}
        order: List[str] = []
        items: List[Tuple[str, List[str], str, str]] = []
        for category, keys, description in self._entries:
            items.append((category, list(keys), description, "keys"))
        for gesture, description in MOUSE_ACTIONS:
            items.append(("Mouse", [gesture], description, "mouse"))

        for category, keys, description, kind in items:
            if category not in sections:
                section = QWidget()
                box = QVBoxLayout(section)
                box.setContentsMargins(0, 0, 0, 0)
                box.setSpacing(6)
                caption = QLabel(category.upper())
                set_role(caption, "caption")
                box.addWidget(caption)
                sections[category] = (section, box)
                order.append(category)
            section, box = sections[category]
            row = self._make_row(keys, description, kind)
            box.addWidget(row)
            search = " ".join([category, description] + keys).lower()
            self._rows.append((row, search, section))

        for category in order:
            section = sections[category][0]
            self._sections.append(section)
            self._content_layout.addWidget(section)

        # One keycap column, as wide as the widest set of keys, so every
        # description starts at the same x without a fixed pixel guess.
        if self._caps:
            width = max(caps.sizeHint().width() for caps in self._caps)
            for caps in self._caps:
                caps.setFixedWidth(width)

    def _make_row(self, keys: List[str], description: str, kind: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)

        caps = QWidget()
        self._caps.append(caps)
        caps_layout = QHBoxLayout(caps)
        caps_layout.setContentsMargins(0, 0, 0, 0)
        caps_layout.setSpacing(4)
        for token in _keycap_tokens(keys, kind == "mouse"):
            if token in ("+", "/"):
                caps_layout.addWidget(self._joiner(token))
                continue
            cap = QLabel(token)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setMinimumWidth(22)
            set_role(cap, "badge", "muted")
            caps_layout.addWidget(cap)
        caps_layout.addStretch(1)
        h.addWidget(caps, 0, Qt.AlignmentFlag.AlignTop)

        text = QLabel(description)
        text.setWordWrap(True)
        h.addWidget(text, 1)
        return row

    @staticmethod
    def _joiner(symbol: str) -> QLabel:
        label = QLabel(symbol)
        set_role(label, "muted")
        return label

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        visible_sections = set()
        for row, search, section in self._rows:
            show = not needle or needle in search
            row.setVisible(show)
            if show:
                visible_sections.add(id(section))
        for section in self._sections:
            section.setVisible(id(section) in visible_sections)
        if visible_sections:
            self._no_match.hide()
        else:
            self._no_match.setText(f"No shortcuts match “{text.strip()}”.")
            self._no_match.show()

    def shortcut_entries(self) -> List[Entry]:
        """The keyboard entries this dialog shows (for tests and tooling)."""
        return list(self._entries)
