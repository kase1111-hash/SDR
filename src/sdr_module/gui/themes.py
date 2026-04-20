"""
GUI stylesheets for dark (Catppuccin Mocha) and light (Catppuccin Latte) themes.
"""

DARK_STYLESHEET = """
QMainWindow, QDialog { background-color: #1e1e2e; color: #cdd6f4; }
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-size: 12px; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px;
    padding: 12px 6px 6px 6px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QToolBar { background-color: #181825; border-bottom: 1px solid #313244;
    spacing: 6px; padding: 4px; }
QToolBar QLabel { background: transparent; color: #a6adc8; padding: 0 2px; }
QToolBar QToolButton { background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 4px 12px; font-weight: bold; }
QToolBar QToolButton:hover { background-color: #45475a; border-color: #585b70; }
QToolBar QToolButton:pressed { background-color: #585b70; }
QToolBar QToolButton:checked { background-color: #f38ba8; color: #1e1e2e;
    border-color: #f38ba8; }
QStatusBar { background-color: #181825; border-top: 1px solid #313244;
    color: #a6adc8; font-size: 11px; }
QStatusBar QLabel { background: transparent; color: #a6adc8; padding: 0 6px; }
QStatusBar::item { border: none; }
QMenuBar { background-color: #181825; color: #cdd6f4;
    border-bottom: 1px solid #313244; }
QMenuBar::item:selected { background-color: #313244; border-radius: 4px; }
QMenu { background-color: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 4px; }
QMenu::item:selected { background-color: #313244; border-radius: 3px; }
QMenu::separator { height: 1px; background-color: #313244; margin: 4px 8px; }
QPushButton { background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 5px 14px; }
QPushButton:hover { background-color: #45475a; border-color: #585b70; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }
QPushButton:disabled { background-color: #1e1e2e; color: #585b70; border-color: #313244; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
    border-radius: 4px; padding: 3px 8px; min-height: 20px; }
QComboBox:hover { border-color: #585b70; }
QComboBox QAbstractItemView { background-color: #1e1e2e; color: #cdd6f4;
    border: 1px solid #45475a; selection-background-color: #313244; }
QSpinBox, QDoubleSpinBox { background-color: #313244; color: #cdd6f4;
    border: 1px solid #45475a; border-radius: 4px; padding: 3px 6px; min-height: 20px; }
QSlider::groove:horizontal { height: 6px; background-color: #313244; border-radius: 3px; }
QSlider::handle:horizontal { background-color: #89b4fa; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #45475a; background-color: #313244; }
QCheckBox::indicator:checked { background-color: #89b4fa; border-color: #89b4fa; }
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px;
    background-color: #1e1e2e; top: -1px; }
QTabBar::tab { background-color: #181825; color: #a6adc8; border: 1px solid #45475a;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 5px 12px; margin-right: 2px; }
QTabBar::tab:selected { background-color: #1e1e2e; color: #89b4fa;
    border-bottom: 2px solid #89b4fa; }
QTableWidget { background-color: #181825; color: #cdd6f4; gridline-color: #313244;
    border: 1px solid #45475a; border-radius: 4px;
    selection-background-color: #313244; }
QHeaderView::section { background-color: #181825; color: #a6adc8; border: none;
    border-bottom: 1px solid #45475a; padding: 4px 8px; font-weight: bold; }
QTextEdit, QPlainTextEdit, QListWidget { background-color: #181825; color: #a6e3a1;
    border: 1px solid #45475a; border-radius: 4px; }
QProgressBar { background-color: #313244; border: 1px solid #45475a;
    border-radius: 4px; text-align: center; color: #cdd6f4; font-size: 10px; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
"""

LIGHT_STYLESHEET = """
QMainWindow, QDialog { background-color: #eff1f5; color: #4c4f69; }
QWidget { background-color: #eff1f5; color: #4c4f69; font-size: 12px; }
QGroupBox { border: 1px solid #bcc0cc; border-radius: 6px; margin-top: 8px;
    padding: 12px 6px 6px 6px; font-weight: bold; color: #1e66f5; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QToolBar { background-color: #e6e9ef; border-bottom: 1px solid #ccd0da;
    spacing: 6px; padding: 4px; }
QToolBar QLabel { background: transparent; color: #6c6f85; padding: 0 2px; }
QToolBar QToolButton { background-color: #ccd0da; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 4px 12px; font-weight: bold; }
QToolBar QToolButton:hover { background-color: #bcc0cc; }
QToolBar QToolButton:checked { background-color: #d20f39; color: #eff1f5;
    border-color: #d20f39; }
QStatusBar { background-color: #e6e9ef; border-top: 1px solid #ccd0da;
    color: #6c6f85; font-size: 11px; }
QStatusBar QLabel { background: transparent; color: #6c6f85; padding: 0 6px; }
QMenuBar { background-color: #e6e9ef; color: #4c4f69;
    border-bottom: 1px solid #ccd0da; }
QMenuBar::item:selected { background-color: #ccd0da; border-radius: 4px; }
QMenu { background-color: #eff1f5; color: #4c4f69; border: 1px solid #bcc0cc;
    border-radius: 4px; padding: 4px; }
QMenu::item:selected { background-color: #ccd0da; border-radius: 3px; }
QPushButton { background-color: #ccd0da; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 5px 14px; }
QPushButton:hover { background-color: #bcc0cc; }
QPushButton:checked { background-color: #1e66f5; color: #eff1f5; border-color: #1e66f5; }
QPushButton:disabled { background-color: #eff1f5; color: #bcc0cc; border-color: #ccd0da; }
QComboBox { background-color: #ccd0da; color: #4c4f69; border: 1px solid #bcc0cc;
    border-radius: 4px; padding: 3px 8px; min-height: 20px; }
QComboBox QAbstractItemView { background-color: #eff1f5; color: #4c4f69;
    border: 1px solid #bcc0cc; selection-background-color: #ccd0da; }
QSpinBox, QDoubleSpinBox { background-color: #ccd0da; color: #4c4f69;
    border: 1px solid #bcc0cc; border-radius: 4px; padding: 3px 6px; min-height: 20px; }
QSlider::groove:horizontal { height: 6px; background-color: #ccd0da; border-radius: 3px; }
QSlider::handle:horizontal { background-color: #1e66f5; width: 14px; height: 14px;
    margin: -4px 0; border-radius: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px;
    border: 1px solid #bcc0cc; background-color: #ccd0da; }
QCheckBox::indicator:checked { background-color: #1e66f5; border-color: #1e66f5; }
QTabWidget::pane { border: 1px solid #bcc0cc; border-radius: 4px;
    background-color: #eff1f5; top: -1px; }
QTabBar::tab { background-color: #e6e9ef; color: #6c6f85; border: 1px solid #bcc0cc;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 5px 12px; margin-right: 2px; }
QTabBar::tab:selected { background-color: #eff1f5; color: #1e66f5;
    border-bottom: 2px solid #1e66f5; }
QTableWidget { background-color: #e6e9ef; color: #4c4f69; gridline-color: #ccd0da;
    border: 1px solid #bcc0cc; border-radius: 4px;
    selection-background-color: #ccd0da; }
QHeaderView::section { background-color: #e6e9ef; color: #6c6f85; border: none;
    border-bottom: 1px solid #bcc0cc; padding: 4px 8px; font-weight: bold; }
QTextEdit, QPlainTextEdit, QListWidget { background-color: #e6e9ef; color: #40a02b;
    border: 1px solid #bcc0cc; border-radius: 4px; }
QProgressBar { background-color: #ccd0da; border: 1px solid #bcc0cc;
    border-radius: 4px; text-align: center; color: #4c4f69; font-size: 10px; }
QProgressBar::chunk { background-color: #1e66f5; border-radius: 3px; }
"""


def get_stylesheet(theme: str) -> str:
    """Return the stylesheet text for the given theme name."""
    return LIGHT_STYLESHEET if theme.lower() == "light" else DARK_STYLESHEET
