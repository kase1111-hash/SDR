"""
Shared pytest fixtures.

Qt only runs ``deleteLater()`` deletions when control returns to an event
loop, which never happens under pytest. Without help, every window a GUI test
"deletes" stays alive for the rest of the session, so later theme switches
restyle thousands of stale widgets (slow, and it has crashed Qt inside
``QApplication.setStyleSheet``). These fixtures flush deferred deletions after
each test and close any top-level windows a test module leaves behind.

An exception raised in a Qt slot or event handler makes PyQt call
``qFatal()`` while ``sys.excepthook`` is Python's default: the whole run
aborts ("Fatal Python error: Aborted") and pytest's captured output, with
the traceback, is lost. The session installs a hook that records such
exceptions instead, and the test they happen in fails with the traceback.
"""

import sys
import traceback

import pytest

# Exceptions raised in Qt slots and event handlers (see the module docstring).
_QT_ERRORS: list = []


def _record_qt_error(exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    _QT_ERRORS.append(text)
    # Also straight to the terminal, should the run die anyway.
    if sys.__stderr__ is not None:
        sys.__stderr__.write(f"\nException in a Qt callback:\n{text}")
        sys.__stderr__.flush()


def _fail_on_qt_errors(when: str) -> None:
    if _QT_ERRORS:
        errors = "\n".join(_QT_ERRORS)
        del _QT_ERRORS[:]
        pytest.fail(
            f"Exception in a Qt slot or event handler {when}:\n{errors}",
            pytrace=False,
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    result = yield
    _fail_on_qt_errors("during this test")
    return result


@pytest.fixture(autouse=True, scope="session")
def _qt_exceptions_fail_tests():
    """Record exceptions from Qt callbacks instead of aborting the run."""
    previous = sys.excepthook
    sys.excepthook = _record_qt_error
    yield
    sys.excepthook = previous


def _qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return None
    return QApplication.instance()


def _current_theme():
    try:
        from sdr_module.gui import themes
    except ImportError:
        return None
    return themes.current_theme()


def _restore_theme_name(theme: str) -> None:
    """Reset the theme module's notion of the active theme (no restyle)."""
    from sdr_module.gui import themes

    themes._current_theme = theme


def _flush_deferred_deletes() -> None:
    from PyQt6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.fixture(autouse=True)
def _qt_flush_deferred_deletes():
    """Run pending ``deleteLater()`` deletions once each test finishes (and
    fail the test if a Qt callback raised meanwhile)."""
    _fail_on_qt_errors("before this test (in an earlier cleanup)")
    yield
    if _qt_app() is not None:
        _flush_deferred_deletes()
    _fail_on_qt_errors("while cleaning up after this test")


@pytest.fixture(autouse=True, scope="module")
def _qt_close_module_windows():
    """Close top-level widgets a test module created and undo its theming.

    A module that applies a theme would otherwise leave the full application
    stylesheet in place, which makes every later widget construction several
    times slower.
    """
    app = _qt_app()
    before = {id(w) for w in app.topLevelWidgets()} if app is not None else set()
    style_sheet = app.styleSheet() if app is not None else ""
    palette = app.palette() if app is not None else None
    theme = _current_theme()
    yield
    app = _qt_app()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if id(widget) in before:
            continue
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:  # already deleted on the C++ side
            pass
    _flush_deferred_deletes()
    # Restyle only after the module's windows are gone.
    if app.styleSheet() != style_sheet:
        app.setStyleSheet(style_sheet)
    if palette is not None:
        app.setPalette(palette)
    if theme is not None:
        _restore_theme_name(theme)


@pytest.fixture(autouse=True, scope="session")
def _temporary_user_folders(tmp_path_factory):
    """File dialogs start in Documents/SDR Recordings (created on first
    use) and Pictures: the tests use temporary ones, not the user's."""
    try:
        from sdr_module.gui import file_dialogs
    except ImportError:
        yield
        return
    for name in ("documents", "pictures"):
        file_dialogs.LOCATION_OVERRIDES[name] = str(tmp_path_factory.mktemp(name))
    yield
    file_dialogs.LOCATION_OVERRIDES.clear()
