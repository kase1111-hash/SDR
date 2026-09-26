"""
Shared pytest fixtures.

Qt only runs ``deleteLater()`` deletions when control returns to an event
loop, which never happens under pytest. Without help, every window a GUI test
"deletes" stays alive for the rest of the session, so later theme switches
restyle thousands of stale widgets (slow, and it has crashed Qt inside
``QApplication.setStyleSheet``). These fixtures flush deferred deletions after
each test and close any top-level windows a test module leaves behind.
"""

import pytest


def _qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        return None
    return QApplication.instance()


def _flush_deferred_deletes() -> None:
    from PyQt6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.fixture(autouse=True)
def _qt_flush_deferred_deletes():
    """Run pending ``deleteLater()`` deletions once each test finishes."""
    yield
    if _qt_app() is not None:
        _flush_deferred_deletes()


@pytest.fixture(autouse=True, scope="module")
def _qt_close_module_windows():
    """Close and delete top-level widgets created by the test module."""
    app = _qt_app()
    before = {id(w) for w in app.topLevelWidgets()} if app is not None else set()
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
