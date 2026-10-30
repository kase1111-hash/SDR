"""
What part of the receiver's span the spectrum and the waterfall show.

The receiver's span is the LO plus or minus half the sample rate. At zoom
x1 the plots show all of it; at x2 ... x16 they show 1/zoom of it, around a
center frequency that stays inside the span. :class:`PlotView` holds that
state (no Qt); the main window gives both plots the same instance, so they
always zoom and pan together, and every plot maps frequencies to pixels
through it (axes, hover, click-to-tune, the passband, measurements).

Zooming keeps a chosen frequency where it is on screen: the tuned frequency
(View > Zoom) or the one under the mouse (Ctrl+wheel). Once zoomed, the view
follows the tuned frequency when tuning moves it out of sight, and stays at
the same frequencies when the LO is retuned (as long as they are still
inside the span).
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Callable, List, Optional, Tuple

#: Zoom factors offered (x1 is the full span).
ZOOM_LEVELS: Tuple[int, ...] = (1, 2, 4, 8, 16)
MAX_ZOOM = ZOOM_LEVELS[-1]


def normalize_zoom(zoom: Any) -> int:
    """The nearest of :data:`ZOOM_LEVELS` to ``zoom`` (1 if not a number)."""
    try:
        value = float(zoom)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(value) or value <= 1:
        return 1
    return min(ZOOM_LEVELS, key=lambda z: abs(math.log2(z) - math.log2(value)))


class PlotView:
    """The frequencies the plots show (see the module docstring)."""

    def __init__(self, center_hz: float = 100e6, span_hz: float = 2.4e6):
        self._full_center = float(center_hz)
        self._full_span = max(1.0, float(span_hz))
        self._zoom = 1
        self._center = self._full_center
        self._listeners: List[Any] = []

    # -- listeners ---------------------------------------------------------
    def subscribe(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after every change. Bound methods are held
        weakly (a deleted plot is dropped)."""
        try:
            ref: Any = weakref.WeakMethod(callback)  # type: ignore[arg-type]
        except TypeError:
            ref = lambda cb=callback: cb  # noqa: E731 - plain function: strong
        self._listeners.append(ref)

    def _notify(self) -> None:
        alive = []
        for ref in self._listeners:
            callback = ref()
            if callback is None:
                continue
            alive.append(ref)
            try:
                callback()
            except RuntimeError:  # pragma: no cover - widget already deleted
                alive.pop()
        self._listeners = alive

    # -- state -------------------------------------------------------------
    @property
    def zoom(self) -> int:
        return self._zoom

    @property
    def full_center(self) -> float:
        return self._full_center

    @property
    def full_span(self) -> float:
        return self._full_span

    @property
    def span(self) -> float:
        """Width of the frequencies shown (Hz)."""
        return self._full_span / self._zoom

    @property
    def center(self) -> float:
        return self._center

    @property
    def start(self) -> float:
        return self._center - self.span / 2

    @property
    def end(self) -> float:
        return self._center + self.span / 2

    def is_zoomed(self) -> bool:
        return self._zoom > 1

    def contains(self, freq_hz: float) -> bool:
        return self.start <= float(freq_hz) <= self.end

    def in_span(self, freq_hz: float) -> bool:
        """Whether ``freq_hz`` lies inside the receiver's (full) span."""
        half = self._full_span / 2
        return self._full_center - half <= float(freq_hz) <= self._full_center + half

    def _clamped(self, center: float) -> float:
        if self._zoom <= 1:
            return self._full_center
        room = (self._full_span - self.span) / 2
        return min(max(center, self._full_center - room), self._full_center + room)

    # -- changes -------------------------------------------------------------
    def set_full(self, center_hz: float, span_hz: float) -> bool:
        """The receiver's span changed (a retune, another sample rate). The
        view stays on the same frequencies where it can."""
        center, span = float(center_hz), max(1.0, float(span_hz))
        if (center, span) == (self._full_center, self._full_span):
            return False
        self._full_center, self._full_span = center, span
        self._center = self._clamped(self._center)
        self._notify()
        return True

    def set_zoom(
        self, zoom: Any, anchor_hz: Optional[float] = None, anchor_frac: float = 0.5
    ) -> bool:
        """Zoom to ``zoom``, keeping ``anchor_hz`` at ``anchor_frac`` of the
        width (0 = left edge; the default: the view's center stays put).
        True when the view changed."""
        new_zoom = normalize_zoom(zoom)
        if anchor_hz is None or not math.isfinite(float(anchor_hz)):
            anchor_hz, anchor_frac = self._center, 0.5
        frac = min(max(float(anchor_frac), 0.0), 1.0)
        before = (self._zoom, self._center)
        self._zoom = new_zoom
        span = self.span
        self._center = self._clamped(float(anchor_hz) + (0.5 - frac) * span)
        if (self._zoom, self._center) == before:
            return False
        self._notify()
        return True

    def zoom_steps(
        self, steps: int, anchor_hz: Optional[float] = None, anchor_frac: float = 0.5
    ) -> bool:
        """Zoom in (positive) or out by ``steps`` factors of two."""
        index = ZOOM_LEVELS.index(self._zoom) + int(steps)
        index = min(max(index, 0), len(ZOOM_LEVELS) - 1)
        return self.set_zoom(ZOOM_LEVELS[index], anchor_hz, anchor_frac)

    def follow(self, freq_hz: float, margin_frac: float = 0.02) -> bool:
        """Pan (zoomed only) so ``freq_hz`` is in sight: centered on it when
        it went out of view. True when the view moved."""
        if self._zoom <= 1:
            return False
        freq = float(freq_hz)
        if not math.isfinite(freq) or not self.in_span(freq):
            return False
        margin = margin_frac * self.span
        if self.start + margin <= freq <= self.end - margin:
            return False
        center = self._clamped(freq)
        if center == self._center:
            return False
        self._center = center
        self._notify()
        return True

    def pan_to(self, center_hz: float) -> bool:
        """Center the view on ``center_hz`` (kept inside the span)."""
        center = self._clamped(float(center_hz))
        if center == self._center:
            return False
        self._center = center
        self._notify()
        return True

    # -- mapping -----------------------------------------------------------
    def frequency_at(self, frac: float) -> float:
        """The frequency at ``frac`` of the width (0 = left edge)."""
        return self.start + float(frac) * self.span

    def fraction_of(self, freq_hz: float) -> float:
        """Where ``freq_hz`` is across the width (0 = left edge, 1 = right)."""
        return (float(freq_hz) - self.start) / self.span

    def describe(self) -> str:
        """``"×4 (600 kHz)"`` style text, for menus and messages."""
        span = self.span
        if span >= 1e6:
            width = f"{span / 1e6:.3g} MHz"
        elif span >= 1e3:
            width = f"{span / 1e3:.3g} kHz"
        else:
            width = f"{span:.0f} Hz"
        return f"×{self._zoom} ({width})"


__all__ = ["MAX_ZOOM", "PlotView", "ZOOM_LEVELS", "normalize_zoom"]
