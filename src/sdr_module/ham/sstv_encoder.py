"""
SSTV encoder: an image to the audio of an SSTV transmission.

Every mode in :data:`sdr_module.ham.sstv.SSTV_MODES` can be sent, with the
standard calibration header and VIS code in front, using the same line
layouts the decoder reads (:func:`sdr_module.ham.sstv.line_templates`).

The tones are phase-continuous and every segment starts at its exact time
(boundaries are rounded once, from the cumulative time), so long images
keep their timing.

Usage:
    from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card

    card = sstv_test_card(320, 240)
    audio = encode_sstv(card, "Robot 36", sample_rate=48000)

Also used to test the decoder and by the demo device's SSTV signal.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .sstv import (
    SSTVMode,
    SSTVModeSpec,
    find_mode,
    line_templates,
    preamble_segments,
    rgb_to_ycc,
    value_to_freq,
    vis_segments,
)

ModeLike = Union[str, int, SSTVMode, SSTVModeSpec]


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """``image`` as ``(height, width, 3)`` floats (nearest-neighbour scaled;
    greyscale expanded to RGB)."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {img.shape}")
    img = img[..., :3].astype(np.float64)
    h, w = img.shape[:2]
    if (h, w) != (height, width):
        rows = np.minimum((np.arange(height) * h) // height, h - 1)
        cols = np.minimum((np.arange(width) * w) // width, w - 1)
        img = img[rows][:, cols]
    return np.clip(img, 0.0, 255.0)


def _channels(image: np.ndarray, spec: SSTVModeSpec) -> Dict[str, np.ndarray]:
    """Per-channel pixel values (rows x width) the mode sends."""
    rgb = _fit_image(image, spec.width, spec.height)
    out: Dict[str, np.ndarray] = {
        "R": rgb[..., 0],
        "G": rgb[..., 1],
        "B": rgb[..., 2],
    }
    if spec.color_format == "YUV":
        y, cr, cb = rgb_to_ycc(rgb)
        out.update({"Y": y, "V": cr, "U": cb})
    return out


def segment_plan(
    image: np.ndarray,
    mode: ModeLike,
    *,
    vis: bool = True,
    lines: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(frequencies Hz, durations ms)`` of every tone of a transmission.

    Args:
        image: RGB image (scaled to the mode's size).
        mode: Mode name, VIS code, :class:`SSTVMode` or spec.
        vis: Send the calibration header and VIS code first.
        lines: Send only the first ``lines`` sync periods (default: all).
    """
    spec = find_mode(mode)
    chans = _channels(image, spec)
    freqs: List[np.ndarray] = []
    durs: List[np.ndarray] = []

    def tone(freq: float, ms: float) -> None:
        freqs.append(np.array([freq]))
        durs.append(np.array([ms]))

    if vis:
        for _kind, ms, freq in vis_segments(spec.vis_code):
            tone(freq, ms)
    for _kind, ms, freq in preamble_segments(spec):
        tone(freq, ms)

    templates = line_templates(spec)
    periods = spec.sync_periods if lines is None else min(lines, spec.sync_periods)
    width = spec.width
    family = spec.family
    for k in range(periods):
        for kind, ms, freq in templates[k % len(templates)]:
            if kind in ("sync", "tone", "parity"):
                tone(freq, ms)
                continue
            if family == "pd":
                rows = (2 * k, min(2 * k + 1, spec.height - 1))
                if kind == "Y0":
                    values = chans["Y"][rows[0]]
                elif kind == "Y1":
                    values = chans["Y"][rows[1]]
                else:  # R-Y / B-Y shared by the two lines
                    values = 0.5 * (chans[kind][rows[0]] + chans[kind][rows[1]])
            elif family == "robot36" and kind in ("U", "V"):
                # Each line sends its own Y and one chroma channel, R-Y on
                # even lines and B-Y on odd ones, averaged over the pair.
                first = k - (k % 2)
                second = min(first + 1, spec.height - 1)
                values = 0.5 * (chans[kind][first] + chans[kind][second])
            else:
                values = chans[kind][k]
            freqs.append(value_to_freq(values))
            durs.append(np.full(width, ms / width))
    return np.concatenate(freqs), np.concatenate(durs)


def synthesize_tones(
    freqs: np.ndarray,
    durations_ms: np.ndarray,
    sample_rate: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Phase-continuous audio (float32) for consecutive tones."""
    fs = float(sample_rate)
    bounds = np.rint(
        np.cumsum(np.asarray(durations_ms, dtype=np.float64)) * fs / 1000.0
    )
    counts = np.diff(np.concatenate(([0.0], bounds))).astype(np.int64)
    counts = np.maximum(counts, 0)
    inst = np.repeat(np.asarray(freqs, dtype=np.float64), counts)
    phase = np.cumsum(inst) * (2.0 * np.pi / fs)
    return (amplitude * np.sin(phase)).astype(np.float32)


def encode_sstv(
    image: np.ndarray,
    mode: ModeLike,
    sample_rate: float = 48000.0,
    *,
    vis: bool = True,
    lines: Optional[int] = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Audio (mono float32) of an SSTV transmission of ``image``.

    Args:
        image: RGB array ``(H, W, 3)`` (scaled to the mode's size if needed).
        mode: Mode name ("Martin 1"), VIS code, :class:`SSTVMode` or spec.
        sample_rate: Audio sample rate in Hz.
        vis: Start with the calibration header and VIS code.
        lines: Send only the first ``lines`` sync periods (default: all).
        amplitude: Peak amplitude of the audio.
    """
    freqs, durs = segment_plan(image, mode, vis=vis, lines=lines)
    return synthesize_tones(freqs, durs, sample_rate, amplitude)


# --------------------------------------------------------------------------- #
# Test card
# --------------------------------------------------------------------------- #

# 5x7 bitmap font for the test card's caption.
_FONT: Dict[str, Tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    " ": ("00000",) * 7,
}

# The eight classic colour bars.
_BARS = (
    (255, 255, 255),
    (255, 255, 0),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (0, 0, 0),
)


def _draw_text(image: np.ndarray, text: str, top: int, scale: int, color) -> None:
    """Draw ``text`` centred horizontally at row ``top`` (5x7 font)."""
    text = text.upper()
    advance = 6 * scale
    width = len(text) * advance - scale
    left = max(0, (image.shape[1] - width) // 2)
    for i, char in enumerate(text):
        glyph = _FONT.get(char, _FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                y0 = top + gy * scale
                x0 = left + i * advance + gx * scale
                image[y0 : y0 + scale, x0 : x0 + scale] = color


def sstv_test_card(
    width: int = 320, height: int = 256, caption: str = "SDR MODULE"
) -> np.ndarray:
    """A colourful RGB test card: colour bars, a caption, a grey ramp, a
    checkerboard and a colour gradient. Big shapes and sharp edges show
    how well an image came through."""
    w, h = int(width), int(height)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    bars_end = int(round(h * 0.34))
    for i, colour in enumerate(_BARS):
        img[:bars_end, i * w // 8 : (i + 1) * w // 8] = colour
    caption_end = int(round(h * 0.56))
    img[bars_end:caption_end] = (24, 40, 96)
    scale = max(
        1, min((caption_end - bars_end - 4) // 8, w // (6 * max(len(caption), 1)))
    )
    text_h = 7 * scale
    _draw_text(
        img,
        caption,
        bars_end + (caption_end - bars_end - text_h) // 2,
        scale,
        (255, 220, 64),
    )
    ramp_end = int(round(h * 0.70))
    ramp = np.linspace(0, 255, w).astype(np.uint8)
    img[caption_end:ramp_end] = ramp[None, :, None]
    check_end = int(round(h * 0.85))
    cell = max(4, w // 16)
    yy, xx = np.mgrid[caption_end:check_end, 0:w]
    checker = ((((yy - ramp_end) // cell) + (xx // cell)) % 2 * 255).astype(np.uint8)
    img[ramp_end:check_end] = checker[ramp_end - caption_end :, :, None]
    rows = h - check_end
    if rows > 0:
        x = np.linspace(0.0, 1.0, w)[None, :]
        y = np.linspace(0.0, 1.0, rows)[:, None]
        img[check_end:, :, 0] = (255 * x * np.ones_like(y)).astype(np.uint8)
        img[check_end:, :, 1] = (255 * (1 - x) * np.ones_like(y)).astype(np.uint8)
        img[check_end:, :, 2] = (255 * y * np.ones_like(x)).astype(np.uint8)
    return img


__all__ = [
    "encode_sstv",
    "segment_plan",
    "synthesize_tones",
    "sstv_test_card",
]
