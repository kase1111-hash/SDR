"""
Voice-like test audio with a real pitch: a glottal source whose pitch
glides and jumps from syllable to syllable (and wobbles a little), shaped
by three formants, spoken in syllables with gaps between them.

Its harmonics move all the time, as a voice's do, so it tells a steady
tone (a heterodyne whistle) from voice the way real speech would; band-
limited noise (``signal_synth.voice``) has no harmonics to mistake.
"""

from __future__ import annotations

import numpy as np

#: The vowel the voice speaks: formants (Hz, half-width Hz, weight).
OPEN_VOWEL = ((700.0, 250.0, 1.0), (1700.0, 300.0, 0.6), (2600.0, 400.0, 0.25))
#: Close vowels (ee, oo): a first formant near 300 Hz, so much of the
#: voice lies just above a radio's 300 Hz voice high-pass.
CLOSE_VOWEL = ((300.0, 100.0, 1.0), (2300.0, 200.0, 0.5))


def pitched_voice(
    seconds: float,
    rate: float = 48000.0,
    seed: int = 0,
    f0: float = 120.0,
    formants=OPEN_VOWEL,
) -> np.ndarray:
    """``seconds`` of voice-like audio at ``rate``, peak 0.6.

    Syllables last 0.12-0.35 s with 0.04-0.25 s gaps; each starts at
    0.8-1.35 times ``f0`` and glides by up to a quarter; harmonics up to
    4 kHz fall off at 6 dB per octave; ``formants`` shape them (by
    default at 700, 1700 and 2600 Hz).
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    envelope = np.zeros(n)
    pitch = np.full(n, float(f0))
    pos = 0
    while pos < n:
        length = int(rng.uniform(0.12, 0.35) * rate)
        gap = int(rng.uniform(0.04, 0.25) * rate)
        start_f = f0 * rng.uniform(0.8, 1.35)
        end_f = start_f * rng.uniform(0.75, 1.25)
        end = min(n, pos + length)
        pitch[pos:end] = np.linspace(start_f, end_f, length)[: end - pos]
        k = np.arange(end - pos)
        ramp = np.minimum(1.0, np.minimum(k, (end - pos) - k) / (0.02 * rate))
        envelope[pos:end] = ramp * rng.uniform(0.5, 1.0)
        after = min(n, end + gap)
        pitch[end:after] = end_f
        pos = after
    t = np.arange(n) / rate
    pitch = pitch * (1.0 + 0.01 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    source = np.zeros(n)
    for h in range(1, 40):
        source += np.where(h * pitch < 4000.0, np.sin(h * phase) / h, 0.0)
    spectrum = np.fft.rfft(source)
    f = np.fft.rfftfreq(n, 1.0 / rate)
    shape = np.zeros(f.size)
    for centre, half_width, weight in formants:
        shape += weight / (1.0 + ((f - centre) / half_width) ** 2)
    low = f < 150.0
    shape[low] *= (f[low] / 150.0) ** 2
    voice = np.fft.irfft(spectrum * shape, n) * envelope
    voice += 0.003 * rng.standard_normal(n)  # a little hiss
    return 0.6 * voice / np.max(np.abs(voice))


def tone(n: int, freq_hz: float, amplitude: float, rate: float = 48000.0, phase=0.3):
    """A steady sine (a heterodyne whistle)."""
    t = np.arange(n) / rate
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)


def band_power(x: np.ndarray, rate: float, low: float, high: float) -> float:
    """Power of ``x`` between ``low`` and ``high`` Hz (whole-signal FFT)."""
    spectrum = np.fft.rfft(np.asarray(x, dtype=float))
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    keep = (f >= low) & (f <= high)
    return float(np.sum(np.abs(spectrum[keep]) ** 2)) / max(len(x), 1) ** 2


def tone_level(x: np.ndarray, freq_hz: float, rate: float = 48000.0) -> float:
    """Amplitude of the ``freq_hz`` component of ``x`` (least squares)."""
    t = np.arange(len(x)) / rate
    basis = np.stack((np.cos(2 * np.pi * freq_hz * t), np.sin(2 * np.pi * freq_hz * t)))
    coef = basis @ np.asarray(x, dtype=float) * 2.0 / len(x)
    return float(np.hypot(*coef))
