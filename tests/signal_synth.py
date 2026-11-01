"""Synthetic test signals for the signal-analysis tests (T10).

Complex baseband generators at a given sample rate, each returning the
clean signal (unit-ish power while on) so a test can add noise at a chosen
SNR with :func:`add_noise`. Used by tests/test_signal_identify.py and
tests/test_signal_bursts.py.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MORSE = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "K": "-.-",
    "M": "--",
    "N": "-.",
    "O": "---",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _lowpass(x: np.ndarray, cutoff: float, rate: float, taps: int = 255) -> np.ndarray:
    t = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff / rate * np.sinc(2 * cutoff / rate * t) * np.hamming(taps)
    h /= np.sum(h)
    return np.convolve(x, h, mode="same")


def voice(n: int, rate: float, seed: int = 0, band=(300.0, 3000.0)) -> np.ndarray:
    """Voice-like audio: band-limited noise with a syllable envelope, peak
    about 1 (real)."""
    rng = _rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    spec[(freqs < band[0]) | (freqs > band[1])] = 0
    audio = np.fft.irfft(spec, n)
    # Syllables: a slow envelope (4 Hz), never quite silent.
    t = np.arange(n) / rate
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.7 * t + rng.uniform(0, 6)) ** 2
    audio *= env
    return audio / (np.max(np.abs(audio)) + 1e-12)


def tones(n: int, rate: float, freqs=(820.0, 1650.0), amps=(0.6, 0.35)) -> np.ndarray:
    t = np.arange(n) / rate
    out = np.zeros(n)
    for f, a in zip(freqs, amps, strict=True):
        out += a * np.sin(2 * np.pi * f * t)
    return out / (np.max(np.abs(out)) + 1e-12)


def carrier(n: int, rate: float, offset: float = 120.0) -> np.ndarray:
    t = np.arange(n) / rate
    return np.exp(2j * np.pi * offset * t)


def fm(
    n: int, rate: float, deviation: float, audio: np.ndarray, offset: float = 0.0
) -> np.ndarray:
    """FM with ``deviation`` Hz at the audio's peak (audio peak 1)."""
    t = np.arange(n) / rate
    phase = 2 * np.pi * np.cumsum(deviation * audio) / rate
    return np.exp(1j * (phase + 2 * np.pi * offset * t))


def am(n: int, rate: float, depth: float, audio: np.ndarray, offset=0.0) -> np.ndarray:
    t = np.arange(n) / rate
    return (1.0 + depth * audio) * np.exp(2j * np.pi * offset * t) / (1 + depth / 2)


def ssb(n: int, rate: float, audio: np.ndarray, upper: bool = True) -> np.ndarray:
    """Single sideband of real ``audio`` (analytic signal)."""
    spec = np.fft.fft(audio)
    freqs = np.fft.fftfreq(n, 1.0 / rate)
    keep = freqs > 0 if upper else freqs < 0
    spec = np.where(keep, 2 * spec, 0)
    out = np.fft.ifft(spec)
    return out / (np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-12)


def morse_keying(
    n: int, rate: float, text: str = "VVV DE SDR", dit_s: float = 0.06, edge_s=0.004
) -> np.ndarray:
    """0/1 keying envelope (raised edges) of ``text`` in Morse, repeated."""
    units: List[int] = []
    for word in text.split():
        for letter in word:
            for symbol in MORSE[letter]:
                units += [1] * (1 if symbol == "." else 3) + [0]
            units += [0, 0]
        units += [0] * 4
    per = int(round(dit_s * rate))
    keyed = np.repeat(np.asarray(units, dtype=float), per)
    keyed = np.resize(keyed, n)
    width = max(1, int(edge_s * rate))
    kernel = np.ones(width) / width
    return np.convolve(keyed, kernel, mode="same")


def cw(n: int, rate: float, offset: float = 80.0, **kw) -> np.ndarray:
    return carrier(n, rate, offset) * morse_keying(n, rate, **kw)


def fsk(
    n: int, rate: float, baud: float, shift: float, seed: int = 0, offset: float = 0.0
) -> np.ndarray:
    """Continuous-phase 2-FSK: +/- ``shift`` Hz."""
    rng = _rng(seed)
    count = int(np.ceil(n * baud / rate)) + 1
    bits = rng.integers(0, 2, count) * 2 - 1
    idx = (np.arange(n) * baud / rate).astype(int)
    freq = shift * bits[idx] + offset
    return np.exp(2j * np.pi * np.cumsum(freq) / rate)


def rrc_taps(sps: int, beta: float = 0.35, span: int = 8) -> np.ndarray:
    t = (np.arange(-span * sps, span * sps + 1)) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            ) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h**2))


def psk(
    n: int, rate: float, baud: float, order: int = 2, seed: int = 0, offset=0.0
) -> np.ndarray:
    """RRC-filtered M-PSK."""
    rng = _rng(seed)
    sps = max(2, int(round(rate / baud)))
    count = n // sps + 20
    symbols = np.exp(1j * (2 * np.pi * rng.integers(0, order, count) / order))
    if order == 4:
        symbols *= np.exp(1j * np.pi / 4)
    up = np.zeros(count * sps, dtype=complex)
    up[::sps] = symbols
    out = np.convolve(up, rrc_taps(sps), mode="same")[:n]
    t = np.arange(n) / rate
    out = out * np.exp(2j * np.pi * offset * t)
    return out / (np.sqrt(np.mean(np.abs(out) ** 2)) + 1e-12)


# --------------------------------------------------------------------------- #
# OOK (rtl_433-style devices) and Mode S pulses
# --------------------------------------------------------------------------- #


def ev1527_pulses(
    code: int, bits: int = 24, short_us: float = 350.0
) -> List[Tuple[float, float]]:
    """(pulse_us, gap_us) pairs of an EV1527/PT2262-style PWM frame: a 1 is
    a long pulse and a short gap, a 0 a short pulse and a long gap, then a
    sync (short pulse, 31 units gap)."""
    out = []
    for i in range(bits - 1, -1, -1):
        if (code >> i) & 1:
            out.append((3 * short_us, short_us))
        else:
            out.append((short_us, 3 * short_us))
    out.append((short_us, 31 * short_us))
    return out


def ppm_sensor_pulses(
    code: int, bits: int = 36, pulse_us=500.0, zero_us=1000.0, one_us=2000.0
) -> List[Tuple[float, float]]:
    """(pulse, gap) pairs of a PPM weather-sensor frame (fixed pulses, the
    gap carries the bit), ending in a long gap."""
    out = []
    for i in range(bits - 1, -1, -1):
        out.append((pulse_us, one_us if (code >> i) & 1 else zero_us))
    out.append((pulse_us, 4000.0))
    return out


def keying_from_pulses(
    pulses: Sequence[Tuple[float, float]],
    rate: float,
    repeats: int = 1,
    lead_us: float = 0.0,
    edge_us: float = 15.0,
) -> np.ndarray:
    """0/1 envelope (soft edges) of (pulse_us, gap_us) pairs, repeated."""
    parts = [np.zeros(int(round(lead_us * 1e-6 * rate)))]
    for _ in range(repeats):
        for on_us, off_us in pulses:
            parts.append(np.ones(int(round(on_us * 1e-6 * rate))))
            parts.append(np.zeros(int(round(off_us * 1e-6 * rate))))
    env = np.concatenate(parts)
    width = max(1, int(round(edge_us * 1e-6 * rate)))
    if width > 1:
        env = np.convolve(env, np.ones(width) / width, mode="same")
    return env


def ook_packet(
    n: int,
    rate: float,
    pulses: Sequence[Tuple[float, float]],
    repeats: int = 4,
    start_s: float = 0.05,
    offset: float = 0.0,
) -> np.ndarray:
    """A carrier keyed by ``pulses`` (repeated), starting at ``start_s``,
    inside ``n`` samples of silence."""
    env = keying_from_pulses(pulses, rate, repeats, lead_us=start_s * 1e6)
    out = np.zeros(n)
    m = min(n, len(env))
    out[:m] = env[:m]
    return carrier(n, rate, offset) * out


def mode_s_frame(rate: float, data: bytes) -> np.ndarray:
    """Envelope of a Mode S reply: the 8 us preamble (pulses at 0, 1, 3.5,
    4.5 us) and PPM data at 1 Mbit/s (0.5 us pulses)."""
    chips = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # 0.5 us chips
    for byte in data:
        for i in range(7, -1, -1):
            chips += [1, 0] if (byte >> i) & 1 else [0, 1]
    chip = 0.5e-6 * rate
    n = int(np.ceil(len(chips) * chip)) + 2
    env = np.zeros(n)
    for k, on in enumerate(chips):
        if on:
            a, b = int(round(k * chip)), int(round((k + 1) * chip))
            env[a : max(b, a + 1)] = 1.0
    return env


def mode_s_replies(
    n: int, rate: float, count: int = 12, seed: int = 0, offset: float = 0.0
) -> np.ndarray:
    """``count`` Mode S replies (random data) at random times."""
    rng = _rng(seed)
    env = np.zeros(n)
    for _ in range(count):
        frame = mode_s_frame(rate, bytes(rng.integers(0, 256, 14).tolist()))
        start = int(rng.integers(0, max(1, n - len(frame))))
        env[start : start + len(frame)] = np.maximum(
            env[start : start + len(frame)], frame
        )
    return carrier(n, rate, offset) * env


# --------------------------------------------------------------------------- #
# Noise
# --------------------------------------------------------------------------- #


def noise(n: int, power: float = 1.0, seed: int = 99) -> np.ndarray:
    rng = _rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(power / 2)


def add_noise(
    signal: np.ndarray,
    rate: float,
    snr_db: float,
    occupied_hz: float,
    seed: int = 7,
    on_power: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """``signal`` plus white noise so that the signal's power (while on:
    ``on_power``, else its mean power) is ``snr_db`` above the noise in
    ``occupied_hz``. Returns ``(samples, noise density per Hz)``."""
    power = on_power if on_power is not None else float(np.mean(np.abs(signal) ** 2))
    n0 = power / (10 ** (snr_db / 10.0) * occupied_hz)
    total = n0 * rate
    out = signal + noise(len(signal), total, seed)
    return out.astype(np.complex64), n0


def channel_filter(x: np.ndarray, rate: float, half_width: float) -> np.ndarray:
    """A receiver's channel filter: low-pass +/- ``half_width``, deep
    stopband (the stopband a classifier must not take for the noise)."""
    re = _lowpass(x.real, half_width, rate, 511)
    im = _lowpass(x.imag, half_width, rate, 511)
    return (re + 1j * im).astype(np.complex64)
