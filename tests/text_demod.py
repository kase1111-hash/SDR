"""Independent demodulators for the text-mode tests (not collected by pytest).

They read what the encoders made the way a receiver would, from the I/Q
alone: an FM discriminator and a UART-style bit clock for RTTY and ASCII
FSK, symbol timing from the signal's own amplitude and a phase comparison
for PSK31, and the repo's streaming Morse decoder (dsp/morse.py) for CW.
The code tables here are typed from the standards (ITA2 with US-TTY
figures, G3PLX Varicode spot checks), not imported from the encoders.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# ITA2, indexed by the code with its first-sent bit as bit 0.
ITA2_LETTERS = (
    None, "E", "\n", "A", " ", "S", "I", "U",
    "\r", "D", "R", "J", "N", "F", "C", "K",
    "T", "Z", "L", "W", "H", "Y", "P", "Q",
    "O", "B", "G", "FIGS", "M", "X", "V", "LTRS",
)  # fmt: skip
ITA2_FIGURES_US = (
    None, "3", "\n", "-", " ", "\a", "8", "7",
    "\r", "$", "4", "'", ",", "!", ":", "(",
    "5", '"', ")", "2", "#", "6", "0", "1",
    "9", "?", "&", "FIGS", ".", "/", ";", "LTRS",
)  # fmt: skip


def mix(iq: np.ndarray, rate: float, freq_hz: float) -> np.ndarray:
    """``iq`` moved down by ``freq_hz`` (that frequency to 0 Hz)."""
    t = np.arange(len(iq)) / float(rate)
    return np.asarray(iq, np.complex128) * np.exp(-2j * np.pi * freq_hz * t)


def analytic(audio: np.ndarray) -> np.ndarray:
    """The analytic signal of real ``audio`` (positive frequencies only)."""
    n = len(audio)
    spectrum = np.fft.fft(np.asarray(audio, np.float64))
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def _smooth(x: np.ndarray, length: int) -> np.ndarray:
    length = max(1, int(length))
    kernel = np.ones(length) / length
    return np.convolve(x, kernel, mode="same")


def instantaneous_frequency(baseband: np.ndarray, rate: float) -> np.ndarray:
    """Frequency (Hz) of each sample of complex ``baseband`` (one fewer)."""
    x = np.asarray(baseband, np.complex128)
    return np.angle(x[1:] * np.conj(x[:-1])) * float(rate) / (2 * np.pi)


def uart_frames(
    mark: np.ndarray,
    rate: float,
    baud: float,
    data_bits: int,
    stop_bits: float = 1.0,
) -> List[int]:
    """Characters of an asynchronous serial stream (``mark``: True while
    the line is at mark, one value per sample): wait for mark (idle), then
    each fall to space is a start bit; the data bits are read at their
    middles, first bit first; a frame whose start bit isn't space or whose
    stop bit isn't mark is a framing error (-1)."""
    bit = float(rate) / float(baud)
    out: List[int] = []
    n = len(mark)
    i = 0
    # Idle first (a stream may start straight into a start bit).
    while i < n and not mark[i]:
        i += 1
    while i < n - 1:
        if mark[i] and not mark[i + 1]:
            start = i + 1
            if start + int((1 + data_bits + 0.5) * bit) >= n:
                break
            if mark[int(start + 0.5 * bit)]:
                i += 1  # a glitch, not a start bit
                continue
            value = 0
            for k in range(data_bits):
                if mark[int(start + (1.5 + k) * bit)]:
                    value |= 1 << k
            stop_at = int(start + (1 + data_bits + 0.5) * bit)
            out.append(value if mark[stop_at] else -1)
            # The next start bit can come once the stop bit is (mostly) sent.
            i = int(start + (1 + data_bits + min(stop_bits, 1.0) * 0.5) * bit)
        else:
            i += 1
    return out


def fsk_mark(iq: np.ndarray, rate: float, center_hz: float, baud: float) -> np.ndarray:
    """Mark (True) / space per sample of FSK centred on ``center_hz``, mark
    above the centre (discriminator, smoothed over a quarter bit)."""
    freq = instantaneous_frequency(mix(iq, rate, center_hz), rate)
    freq = _smooth(freq, rate / baud / 4.0)
    return np.append(freq > 0, freq[-1] > 0)


def ita2_text(codes: Sequence[int], figures=ITA2_FIGURES_US) -> str:
    """Text of ITA2 ``codes``, following LTRS and FIGS (no unshift on
    space)."""
    in_figures = False
    out = []
    for code in codes:
        if code < 0:
            out.append("~")  # framing error
            continue
        char = (figures if in_figures else ITA2_LETTERS)[code]
        if char == "LTRS":
            in_figures = False
        elif char == "FIGS":
            in_figures = True
        elif char is not None:
            out.append(char)
    return "".join(out)


def ita2_text_usos(codes: Sequence[int], figures=ITA2_FIGURES_US) -> str:
    """As :func:`ita2_text`, for a receiver that unshifts on space (USOS)."""
    in_figures = False
    out = []
    for code in codes:
        if code < 0:
            out.append("~")
            continue
        char = (figures if in_figures else ITA2_LETTERS)[code]
        if char == "LTRS":
            in_figures = False
        elif char == "FIGS":
            in_figures = True
        elif char is not None:
            out.append(char)
            if char == " ":
                in_figures = False
    return "".join(out)


def rtty_decode(
    iq: np.ndarray,
    rate: float,
    center_hz: float = 0.0,
    baud: float = 45.45,
    usos: bool = False,
) -> str:
    """Text of RTTY I/Q (5 data bits, mark above ``center_hz``)."""
    mark = fsk_mark(iq, rate, center_hz, baud)
    codes = uart_frames(mark, rate, baud, 5, 1.5)
    return (ita2_text_usos if usos else ita2_text)(codes)


def ascii_decode(
    iq: np.ndarray, rate: float, center_hz: float = 0.0, baud: float = 300.0
) -> bytes:
    """Bytes of 8N1 FSK I/Q (mark above ``center_hz``)."""
    mark = fsk_mark(iq, rate, center_hz, baud)
    return bytes(c & 0xFF for c in uart_frames(mark, rate, baud, 8, 1.0) if c >= 0)


def psk31_bits(
    iq: np.ndarray, rate: float, center_hz: float = 0.0, baud: float = 31.25
) -> List[int]:
    """PSK31 bits (1: no phase change, 0: a reversal) of I/Q with the
    carrier at ``center_hz``. The symbol clock is found from the signal: the
    amplitude peaks mid-symbol, between the zero crossings of reversals."""
    x = mix(iq, rate, center_hz)
    sps = float(rate) / float(baud)
    x = np.convolve(x, np.ones(max(1, int(sps / 8))) / max(1, int(sps / 8)), "same")
    mag = np.abs(x)
    best, best_score = 0.0, -1.0
    for offset in np.arange(0.0, sps, sps / 32.0):
        idx = (offset + np.arange(int((len(x) - offset) / sps)) * sps).astype(int)
        score = float(np.mean(mag[idx])) if len(idx) else 0.0
        if score > best_score:
            best, best_score = offset, score
    idx = (best + np.arange(int((len(x) - best) / sps)) * sps).astype(int)
    samples = x[idx]
    strong = np.abs(samples) > 0.25 * float(np.max(np.abs(samples)) or 1.0)
    bits = []
    for k in range(1, len(samples)):
        if not (strong[k] and strong[k - 1]):
            continue
        bits.append(1 if (samples[k] * np.conj(samples[k - 1])).real > 0 else 0)
    return bits


def varicode_text(bits: Sequence[int], table: Dict[str, str]) -> str:
    """Characters of Varicode ``bits`` (``table``: code -> character);
    characters are separated by two zeros."""
    out = []
    code = ""
    zeros = 0
    for bit in bits:
        if bit:
            if zeros == 1:
                code += "0"
            zeros = 0
            code += "1"
        else:
            zeros += 1
            if zeros == 2:
                if code:
                    out.append(table.get(code, "~"))
                code = ""
    return "".join(out)


def psk31_decode(
    iq: np.ndarray, rate: float, table: Dict[str, str], center_hz: float = 0.0
) -> str:
    """Text of PSK31 I/Q, ``table``: code -> character."""
    return varicode_text(psk31_bits(iq, rate, center_hz), table)


def cw_decode(iq: np.ndarray, rate: float, center_hz: float = 0.0) -> str:
    """Text of a keyed carrier (CW) through the repo's streaming Morse
    decoder (dsp/morse.py), fed in 33 ms blocks like the live decoder."""
    from sdr_module.dsp.morse import KeyingDetector, MorseDecoder

    x = mix(iq, rate, center_hz)
    env = _smooth(np.abs(x), rate * 0.002)
    # Work at about 8 kHz (the decoder reads about 1 kHz of decisions).
    step = max(1, int(rate // 8000))
    env = env[::step]
    work = float(rate) / step
    keyer = KeyingDetector(work)
    decoder = MorseDecoder(work)
    # Silence before and after, as on the air between transmissions.
    pad = np.zeros(int(0.5 * work))
    env = np.concatenate((pad, env, pad, pad, pad, pad, pad, pad))
    block = int(0.033 * work)
    lines = []
    for start in range(0, len(env), block):
        keying = keyer.process(env[start : start + block])
        lines += decoder.process(keying)
    lines += decoder.flush()
    return " ".join(line.text for line in lines)


def tone_frequencies(signal: np.ndarray, rate: float) -> List[float]:
    """The two tones (Hz, lowest first) of FSK ``signal`` (real audio or
    complex I/Q): the two clusters of its instantaneous frequency (exact
    for FSK, unlike spectrum peaks, which the keying sidebands shift)."""
    x = np.asarray(signal)
    if not np.iscomplexobj(x):
        x = analytic(x)
    strong = np.abs(x) > 0.5 * float(np.max(np.abs(x)) or 1.0)
    freq = instantaneous_frequency(x, rate)[strong[1:] & strong[:-1]]
    low, high = np.percentile(freq, 5), np.percentile(freq, 95)
    for _ in range(20):
        split = (low + high) / 2.0
        low = float(np.median(freq[freq < split]))
        high = float(np.median(freq[freq >= split]))
    return [low, high]


def occupied_bandwidth(iq: np.ndarray, rate: float, fraction: float = 0.99) -> float:
    """The bandwidth (Hz) holding ``fraction`` of the power of ``iq``,
    centred on its spectrum's centre of power (as the ITU defines it, with
    0.5% outside on each side)."""
    n = len(iq)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / rate))
    cumulative = np.cumsum(spectrum) / np.sum(spectrum)
    tail = (1.0 - fraction) / 2.0
    low = freqs[int(np.searchsorted(cumulative, tail))]
    high = freqs[int(np.searchsorted(cumulative, 1.0 - tail))]
    return float(high - low)


def first(seq: Sequence, default: Optional[object] = None):
    return seq[0] if len(seq) else default
