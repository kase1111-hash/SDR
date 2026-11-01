"""A fake ``python_hackrf`` backend whose TX callback runs in its own thread,
the way libhackrf's USB thread pulls samples, plus helpers that measure what
went "on air" (shared by the transmit tests; not collected by pytest). It
receives too: its RX callback gets a transfer at a time from a thread of
its own, paced at the sample rate, holding ``rx_stations`` around the
frequency tuned (for scans with a real ``HackRFDevice``); without
``rx_stations`` it receives nothing, as before.

The fake device only answers to the real python_hackrf device method names
(see ``tests/test_hackrf_driver.py``), so a driver call the real library
doesn't have fails loudly instead of passing against a permissive mock.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Dict, List, Optional

import numpy as np

BUFFER_LENGTH = 262144  # bytes per USB transfer (131072 I/Q samples)
_FILTERS = [
    1750000,
    2500000,
    3500000,
    5000000,
    5500000,
    6000000,
    7000000,
    8000000,
    9000000,
    10000000,
    12000000,
    14000000,
    15000000,
    20000000,
    24000000,
    28000000,
]
_ALLOWED = {
    "pyhackrf_close",
    "pyhackrf_serialno_read",
    "pyhackrf_set_freq",
    "pyhackrf_set_sample_rate",
    "pyhackrf_set_baseband_filter_bandwidth",
    "pyhackrf_set_lna_gain",
    "pyhackrf_set_vga_gain",
    "pyhackrf_set_txvga_gain",
    "pyhackrf_set_amp_enable",
    "pyhackrf_set_antenna_enable",
    "pyhackrf_is_streaming",
    "set_rx_callback",
    "set_tx_callback",
    "pyhackrf_start_rx",
    "pyhackrf_stop_rx",
    "pyhackrf_start_tx",
    "pyhackrf_stop_tx",
}


class _CInt:
    def __init__(self, value: int = 0) -> None:
        self.value = value


class ThreadedFakeHackRF:
    """Fake device: TX transfers are pulled by a thread of its own.

    ``pace``: take each transfer in real time at the sample rate (like the
    hardware) instead of as fast as possible. ``stall``: never pull samples
    (a hung USB link). ``tx_log`` records (what, time, rate, freq, filter).
    ``rx_stations``: carriers ``{freq: dBFS}`` heard while receiving (each
    RX transfer is made on the frequency tuned when it is handed over; {}:
    noise only). None (the default): receiving delivers nothing.
    """

    def __init__(
        self,
        pace: bool = False,
        stall: bool = False,
        rx_stations: Optional[Dict[float, float]] = None,
    ):
        self.pace = pace
        self.stall = stall
        self.receives = rx_stations is not None
        self.rx_stations = dict(rx_stations or {})
        #: RX transfers handed to the callback.
        self.rx_transfers = 0
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_stop = threading.Event()
        self.freq: Optional[int] = None
        self.sample_rate: Optional[float] = None
        self.bandwidth: Optional[int] = None
        self.txvga: Optional[int] = None
        self.amp: Optional[bool] = None
        self.antenna: Optional[bool] = None
        self.lna: Optional[int] = None
        self.vga: Optional[int] = None
        self._rx_cb = None
        self._tx_cb = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.chunks: List[np.ndarray] = []
        self.tx_log: List[tuple] = []
        self.closed = False

    def __getattr__(self, name):
        if name not in _ALLOWED:
            raise AttributeError(f"python_hackrf device has no attribute {name!r}")
        raise AttributeError(name)

    def pyhackrf_serialno_read(self):
        return "0000000000fake77"

    def pyhackrf_set_freq(self, freq):
        self.freq = freq

    def pyhackrf_set_sample_rate(self, rate):
        self.sample_rate = rate

    def pyhackrf_set_baseband_filter_bandwidth(self, bw):
        self.bandwidth = bw

    def pyhackrf_set_lna_gain(self, gain):
        self.lna = gain

    def pyhackrf_set_vga_gain(self, gain):
        self.vga = gain

    def pyhackrf_set_txvga_gain(self, gain):
        self.txvga = gain

    def pyhackrf_set_amp_enable(self, on):
        self.amp = bool(on)

    def pyhackrf_set_antenna_enable(self, on):
        self.antenna = bool(on)

    def set_rx_callback(self, fn):
        self._rx_cb = fn

    def set_tx_callback(self, fn):
        self._tx_cb = fn

    def pyhackrf_start_rx(self):
        if not self.receives:
            return
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_pump, daemon=True)
        self._rx_thread.start()

    def _rx_pump(self):
        """Hand the RX callback a transfer at a time in real time, as
        libhackrf's USB thread does."""
        rng = np.random.default_rng(7)
        count = BUFFER_LENGTH // 2
        due = time.monotonic()
        while not self._rx_stop.is_set():
            rate = float(self.sample_rate or 2e6)
            due = max(due + count / rate, time.monotonic())
            if self._rx_stop.wait(max(0.0, due - time.monotonic())):
                break
            buf = self._rx_transfer(rng, count, rate)
            rc = self._rx_cb(self, buf, BUFFER_LENGTH, BUFFER_LENGTH)
            self.rx_transfers += 1
            if rc != 0:
                break

    def _rx_transfer(self, rng, count: int, rate: float) -> np.ndarray:
        """One transfer of int8 I/Q around the frequency tuned now."""
        t = np.arange(count) / rate
        iq = (rng.standard_normal(count) + 1j * rng.standard_normal(count)) * 0.015
        centre = float(self.freq or 0)
        for freq, level in self.rx_stations.items():
            offset = freq - centre
            if abs(offset) < 0.5 * rate:
                iq = iq + 10 ** (level / 20) * np.exp(2j * np.pi * offset * t)
        raw = np.empty(2 * count, dtype=np.float32)
        raw[0::2], raw[1::2] = iq.real, iq.imag
        return np.clip(np.round(raw * 127.0), -127, 127).astype(np.int8)

    def pyhackrf_stop_rx(self):
        self._rx_stop.set()
        thread = self._rx_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(5)

    def pyhackrf_start_tx(self):
        self.chunks = []
        self.tx_log.append(
            ("start_tx", time.monotonic(), self.sample_rate, self.freq, self.bandwidth)
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        due = time.monotonic()
        while not self._stop.is_set():
            if self.stall:
                time.sleep(0.01)
                continue
            buf = np.zeros(BUFFER_LENGTH, dtype=np.int8)
            length = _CInt(0)
            rc = self._tx_cb(self, buf, BUFFER_LENGTH, length)
            n = int(length.value)
            if n > 0:
                self.chunks.append(buf[:n].copy())
                if self.pace:
                    due += (n / 2) / float(self.sample_rate or 2e6)
                    while not self._stop.is_set() and time.monotonic() < due:
                        time.sleep(0.002)
            if rc != 0:
                self.tx_log.append(("callback_end", time.monotonic(), rc))
                break
        self.tx_log.append(("pump_end", time.monotonic()))

    def pyhackrf_stop_tx(self):
        self.tx_log.append(("stop_tx", time.monotonic()))
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(5)

    def pyhackrf_close(self):
        self.closed = True

    # --- test helpers (not part of the real API) ---
    def tx_iq(self, decimate: int = 1) -> np.ndarray:
        """Everything transmitted, as complex samples (+/-1 full scale).

        ``decimate``: average each run of that many samples (a cheap
        low-pass), so a long transmission can be measured quickly at a
        lower rate (the rate divided by ``decimate``).
        """
        if not self.chunks:
            return np.zeros(0, np.complex64)
        factor = max(1, int(decimate))
        parts = []
        leftover = np.zeros(0, np.complex64)
        for chunk in self.chunks:
            raw = chunk.astype(np.float32)
            iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64) / np.float32(127.0)
            if factor > 1:
                iq = np.concatenate((leftover, iq))
                usable = len(iq) - len(iq) % factor
                leftover = iq[usable:]
                iq = iq[:usable].reshape(-1, factor).mean(axis=1)
            parts.append(iq.astype(np.complex64))
        return np.concatenate(parts)

    def tx_count(self) -> int:
        """How many I/Q samples were transmitted."""
        return sum(len(chunk) for chunk in self.chunks) // 2

    def start_rate(self) -> Optional[float]:
        """The sample rate the last transmission started at."""
        starts = [entry for entry in self.tx_log if entry[0] == "start_tx"]
        return starts[-1][2] if starts else None


def install_fake_backend(monkeypatch, device: ThreadedFakeHackRF):
    """Make ``import python_hackrf`` give a backend that opens ``device``."""
    import sdr_module.devices.hackrf as hackrf_mod

    module = types.ModuleType("python_hackrf")
    backend = types.ModuleType("python_hackrf.pyhackrf")

    class _DevList:
        device_count = 1
        serial_numbers = ["0000000000fake77"]

    def compute_filter(bw):
        best = _FILTERS[0]
        for value in _FILTERS:
            if value <= bw:
                best = value
        return best

    backend.pyhackrf_init = lambda: None
    backend.pyhackrf_exit = lambda: None
    backend.pyhackrf_device_list = lambda: _DevList()
    backend.pyhackrf_open = lambda: device
    backend.pyhackrf_device_list_open = lambda dl, i: device
    backend.pyhackrf_compute_baseband_filter_bw = compute_filter
    module.pyhackrf = backend
    monkeypatch.setitem(sys.modules, "python_hackrf", module)
    monkeypatch.setitem(sys.modules, "python_hackrf.pyhackrf", backend)
    monkeypatch.setattr(hackrf_mod, "_initialized", False)
    return backend


def open_hackrf(monkeypatch, rate: float = 2.4e6, **fake_args):
    """A HackRFDevice opened on a threaded fake, at ``rate`` (as Device >
    Connect leaves it, filter 0.75 x rate). Returns (device, fake)."""
    from sdr_module.devices.hackrf import HackRFDevice

    fake = ThreadedFakeHackRF(**fake_args)
    install_fake_backend(monkeypatch, fake)
    device = HackRFDevice()
    assert device.open(0)
    device.set_sample_rate(rate)
    device.set_bandwidth(0.75 * rate)
    return device, fake


# --------------------------------------------------------------------------- #
# What went on air
# --------------------------------------------------------------------------- #

_MORSE = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    "-..-.": "/",
}


def _runs(on: np.ndarray) -> List[tuple]:
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    bounds = np.concatenate(([0], edges, [len(on)]))
    return [
        (bool(on[a]), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def decode_keying(on: np.ndarray, unit_s: float) -> Dict[str, float]:
    """Morse text, dit length (s), WPM and keyed span (s) of an on/off
    keying sampled every ``unit_s`` seconds."""
    runs = _runs(on)
    lengths = [n * unit_s for key, n in runs if key]
    if not lengths:
        return {"text": "", "wpm": 0.0, "dit_s": 0.0, "keyed_s": 0.0}
    shortest = min(lengths)
    dit = float(np.median([d for d in lengths if d < 1.8 * shortest]))
    text, symbol = "", ""
    for key, n in runs:
        d = n * unit_s
        if key:
            symbol += "." if d < 2 * dit else "-"
        elif d >= 5 * dit:
            text += (_MORSE.get(symbol, "?") if symbol else "") + " "
            symbol = ""
        elif d >= 2 * dit:
            text += _MORSE.get(symbol, "?") if symbol else ""
            symbol = ""
    if symbol:
        text += _MORSE.get(symbol, "?")
    first = next(i for i, (key, _n) in enumerate(runs) if key)
    last = max(i for i, (key, _n) in enumerate(runs) if key)
    keyed = sum(n for _key, n in runs[first : last + 1]) * unit_s
    return {"text": text.strip(), "wpm": 1.2 / dit, "dit_s": dit, "keyed_s": keyed}


def analyse_cw(iq: np.ndarray, rate: float) -> Dict[str, float]:
    """A keyed-carrier (A1A) ID: its keying from the envelope."""
    win = max(1, int(rate * 0.001))
    count = len(iq) // win
    env = np.abs(iq[: count * win]).reshape(count, win).mean(axis=1)
    on = env > 0.5 * float(env.max() or 1.0)
    out = decode_keying(on, win / rate)
    out["carrier_off_level"] = float(np.percentile(env[~on], 50)) if (~on).any() else 0
    out["peak"] = float(env.max())
    return out


def analyse_mcw(iq: np.ndarray, rate: float) -> Dict[str, float]:
    """An FM tone ID (MCW): the tone, its peak deviation and its keying."""
    phase = np.unwrap(np.angle(iq.astype(np.complex128)))
    freq = np.diff(phase) * rate / (2 * np.pi)
    win = max(1, int(rate * 0.001))
    count = len(freq) // win
    rms = np.sqrt((freq[: count * win].reshape(count, win) ** 2).mean(axis=1))
    on = rms > 0.3 * float(rms.max() or 1.0)
    out = decode_keying(on, win / rate)
    keyed = np.repeat(on, win)[: len(freq)]
    segment = freq[: len(keyed)][keyed]
    if len(segment) > 4096:
        size = 1 << int(np.log2(len(segment)))
        segment = segment[:size]
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(size)))
        bins = np.fft.rfftfreq(size, 1 / rate)
        out["tone_hz"] = float(bins[np.argmax(spectrum[1:]) + 1])
    # Peak deviation from the phase over 20 us (the int8 samples make the
    # per-sample discriminator noisy).
    lag = max(1, int(rate / 50e3))
    smooth = (phase[lag:] - phase[:-lag]) * rate / (2 * np.pi * lag)
    out["peak_dev_hz"] = float(np.percentile(np.abs(smooth), 99.9))
    out["envelope_min"] = float(np.min(np.abs(iq)))
    return out
