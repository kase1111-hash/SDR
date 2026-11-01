"""
Transmitters for the protocols the live decoders read: POCSAG and FLEX
pagers, AX.25/APRS packet (Bell 202 AFSK on FM), ACARS (MSK on AM),
Morse code (on/off keyed carrier) and RDS (on the FM broadcast composite).

Each protocol is built in layers, the way a real transmitter builds it:

- the protocol's bits (codewords, frames, characters with parity and CRC);
- the modulating signal (NRZ levels, AFSK or MSK audio);
- the radio signal as complex baseband at any sample rate (FM or AM),
  ready to be mixed into a simulated span.

The tests use them to check the decoders end to end, and the demo device
(:mod:`sdr_module.gui.device_dialog`) precomputes its pager, APRS and ACARS
transmissions with them. NumPy only; no Qt.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .protocols import ACARSDecoder, AX25Decoder, POCSAGDecoder, bch3121_encode

__all__ = [
    "bit_levels",
    "fm_phase",
    "fm_modulate",
    "tone_audio",
    "pocsag_codeword",
    "pocsag_bits",
    "pocsag_baseband",
    "flex_codeword_bits",
    "flex_frame_bits",
    "flex_words_bits",
    "flex_baseband",
    "ax25_frame",
    "hdlc_bits",
    "afsk_audio",
    "aprs_baseband",
    "acars_block",
    "acars_bits",
    "msk_audio",
    "acars_baseband",
    "morse_units",
    "morse_keying",
    "cw_baseband",
    "rds_block",
    "rds_ps_groups",
    "rds_radiotext_groups",
    "rds_clock_group",
    "rds_ecc_group",
    "rds_bits",
    "rds_baseband",
    "fm_broadcast_composite",
]

POCSAG_DEVIATION_HZ = 4500.0
FLEX_DEVIATION_HZ = 4800.0
APRS_DEVIATION_HZ = 3000.0
ACARS_MODULATION_DEPTH = 0.7


# --------------------------------------------------------------------------- #
# Modulation
# --------------------------------------------------------------------------- #


def bit_levels(bits: Sequence[int], rate: float, baud: float) -> np.ndarray:
    """The bits as NRZ levels (+1 for 1, -1 for 0) sampled at ``rate``;
    sample k belongs to bit ``floor(k * baud / rate)`` (exact timing for
    any number of samples per bit)."""
    bits_arr = np.asarray(bits, dtype=np.int8)
    count = int(np.ceil(len(bits_arr) * float(rate) / float(baud)))
    index = np.minimum(
        (np.arange(count) * (float(baud) / float(rate))).astype(np.int64),
        len(bits_arr) - 1,
    )
    return np.where(bits_arr[index] > 0, 1.0, -1.0)


def fm_phase(deviation_hz: np.ndarray, rate: float) -> np.ndarray:
    """Phase (radians, float64) of an FM signal whose instantaneous
    frequency is ``deviation_hz`` (per sample)."""
    return (2.0 * np.pi / float(rate)) * np.cumsum(
        np.asarray(deviation_hz, dtype=np.float64)
    )


def fm_modulate(deviation_hz: np.ndarray, rate: float) -> np.ndarray:
    """Unit-amplitude FM baseband (complex64) for ``deviation_hz``."""
    return np.exp(1j * fm_phase(deviation_hz, rate)).astype(np.complex64)


def tone_audio(
    freqs: Sequence[float],
    rate: float,
    baud: float,
    amplitudes: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Phase-continuous audio sending tone ``freqs[i]`` (Hz) for bit i."""
    freq_arr = np.asarray(freqs, dtype=np.float64)
    count = int(np.ceil(len(freq_arr) * float(rate) / float(baud)))
    index = np.minimum(
        (np.arange(count) * (float(baud) / float(rate))).astype(np.int64),
        len(freq_arr) - 1,
    )
    phase = (2.0 * np.pi / float(rate)) * np.cumsum(freq_arr[index])
    audio = np.sin(phase)
    if amplitudes is not None:
        audio *= np.asarray(amplitudes, dtype=np.float64)[index]
    return audio


# --------------------------------------------------------------------------- #
# POCSAG
# --------------------------------------------------------------------------- #


def pocsag_codeword(flag: int, data20: int) -> int:
    """A 32-bit POCSAG codeword: the flag (0 address, 1 message), 20 data
    bits, the BCH(31,21) check bits and even parity."""
    codeword = bch3121_encode(((flag & 1) << 20) | (data20 & 0xFFFFF))
    return (codeword << 1) | (bin(codeword).count("1") & 1)


_NUMERIC_CODES = {char: i for i, char in enumerate(POCSAGDecoder.NUMERIC_CHARS[:16])}


def _pocsag_message_bits(text: str, numeric: bool) -> List[int]:
    """The message bits of a page: 4-bit digits or 7-bit characters, each
    least significant bit first, padded to whole 20-bit codewords."""
    bits: List[int] = []
    if numeric:
        for char in text:
            code = _NUMERIC_CODES.get(char, _NUMERIC_CODES[" "])
            bits += [(code >> i) & 1 for i in range(4)]
        pad = _NUMERIC_CODES[" "]
        while len(bits) % 20:
            bits += [(pad >> i) & 1 for i in range(4)]
    else:
        for char in text + "\x04":  # EOT ends the message
            code = ord(char) & 0x7F
            bits += [(code >> i) & 1 for i in range(7)]
        bits += [0] * (-len(bits) % 20)  # NUL fill
    return bits


def pocsag_bits(
    pages: Sequence[Tuple[int, int, Optional[str]]],
    preamble_bits: int = POCSAGDecoder.PREAMBLE_BITS,
) -> List[int]:
    """A POCSAG transmission: preamble, then batches of a sync word and 16
    codewords carrying ``pages``, each ``(address, function, text)``.

    Function 0 sends a numeric page and 1-3 an alphanumeric one (as the
    decoder reads them); text None sends a tone-only page. Each address
    codeword goes in the frame its low three bits name; message codewords
    follow it, across batches if need be; unused positions hold idle
    codewords.
    """
    words: List[int] = []  # codewords in transmission order (16 per batch)
    for address, function, text in pages:
        frame = address & 7
        while (len(words) % 16) // 2 != frame:
            words.append(POCSAGDecoder.IDLE_WORD)
        words.append(pocsag_codeword(0, ((address >> 3) << 2) | (function & 3)))
        if text is None:
            continue  # tone only
        message = _pocsag_message_bits(text, numeric=function == 0)
        for i in range(0, len(message), 20):
            data = 0
            for bit in message[i : i + 20]:
                data = (data << 1) | bit
            words.append(pocsag_codeword(1, data))
    words.append(POCSAGDecoder.IDLE_WORD)  # every page is closed
    while len(words) % 16:
        words.append(POCSAGDecoder.IDLE_WORD)

    bits = [1 - (i & 1) for i in range(preamble_bits)]
    for start in range(0, len(words), 16):
        for word in [POCSAGDecoder.SYNC_WORD] + words[start : start + 16]:
            bits += [(word >> (31 - i)) & 1 for i in range(32)]
    return bits


def pocsag_baseband(
    pages: Sequence[Tuple[int, int, Optional[str]]],
    rate: float,
    baud: int = 1200,
    deviation_hz: float = POCSAG_DEVIATION_HZ,
) -> np.ndarray:
    """A POCSAG transmission as FM baseband (complex64) at ``rate``: a 1 is
    sent as the lower frequency, a 0 as the higher one."""
    levels = bit_levels(pocsag_bits(pages), rate, baud)
    return fm_modulate(-deviation_hz * levels, rate)


# --------------------------------------------------------------------------- #
# FLEX (1600 bps, 2-level FSK)
# --------------------------------------------------------------------------- #

_FLEX_IDLE = 0x1FFFFF
_FLEX_CODE_1600_2 = 0x870C


def flex_codeword_bits(value: int) -> List[int]:
    """The 32 bits (in the order sent) of a FLEX codeword carrying the
    21-bit ``value``: its bit 0 first, then the BCH check bits and even
    parity."""
    value &= 0x1FFFFF
    data = 0
    for i in range(21):
        data = (data << 1) | ((value >> i) & 1)
    codeword = bch3121_encode(data)
    bits = [(codeword >> (30 - i)) & 1 for i in range(31)]
    return bits + [sum(bits) & 1]


def _flex_checksum(value: int) -> int:
    """``value`` (bits 4-20 filled in) with the 4-bit checksum in bits 0-3
    that makes its nibbles (and bit 20) add up to 0xF."""
    total = (value >> 20) & 1
    for shift in range(4, 20, 4):
        total += (value >> shift) & 0xF
    return (value & ~0xF) | ((0xF - total) & 0xF)


def _flex_alpha_words(text: str, number: int) -> List[int]:
    """Header word and character words of an alphanumeric page."""
    codes = [ord(c) & 0x7F for c in text]
    signature = (~sum(codes)) & 0x7F
    slots = [signature] + codes
    slots += [0x03] * (-len(slots) % 3)  # ETX fill
    words = [
        slots[i] | (slots[i + 1] << 7) | (slots[i + 2] << 14)
        for i in range(0, len(slots), 3)
    ]
    check = (~sum(words)) & 0x3FF
    # K (fragment check), C=0 (last fragment), F=3 (first fragment), N.
    header = check | (0 << 10) | (3 << 11) | ((number & 0x3F) << 13)
    return [header] + words


def flex_frame_bits(
    pages: Sequence[Tuple[int, str]],
    cycle: int = 0,
    frame: int = 0,
    bit_sync_bits: int = 32,
) -> List[int]:
    """One FLEX frame (1600 bps, 2-level) carrying alphanumeric ``pages``,
    each ``(capcode, text)`` with a short address (capcode 1-1933312).

    The sync pattern is sent with the opposite polarity to the data, as
    multimon-ng reads FLEX (see FLEXDecoder).
    """
    count = len(pages)
    first_vector = 1 + count
    values = [_FLEX_IDLE] * 88
    values[0] = _flex_checksum(first_vector << 10)  # BIW: one address block
    position = first_vector + count
    for i, (capcode, text) in enumerate(pages):
        if not 1 <= capcode <= 1933312:
            raise ValueError(f"capcode {capcode} needs a long address")
        words = _flex_alpha_words(text, i)
        if position + len(words) > 88:
            raise ValueError("pages do not fit in one frame")
        values[1 + i] = capcode + 0x8000
        values[first_vector + i] = _flex_checksum(
            (5 << 4) | (position << 7) | (len(words) << 14)
        )
        values[position : position + len(words)] = words
        position += len(words)
    return flex_words_bits(values, cycle, frame, bit_sync_bits)


def flex_words_bits(
    values: Sequence[int],
    cycle: int = 0,
    frame: int = 0,
    bit_sync_bits: int = 32,
) -> List[int]:
    """One FLEX frame (1600 bps, 2-level) whose phase holds the 88 21-bit
    codeword ``values`` as given (block information word first): the
    header, then the 11 interleaved blocks."""
    values = list(values)[:88] + [_FLEX_IDLE] * max(0, 88 - len(values))
    sync = (_FLEX_CODE_1600_2 << 48) | (0xA6C6AAAA << 16) | (_FLEX_CODE_1600_2 ^ 0xFFFF)
    bits = [1 - (i & 1) for i in range(bit_sync_bits)]
    bits += [((sync >> (63 - i)) & 1) ^ 1 for i in range(64)]
    bits += [((0x5939 >> (15 - i)) & 1) ^ 1 for i in range(16)]
    fiw = _flex_checksum(((cycle & 0xF) << 4) | ((frame & 0x7F) << 8))
    bits += flex_codeword_bits(fiw)
    bits += [1 - (i & 1) for i in range(40)]  # second sync (25 ms)
    codewords = [flex_codeword_bits(v) for v in values]
    for block in range(11):
        for i in range(32):
            for j in range(8):
                bits.append(codewords[block * 8 + j][i])
    return bits


def flex_baseband(
    pages: Sequence[Tuple[int, str]],
    rate: float,
    cycle: int = 0,
    frame: int = 0,
    deviation_hz: float = FLEX_DEVIATION_HZ,
    bit_sync_bits: int = 32,
) -> np.ndarray:
    """A FLEX frame as FM baseband (complex64): a 1 is the higher frequency."""
    bits = flex_frame_bits(pages, cycle, frame, bit_sync_bits)
    return fm_modulate(deviation_hz * bit_levels(bits, rate, 1600), rate)


# --------------------------------------------------------------------------- #
# AX.25 / APRS
# --------------------------------------------------------------------------- #


def _ax25_address(callsign: str, last: bool) -> bytes:
    call, _, ssid_text = callsign.upper().partition("-")
    ssid = int(ssid_text) if ssid_text else 0
    out = bytes((ord(c) << 1) & 0xFE for c in call.ljust(6)[:6])
    return out + bytes([0x60 | ((ssid & 0x0F) << 1) | (1 if last else 0)])


def ax25_frame(
    source: str, destination: str, info: str, path: Sequence[str] = ()
) -> bytes:
    """An AX.25 UI frame (control 0x03, PID 0xF0) with its FCS."""
    calls = [destination, source] + list(path)
    address = b"".join(
        _ax25_address(call, i == len(calls) - 1) for i, call in enumerate(calls)
    )
    payload = address + bytes([0x03, 0xF0]) + info.encode("ascii", "replace")
    fcs = AX25Decoder(1200 * 8)._compute_crc(payload)
    return payload + bytes([fcs & 0xFF, fcs >> 8])


def hdlc_bits(frame: bytes, flags: int = 32, tail_flags: int = 3) -> List[int]:
    """The line bits of an HDLC frame: opening flags, the frame (bytes least
    significant bit first, a 0 stuffed after five 1s), closing flags, all
    NRZI coded (a 0 toggles the line, a 1 keeps it)."""
    data: List[int] = []
    ones = 0
    for byte in frame:
        for i in range(8):
            bit = (byte >> i) & 1
            data.append(bit)
            ones = ones + 1 if bit else 0
            if ones == 5:
                data.append(0)
                ones = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    raw = flag * flags + data + flag * tail_flags
    line, level = [], 0
    for bit in raw:
        if bit == 0:
            level ^= 1
        line.append(level)
    return line


def afsk_audio(
    line_bits: Sequence[int],
    rate: float,
    baud: float = 1200.0,
    mark_hz: float = 1200.0,
    space_hz: float = 2200.0,
    twist_db: float = 0.0,
) -> np.ndarray:
    """Bell 202 AFSK audio for the line bits (1 mark, 0 space); a positive
    ``twist_db`` sends the space tone that much louder (pre-emphasis)."""
    bits = np.asarray(line_bits)
    freqs = np.where(bits > 0, mark_hz, space_hz)
    gain = 10.0 ** (twist_db / 20.0)
    amps = np.where(bits > 0, 1.0, gain) / max(1.0, gain)
    return tone_audio(freqs, rate, baud, amps)


def aprs_baseband(
    source: str,
    destination: str,
    info: str,
    rate: float,
    path: Sequence[str] = (),
    deviation_hz: float = APRS_DEVIATION_HZ,
    flags: int = 32,
    twist_db: float = 0.0,
) -> np.ndarray:
    """An APRS packet as narrowband FM baseband (complex64): AFSK1200 audio
    at ``deviation_hz`` peak deviation."""
    line = hdlc_bits(ax25_frame(source, destination, info, path), flags)
    audio = afsk_audio(line, rate, twist_db=twist_db)
    return fm_modulate(deviation_hz * audio, rate)


# --------------------------------------------------------------------------- #
# ACARS
# --------------------------------------------------------------------------- #


def acars_block(
    registration: str,
    label: str,
    text: str = "",
    mode: str = "2",
    block_id: str = "2",
    ack: Optional[str] = None,
    message_number: str = "",
    flight_id: str = "",
    end: int = ACARSDecoder.ETX,
) -> bytes:
    """The characters of an ACARS block from mode to ETX (or ETB), odd
    parity added. A downlink (``block_id`` a digit) puts the message number
    and flight ID before the text."""
    chars = [ord(mode[:1] or "2")]
    chars += [ord(c) for c in registration.rjust(7, ".")[-7:]]
    chars.append(ACARSDecoder.NAK if ack is None else ord(ack[:1]))
    chars += [ord(c) for c in label.ljust(2)[:2]]
    chars.append(ord(block_id[:1] or "2"))
    body = text
    if block_id.isdigit() and (message_number or flight_id):
        body = message_number.ljust(4)[:4] + flight_id.ljust(6)[:6] + text
    if body:
        chars.append(ACARSDecoder.STX)
        chars += [ord(c) for c in body]
    chars.append(end)
    return bytes(ACARSDecoder.with_parity(c) for c in chars)


def acars_bits(block: bytes, prekey_bits: int = 128) -> List[int]:
    """The data bits of an ACARS transmission: pre-key (ones), '+' '*'
    (bit sync), SYN SYN, SOH, the block, its CRC and DEL; each character
    least significant bit first."""
    crc = ACARSDecoder(48000)._compute_crc(block)
    sent = bytes(ACARSDecoder.with_parity(c) for c in b"+*\x16\x16\x01")
    sent += block + bytes([crc & 0xFF, crc >> 8, ACARSDecoder.DEL])
    bits = [1] * prekey_bits
    for byte in sent:
        bits += [(byte >> i) & 1 for i in range(8)]
    return bits


def msk_audio(
    data_bits: Sequence[int], rate: float, baud: float = 2400.0
) -> np.ndarray:
    """ACARS MSK audio: for each bit a 2400 Hz tone when it equals the
    previous bit and 1200 Hz when it changes (the bit before the first is
    taken as 1, the pre-key's)."""
    freqs = []
    last = 1
    for bit in data_bits:
        freqs.append(2400.0 if bit == last else 1200.0)
        last = bit
    return tone_audio(freqs, rate, baud)


def acars_baseband(
    block: bytes,
    rate: float,
    depth: float = ACARS_MODULATION_DEPTH,
    prekey_bits: int = 128,
) -> np.ndarray:
    """An ACARS transmission as AM baseband (complex64): carrier plus the
    MSK audio at ``depth``."""
    audio = msk_audio(acars_bits(block, prekey_bits), rate)
    return (1.0 + depth * audio).astype(np.complex64)


# --------------------------------------------------------------------------- #
# Morse code (CW)
# --------------------------------------------------------------------------- #


def morse_units(text: str) -> List[int]:
    """The keying of ``text`` in dit units (1: key down): a dit is one unit
    and a dash three, with one unit between the elements of a letter, three
    between letters and seven between words. Characters without a Morse
    code are left out."""
    from .morse import MORSE_CODE

    codes = {char: code for code, char in MORSE_CODE.items() if len(char) == 1}
    units: List[int] = []
    words = [w for w in str(text).upper().split() if any(c in codes for c in w)]
    for wi, word in enumerate(words):
        letters = [codes[c] for c in word if c in codes]
        for li, code in enumerate(letters):
            for ei, element in enumerate(code):
                units += [1] * (1 if element == "." else 3)
                if ei < len(code) - 1:
                    units.append(0)
            if li < len(letters) - 1:
                units += [0] * 3
        if wi < len(words) - 1:
            units += [0] * 7
    return units


def morse_keying(
    text: str,
    rate: float,
    wpm: float = 20.0,
    edge_s: float = 0.005,
    lead_s: float = 0.5,
    tail_s: float = 0.5,
) -> np.ndarray:
    """The key envelope (0 to 1, float64) of ``text`` sent at ``wpm`` words
    per minute, sampled at ``rate``: smooth edges about ``edge_s`` long (no
    key clicks; two centred moving averages, so the timing is kept) and
    silence before and after."""
    rate = float(rate)
    unit = 1.2 / float(wpm) * rate  # samples per dit
    units = morse_units(text)
    count = int(round(len(units) * unit))
    index = np.minimum((np.arange(count) / unit).astype(np.int64), len(units) - 1)
    key = np.asarray(units, dtype=np.float64)[index] if units else np.zeros(0)
    half = int(round(edge_s * rate / 4.0))
    if half > 0 and len(key):
        length = 2 * half + 1
        for _ in range(2):
            padded = np.concatenate((np.zeros(half), key, np.zeros(half)))
            sums = np.concatenate(([0.0], np.cumsum(padded)))
            key = (sums[length:] - sums[:-length]) / length
    lead = np.zeros(int(round(lead_s * rate)))
    tail = np.zeros(int(round(tail_s * rate)))
    return np.concatenate((lead, key, tail))


def cw_baseband(
    text: str,
    rate: float,
    wpm: float = 20.0,
    offset_hz: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A CW transmission of ``text`` as complex baseband (complex64): a
    carrier ``offset_hz`` from 0 Hz keyed by :func:`morse_keying`."""
    key = morse_keying(text, rate, wpm)
    t = np.arange(len(key)) / float(rate)
    carrier = np.exp(2j * np.pi * float(offset_hz) * t)
    return (float(amplitude) * key * carrier).astype(np.complex64)


# --------------------------------------------------------------------------- #
# RDS (on the FM broadcast composite)
# --------------------------------------------------------------------------- #

RDS_BAUD = 1187.5
RDS_CARRIER_HZ = 57e3
RDS_PILOT_HZ = 19e3
#: Group type codes (block B bits 15-11).
_RDS_GROUP = {"0A": 0x00, "0B": 0x01, "1A": 0x02, "2A": 0x04, "2B": 0x05, "4A": 0x08}


def rds_block(data: int, offset: str) -> int:
    """A 26-bit RDS block: the 16 data bits and the check word with offset
    word ``offset`` ("A", "B", "C", "C'" or "D") added."""
    from .protocols import RDSDecoder, _rds_poly_mod

    data &= 0xFFFF
    return (data << 10) | (_rds_poly_mod(data << 10) ^ RDSDecoder.OFFSETS[offset])


def _rds_b(group: str, pty: int, tp: bool, low5: int) -> int:
    """Block B: group type and version, TP, PTY and the low five bits."""
    return (
        (_RDS_GROUP[group] << 11)
        | (int(bool(tp)) << 10)
        | ((int(pty) & 0x1F) << 5)
        | (low5 & 0x1F)
    )


def _rds_chars(text: str, length: int) -> List[int]:
    """``text`` as RDS bytes (ASCII; others as spaces), padded to ``length``."""
    codes = [ord(c) if 32 <= ord(c) < 127 else 0x20 for c in str(text)[:length]]
    return codes + [0x20] * (length - len(codes))


def rds_ps_groups(
    pi: int,
    ps: str,
    pty: int = 0,
    tp: bool = False,
    ta: bool = False,
    music: bool = True,
    af_mhz: Sequence[float] = (),
) -> List[Tuple[int, int, int, int]]:
    """The four type 0A groups that send the station name ``ps`` (8
    characters), with the alternative frequencies ``af_mhz`` in block C."""
    chars = _rds_chars(ps, 8)
    afs = [int(round((f - 87.5) * 10)) for f in af_mhz if 87.6 <= f <= 107.9]
    codes = ([224 + len(afs)] + afs) if afs else [224]
    codes += [205] * (8 - len(codes) % 8)  # fillers
    groups = []
    for segment in range(4):
        b = _rds_b("0A", pty, tp, (int(ta) << 4) | (int(music) << 3) | segment)
        c = (codes[2 * segment % len(codes)] << 8) | codes[
            (2 * segment + 1) % len(codes)
        ]
        d = (chars[2 * segment] << 8) | chars[2 * segment + 1]
        groups.append((pi, b, c, d))
    return groups


def rds_radiotext_groups(
    pi: int, text: str, pty: int = 0, tp: bool = False, ab: int = 0
) -> List[Tuple[int, int, int, int]]:
    """The type 2A groups that send RadioText ``text`` (up to 64
    characters; a shorter text ends with a carriage return)."""
    codes = _rds_chars(text, min(len(text), 64))
    if len(codes) < 64:
        codes.append(0x0D)
    codes += [0x20] * (-len(codes) % 4)
    groups = []
    for segment in range(len(codes) // 4):
        b = _rds_b("2A", pty, tp, ((ab & 1) << 4) | segment)
        part = codes[4 * segment : 4 * segment + 4]
        groups.append((pi, b, (part[0] << 8) | part[1], (part[2] << 8) | part[3]))
    return groups


def rds_clock_group(
    pi: int,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    offset_half_hours: int = 0,
    pty: int = 0,
    tp: bool = False,
) -> Tuple[int, int, int, int]:
    """A type 4A group: UTC date and time and the local offset (half hours)."""
    import datetime as _dt

    mjd = (_dt.date(year, month, day) - _dt.date(1858, 11, 17)).days
    b = _rds_b("4A", pty, tp, (mjd >> 15) & 0x03)
    c = ((mjd & 0x7FFF) << 1) | ((hour >> 4) & 0x01)
    sign = 1 if offset_half_hours < 0 else 0
    d = ((hour & 0x0F) << 12) | ((minute & 0x3F) << 6) | (sign << 5)
    d |= abs(int(offset_half_hours)) & 0x1F
    return (pi, b, c, d)


def rds_ecc_group(pi: int, ecc: int, pty: int = 0) -> Tuple[int, int, int, int]:
    """A type 1A group carrying the extended country code (variant 0)."""
    return (pi, _rds_b("1A", pty, False, 0), ecc & 0xFF, 0)


def rds_bits(groups: Sequence[Tuple[int, int, int, int]]) -> List[int]:
    """The data bits of ``groups`` (block C of a version B group gets the
    C' offset)."""
    bits: List[int] = []
    for a, b, c, d in groups:
        third = "C'" if (b >> 11) & 1 else "C"
        for data, offset in ((a, "A"), (b, "B"), (c, third), (d, "D")):
            block = rds_block(data, offset)
            bits.extend((block >> (25 - i)) & 1 for i in range(26))
    return bits


def _rds_pulse(t: np.ndarray) -> np.ndarray:
    """Impulse response of the RDS data-shaping filter, cos(pi f td / 4)
    up to 2/td (td: one bit), for times ``t`` in seconds."""
    a = 1.0 / (4.0 * RDS_BAUD)
    return np.sinc(t / a + 0.5) + np.sinc(t / a - 0.5)


def rds_baseband(
    bits: Sequence[int], rate: float, periodic: bool = False, span_bits: int = 8
) -> np.ndarray:
    """The RDS data signal (before the 57 kHz carrier) for data ``bits``
    at ``rate``: differentially encoded, biphase (an impulse pair per bit)
    and shaped by the RDS filter, peaking at about 1 (float64).

    With ``periodic`` the bits repeat forever: the result is one seamless
    period of ``len(bits) / 1187.5`` seconds (the demo device loops it).
    """
    rate = float(rate)
    count = len(bits)
    if not count:
        return np.zeros(0)
    level = np.empty(2 * count)
    last = 0
    for k, bit in enumerate(bits):
        last ^= 1 if bit else 0  # differential coding
        level[2 * k] = 1.0 if last else -1.0
    level[1::2] = -level[0::2]  # biphase: the second half is the opposite
    half = 1.0 / (2.0 * RDS_BAUD)  # seconds per half bit
    samples = int(round(count / RDS_BAUD * rate))
    # A lone bit's peak: the scale for a peak of about 1.
    probe = np.linspace(-half, 2 * half, 301)
    peak = np.max(
        np.abs(_rds_pulse(probe - 0.5 * half) - _rds_pulse(probe - 1.5 * half))
    )
    if periodic and samples > 4 * count + 1:
        # One period exactly, by its Fourier series: the impulses' spectrum
        # (a DFT of their levels) times the filter's, cos(pi f td / 4) up to
        # 2/td (2 * count harmonics of the period).
        period = count / RDS_BAUD
        k = np.arange(2 * count + 1)
        impulses = np.fft.fft(level)[k % (2 * count)] * np.exp(
            -1j * np.pi * k / (2 * count)
        )
        a = 1.0 / (4.0 * RDS_BAUD)
        spectrum = np.zeros(samples, dtype=np.complex128)
        spectrum[k] = 2 * a * np.cos(np.pi * a * k / period) * impulses / period
        spectrum[samples - k[1:]] = np.conj(spectrum[k[1:]])
        return np.real(np.fft.ifft(spectrum)) * samples / peak
    span = 2 * span_bits
    out = np.empty(samples)
    chunk = 65536
    for start in range(0, samples, chunk):
        t = np.arange(start, min(start + chunk, samples)) / rate
        centre = np.floor(t / half).astype(np.int64)
        j = centre[:, None] + np.arange(-span, span + 1)[None, :]
        dt = t[:, None] - (j + 0.5) * half
        if periodic:
            weights = level[np.mod(j, 2 * count)]
        else:
            inside = (j >= 0) & (j < 2 * count)
            weights = np.where(inside, level[np.clip(j, 0, 2 * count - 1)], 0.0)
        out[start : start + len(t)] = np.sum(weights * _rds_pulse(dt), axis=1)
    return out / peak


def fm_broadcast_composite(
    rate: float,
    seconds: float,
    rds_bits_: Optional[Sequence[int]] = None,
    pilot: bool = True,
    audio_hz: Sequence[float] = (440.0, 1300.0),
    audio_level: float = 0.7,
    pilot_level: float = 0.09,
    rds_level: float = 0.04,
    rds_phase: float = 0.0,
    deviation_hz: float = 75e3,
) -> np.ndarray:
    """An FM broadcast multiplex (the instantaneous frequency, Hz): program
    audio, the 19 kHz stereo pilot and the RDS signal on a 57 kHz carrier
    locked to the pilot's third harmonic (``rds_phase`` from it: 0 in
    phase, pi/2 in quadrature). Levels are fractions of the peak
    deviation."""
    rate = float(rate)
    n = int(round(seconds * rate))
    t = np.arange(n) / rate
    composite = np.zeros(n)
    for i, freq in enumerate(audio_hz):
        composite += audio_level / len(audio_hz) * np.sin(2 * np.pi * freq * t + i)
    theta = 2 * np.pi * RDS_PILOT_HZ * t
    if pilot:
        composite += pilot_level * np.cos(theta)
    if rds_bits_ is not None and len(rds_bits_):
        data = rds_baseband(rds_bits_, rate, periodic=True)
        reps = -(-n // len(data))
        data = np.tile(data, reps)[:n]
        composite += rds_level * data * np.cos(3 * theta + rds_phase)
    return deviation_hz * composite
