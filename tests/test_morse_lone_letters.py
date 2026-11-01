"""K-D2: a lone E or T keyed by an operator is shown; static stays out.

A word that is one E or one T is what a crash of static keys, so the
decoder dropped it next to words of several letters and in a line of
nothing else: "TU E E" (a sign-off) read "TU" and "T T T" read nothing.
Now such a word shows when the speed is established by the code around it
(the recent marks are dits and dashes of one speed), its mark is a dit or
a dash at that speed and a steady carrier (the envelope of a crash of
static is not), and a word gap at that speed separates it from the code
before or after it, as an operator keys it. The crash models are those of
tests/test_morse_decoder.py.
"""

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from tests import test_morse_decoder as models
from tests.test_morse_decoder import MESSAGE, _decode

RATE = 2000.0


def _channel(parts, wpm, snr_db=12.0, seed=1, crashes=False):
    """CW at ``wpm`` sending ``parts`` (texts, or pauses in seconds),
    ``snr_db`` above band noise (in 500 Hz), with static crashes in the
    pauses when ``crashes``."""
    pieces = [np.zeros(int(1.0 * RATE), np.complex64)]
    quiet = [np.ones(len(pieces[0]), bool)]
    for part in parts:
        if isinstance(part, str):
            x = enc.cw_baseband(part, RATE, wpm, offset_hz=30.0)
            quiet.append(np.zeros(len(x), bool))
        else:
            x = np.zeros(int(part * RATE), np.complex64)
            quiet.append(np.ones(len(x), bool))
        pieces.append(x)
    pieces.append(np.zeros(int(3.0 * RATE), np.complex64))
    quiet.append(np.ones(len(pieces[-1]), bool))
    signal, quiet = np.concatenate(pieces), np.concatenate(quiet)
    n = int(len(signal) / RATE * RATE)
    level = np.ones(n)
    if crashes:
        # Crashes only where nothing is sent, and not within 0.55 s of it
        # (closer, they are part of a word at 12 WPM: "ECQ").
        level = models.TestSpeeds._crashes(n / RATE, seed)[:n]
        near = np.convolve(~quiet[:n], np.ones(int(1.1 * RATE)), "same") > 0
        level[near] = 1.0
    noise = models.TestSpeeds._shaped_noise(n / RATE, level, seed)
    return np.sqrt(10 ** (snr_db / 10)) * signal[:n] + noise


@pytest.mark.parametrize("wpm", [12, 20, 30])
def test_a_sign_off_keeps_its_es(wpm):
    text, _morse = _decode(_channel([MESSAGE + " TU E E"], wpm), RATE, block=66)
    assert text == [MESSAGE + " TU E E"]


@pytest.mark.parametrize("wpm", [12, 20, 30])
def test_ts_a_word_apart_after_a_message(wpm):
    text, _morse = _decode(_channel([MESSAGE + " T T T"], wpm), RATE, block=66)
    assert text == [MESSAGE + " T T T"]


@pytest.mark.parametrize("wpm", [12, 20, 30])
@pytest.mark.parametrize("line", ["T T T", "E E", "TU E E", "TT"])
def test_a_line_of_lone_letters_once_the_speed_is_known(wpm, line):
    # The station sent a message first: its speed is known. (A line of one
    # E or T, a pause on both sides, is timed like a crash of static: it
    # is still not shown.)
    x = _channel([MESSAGE, 3.5, line], wpm, snr_db=20.0, seed=2)
    text, _morse = _decode(x, RATE, block=66)
    assert text == [MESSAGE, line]


@pytest.mark.parametrize("wpm", [12, 20, 30])
def test_blips_of_noise_in_the_pause_before_them(wpm):
    # Weaker (12 dB): the noise keys blips shorter than a dit in the pause,
    # which are not letters and don't cut the pause short.
    x = _channel([MESSAGE, 4.0, "E E"], wpm, seed=2)
    text, _morse = _decode(x, RATE, block=66)
    assert text[-1] == "E E"


@pytest.mark.parametrize("wpm", [12, 20, 30])
def test_lone_letters_in_the_middle_and_at_the_start(wpm):
    for line in ("E E " + MESSAGE, MESSAGE.replace("DE", "DE T")):
        text, _morse = _decode(_channel([line], wpm, seed=3), RATE, block=66)
        assert text == [line], line


@pytest.mark.parametrize("wpm", [12, 20, 30])
@pytest.mark.parametrize("seed", [2, 3, 4, 5, 6, 7, 8, 9])
def test_static_around_a_sign_off_adds_no_lone_letters(seed, wpm):
    # Crashes of static before and after the message, as
    # tests/test_morse_decoder.py has them around MESSAGE, some a word gap
    # from it: the operator's Es show, the crashes' marks (noise, not a
    # steady carrier) don't. (A crash that keys twice can still make an I
    # or an N now and then, as before.)
    x = _channel([2.0, MESSAGE + " TU E E", 3.0], wpm, seed=seed, crashes=True)
    text, _morse = _decode(x, RATE, block=66)
    words = " ".join(text).split()
    end = words.index("73")
    assert words[end : end + 4] == ["73", "TU", "E", "E"], (seed, text)
    # The only lone Es and Ts are the operator's.
    assert [w for w in words if w in ("E", "T")] == ["E", "E"], (seed, text)


def test_the_cw_demodulator_reads_them_too():
    # CWDemodulator.decode_morse takes the steadiness of the keying its
    # detect_keying just gave.
    from sdr_module.dsp.demodulators import CWDemodulator

    x = _channel([MESSAGE + " TU E E"], 20)
    demod = CWDemodulator(RATE, wpm=20)
    text = ""
    for start in range(0, len(x), 66):
        text += demod.decode_morse(demod.detect_keying(np.abs(x[start : start + 66])))
    text += demod.flush()
    assert text.strip() == MESSAGE + " TU E E"
    # Keying from elsewhere has no steadiness: a lone E stays out, as before.
    other = CWDemodulator(RATE, wpm=20)
    keying = other.detect_keying(np.abs(x)).copy()
    assert (other.decode_morse(keying) + other.flush()).strip() == MESSAGE + " TU"


@pytest.mark.parametrize("wpm", [12, 25])
def test_the_cw_live_decoder_reads_them_too(wpm):
    # The Decoder tab's 'CW (Morse)' decoder (live, playback and Decode
    # Recording) keys the Morse decoder with its own KeyingDetector: it
    # must pass the steadiness on, or a lone E or T stays out there. A
    # carrier 300 Hz from the LO, 20 dB above the noise in 500 Hz, in
    # display-frame blocks.
    from sdr_module.dsp.live_decoders import create_live_decoder

    rate, offset, lines = 48e3, 300.0, []
    rng = np.random.default_rng(wpm)
    parts = [np.zeros(int(0.5 * rate), np.complex64)]
    for item in (MESSAGE + " TU E E", 3.5, "T T T", 3.5):
        if isinstance(item, str):
            parts.append(enc.cw_baseband(item, rate, wpm, offset, amplitude=0.1))
        else:
            parts.append(np.zeros(int(item * rate), np.complex64))
    x = np.concatenate(parts)
    sigma = 0.1 * np.sqrt(rate / 500.0 / 10 ** (20.0 / 10) / 2)
    noise = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    x = (x + sigma * noise).astype(np.complex64)
    decoder = create_live_decoder("CW (Morse)", rate, offset)
    block = int(rate / 30)
    for start in range(0, len(x), block):
        lines += decoder.process(x[start : start + block])
    lines += decoder.flush()
    words = " ".join(m.text for m in lines).split()
    assert words == (MESSAGE + " TU E E T T T").split()


def test_an_empty_block_has_no_steadiness_left_over():
    # The steadiness is per sample of the last block, an empty one too
    # (decode_morse of an empty keying raised: one value per decision).
    from sdr_module.dsp.demodulators import CWDemodulator
    from sdr_module.dsp.morse import KeyingDetector

    keyer = KeyingDetector(RATE)
    keyer.process(np.ones(500))
    assert len(keyer.process(np.zeros(0))) == len(keyer.steady) == 0
    demod = CWDemodulator(RATE)
    demod.decode_morse(demod.detect_keying(np.ones(500)))
    assert demod.decode_morse(demod.detect_keying(np.zeros(0))) == ""
