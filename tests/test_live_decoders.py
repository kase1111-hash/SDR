"""Live protocol decoders on real signals (FG-B01, TOOLS-01).

Every protocol is sent the way a real transmitter sends it (the encoders in
``sdr_module.dsp.protocol_encoders``: BCH codewords, HDLC frames, Bell 202
AFSK on FM, MSK on AM, ...), placed in a wide I/Q span like the receiver's
and decoded by ``create_live_decoder`` in GUI-sized blocks: clean, with
noise, next to a much stronger station and away from the LO.

Before these decoders the window decoded nothing but ADS-B: it fed the raw
span to an FM discriminator, AFSK and AM-MSK went through that same
discriminator, every slicer dropped a bit and restarted its phase at each
block, and POCSAG auto-baud locked to 2400 baud on any noise.
"""

import functools

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import (
    LIVE_DECODERS,
    ACARSLiveDecoder,
    ADSBLiveDecoder,
    AX25LiveDecoder,
    CWLiveDecoder,
    FLEXLiveDecoder,
    LiveDecoderStatus,
    POCSAGLiveDecoder,
    RDSLiveDecoder,
    create_live_decoder,
    describe_message,
    live_decoder_class,
)
from sdr_module.dsp.protocols import (
    ACARSDecoder,
    ADSBDecoder,
    ADSBMessage,
    AX25Decoder,
    BitSlicer,
    FLEXDecoder,
    POCSAGDecoder,
    bch3121_correct,
    bch3121_encode,
)

GUI_BLOCK_S = 0.034  # one display frame at 30 Hz

APRS_INFO = "!4903.50N/07201.75W>Test station"
ACARS_BLOCK = enc.acars_block(
    "N601SD", "H1", "TEST MESSAGE", message_number="M01A", flight_id="SD0601"
)

# name -> (baseband maker, expected (address, content) rows, channel width
# the SNR is measured in)
CASES = {
    "POCSAG": (
        lambda rate: enc.pocsag_baseband(
            [(1234567, 3, "TEST PAGE ONE"), (1100004, 0, "555-0123")], rate
        ),
        [("1234567 (F3)", "TEST PAGE ONE"), ("1100004 (F0)", "555-0123")],
        25e3,
    ),
    "FLEX": (
        lambda rate: enc.flex_baseband([(1234567, "FLEX TEST PAGE")], rate),
        [("1234567 (A, 1600 bps)", "FLEX TEST PAGE")],
        25e3,
    ),
    # (Rows read the APRS position and symbol, and the ACARS flight.)
    "AX.25/APRS": (
        lambda rate: enc.aprs_baseband(
            "N0CALL-9", "APZ001", APRS_INFO, rate, ["WIDE1-1"]
        ),
        [("N0CALL-9>APZ001", "49.0583, -72.0292  ·  Car  ·  Test station")],
        25e3,
    ),
    "ACARS": (
        lambda rate: enc.acars_baseband(ACARS_BLOCK, rate),
        [("N601SD", "Flight SD0601  ·  H1  ·  TEST MESSAGE")],
        10e3,
    ),
}
PROTOCOLS = list(CASES)


@functools.lru_cache(maxsize=16)
def _baseband(name, rate):
    return CASES[name][0](rate)


def _scene(
    name,
    rate,
    offset=0.0,
    snr_db=None,
    stations=(),
    tuning_error=0.0,
    seed=0,
    lead_s=0.2,
    tail_s=0.25,
):
    """One transmission of ``name`` in a span sampled at ``rate``: ``offset``
    Hz from the centre, with white noise for ``snr_db`` in the protocol's
    channel and ``stations`` ((offset Hz, dB relative), ...) of broadcast
    FM."""
    rng = np.random.default_rng(seed)
    bb = _baseband(name, rate)
    lead, tail = int(lead_s * rate), int(tail_s * rate)
    x = np.zeros(lead + len(bb) + tail, dtype=np.complex64)
    x[lead : lead + len(bb)] = bb
    t = np.arange(len(x)) / rate
    if offset or tuning_error:
        x *= np.exp(2j * np.pi * (offset + tuning_error) * t).astype(np.complex64)
    for where, db in stations:
        phase = 2 * np.pi * where * t + 50.0 * np.sin(2 * np.pi * 1e3 * t)
        x += (10 ** (db / 20) * np.exp(1j * phase)).astype(np.complex64)
    if snr_db is not None:
        width = CASES[name][2]
        sigma = np.sqrt((rate / width) / 10 ** (snr_db / 10) / 2)
        noise = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
        x += (sigma * noise).astype(np.complex64)
    return x


def _decode(decoder, x, block):
    messages = []
    for start in range(0, len(x), block):
        messages += decoder.process(x[start : start + block])
    return messages


def _rows(messages):
    return [describe_message(m) for m in messages if m.valid]


# --------------------------------------------------------------------------- #
# Every protocol, end to end
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PROTOCOLS)
class TestEveryProtocolDecodesRealSignals:
    def test_clean_transmission_in_gui_sized_blocks(self, name):
        rate = 2.4e6
        x = _scene(name, rate)
        block = int(rate * GUI_BLOCK_S)
        blocks = _decode(create_live_decoder(name, rate), x, block)
        assert _rows(blocks) == CASES[name][1]
        # The same as decoding the whole stream at once, and as odd blocks.
        whole = create_live_decoder(name, rate).process(x)
        assert _rows(whole) == _rows(blocks)
        odd = _decode(create_live_decoder(name, rate), x, 7919)
        assert _rows(odd) == _rows(blocks)

    def test_realistic_in_channel_snr(self, name):
        rate = 480e3
        x = _scene(name, rate, snr_db=15.0, seed=1)
        messages = _decode(create_live_decoder(name, rate), x, int(rate * GUI_BLOCK_S))
        assert _rows(messages) == CASES[name][1]
        assert not [m for m in messages if not m.valid]

    def test_a_much_stronger_station_elsewhere_in_the_span(self, name):
        rate = 1.2e6
        stations = ((450e3, 40.0), (-320e3, 30.0))
        x = _scene(name, rate, snr_db=20.0, stations=stations, seed=2)
        messages = _decode(create_live_decoder(name, rate), x, int(rate * GUI_BLOCK_S))
        assert _rows(messages) == CASES[name][1]

    def test_signal_away_from_the_lo(self, name):
        rate = 1.2e6
        x = _scene(name, rate, offset=350e3, snr_db=20.0, seed=3)
        block = int(rate * GUI_BLOCK_S)
        decoder = create_live_decoder(name, rate, offset_hz=350e3)
        assert decoder.offset_hz == 350e3
        assert _rows(_decode(decoder, x, block)) == CASES[name][1]
        # Listening at the LO instead hears nothing of it.
        assert _rows(_decode(create_live_decoder(name, rate), x, block)) == []

    def test_following_the_vfo_to_a_new_offset(self, name):
        rate = 1.2e6
        x = _scene(name, rate, offset=-275e3, snr_db=20.0, seed=4, lead_s=0.3)
        decoder = create_live_decoder(name, rate, offset_hz=100e3)
        block = int(rate * GUI_BLOCK_S)
        messages = decoder.process(x[: 3 * block])
        decoder.set_offset(-275e3)
        messages += _decode(decoder, x[3 * block :], block)
        assert _rows(messages) == CASES[name][1]


@pytest.mark.parametrize("name", ["POCSAG", "FLEX", "AX.25/APRS"])
@pytest.mark.parametrize("error_hz", [-1500.0, 1500.0])
def test_fm_protocols_tolerate_a_tuning_error(name, error_hz):
    rate = 480e3
    x = _scene(name, rate, snr_db=18.0, tuning_error=error_hz, seed=5)
    messages = _decode(create_live_decoder(name, rate), x, int(rate * GUI_BLOCK_S))
    assert _rows(messages) == CASES[name][1]


def test_ads_b_in_gui_sized_blocks_away_from_the_lo():
    # ADS-B is decoded from the pulse envelope at the full rate, which does
    # not depend on where in the span the replies are.
    from sdr_module.gui import device_dialog as dd

    rate = 2.4e6
    frame = dd._extended_squitter(0x4840D6, dd._ident_me("KLM1023"))
    pulses = dd._modes_pulses(frame)
    rng = np.random.default_rng(6)
    x = (0.02 * (rng.standard_normal(int(0.2 * rate)) + 0j)).astype(np.complex64)
    x += (0.02j * rng.standard_normal(len(x))).astype(np.complex64)
    t = np.arange(len(x)) / rate
    carrier = np.exp(2j * np.pi * 300e3 * t).astype(np.complex64)
    for k in range(5):
        # On the decoder's 0.5 us slot grid (1.2 samples), across blocks.
        start, env = dd._pulse_envelope(pulses, 7200.0 + k * 90_000.0, rate)
        x[start : start + len(env)] += 0.5 * env * carrier[start : start + len(env)]
    decoder = create_live_decoder("ADS-B", rate, offset_hz=300e3)
    messages = _decode(decoder, x, int(rate * GUI_BLOCK_S))
    assert [(m.icao_address, m.callsign) for m in messages] == [
        ("4840D6", "KLM1023")
    ] * 5
    assert decoder.status.valid == 5


@pytest.mark.parametrize("rate", [2.2e6, 2.4e6, 2.56e6, 8e6, 20e6])
def test_ads_b_replies_decode_wherever_they_start(rate):
    # DEC-06: the magnitude was read in 0.5 us slots on one fixed grid, so
    # a reply starting between two slot edges spread every pulse over two
    # slots: at 2.4 MS/s about half of the strong replies were lost (none
    # that started 0.2-0.3 us into a slot). Here 20 replies start 0 to
    # 0.475 us into a slot.
    from sdr_module.gui import device_dialog as dd

    frame = dd._extended_squitter(0x4840D6, dd._ident_me("KLM1023"))
    pulses = dd._modes_pulses(frame)
    rng = np.random.default_rng(16)
    x = (0.02 * rng.standard_normal(int(0.012 * rate))).astype(np.complex64)
    x += (0.02j * rng.standard_normal(len(x))).astype(np.complex64)
    slot = 0.5e-6 * rate  # samples
    for k in range(20):
        start, env = dd._pulse_envelope(pulses, (100 + 400 * k + k / 20) * slot, rate)
        x[start : start + len(env)] += 0.5 * env
    rows = []
    for block in (len(x), int(rate * GUI_BLOCK_S), 997):
        decoder = create_live_decoder("ADS-B", rate)
        rows.append([(m.icao_address, m.callsign) for m in _decode(decoder, x, block)])
    assert rows[0] == [("4840D6", "KLM1023")] * 20
    assert rows[1] == rows[0] and rows[2] == rows[0]


@pytest.mark.parametrize("rate, phases", [(8e6, 4), (10e6, 5), (20e6, 5)])
def test_ads_b_sums_whole_samples_at_whole_ratios(rate, phases):
    # DEC-07: at 4, 5 or 10 samples per 0.5 us slot every block still went
    # through a float64 running sum and an interpolation: 9-12 ms of each
    # 33 ms frame at 20 MS/s. Now the slot values are sums of whole
    # samples (one value per offset, every 0.1-0.125 us).
    decoder = ADSBDecoder(rate)
    assert (decoder._phases, decoder._points) == (phases, 1)
    x = np.zeros(int(rate * GUI_BLOCK_S), dtype=np.float32)
    x[1000:1004] = 1.0
    slots = decoder._to_work_rate(x)
    assert len(slots) == (len(x) - decoder._need) // decoder._period + 1
    assert slots.max() == pytest.approx(4 / (rate * 0.5e-6), rel=1e-6)


@pytest.mark.parametrize("rate", [2e6, 2.4e6, 8e6, 20e6])
def test_ads_b_absurd_values_decode_to_nothing_quietly(rate):
    # Values no receiver gives (up to 3e38, as random bytes read as
    # float32 are) overflowed the float32 slot sums and the clarity of the
    # bits: a RuntimeWarning per block (in the fuzz tests too).
    import warnings

    rng = np.random.default_rng(4)
    decoder = ADSBDecoder(rate)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(10):
            data = rng.integers(0, 256, 4 * 40000, dtype=np.uint8).view(np.float32)
            assert decoder.decode(data) == []


def test_noise_alone_decodes_nothing_and_fails_nothing():
    rate = 240e3
    rng = np.random.default_rng(9)
    for name in PROTOCOLS:
        decoder = create_live_decoder(name, rate)
        for _ in range(60):  # 2 s
            n = int(rate * GUI_BLOCK_S)
            noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            assert decoder.process(noise.astype(np.complex64)) == []
        status = decoder.status
        assert (status.valid, status.failed) == (0, 0), name


def test_any_array_layout_and_complex_type():
    # Regression: a strided view (e.g. one column of a buffer) made the
    # channelizer's finite check raise ValueError, so every block failed.
    rate = 240e3
    x = _scene("AX.25/APRS", rate)
    buffer = np.zeros((len(x), 2), dtype=np.complex64)
    buffer[:, 0] = x
    strided = buffer[:, 0]
    assert not strided.flags.c_contiguous
    block = int(rate * GUI_BLOCK_S)
    expected = CASES["AX.25/APRS"][1]
    for samples in (strided, x.astype(np.complex128)):
        decoder = create_live_decoder("AX.25/APRS", rate)
        assert _rows(_decode(decoder, samples, block)) == expected


# --------------------------------------------------------------------------- #
# The channel at every rate
# --------------------------------------------------------------------------- #

CHANNEL_DECODERS = [n for n, c in LIVE_DECODERS.items() if c is not ADSBLiveDecoder]


def _channel_gain_db(channelizer, freqs):
    """Gain (dB) of ``channelizer`` for complex tones ``freqs`` Hz from the
    channel, exactly: each stage filters a tone, then its decimation moves
    the tone to its alias."""
    f = np.asarray(freqs, dtype=np.float64)
    gain = np.zeros(len(f))
    rate = channelizer.rate
    for stage in channelizer._stages:
        taps = np.asarray(stage.taps, dtype=np.float64)
        size = 1 << max(16, int(np.ceil(np.log2(32 * len(taps)))))
        response = np.abs(np.fft.fft(taps, size)) / abs(np.sum(taps))
        gain += 20 * np.log10(response[np.round(f / rate * size).astype(int) % size])
        rate /= stage.factor
        f = (f + rate / 2) % rate - rate / 2
    return gain


@pytest.mark.parametrize("rate", [2.4e6, 8e6, 20e6])
@pytest.mark.parametrize("name", CHANNEL_DECODERS)
def test_the_channel_keeps_out_everything_beyond_its_edge(name, rate):
    # DEC-02: the CW channel's first stage decimated by 600 to 5000 with a
    # filter that needed 2.6k-22k taps; it was cut to 2047, so tones near
    # 4 kHz folded into the +/-250 Hz channel (-40 dB at 2.4 MS/s, -1 dB
    # at 20 MS/s) and a station 4 kHz away was decoded.
    cls = LIVE_DECODERS[name]
    channelizer = cls(rate)._channelizer
    stop, keep = channelizer.stop_hz, channelizer.pass_hz
    freqs = np.arange(-rate / 2, rate / 2, stop / 20.0)
    gain = _channel_gain_db(channelizer, freqs)
    assert gain[np.abs(freqs) >= stop].max() < -55.0
    assert np.abs(gain[np.abs(freqs) <= keep]).max() < 0.1


@pytest.mark.parametrize("rate", [2.4e6, 8e6, 20e6])
def test_a_morse_station_4_khz_away_is_not_heard(rate):
    # DEC-02 as reported: a carrier 3.75 kHz from the tuned frequency came
    # through the CW channel at -40 dB (2.4 MS/s) to -1 dB (20 MS/s).
    def channel_power(offset):
        decoder = CWLiveDecoder(rate)
        t = np.arange(int(0.15 * rate)) / rate
        out = decoder._channelizer.process(np.exp(2j * np.pi * offset * t))
        return np.mean(np.abs(out[len(out) // 2 :]) ** 2)

    assert 10 * np.log10(channel_power(3750.0) / channel_power(0.0)) < -55.0


# --------------------------------------------------------------------------- #
# Factory, registry and status
# --------------------------------------------------------------------------- #


class TestFactory:
    def test_registry_names_match_the_decoder_panel(self):
        from sdr_module.gui.decoder_panel import PROTOCOL_INFO, PROTOCOL_UNAVAILABLE

        live = set(PROTOCOL_INFO) - set(PROTOCOL_UNAVAILABLE)
        assert set(LIVE_DECODERS) == live
        assert LIVE_DECODERS["POCSAG"] is POCSAGLiveDecoder
        assert LIVE_DECODERS["FLEX"] is FLEXLiveDecoder
        assert LIVE_DECODERS["AX.25/APRS"] is AX25LiveDecoder
        assert LIVE_DECODERS["ACARS"] is ACARSLiveDecoder
        assert LIVE_DECODERS["ADS-B"] is ADSBLiveDecoder

    def test_aliases(self):
        for alias, cls in (
            ("aprs", AX25LiveDecoder),
            ("ax25", AX25LiveDecoder),
            ("pocsag", POCSAGLiveDecoder),
            ("adsb", ADSBLiveDecoder),
        ):
            assert live_decoder_class(alias) is cls
        # RDS decodes from raw I/Q now (it needed 57 kHz subcarrier recovery).
        assert live_decoder_class("rds") is RDSLiveDecoder
        assert live_decoder_class("cw") is CWLiveDecoder
        assert live_decoder_class("teletext") is None
        with pytest.raises(ValueError):
            create_live_decoder("Off", 2.4e6)

    def test_a_rate_too_low_is_refused_with_a_reason(self):
        with pytest.raises(ValueError, match="sample rate is too low for POCSAG"):
            create_live_decoder("POCSAG", 8000)
        with pytest.raises(ValueError, match="positive"):
            create_live_decoder("ACARS", 0)

    def test_ads_b_below_2_ms_s_reports_a_problem(self):
        decoder = create_live_decoder("ADS-B", 1.8e6)
        assert decoder.status.problem == (
            "ADS-B needs at least 2 MS/s; the device runs at 1.8 MS/s"
        )
        assert decoder.process(np.ones(10000, dtype=np.complex64)) == []
        assert create_live_decoder("ADS-B", 2.4e6).status.problem is None

    def test_a_recording_too_slow_says_so(self):
        """P-05: a recording playing below ADS-B's rate was worded for the
        device ("the device runs at 1 MS/s. Choose 2.4 MS/s or more in
        Receiver > Sample rate"): a recording's rate can't be changed.
        The device keeps its advice."""
        status = create_live_decoder("ADS-B", 1e6, recording=True).status
        assert status.problem == "ADS-B needs at least 2 MS/s; this recording is 1 MS/s"
        assert status.remedy is None
        assert ADSBLiveDecoder(1e6, recording=True).status.problem == status.problem
        device = create_live_decoder("ADS-B", 1e6).status
        assert (
            device.problem == "ADS-B needs at least 2 MS/s; the device runs at 1 MS/s"
        )
        assert "Receiver > Sample rate" in device.remedy
        assert (
            create_live_decoder("ADS-B", 2.4e6, recording=True).status.problem is None
        )
        with pytest.raises(ValueError, match="^POCSAG needs at least 16 kS/s; this re"):
            create_live_decoder("POCSAG", 8000, recording=True)

    def test_channel_rate_is_about_48_khz(self):
        for rate in (1.024e6, 1.8e6, 2.048e6, 2.4e6, 3.2e6, 8e6, 20e6):
            decoder = create_live_decoder("POCSAG", rate)
            assert 40e3 <= decoder.status.channel_rate <= 60e3, rate
            assert decoder.status.sample_rate == rate
        # A recording already near the protocol rate is used as it is.
        assert create_live_decoder("AX.25/APRS", 38400).status.channel_rate == 38400


class TestStatus:
    def test_counts_and_last_time(self):
        rate = 480e3
        clock = iter([100.0, 200.0])
        decoder = POCSAGLiveDecoder(rate, clock=lambda: next(clock))
        status = decoder.status
        assert isinstance(status, LiveDecoderStatus)
        assert (status.name, status.unit) == ("POCSAG", "page")
        assert (status.frames, status.valid, status.failed) == (0, 0, 0)
        assert status.last_time is None
        _decode(decoder, _scene("POCSAG", rate), int(rate * GUI_BLOCK_S))
        status = decoder.status
        assert status.valid == 2
        assert status.frames >= 1
        assert status.failed == 0
        assert status.last_time in (100.0, 200.0)
        decoder.reset()
        assert decoder.status.valid == 0

    def test_signal_present_and_undecoded(self):
        rate = 240e3
        decoder = create_live_decoder("POCSAG", rate)
        rng = np.random.default_rng(4)
        n = int(rate * 0.05)

        def noise():
            return (
                0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            ).astype(np.complex64)

        for _ in range(80):  # 4 s of noise: the floor
            decoder.process(noise())
        assert decoder.status.signal_measured
        assert not decoder.status.signal_present
        # A carrier with voice-like FM, which no pager decoder can read.
        t = np.arange(n * 160) / rate
        voice = np.exp(1j * 2000 * np.sin(2 * np.pi * 400 * t) / 400).astype(
            np.complex64
        )
        for i in range(160):
            decoder.process(voice[i * n : (i + 1) * n] + noise())
        status = decoder.status
        assert status.signal_present
        assert status.undecoded_signal_s > 6.0
        assert status.valid == 0
        # A page resets the count.
        page = _scene("POCSAG", rate)
        _decode(decoder, page, n)
        assert decoder.status.valid == 2
        assert decoder.status.undecoded_signal_s < 2.0

    def test_status_is_a_snapshot(self):
        decoder = create_live_decoder("ACARS", 240e3)
        status = decoder.status
        status.valid = 99
        assert decoder.status.valid == 0

    @staticmethod
    def _undecodable_signal(decoder, rate, seconds, rng):
        """Noise for the floor, then ``seconds`` of voice-like FM."""
        n = int(rate * 0.05)

        def noise():
            return (
                0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            ).astype(np.complex64)

        for _ in range(60):  # 3 s of noise: the floor
            decoder.process(noise())
        t = np.arange(n) / rate
        for i in range(int(seconds / 0.05)):
            phase = 5.0 * np.sin(2 * np.pi * 400 * (t + i * 0.05))
            decoder.process(np.exp(1j * phase).astype(np.complex64) + noise())
        return noise

    def test_a_new_channel_forgets_the_signal_heard_on_the_old_one(self):
        # Regression: after 6 s of something undecodable the status said
        # "Signal present but nothing decodes" and kept saying it after the
        # receiver was retuned to a quiet channel.
        rate = 48e3
        rng = np.random.default_rng(12)
        decoder = create_live_decoder("POCSAG", rate)
        self._undecodable_signal(decoder, rate, 8.0, rng)
        assert decoder.status.undecoded_signal_s > 6.0
        decoder.channel_changed()
        status = decoder.status
        assert (status.undecoded_signal_s, status.signal_present) == (0.0, False)
        # Following the VFO to another offset is a new channel too.
        self._undecodable_signal(decoder, rate, 8.0, rng)
        decoder.set_offset(0.0)  # unchanged: nothing is forgotten
        assert decoder.status.undecoded_signal_s > 6.0
        decoder.set_offset(5e3)
        assert decoder.status.undecoded_signal_s == 0.0

    def test_undecoded_signal_is_forgotten_once_the_channel_is_quiet(self):
        rate = 48e3
        rng = np.random.default_rng(13)
        decoder = create_live_decoder("POCSAG", rate)
        noise = self._undecodable_signal(decoder, rate, 8.0, rng)
        for _ in range(int(20 / 0.05)):  # a short pause keeps it
            decoder.process(noise())
        assert decoder.status.undecoded_signal_s > 6.0
        for _ in range(int(12 / 0.05)):  # 32 s of quiet: gone
            decoder.process(noise())
        status = decoder.status
        assert (status.undecoded_signal_s, status.signal_present) == (0.0, False)


# --------------------------------------------------------------------------- #
# The protocol layer
# --------------------------------------------------------------------------- #


class TestBch:
    def test_corrects_one_and_two_errors(self):
        rng = np.random.default_rng(0)
        for data in rng.integers(0, 1 << 21, 20).tolist():
            word = bch3121_encode(data)
            assert bch3121_correct(word) == (True, word, 0)
            i, j = rng.choice(31, 2, replace=False).tolist()
            assert bch3121_correct(word ^ (1 << i)) == (True, word, 1)
            assert bch3121_correct(word ^ (1 << i) ^ (1 << j)) == (True, word, 2)

    def test_pocsag_word_with_three_errors_is_rejected(self):
        decoder = POCSAGDecoder(48000)
        word = enc.pocsag_codeword(1, 0x12345)
        assert decoder._check_bch(word) == (True, word)
        rejected = 0
        rng = np.random.default_rng(1)
        for _ in range(50):
            flips = rng.choice(32, 3, replace=False).tolist()
            bad = word
            for bit in flips:
                bad ^= 1 << bit
            ok, _fixed = decoder._check_bch(bad)
            rejected += not ok
        assert rejected >= 45  # parity catches nearly every triple error


class TestBitSlicer:
    @pytest.mark.parametrize("integrate", [True, False])
    @pytest.mark.parametrize("sps", [5.0, 20.0, 42.67, 93.75])
    def test_blocks_of_any_size_give_the_same_bits(self, integrate, sps):
        rng = np.random.default_rng(int(sps))
        bits = rng.integers(0, 2, 300)
        x = np.repeat(bits * 2.0 - 1, int(sps)) + 0.7 * rng.standard_normal(
            300 * int(sps)
        )
        whole = BitSlicer(sps * 1000, 1000, integrate).process(x)
        slicer = BitSlicer(sps * 1000, 1000, integrate)
        parts, pos = [], 0
        while pos < len(x):
            size = int(rng.integers(1, int(sps * 6)))
            parts += slicer.process(x[pos : pos + size])
            pos += size
        assert parts == whole

    def test_recovers_the_clock_of_a_slightly_fast_sender(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 2000)
        rate, baud = 48000.0, 1200.0 * 1.002  # 2000 ppm fast
        x = enc.bit_levels(bits.tolist(), rate, baud) + 0.5 * rng.standard_normal(
            int(np.ceil(2000 * rate / baud))
        )
        got = np.array(BitSlicer(rate, 1200.0).process(x))
        # Align and count errors (no bit is slipped).
        best = min(
            int(np.sum(got[off : off + 1900] != bits[:1900])) for off in range(0, 10)
        )
        assert best == 0


class TestPOCSAG:
    def _decode(self, bits, baud=1200, rate=48000.0, invert=False, **kwargs):
        levels = enc.bit_levels(bits, rate, baud)
        if invert:
            levels = -levels
        decoder = POCSAGDecoder(rate, **kwargs)
        messages = []
        x = np.concatenate([levels, np.zeros(int(rate * 0.2))])
        for start in range(0, len(x), 1601):
            messages += decoder.decode(x[start : start + 1601])
        return decoder, messages

    @pytest.mark.parametrize("baud", [512, 1200, 2400])
    def test_every_baud_rate_at_once(self, baud):
        bits = enc.pocsag_bits([(1234567, 3, "HELLO")])
        decoder, messages = self._decode(bits, baud)
        assert [(m.address, m.content, m.baud_rate) for m in messages] == [
            (1234567, "HELLO", baud)
        ]
        assert decoder.baud_rate == baud

    def test_noise_does_not_lock_the_baud_rate(self):
        # Regression: auto-baud scored raw transition counts, so noise always
        # picked 2400 baud and a 1200 baud page then never decoded.
        rng = np.random.default_rng(0)
        rate = 48000.0
        decoder = POCSAGDecoder(rate)
        decoder.decode(rng.standard_normal(int(rate)))
        levels = enc.bit_levels(enc.pocsag_bits([(8, 3, "AFTER NOISE")]), rate, 1200)
        messages = decoder.decode(np.concatenate([levels, np.zeros(9600)]))
        assert [m.content for m in messages] == ["AFTER NOISE"]

    def test_either_polarity(self):
        bits = enc.pocsag_bits([(1234567, 3, "HELLO")])
        _decoder, messages = self._decode(bits, invert=True)
        assert [m.content for m in messages] == ["HELLO"]

    def test_long_page_continues_across_batches(self):
        text = "A long page that needs more than one batch of codewords to send"
        _decoder, messages = self._decode(enc.pocsag_bits([(1234560, 3, text)]))
        assert [(m.address, m.content, m.valid) for m in messages] == [
            (1234560, text, True)
        ]

    def test_numeric_digits_are_sent_least_significant_bit_first(self):
        _decoder, messages = self._decode(
            enc.pocsag_bits([(800, 0, "0123456789-U ()")])
        )
        assert [(m.message_type, m.content) for m in messages] == [
            ("numeric", "0123456789-U ()")
        ]

    def test_tone_only_page(self):
        _decoder, messages = self._decode(enc.pocsag_bits([(16, 1, None)]))
        assert [(m.address, m.message_type, m.content) for m in messages] == [
            (16, "tone", "")
        ]

    def test_a_damaged_codeword_makes_the_page_invalid(self):
        bits = enc.pocsag_bits([(1234567, 3, "DAMAGED PAGE TEXT")])
        start = 576 + 32 + 15 * 32  # the first message codeword
        for i in (3, 9, 17):
            bits[start + i] ^= 1  # three errors: uncorrectable
        decoder, messages = self._decode(bits)
        assert len(messages) == 1
        assert not messages[0].valid
        assert "BCH" in messages[0].error_message
        assert decoder.frames_failed == 1

    LONG_TEXT = (
        "This is a long page that continues into the next batch so that a "
        "lost sync word cuts it short in the middle, ok"
    )

    @pytest.mark.parametrize("errors", [4, 5])
    def test_a_page_cut_by_a_lost_batch_is_not_valid(self, errors):
        # DEC-05: the second batch's sync word after a short fade (more
        # than 4 bit errors) ended the transmission, and the open page came
        # out valid as "Th" instead of its 111 characters.
        bits = enc.pocsag_bits([(1234567, 3, self.LONG_TEXT)])  # frame 7
        sync = POCSAGDecoder.PREAMBLE_BITS + 32 * 17  # the 2nd sync word
        for k in range(errors):
            bits[sync + 3 + 6 * k] ^= 1
        decoder, messages = self._decode(bits)
        page = messages[0]
        if errors == 4:  # still found: the whole page
            assert (page.content, page.valid) == (self.LONG_TEXT, True)
            return
        assert page.content == "Th"
        assert not page.valid
        assert page.error_message == "cut short: the next batch was lost"
        assert decoder.frames_failed >= 1

    def test_a_page_cut_at_its_longest_is_not_valid(self):
        text = ("0123456789abcdefghij" * 40)[:780]  # 273 message codewords
        decoder, messages = self._decode(enc.pocsag_bits([(8, 3, text)]))
        page = messages[0]
        assert text.startswith(page.content) and len(page.content) < len(text)
        assert not page.valid
        assert page.error_message == "cut short after 200 codewords"

    def test_one_or_two_bit_errors_are_corrected(self):
        bits = enc.pocsag_bits([(1234567, 3, "FIXED")])
        start = 576 + 32 + 15 * 32
        bits[start + 4] ^= 1
        bits[start + 40] ^= 1
        bits[start + 44] ^= 1
        _decoder, messages = self._decode(bits)
        assert [(m.content, m.valid) for m in messages] == [("FIXED", True)]

    def test_noise_that_matches_the_sync_word_is_not_a_transmission(self):
        # Regression: noise matches the sync word (within two bit errors, at
        # one of three speeds) a few times an hour. The random codewords
        # after it were taken as a transmission: garbage pages, and the
        # status line warned that pages "failed their checks" on an idle
        # channel.
        rng = np.random.default_rng(11)
        check = POCSAGDecoder(48000.0)._check_bch
        junk = []
        while len(junk) < 16:
            word = int(rng.integers(0, 1 << 32))
            if not check(word)[0]:
                junk.append(word)
        bits = rng.integers(0, 2, 100).tolist()
        for word in [POCSAGDecoder.SYNC_WORD ^ (1 << 7)] + junk:
            bits += [(word >> (31 - i)) & 1 for i in range(32)]
        # A page right after it still decodes.
        bits += rng.integers(0, 2, 8).tolist()
        bits += enc.pocsag_bits([(1234567, 3, "AFTER THE NOISE")])
        decoder, messages = self._decode(bits)
        assert [(m.address, m.content, m.valid) for m in messages] == [
            (1234567, "AFTER THE NOISE", True)
        ]
        assert (decoder.frames_seen, decoder.frames_failed) == (1, 0)


class TestFLEX:
    def _decode(self, bits, rate=48000.0, invert=False):
        levels = enc.bit_levels(bits, rate, 1600)
        decoder = FLEXDecoder(rate)
        x = np.concatenate([-levels if invert else levels, np.zeros(4800)])
        messages = []
        for start in range(0, len(x), 1601):
            messages += decoder.decode(x[start : start + 1601])
        return decoder, messages

    def test_frame_with_two_pages(self):
        pages = [(1234567, "First FLEX page"), (1000123, "Second: 10:30")]
        decoder, messages = self._decode(enc.flex_frame_bits(pages, cycle=4, frame=99))
        assert [(m.capcode, m.content, m.message_type) for m in messages] == [
            (1234567, "First FLEX page", "alpha"),
            (1000123, "Second: 10:30", "alpha"),
        ]
        assert {(m.cycle, m.frame, m.phase, m.valid) for m in messages} == {
            (4, 99, "A", True)
        }
        assert (decoder.frames_seen, decoder.frames_failed) == (1, 0)

    def test_either_polarity(self):
        _decoder, messages = self._decode(
            enc.flex_frame_bits([(77777, "INVERTED")]), invert=True
        )
        assert [m.content for m in messages] == ["INVERTED"]

    def test_damaged_message_word_makes_the_page_invalid(self):
        bits = enc.flex_frame_bits([(1234567, "DAMAGED FLEX PAGE")])
        data = 32 + 64 + 16 + 32 + 40  # where the 11 blocks start
        # Word 4 (the first character word) is bits 4, 12, 20, ... of block 0.
        for i in (0, 5, 11):
            bits[data + 4 + 8 * i] ^= 1
        decoder, messages = self._decode(bits)
        assert len(messages) == 1 and not messages[0].valid
        assert "BCH" in messages[0].error_message
        assert decoder.frames_failed == 1

    @pytest.mark.parametrize(
        "code, speed",
        [
            # ARIB STD-43A sync codes, as multimon-ng reads them (0xDEA0 was
            # reported as "3200 bps 4-level", and 0x7B18 / 0xB068 swapped).
            (0x7B18, "3200 bps 2-level"),
            (0xB068, "3200 bps 4-level"),
            (0xDEA0, "6400 bps 4-level"),
            (0x4C7C, "6400 bps 4-level"),
        ],
    )
    def test_other_speeds_are_recognised_and_read_at_their_speed(self, code, speed):
        # K-D4: they were "not supported yet" (unsupported_frames); now the
        # data after the header is read at the speed the sync code names
        # (tests/test_flex_speeds.py decodes every speed and phase).
        bits = enc.flex_frame_bits([(1234567, "AT 1600")])
        # Make the sync pattern name another speed.
        sync = (code << 48) | (0xA6C6AAAA << 16) | (code ^ 0xFFFF)
        bits[32:96] = [((sync >> (63 - i)) & 1) ^ 1 for i in range(64)]
        decoder, messages = self._decode(bits)
        assert decoder.frames_seen == 1
        assert (decoder.unsupported_frames, decoder.last_unsupported) == (0, "")
        assert {m.baud_rate for m in messages} <= {int(speed.split()[0])}

    @staticmethod
    def _fast_frames(count, gap_s=1.875, rate=240e3):
        """``count`` FLEX frame headers saying 3200 bps 2-level, ``gap_s``
        apart, on an FM channel with a little noise."""
        rng = np.random.default_rng(3)
        code = 0x7B18
        sync = (code << 48) | (0xA6C6AAAA << 16) | (code ^ 0xFFFF)
        bits = [1 - (i & 1) for i in range(64)]
        bits += [((sync >> (63 - i)) & 1) ^ 1 for i in range(64)]
        bits += [1 - (i & 1) for i in range(200)]
        header = enc.fm_modulate(
            enc.FLEX_DEVIATION_HZ * enc.bit_levels(bits, rate, 1600), rate
        )
        gap = np.zeros(int(gap_s * rate) - len(header), dtype=np.complex64)
        x = np.concatenate([np.concatenate([gap, header]) for _ in range(count)])
        noise = rng.standard_normal(2 * len(x)).view(np.complex128)
        return (x + 0.001 * noise).astype(np.complex64)

    def _feed(self, decoder, x, block=8000):
        for start in range(0, len(x), block):
            decoder.process(x[start : start + block])

    @staticmethod
    def _slow_reader(rate):
        """A FLEX live decoder that can't read 3200 symbols/s, as at a
        sample rate below 9.6 kS/s (K-D4: at the channel's own rate every
        speed decodes)."""
        decoder = FLEXLiveDecoder(rate)
        decoder._decoder._fast_ok = False
        return decoder

    def test_a_speed_it_cant_decode_is_a_problem_while_it_is_heard(self):
        # DEC-09: the problem came from a lifetime count, so it stayed on
        # a quiet channel, after retuning and after Clear.
        rate = 240e3
        decoder = self._slow_reader(rate)
        self._feed(decoder, self._fast_frames(4))
        problem = decoder.status.problem
        assert problem == (
            "This channel sends FLEX at 3200 bps 2-level, which the decoder "
            "can't read at this sample rate"
        )
        # Another channel: forgotten at once.
        decoder.set_offset(50e3)
        assert decoder.status.problem is None
        self._feed(decoder, np.zeros(int(9 * rate), dtype=np.complex64))
        assert decoder.status.problem is None
        # Back on it: there again with its next frame, and gone a few frames
        # after the channel goes quiet.
        decoder.set_offset(0.0)
        self._feed(decoder, self._fast_frames(2))
        assert decoder.status.problem == problem
        quiet = np.zeros(int(4 * rate), dtype=np.complex64)
        self._feed(decoder, quiet)
        assert decoder.status.problem == problem
        self._feed(decoder, quiet)
        assert decoder.status.problem is None

    def test_clear_forgets_a_speed_it_cant_decode(self):
        # Leftover 29: the Decoder's Clear left "This channel sends FLEX at
        # 3200 bps" on the status line until 6 s after the last such frame.
        decoder = self._slow_reader(240e3)
        self._feed(decoder, self._fast_frames(4))
        problem = decoder.status.problem
        assert problem
        decoder.messages_cleared()
        assert decoder.status.problem is None
        # Still sent: back with the next such frame.
        self._feed(decoder, self._fast_frames(1))
        assert decoder.status.problem == problem

    def test_reset_forgets_other_speeds(self):
        decoder = FLEXDecoder(48000.0)
        decoder.unsupported_frames, decoder.last_unsupported = 3, "3200 bps 2-level"
        decoder.reset()
        assert (decoder.unsupported_frames, decoder.last_unsupported) == (0, "")

    @staticmethod
    def _long_address(capcode):
        """The two address words of a long address (set 1-2)."""
        first = (capcode - 2068480) % 32768
        second = 2097151 - (capcode - 2068480) // 32768
        assert 1 <= first <= 32768 and 2064383 <= second <= 2097150
        return [first, second]

    def test_long_address_numeric_page(self):
        # Regression: with a long address the first message word is the
        # second vector word (multimon-ng, ARIB STD-43A 3.9.1); the digits
        # were read from the message field only, one word too far.
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        stream = [0, 0] + [(d >> i) & 1 for d in digits for i in range(4)]
        body = [sum(b << i for i, b in enumerate(stream[k : k + 21])) for k in (0, 21)]
        values = [enc._flex_checksum(3 << 10)]  # BIW: vectors start at word 3
        values += self._long_address(2232321)  # words 1-2
        values += [(3 << 4) | (5 << 7) | (1 << 14)]  # numeric, 2 words, from 5
        values += body  # word 4 (the second vector word) and word 5
        _decoder, messages = self._decode(enc.flex_words_bits(values))
        assert [(m.capcode, m.message_type, m.content, m.valid) for m in messages] == [
            (2232321, "numeric", "1234567890", True)
        ]

    def test_long_address_alpha_page(self):
        text = "LONG ADDRESS PAGE"
        words = enc._flex_alpha_words(text, 0)
        values = [enc._flex_checksum(3 << 10)]
        values += self._long_address(2232321)
        # Alphanumeric: the header is the second vector word, the
        # characters follow from word 5.
        values += [(5 << 4) | (5 << 7) | (len(words) << 14), words[0]] + words[1:]
        _decoder, messages = self._decode(enc.flex_words_bits(values))
        assert [(m.capcode, m.content, m.valid) for m in messages] == [
            (2232321, text, True)
        ]

    def test_a_long_address_cut_off_by_the_vectors_is_skipped(self):
        # Its second word would be the first vector word: no page (it was
        # shown with a negative capcode).
        values = [enc._flex_checksum(2 << 10), 5, (5 << 4) | (3 << 7) | (2 << 14)]
        values += enc._flex_alpha_words("X", 0)
        _decoder, messages = self._decode(enc.flex_words_bits(values))
        assert messages == []

    def test_numeric_page_layout(self):
        # Digits are 4 bits, least significant first, after 2 header bits,
        # packed across the 21-bit words (multimon-ng's layout).
        digits = [1, 2, 3, 4, 5] + [0xC] * 5  # 0xC fills the last word
        stream = [0, 0] + [(d >> i) & 1 for d in digits for i in range(4)]
        words = [sum(b << i for i, b in enumerate(stream[k : k + 21])) for k in (0, 21)]
        assert FLEXDecoder(48000)._decode_numeric(words) == "12345"


class TestAX25:
    def test_damaged_frame_is_returned_invalid_and_counted(self):
        frame = bytearray(enc.ax25_frame("N0CALL", "APZ001", "Damaged packet"))
        frame[-5] ^= 0x10  # inside the info field
        line = enc.hdlc_bits(bytes(frame))
        decoder = AX25Decoder(48000.0)
        frames = decoder.decode(
            np.concatenate([enc.bit_levels(line, 48000.0, 1200), np.zeros(4800)])
        )
        assert [(f.valid, f.error_message) for f in frames] == [(False, "CRC mismatch")]
        assert (decoder.frames_seen, decoder.frames_failed) == (1, 1)

    def test_flags_and_noise_between_them_are_not_frames(self):
        rng = np.random.default_rng(3)
        decoder = AX25Decoder(48000.0)
        decoder.decode(rng.standard_normal(48000 * 3))
        assert (decoder.frames_seen, decoder.frames_failed) == (0, 0)


class TestACARS:
    def test_block_layout(self):
        block = ACARS_BLOCK
        # Every character carries odd parity.
        assert all(bin(b).count("1") % 2 == 1 for b in block)
        assert block[0] & 0x7F == ord("2")
        assert bytes(b & 0x7F for b in block[1:8]) == b".N601SD"
        assert block[-1] & 0x7F == ACARSDecoder.ETX
        bits = enc.acars_bits(block)
        # '+' is sent as 0xAB: the verifier's real-world sync example.
        first = bits[128:136]
        assert sum(b << i for i, b in enumerate(first)) == 0xAB

    def test_a_damaged_block_is_returned_invalid_with_its_fields(self):
        bits = enc.acars_bits(ACARS_BLOCK)
        # One bit of a text character (after the pre-key and the 5 sync
        # characters): a parity error and a CRC mismatch.
        bits[128 + 8 * (5 + 20) + 2] ^= 1
        levels = np.array(bits, dtype=float) * 2 - 1
        decoder = ACARSDecoder(48000.0)
        messages = decoder.decode(np.repeat(levels, 20))
        assert len(messages) == 1
        msg = messages[0]
        assert not msg.valid
        assert "CRC mismatch" in msg.error_message
        assert msg.registration == "N601SD" and msg.label == "H1"
        assert decoder.frames_failed == 1

    def test_tone_input_with_differential_decoding(self):
        bits = enc.acars_bits(ACARS_BLOCK)
        tones, last = [], 1
        for bit in bits:
            tones.append(1.0 if bit == last else -1.0)
            last = bit
        decoder = ACARSDecoder(48000.0, differential=True)
        messages = decoder.decode(np.repeat(np.array(tones), 20))
        assert [(m.flight_id, m.message_number, m.text) for m in messages] == [
            ("SD0601", "M01A", "TEST MESSAGE")
        ]


class TestDescribeMessage:
    def test_rows(self):
        from sdr_module.dsp.protocols import (
            ACARSMessage,
            AX25Frame,
            FLEXMessage,
            POCSAGMessage,
            ProtocolType,
        )

        base = dict(timestamp=0.0, raw_bits=b"", valid=True)
        assert describe_message(
            POCSAGMessage(protocol=ProtocolType.POCSAG, address=8, function=3,
                          content="HI", **base)
        ) == ("8 (F3)", "HI")  # fmt: skip
        assert describe_message(
            FLEXMessage(protocol=ProtocolType.FLEX, capcode=12, content="X", **base)
        ) == ("12 (1600 bps)", "X")
        assert describe_message(
            AX25Frame(protocol=ProtocolType.AX25, source="A", destination="B",
                      info="!x", **base)
        ) == ("A>B", "!x")  # fmt: skip
        assert describe_message(
            ACARSMessage(protocol=ProtocolType.ACARS, registration="N1",
                         label="H1", text="T", **base)
        ) == ("N1", "H1  ·  T")  # fmt: skip
        assert describe_message(
            ADSBMessage(protocol=ProtocolType.ADSB, icao_address="4840D6",
                        callsign="KLM1023", altitude=38000, **base)
        ) == ("4840D6", "KLM1023  ·  38,000 ft")  # fmt: skip
