#!/usr/bin/env python3
"""
The display loop's cost at high sample rates (the final perf fixes).

At 20 MS/s (a HackRF) a display frame carries 666,667 samples every 33 ms.
The end-to-end checks found broadcast FM, None (I/Q) and the Decoder's RDS
taking more than that frame's time on a 4-core computer: the channel filter
computed each kept output in several passes (or filtered every sample by
FFT and kept one in five), the Decoder's RDS picked the station out of the
full-rate block a second time, and the plots transformed every sample even
of a catch-up frame several times longer. These tests pin the work each
part does (counts, not timings: they hold on a busy computer too).

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_perf.py``
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import _WindowTestCase, require_pyqt6  # noqa: E402

RATE = 20e6
FRAME = 666_667  # one display frame at 20 MS/s


def _mw():
    from sdr_module.gui import main_window

    return main_window


def _noise(n, seed=1):
    rng = np.random.default_rng(seed)
    return (0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )


class _CountDots:
    """Counts the dot products (rows) ``module._row_dot`` computes."""

    def __init__(self, module):
        self.calls = 0
        self.rows = 0
        self._module = module
        self._orig = module._row_dot

    def __enter__(self):
        def spy(rows, weights):
            self.calls += 1
            self.rows += int(np.asarray(rows).shape[0])
            return self._orig(rows, weights)

        self._patch = mock.patch.object(self._module, "_row_dot", spy)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


# ---------------------------------------------------------------------------
# LISTEN-06, PLOTS-E2E-5, HAM-E2E-3: the receiver's channel filter
# ---------------------------------------------------------------------------


class TestReceiverChannelFilterCost(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def test_broadcast_fm_at_20_msps_computes_each_kept_output_once(self):
        # The 200 kHz channel's first stage (67 taps, decimating by 15) was
        # computed in five passes of short rows (one per group of 15 taps):
        # 7-9 ms of each 33 ms frame. Each kept output is one dot product.
        chain = self.mw._ReceiverChain(RATE, "FM", 200e3)
        outputs = 0
        with _CountDots(self.mw) as dots:
            x = _noise(FRAME)
            for stage in chain._channel:
                x = stage.process(x)
                if stage._groups:
                    outputs += len(x)
        self.assertGreater(outputs, 0)
        self.assertEqual(dots.rows, outputs)
        self.assertEqual(dots.calls, sum(1 for s in chain._channel if s._groups))

    def test_no_channel_at_20_msps_filters_every_sample_by_fft(self):
        # None (I/Q) with its 2.4 MHz filter decimates by 5: the 243-tap
        # filter ran as an FFT over all 666,667 samples and kept one output
        # in five (15-29 ms a frame, so I/Q at 20 MS/s could not keep up).
        for mode, width in (("None (I/Q)", 2.4e6), ("FM", 200e3), ("FM", 25e3)):
            chain = self.mw._ReceiverChain(RATE, mode, width)
            first = chain._channel[0]
            self.assertGreater(first._factor, 1, mode)
            self.assertTrue(first._groups, (mode, width))

    def test_kept_outputs_match_filtering_every_sample(self):
        # The direct form gives what filtering every sample by FFT and
        # keeping every factor-th one gives, for complex and real streams
        # cut into blocks of any size.
        rng = np.random.default_rng(3)
        for taps_n, factor in ((67, 15), (243, 5), (15, 2), (195, 3), (607, 135)):
            taps = np.hanning(taps_n + 2)[1:-1].astype(np.float32)
            taps /= taps.sum()
            for complex_ in (True, False):
                n = 50_021
                x = rng.standard_normal(n).astype(np.float32)
                if complex_:
                    x = (x + 1j * rng.standard_normal(n)).astype(np.complex64)
                fast = self.mw._FirDecimator(taps, factor)
                fast._groups = fast._groups or 1  # (the direct form)
                slow = self.mw._FirDecimator(taps, factor)
                slow._groups = 0  # FFT overlap-save, then every factor-th
                cuts = np.unique(
                    np.concatenate(([0, n], rng.integers(0, n, 9), [1, 2, 3]))
                )
                a = np.concatenate(
                    [
                        fast.process(x[i:j])
                        for i, j in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                b = slow.process(x)
                self.assertEqual(len(a), len(b), (taps_n, factor, complex_))
                self.assertEqual(a.dtype, b.dtype)
                np.testing.assert_allclose(
                    a, b, atol=2e-5, err_msg=str((taps_n, factor, complex_))
                )


# ---------------------------------------------------------------------------
# Leftover 6, PLOTS-E2E-12: the decoders' and Identify's channel
# ---------------------------------------------------------------------------


def _channelizer():
    from sdr_module.dsp import channelizer

    return channelizer


class TestDecoderChannelCost(unittest.TestCase):
    def _counted(self, chan, x):
        """``(dot products, kept outputs)`` of one block through ``chan``."""
        module = _channelizer()
        outputs = 0
        with _CountDots(module) as dots:
            y = chan._nco.process(x)
            for stage in chan._stages:
                y = stage.process(y)
                outputs += len(y)
        return dots.rows, outputs

    def test_identify_at_20_msps_computes_each_kept_output_once(self):
        # Identify on a 2.4 MHz I/Q channel at 20 MS/s took 32-115 ms of
        # each 33 ms block (each of its taps a strided pass over the
        # block), and 16 ms on a broadcast FM channel: each kept output is
        # now one dot product.
        from sdr_module.dsp.signal_id import SignalIdentifier, analysis_band

        for mode, width in (("None (I/Q)", 2.4e6), ("FM", 200e3), ("FM", 25e3)):
            band, _reach = analysis_band(mode, width, RATE)
            chan = SignalIdentifier(RATE, 150e3, band)._channel
            self.assertIsNotNone(chan, mode)
            rows, outputs = self._counted(chan, _noise(FRAME))
            self.assertEqual(rows, outputs, (mode, width))

    def test_live_decoders_at_20_msps_compute_each_kept_output_once(self):
        from sdr_module.dsp.live_decoders import LIVE_DECODERS, ADSBLiveDecoder

        for name, cls in LIVE_DECODERS.items():
            if cls is ADSBLiveDecoder:
                continue  # (no channel: the pulses at the full rate)
            chan = cls(RATE, 25e3)._channelizer
            rows, outputs = self._counted(chan, _noise(FRAME))
            self.assertEqual(rows, outputs, name)

    def test_stages_match_the_filter_computed_tap_by_tap(self):
        # Each stage's kept outputs are those of the plain convolution, for
        # complex and real streams cut into blocks of any size.
        module = _channelizer()
        rng = np.random.default_rng(5)
        for taps_n, factor in ((19, 3), (85, 2), (567, 135), (15, 1), (5, 2)):
            taps = rng.standard_normal(taps_n)
            taps = (taps / np.sum(np.abs(taps))).astype(np.float32)
            for complex_ in (True, False):
                n = 20_011
                x = rng.standard_normal(n).astype(np.float32)
                if complex_:
                    x = (x + 1j * rng.standard_normal(n)).astype(np.complex64)
                stage = module.FirDecimator(taps, factor)
                cuts = np.unique(
                    np.concatenate(([0, n], rng.integers(0, n, 9), [1, 2, 3]))
                )
                got = np.concatenate(
                    [
                        stage.process(x[i:j])
                        for i, j in zip(cuts[:-1], cuts[1:], strict=True)
                    ]
                )
                full = np.convolve(x.astype(np.complex128), taps.astype(np.float64))
                want = full[:n:factor]
                self.assertEqual(len(got), len(want), (taps_n, factor))
                np.testing.assert_allclose(
                    got, want, atol=2e-5, err_msg=str((taps_n, factor, complex_))
                )


# ---------------------------------------------------------------------------
# Leftover 6, DEV-E2E-19: the plots' spectrum of a long frame
# ---------------------------------------------------------------------------


def _tone(n, freq, rate, amplitude=1.0):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


class TestFrameSpectrumWorkIsBounded(unittest.TestCase):
    CAP = 1 << 19

    def test_a_long_frame_is_analysed_in_evenly_spread_segments(self):
        # Every FFT segment of every frame: 325 at 20 MS/s (13-17 ms a
        # frame), 1024 for a 2 M-sample catch-up frame after a stall
        # (60-330 ms: a visible hitch that made the next frame later
        # still). With a cap, a long frame is analysed in evenly spread
        # segments, the newest included.
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        size = 2048
        n = 1 << 21
        data = _noise(n, seed=4) * np.float32(0.01) + _tone(n, 3e6, RATE, 0.1)
        whole = FrameSpectrumAnalyzer(size, threads=1).process(data)
        capped = FrameSpectrumAnalyzer(size, threads=1, max_samples=self.CAP)
        spread = capped.process(data)
        self.assertEqual(whole.segments, n // size)
        self.assertLessEqual(spread.segments, self.CAP // size)
        self.assertGreaterEqual(spread.segments, self.CAP // size // 2)
        # The same spectrum: the tone and the noise floor read the same.
        b = size // 2 + int(round(3e6 / (RATE / size)))
        self.assertAlmostEqual(
            float(spread.mean_db[b]), float(whole.mean_db[b]), delta=0.2
        )
        self.assertAlmostEqual(spread.noise_floor_db, whole.noise_floor_db, delta=0.5)

    def test_the_newest_segment_and_longer_bursts_still_show(self):
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        size = 2048
        n = 1 << 21
        burst = _tone(4 * size, -2e6, RATE, 0.1)
        b = size // 2 + int(round(-2e6 / (RATE / size)))
        for start in (0, 777_777, n - 4 * size, n - size):
            data = _noise(n, seed=6) * np.float32(0.001)
            stop = min(n, start + len(burst))
            data[start:stop] += burst[: stop - start]
            a = FrameSpectrumAnalyzer(size, threads=1, max_samples=self.CAP)
            spec = a.process(data)
            floor = float(np.median(spec.max_db))
            self.assertGreater(float(spec.max_db[b]), floor + 30, start)

    def test_frames_within_the_cap_are_analysed_whole(self):
        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        a = FrameSpectrumAnalyzer(2048, threads=1, max_samples=self.CAP)
        self.assertEqual(a.max_samples, self.CAP)
        self.assertEqual(a.process(_noise(80_000)).segments, 39)  # 2.4 MS/s
        self.assertEqual(a.process(_noise(333_334)).segments, 162)  # 10 MS/s
        a.max_samples = None
        self.assertEqual(a.process(_noise(1 << 21)).segments, 1024)


class TestPlotsInTheWindow(_WindowTestCase):
    demo = True

    def test_a_catch_up_frame_costs_the_plots_no_more_than_a_usual_one(self):
        analyzer = self.win._frame_spectrum
        self.assertEqual(analyzer.max_samples, self.mw._PLOT_MAX_SAMPLES)
        frames = []
        process = analyzer.process

        def spy(samples):
            frame = process(samples)
            frames.append(frame)
            return frame

        analyzer.process = spy
        self.win._plot_frame(_noise(80_000))  # 2.4 MS/s: every sample
        self.win._plot_frame(_noise(1 << 21))  # a stall's backlog at 20 MS/s
        self.assertEqual(frames[0].segments, 80_000 // 2048)
        self.assertLessEqual(
            frames[1].segments, self.mw._PLOT_MAX_SAMPLES // analyzer.size
        )


# ---------------------------------------------------------------------------
# ROB-06: the Decoder's RDS and the receiver's broadcast FM channel
# ---------------------------------------------------------------------------


class TestRdsFromTheReceiversChannel(unittest.TestCase):
    def test_a_raw_decoder_decodes_a_channel_handed_to_it(self):
        from sdr_module.dsp.live_decoders import RDSLiveDecoder, describe_message
        from tests.test_rds import _wfm

        # The receiver's broadcast FM channel at 20 MS/s: 444 kS/s at 0 Hz.
        decoder = RDSLiveDecoder(RATE, 0.0)
        rate = RATE / 45
        x = _wfm(rate, 2.5, 0.0)
        own = mock.Mock(wraps=decoder._channelizer.process)
        decoder._channelizer.process = own
        messages = []
        block = int(rate / 30)
        for i in range(0, len(x), block):
            messages += decoder.process_channel(x[i : i + block], rate)
        rows = [describe_message(m) for m in messages]
        self.assertIn(("PI C0DE", "Station name: SDR DEMO"), rows)
        self.assertEqual(own.call_count, 0)
        status = decoder.status
        self.assertEqual(status.channel_rate, rate)
        self.assertGreater(status.valid, 0)
        self.assertAlmostEqual(status.stream_s, len(x) / rate, places=3)
        # Raw blocks again: its own channel, at its own rate (the station
        # it knows stays known).
        decoder.process(np.zeros(int(RATE / 30), dtype=np.complex64))
        self.assertEqual(own.call_count, 1)
        self.assertEqual(decoder.status.channel_rate, 250e3)
        self.assertEqual(decoder.station.ps_name, "SDR DEMO")


class TestDecoderRdsInTheWindow(_WindowTestCase):
    demo = True

    def test_the_decoders_rds_decodes_the_receivers_fm_channel(self):
        # ROB-06: with the receiver on broadcast FM the Decoder's RDS
        # picked the same station out of the raw block a second time (8-14
        # ms of each 33 ms frame at 20 MS/s), where the background RDS
        # took 1.2 ms from the receiver's own channel.
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.win._decoder_panel._proto_combo.setCurrentText("RDS")
        self.settle(1)
        decoder = self.win._decoder
        own = mock.Mock(wraps=decoder._channelizer.process)
        decoder._channelizer.process = own
        for _ in range(300):  # (up to 10 s of the demo station)
            self.win._process_block(self.win._device.read_samples(80000))
            if decoder.station.ps_complete:
                break
        self.assertTrue(decoder.station.ps_complete)
        self.assertEqual(own.call_count, 0)
        self.assertTrue(self.win._decoder_panel.count_for("RDS"))
        # A narrow FM channel can't carry the station's RDS: the decoder
        # picks the station out of the raw block itself again.
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.win._process_block(self.win._device.read_samples(80000))
        self.assertEqual(own.call_count, 1)


# ---------------------------------------------------------------------------
# DEC-E2E-03: a decoder that gets only part of the signal says so
# ---------------------------------------------------------------------------


def _hardware_case():
    from tests.test_review_mainloop import _HardwareCase

    return _HardwareCase


class TestDecoderSaysWhenSignalIsSkipped(_hardware_case()):
    """A HackRF at 20 MS/s on a computer whose frames take 1.5x real time
    (the pacer reads one transfer a frame; the driver drops the rest)."""

    def _decode(self, protocol):
        self.win._set_demod_mode("None (I/Q)", bandwidth="2 MHz")
        self.win._decoder_panel._proto_combo.setCurrentText(protocol)
        self.settle(1)
        self.win._decoder_panel.set_receiving(True)

    def test_the_decoder_status_says_it_misses_part_of_the_signal(self):
        # ADS-B's Tune to puts the receiver in None (I/Q): no speaker, so
        # the "can't keep up" hint never came, and the status line stayed
        # green while 72 % of the samples were dropped.
        self._decode("ADS-B")
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)
        panel = self.win._decoder_panel
        line = panel.decoder_status_text()
        self.assertIn("misses part of the signal", line)
        self.assertIn("20 MS/s", line)
        self.assertIn("Sample rate", line)
        self.assertEqual(panel._status_label.property("tone"), "warning")
        self.assertIn("misses part of the signal", self.win._message_label.text())
        # The computer keeps up again: the line is the decoder's own, and
        # the status bar no longer says it can't keep up.
        self.LOAD = 0.3
        self.run_frames(300)
        self.assertFalse(self.win._pacer.overloaded)
        self.assertNotIn("misses", panel.decoder_status_text())
        self.assertNotIn("misses", self.win._message_label.text())

    def test_no_decoder_or_decoding_paused_says_nothing_of_it(self):
        self._decode("ADS-B")
        self.win._decoder_panel._enabled_check.setChecked(False)
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)
        self.assertNotIn("misses", self.win._decoder_panel.decoder_status_text())
        self.assertNotIn("misses", self.win._message_label.text())

    def _lines_while(self, frames, per_frame=5):
        """The Decoder's status lines over ``frames`` display frames."""
        lines = []
        for _ in range(frames):
            self.run_frames(1, per_frame)
            line = self.win._decoder_panel.decoder_status_text()
            if not lines or lines[-1] != line:
                lines.append(line)
        return lines

    def test_a_computer_catching_up_is_not_blamed_on_the_device(self):
        # Once the computer keeps up again, the two seconds real time is
        # judged by still hold the one-transfer frames: the line went on
        # with "The device isn't delivering 20 MS/s in real time" for a
        # few seconds, blaming the HackRF for what this computer dropped.
        self._decode("ADS-B")
        self.run_frames(40)
        self.assertTrue(self.win._pacer.overloaded)
        self.LOAD = 0.3
        lines = self._lines_while(300)
        self.assertFalse(self.win._pacer.overloaded)
        skipping = [line for line in lines if "misses part of the signal" in line]
        self.assertTrue(skipping)
        for line in skipping:
            self.assertIn("This computer can't process 20 MS/s", line)
        self.assertNotIn("misses", lines[-1])

    def test_a_device_short_of_its_rate_is_named(self):
        # A computer that keeps up, but the device delivers a fifth of its
        # sample rate (one transfer a frame): that is the device.
        self.LOAD = 0.3
        self._decode("ADS-B")
        lines = self._lines_while(120, per_frame=1)
        self.assertFalse(self.win._pacer.overloaded)
        self.assertIn(
            "The device isn't delivering 20 MS/s in real time, so the ADS-B "
            "decoder misses part of the signal. A lower sample rate "
            "(Receiver > Sample rate) helps.",
            lines,
        )


class TestFrameSpectrumNeverWaitsForABusyHelper(unittest.TestCase):
    def test_the_frame_is_done_while_the_helper_cannot_run(self):
        # On a busy computer the helper thread may not run for a while; the
        # frame waited for it (its half of the FFTs), so frames took longer
        # in wall time than their work and the window judged that it can't
        # keep up. The caller now does the helper's half itself then.
        import threading

        from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer

        data = _noise(1 << 18) + _tone(1 << 18, 1e6, RATE, 0.1)
        want = FrameSpectrumAnalyzer(2048, threads=1).process(data)
        analyzer = FrameSpectrumAnalyzer(2048, threads=2)
        analyzer.process(_noise(1 << 18))  # (starts the helper thread)
        gate = threading.Event()
        self.addCleanup(analyzer.close)
        self.addCleanup(gate.set)
        analyzer._pool.submit(gate.wait, 30)  # the helper is busy elsewhere
        result = []
        worker = threading.Thread(target=lambda: result.append(analyzer.process(data)))
        worker.start()
        worker.join(10)
        self.assertFalse(worker.is_alive(), "the frame waited for the helper")
        np.testing.assert_allclose(result[0].mean_db, want.mean_db, atol=1e-4)
        np.testing.assert_allclose(result[0].max_db, want.max_db, atol=1e-4)
        gate.set()


# ---------------------------------------------------------------------------
# ROB-14: the demo device's first frames at a new rate or in a new band
# ---------------------------------------------------------------------------


def _largest_prime_factor(n):
    n, factor, largest = int(n), 2, 1
    while factor * factor <= n:
        while n % factor == 0:
            largest, n = factor, n // factor
        factor += 1
    return max(largest, n)


class TestDemoLoopsAreQuickToBuild(unittest.TestCase):
    def test_the_music_loop_has_an_fft_friendly_length_at_every_rate(self):
        # At 3.2 and 1.024 MS/s the demo tune's loop was 382,090 and
        # 390,095 samples long (prime factors 2011 and 1279): its 75 us
        # pre-emphasis (an FFT around the loop) took 200 ms a channel, and
        # the first frame at the new rate stalled the window for 0.4-0.5 s.
        from sdr_module.gui import device_dialog as dd

        for rate in (1.024e6, 2.048e6, 2.4e6, 3.2e6, 8e6, 20e6):
            program = dd._loops("wfm", dd._MUSIC_SEED, rate)[0]
            self.assertLessEqual(_largest_prime_factor(len(program.values)), 7, rate)
            am = dd._loops("amb", 0, rate)
            self.assertLessEqual(_largest_prime_factor(len(am.values)), 7, rate)

    def test_the_am_stations_reuse_the_fm_stations_tune(self):
        # (Built once per loop rate: the AM band's stations took 0.08-0.45 s
        # to build the same tune again.)
        from sdr_module.gui import device_dialog as dd

        rate = 1.5e6
        with (
            mock.patch.dict(dd._LOOP_CACHE, clear=True),
            mock.patch.dict(dd._AM_MUSIC, clear=True),
            mock.patch.dict(dd._MUSIC_CACHE, clear=True),
        ):
            dd._loops("wfm", dd._MUSIC_SEED, rate)
            with mock.patch.object(dd, "_music_tune", wraps=dd._music_tune) as tune:
                dd._loops("amb", 1, rate)
        self.assertEqual(tune.call_count, 0)


# ---------------------------------------------------------------------------
# DEV-E2E-04: the Second Receiver's plots while nobody sees them
# ---------------------------------------------------------------------------


class TestSecondReceiverPlotsOnlyWhenSeen(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.rx = self.win._second_rx.show()
        self.rx._status_timer.stop()
        self.settle()

    def tearDown(self):
        rx = self.win._second_rx.receiver
        if rx is not None:
            rx.shutdown()
        super().tearDown()

    def test_a_minimized_second_receiver_skips_its_plots_not_its_audio(self):
        # Two demo receivers share one GUI thread (DEV-E2E-04): the second
        # one's plots cost a quarter of its frame even with its window
        # minimized. It still demodulates (Listen) then.
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        self.settle()
        analyzed = []
        process = rx._frame_spectrum.process
        rx._frame_spectrum.process = lambda s: analyzed.append(len(s)) or process(s)
        heard = []
        dsp_process = rx._dsp.process
        rx._dsp.process = lambda s, play: heard.append(len(s)) or dsp_process(s, play)
        block = rx.device.read_samples(80_000)
        rx._process(block)
        self.assertEqual(analyzed, [80_000])
        rx.showMinimized()
        self.settle()
        rx._process(block)
        self.assertEqual(analyzed, [80_000])  # no plots
        self.assertEqual(heard, [80_000, 80_000])  # but the receiver runs
        rx.showNormal()
        self.settle()
        rx._process(block)
        self.assertEqual(analyzed, [80_000, 80_000])


# ---------------------------------------------------------------------------
# The spectrum's trace drawing (ROB-06's 5-6 ms of paints a frame)
# ---------------------------------------------------------------------------


class TestTraceDrawing(_WindowTestCase):
    def _trace(self):
        rng = np.random.default_rng(8)
        spectrum = self.win._spectrum
        spectrum.update_spectrum(
            (-90 + 5 * rng.standard_normal(2048)).astype(np.float32)
        )
        return spectrum

    def test_the_trace_is_not_built_point_by_point(self):
        # The trace was built as one Python QPointF per pixel column, each
        # frame (and again for peak and min hold): about a quarter of the
        # spectrum's paint.
        from sdr_module.gui import spectrum_widget as sw

        spectrum = self._trace()
        made = []
        point = sw.QPointF

        def counting(*args):
            made.append(args)
            return point(*args)

        with mock.patch.object(sw, "QPointF", counting):
            spectrum._canvas.repaint()
        # (The axes' ticks and labels make a few dozen; the trace made 900.)
        self.assertLess(len(made), 200)

    def test_the_trace_is_drawn_exactly_as_before(self):
        from sdr_module.gui import spectrum_widget as sw

        spectrum = self._trace()
        spectrum._show_peak = True
        spectrum._peak_hold = spectrum._spectrum + 6.0
        fast = spectrum._canvas.grab().toImage()

        def slow(xs, ys):  # (a QPointF per point, as it was)
            return sw.QPolygonF(
                [sw.QPointF(x, y) for x, y in zip(list(xs), list(ys), strict=True)]
            )

        with mock.patch.object(sw, "polygon_of", slow):
            reference = spectrum._canvas.grab().toImage()
        self.assertEqual(fast, reference)

    def test_polygons_of_any_length(self):
        from sdr_module.gui.spectrum_widget import QPointF, polygon_of

        for n in (0, 1, 2, 7, 1300):
            xs = np.linspace(-3.5, 1000.25, n)
            ys = np.cos(xs)
            poly = polygon_of(xs, ys)
            self.assertEqual(poly.count(), n)
            for i in range(n):
                self.assertEqual(poly.at(i), QPointF(float(xs[i]), float(ys[i])))


# ---------------------------------------------------------------------------
# LISTEN-06: Remove DC spike and Balance I/Q at 20 MS/s
# ---------------------------------------------------------------------------


def _iq_reference(x, rate, conditioner_state, remove_dc, balance):
    """What the conditioner computed before (whole-block temporaries)."""
    from sdr_module.gui.audio_chain import correct_iq

    start, end, gain, phase = conditioner_state
    n = len(x)
    y = x
    if remove_dc:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        y = y - (np.complex64(start) + np.complex64(end - start) * ramp)
    if balance:
        y = correct_iq(y, gain, phase)
    return y


class TestIqConditionerAtHighRates(unittest.TestCase):
    @staticmethod
    def _blocks(count):
        for block in range(count):
            x = _noise(FRAME, seed=block) + np.complex64(0.02 - 0.01j * block)
            q = 1.05 * (x.imag * np.cos(0.03) + x.real * np.sin(0.03))
            yield (x.real + 1j * q).astype(np.complex64)

    def test_the_same_samples_as_before(self):
        from sdr_module.gui.audio_chain import IqConditioner

        for remove_dc, balance in ((True, True), (True, False), (False, True)):
            conditioner = IqConditioner()
            conditioner.remove_dc, conditioner.balance_iq = remove_dc, balance
            for block, x in enumerate(self._blocks(3)):
                dc_before = conditioner._dc
                got = conditioner.process(x, RATE)
                gain, phase = conditioner.imbalance
                start = dc_before if dc_before is not None else conditioner._dc
                want = _iq_reference(
                    x, RATE, (start, conditioner._dc, gain, phase), remove_dc, balance
                )
                self.assertEqual(got.dtype, np.complex64)
                np.testing.assert_allclose(
                    got, want, atol=1e-6, err_msg=str((remove_dc, balance, block))
                )

    def test_no_temporaries_the_size_of_the_block(self):
        # With Remove DC spike and Balance I/Q on, the conditioner made half
        # a dozen temporaries the size of the block (666,667 samples at 20
        # MS/s), the largest part of a frame with both on. It now works in
        # pieces that stay in the processor's cache.
        import tracemalloc

        from sdr_module.gui.audio_chain import IqConditioner

        conditioner = IqConditioner()
        conditioner.remove_dc = conditioner.balance_iq = True
        first, second = self._blocks(2)
        conditioner.process(first, RATE)  # (the estimates start)
        tracemalloc.start()
        try:
            out = conditioner.process(second, RATE)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 1.5 * out.nbytes)


# ---------------------------------------------------------------------------
# REC-E2E-05 (part 4): a recording playing back on a computer that can't
# keep up, and the decoder hint while one plays
# ---------------------------------------------------------------------------


class _Clock:
    """The ``time`` module of main_window, run by hand."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)

    def __getattr__(self, name):  # strftime, time...
        import time

        return getattr(time, name)


class TestSlowRecordingRidesOutHiccups(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_a_late_frame_is_made_up_in_full(self):
        # A 48 kHz recording's frames (1,600 samples) are never timed, so
        # each could take only the probe limit (50 ms of signal) for good,
        # and what was left over a block was dropped: every 90 ms hiccup
        # of the window cost ~30 ms of the recording's real time (audio).
        mw = _mw()
        pacer = mw._Pacer()
        rate, now, read = 48e3, 100.0, 0
        pacer.paced_request(rate, now)
        start = now
        for frame in range(90):
            now += 0.033 + (0.09 if frame % 8 == 7 else 0.0)
            read += pacer.paced_request(rate, now)
        self.assertAlmostEqual(
            read / rate, now - start, delta=mw.DISPLAY_BLOCK / rate + 1e-3
        )


def _playback_case():
    from tests.test_gui_playback import _PlaybackCase

    return _PlaybackCase


class TestPlaybackWhileOverloaded(_playback_case()):
    """A 1 MS/s recording (looped) on a fake clock; processing costs
    ``load`` times the time its samples cover at 1x."""

    PLAY_RATE = 1e6

    def setUp(self):
        super().setUp()
        from tests.test_gui_playback import _fm_tone

        x = _fm_tone(1.0, rate=self.PLAY_RATE)
        self.play(self.make_file(x, name="p.sigmf-data", rate=self.PLAY_RATE))
        self.win._playback_bar.set_loop(True)
        self.assertTrue(self.win._playback.active)
        self.clock = _Clock()
        patcher = mock.patch.object(self.mw, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.load = 0.0
        self.read_wait = 0.0
        self.played = 0
        process = self.win._process_block

        def slow_process(samples):
            self.played += len(samples)
            self.clock.advance(self.load * len(samples) / self.PLAY_RATE)
            process(samples)

        self.win._process_block = slow_process
        device = self.win._device
        read = device.read_samples

        def slow_read(count):
            if count:
                self.clock.advance(self.read_wait)  # (waiting for the disk)
            return read(count)

        device.read_samples = slow_read

    def run_frames(self, seconds):
        """Seconds of the recording played in ``seconds`` of display frames
        (one each 33 ms, or as soon as the last one ended)."""
        win = self.win
        win._pacer.restart()
        win._update_display()
        self.played = 0
        start = self.clock.now
        while self.clock.now - start < seconds:
            tick = self.clock.now
            win._update_display()
            self.clock.now = max(self.clock.now, tick + 0.033)
        return self.played / self.PLAY_RATE / (self.clock.now - start)

    def test_too_slow_for_4x_still_plays_in_real_time(self):
        # 4x needs 1.2x real time here: frames can't keep up. One block a
        # frame (and a probe a second) played the recording at about 0.5x
        # (0.3x in the window), though 1x was in reach.
        self.win._playback_bar.set_speed(4.0)
        self.load = 0.3
        speed = self.run_frames(4.0)
        self.assertTrue(self.win._pacer.overloaded)
        self.assertGreater(speed, 0.95)
        self.assertLess(speed, 2.0)

    def test_too_slow_for_1x_plays_as_fast_as_it_can(self):
        # Frames can't keep up even at 1x (1.25x real time): the recording
        # plays as fast as this computer manages (it played at 0.17x, a
        # block a frame).
        self.load = 1.25
        speed = self.run_frames(4.0)
        self.assertTrue(self.win._pacer.overloaded)
        self.assertGreater(speed, 0.6)
        self.assertLessEqual(speed, 1.0)

    def test_waiting_for_the_disk_isnt_a_computer_that_cant_keep_up(self):
        # A file read cold waits for the disk (60 ms a frame here) with the
        # processor idle: frames judged by the clock looked overloaded, and
        # playback dropped to a block a frame.
        self.load = 0.1
        self.read_wait = 0.06
        speed = self.run_frames(4.0)
        self.assertFalse(self.win._pacer.overloaded)
        self.assertGreater(speed, 0.9)

    def test_the_decoder_doesnt_say_a_recording_is_missed(self):
        # A recording is read in full however slowly it plays: the decoder
        # hears all of it. Its line said it "misses part of the signal" and
        # sent the user to Receiver > Sample rate, refused while a file
        # plays, under the status bar's "Playing at about 0.3x, not 4x".
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("POCSAG")
        self.settle(1)
        win._playback_bar.set_speed(4.0)
        self.load = 0.3
        self.run_frames(3.0)
        self.assertTrue(win._pacer.overloaded)
        panel = win._decoder_panel
        self.assertTrue(panel.decoder_status_text())
        self.assertNotIn("misses part of the signal", panel.decoder_status_text())
        self.assertNotIn("misses part of the signal", win._message_label.text())
