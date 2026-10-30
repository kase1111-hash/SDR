#!/usr/bin/env python3
"""
Tuning helpers without a window: the NCO that moves the tuned channel to
0 Hz, reading typed frequencies, tuning steps, click-to-tune snapping and
the tuning history (src/sdr_module/dsp/nco.py, src/sdr_module/gui/tuning.py).
"""

import json
import math
import unittest

import numpy as np

from sdr_module.dsp.nco import TABLE_SIZE, Nco
from sdr_module.gui.tuning import (
    ACCEPTABLE,
    INTERMEDIATE,
    INVALID,
    STEP_CHOICES_HZ,
    STEP_MODE_DEFAULTS,
    TunedEntry,
    TuningHistory,
    default_step_hz,
    format_mhz_range,
    format_step,
    nearest_step,
    parse_frequency_text,
    snap_click_frequency,
    step_frequency,
    step_index,
    step_mode_key,
)

RATE = 2.4e6


def _noise(n, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


class TestNco(unittest.TestCase):
    def test_matches_an_exact_mix(self):
        x = _noise(50_001)
        for freq in (123456.7, -987654.3, 1.0, 1.1e6):
            exact = x.astype(np.complex128) * np.exp(
                -2j * np.pi * freq * np.arange(len(x)) / RATE
            )
            out = Nco(RATE, freq).mix(x)
            self.assertEqual(out.dtype, np.complex64)
            self.assertLess(np.max(np.abs(out - exact)), 1e-5, freq)

    def test_blocks_join_seamlessly(self):
        x = _noise(3 * TABLE_SIZE + 777)
        whole = Nco(RATE, 312.5e3).mix(x)
        nco = Nco(RATE, 312.5e3)
        cuts = [0, 1, TABLE_SIZE - 1, TABLE_SIZE, TABLE_SIZE + 1, 9000, len(x)]
        parts = np.concatenate(
            [nco.mix(x[a:b]) for a, b in zip(cuts, cuts[1:], strict=False)]
        )
        np.testing.assert_allclose(parts, whole, atol=2e-6)

    def test_a_signal_at_the_offset_comes_out_at_zero_hertz(self):
        n = 20_000
        tone = np.exp(2j * np.pi * 400e3 * np.arange(n) / RATE).astype(np.complex64)
        out = Nco(RATE, 400e3).mix(tone)
        # A constant phasor: no rotation left.
        self.assertLess(np.max(np.abs(np.diff(np.angle(out)))), 1e-4)

    def test_retuning_bends_the_phase_without_a_jump(self):
        nco = Nco(RATE, 10e3)
        ones = np.ones(5000, dtype=np.complex64)
        a = nco.mix(ones)
        nco.frequency_hz = 20e3
        b = nco.mix(ones)
        steps = np.diff(np.unwrap(np.angle(np.concatenate([a, b])))) / (2 * np.pi)
        # Per-sample phase steps of -10 kHz, then -20 kHz: no extra jump.
        hz = steps * RATE
        self.assertAlmostEqual(float(np.median(hz[:4990])), -10e3, delta=5)
        self.assertAlmostEqual(float(np.median(hz[5010:])), -20e3, delta=5)
        self.assertLess(abs(hz[4999] - hz[5000]), 10.5e3)

    def test_zero_shift_passes_samples_through(self):
        x = _noise(1000)
        self.assertIs(Nco(RATE, 0.0).mix(x), x)

    def test_moving_to_zero_hertz_and_back_keeps_the_phase(self):
        """At 0 Hz the phase stands still where it got to: returning the
        input unturned jumped the phase (a click in SSB/CW audio when
        fine-tuning across the LO, and on the first step after a recenter)."""
        ones = np.ones(TABLE_SIZE + 333, dtype=np.complex64)
        nco = Nco(RATE, 1234.5)
        parts = [nco.mix(ones)]
        for freq in (0.0, -10.0, 0.0, 777.0):
            nco.frequency_hz = freq
            parts.append(nco.mix(ones))
        out = np.concatenate(parts)
        exact = np.concatenate(
            [
                np.full(len(ones), f)
                for f in (1234.5, 0.0, -10.0, 0.0, 777.0)  # Hz, per part
            ]
        )
        phase = -2 * np.pi * np.cumsum(np.concatenate([[0.0], exact[:-1]])) / RATE
        np.testing.assert_allclose(out, np.exp(1j * phase), atol=2e-4)
        steps = np.abs(np.angle(out[1:] * np.conj(out[:-1])))
        self.assertLess(float(np.max(steps)), 2 * np.pi * 1300 / RATE)

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            Nco(0.0)
        nco = Nco(RATE)
        with self.assertRaises(ValueError):
            nco.frequency_hz = float("nan")


class TestParseFrequencyText(unittest.TestCase):
    MIN, MAX = 1e3, 6e9

    def parse(self, text, unit=1e6, **kw):
        kw.setdefault("grouped", unit in (1.0, 1e3))
        return parse_frequency_text(text, unit, self.MIN, self.MAX, **kw)

    def assertHz(self, text, hz, unit=1e6, **kw):  # noqa: N802 - unittest style
        result = self.parse(text, unit, **kw)
        self.assertEqual(result.state, ACCEPTABLE, (text, result))
        self.assertAlmostEqual(result.hz, hz, places=0, msg=text)
        return result

    def test_unit_suffixes_in_any_case_and_with_spaces(self):
        self.assertHz("433.92M", 433.92e6)
        self.assertHz("433.92 mhz", 433.92e6)
        self.assertHz("433.92 MHz", 433.92e6)
        self.assertHz("7074k", 7.074e6)
        self.assertHz("7074 KHZ", 7.074e6)
        self.assertHz("1.09G", 1.09e9)
        self.assertHz("1.09 ghz", 1.09e9)
        self.assertHz("145800000 Hz", 145.8e6)
        # A suffix wins over the field's unit.
        self.assertHz("146.52M", 146.52e6, unit=1e3)

    def test_e_notation(self):
        result = self.assertHz("7.074e6", 7.074e6)
        self.assertTrue(result.read_as_hz)  # 7.074e6 MHz can't be meant
        self.assertHz("1e3", 1e9)  # 1000 MHz: a plain value in the unit
        self.assertHz("1.4652E2", 146.52e6)

    def test_decimal_comma_and_digit_grouping(self):
        self.assertHz("144,39", 144.39e6)  # MHz shows no grouping
        self.assertHz("1,090.5", 1090.5e6)
        self.assertHz("145,800,000", 145.8e6, unit=1.0)
        self.assertHz("145'800'000", 145.8e6, unit=1.0)
        self.assertHz("7,074", 7.074e6, unit=1e3)  # grouped kHz
        self.assertHz("7,074 kHz", 7.074e6)  # a kHz suffix groups too
        # A German locale: '.' groups, ',' is the decimal point.
        self.assertHz("7.074", 7.074e6, unit=1e3, group_separator=".")
        self.assertHz("100.100,5", 100100.5e3, unit=1e3, group_separator=".")

    def test_a_bare_number_in_hz_typed_into_the_mhz_field(self):
        result = self.assertHz("162550000", 162.55e6)
        self.assertTrue(result.read_as_hz)
        self.assertHz("2400000000", 2.4e9, unit=1e9)
        self.assertHz("162550000", 162.55e6, unit=1e3)  # 162.55 GHz in kHz

    def test_out_of_range_is_intermediate_and_explained(self):
        result = self.parse("7000")
        self.assertEqual(result.state, INTERMEDIATE)  # digits are kept
        self.assertEqual(result.hz, 7e9)
        self.assertIn("7000 MHz is out of range", result.problem)
        self.assertIn("0.001–6000 MHz", result.problem)
        self.assertIn("For 7000 kHz, type 7000k.", result.problem)
        self.assertEqual(self.parse("16255").state, INTERMEDIATE)
        low = self.parse("100 Hz")
        self.assertEqual(low.state, INTERMEDIATE)
        self.assertIn("100 Hz is out of range", low.problem)

    def test_the_device_range_names_the_device(self):
        result = parse_frequency_text(
            "2400",
            1e6,
            500e3,
            1.766e9,
            range_text="RTL-SDR tunes 0.5–1766 MHz",
        )
        self.assertEqual(result.state, INTERMEDIATE)
        self.assertEqual(
            result.problem, "2400 MHz is out of range: RTL-SDR tunes 0.5–1766 MHz."
        )

    def test_unfinished_text_is_intermediate(self):
        for text in ("", " ", ".", "7.074e", "7.074e+", "162.55 MH", "433 kh"):
            self.assertEqual(self.parse(text).state, INTERMEDIATE, text)
        self.assertIn("Can't read", self.parse("7.074e").problem)

    def test_characters_that_cant_belong_to_a_frequency_are_invalid(self):
        for text in ("-5", "+5", "abc", "12x", "1e6k", "1.2.3,4,5"):
            state = self.parse(text).state
            self.assertIn(state, (INVALID, INTERMEDIATE), text)
        for text in ("+5", "--5", "-abc", "abc", "12x"):
            self.assertEqual(self.parse(text).state, INVALID, text)

    def test_a_negative_number_is_kept_and_refused(self):
        """Dropping the minus (Invalid) left "5": Enter tuned 5 MHz."""
        result = self.parse("-5")
        self.assertEqual(result.state, INTERMEDIATE)
        self.assertEqual(result.hz, -5e6)  # a number: the field turns red
        self.assertEqual(result.problem, "A frequency can't be negative.")
        self.assertEqual(self.parse("-").state, INTERMEDIATE)

    def test_separators_that_dont_group_thousands_are_not_a_number(self):
        """A slip ("12..5") was read as grouping and tuned 125 MHz;
        "145.800.0" (on the way to 145.800.000) tuned 1.458 MHz."""
        for text in ("12..5", "1.2.3", "145.800.0", "..5", "5..", "1,2,3"):
            result = self.parse(text)
            self.assertEqual(result.state, INTERMEDIATE, text)
            self.assertIsNone(result.hz, text)
            self.assertIn("Can't read", result.problem)
        for text in ("12,34.5", "1.,5", "1,2.3,4"):
            self.assertEqual(self.parse(text).state, INTERMEDIATE, text)
        # Well-formed grouping still reads.
        self.assertHz("145.800.000", 145.8e6)
        self.assertHz("1,234.5", 1234.5e6)
        self.assertHz("1.234,5", 1234.5e6)

    def test_the_smaller_unit_hint_needs_whole_hertz(self):
        self.assertIn("For 7000 kHz, type 7000k.", self.parse("7000").problem)
        problem = self.parse("6000.000001").problem  # 1 Hz above 6 GHz
        self.assertIn("6000.000001 MHz is out of range", problem)
        self.assertNotIn("For ", problem)

    def test_range_text(self):
        self.assertEqual(format_mhz_range(500e3, 1.766e9), "0.5–1766 MHz")
        self.assertEqual(format_mhz_range(1e6, 6e9), "1–6000 MHz")


class TestSteps(unittest.TestCase):
    def test_the_step_choices(self):
        self.assertEqual(
            [format_step(s) for s in STEP_CHOICES_HZ],
            [
                "10 Hz",
                "50 Hz",
                "100 Hz",
                "500 Hz",
                "1 kHz",
                "2.5 kHz",
                "5 kHz",
                "6.25 kHz",
                "8.33 kHz",
                "9 kHz",
                "10 kHz",
                "12.5 kHz",
                "25 kHz",
                "50 kHz",
                "100 kHz",
                "200 kHz",
                "1 MHz",
            ],
        )

    def test_mode_keys_and_defaults(self):
        self.assertEqual(step_mode_key("FM", 75e3), "wfm")
        self.assertEqual(step_mode_key("FM", 5e3), "nfm")
        self.assertEqual(step_mode_key("AM"), "am")
        self.assertEqual(step_mode_key("USB"), "usb")
        self.assertEqual(step_mode_key("LSB"), "lsb")
        self.assertEqual(step_mode_key("CW"), "cw")
        self.assertEqual(step_mode_key("None (I/Q)"), "iq")
        expected = {
            "wfm": 100e3,
            "nfm": 12.5e3,
            "am": 10e3,
            "usb": 100.0,
            "lsb": 100.0,
            "cw": 10.0,
            "iq": 10e3,
        }
        self.assertEqual(STEP_MODE_DEFAULTS, expected)
        for key, step in expected.items():
            self.assertEqual(default_step_hz(key), step)
            self.assertIn(step, STEP_CHOICES_HZ)

    def test_nearest_step_and_index(self):
        self.assertEqual(nearest_step(8333), 25e3 / 3)
        self.assertEqual(nearest_step(12e3), 12.5e3)
        self.assertEqual(nearest_step("junk"), 10e3)
        self.assertEqual(nearest_step(-5), 10e3)
        self.assertEqual(step_index(10.0), 0)
        self.assertEqual(step_index(1e6), len(STEP_CHOICES_HZ) - 1)

    def test_stepping_keeps_an_offset_but_not_rounding_drift(self):
        self.assertEqual(step_frequency(145.800001e6, 10e3, 1), 145.810001e6)
        self.assertEqual(step_frequency(100.1e6, 100e3, -10), 99.1e6)
        self.assertEqual(step_frequency(100.1e6, 100e3, 0), 100.1e6)
        # 8.33 kHz steps, rounded to whole Hz by the field each time.
        freq = 118e6
        for _ in range(3):
            freq = round(step_frequency(freq, 25e3 / 3, 1))
        self.assertEqual(freq, 118.025e6)
        for _ in range(3):
            freq = round(step_frequency(freq, 25e3 / 3, -1))
        self.assertEqual(freq, 118e6)


class TestSnapClick(unittest.TestCase):
    N = 2048
    CENTER = 100e6
    START = CENTER - RATE / 2
    HZ_PER_PX = RATE / 1000  # a 1000 px plot

    def spectrum(self, signals=(), seed=3):
        """Noise at -90 dB plus (offset, width_hz, level_db) signals."""
        rng = np.random.default_rng(seed)
        power = -90.0 + 3.0 * rng.standard_normal(self.N)
        freqs = self.START + np.arange(self.N) * RATE / self.N
        for offset, width, level in signals:
            f0 = self.CENTER + offset
            if width <= 0:  # a carrier with a Hann main lobe
                x = (freqs - f0) / (RATE / self.N)
                lobe = np.where(
                    np.abs(x) < 2, 20 * np.log10(np.abs(np.sinc(x / 2)) + 1e-9), -200
                )
                power = np.maximum(power, level + lobe)
            else:
                inside = np.abs(freqs - f0) <= width / 2
                ripple = 2.0 * rng.standard_normal(self.N)
                power = np.where(inside, np.maximum(power, level + ripple), power)
        return power

    def snap(self, click_offset, power, step, mode=""):
        return snap_click_frequency(
            self.CENTER + click_offset,
            power,
            self.START,
            RATE,
            self.HZ_PER_PX,
            step,
            mode,
        )

    def test_a_click_near_a_broadcast_station_lands_on_its_channel(self):
        power = self.spectrum([(500e3, 180e3, -40.0)])
        self.assertEqual(self.snap(510e3, power, 100e3, "wfm"), 100.5e6)
        self.assertEqual(self.snap(490e3, power, 100e3, "wfm"), 100.5e6)

    def test_a_carrier_is_found_to_well_inside_a_cw_filter(self):
        true = 123_456.0
        power = self.spectrum([(true, 0, -30.0)])
        landed = self.snap(true + 4e3, power, 10.0, "cw")
        self.assertLess(abs(landed - (self.CENTER + true)), 150.0)
        self.assertAlmostEqual(landed / 10.0, round(landed / 10.0))

    def test_a_signal_off_the_step_raster_is_not_dragged_onto_it(self):
        # 2 m simplex on a 20 kHz raster: 100.520 MHz here, 12.5 kHz step.
        power = self.spectrum([(520e3, 0, -35.0)])
        landed = self.snap(522e3, power, 12.5e3, "nfm")
        self.assertLess(abs(landed - 100.52e6), 1.2e3)
        self.assertNotEqual(landed, 100.525e6)

    def test_empty_spectrum_goes_to_the_raster_or_the_pixel(self):
        power = self.spectrum()
        self.assertEqual(self.snap(234_567.0, power, 100e3), 100.2e6)
        landed = self.snap(234_567.0, power, 10.0)
        self.assertEqual(landed, 100.235e6)  # the pixel's 1 kHz resolution
        self.assertEqual(self.snap(234_567.0, None, 25e3), 100.225e6)

    def test_sideband_modes_dont_move_onto_the_signal(self):
        power = self.spectrum([(301_500.0, 2.4e3, -40.0)])
        landed = self.snap(300e3, power, 100.0, "usb")
        self.assertEqual(landed, 100.3e6)

    def test_noise_alone_rarely_counts_as_a_signal(self):
        """FFT noise bins have exponentially distributed power: a click on
        an empty stretch must not hop to a noise blip (then off the raster)."""
        rng = np.random.default_rng(11)
        trials, hops = 400, 0
        for _ in range(trials):
            power = 10 * np.log10(rng.exponential(1.0, self.N)) - 90.0
            click = self.CENTER + rng.uniform(-1e6, 1e6)
            landed = snap_click_frequency(
                click, power, self.START, RATE, self.HZ_PER_PX, 12.5e3, "nfm"
            )
            hops += landed != round(click / 12.5e3) * 12.5e3
        self.assertLess(hops / trials, 0.05)

    def test_bad_spectrum_data_is_ignored(self):
        power = self.spectrum([(500e3, 180e3, -40.0)])
        power[10] = np.nan
        self.assertEqual(self.snap(510e3, power, 100e3), 100.5e6)  # the raster


class TestTuningHistory(unittest.TestCase):
    def test_back_and_forward_like_a_browser(self):
        h = TuningHistory()
        for freq in (100e6, 101e6, 102e6):
            h.visit(TunedEntry(freq, "FM"))
        self.assertEqual(h.back().freq_hz, 101e6)
        self.assertEqual(h.back().freq_hz, 100e6)
        self.assertIsNone(h.back())
        self.assertEqual(h.forward().freq_hz, 101e6)
        self.assertEqual(h.peek(1).freq_hz, 102e6)
        # Somewhere new after going back drops what was ahead.
        h.visit(TunedEntry(103e6, "AM"))
        self.assertFalse(h.can_go_forward())
        self.assertEqual(h.back().freq_hz, 101e6)

    def test_the_same_frequency_updates_its_mode(self):
        h = TuningHistory()
        h.visit(TunedEntry(100e6, "FM"))
        self.assertTrue(h.visit(TunedEntry(100e6, "AM")))
        self.assertEqual(h.current().mode, "AM")
        self.assertFalse(h.can_go_back())
        self.assertFalse(h.visit(TunedEntry(100e6 + 0.2, "AM")))

    def test_recent_frequencies_are_unique_most_recent_first_and_capped(self):
        h = TuningHistory(recent_limit=20)
        for i in range(25):
            h.visit(TunedEntry(100e6 + i * 1e3, "FM"))
        h.visit(TunedEntry(100.005e6, "AM"))
        recent = h.recent()
        self.assertEqual(len(recent), 20)
        self.assertEqual(recent[0], TunedEntry(100.005e6, "AM"))
        self.assertEqual(len({e.freq_hz for e in recent}), 20)

    def test_saved_and_restored(self):
        h = TuningHistory()
        h.visit(TunedEntry(146.52e6, "FM", "5 kHz"))
        h.visit(TunedEntry(7.074e6, "USB"))
        raw = h.to_json()
        self.assertEqual(json.loads(raw)[0]["freq_hz"], 7.074e6)
        again = TuningHistory(TuningHistory.entries_from_json(raw))
        self.assertEqual(again.recent(), h.recent())
        # After a restart Back goes to the frequency before the last one.
        self.assertEqual(again.current().freq_hz, 7.074e6)
        self.assertEqual(again.back(), TunedEntry(146.52e6, "FM", "5 kHz"))
        for bad in ("", "nope", "{}", '[{"freq_hz": "x"}, 5, {"freq_hz": -1}]'):
            self.assertEqual(TuningHistory.entries_from_json(bad), [])

    def test_clear_keeps_where_you_are(self):
        h = TuningHistory()
        h.visit(TunedEntry(100e6))
        h.visit(TunedEntry(101e6))
        h.clear_recent()
        self.assertEqual(h.recent(), [])
        self.assertFalse(h.can_go_back())
        self.assertEqual(h.current().freq_hz, 101e6)
        self.assertTrue(math.isfinite(h.current().freq_hz))


if __name__ == "__main__":
    unittest.main()
