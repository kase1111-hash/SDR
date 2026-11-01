#!/usr/bin/env python3
"""
Tests for the plots' zoom model (``sdr_module.gui.plot_view``, no Qt):
zoom levels, anchors, staying inside the span, following the tuned
frequency, retunes and listeners.
"""

import gc
import unittest

from sdr_module.gui.plot_view import ZOOM_LEVELS, PlotView, normalize_zoom


class TestZoom(unittest.TestCase):
    def setUp(self):
        self.view = PlotView(100e6, 2.4e6)

    def test_full_span_at_x1(self):
        v = self.view
        self.assertEqual(v.zoom, 1)
        self.assertEqual((v.start, v.end), (98.8e6, 101.2e6))
        self.assertFalse(v.is_zoomed())

    def test_zoom_keeps_the_anchor_where_it_is(self):
        v = self.view
        anchor = 99.4e6  # a quarter of the way across
        frac = v.fraction_of(anchor)
        self.assertAlmostEqual(frac, 0.25)
        self.assertTrue(v.set_zoom(4, anchor, frac))
        self.assertEqual(v.span, 600e3)
        self.assertAlmostEqual(v.fraction_of(anchor), 0.25)
        self.assertAlmostEqual(v.frequency_at(0.25), anchor)

    def test_the_view_stays_inside_the_span(self):
        v = self.view
        v.set_zoom(8, 101.19e6, 0.5)  # centered near the top edge
        self.assertAlmostEqual(v.end, 101.2e6)
        self.assertAlmostEqual(v.span, 300e3)
        v.set_zoom(16, 98.0e6, 0.5)  # outside the span: clamped
        self.assertAlmostEqual(v.start, 98.8e6)

    def test_zoom_steps_and_limits(self):
        v = self.view
        for expected in (2, 4, 8, 16, 16):
            v.zoom_steps(1, 100e6)
            self.assertEqual(v.zoom, expected)
        self.assertFalse(v.zoom_steps(1, 100e6))
        for expected in (8, 4, 2, 1, 1):
            v.zoom_steps(-1)
            self.assertEqual(v.zoom, expected)
        self.assertEqual(v.center, 100e6)

    def test_normalize_zoom(self):
        self.assertEqual(normalize_zoom(3), 4)  # nearest power of two
        self.assertEqual(normalize_zoom("8"), 8)
        self.assertEqual(normalize_zoom(1000), 16)
        self.assertEqual(normalize_zoom(None), 1)
        self.assertEqual(normalize_zoom(float("nan")), 1)
        self.assertEqual(ZOOM_LEVELS, (1, 2, 4, 8, 16))

    def test_follow_pans_to_a_frequency_out_of_sight(self):
        v = self.view
        v.set_zoom(4, 100e6, 0.5)  # 99.7-100.3 MHz
        self.assertFalse(v.follow(100.1e6))
        self.assertTrue(v.follow(100.9e6))
        self.assertTrue(v.contains(100.9e6))
        self.assertAlmostEqual(v.center, 100.9e6)
        self.assertFalse(v.follow(105e6))  # outside the span: nothing to do
        self.view.set_zoom(1)
        self.assertFalse(v.follow(100.9e6))  # not zoomed: never pans

    def test_a_retune_keeps_the_view_on_the_same_frequencies(self):
        v = self.view
        v.set_zoom(4, 100.5e6, 0.5)
        self.assertTrue(v.set_full(100.4e6, 2.4e6))
        self.assertAlmostEqual(v.center, 100.5e6)
        v.set_full(110e6, 2.4e6)  # far away: clamped into the new span
        self.assertTrue(v.in_span(v.center))
        self.assertGreaterEqual(v.start, 110e6 - 1.2e6 - 1e-6)
        v.set_zoom(1)
        v.set_full(120e6, 1e6)
        self.assertEqual((v.center, v.span), (120e6, 1e6))

    def test_listeners_are_told_and_dropped_with_their_owner(self):
        calls = []

        class Owner:
            def changed(self):
                calls.append(1)

        owner = Owner()
        self.view.subscribe(owner.changed)
        self.view.set_zoom(2)
        self.assertEqual(len(calls), 1)
        self.view.set_zoom(2)  # no change, no call
        self.assertEqual(len(calls), 1)
        del owner
        gc.collect()
        self.view.set_zoom(4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.view._listeners, [])

    def test_describe(self):
        self.view.set_zoom(4)
        self.assertEqual(self.view.describe(), "×4 (600 kHz)")


if __name__ == "__main__":
    unittest.main()
