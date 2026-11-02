"""J-decode: a FLEX page's phase and speed show in the Decoder's window.

Since FLEX at 3200 and 6400 bps decodes in every phase (A to D), pages
arrive from different phases and speeds, but neither showed anywhere: the
Messages table, the details card and Export gave the capcode alone
(FLEXMessage.phase and baud_rate carried them). Now the address reads
like POCSAG's ("1234567 (F3)"): "1234567 (C, 6400 bps)", which the Log
and Export repeat, and the details card says the phase, speed, frame and
cycle and what kind of page it is.
"""

import csv
import io
import unittest

import numpy as np
import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import (
    FLEXLiveDecoder,
    describe_message,
    message_details,
)
from sdr_module.dsp.protocols import FLEXMessage, ProtocolType
from tests.test_gui_chrome import _WindowTestCase

RATE = 48000.0
SPEEDS = [(1600, 2), (3200, 2), (3200, 4), (6400, 4)]


def _page(**fields):
    base = dict(protocol=ProtocolType.FLEX, timestamp=0.0, raw_bits=b"", valid=True)
    return FLEXMessage(**{**base, **fields})


def test_the_row_names_the_phase_and_speed():
    page = _page(capcode=1234567, phase="C", baud_rate=6400, content="Hello")
    assert describe_message(page) == ("1234567 (C, 6400 bps)", "Hello")
    first = _page(capcode=1000123, phase="A", baud_rate=1600, content="Hi")
    assert describe_message(first) == ("1000123 (A, 1600 bps)", "Hi")
    # (Without a phase, the speed alone.)
    assert describe_message(_page(capcode=12, content="X")) == ("12 (1600 bps)", "X")


def test_the_details_say_phase_speed_frame_and_kind():
    page = _page(
        capcode=1000456,
        cycle=3,
        frame=18,
        phase="B",
        baud_rate=3200,
        message_type="alpha",
        content="Call the office",
    )
    assert message_details(page) == [
        "Phase B at 3200 bps, frame 18 of cycle 3",
        "Alphanumeric page",
    ]
    numeric = _page(
        capcode=7, frame=5, phase="D", baud_rate=6400, message_type="numeric"
    )
    assert message_details(numeric) == [
        "Phase D at 6400 bps, frame 5 of cycle 0",
        "Numeric page",
    ]
    tone = _page(capcode=7, phase="A", message_type="tone")
    assert message_details(tone)[1] == "Tone-only page (no message)"


@pytest.mark.parametrize("speed", SPEEDS)
def test_every_phase_and_speed_reaches_the_row(speed):
    phases = enc.FLEX_SPEEDS[speed][2]
    pages = {p: [(1000000 + ord(p), f"Phase {p} at {speed[0]}")] for p in phases}
    f = enc.flex_frequency(pages, RATE, 3, 17, speed=speed)
    x = enc.fm_modulate(np.concatenate([np.zeros(4800), f, np.zeros(4800)]), RATE)
    decoder = FLEXLiveDecoder(RATE)
    rows = []
    block = int(RATE * 0.034)
    for start in range(0, len(x), block):
        rows += [describe_message(m) for m in decoder.process(x[start : start + block])]
    assert sorted(rows) == sorted(
        (f"{1000000 + ord(p)} ({p}, {speed[0]} bps)", f"Phase {p} at {speed[0]}")
        for p in phases
    )


class TestTheWindowShowsThem(_WindowTestCase):
    def test_table_details_card_and_export(self):
        win = self.win
        panel = win._decoder_panel
        page = _page(
            capcode=1000999,
            cycle=2,
            frame=18,
            phase="D",
            baud_rate=6400,
            message_type="alpha",
            content="Phase D: 4-level FSK at 3200 baud",
        )
        win._push_decoded_message(page)
        row = panel._table.rowCount() - 1
        from sdr_module.gui.decoder_panel import COL_ADDRESS

        self.assertEqual(
            panel._table.item(row, COL_ADDRESS).text(), "1000999 (D, 6400 bps)"
        )
        panel._table.selectRow(row)
        panel._update_detail()
        card = panel._detail
        self.assertIn("1000999 (D, 6400 bps)", card.meta.text())
        self.assertEqual(
            card.details.text(),
            "Phase D at 6400 bps, frame 18 of cycle 2\nAlphanumeric page",
        )
        out = io.BytesIO()
        panel._spool.copy_rows(out, panel._spool.snapshot())
        rows = list(csv.reader(io.StringIO(out.getvalue().decode("utf-8"))))
        self.assertEqual(rows[-1][3], "1000999 (D, 6400 bps)")


if __name__ == "__main__":
    unittest.main()
