"""Two demo devices at once, at two sample rates (K-X1).

A wideband scan on two demo devices, or Device > Second Receiver... on a
second demo device, runs two ``MockDevice`` at once, often at different
sample rates (the main window's 2.4 MS/s, the second receiver's 3.2 MS/s).
Before, the demo kept its SSTV transmission for one sample rate at a time
and its modulation loops for 24 signals in all (one rate needs 22): two
demo devices at 2.4 and 3.2 MS/s near 145.800 MHz rebuilt the SSTV
transmission (0.1 s, 3.6 MB) on every read, 1.26 us a sample instead of
about 0.01, and two sweeping every demo station rebuilt 166 loops in three
passes instead of 44. Work is counted here, not timed.
"""

from unittest import mock

import pytest

from sdr_module.gui import device_dialog as dd
from sdr_module.gui.device_dialog import MockDevice


@pytest.fixture
def fresh_caches(monkeypatch):
    """Empty demo caches, and every sample read (no skipping ahead)."""
    monkeypatch.setattr(dd, "_MAX_LAG_S", 1e9)
    monkeypatch.setattr(dd, "_SSTV_CACHE", {})
    monkeypatch.setattr(dd, "_LOOP_CACHE", {})


def _device(rate, freq):
    device = MockDevice()
    device.set_sample_rate(rate)
    device.set_frequency(freq)
    device.start_rx()
    return device


def _frequencies():
    return sorted({freq for freq, *_rest in dd._DEMO_SIGNALS})


def test_two_rates_at_the_sstv_downlink_build_it_once_each(fresh_caches):
    import sdr_module.ham.sstv_encoder as encoder

    first = _device(2.4e6, 145.8e6)
    second = _device(3.2e6, 145.8e6)
    with mock.patch.object(encoder, "encode_sstv", wraps=encoder.encode_sstv) as enc:
        for _ in range(6):
            first.read_samples(24000)
            second.read_samples(32000)
    assert enc.call_count == 2  # once per rate


def test_two_rates_sweeping_every_station_build_each_loop_once(fresh_caches):
    built = []
    real = dd._build_loops

    def counting(kind, variant, rate):
        built.append((kind, variant, rate))
        return real(kind, variant, rate)

    first = _device(2.4e6, 88e6)
    second = _device(3.2e6, 88e6)
    stations = _frequencies()
    with mock.patch.object(dd, "_build_loops", side_effect=counting):
        for _ in range(3):
            for low, high in zip(stations, reversed(stations), strict=True):
                first.set_frequency(low)
                first.read_samples(24000)
                second.set_frequency(high)
                second.read_samples(32000)
    assert built and len(built) == len(set(built))


def test_a_third_rate_lets_the_oldest_go(fresh_caches):
    """Memory stays bounded: the loops of the two rates used last are kept."""
    for rate in (2.4e6, 3.2e6, 1.0e6):
        _device(rate, 100.1e6).read_samples(int(rate * 0.01))
    rates = {key[2] for key in dd._LOOP_CACHE}
    assert rates == {3.2e6, 1.0e6}
    for rate in (2.4e6, 3.2e6, 1.0e6):
        _device(rate, 145.8e6).read_samples(int(rate * 0.01))
    assert set(dd._SSTV_CACHE) == {3.2e6, 1.0e6}
