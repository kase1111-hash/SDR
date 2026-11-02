"""Which dual-SDR operation modes the controller really implements (FG-B16).

``OperationMode`` lists five modes. Choosing relay used to succeed
(``set_mode`` returned True) and then nothing happened; now it is refused
with an error that says why. Wideband scan was refused the same way until
the controller implemented it (K-X1, ``tests/test_dual_sdr_wideband.py``).
"""

import pytest

from sdr_module.core.dual_sdr import (
    NOT_IMPLEMENTED_MODES,
    DualSDRController,
    OperationMode,
)
from tests.test_dual_sdr import MockHackRFDevice, MockRTLSDRDevice


def _controller() -> DualSDRController:
    controller = DualSDRController()
    controller._rtlsdr = MockRTLSDRDevice()
    controller._hackrf = MockHackRFDevice()
    return controller


@pytest.mark.parametrize("mode", [OperationMode.RELAY])
def test_set_mode_refuses_a_mode_that_is_not_implemented(mode):
    controller = _controller()
    with pytest.raises(NotImplementedError) as raised:
        controller.set_mode(mode)
    message = str(raised.value)
    assert "not implemented" in message
    assert message == NOT_IMPLEMENTED_MODES[mode]
    # The mode in use doesn't change.
    assert controller.state.mode == OperationMode.DUAL_RX


def test_the_reasons_point_somewhere_useful():
    relay = NOT_IMPLEMENTED_MODES[OperationMode.RELAY]
    assert "transmit" in relay.lower()


def test_implemented_flag_and_supported_modes():
    implemented = (
        OperationMode.DUAL_RX,
        OperationMode.FULL_DUPLEX,
        OperationMode.TX_MONITOR,
        OperationMode.WIDEBAND_SCAN,
    )
    for mode in OperationMode:
        assert mode.implemented == (mode in implemented)
    assert DualSDRController.supported_modes() == implemented
    # Every mode is either implemented or says why it isn't.
    assert set(NOT_IMPLEMENTED_MODES) == set(OperationMode) - set(implemented)


def test_implemented_modes_still_switch():
    controller = _controller()
    for mode in DualSDRController.supported_modes():
        assert controller.set_mode(mode)
        assert controller.state.mode == mode


def test_unimplemented_mode_is_refused_even_while_streaming():
    """A programming error is reported as such, not as "stop streaming"."""
    controller = _controller()
    controller.start_dual_rx()
    with pytest.raises(NotImplementedError):
        controller.set_mode(OperationMode.RELAY)
    assert controller.state.rtlsdr_streaming
    controller.stop_all()
