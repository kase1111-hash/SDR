# sdr-antenna-array

Multi-SDR antenna array support for beamforming and direction finding.

Part of the [SDR](https://github.com/kase1111-hash/SDR) project. Provides:

- **Array configuration** — linear/circular geometries with per-element
  positions and presets (`array_config`)
- **Beamforming** — conventional delay-and-sum plus adaptive MVDR
  (`beamformer`, `adaptive_beamformer`)
- **Direction of arrival** — MUSIC and correlation-based DoA estimation (`doa`)
- **Calibration** — phase/gain calibration across elements (`calibration`)
- **Array control** — synchronized N-device streaming with timestamped
  buffers (`array_controller`, `timestamped_buffer`, `cross_correlator`)

## Installation

```bash
pip install -e packages/sdr-antenna-array
```

Install with the `sdr` extra to integrate with live SDR hardware via
`sdr-module`:

```bash
pip install -e "packages/sdr-antenna-array[sdr]"
```

## Quick start

```python
import numpy as np
from sdr_antenna_array import Beamformer, create_linear_4_element

config = create_linear_4_element(frequency=433e6)
beamformer = Beamformer(config)
result = beamformer.delay_and_sum(signals, steering_azimuth=np.radians(30))
```

See the test suite under `tests/` for more usage examples.
