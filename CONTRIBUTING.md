# Contributing to SDR Module

Thank you for your interest in contributing to the SDR Module project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/SDR.git
   cd SDR
   ```
3. Add the upstream repository as a remote:
   ```bash
   git remote add upstream https://github.com/kase1111-hash/SDR.git
   ```
4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

Install the package in development mode with all dependencies:

```bash
pip install -e ".[dev,full]"
```

This installs:
- Core dependencies (NumPy)
- Hardware support (pyrtlsdr, hackrf)
- Development tools (pytest, black, isort, mypy, ruff)
- Optional dependencies (scipy, matplotlib)

### Hardware Setup (Optional)

For testing with actual hardware:

- **RTL-SDR**: Install RTL-SDR drivers for your operating system
- **HackRF One**: Install HackRF tools and drivers

The codebase includes demo modes for testing without hardware.

## Code Style

This project follows strict code style guidelines:

### Formatting

- **Black** for code formatting (88 character line length)
- **isort** for import sorting (Black-compatible profile)

Run formatters before committing:

```bash
black src/ tests/
isort src/ tests/
```

### Linting

- **Ruff** for fast Python linting
- **Mypy** for static type checking

Run linters:

```bash
ruff check src/ tests/
mypy src/sdr_module
```

### Style Guidelines

- Use descriptive variable and function names
- Add docstrings to all public modules, classes, and functions
- Use type hints for function parameters and return values
- Keep functions focused and single-purpose
- Prefer composition over inheritance

### Example

```python
from typing import Optional
import numpy as np

def compute_signal_power(
    samples: np.ndarray,
    sample_rate: float,
    window_size: Optional[int] = None
) -> float:
    """
    Compute the average power of a signal.

    Args:
        samples: Complex I/Q samples
        sample_rate: Sample rate in Hz
        window_size: Optional window size for averaging

    Returns:
        Average signal power in dB
    """
    if window_size is None:
        window_size = len(samples)

    power = np.mean(np.abs(samples[:window_size]) ** 2)
    return 10 * np.log10(power + 1e-10)
```

## Testing

### Running Tests

Run the full test suite:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=sdr_module --cov-report=html
```

Run specific test files:

```bash
pytest tests/test_dual_sdr.py -v
pytest tests/test_dsp.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for common setup
- Aim for comprehensive coverage of new features

Example test:

```python
import pytest
import numpy as np
from sdr_module.dsp import SpectrumAnalyzer

class TestSpectrumAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return SpectrumAnalyzer(fft_size=1024)

    def test_compute_spectrum_basic(self, analyzer):
        samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        result = analyzer.compute_spectrum(
            samples,
            center_freq=100e6,
            sample_rate=2.4e6
        )
        assert result is not None
        assert len(result.magnitudes) == 1024

    def test_compute_spectrum_empty_input(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.compute_spectrum(np.array([]), 100e6, 2.4e6)
```

## Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

Examples:

```
Add GFSK demodulator support

Implement GFSK demodulation with configurable modulation index
and symbol rate detection. Includes support for Bluetooth-style
GFSK with 0.5 modulation index.

Fixes #123
```

```
Fix spectrum analyzer peak detection

Peak detection was failing for signals near the noise floor.
Adjusted the threshold calculation to account for the noise
level in the FFT bins.
```

### Before Submitting

1. Ensure all tests pass:
   ```bash
   pytest
   ```

2. Run all code quality checks:
   ```bash
   black src/ tests/
   isort src/ tests/
   ruff check src/ tests/
   mypy src/sdr_module
   ```

3. Update documentation if needed

4. Update CHANGELOG.md for significant changes

## Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Provide a clear title and description
   - Reference any related issues
   - Include test results and screenshots if applicable

4. **Address review feedback** promptly

5. **Squash commits** if requested, keeping history clean

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commit messages are clear and descriptive

## Areas for Contribution

We welcome contributions in these areas:

### New Protocol Decoders

Add support for additional protocols:
- Weather station protocols
- Smart home devices
- Industrial IoT protocols
- Digital voice modes (D-STAR, Yaesu System Fusion)

### Additional Modulation Support

Implement new demodulators:
- OFDM
- Spread spectrum
- Advanced QAM variants

### GUI Improvements

- New visualization widgets
- Improved user experience
- Accessibility features
- Theme support

### Documentation

- Tutorial content
- API documentation improvements
- Example scripts
- Translation to other languages

### Performance Optimization

- DSP algorithm improvements
- Memory optimization
- GPU acceleration (CUDA/OpenCL)

### Bug Fixes

Check the [issue tracker](https://github.com/kase1111-hash/SDR/issues) for known bugs.

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- Python version and operating system
- SDR hardware being used (if applicable)
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Error messages and stack traces
- Relevant configuration or settings

### Feature Requests

For feature requests, include:

- Clear description of the feature
- Use case and motivation
- Any relevant examples or references
- Willingness to contribute the feature

## Questions?

If you have questions about contributing:

1. Check existing [issues](https://github.com/kase1111-hash/SDR/issues) and [discussions](https://github.com/kase1111-hash/SDR/discussions)
2. Open a new issue with your question
3. Tag it with the "question" label

Thank you for contributing to SDR Module!
