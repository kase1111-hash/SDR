# Security Policy

## Supported Versions

The following versions of SDR Module are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in SDR Module, please report it responsibly.

### How to Report

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers via GitHub's private vulnerability reporting feature
3. Or open a private security advisory at: https://github.com/kase1111-hash/SDR/security/advisories/new

### What to Include

When reporting a vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes or mitigations
- Your contact information for follow-up

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues

### After Reporting

1. You will receive acknowledgment of your report
2. We will investigate and validate the vulnerability
3. We will work on a fix and coordinate disclosure timing
4. You will be credited in the security advisory (unless you prefer anonymity)

## Security Considerations

### RF Transmission Safety

SDR Module includes critical safety features for radio transmission:

#### TX Frequency Lockouts

The software implements hard-coded frequency lockouts to prevent transmission on protected frequencies:

- **GPS/GNSS**: L1 (1575.42 MHz), L2 (1227.60 MHz), L5 (1176.45 MHz), GLONASS, Galileo, BeiDou
- **Aviation Emergency**: 121.5 MHz, 243.0 MHz
- **ADS-B/Mode S**: 1030 MHz, 1090 MHz
- **Emergency Beacons**: 406.0-406.1 MHz (COSPAS-SARSAT)
- **Marine Distress**: 156.8 MHz (VHF Channel 16)
- **Cellular Bands**: Various protected cellular frequencies

These lockouts are implemented in the device drivers and cannot be bypassed through the API.

#### Regulatory Compliance

Users are responsible for:

- Obtaining appropriate licenses for transmission (e.g., amateur radio license)
- Complying with local regulations regarding RF transmission
- Operating within authorized frequency bands and power levels
- Proper identification requirements (e.g., callsign transmission)

### Data Security

#### I/Q Recordings

- I/Q recordings may contain sensitive information depending on the signals captured
- Store recordings securely and be aware of legal implications
- SigMF metadata files may contain location and equipment information

#### Configuration Files

- Configuration files may contain frequency presets and hardware settings
- Do not share configurations that reveal sensitive monitoring setups

### Plugin Security

When using third-party plugins:

- Only install plugins from trusted sources
- Review plugin code before installation when possible
- Plugins have access to hardware and file system through the API
- Report suspicious plugins to the maintainers

### Network Security

If using network features (future planned features):

- Do not expose SDR control interfaces to untrusted networks
- Use authentication when available
- Be cautious with remote control capabilities

## Known Security Considerations

### Hardware Access

- SDR devices require direct hardware access
- Running with elevated privileges may be required for USB device access
- Use udev rules on Linux to avoid running as root

### Signal Processing

- Malformed I/Q data could potentially cause crashes
- Implement input validation when processing untrusted data sources

## Security Best Practices

### For Users

1. Keep the software updated to the latest version
2. Use the minimum required privileges
3. Be aware of what frequencies you are monitoring (legal considerations)
4. Secure your I/Q recordings appropriately
5. Validate plugins before installation

### For Contributors

1. Follow secure coding practices
2. Validate all user inputs
3. Handle errors gracefully without exposing sensitive information
4. Do not bypass TX frequency lockouts
5. Test security-critical code thoroughly

## Disclosure Policy

We follow a coordinated disclosure policy:

1. Security issues are fixed before public disclosure
2. Fixes are released as soon as practically possible
3. Security advisories are published after fixes are available
4. Credit is given to reporters (with permission)

## Contact

For security-related inquiries:

- GitHub Security Advisories: https://github.com/kase1111-hash/SDR/security/advisories
- General Issues: https://github.com/kase1111-hash/SDR/issues

Thank you for helping keep SDR Module secure.
