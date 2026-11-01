"""What decoded messages say: rows, the detail card's lines and APRS parsing.

tl:UX-06 / tl:G-03 / fg:FG-G07: ADS-B rows dropped the track, vertical rate
and category ("450 kt" alone), DF11 rows showed a dash, ACARS rows hid the
flight when a registration was there, AX.25 rows hid the digipeater path
and APRS reports were shown raw ("!4903.50N/07201.75W>...") although
parse_aprs existed.
"""

import pytest

from sdr_module.dsp import protocol_encoders as enc
from sdr_module.dsp.live_decoders import describe_message, message_details
from sdr_module.dsp.morse import MorseMessage
from sdr_module.dsp.protocols import (
    ACARSMessage,
    ADSBDecoder,
    ADSBMessage,
    APRSMessage,
    AX25Decoder,
    AX25Frame,
    ProtocolType,
    RDSData,
    parse_aprs_info,
)

BASE = dict(timestamp=0.0, raw_bits=b"", valid=True)


def adsb(**fields):
    return ADSBMessage(
        protocol=ProtocolType.ADSB, icao_address="4840D6", **BASE, **fields
    )


class TestADSB:
    def test_identification_has_the_category(self):
        msg = adsb(
            downlink_format=17, type_code=4, callsign="KLM1023", category="Heavy"
        )
        assert describe_message(msg) == ("4840D6", "KLM1023  ·  Heavy")
        assert message_details(msg) == [
            "DF17 extended squitter, type code 4 (identification)",
            "Category: Heavy",
        ]

    @pytest.mark.parametrize(
        "rate,text",
        [
            (0, "level"),
            (1216, "climbing 1,216 ft/min"),
            (-832, "descending 832 ft/min"),
        ],
    )
    def test_velocity_has_track_and_vertical_rate(self, rate, text):
        msg = adsb(
            downlink_format=17,
            type_code=19,
            velocity=450.0,
            heading=265.0,
            vertical_rate=rate,
        )
        assert describe_message(msg)[1] == f"450 kt  ·  track 265°  ·  {text}"

    def test_no_vertical_rate_information_is_not_level(self):
        # An all-zero vertical rate field means "no information"; it was
        # read as -64 ft/min ("descending 64 ft/min").
        from sdr_module.gui import device_dialog as dd

        me = dd._velocity_me(450.0, 265.0) & ~(0x1FF << 10)  # no rate
        frame = dd._extended_squitter(dd._AIRCRAFT_ICAO, me)
        msg = ADSBDecoder(2e6)._parse_frame(frame)
        assert (msg.vertical_rate, msg.vertical_rate_known) == (0, False)
        assert describe_message(msg)[1] == "450 kt  ·  track 265°"
        level = ADSBDecoder(2e6)._parse_frame(
            dd._extended_squitter(dd._AIRCRAFT_ICAO, dd._velocity_me(450.0, 265.0))
        )
        assert level.vertical_rate_known and describe_message(level)[1].endswith(
            "level"
        )

    def test_track_rounds_into_the_compass(self):
        msg = adsb(downlink_format=17, type_code=19, velocity=300.0, heading=359.7)
        assert "track 0°" in describe_message(msg)[1]

    def test_unread_squitters_say_what_they_are(self):
        assert describe_message(adsb(downlink_format=17, type_code=31))[1] == (
            "Operational status (not read)"
        )
        assert describe_message(adsb(downlink_format=17, type_code=19))[1] == (
            "Airborne velocity (not read)"
        )
        assert describe_message(adsb(downlink_format=18, type_code=27))[1] == (
            "Type code 27 (not read)"
        )

    def test_a_track_of_north_is_shown(self):
        msg = adsb(downlink_format=17, type_code=19, velocity=300.0, heading=0.0)
        assert "track 0°" in describe_message(msg)[1]

    def test_address_only_replies_say_so(self):
        assert describe_message(adsb(downlink_format=11))[1] == (
            "Address only (all-call reply)"
        )
        assert describe_message(adsb(downlink_format=0))[1] == "Address only"

    def test_squawk_altitude_position_and_ground(self):
        assert (
            describe_message(adsb(downlink_format=5, squawk="7700"))[1] == "Squawk 7700"
        )
        msg = adsb(
            downlink_format=17,
            type_code=11,
            altitude=38000,
            latitude=52.25,
            longitude=3.9,
        )
        assert describe_message(msg)[1] == "38,000 ft  ·  52.2500, 3.9000"
        assert (
            describe_message(adsb(downlink_format=11, on_ground=True))[1]
            == "on the ground"
        )

    def test_categories_by_set(self):
        cats = ADSBDecoder.CATEGORIES
        assert cats[4][5] == "Heavy" and cats[4][7] == "Rotorcraft"
        assert cats[3][1] == "Glider" and cats[2][1] == "Emergency vehicle"
        assert cats[1][3] == ""

    def test_no_category_name_for_no_category(self):
        # The demo's KLM1023 sends the classic frame, category 0 (none).
        from sdr_module.gui import device_dialog as dd

        frame = dd._extended_squitter(dd._AIRCRAFT_ICAO, dd._ident_me("KLM1023"))
        assert frame.hex().upper() == "8D4840D6202CC371C32CE0576098"
        msg = ADSBDecoder(2e6)._parse_frame(frame)
        assert (msg.callsign, msg.category) == ("KLM1023", "")
        assert describe_message(msg) == ("4840D6", "KLM1023")


class TestACARS:
    def test_flight_beside_the_registration(self):
        msg = ACARSMessage(
            protocol=ProtocolType.ACARS,
            registration="N601SD",
            flight_id="SD0601",
            label="H1",
            text="ETA 1435Z",
            block_id="2",
            message_number="M01A",
            mode="2",
            **BASE,
        )
        assert describe_message(msg) == ("N601SD", "Flight SD0601  ·  H1  ·  ETA 1435Z")
        assert message_details(msg) == [
            "Registration N601SD, flight SD0601",
            "Label H1, block 2, message M01A, mode 2",
        ]

    def test_flight_as_the_address_without_a_registration(self):
        msg = ACARSMessage(
            protocol=ProtocolType.ACARS,
            flight_id="SD0601",
            label="H1",
            text="X",
            **BASE,
        )
        assert describe_message(msg) == ("SD0601", "H1  ·  X")


class TestAX25:
    def frame(self, info, path=("WIDE1-1",)):
        return AX25Frame(
            protocol=ProtocolType.AX25,
            source="N0CALL-9",
            destination="APZ001",
            digipeaters=list(path),
            info=info,
            **BASE,
        )

    def test_position_reports_are_read(self):
        msg = self.frame("!4903.50N/07201.75W>Demo mobile")
        assert describe_message(msg) == (
            "N0CALL-9>APZ001",
            "49.0583, -72.0292  ·  Car  ·  Demo mobile",
        )
        assert message_details(msg) == [
            "Path: via WIDE1-1",
            "APRS: Position · Car (/>)",
            "Sent: !4903.50N/07201.75W>Demo mobile",
        ]

    def test_course_speed_and_altitude(self):
        msg = self.frame("@092345z4903.50N/07201.75W>088/036/A=001234 moving")
        assert describe_message(msg)[1] == (
            "49.0583, -72.0292  ·  Car  ·  36 kt, course 88°  ·  1,234 ft  ·  moving"
        )
        assert "Position with time (accepts messages)" in message_details(msg)[1]

    def test_messages_status_and_others(self):
        assert describe_message(self.frame(":N0CALL-7 :Hello there{12"))[1] == (
            "To N0CALL-7: Hello there"
        )
        assert describe_message(self.frame(">Net tonight"))[1] == "Net tonight"
        # Not APRS, or a position that doesn't read: as sent.
        assert describe_message(self.frame("Hello world"))[1] == "Hello world"
        assert describe_message(self.frame("!x"))[1] == "!x"
        assert message_details(self.frame("Hi", path=()))[0] == "Path: direct"

    def test_a_repeated_digipeater_is_marked(self):
        data = bytearray(
            enc.ax25_frame("N0CALL-9", "APZ001", "!x", ["WIDE1-1", "WIDE2-1"])
        )
        data[14 + 6] |= 0x80  # the first digipeater has repeated it
        payload = bytes(data[:-2])
        fcs = AX25Decoder(9600)._compute_crc(payload)
        frame = AX25Decoder(9600)._parse_frame_bytes(
            payload + bytes([fcs & 0xFF, fcs >> 8])
        )
        assert frame.digipeaters == ["WIDE1-1*", "WIDE2-1"]
        assert message_details(frame)[0] == "Path: via WIDE1-1*, WIDE2-1"

    def test_parse_aprs_keeps_its_api(self):
        decoder = AX25Decoder(9600)
        msg = decoder.parse_aprs(self.frame("=4850.00N/07230.00W-Home station"))
        assert isinstance(msg, APRSMessage)
        assert (msg.data_type, msg.symbol, msg.symbol_name) == (
            "position_msg",
            "/-",
            "Home",
        )
        assert msg.path == ["WIDE1-1"]
        assert describe_message(msg)[1] == "48.8333, -72.5000  ·  Home  ·  Home station"


class TestParseAprsInfo:
    def test_compressed_position_from_the_spec(self):
        # APRS 1.0.1, chapter 9: "/5L!!<*e7>7P[" is 49 30'N 72 45'W,
        # course 88, speed 36.2 knots, a car.
        fields = parse_aprs_info("!/5L!!<*e7>7P[")
        assert fields["latitude"] == pytest.approx(49.5, abs=1e-4)
        assert fields["longitude"] == pytest.approx(-72.75, abs=1e-4)
        assert (fields["course"], fields["speed"]) == (88.0, 36.2)
        assert fields["symbol_name"] == "Car"

    def test_ambiguous_position(self):
        fields = parse_aprs_info("!4903.  N/07201.  W-")
        assert fields["latitude"] == pytest.approx(49.05, abs=1e-6)

    def test_object(self):
        fields = parse_aprs_info(";LEADER   *092345z4903.50N/07201.75W>088/036")
        assert fields["data_type"] == "object" and fields["addressee"] == "LEADER"
        assert fields["latitude"] == pytest.approx(49.0583, abs=1e-4)

    def test_empty_and_unknown(self):
        assert parse_aprs_info("")["data_type"] == "unknown"
        assert parse_aprs_info("{{weird")["data_type"] == "unknown"

    def test_a_course_and_speed_that_are_not_base_91_are_left_out(self):
        # DEC-08: a speed byte of 0xFF (it arrives as U+FFFD) was read as
        # 1.08 ** 65500 and raised OverflowError: the row never showed and
        # sdr-scan decode stopped listing at it.
        for bad in ("\ufffd", "\x7f", " "):
            fields = parse_aprs_info(f"!/5L!!<*e7>!{bad}A comment")
            assert fields["latitude"] == pytest.approx(49.5, abs=1e-4)
            assert (fields["course"], fields["speed"]) == (0.0, 0.0)
        frame = AX25Frame(
            protocol=ProtocolType.AX25,
            timestamp=0.0,
            raw_bits=b"",
            valid=True,
            source="N0CALL-1",
            destination="APRS",
            info="!/5L!!<*e7>!\ufffdA bad speed byte",
        )
        assert describe_message(frame)[1].startswith("49.5000, -72.7500")
        assert "APRS: Position" in " ".join(message_details(frame))


class TestOtherDetails:
    def test_rds(self):
        msg = RDSData(
            protocol=ProtocolType.RDS,
            pi_code=0xC0DE,
            pty=9,
            pty_name="Top 40",
            tp=True,
            ms=True,
            af_list=[99.5, 100.1],
            **BASE,
        )
        assert message_details(msg) == [
            "PI C0DE · program type 9 (Top 40)",
            "Traffic program, music",
            "Also on 99.5, 100.1 MHz",
        ]

    def test_morse(self):
        msg = MorseMessage(protocol=ProtocolType.CW, text="CQ", wpm=18.0, **BASE)
        assert message_details(msg) == ["Sent at 18 words per minute"]
