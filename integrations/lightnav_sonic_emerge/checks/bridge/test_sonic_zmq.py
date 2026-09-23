import json
import struct
import unittest

from integrations.lightnav_sonic_emerge.bridge.adapter import SonicPlannerCommand
from integrations.lightnav_sonic_emerge.bridge.sonic_zmq import HEADER_SIZE, build_command_packet, build_planner_packet


def _header(packet: bytes, topic: bytes) -> dict:
    offset = len(topic)
    return json.loads(packet[offset : offset + HEADER_SIZE].rstrip(b"\x00"))


class SonicZmqPacketTests(unittest.TestCase):
    def test_command_packet_uses_standard_fields(self) -> None:
        packet = build_command_packet(start=True, stop=False, planner=True)
        self.assertTrue(packet.startswith(b"command"))
        header = _header(packet, b"command")
        self.assertEqual(header["v"], 1)
        self.assertEqual([field["name"] for field in header["fields"]], ["start", "stop", "planner"])
        self.assertEqual(packet[-3:], b"\x01\x00\x01")

    def test_planner_packet_matches_sonic_zmq_manager_layout(self) -> None:
        command = SonicPlannerCommand(
            mode=1,
            movement_direction=(0.6, 0.8, 0.0),
            facing_direction=(1.0, 0.0, 0.0),
            movement_speed=0.25,
            height=-1.0,
        )
        packet = build_planner_packet(command)
        self.assertTrue(packet.startswith(b"planner"))
        header = _header(packet, b"planner")
        self.assertEqual(
            [field["name"] for field in header["fields"]],
            ["mode", "movement", "facing", "speed", "height"],
        )
        payload = packet[len(b"planner") + HEADER_SIZE :]
        self.assertEqual(struct.unpack_from("<i", payload, 0)[0], 1)
        self.assertAlmostEqual(struct.unpack_from("<f", payload, 28)[0], 0.25)
        self.assertAlmostEqual(struct.unpack_from("<f", payload, 32)[0], -1.0)

    def test_zero_facing_is_rejected(self) -> None:
        command = SonicPlannerCommand(1, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.2)
        with self.assertRaisesRegex(ValueError, "facing_direction"):
            build_planner_packet(command)


if __name__ == "__main__":
    unittest.main()
