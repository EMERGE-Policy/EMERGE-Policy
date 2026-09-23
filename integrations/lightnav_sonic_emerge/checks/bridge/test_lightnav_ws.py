import unittest

from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavProtocolError, parse_next_response


def _response(*, seq=4, stop=False, actions=None, rc=0):
    if actions is None:
        actions = [[0.25, 0.0, 0.1], [0.5, 0.0, 0.2]]
    return {
        "action": "next",
        "data": {
            "rc": rc,
            "seq": seq,
            "actions": {"step": 42, "actions": actions},
            "stop": stop,
            "latency_ms": 12.5,
            "raw_text": "<traj_1>",
        },
    }


class LightNavResponseTests(unittest.TestCase):
    def test_valid_response_preserves_first_waypoint(self) -> None:
        reply = parse_next_response(_response(), expected_seq=4)
        self.assertEqual(reply.first_waypoint, (0.25, 0.0, 0.1))
        self.assertFalse(reply.stop)
        self.assertEqual(reply.latency_ms, 12.5)

    def test_stop_response_is_valid(self) -> None:
        reply = parse_next_response(_response(stop=True))
        self.assertTrue(reply.stop)

    def test_sequence_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(LightNavProtocolError, "sequence mismatch"):
            parse_next_response(_response(seq=5), expected_seq=4)

    def test_server_error_is_rejected(self) -> None:
        with self.assertRaisesRegex(LightNavProtocolError, "rc=500"):
            parse_next_response(_response(rc=500))

    def test_incomplete_waypoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(LightNavProtocolError, "exactly three"):
            parse_next_response(_response(actions=[[0.1, 0.2]]))

    def test_non_finite_waypoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(LightNavProtocolError, "finite"):
            parse_next_response(_response(actions=[[float("inf"), 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
