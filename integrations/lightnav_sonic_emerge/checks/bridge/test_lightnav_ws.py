import json
import unittest

from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import (
    LightNavProtocolError,
    LightNavWsClient,
    parse_next_response,
)


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
    def test_optional_request_timeout_is_passed_to_socket(self) -> None:
        class Socket:
            def __init__(self):
                self.timeout = None

            def send(self, _message):
                pass

            def recv(self, *, timeout):
                self.timeout = timeout
                return json.dumps({"action": "login", "data": {"rc": 0}})

        client = LightNavWsClient("ws://127.0.0.1:8051", request_timeout_s=10)
        socket = Socket()
        client._socket = socket
        client._send_expect_ok("login", {})
        self.assertEqual(socket.timeout, 10.0)
        with self.assertRaisesRegex(ValueError, "request_timeout_s"):
            LightNavWsClient("ws://127.0.0.1:8051", request_timeout_s=-1)

    def test_valid_response_preserves_first_waypoint(self) -> None:
        reply = parse_next_response(_response(), expected_seq=4)
        self.assertEqual(reply.first_waypoint, (0.25, 0.0, 0.1))
        self.assertFalse(reply.stop)
        self.assertIsNone(reply.visible)
        self.assertEqual(reply.latency_ms, 12.5)

    def test_visibility_flag_is_preserved_when_available(self) -> None:
        response = _response()
        response["data"]["visible"] = True
        self.assertTrue(parse_next_response(response).visible)
        response["data"]["visible"] = 1
        with self.assertRaisesRegex(LightNavProtocolError, "visible"):
            parse_next_response(response)

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
