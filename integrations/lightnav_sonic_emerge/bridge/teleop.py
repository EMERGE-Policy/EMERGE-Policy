"""Keyboard recovery teleoperation for the SONIC navigation simulator.

Run this only after the standalone bridge runner or EMERGE Controller has exited. It does not
contact LightNav and never resets MuJoCo: it merely publishes standard SONIC
planner commands to help position or reorient G1 before another VLN attempt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import select
import sys
import termios
import time
import tty

from .adapter import SONIC_SLOW_WALK_MODE, LightNav2SonicAdapter, SonicPlannerCommand
from .runner import SonicYawSubscriber
from .sonic_zmq import SonicZmqPublisher


_EPSILON = 1e-8


def wrap_angle(angle: float) -> float:
    """Normalize an angle to ``[-pi, pi]``."""

    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class TeleopState:
    """Persistent local movement and world-heading targets for keyboard input."""

    forward: float = 0.0
    lateral: float = 0.0
    target_yaw: float | None = None
    speed_mps: float = 0.18

    @property
    def has_translation(self) -> bool:
        return math.hypot(self.forward, self.lateral) > _EPSILON

    def stop(self, robot_yaw: float) -> None:
        self.forward = 0.0
        self.lateral = 0.0
        self.target_yaw = robot_yaw

    def set_translation(self, forward: float, lateral: float, robot_yaw: float) -> None:
        self.forward = forward
        self.lateral = lateral
        # Selecting a movement key cancels any unfinished turn and faces the
        # current body heading, which makes keyboard recovery predictable.
        self.target_yaw = robot_yaw

    def turn(self, delta_yaw: float, robot_yaw: float) -> None:
        self.forward = 0.0
        self.lateral = 0.0
        base_yaw = robot_yaw if self.target_yaw is None else self.target_yaw
        self.target_yaw = wrap_angle(base_yaw + delta_yaw)


def command_from_state(
    state: TeleopState,
    robot_yaw: float,
    *,
    heading_tolerance_rad: float,
) -> SonicPlannerCommand:
    """Build one standard world-frame SONIC command without any transport I/O."""

    target_yaw = robot_yaw if state.target_yaw is None else state.target_yaw
    facing = (math.cos(target_yaw), math.sin(target_yaw), 0.0)
    if not state.has_translation:
        if abs(wrap_angle(target_yaw - robot_yaw)) <= heading_tolerance_rad:
            return LightNav2SonicAdapter().idle(robot_yaw, reason="keyboard stop")
        return SonicPlannerCommand(
            mode=SONIC_SLOW_WALK_MODE,
            movement_direction=(0.0, 0.0, 0.0),
            facing_direction=facing,
            movement_speed=-1.0,
            height=-1.0,
            reason="keyboard turn-in-place",
        )

    local_norm = math.hypot(state.forward, state.lateral)
    forward = state.forward / local_norm
    lateral = state.lateral / local_norm
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    world_x = cos_yaw * forward - sin_yaw * lateral
    world_y = sin_yaw * forward + cos_yaw * lateral
    return SonicPlannerCommand(
        mode=SONIC_SLOW_WALK_MODE,
        movement_direction=(world_x, world_y, 0.0),
        facing_direction=facing,
        movement_speed=state.speed_mps,
        height=-1.0,
        reason="keyboard translation",
    )


class CbreakInput:
    """Read one-byte terminal keys while keeping Ctrl-C enabled.

    ``conda run`` often pipes stdin into its child process.  Opening the
    controlling terminal explicitly keeps keyboard teleop interactive in that
    common launch mode.
    """

    def __enter__(self) -> CbreakInput:
        try:
            self._stream = open("/dev/tty", "r", encoding="utf-8", buffering=1)
        except OSError as exc:
            raise RuntimeError("teleop requires a controlling interactive terminal") from exc
        self._fd = self._stream.fileno()
        self._settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def read_pending(self) -> list[str]:
        keys: list[str] = []
        while select.select([self._stream], [], [], 0.0)[0]:
            key = self._stream.read(1)
            if key:
                keys.append(key)
        return keys

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
        self._stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-bind", default="tcp://*:5556")
    parser.add_argument("--sonic-feedback-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--sonic-feedback-topic", default="g1_debug")
    parser.add_argument("--control-hz", type=float, default=4.0)
    parser.add_argument("--speed-mps", type=float, default=0.18)
    parser.add_argument("--turn-step-deg", type=float, default=30.0)
    parser.add_argument("--heading-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--input-timeout-s", type=float, default=0.75)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--control-hz": args.control_hz,
        "--speed-mps": args.speed_mps,
        "--turn-step-deg": args.turn_step_deg,
        "--heading-tolerance-deg": args.heading_tolerance_deg,
        "--input-timeout-s": args.input_timeout_s,
    }
    invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0.0]
    if invalid:
        raise ValueError(f"arguments must be finite and positive: {', '.join(invalid)}")
    if args.speed_mps > 0.25:
        raise ValueError("--speed-mps must not exceed the bridge's conservative 0.25 m/s limit")


def _print_help() -> None:
    print(
        "keys: W forward | S backward | A left | D right | "
        "Q/E turn left/right one step | SPACE stop | +/- speed | H help | X exit"
    )


def _handle_key(key: str, state: TeleopState, yaw: float, turn_step_rad: float) -> bool:
    """Apply a keypress. Return ``True`` when the operator requests exit."""

    lower = key.lower()
    if lower == "w":
        state.set_translation(1.0, 0.0, yaw)
    elif lower == "s":
        state.set_translation(-1.0, 0.0, yaw)
    elif lower == "a":
        state.set_translation(0.0, 1.0, yaw)
    elif lower == "d":
        state.set_translation(0.0, -1.0, yaw)
    elif lower == "q":
        state.turn(turn_step_rad, yaw)
    elif lower == "e":
        state.turn(-turn_step_rad, yaw)
    elif key == " ":
        state.stop(yaw)
    elif key == "+":
        state.speed_mps = min(0.25, state.speed_mps + 0.02)
        print(f"speed: {state.speed_mps:.2f} m/s")
    elif key == "-":
        state.speed_mps = max(0.02, state.speed_mps - 0.02)
        print(f"speed: {state.speed_mps:.2f} m/s")
    elif lower == "h":
        _print_help()
    elif lower == "x":
        return True
    return False


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    period_s = 1.0 / args.control_hz
    turn_step_rad = math.radians(args.turn_step_deg)
    heading_tolerance_rad = math.radians(args.heading_tolerance_deg)
    adapter = LightNav2SonicAdapter()
    publisher = SonicZmqPublisher(args.sonic_bind)
    yaw_source = SonicYawSubscriber(args.sonic_feedback_endpoint, args.sonic_feedback_topic)
    state = TeleopState(speed_mps=args.speed_mps)
    last_yaw: float | None = None
    try:
        print(f"SONIC keyboard teleop bound at {args.sonic_bind}; waiting for planner subscriber")
        publisher.activate_planner()
        _print_help()
        with CbreakInput() as keyboard:
            while True:
                tick_started = time.monotonic()
                yaw = yaw_source.latest(args.input_timeout_s)
                if yaw is None:
                    publisher.send_idle(adapter.idle(last_yaw, reason="stale or missing robot yaw feedback"))
                    print("idle: stale or missing robot yaw feedback")
                else:
                    last_yaw = yaw
                    for key in keyboard.read_pending():
                        if _handle_key(key, state, yaw, turn_step_rad):
                            return 0
                    command = command_from_state(
                        state,
                        yaw,
                        heading_tolerance_rad=heading_tolerance_rad,
                    )
                    if command.is_idle:
                        publisher.send_idle(command)
                    else:
                        publisher.send_planner(command)
                remaining = period_s - (time.monotonic() - tick_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("interrupted; sending SONIC locomotion IDLE")
        return 130
    finally:
        try:
            publisher.send_idle(adapter.idle(last_yaw, reason="keyboard teleop shutdown"))
        finally:
            publisher.close()
            yaw_source.close()


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
