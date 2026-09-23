"""Bounded LightNav-0 → SONIC execution owned by one EMERGE action."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable

try:
    from .observation_writer import EgoObservationWriter
except ImportError:  # External-plugin loading adds this directory to sys.path.
    from observation_writer import EgoObservationWriter


CancelCheck = Callable[[], str | None]
StateCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class NavigationConfig:
    lightnav_server: str
    camera_endpoint: str = "tcp://127.0.0.1:5555"
    camera_name: str = "ego_view"
    sonic_feedback_endpoint: str = "tcp://127.0.0.1:5557"
    sonic_feedback_topic: str = "g1_debug"
    sonic_bind: str = "tcp://*:5556"
    client_id: str = "emerge-lightnav-sonic-g1"
    control_hz: float = 4.0
    input_timeout_s: float = 0.75
    max_stale_ticks: int = 8
    elastic_band_release_file: Path | None = None

    def __post_init__(self) -> None:
        release_file = self.elastic_band_release_file
        if release_file is not None and not isinstance(release_file, Path):
            object.__setattr__(
                self,
                "elastic_band_release_file",
                Path(release_file).expanduser().resolve(),
            )
        if not self.lightnav_server.strip():
            raise ValueError("lightnav_server must be non-empty")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if not math.isfinite(self.input_timeout_s) or self.input_timeout_s <= 0:
            raise ValueError("input_timeout_s must be finite and positive")
        if self.max_stale_ticks < 1:
            raise ValueError("max_stale_ticks must be at least one")


@dataclass(frozen=True, slots=True)
class NavigationResult:
    status: str
    reason: str
    ticks: int
    command_steps: int
    elapsed_s: float
    last_waypoint: tuple[float, float, float] | None = None
    last_yaw_rad: float | None = None
    lightnav_stop: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NavigationBackend:
    """Run a cancellable navigation action while owning the SONIC PUB socket."""

    def __init__(
        self,
        config: NavigationConfig,
        observation_writer: EgoObservationWriter,
        *,
        camera_factory: Callable[[str, str], Any] | None = None,
        yaw_factory: Callable[[str, str], Any] | None = None,
        publisher_factory: Callable[[str], Any] | None = None,
        client_factory: Callable[[str, str], Any] | None = None,
        adapter_factory: Callable[[float], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.observation_writer = observation_writer
        self._monotonic = monotonic
        self._sleep = sleep

        if any(
            factory is None
            for factory in (
                camera_factory,
                yaw_factory,
                publisher_factory,
                client_factory,
                adapter_factory,
            )
        ):
            defaults = self._default_factories()
            camera_factory = camera_factory or defaults[0]
            yaw_factory = yaw_factory or defaults[1]
            publisher_factory = publisher_factory or defaults[2]
            client_factory = client_factory or defaults[3]
            adapter_factory = adapter_factory or defaults[4]

        self._camera_factory = camera_factory
        self._yaw_factory = yaw_factory
        self._publisher_factory = publisher_factory
        self._client_factory = client_factory
        self._adapter_factory = adapter_factory

    @staticmethod
    def _default_factories() -> tuple[Callable[..., Any], ...]:
        try:
            from .bridge.adapter import AdapterLimits, LightNav2SonicAdapter
            from .bridge.lightnav_ws import LightNavWsClient
            from .bridge.runner import SonicCameraSubscriber, SonicYawSubscriber
            from .bridge.sonic_zmq import SonicZmqPublisher
        except ImportError as exc:
            try:
                from bridge.adapter import AdapterLimits, LightNav2SonicAdapter
                from bridge.lightnav_ws import LightNavWsClient
                from bridge.runner import SonicCameraSubscriber, SonicYawSubscriber
                from bridge.sonic_zmq import SonicZmqPublisher
            except ImportError as fallback_exc:
                raise RuntimeError(
                    "the vendored LightNav-SONIC bridge is not importable from the "
                    "external plugin directory"
                ) from fallback_exc

        return (
            SonicCameraSubscriber,
            SonicYawSubscriber,
            SonicZmqPublisher,
            lambda endpoint, client_id: LightNavWsClient(endpoint, client_id=client_id),
            lambda period: LightNav2SonicAdapter(
                AdapterLimits(control_period_s=period)
            ),
        )

    def run(
        self,
        *,
        instruction: str,
        max_steps: int,
        cancel_check: CancelCheck | None = None,
        state_callback: StateCallback | None = None,
    ) -> NavigationResult:
        if not instruction.strip():
            raise ValueError("navigation instruction must be non-empty")
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")

        period_s = 1.0 / self.config.control_hz
        started = self._monotonic()
        camera = yaw_source = publisher = client = None
        yaw: float | None = None
        last_waypoint: tuple[float, float, float] | None = None
        ticks = command_steps = stale_ticks = 0
        lightnav_stop = False
        adapter = self._adapter_factory(period_s)

        def finish(status: str, reason: str, error: str | None = None) -> NavigationResult:
            result = NavigationResult(
                status=status,
                reason=reason,
                ticks=ticks,
                command_steps=command_steps,
                elapsed_s=max(0.0, self._monotonic() - started),
                last_waypoint=last_waypoint,
                last_yaw_rad=yaw,
                lightnav_stop=lightnav_stop,
                error=error,
            )
            if state_callback is not None:
                state_callback({"status": status, **result.to_dict()})
            return result

        try:
            camera = self._camera_factory(
                self.config.camera_endpoint, self.config.camera_name
            )
            yaw_source = self._yaw_factory(
                self.config.sonic_feedback_endpoint,
                self.config.sonic_feedback_topic,
            )
            publisher = self._publisher_factory(self.config.sonic_bind)
            publisher.activate_planner()
            client = self._client_factory(
                self.config.lightnav_server, self.config.client_id
            )
            client.connect()

            while ticks < max_steps:
                tick_started = self._monotonic()
                cancellation = cancel_check() if cancel_check is not None else None
                if cancellation:
                    return finish("cancelled", str(cancellation))

                ticks += 1
                jpeg = camera.latest(self.config.input_timeout_s)
                yaw = yaw_source.latest(self.config.input_timeout_s)
                if jpeg is None or yaw is None:
                    stale_ticks += 1
                    missing = "camera" if jpeg is None else "robot yaw feedback"
                    self._send_idle(publisher, adapter, yaw, f"stale or missing {missing}")
                    self._update_running_state(
                        state_callback,
                        instruction=instruction,
                        ticks=ticks,
                        command_steps=command_steps,
                        max_steps=max_steps,
                        yaw=yaw,
                        waypoint=last_waypoint,
                        last_error=f"stale or missing {missing}",
                    )
                    if stale_ticks >= self.config.max_stale_ticks:
                        return finish("failed", "stale_input", f"stale or missing {missing}")
                else:
                    stale_ticks = 0
                    self.observation_writer.write_jpeg(jpeg)
                    self._release_elastic_band()
                    reply = client.predict(jpeg, instruction)
                    last_waypoint = tuple(reply.first_waypoint)
                    lightnav_stop = bool(reply.stop)
                    if lightnav_stop:
                        self._send_idle(publisher, adapter, yaw, "LightNav stop=true")
                        return finish("completed", "lightnav_stop")

                    command = adapter.convert_or_idle(last_waypoint, yaw)
                    publisher.send_planner(command)
                    command_steps += 1
                    self._update_running_state(
                        state_callback,
                        instruction=instruction,
                        ticks=ticks,
                        command_steps=command_steps,
                        max_steps=max_steps,
                        yaw=yaw,
                        waypoint=last_waypoint,
                        last_error=None,
                    )

                remaining = period_s - (self._monotonic() - tick_started)
                if remaining > 0:
                    self._sleep(remaining)

            return finish("budget_exhausted", "max_steps")
        except Exception as exc:
            return finish("failed", "exception", f"{type(exc).__name__}: {exc}")
        finally:
            self._send_idle(publisher, adapter, yaw, "navigation action shutdown")
            for resource in (client, publisher, camera, yaw_source):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass

    def _release_elastic_band(self) -> None:
        path = self.config.elastic_band_release_file
        if path is None or path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    @staticmethod
    def _send_idle(publisher: Any, adapter: Any, yaw: float | None, reason: str) -> None:
        if publisher is None or not getattr(publisher, "activated", False):
            return
        try:
            publisher.send_idle(adapter.idle(yaw, reason=reason))
        except Exception:
            pass

    @staticmethod
    def _update_running_state(
        callback: StateCallback | None,
        *,
        instruction: str,
        ticks: int,
        command_steps: int,
        max_steps: int,
        yaw: float | None,
        waypoint: tuple[float, float, float] | None,
        last_error: str | None,
    ) -> None:
        if callback is None:
            return
        callback(
            {
                "status": "running",
                "instruction": instruction,
                "ticks": ticks,
                "command_steps": command_steps,
                "max_steps": max_steps,
                "last_yaw_rad": yaw,
                "last_waypoint": list(waypoint) if waypoint is not None else None,
                "last_error": last_error,
            }
        )
