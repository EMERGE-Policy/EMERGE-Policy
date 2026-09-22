# Model service infrastructure

Shared infrastructure lives in `external_model_server/model_service/`. Model
adapters live beside it. Importing contracts and health discovery doesn't load
Torch or any model. The public protocol replaces the previous `OK` health and
unversioned inference messages; upgrade public clients and servers together.

## Start and inspect

Use [the launcher](../scripts/model_server/README.md) to start the actual models.
OpenPI is a single process on port 8000. Its adapter loads the official OpenPI
policy directly, while the shared runtime owns the public protocol, health,
queue, and batching lifecycle.

Public `GET /healthz` returns JSON:

```json
{
  "schema": "emerge.model-health.v1",
  "protocol_version": 1,
  "service": "vggt",
  "model_id": "model.pt",
  "instance_id": "a-new-identifier-on-each-start",
  "input_schema": "emerge.vggt.request.v1",
  "output_schema": "emerge.vggt.response.v1",
  "capabilities": ["infer", "batch"],
  "status": "ready",
  "detail": ""
}
```

`ready` returns HTTP 200; `starting`, `draining`, and `failed` return 503 with the
same identity fields. Loading starts after binding the public listener. Health
reads runtime state; it doesn't run GPU inference.

## Add a local model

Implement `ModelAdapter`'s descriptor, load, infer, and close methods. The runtime
calls all three lifecycle/compute methods on the same dedicated execution thread.
Constructors and descriptors must remain lightweight.

```python
from external_model_server.model_service.contracts import ServiceDescriptor
from external_model_server.model_service.runtime import ModelServerRuntime


class DoubleAdapter:
    descriptor = ServiceDescriptor(
        "double", "example-v1", "example.double.input.v1", "example.double.output.v1"
    )

    def load(self):
        pass

    def infer(self, request):
        return {"value": request["value"] * 2}

    def close(self):
        pass


ModelServerRuntime(DoubleAdapter(), port=8008).serve_forever()
```

For batching, additionally implement `batch_key(request)` and
`infer_batch(requests)`, advertise `batch`, and set `max_batch_size` and
`batch_wait_ms`. Keys describe compatible requests. Results must match the input
count and order. The adapter owns tensor stacking, padding and output splitting;
the runtime owns queues and scheduling. Reject invalid input with `ValueError`;
use `ModelFailureError` only when the model truly cannot serve any more requests.

`close()` must also work after a partly failed load. Avoid loading a model or
importing its heavyweight backend at module import time. Put business schemas
in a lightweight module usable by both the adapter and its business client.

## Wrap an existing server

Implement async `load`, `check`, `infer`, and `close` with a descriptor, then use
`ModelServerRuntime(adapter, remote=True, concurrency=4)`. `check` must raise
`ServiceError("NOT_READY", ...)` when the backend is unavailable. `load` may wait
for the backend while the public endpoint reports `starting`; it must be cancellable.
The adapter translates the existing backend's protocol to public results.
Adapters for local models use the same local runtime path; remote adapters are
available for integrations that genuinely need to wrap an existing service.

Remote I/O may be cancelled; local GPU execution isn't forcibly interrupted.

## Clients and discovery

```python
from external_model_server.model_service.client import ModelClient
from external_model_server.model_service.contracts import ServiceExpectation
from external_model_server.model_service.discovery import ServiceDiscovery

expected = ServiceExpectation(
    "double", "example.double.input.v1", "example.double.output.v1"
)
with ModelClient(expected, discovery=ServiceDiscovery()) as client:
    print(client.infer({"value": 21}))
```

`AsyncModelClient` provides the same operations using `async with` and `await`.
Each connection handles one in-flight request; separate connections can be
batched by the server. The client validates the handshake before sending input.
Health queries also verify identity, but aren't required before every inference.

`ServiceDiscovery` finds exactly one matching instance from the configured local
port range. A `DiscoveryConfig` customizes the scan range and timeouts.
Discovery has bounded concurrency and a whole-scan deadline. Address aliases
for the same instance are deduplicated; genuine replicas remain distinct.
An incomplete scan cannot select a service automatically. The client retains
its selected endpoint; reconnecting revalidates the handshake at that address.

## Wire format and failure behavior

The first websocket message is binary metadata: `type: metadata` plus the same
description fields as health (without the health schema). Requests contain
`request_id`, `protocol_version`, `operation: infer`, `input_schema`, `timeout_s`,
and `payload`. Responses contain the same ID and either `ok: true, result, timing`
or `ok: false, error: {code, message, retryable}`. Arrays retain their dtype and
shape through msgpack; object/structured/complex arrays are rejected.

The default message limit is 256 MiB, queue capacity is 64, and local execution
is serialized. A full queue returns `OVERLOADED`. Expired queued requests are
removed; timed-out running requests may still compute. Timeouts close the client
connection so a delayed reply can't be mistaken for the next result. Requests
that were sent are never automatically replayed, even when `retryable` is true.
Request IDs provide correlation, not deduplication or exactly-once execution.

Shutdown enters `draining`, rejects queued and new work, waits for active local
execution, and then closes the adapter on its execution thread. If the shutdown
deadline expires, the runtime logs it and doesn't free a model still in use;
the launcher terminates the remaining process. Logs include service, instance, request ID,
failure code and queue/inference latency, not image or array payloads.
