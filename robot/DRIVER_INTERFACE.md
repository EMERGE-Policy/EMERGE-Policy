# Robot driver interface

Every controller driver must subclass `robot.drivers.base_driver.BaseDriver`.
The abstract methods are the integration contract; `load_driver()` rejects a
class that leaves any of them unimplemented.

## Required lifecycle

```python
class MyDriver(BaseDriver):
    def load_environment(self) -> None: ...
    def reset_environment(self) -> None: ...
    def get_scene_catalog(self) -> SceneCatalog: ...
    def switch_scene(self, scene_id: str) -> None: ...
    def close(self) -> None: ...
```

`load_environment()` creates the environment from the driver's initial
configuration. `reset_environment()` creates a fresh initial state from that
configuration; the Controller calls it after `/reset` has closed the old driver
and cleared the workspace. `switch_scene(scene_id)` loads a different scene in
the same driver family. It must accept a new driver instance, validate the ID
before changing resources, and publish state and observations before returning.

`get_scene_catalog()` is called before loading so the TUI can show choices. It
returns a `SceneCatalog`:

```python
{
    "root": "My environments",
    "current": "kitchen/task-01",
    "entries": [
        {"id": "kitchen/task-01", "label": "Kitchen/Task 1"},
    ],
}
```

IDs are stable driver-owned values. Labels are display paths; `/` lets the TUI
group entries into folders. A driver may use task names, database keys, URLs or
device identifiers. The interface does not require BDDL or filesystem paths.
`current` is `None` while no scene is loaded.

`close()` must stop environment writes and release all resources. It must be safe
before a load, after a partial load, and when called more than once. The
Controller only clears `ACTION.md`, plans and artifacts after `close()` has
returned successfully.

`get_profile_path()`, `execute_action()` and `get_runtime_state()` remain part
of the same required contract. A driver should keep scene-specific parsing and
resource creation inside these lifecycle methods; the TUI and Controller only
deal with scene IDs and the generic catalog.

## Controller behavior

- Startup calls `load_environment()`.
- `/reset` closes the driver, clears the task workspace, constructs a new driver,
  and calls `reset_environment()`.
- `/scene` validates the requested ID against the published catalog, closes the
  driver, clears the task workspace, constructs a new driver, and calls
  `switch_scene(scene_id)`.
- A failed switch leaves the environment unavailable and preserves the catalog
  so the user can retry from `/scene`. The Controller never replays an action or
  silently restores the previous scene.

The LIBERO driver implements this contract by mapping IDs to relative BDDL paths
under `libero.bddl_root`. Other drivers can expose a completely different scene
model while keeping the same TUI and Controller integration.
