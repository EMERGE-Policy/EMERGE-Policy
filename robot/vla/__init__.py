"""robot.vla — VLA (π0.5) hybrid execution module.

IMPORTANT (see dev/vla_hybrid_execution_plan.md §3.1):
This package is imported from BOTH conda environments:

  * ``Emerge_VLA`` (client) — runs LIBERO/MuJoCo + the VLA closed-loop executor.
  * ``pi05_server``   (server) — runs the π0.5 inference service, has NO pybullet.

To keep the server environment importable, this package MUST stay lightweight:
do NOT import client-side modules (``observation.py`` / ``vla_executor.py``)
here, because they transitively import LIBERO / robosuite. ``openpi_bridge.py``
contains the public model-service client. ``external_model_server/openpi_server.py``
loads the official OpenPI policy through an adapter and exposes the shared
model-service contract; clients never import the model objects.
"""
