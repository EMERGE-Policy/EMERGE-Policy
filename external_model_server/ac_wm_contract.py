"""Cosmos3 forward-dynamics service, separate from Cosmos Policy WAM."""

from external_model_server.model_service.contracts import ServiceExpectation

SERVICE = "cosmos3_ac_wm"
INPUT_SCHEMA = "emerge.ac_wm.rollout.request.v1"
OUTPUT_SCHEMA = "emerge.ac_wm.rollout.response.v1"
EXPECTATION = ServiceExpectation(SERVICE, INPUT_SCHEMA, OUTPUT_SCHEMA)
