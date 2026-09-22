"""Public identifiers for the model-specific payloads (no backend imports)."""

from external_model_server.model_service.contracts import ServiceExpectation

VGGT = ServiceExpectation("vggt", "emerge.vggt.request.v1", "emerge.vggt.response.v1")
SAM3 = ServiceExpectation("sam3", "emerge.sam3.request.v1", "emerge.sam3.response.v1")
OPENPI = ServiceExpectation("openpi", "emerge.openpi.request.v1", "emerge.openpi.response.v1")
