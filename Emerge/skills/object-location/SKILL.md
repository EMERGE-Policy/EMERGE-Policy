---
name: object-localization
description: Identify requested objects from calibrated multi-camera images, verify appearance-based SAM3 candidate masks, locate confirmed targets in the world frame, and separately report nearby scene context. Use for embodied tasks that require finding, grasping, placing, or safely approaching visible objects.
---

# Object Localization

Follow this sequence:

1. Treat each incoming object name together with any support, containment, proximity, direction, or between relation as the semantic target identity. Preserve every identity-defining relation and its visible reference object.
2. Call `observe_scene` and inspect every full camera view before describing any target.
3. Use general visual and semantic knowledge to compare the requested object kind with evidence visible in the current scene. Cross-check each physical candidate, relation reference object, and requested spatial relation across views. Never invent an appearance attribute or relation that the images do not show.
4. Inventory all distinct physical candidates and visible reference objects that could satisfy the request before choosing one. Do not select only the most prominent candidate.
5. Compare appearance and identity-defining support, containment, proximity, direction, or between relations. When same-kind candidates look alike, the requested relation is mandatory identity evidence.
6. Assign each physical candidate a unique `candidate_id`. Generate a concise English prompt grounded in its visible appearance and, when required to distinguish instances, its visible relation to the named reference object.
7. Include a requested visible spatial relation in the SAM3 prompt when it identifies the intended instance, for example `patterned bowl next to the red cookies box`. Exclude coordinates and irrelevant nearby objects. Never borrow attributes or relations from another object.
8. Call `segment_candidates` with all plausible candidates together. Inspect every returned full-view overlay separately. Record the exact per-view instance covering the same physical object and verify its requested relation to the same reference object.
9. A relational candidate is usable only when at least two verified views support both the same object and the requested relation. A mask in two views is insufficient while instance identity remains unresolved. VGGT geometry is reused within the run.
10. Decide semantic identity only after reviewing the original images and candidate overlays. Reject an identical-looking candidate that violates the requested relation; SAM3 confidence is not identity evidence.
11. Call `locate_candidates` with the confirmed `candidate_id`, requested semantic `object_key`, and explicitly checked `verified_instances`, pairing each view with the correct instance number. Never assume instance ranks stay stable across views.
12. Select the best-supported candidate using both appearance and requested relation. Report ambiguity rather than guessing when the relation cannot distinguish the remaining candidates. Never localize an unconfirmed target.
13. Reuse measured coordinates from `locate_candidates`; never estimate coordinates from images. In the final response, provide only brief qualitative scene context. Keep irrelevant context out of SAM3 prompts, but preserve every identity-defining relation and reference object.

Treat `found: false` as unlocated. A successful mask is not sufficient by itself: appearance and every requested spatial relation must identify the same physical instance.
