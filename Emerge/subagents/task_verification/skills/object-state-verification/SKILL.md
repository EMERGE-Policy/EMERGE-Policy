---
name: object-state-verification
description: Verify whether requested object states and visible spatial relations are currently satisfied from multi-camera images. Use after embodied actions to check outcomes such as held, released, inside, on, open, closed, upright, clear, or obstructed without assuming that action completion means task success.
---

# Object State Verification

1. Call `observe_scene` and inspect every full camera view before judging the outcome.
2. Decompose the requested outcome into the smallest set of required, visually observable predicates. Express only positive success requirements; do not submit a failure alternative as another required predicate. Verify only conditions relevant to the request.
3. Cross-check object identity and each relation across views. Use `satisfied` only with direct visible support, `not_satisfied` only with direct contradictory evidence, and `uncertain` when identity, occlusion, or viewpoint prevents a reliable decision. For visually identical instances, another instance remaining at the source is not evidence that the moved one has the wrong provenance; if the current images cannot distinguish them, mark identity uncertain.
4. Judge containment from the target relative to the container opening and interior, not from simple image overlap. Judge release from visible separation between the gripper and object. Judge held only when the intended object is visibly secured and unsupported in the gripper. Fingers touching or surrounding an object that still rests on its original support establish contact or alignment, not a secure grasp. When the requested phase includes lifting, require the object to be visibly lifted from that support with the gripper.
5. Call `submit_task_verification` exactly once with every required predicate and one brief scene-context sentence. The tool computes the overall outcome from the predicate states.
6. Finish with at most one short sentence. Do not add claims that were not submitted.

Do not estimate coordinates, plan recovery, or control the robot. Do not infer hidden contact, force, stability, or containment from the requested action or from the fact that execution stopped. Report an unobservable condition as `uncertain`.
