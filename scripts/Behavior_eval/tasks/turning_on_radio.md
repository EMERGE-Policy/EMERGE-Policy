## turning_on_radio execution contract

This benchmark task requires manipulating the radio before operating its power
control. Treat the following phase order as part of the task definition, while
using EMERGE's generic skills and the active R1Pro action contract to implement
each phase.

1. **Locate the whole radio with the head RGB-D view.** Ask `object_location`
   for the radio receiver as one object in `robot_base`. Use wrist views only as
   supporting evidence. Do not search for or localize the red power button in
   the radio's initial pose.
2. **Approach the radio body.** Use the measured radio geometry for a safe
   pre-grasp approach. The target of this phase is the radio body, not a knob,
   dial, or button.
3. **Grasp and reorient the radio.** Use both arms as needed. One arm must
   secure and lift the radio, then turn it so the face containing the red power
   button is visibly facing upward. Use the other arm to stabilize the object
   when required. A hand merely touching the radio does not complete this
   phase. Keep the current PLAN subgoal limited to grasping and reorientation
   so the visual monitor can stop a VLA chunk once that physical state is
   achieved.
4. **Verify the new pose.** From fresh head and wrist observations, confirm
   that the radio is stably supported and the red button is exposed on top.
   Until this condition is visually established, do not localize or press the
   button.
5. **Relocalize the exposed button.** Only now ask `object_location` for the
   red power button using the new RGB-D observations. Reject a mask that covers
   the surrounding dial or radio housing.
6. **Press and hold with the other arm.** Approach the measured exposed button
   with the arm that is not securing the radio. Maintain contact until the
   button or indicator changes from red to green, then read the official
   evaluator state. Visual color evidence guides the action, but official
   `evaluation.success` remains the final success signal.

If grasping or reorientation fails, recover that phase before reasoning about
the button. Never replace the grasp-and-turn phase with a direct press against
the radio in its initial pose.
