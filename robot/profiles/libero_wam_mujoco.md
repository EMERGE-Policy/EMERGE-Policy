# LIBERO MuJoCo WAM Embodiment

Driver: libero_mujoco
Policy backend: Cosmos Policy WAM

The embodiment uses a Panda robot controlled by LIBERO / robosuite OSC_POSE at
20 Hz. All poses use metres and world-frame XYZW quaternions.

## Success Check

After every execute_robot_action, re-read ROBOT_STATE.md. If the fresh
robots.libero_mujoco.success value is true, stop immediately. A WAM
step_completed result is not task success.

## WAM Action Contract

Use wam_execute for contact-sensitive control. The evaluator owns the complete
task_instruction and conditioning_mode. Supply one phase_instruction describing
exactly the current unfinished phase plus a positive step budget. Do not call
vla_execute in this profile.

The configured conditioning mode may encode the original task, the current
phase, or both, but PLAN.md still determines the current execution boundary.
Candidate search is automatic: Cosmos generates and scores the configured
best-of-N chunks, and Emerge selects the best candidate locally.

## Task Routes

Choose the route before the first robot action. An earlier route takes priority.

### Continuous WAM Tasks

For microwave, wine-bottle, and moka-pot tasks, use one continuous wam_execute
call with the complete single-task semantics and the remaining evaluator action
budget. Do not split it into grasp, transport, placement, or articulation
subphases; do not call task_verification; stop after that call rather than
inserting recovery moves or an independent gripper command.

For a wine bottle going to a wine rack, start WAM directly so it controls the
large wrist rotation and grasp alignment. For a wine bottle going to a cabinet
top, one non-contact pre-grasp approach is allowed only when the bottle is
upright, current geometry proves a clear path, and wrist orientation is
preserved. Otherwise start WAM directly.

For bowl-grasp tasks, prefer WAM directly. A bowl rim admits many viable grasp
poses and a fixed rule approach can constrain or disturb alignment.

### Multi-Object Tasks

Handle one selected object at a time. Once its grasp-to-placement phase begins,
prefer one continuous WAM call with that object's complete semantics. Use a rule
move before the WAM phase only when a verified clearance or collision-avoidance
gate requires it. Never mention later objects in the active phase_instruction.

### General Geometry-Assisted Route

For other tasks, a geometry-assisted handoff is optional:

localize object -> optional non-contact pre-position -> WAM contact phase ->
verify -> optional clear-space transport -> WAM final placement

Use one pre-position only when object geometry is current, the end effector is
outside a useful pre-grasp region, and a collision-free offset endpoint is
known. Preserve wrist orientation and stop outside contact. Keep WAM in control
if any gate is uncertain or it is already making useful visual-contact progress.

After a verified grasp, use rule transport only when the held object,
destination geometry, and a collision-free route are all current. WAM performs
final alignment, descent, placement, and release.

## Collision and Clearance Rules

Before horizontal motion near clutter, raise above the tallest nearby movable
object and fixture. Keep high clearance through horizontal translation and wrist
adjustment, then descend vertically only above the selected target. Never sweep
laterally at object height.

After grasping beside an open drawer or fixture, withdraw a short measured
distance outward along a clear path before lifting. Do not lift through the
fixture collision envelope.

Before carrying beside, above, or through a fixed structure, localize the
relevant structure and movable objects and plan from full size_m extents, not
only centers. Leave clearance for both the gripper and held object.

## Fixed Structures and Destinations

- Microwave: microwave_1; interior microwave_1_heating_region.
- Drawer cabinets: wooden_cabinet_1 or white_cabinet_1; interiors top_region,
  middle_region, and bottom_region; cabinet top top_side.
- Two-layer shelf: wooden_two_layer_shelf_1; top_region, bottom_region, top_side.
- Wine rack: wine_rack_1; destination wine_rack_1_top_region.

object_location measures geometry only. WAM handles visual approach, handle
contact, and opening or closing a drawer, cabinet door, or microwave door as one
coherent action. Do not use attach to pull an articulated structure.

If an initial drawer relation only identifies a movable object, it does not
require opening or closing the drawer when the object is already visible and
accessible.

## Target Identity

Bind ambiguous targets with object class, visual signature, and original task
relation. Track the bound instance through motion and action history after it
moves. Never substitute another same-class object at the original source.

Useful appearance distinctions:

- Salad dressing: tall white bottle, dark-green cap, Creamy Ranch label.
- Ketchup: tall bright-orange squeeze bottle with light cap; not shorter BBQ sauce.
- Tomato sauce: short metal can with red-green tomato label; not a squeeze bottle.
- Butter: thin red-orange carton with FARM FRESH / BUTTER; not pudding box.
- Chocolate pudding: thicker brown box with light-blue border.
- Milk: tall red folded-top carton with cow graphic; not orange juice.
- Cream cheese: low pale rectangular box with dark blue-purple oval label.
- Plate: shallow cream plate with two dark-red rings.
- Akita bowl: deep bowl with white interior, dark radial motifs, thin yellow rim.
- Stove: square silver hot plate with black spiral burner and control knob.

The object_location specialist does not inherit this profile. Make each request
self-contained with appearance, original relation, and exclusions.

## WAM Budgets and Recovery

- 40–60 steps: local grasp or correction after a verified useful approach.
- 80–110 steps: ordinary visual approach, placement, or insertion.
- 220–300 steps: one continuous fixed articulated mechanism phase.

Outside fixed mechanisms, keep a call at or below 110 steps. Do not use a larger
budget to recover wrong-target or repeated no-progress behavior. After two
no-progress calls, change setup or strategy.

Continue WAM only during the same purposeful contact phase. Hand off only after
the intended local state is visibly established and a measured clear-space rule
target exists. Re-anchor if WAM affects the wrong object, loses the target,
disturbs an acceptable relation, or creates unsafe contact.

## Tool Responsibilities

- object_location: measure movable objects, destinations, and clearance-critical
  fixtures.
- attach: coarse non-contact pre-position and verified clear-space transport.
- wam_execute: visual approach, grasp, final alignment, placement, insertion,
  release, and articulated contact.
- task_verification: check uncertain relations or grasp state except where a
  continuous route forbids it.
