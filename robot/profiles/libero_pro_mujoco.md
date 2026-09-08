# LIBERO Pro MuJoCo Embodiment

Driver: `libero_mujoco`

The embodiment uses a Panda robot controlled by LIBERO / robosuite `OSC_POSE`
at 20 Hz. All poses use metres and world-frame XYZW quaternions.

## Success Check

After every `execute_robot_action`, re-read `ROBOT_STATE.md`. If the fresh
`robots.libero_mujoco.success` value is `true`, stop immediately; otherwise,
reassess the current subgoal.

## Location Result Acceptance

Before using any returned `position_m` or `size_m` for robot motion:

- `status: success` and the returned object name only mean the call completed;
  every required target and relation reference must be found and unambiguous.
- Check the result against the original identity binding, latest verified
  relation, action history, and last known location. Reject any contradiction.
- Mutually exclusive identities must not resolve to the same or substantially
  overlapping footprint. Never move to a rejected result; retry by locating all
  plausible candidates and references together, or keep the target unlocated.

## Task Route

Choose the applicable route before the first robot action. A contact-control
task still follows the fixed-structure and movable-object constraints below
whenever they apply.

### VLA Contact-Control Tasks

For microwave, wine-bottle, moka-pot, drawer, and cabinet tasks, VLA remains
the contact controller but **MUST** execute one verified phase at a time.
Localize the current target or fixture before contact. Once a phase succeeds,
later VLA requests must state that it is complete and forbid repeating it.
Never give VLA a complete multi-phase task in one request; after no progress,
retreat high, re-localize, and change the contact geometry.
`task_verification` is allowed between phases.

### Fixed Structures and Large Destinations

Use the LIBERO fixture and region names that correspond to the task:

For microwave ovens, drawer cabinets, shelves, and wine racks: before approaching any of these structures, or before moving an object beside, above, or through them, call object_location for the relevant structure and the object(s) involved to obtain their `position_m` and `size_m`. Plan your motion based on their full spatial extents, not merely their center points. Ensure sufficient clearance for both the gripper and the held object, so the path does not intersect the microwave body, cabinet body, any open drawer, or the shelf

When placing onto a high destination such as a cabinet or shelf top, first lift
vertically until the held object clears the destination's full top extent, then
move horizontally above it. Never approach a high destination at tabletop
height.

`object_location` only measures geometry. Let VLA handle visual approach,
handle contact, and opening or closing a drawer, cabinet door, or microwave door
as one coherent action. Do not use attach to pull an articulated structure.

After opening or pulling a drawer/cabinet, turning a knob, or placing an object
inside a fixture, **MUST** first move vertically to a high-clearance position
without rotating; only then reset with `move_to_pose` to a straight top-down
orientation (`orientation_euler: [-3.14156, 0.05681, 0.00049]`) before the next
movable-object attach or fixture re-engagement. Never rotate near the fixture or
reuse the preceding contact orientation.

When a task uses a relation such as "initially in the drawer" only to identify a
movable object, that relation does not itself require opening or closing the
drawer. If the object is already visible and accessible, localize it directly.
If it is unclear whether the structure blocks visibility or access, call
`task_verification` before the first robot action.

### Movable-Object Tasks

Use the following route for a normal pick-and-place task:

`localize object -> attach approach -> VLA grasp -> localize destination -> attach transport -> VLA place`

Example — `Put the bowl on the top of the drawer`:

`object_location(bowl + cabinet top) -> attach bowl -> VLA grasp -> high vertical lift -> object_location(cabinet top) -> horizontal transport above cabinet -> VLA place -> check success`

Use the bowl's original relation for identity, and use the measured cabinet body
and top-surface extent for transport and placement. Do not skip destination
localization because the cabinet is fixed.

Contact or visual overlap is not a verified grasp. First lift only in `+Z` by
at most `0.05 m` and require the same bound object to follow unsupported. If
the grasp is false or uncertain, do not transport or place.

Before every place, move the held object geometrically above the measured
destination center or opening; VLA performs only the final descent and release.
After release, rise vertically by `0.10 m` and verify that the object is stable
and the gripper is empty.

For a top-down bowl or mug grasp, prefer the right edge at
`[x, y + clamp(0.35 * size_y, 0.02, 0.04), z_clear]`, so the fingers straddle
the rim; a center grasp remains allowed when the edge is unsafe or occluded.
For a flat carton, prefer an exposed side edge instead of pressing its top.

On LIBERO tables, front/near is `+X`, back is `-X`, left is `-Y`, and right is
`+Y`. For "in front of a fixture", keep the destination outside the fixture
footprint: `target_x >= fixture_x + fixture_size_x/2 + object_size_x/2 + 0.02`,
with `target_y` near `fixture_y`. Push along that world-frame direction; never
push toward the fixture center or onto it.

Preserve task verbs and object roles: a push remains a push, and a destination
is never grasped unless the task explicitly asks to move it. Never repeat an
unchanged localization or VLA request. After two failed grasps or placements,
retreat high, re-localize, and change the approach. Do not move a distractor
merely to clear access unless the task requires it.

## Target Identity

For an ambiguous target, combine:

`object class + visual signature + original task relation`

Lighting may alter any single color cue, so prefer a combination of shape,
color, and label.

### Object Appearance Hints

- **Salad dressing:** tall white bottle, dark-green cap, "Creamy Ranch
  Dressing" label.
- **Ketchup:** tall, narrow bright-orange squeeze bottle with a short pale cap,
  tapered shoulders, and "Tomato Ketchup" label; never use the shorter, wider
  dark orange-brown BBQ bottle.
- **BBQ sauce:** shorter, wider squeeze bottle with a broad lower body;
  dark orange-brown with an orange flip-top cap and "BBQ Sauce" label. Never
  use the tall bright-orange ketchup bottle.
- **Tomato sauce:** short cylindrical metal can with a red-and-green tomato
  label and a silver top; not a squeeze bottle.
- **Butter:** very thin red-orange rectangular carton, usually lying flat, with
  yellow "FARM FRESH", white "BUTTER", and a cow on a light-blue panel; not the
  thicker brown-and-blue chocolate pudding box.
- **Chocolate pudding:** thicker brown box of instant pudding mix with a
  light-blue border and "CHOCOLATE PUDDING" text; not the thin red-orange butter
  carton.
- **Milk:** tall red carton with a folded triangular top, large white "Milk"
  text, and a black-and-white cow on a light-blue panel with a green bottom edge;
  not the yellow-orange juice carton with a large orange image.
- **Cream cheese:** low white or pale-blue rectangular box with a dark
  blue-purple oval label and white "Cream Cheese" text; not butter, chocolate
  pudding, or milk.
- **Porcelain mug:** all-white mug with a lightly dotted or textured body.
- **White-yellow mug:** white mug with prominent yellow outer panels and a
  yellow handle.
- **Plate:** shallow, nearly flat round white or cream plate with two dark-red
  concentric rings near the rim and a mostly plain center; no dense black radial
  motifs, no thin yellow rim, no black spiral, and no square metal base.
- **Akita black bowl / bowl:** deep round bowl with a white interior densely
  covered by black or charcoal radial motifs and a thin yellow rim; not the
  red-ring plate or the flat black spiral burner.
- **Stove:** separate square gray-silver hot plate with a dark circular spiral
  burner and a black control knob; distinguish it from the red-ring plate.
- **Basket:** light beige or cream woven rectangular basket with an open top and
  cut-out handles.
- **Moka pot:** silver metal octagonal stovetop coffee pot with a black handle
  and top knob.
- **Wine bottle:** slender dark-green or nearly black glass bottle with a light
  rectangular wine label and a cork-colored top.
- **Wine rack:** angled wooden bottle rack with light tan panels and darker wood
  supports.
- **Wooden cabinet:** dark charcoal-brown wood drawer cabinet with metal
  handles.

### Akita Bowl and Plate Disambiguation

These objects are easy to confuse from above. The bowl is deep with dense black
radial motifs and a thin yellow rim; the plate is shallow with a plain center
and two dark-red rings. Each SAM3 prompt **MUST** explicitly exclude the other.

- Localize a source bowl and destination plate separately. Reject the result if
  they use the same mask or have nearly identical coordinates and dimensions.
- The bowl must satisfy its original relation in at least two views, such as
  being on the stove or inside the top drawer rather than on the cabinet top.

### Stove and Plate Disambiguation

When locating either one, request stove and plate as separate candidates and
decide by structure, never color. A stove match **MUST** include the square base,
circular spiral burner, and separate black knob; reject a burner-only mask. A
plate match **MUST** be a thin round dish with no square base, spiral, or knob.
Reject overlapping masks or nearly identical coordinates.

### Alternate Color Retry

For every object instance, the first query **MUST** use only its standard hint.
For ketchup or BBQ sauce, localize both classes together but still use only
their standard cues in that first request and reject overlapping or identical
results. Retry once with the instance's alternate color only after the standard
query explicitly finds no reliable match; another object, visible color,
benchmark knowledge, timeout, or error never unlocks the retry. Preserve all
non-color cues and, after a successful retry, use that color in all later
location, verification, and VLA requests for the instance.

- **Salad dressing:** replace the white and dark-green color cues with a
  red-dominant bottle and label.
- **Ketchup:** replace the bright-orange bottle cue with vivid blue; retain the
  tall, narrow squeeze-bottle shape, short cap, and tapered shoulders.
- **BBQ sauce:** replace the dark orange-brown bottle cue with olive green;
  retain the shorter, wider package, broad lower body, cap, and label.
- **Tomato sauce:** replace the red-and-green label cue with a yellow-dominant
  can and label.
- **Butter:** replace the red-orange carton cue with green.
- **Chocolate pudding:** replace the brown carton cue with green.
- **Milk:** replace the red carton cue with yellow.
- **Cream cheese:** replace the white or pale-blue box cue with red.
- **Plate:** replace the white or cream plate cue with yellow; retain the
  shallow round plate shape.
- **Stove:** replace the gray-silver hot-plate cue with yellow; retain the dark
  circular burner and control-knob structure.
- **Basket:** replace the light beige or cream woven cue with red or pink-red.
- **Moka pot:** replace the silver metal body cue with yellow or gold; retain
  the black handle and octagonal coffee-pot shape.
- **Wine bottle:** replace the dark-green or nearly black bottle cue with a
  white or pale-green bottle body and a light or plain label.
- **Wine rack:** replace the light-tan wood cue with predominantly dark-brown
  wood.
- **Wooden cabinet:** replace the dark charcoal-brown wood cue with yellow or
  pale-yellow wood.

### Identity Binding and Tracking

- Bind an instance using its original task relation, such as "the Akita bowl
  initially in the top drawer" or "the plate initially on the left".
- Localize visually identical left/right objects together and record both
  initial bindings before either moves. These roles are immutable and must not
  be reassigned from current image position or a later single-object result.
- If the task identifies the manipulated object by an explicit spatial relation,
  assume that two or more similar instances are present. Every subagent request
  **MUST** emphasize that the exact relation is identity-defining and preserve it
  in observation, candidate prompts, selection, and retries; never reduce the
  target to its generic object class or drop the relation.
- Every `object_location` request **MUST** name the original spatial relation
  and its reference object(s). Preserve the relation in the SAM3 prompt and
  cross-view instance selection. If it cannot be verified, localize the target
  and references together and report ambiguity rather than guessing.
- After an object moves, preserve its original binding while updating its
  current relation from verified observation and action history. Recovery
  requests must include its last location, current support or reference, all
  same-class candidates, and explicit exclusions.
- Reject a recovery result that selects an untouched distractor or violates the
  verified support, relative relation, or height. Repeated agreement does not
  make the same contradictory localization valid.
- Another same-class object near the original source does not invalidate an
  identity already tracked through motion.
- If continuity is lost, re-establish it from action history and location; do
  not invent a visual difference.

### Identity Information for Subagents

The `object_location` specialist does not inherit this profile. Every request
**MUST** include the target appearance, original relation and reference(s), and
exclusions. Preserve the relation in SAM3 prompts and per-view selection. On an
alternate-color retry, replace conflicting color cues.

Each VLA instruction describes only the current action phase while preserving
the object name, attributes, and target relation from the original LIBERO task.
Do not give VLA later phases in the same instruction.

## Tool Responsibilities

- `object_location`: localize the current movable object, destination, or
  clearance-critical fixture while preserving every identity-defining spatial
  relation and its reference object(s).
- `attach`: perform measured coarse approach, lifting, and transport while
  respecting measured object and fixture extents.
- `vla_execute`: perform grasping, final alignment and placement, and visually
  guided contact with articulated structures.
- `task_verification`: check task relations, grasp state, or uncertain structure
  state between action phases.
