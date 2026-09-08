# LIBERO MuJoCo Embodiment

Driver: `libero_mujoco`

The embodiment uses a Panda robot controlled by LIBERO / robosuite `OSC_POSE`
at 20 Hz. All poses use metres and world-frame XYZW quaternions.
##
**Remember to update your plan on time.**

The plan should be updated and checked on time according to the planning phase.

call `task_verification`
## Success Check

After every `execute_robot_action`, re-read `ROBOT_STATE.md`. If the fresh
`robots.libero_mujoco.success` value is `true`, stop immediately; otherwise,
reassess the current subgoal.

## Task Route

Choose one of the following routes before the first robot action. An earlier
route takes priority over a later route.

### VLA-Only Tasks

For microwave, wine bottle, and moka pot tasks, use only `vla_execute` for robot
control, with `step: 320`, and never call `task_verification`. `object_location`
may still measure targets and obstacles because observation is not robot
control. Give VLA the complete task semantics directly instead of splitting the
instruction into grasp, transport, placement, or articulation subphases. This
route overrides the default control workflow below.

### Fixed Structures and Large Destinations

Use the LIBERO fixture and region names that correspond to the task:

- Microwave: `microwave_1`; its interior placement region is
  `microwave_1_heating_region`.
- Drawer cabinets: `wooden_cabinet_1` or `white_cabinet_1`; their drawer
  interiors are `top_region`, `middle_region`, and `bottom_region`, producing
  names such as `wooden_cabinet_1_top_region` and
  `white_cabinet_1_bottom_region`. The cabinet top is `top_side`.
- Two-layer shelf: `wooden_two_layer_shelf_1`; its regions are
  `wooden_two_layer_shelf_1_top_region`,
  `wooden_two_layer_shelf_1_bottom_region`, and
  `wooden_two_layer_shelf_1_top_side`.
- Wine rack: `wine_rack_1`; its bottle destination is
  `wine_rack_1_top_region`.

Before approaching one of these structures, or carrying an object beside,
above, or through it, call `object_location` for the relevant structure and
objects to measure `position_m` and `size_m`. Plan from their full spatial
extents, not only their center points. Leave clearance for the gripper and held
object so the path does not cross a microwave body, cabinet body, open drawer,
or shelf.

`object_location` only measures geometry. Let VLA handle visual approach,
handle contact, and opening or closing a drawer, cabinet door, or microwave door
as one coherent action. Do not use attach to pull an articulated structure.

When a task uses a relation such as "initially in the drawer" only to identify a
movable object, that relation does not itself require opening or closing the
drawer. If the object is already visible and accessible, localize it directly.
If it is unclear whether the structure blocks visibility or access, call
`task_verification` before the first robot action, except on the VLA-only route.

### Movable-Object Tasks

Use the following route for a normal pick-and-place task:

`localize object -> attach approach -> VLA grasp -> localize destination -> attach transport -> VLA place`

## Target Identity

For an ambiguous target, combine:

`object class + visual signature + original task relation`

Lighting may alter any single color cue, so prefer a combination of shape,
color, and label.

### Object Appearance Hints

- **Salad dressing:** tall white bottle, dark-green cap, "Creamy Ranch
  Dressing" label.
- **Ketchup:** tall bright-orange squeeze bottle, light gray or white cap,
  "Tomato Ketchup" label; not the shorter dark orange-brown BBQ bottle.
- **BBQ sauce:** shorter dark orange-brown squeeze bottle, orange flip-top cap,
  "BBQ Sauce" label.
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
- **Plate:** shallow round white or cream plate with two dark-red concentric
  rings near the rim; no black spiral and no square metal base.
- **Akita black bowl / bowl:** deep round bowl with a white interior densely
  covered by black or charcoal radial motifs and a thin yellow rim; not the
  red-ring plate or the flat black spiral burner.
- **Stove:** separate square gray-silver hot plate with a dark circular spiral
  burner and a black control knob; distinguish it from the red-ring plate.

### Identity Binding and Tracking

- Bind an instance using its original task relation, such as "the Akita bowl
  initially in the top drawer" or "the plate initially on the left".
- Bind visually identical bowls or plates by their initial relation and
  location. Left and right refer to initial scene roles.
- After an object moves, track the same instance through observed motion, action
  history, and its last known location.
- Another same-class object near the original source does not invalidate an
  identity already tracked through motion.
- If continuity is lost, re-establish it from action history and location; do
  not invent a visual difference.

### Identity Information for Subagents

The `object_location` specialist does not inherit this profile. Make every
request self-contained with the target's full appearance, original task
relation, and exclusions; a name such as ketchup, butter, or bowl alone is not
enough. Use the same identity in `task_verification` requests.

Except on the VLA-only route, each VLA instruction describes only the current
action phase while preserving the object name, attributes, and target relation
from the original LIBERO task. Do not give VLA later phases in the same
instruction.

## Tool Responsibilities

- `object_location`: localize the current movable object, destination, or
  clearance-critical fixture.
- `attach`: perform measured coarse approach, lifting, and transport while
  respecting measured object and fixture extents.
- `vla_execute`: perform grasping, final alignment and placement, and visually
  guided contact with articulated structures.
- `task_verification`: check task relations, grasp state, or uncertain structure
  state, except where the active task route forbids it.
