# Adding robot assets and definitions

NewRL separates asset preparation from robot configuration. An asset supplies
the geometry, articulation, inertial properties and collision shapes. A robot
factory supplies the asset path, initial pose, actuator settings, action
scales and joint/body symmetry mappings through `RobotConfig`.

## Asset ownership

- Keep each core robot factory and its assetx recipe in one [robot family module](../src/newrl/assets/robots/). Shared cache handling lives in [cooking.py](../src/newrl/assets/cooking.py).
- Keep project-specific assets, factories and acquisition code in the
  project's own package. Projects may ship USD files and their dependencies,
  fetch prepared assets from cloud storage, or cook vendor sources with assetx.
- Resolve shipped assets relative to the owning package and keep references
  to meshes and other USD layers portable. Include package data when
  distributing the project as a wheel.
- Acquire assets lazily, when a robot factory is invoked. Importing a module,
  registering presets and listing available robots should be inexpensive and
  work offline. Reuse complete cached assets.
- Prefer pinned vendor revisions or versioned prepared bundles so experiments
  can reproduce their geometry and dynamics.

## Register an assetx recipe

Assetx is the core workflow for converting and adapting vendor MJCF or URDF
assets into USD. It is also available to projects.

Define a module-level function accepting the vendor directory as a
`Path` and returning a `MujocoAsset`. Compose assetx transformations
to normalize names, adapt collision geometry and remove source-specific
actuators, sensors and scene settings where appropriate. Register a
`Recipe(name, build_function, vendor_url)` in the `asset` group.

For core recipes, add the entry alongside the existing recipes in
`newrl.assets.robots`. For project recipes, use the project registration callback
and a namespaced name, such as `my_project:custom_robot`. Use the same
name in the recipe and its registration.

The robot factory resolves the cooked asset using
`cooked_usd(cooked_robot(name, robot_model_dir()))`.
Complete bundles are reused even after recipe changes. Missing or incomplete
bundles are cooked on demand. The default cache is `.cache/robots` under
the NewRL installation, with `ROBOT_MODEL_DIR` available to override it.
Namespaced recipes occupy separate subdirectories, such as
`.cache/robots/my_project/custom_robot`.

```sh
newrl assets list
newrl assets rebuild as2w
newrl assets rebuild my_project:custom_robot
```

Use explicit rebuilds to apply changes to cooking recipes. These commands
manage registered assetx recipes. Projects manage other asset sources through
their own acquisition code.

## Define a robot factory

Supply a zero-argument function at module scope that returns a fresh
`RobotConfig`. The scene calls this function when constructing the robot.
Its qualified Python name is saved in checkpoints, so keep the owning package
installable and the factory importable.

For example, a project shipping an AS2W-compatible USD can reuse the core
actuator and symmetry settings through a shared configuration factory.
For a completely independent robot definition, construct the configuration
directly using the following fields.

- `asset` is the resolved path to the USD articulation.
- `position` and `joint_position` define root placement in meters and
  nominal joint positions in radians or meters, as appropriate to each joint.
- `actuators` defines groups of joints with PD gains, armature, effort and
  velocity limits, friction, and optional passive `viscous_damping`. The
  alternative is to provide explicit stiffness, damping and armature values.
- `action_scale` defines the action-to-target scaling. Use
  `velocity_joints` for joints controlled by velocity targets, such as wheels.
- `joint_mirror` and `body_mirror` describe symmetry for mirrored
  observations and actions when using symmetry augmentation.

Use joint names or full-match regular expressions for per-joint settings
and actuator groups. Check the resulting joint ordering, nominal pose and
actuator coverage against the imported articulation.

For a core robot, register its factory in the `robot` group in
`newrl.assets.robots`, alongside the existing factories.

```python
cs.store("custom_robot", custom_robot, group="robot")
```

For a project robot, register the factory using the project's entry point.
For example, the project's `pyproject.toml` contains

```toml
[project.entry-points."newrl.projects"]
my_project = "my_project:register"
```

and its package exposes

```python
from .robots import custom_robot


def register(store, project):
    store.store(f"{project}:custom_robot", custom_robot, group="robot")
```

Register optional assetx recipes in the same callback. Install the project
into the environment used to run NewRL. Custom scripts can call
`newrl.projects.load_projects()` before resolving registered presets.

## Use and validate the robot

Use the factory directly in task configurations.

```python
task = EnvironmentConfig(
    robots={"robot": custom_robot},
    # Specify the MDP, terrain and other task settings here.
)
```

Use the CLI to discover registered factories and replace a compatible robot
in a task. The robot must satisfy that task's body and joint selectors,
observation dimensions and action interface.

```sh
newrl robots list
newrl play task=AS2WSphereLocoFlat task.robots.robot=newrl-atec:ATECAS2W
newrl play task=AS2WSphereLocoFlat task.robots.robot=newrl-atec:ATECAS2W viewer=none task.num_envs=4 steps=20
```

The ATEC example uses the project's independent spherical-wheel AS2W
definition and requires the ATEC package to be installed.

Before finishing a robot addition, verify that

- The factory appears in `newrl robots list` without triggering asset
  acquisition, and the selected factory survives a config save/load round trip.
- The factory returns a valid `RobotConfig` and its `build()` method
  imports the USD and configures the expected bodies and joints.
- A small rollout resets and steps successfully using the intended solver.
  Inspect the initial pose, collision geometry, joint motion and action scales
  in the viewer.
- For assetx recipes, a complete cached bundle is reused and an explicit
  rebuild applies recipe changes. Run `tests/test_assets.py` and
  `tests/test_robots.py` when changing these paths.
