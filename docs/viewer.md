# Viewer guide

## Playback and Viser

Play a trained on-policy learner with the task saved in its checkpoint.

```sh
newrl play checkpoint=path/to/checkpoint.pt
```

Playback defaults to four worlds and deterministic actions. Use `num_envs=1`
for a single world, `deterministic=false` to sample actions, or
`viewer=none steps=100` for a headless run.
Use `uv run --extra viser newrl play task=AnymalLocoFlat viewer=viser`
to open playback in a browser. Both web viewers use port `8081` by default.
If it is unavailable, they request a free port from the operating system and
print the resulting URL. Set `viewer.port` to choose a different preferred
port. A value of `0` always requests an OS-assigned port. `viewer.host` changes
Viser's listen address.
In the browser's **Robot view** panel, choose a world and robot, then enable
**Follow robot**. Enable **Camera frustums** to see the selected robot's
camera mount and field of view. Use **Camera** if the robot has several
cameras. Active cameras show their latest RGB or depth image on the frustum.
For a Gaussian camera, enable **Gaussian scene** to display its full splat cloud
in the selected world. The first enable loads and uploads the cloud to that
browser client, which can take time for large scenes.
In depth images, near surfaces appear brighter and invalid pixels are black.
Training discards unused observations by default. For playback, set
`discard_unused_obs=true` to skip observation groups that the learner does
not use. This is the default for `play-timeline`. Playback without a learner
can skip all observation groups using this option. Set
`discard_unused_obs=false` to keep simulator camera images available in Viser.

## Custom viewer

The Python viewer backends and presets live in `newrl.visualizations`. The
frontend source and npm configuration live in `viewer/`.

The experimental browser viewer renders Newton meshes and static Gaussian
scenes with Three.js and Gaussian Splat Lite. Build it from a source checkout
with Node.js 22.12 or newer.

```sh
npm --prefix viewer ci
npm --prefix viewer run build
uv run newrl play task=AnymalLocoFlat viewer=browser num_envs=1
```

Open the loopback URL printed by playback. The browser interpolates poses and
smooths the selected follow target while preserving manual orbit controls.
Turn off **Follow** and click the viewport to move with **WASD**. Use **Q**
to descend and **E** to ascend. Hold **Shift** to move faster or **Ctrl** to
move more slowly. **X** also slows movement when the viewport has focus,
including for keys that Chrome reserves as Ctrl shortcuts.
Press **Space** in the viewport or click **Pause simulation** to stop
simulation stepping. The camera and recording remain active while paused.
Use `viewer.port` to select the preferred port. The **Scene** panel toggles
ground, robot visuals, collision geometry, camera frustums, and individual
Gaussian clouds. Frustums follow the selected world and remain available
when playback skips camera observations.
**Shadows** is on by default for mesh and plane terrain and turns off while
a Gaussian cloud is visible. Only robot geometry in view casts shadows.
Enable **1 m grid** to project a world-aligned, one-meter grid across the
ground. Fine lines fade smoothly with distance.
Gaussian clouds load the first time they are enabled. Pass
`viewer.splat_path=/path/to/scene.ply` (or `.spz`) to add a cloud already aligned to
simulation coordinates.

Generated terrains show a coordinate frame at each tile origin by default.
Toggle **Terrain origins** under **Markers**, or start with
`viewer.terrain_origin_frames=false` to hide them initially. Custom playback
scripts can register a `Marker` from `newrl.visualizations.markers` with
`viewer.add_marker()` before the first frame. Markers accept world poses or a
Newton body index and a body-relative pose. Use `viewer.set_marker_pose()`
to update a marker during playback.

Velocity tracking tasks draw the commanded planar velocity in blue and the
measured planar velocity in orange for the selected world. Both arrows use
the same scale. The command is the one the policy sees for the next decision.
Toggle either arrow under **Markers**, or set `viewer.task_visualizations=false`
to skip task and sensor marker reads entirely. Height-map sensors draw green
dots at valid terrain intersections for the selected world. Toggle each map
under **Markers**. Contact sensors configured with `track_contact_position=True`
draw pink dots at each body's most recent valid touchdown. These remain visible
in the air until the next touchdown or reset. Toggle them under **Last contacts**.
Timeline frames keep the marker values
that were drawn when the frame was published.

`G1LAFAN1Tracking` shows a translucent G1 ghost at each world's current
reference pose. Toggle **Reference motion** under **Scene**. Each ghost
follows its own clip and frame, including when scrubbing the timeline.

Open **Camera preview** to inspect the selected world's cached RGB or depth
images. Enable **Show on frustum** to display the same image in the 3D scene.
Cameras already used by the simulator are available by default. Set
`viewer.camera_preview=true` to activate other configured cameras, including
Gaussian cameras whose observations were pruned. The latter can trigger
the gsplat CUDA build. Camera images are available during live playback,
including while paused.

The **Performance** panel shows browser FPS, main thread time per frame, draw
calls, and simulation speed relative to wall time. When `nvidia-smi` is
available, it also reports device-wide utilization and memory for the host GPU
running the simulation. The browser can render on a different GPU.

Use **Record** and **Stop & Save** to capture the live viewport at up to 30 FPS.
NewRL saves the video in `recordings/` under the directory where you launched
`newrl play`. Set `viewer.recording_dir=/path/to/recordings` to choose a different
directory. The viewer uses H.264 MP4 when the browser supports it and WebM
otherwise. The saved path appears in the viewer. Keep the tab and playback
open until the save completes.

## Rewind and branch a rollout

```sh
uv run newrl play-timeline task=AnymalLocoFlat
uv run newrl play-timeline checkpoint=path/to/checkpoint.pt deterministic=false
```

`play-timeline` opens one world in the custom viewer. The bottom timeline
shows simulation seconds and the retained time range. Scrub it or select
**Play history** to pause the live simulation and inspect the recorded poses.
Click an orange episode marker to jump to the reset at that time. Use
**Show terminal** or **Show reset** to switch between the two poses.
**Branch from here** restores the selected decision frame and
resumes simulation. With `deterministic=false`, the policy samples fresh actions.
The original run remains selectable while its frames are retained. Terminal
poses can be viewed, and their following reset poses are valid branch points.
**Return to live** shows the current simulation, which stays paused until resumed.

History lives in RAM for the current session. New frames evict the oldest
across all branches when the history reaches `history_frames` (3000 by default)
or `history_mib` (256 MiB of pose and tensor data by default).

## Gaussian camera sensors

Projects with a static Gaussian splat PLY can use `GaussianCameraConfig`
from `newrl.sensors` with `RGBCameraObservation` or `DepthCameraObservation`.
Set `mode` to `rgb`, `depth`, or `rgbd`. Configure `world_from_cloud` to
align a standalone PLY and install the `gsrender` extra with `uv sync --extra gsrender`.
Gaussian cameras default to 20 Hz. Set `frequency_hz` to change the rate or
`update_every` to specify a fixed number of control steps between frames.
Gaussian depth measures expected forward-axis depth and reports pixels below
`min_alpha` as invalid.

## Reconstructed scenes

Use the standalone gsrender editor to record a camera trajectory and export
`terrain.ply` and `terrain.json`. Set `ReconstructedTerrainConfig.directory`
to the exported terrain directory. Use the source `scene.json` as
`GaussianCameraConfig.source` or `viewer.splat_path` to apply the same
`worldFromSource` transform used during terrain export. For Gaussian camera
sensors, the manifest needs a 3DGS PLY in `cloud` or `renderCloud`.

The editor setup and manifest format are documented in the [gsrender repository](https://github.com/LearningFromEmbodiedExperienceLab/gsrender).

On a development machine with a sibling gsrender checkout, first install the
extra to resolve the backend dependencies, then overlay the local package.

```sh
uv sync --extra gsrender
uv pip install --python .venv/bin/python --no-deps -e ../gsrender
uv run --no-sync newrl play ...
```

Run the editable install again after a project sync to restore the overlay.

## Chrome WebGPU troubleshooting

On Linux with an NVIDIA hybrid graphics setup, check `chrome://gpu` if Chrome
reports that WebGPU is unavailable or picks the integrated GPU. If you have
the `webgpu-chrome` launcher, pass it the URL printed by playback. The
equivalent command uses a dedicated Chrome profile and forces NVIDIA offload
with Vulkan on X11. Set `VIEWER_URL` to the printed URL before running it:

```sh
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
google-chrome-stable \
  --user-data-dir="${XDG_CONFIG_HOME:-$HOME/.config}/webgpu-chrome" \
  --ozone-platform=x11 \
  --enable-unsafe-webgpu \
  --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE \
  "$VIEWER_URL"
```

Use that profile for local viewer sessions. Append `?renderer=webgl` to the
viewer URL to explicitly select WebGL2 if WebGPU is still unavailable or unstable.
