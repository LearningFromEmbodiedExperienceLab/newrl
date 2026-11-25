import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GaussianSplatRenderer, SplatMesh } from "gaussian-splat-lite";
import { Axis3d, Camera, createIcons, Layers3 } from "lucide";
import { setupCameraPreview } from "./camera-preview.js";
import { setupRecording } from "./recording.js";
import { setupTimeline } from "./timeline.js";

const status = document.querySelector("#status");
const targetSelect = document.querySelector("#target");
const follow = document.querySelector("#follow");
const entities = document.querySelector("#entities");
const freeCameraKeys = document.querySelector("#free-camera-keys");
const updateCameraHint = () => { freeCameraKeys.hidden = follow.checked; };
follow.addEventListener("change", updateCameraHint);
updateCameraHint();

function visibilityControl(name, checked, container = entities) {
  const row = document.createElement("label");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;
  const message = document.createElement("span");
  message.className = "asset-status";
  message.setAttribute("role", "status");
  row.append(input, ` ${name}`, message);
  container.append(row);
  return {input, message};
}

async function latest(url, onHeaders) {
  const response = await fetch(url, {cache: "no-store"});
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  onHeaders?.(response.headers);
  if (response.status === 204) return null;
  return response.json();
}
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function cameraFrustum({fovY, aspect, mount}, material) {
  // Both sensor types provide an OpenGL mount with -Z forward and +Y up.
  const depth = 0.55;
  const height = depth * Math.tan(fovY / 2);
  const width = height * aspect;
  const corners = [
    [-width, -height, -depth], [width, -height, -depth],
    [width, height, -depth], [-width, height, -depth],
  ];
  const vertices = corners.flatMap((corner, i) => [
    0, 0, 0, ...corner, ...corner, ...corners[(i + 1) % 4],
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  const wireframe = new THREE.LineSegments(geometry, material);
  wireframe.position.fromArray(mount);
  wireframe.quaternion.fromArray(mount, 3);
  const image = new THREE.Mesh(
    new THREE.PlaneGeometry(2 * width, 2 * height),
    new THREE.MeshBasicMaterial({side: THREE.DoubleSide, toneMapped: false})
  );
  image.position.z = -depth;
  image.visible = false;
  wireframe.add(image);
  const body = new THREE.Group();
  body.add(wireframe);
  body.visible = false;
  return {body, image};
}

function addMarkers(scene, specifications) {
  const panel = document.querySelector("#markers");
  panel.hidden = !specifications.length;
  const controls = new Map();
  return specifications.map(spec => {
    const group = spec.group ?? {frame: "Frames", arrow: "Arrows", point: "Points"}[spec.kind];
    if (!controls.has(group)) {
      controls.set(group, visibilityControl(group, spec.visible, panel.querySelector("div")).input);
    }
    const input = controls.get(group);
    let geometry;
    if (spec.kind === "frame") {
      geometry = new THREE.AxesHelper(spec.size);
      geometry.material.depthTest = false;
      geometry.renderOrder = 2;
    } else if (spec.kind === "arrow") {
      geometry = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0), new THREE.Vector3(), spec.size, spec.color
      );
    } else {
      geometry = new THREE.Mesh(
        new THREE.SphereGeometry(spec.size, 12, 8),
        new THREE.MeshBasicMaterial({color: spec.color})
      );
    }
    const root = new THREE.Group();
    root.add(geometry);
    root.position.fromArray(spec.offset);
    geometry.position.fromArray(spec.pose);
    geometry.quaternion.fromArray(spec.pose, 3);
    root.visible = input.checked && spec.body === null;
    scene.add(root);
    return {spec, root, geometry, input};
  });
}

function freeCameraMovement(camera, orbit, canvas) {
  const keys = new Set();
  const movementKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"]);
  const modifierKeys = new Set(["ControlLeft", "ControlRight", "ShiftLeft", "ShiftRight"]);
  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const movement = new THREE.Vector3();
  canvas.tabIndex = 0;
  canvas.addEventListener("pointerdown", () => canvas.focus({preventScroll: true}));

  addEventListener("keydown", event => {
    if (follow.checked || event.metaKey || event.altKey ||
        event.target.closest?.("select, textarea, input:not([type=checkbox]), [contenteditable]")) return;
    if (modifierKeys.has(event.code)) {
      keys.add(event.code);
    } else if (movementKeys.has(event.code) || (event.code === "KeyX" && event.target === canvas)) {
      event.preventDefault();
      keys.add(event.code);
    }
  });
  addEventListener("keyup", event => keys.delete(event.code));
  addEventListener("blur", () => keys.clear());
  document.addEventListener("visibilitychange", () => keys.clear());
  follow.addEventListener("change", () => keys.clear());

  return dt => {
    if (follow.checked) return;
    const ahead = Number(keys.has("KeyW")) - Number(keys.has("KeyS"));
    const sideways = Number(keys.has("KeyD")) - Number(keys.has("KeyA"));
    const vertical = Number(keys.has("KeyE")) - Number(keys.has("KeyQ"));
    if (!ahead && !sideways && !vertical) return;

    forward.subVectors(orbit.target, camera.position);
    forward.z = 0;
    forward.normalize();
    right.crossVectors(forward, camera.up);
    movement.copy(forward).multiplyScalar(ahead).addScaledVector(right, sideways);
    movement.z = vertical;
    const fine = keys.has("ControlLeft") || keys.has("ControlRight") || keys.has("KeyX");
    const fast = keys.has("ShiftLeft") || keys.has("ShiftRight");
    const speed = fine ? 0.75 : fast ? 12 : 3;
    movement.normalize().multiplyScalar(speed * dt);
    camera.position.add(movement);
    orbit.target.add(movement);
  };
}

async function main() {
  createIcons({icons: {Axis3d, Camera, Layers3}});
  const forceWebGL = new URLSearchParams(location.search).get("renderer") === "webgl";
  const renderer = new WebGPURenderer({antialias: false, forceWebGL});
  await renderer.init();
  document.querySelector("#backend").textContent = renderer.backend.isWebGPUBackend ? "WebGPU" : "WebGL2";
  const viewport = document.querySelector("#viewport");
  viewport.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1c2127);
  const camera = new THREE.PerspectiveCamera(55, 1, 0.04, 1000);
  camera.up.set(0, 0, 1);
  camera.position.set(3, -4, 2.3);
  const orbit = new OrbitControls(camera, renderer.domElement);
  const moveFreeCamera = freeCameraMovement(camera, orbit, renderer.domElement);
  orbit.target.set(0, 0, 0.45);
  orbit.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x777f89, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2);
  light.position.set(3, -4, 7);
  scene.add(light);
  const splats = new GaussianSplatRenderer({renderer});
  scene.add(splats);
  let recordingActive = () => false;
  const resize = () => {
    if (recordingActive()) return;
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    const width = Math.max(1, viewport.clientWidth);
    const height = Math.max(1, viewport.clientHeight);
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  addEventListener("resize", resize);
  new ResizeObserver(resize).observe(viewport);
  resize();

  let manifest;
  while (!(manifest = await latest("/scene.json"))) await delay(150);
  recordingActive = setupRecording(renderer.domElement, resize);
  const geometries = new Map();
  for (const [name, mesh] of Object.entries(manifest.meshes)) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(Uint8Array.fromBase64(mesh.points).buffer), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(Uint8Array.fromBase64(mesh.indices).buffer), 1));
    if (mesh.normals) geometry.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(Uint8Array.fromBase64(mesh.normals).buffer), 3));
    else geometry.computeVertexNormals();
    geometries.set(name, geometry);
  }
  const layers = new Map();
  for (const batch of Object.values(manifest.instances)) {
    for (const name of batch.layers.flat()) layers.set(name, null);
  }
  for (const name of [...layers.keys()].sort()) {
    layers.set(name, visibilityControl(name, !name.endsWith(" collision")).input);
  }
  const batches = new Map();
  for (const [name, batch] of Object.entries(manifest.instances)) {
    const geometry = geometries.get(batch.mesh);
    if (!geometry) continue;
    const objects = Array.from({length: batch.count}, (_, i) => {
      const opacity = batch.opacities?.[i] ?? 1;
      const color = batch.colors?.[i];
      const material = new THREE.MeshStandardMaterial({
        color: color ? new THREE.Color(...color) : 0xaeb7c3,
        opacity, transparent: opacity < 1, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.visible = false;
      if (batch.scales) mesh.scale.fromArray(batch.scales[i]);
      scene.add(mesh);
      return {mesh, controls: batch.layers[i].map(name => layers.get(name))};
    });
    batches.set(name, objects);
  }

  manifest.targets.forEach(({world, name}, index) => {
    targetSelect.add(new Option(`World ${world} · ${name}`, index));
  });
  const frustumMaterial = new THREE.LineBasicMaterial({color: 0x40c0ff});
  const frustums = manifest.cameras.map(spec => {
    const {input} = visibilityControl(`Camera frustum · ${spec.name}`, false);
    const {body, image} = cameraFrustum(spec, frustumMaterial);
    scene.add(body);
    return {input, body, image};
  });
  const markers = addMarkers(scene, manifest.markers);
  const cloudOffset = new THREE.Vector3();
  function placeCloud(cloud, spec) {
    cloudOffset.fromArray(manifest.targets[Number(targetSelect.value)]?.offset ?? [0, 0, 0]);
    cloud.position.fromArray(spec.transform).add(cloudOffset);
    cloud.quaternion.fromArray(spec.transform, 3);
  }
  const clouds = new Map();
  for (const spec of manifest.clouds) {
    const {input, message} = visibilityControl(`Gaussians · ${spec.name}`, false);
    let pending;
    input.addEventListener("change", async () => {
      if (!input.checked) {
        clouds.get(spec)?.removeFromParent();
        message.textContent = "";
        return;
      }
      if (!clouds.has(spec)) {
        message.textContent = " · Loading…";
        try {
          pending ??= (async () => {
            const cloud = new SplatMesh({url: spec.url});
            try {
              await cloud.initialized;
              clouds.set(spec, cloud);
            } catch (error) {
              cloud.dispose();
              throw error;
            }
          })();
          await pending;
        } catch (error) {
          if (input.checked) {
            input.checked = false;
            message.textContent = ` · ${error.message}`;
          }
          return;
        } finally {
          pending = undefined;
        }
      }
      if (input.checked) {
        const cloud = clouds.get(spec);
        placeCloud(cloud, spec);
        scene.add(cloud);
        message.textContent = "";
      }
    });
  }
  targetSelect.addEventListener("change", () => {
    for (const [spec, cloud] of clouds) placeCloud(cloud, spec);
  });

  const frames = [];
  let liveFrame;
  let lastRender = performance.now();
  let statsAt = lastRender;
  let renderCount = 0;
  let cpuMs = 0;
  let drawCount = 0;
  let simulationTime;
  let previousSimulationTime;
  const stats = Object.fromEntries(
    ["fps", "cpu", "draws", "sim-speed", "gpu-util", "gpu-memory"]
      .map(id => [id, document.getElementById(id)])
  );
  async function pollGpu() {
    let gpu;
    try {
      gpu = await latest("/gpu.json");
    } catch {
      // The viewer may have closed while the browser tab stays open.
    }
    stats["gpu-util"].textContent = gpu?.utilization == null ? "Unavailable" : `${gpu.utilization}%`;
    stats["gpu-memory"].textContent = gpu?.usedMiB == null || gpu?.totalMiB == null
      ? "Unavailable" : `${gpu.usedMiB} / ${gpu.totalMiB} MiB`;
    setTimeout(pollGpu, 1000);
  }
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const worldOffset = new THREE.Vector3();
  function interpolatePose(object, from, to, alpha) {
    object.position.fromArray(from);
    position.fromArray(to);
    const t = object.position.distanceToSquared(position) > 1 ? 1 : alpha;
    object.position.lerp(position, t);
    object.quaternion.fromArray(from, 3);
    rotation.fromArray(to, 3);
    object.quaternion.slerp(rotation, t);
  }
  const followTarget = new THREE.Vector3();
  const delta = new THREE.Vector3();
  const pauseButton = document.querySelector("#pause");
  let paused = false;
  let pauseRequest = 0;
  function updatePauseButton(value) {
    paused = value;
    pauseButton.textContent = paused ? "Resume simulation" : "Pause simulation";
  }
  async function setPaused(value) {
    pauseRequest++;
    pauseButton.disabled = true;
    try {
      const action = value ? "pause" : "resume";
      const response = await fetch(`/playback/${action}`, {method: "POST"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      updatePauseButton((await response.json()).paused);
    } finally {
      pauseButton.disabled = false;
    }
  }
  function showFrame(frame, jump = false) {
    if (jump || (frames.length && frames.at(-1).frame.epoch !== frame.epoch)) frames.length = 0;
    frames.push({time: performance.now(), frame});
    if (frames.length > 16) frames.shift();
  }
  const timeline = manifest.timeline ? setupTimeline({
    setPaused,
    showFrame,
    showLive: () => {
      if (liveFrame) showFrame(liveFrame, true);
    },
  }) : null;
  const previewTexture = setupCameraPreview(
    manifest.cameras,
    () => manifest.targets[Number(targetSelect.value)]?.world,
    () => timeline?.isHistorical()
  );
  function togglePause() {
    if (!pauseButton.disabled) {
      if (timeline?.isHistorical()) timeline.returnToLive();
      setPaused(!paused).catch(error => {
        status.textContent = `Could not change simulation pause. ${error.message}`;
      });
    }
  }
  pauseButton.disabled = false;
  pauseButton.addEventListener("click", togglePause);
  addEventListener("keydown", event => {
    if (event.code !== "Space" || event.ctrlKey || event.metaKey || event.altKey ||
        event.target.closest?.("button, input, select, textarea, summary, a, [contenteditable]")) return;
    event.preventDefault();
    if (!event.repeat) {
      if (timeline?.isHistorical()) timeline.toggleHistory();
      else togglePause();
    }
  });
  async function poll() {
    try {
      const request = pauseButton.disabled ? null : pauseRequest;
      const frame = await latest("/frame.json", headers => {
        if (request === pauseRequest) {
          updatePauseButton(headers.get("X-Simulation-Paused") === "1");
        }
      });
      if (frame && frame.id !== liveFrame?.id) {
        if (frame.epoch !== liveFrame?.epoch) previousSimulationTime = undefined;
        liveFrame = frame;
        simulationTime = frame.time;
        if (!timeline?.isHistorical()) showFrame(frame);
      }
    } catch {
      status.textContent = "Playback disconnected";
    }
    setTimeout(poll, 20);
  }
  status.textContent = "Live";
  poll();
  pollGpu();
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    const dt = Math.min((now - lastRender) / 1000, 0.1);
    lastRender = now;
    if (frames.length) {
      // Render behind the newest packet so regular jitter cannot skip poses.
      const renderTime = now - 60;
      while (frames.length > 1 && frames[1].time <= renderTime) frames.shift();
      const start = frames[0];
      const end = frames[1] ?? start;
      const from = start.frame;
      const to = end.frame;
      const alpha = start === end ? 1 :
        THREE.MathUtils.clamp((renderTime - start.time) / (end.time - start.time), 0, 1);
      for (const [name, objects] of batches) {
        const poses = to.poses[name];
        const previous = from.poses[name] ?? poses;
        if (!poses) continue;
        for (let i = 0; i < objects.length; i++) {
          const {mesh, controls} = objects[i];
          mesh.visible = controls.some(input => input.checked);
          if (!mesh.visible) continue;
          interpolatePose(mesh, previous[i], poses[i], alpha);
        }
      }
      const target = Number(targetSelect.value);
      const selected = manifest.targets[target];
      if (selected) worldOffset.fromArray(selected.offset);
      for (let i = 0; i < frustums.length; i++) {
        const {input, body, image} = frustums[i];
        const texture = previewTexture(i);
        body.visible = selected !== undefined && (input.checked || texture !== null);
        image.visible = texture !== null;
        if (texture && texture !== image.material.map) {
          image.material.map = texture;
          image.material.needsUpdate = true;
        }
        if (!body.visible) continue;
        const world = selected.world;
        interpolatePose(body, from.cameraBodies[i][world], to.cameraBodies[i][world], alpha);
        body.position.add(worldOffset);
      }
      for (const {spec, root, geometry, input} of markers) {
        root.visible = input.checked;
        if (!root.visible) continue;
        if (spec.body !== null) {
          interpolatePose(root, from.markerBodies[spec.body], to.markerBodies[spec.body], alpha);
          root.position.add(worldOffset.fromArray(spec.offset));
        }
        const previous = Object.hasOwn(from.markerPoses, spec.name) ? from.markerPoses[spec.name] : spec.pose;
        const next = Object.hasOwn(to.markerPoses, spec.name) ? to.markerPoses[spec.name] : spec.pose;
        interpolatePose(geometry, previous, next, alpha);
      }
      if (follow.checked && to.targets[target]) {
        followTarget.fromArray(from.targets[target]);
        position.fromArray(to.targets[target]);
        followTarget.lerp(position, followTarget.distanceToSquared(position) > 1 ? 1 : alpha);
        delta.subVectors(followTarget, orbit.target).multiplyScalar(1 - Math.exp(-12 * dt));
        orbit.target.add(delta);
        camera.position.add(delta);
      }
    }
    moveFreeCamera(dt);
    orbit.update(dt);
    renderer.render(scene, camera);
    renderCount++;
    cpuMs += performance.now() - now;
    drawCount += renderer.info.render.drawCalls;
    const elapsed = now - statsAt;
    if (elapsed >= 1000) {
      stats.fps.textContent = (1000 * renderCount / elapsed).toFixed(1);
      stats.cpu.textContent = `${(cpuMs / renderCount).toFixed(1)} ms`;
      stats.draws.textContent = (drawCount / renderCount).toFixed(0);
      if (previousSimulationTime !== undefined) {
        stats["sim-speed"].textContent = `${(1000 * (simulationTime - previousSimulationTime) / elapsed).toFixed(2)}×`;
      }
      previousSimulationTime = simulationTime;
      renderCount = cpuMs = drawCount = 0;
      statsAt = now;
    }
  });
}

main().catch(error => {
  status.textContent = error.message;
  console.error(error);
});
