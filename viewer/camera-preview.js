import * as THREE from "three";

export function setupCameraPreview(cameras, selectedWorld, isHistorical) {
  if (!cameras.length) return () => null;
  const panel = document.querySelector("#camera-preview");
  const selector = document.querySelector("#preview-camera");
  const channel = document.querySelector("#preview-channel");
  const inScene = document.querySelector("#preview-frustum");
  const canvas = document.querySelector("#preview-image");
  const status = document.querySelector("#preview-status");
  const context = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;
  let currentRoute;
  let tag;
  let ready = false;

  function selection() {
    const world = selectedWorld();
    if (!panel.open || world === undefined || isHistorical()) return;
    if (!channel.value) return;
    return `/camera/${selector.value}/${world}/${channel.value}`;
  }

  function updateFields() {
    const spec = cameras[Number(selector.value)];
    channel.replaceChildren(...spec.previewFields.map(field =>
      new Option(field === "rgb" ? "RGB" : "Depth", field)));
    channel.disabled = !spec.previewFields.length;
    inScene.disabled = !spec.previewFields.length;
  }

  async function poll() {
    const route = selection();
    if (route !== currentRoute) {
      currentRoute = route;
      tag = undefined;
      ready = false;
      canvas.hidden = true;
    }
    try {
      if (!panel.open) return;
      if (isHistorical()) {
        status.textContent = "Camera images are available in live playback.";
        return;
      }
      if (!route) {
        status.textContent = channel.value ? "Select a world." : "Start with viewer.camera_preview=true to activate this sensor.";
        return;
      }
      if (!ready) status.textContent = "Waiting for sensor frame…";
      const response = await fetch(route, {
        cache: "no-store",
        headers: tag ? {"If-None-Match": tag} : {},
      });
      if (response.status === 204 || response.status === 304) return;
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (route !== selection()) return;
      const {width, height} = cameras[Number(selector.value)];
      const stride = channel.value === "rgb" ? 3 : 1;
      if (bytes.length !== width * height * stride) throw new Error("Unexpected camera image size.");
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        // A different image size requires fresh Three.js texture storage.
        texture.dispose();
      }
      const image = context.createImageData(width, height);
      for (let i = 0; i < width * height; i++) {
        const src = i * stride;
        const dst = i * 4;
        image.data[dst] = bytes[src];
        image.data[dst + 1] = bytes[stride === 3 ? src + 1 : src];
        image.data[dst + 2] = bytes[stride === 3 ? src + 2 : src];
        image.data[dst + 3] = 255;
      }
      context.putImageData(image, 0, 0);
      texture.needsUpdate = true;
      canvas.hidden = false;
      status.textContent = channel.value === "depth" ? "Near surfaces are brighter. Invalid pixels are black." : "";
      tag = response.headers.get("ETag");
      ready = true;
    } catch (error) {
      if (route === selection()) status.textContent = `Camera preview unavailable. ${error.message}`;
    } finally {
      setTimeout(poll, panel.open ? 67 : 150);
    }
  }

  panel.hidden = false;
  cameras.forEach(({name}, index) => selector.add(new Option(name, index)));
  const active = cameras.findIndex(camera => camera.previewFields.length);
  if (active >= 0) selector.selectedIndex = active;
  updateFields();
  selector.addEventListener("change", updateFields);
  poll();
  return index => ready && inScene.checked && Number(selector.value) === index &&
    currentRoute === selection() ? texture : null;
}
