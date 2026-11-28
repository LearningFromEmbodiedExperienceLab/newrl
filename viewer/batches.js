import * as THREE from "three";
import { MeshStandardNodeMaterial } from "three/webgpu";
import { color as nodeColor, Fn, If, materialColor, mix, normalWorld, positionWorld, smoothstep, vec2, vec4 } from "three/tsl";

function groundGridColor(enabled) {
  return Fn(() => {
    const albedo = vec4(materialColor);
    const base = albedo.rgb.toVar();
    If(enabled.greaterThan(0.5), () => {
      const xy = positionWorld.xy;
      const nearestLine = xy.add(0.5).fract().sub(0.5).abs();
      const footprint = xy.fwidth().max(0.0001);
      const lines = smoothstep(footprint.mul(0.18), footprint.mul(1.15), nearestLine).oneMinus();
      const fade = smoothstep(vec2(0.12), vec2(0.45), footprint).oneMinus();
      const coverage = lines.mul(fade);
      const slope = smoothstep(0.15, 0.45, normalWorld.z.abs());
      base.assign(mix(base, nodeColor(0xadc8d7), coverage.x.max(coverage.y).mul(slope).mul(0.42)));
    });
    return vec4(base, albedo.a);
  })();
}

function instances(scene, geometry, material, capacity) {
  if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
  const mesh = new THREE.InstancedMesh(geometry, material, capacity);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false; // Culling happens before compacting the instance buffer.
  mesh.count = 0;
  scene.add(mesh);
  return mesh;
}

function uploadVisible(attribute, count, width) {
  if (!count) return;
  attribute.clearUpdateRanges();
  attribute.addUpdateRange(0, count * width);
  attribute.needsUpdate = true;
}

export function buildMeshBatches(scene, specifications, geometries, layers, gridEnabled) {
  const opaqueMaterials = new Map();
  const groundMaterials = new Map();
  const gridColor = groundGridColor(gridEnabled);
  return Object.entries(specifications).flatMap(([name, spec]) => {
    const geometry = geometries.get(spec.mesh);
    if (!geometry) return [];
    const {map = null, hasAlpha = false} = geometry.userData.baseColor ?? {};
    const opaque = new Map();
    const transparent = [];
    for (let i = 0; i < spec.count; i++) {
      const opacity = spec.opacities?.[i] ?? 1;
      const rgb = spec.colors?.[i];
      const color = rgb
        ? new THREE.Color().setRGB(...rgb, THREE.SRGBColorSpace)
        : new THREE.Color(map ? 0xffffff : 0xaeb7c3);
      const scale = spec.scales ? new THREE.Vector3(...spec.scales[i]) : new THREE.Vector3(1, 1, 1);
      const controls = spec.layers[i].map(layer => layers.get(layer));
      const ground = spec.layers[i].includes("Ground");
      if (opacity < 1 || hasAlpha) {
        if (opacity <= 0) continue;
        const Material = ground ? MeshStandardNodeMaterial : THREE.MeshStandardMaterial;
        const material = new Material({
          color, map, opacity, transparent: true, side: THREE.DoubleSide,
        });
        if (ground) material.colorNode = gridColor;
        const mesh = new THREE.Mesh(geometry, material);
        mesh.scale.copy(scale);
        mesh.visible = false;
        scene.add(mesh);
        transparent.push({mesh, controls, index: i});
      } else {
        const key = JSON.stringify([spec.layers[i], ground ? color.getHex() : null]);
        if (!opaque.has(key)) opaque.set(key, {controls, rows: [], layers: spec.layers[i], ground});
        opaque.get(key).rows.push({index: i, color, scale});
      }
    }
    const groups = Array.from(opaque.values(), ({controls, rows, layers, ground}) => {
      const uniformColor = rows.every(({color}) => color.equals(rows[0].color));
      const color = uniformColor ? rows[0].color : new THREE.Color(0xffffff);
      let material;
      if (ground) {
        if (!groundMaterials.has(map)) groundMaterials.set(map, new Map());
        const materials = groundMaterials.get(map);
        const key = color.getHex();
        if (!materials.has(key)) {
          const material = new MeshStandardNodeMaterial({color, map, side: THREE.DoubleSide});
          material.colorNode = gridColor;
          materials.set(key, material);
        }
        material = materials.get(key);
      } else {
        if (!opaqueMaterials.has(map)) opaqueMaterials.set(map, new Map());
        const materials = opaqueMaterials.get(map);
        const key = color.getHex();
        if (!materials.has(key)) {
          materials.set(key, new THREE.MeshStandardMaterial({color, map, side: THREE.DoubleSide}));
        }
        material = materials.get(key);
      }
      const mesh = instances(scene, geometry, material, rows.length);
      mesh.castShadow = layers.some(layer => layer.endsWith(" visuals"));
      mesh.receiveShadow = ground;
      let colorSlots = null;
      if (!uniformColor) {
        // Declare the attribute before the renderer builds the material pipeline.
        mesh.setColorAt(0, rows[0].color);
        mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        colorSlots = new Int32Array(rows.length).fill(-1);
      }
      return {mesh, controls, rows, colorSlots};
    });
    return [{name, groups, transparent}];
  });
}

export function buildGhost(scene, spec, geometries, offsets) {
  const material = new THREE.MeshBasicMaterial({
    color: spec.color, transparent: true, opacity: spec.opacity, depthWrite: false,
  });
  const shapes = spec.shapes.map(shape => {
    const mesh = instances(scene, geometries.get(shape.mesh), material, offsets.length);
    const local = new THREE.Matrix4().compose(
      new THREE.Vector3().fromArray(shape.pose),
      new THREE.Quaternion().fromArray(shape.pose, 3),
      new THREE.Vector3().fromArray(shape.scale),
    );
    return {mesh, body: shape.body, local};
  });
  const bodies = Array.from({length: offsets.length * spec.bodyCount}, () => new THREE.Matrix4());
  return {spec, offsets, shapes, bodies};
}

export function instanceUpdater(interpolatePose) {
  const transform = new THREE.Object3D();
  const frustum = new THREE.Frustum();
  const projection = new THREE.Matrix4();
  const sphere = new THREE.Sphere();
  const matrix = new THREE.Matrix4();
  const enabled = controls => controls.some(input => input.checked);
  const inView = (mesh, pose) => frustum.intersectsSphere(sphere.copy(mesh.geometry.boundingSphere).applyMatrix4(pose));

  return (camera, batches, ghosts, from, to, alpha) => {
    camera.updateMatrixWorld();
    projection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    frustum.setFromProjectionMatrix(projection, camera.coordinateSystem, camera.reversedDepth);
    for (const {name, groups, transparent} of batches) {
      const poses = to.poses[name];
      if (!poses) continue;
      const previous = from.poses[name] ?? poses;
      for (const {mesh, controls, rows, colorSlots} of groups) {
        mesh.visible = enabled(controls);
        if (!mesh.visible) continue;
        let count = 0;
        let colorsChanged = false;
        for (const {index, scale, color} of rows) {
          interpolatePose(transform, previous, poses, alpha, index * 7);
          transform.scale.copy(scale);
          transform.updateMatrix();
          if (!inView(mesh, transform.matrix)) continue;
          mesh.setMatrixAt(count, transform.matrix);
          if (colorSlots && colorSlots[count] !== index) {
            mesh.setColorAt(count, color);
            colorSlots[count] = index;
            colorsChanged = true;
          }
          count++;
        }
        mesh.count = count;
        mesh.visible = count > 0;
        uploadVisible(mesh.instanceMatrix, count, 16);
        if (colorsChanged) uploadVisible(mesh.instanceColor, count, 3);
      }
      for (const {mesh, controls, index} of transparent) {
        mesh.visible = enabled(controls);
        if (mesh.visible) interpolatePose(mesh, previous, poses, alpha, index * 7);
      }
    }
    transform.scale.set(1, 1, 1);
    for (const {spec, input, shapes, bodies, offsets} of ghosts) {
      const poses = to.ghostPoses[spec.name];
      const show = input.checked && !!poses;
      for (const {mesh} of shapes) mesh.visible = show;
      if (!show) continue;
      const previous = from.ghostPoses[spec.name] ?? poses;
      for (let i = 0; i < bodies.length; i++) {
        interpolatePose(transform, previous, poses, alpha, i * 7);
        transform.position.add(offsets[Math.floor(i / spec.bodyCount)]);
        transform.updateMatrix();
        bodies[i].copy(transform.matrix);
      }
      for (const {mesh, body, local} of shapes) {
        let count = 0;
        for (let world = 0; world < offsets.length; world++) {
          matrix.multiplyMatrices(bodies[world * spec.bodyCount + body], local);
          if (inView(mesh, matrix)) mesh.setMatrixAt(count++, matrix);
        }
        mesh.count = count;
        mesh.visible = count > 0;
        uploadVisible(mesh.instanceMatrix, count, 16);
      }
    }
  };
}
