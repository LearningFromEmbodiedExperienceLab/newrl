"""Generate Newton rigid contacts between robot primitives and a static terrain mesh.

TerrainCollision applies the ground pose and scale once and builds a world-space
Warp mesh whose bounding volume hierarchy (BVH) is shared by all worlds. Each
primitive paired with the ground owns one query row. Rows are ordered by shape
type and within-world shape index so adjacent GPU lanes process corresponding
links across replicas. Cached face IDs are stored by candidate slot, then row,
so neighboring lanes can read consecutive entries.

Each collision update clears the contact buffer and runs three kernels:

* ``_prepare`` bounds each primitive at its current orientation and expands its
  axis-aligned box by both shapes' gaps and margins. On a cache miss it queries
  the BVH with an extra ``CACHE_SKIN`` on each side and saves the candidate faces.
  Small motions can reuse this list as long as the current query box fits inside
  the saved box. Checking containment each update also handles rotations,
  teleports, and margin changes. If the padded query exceeds the face capacity,
  the row records ``USE_BVH`` and detection traverses the current query box in full.
* ``_detect`` recomputes contacts at the current pose. Spheres use the closest
  point on each front-facing triangle and group contacts by face orientation.
  Boxes, capsules, and cylinders use separating-plane tests before Newton's
  convex contact routine. Their manifold retains the deepest contact and the
  extreme contact along each of four tetrahedral directions to preserve support
  across a resting face.
* ``_emit`` removes duplicate manifold entries and writes up to five contacts per
  primitive. Each record carries a world-space ground witness and normal, the
  primitive witness in its body's frame, and the offsets and margins Newton
  uses to reconstruct the surface points during solver substeps.
"""

from typing import Any

import newton
import numpy as np
import warp as wp
from newton import GeoType
from newton._src.geometry.collision_core import create_compute_gjk_mpr_contacts
from newton._src.geometry.contact_data import (
    ContactData,
    contact_passes_gap_check,
    prepare_speculative_contact,
)
from newton._src.geometry.flags import MeshProperties
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    closest_point_on_triangle,
    create_triangle_prism_penetration_refiner,
    support_map,
)
from newton._src.sim.collide import _estimate_rigid_contact_max

# This limits per-query memory only. Overfull regions use the complete mesh iterator.
MAX_CACHED_FACES = 256
INVALID_CACHE = -2
USE_BVH = -1
CACHE_SKIN = 0.04
MANIFOLD_SIZE = 5
_SUPPORTED_PRIMITIVES = (GeoType.SPHERE, GeoType.CAPSULE, GeoType.BOX, GeoType.CYLINDER)


@wp.struct
class _Manifold:
    row: int
    center: wp.vec3
    sphere: bool
    scores: wp.array2d[float]
    candidates: wp.array2d[ContactData]


@wp.func
def _retain(contact: ContactData, manifold: _Manifold, output_index: int):
    if not contact_passes_gap_check(contact):
        return
    # Deepest contact, plus extrema along four tetrahedral directions. Keeping
    # actual witnesses preserves torque arms when a face rests on the terrain.
    delta = contact.contact_point_center - manifold.center
    scores = wp.vec4(
        delta[0] + delta[1] + delta[2],
        delta[0] - delta[1] - delta[2],
        -delta[0] + delta[1] - delta[2],
        -delta[0] - delta[1] + delta[2],
    )
    for i in range(MANIFOLD_SIZE):
        score = -contact.contact_distance
        if i > 0:
            score = scores[i - 1]
        if score > manifold.scores[manifold.row, i]:
            manifold.scores[manifold.row, i] = score
            manifold.candidates[manifold.row, i] = contact


_convex_contacts = create_compute_gjk_mpr_contacts(
    _retain,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support_map),
)


@wp.func
def _extent(kind: int, scale: wp.vec3, axis: wp.vec3):
    """Upper bound on support distance along a primitive-local unit axis."""
    if kind == int(GeoType.BOX):
        return wp.dot(scale, wp.abs(axis))
    axial = wp.abs(axis[2])
    if kind == int(GeoType.CAPSULE):
        return scale[0] + scale[1] * axial
    radius = scale[0]
    if scale[2] > 0.0:
        # Enclose a barrel in the cylinder through its widest cross-section.
        height_sq = scale[1] * scale[1]
        radius += height_sq / (scale[2] + wp.sqrt(wp.max(0.0, scale[2] * scale[2] - height_sq)))
    return radius * wp.length(wp.vec2(axis[0], axis[1])) + scale[1] * axial


@wp.func
def _test_triangle(
    mesh: wp.uint64,
    face: int,
    primitive: GenericShapeData,
    rotation: wp.quat,
    ground: int,
    shape: int,
    gap: float,
    margin0: float,
    margin1: float,
    manifold: _Manifold,
):
    a = wp.mesh_eval_position(mesh, face, 1.0, 0.0)
    b = wp.mesh_eval_position(mesh, face, 0.0, 1.0)
    c = wp.mesh_eval_position(mesh, face, 0.0, 0.0)
    cross = wp.cross(b - a, c - a)
    length = wp.length(cross)
    if length == 0.0 or wp.dot(cross, manifold.center - a) < 0.0:
        return
    if manifold.sphere:
        ground_pt = closest_point_on_triangle(manifold.center, a, b, c)
        diff = manifold.center - ground_pt
        dist = wp.length(diff)
        radius = primitive.scale[0]
        normal = cross / length
        if dist > 1.0e-9:
            normal = diff / dist
        contact = ContactData()
        contact.shape_a = ground
        contact.shape_b = shape
        contact.contact_distance = dist - radius
        contact.contact_normal_a_to_b = normal
        contact.contact_point_center = ground_pt + 0.5 * normal * contact.contact_distance
        contact.margin_a = margin0
        contact.margin_b = margin1
        contact.radius_eff_b = radius
        contact.gap_sum = gap
        if not contact_passes_gap_check(contact):
            return
        # Keep the closest feature for each face orientation. This suppresses
        # internal edges on a tiled plane while retaining both sides of a valley.
        contact.sort_sub_key = face
        slot = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
        for i in range(MANIFOLD_SIZE):
            score = manifold.scores[manifold.row, i]
            if score == -1.0e30:
                slot = i
                break
            previous = manifold.candidates[manifold.row, i]
            if wp.dot(cross / length, wp.mesh_eval_face_normal(mesh, previous.sort_sub_key)) > 0.9999:
                slot = i
                break
            if score < manifold.scores[manifold.row, slot]:
                slot = i
        if -contact.contact_distance > manifold.scores[manifold.row, slot]:
            manifold.scores[manifold.row, slot] = -contact.contact_distance
            manifold.candidates[manifold.row, slot] = contact
    else:
        # Cheap separating-plane test before entering the iterative convex solver.
        n = cross / length
        shell = gap + margin0 + margin1
        kind = primitive.shape_type
        scale = primitive.scale
        axis = wp.quat_rotate_inv(rotation, n)
        if wp.dot(n, manifold.center - a) > _extent(kind, scale, axis) + shell + 1.0e-6:
            return
        # A second separating plane rejects triangles near the edge of the AABB.
        closest = closest_point_on_triangle(manifold.center, a, b, c)
        delta = manifold.center - closest
        distance = wp.length(delta)
        if distance > 1.0e-8:
            direction = delta / distance
            axis = wp.quat_rotate_inv(rotation, direction)
            # Use the extreme triangle projection even if the nearest point was rounded.
            separation = wp.min(
                wp.dot(direction, manifold.center - a),
                wp.min(wp.dot(direction, manifold.center - b), wp.dot(direction, manifold.center - c)),
            )
            if separation > _extent(kind, scale, axis) + shell + 1.0e-5:
                return
        triangle = GenericShapeData()
        triangle.shape_type = int(GeoTypeEx.TRIANGLE)
        triangle.scale = b - a
        triangle.auxiliary = c - a
        _convex_contacts(
            triangle,
            primitive,
            wp.quat_identity(),
            rotation,
            a,
            manifold.center,
            gap,
            ground,
            shape,
            margin0,
            margin1,
            SupportMapDataProvider(),
            manifold,
            face << 1,
        )


@wp.kernel(enable_backward=False)
def _prepare(
    body_q: wp.array[wp.transform],
    queries: wp.array[int],
    bodies: wp.array[int],
    local: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
    kinds: wp.array[int],
    margins: wp.array[float],
    gaps: wp.array[float],
    ground: int,
    mesh: wp.uint64,
    poses: wp.array[wp.transform],
    bounds: wp.array[wp.vec3],
    cached_lo: wp.array[wp.vec3],
    cached_hi: wp.array[wp.vec3],
    counts: wp.array[int],
    faces: wp.array2d[Any],
):
    row = wp.tid()
    shape = queries[row]
    pose = local[shape]
    if bodies[shape] >= 0:
        pose = wp.transform_multiply(body_q[bodies[shape]], pose)
    center = wp.transform_get_translation(pose)
    rot = wp.transform_get_rotation(pose)
    kind = kinds[shape]
    scale = scales[shape]
    gap = gaps[shape] + gaps[ground]
    extent = wp.vec3(scale[0])
    if kind != int(GeoType.SPHERE):
        rotation = wp.quat_to_matrix(rot)
        for j in range(3):
            extent[j] = _extent(kind, scale, rotation[j])
    half = extent + wp.vec3(gap + margins[shape] + margins[ground] + 1.0e-6)
    poses[row] = pose
    bounds[row] = half
    lo = center - half
    hi = center + half
    # Containment checks both translation and rotation, including teleports and shell changes.
    if counts[row] != INVALID_CACHE and wp.min(lo - cached_lo[row]) >= 0.0 and wp.min(cached_hi[row] - hi) >= 0.0:
        return
    lo -= wp.vec3(CACHE_SKIN)
    hi += wp.vec3(CACHE_SKIN)
    cached_lo[row] = lo
    cached_hi[row] = hi
    query = wp.mesh_query_aabb(mesh, lo, hi)
    face = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    count = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    while wp.mesh_query_aabb_next(query, face):
        if count == faces.shape[0]:
            counts[row] = USE_BVH
            return
        faces[count, row] = faces.dtype(face)
        count += 1
    counts[row] = count


@wp.kernel(enable_backward=False)
def _detect(
    queries: wp.array[int],
    poses: wp.array[wp.transform],
    bounds: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    kinds: wp.array[int],
    margins: wp.array[float],
    gaps: wp.array[float],
    ground: int,
    mesh: wp.uint64,
    counts: wp.array[int],
    faces: wp.array2d[Any],
    scores: wp.array2d[float],
    candidates: wp.array2d[ContactData],
):
    row = wp.tid()
    shape = queries[row]
    pose = poses[row]
    primitive = GenericShapeData()
    primitive.shape_type = kinds[shape]
    primitive.scale = scales[shape]
    primitive.shape_index = shape
    center = wp.transform_get_translation(pose)
    rot = wp.transform_get_rotation(pose)
    gap = gaps[shape] + gaps[ground]
    manifold = _Manifold()
    manifold.row = row
    manifold.center = center
    manifold.sphere = primitive.shape_type == int(GeoType.SPHERE)
    manifold.scores = scores
    manifold.candidates = candidates
    for i in range(MANIFOLD_SIZE):
        scores[row, i] = -1.0e30

    count = counts[row]
    if count >= 0:
        for i in range(count):
            _test_triangle(
                mesh,
                int(faces[i, row]),
                primitive,
                rot,
                ground,
                shape,
                gap,
                margins[ground],
                margins[shape],
                manifold,
            )
        return
    # The padded region exceeded the cache. Test every hit on the current tight bounds.
    half = bounds[row]
    query = wp.mesh_query_aabb(mesh, center - half, center + half)
    face = int(0)  # noqa: UP018, RUF046 - mutable Warp loop variable
    while wp.mesh_query_aabb_next(query, face):
        _test_triangle(mesh, face, primitive, rot, ground, shape, gap, margins[ground], margins[shape], manifold)


@wp.func
def _unique_contact(
    scores: wp.array2d[float],
    candidates: wp.array2d[ContactData],
    row: int,
    i: int,
    center: wp.vec3,
    normal: wp.vec3,
) -> bool:
    for j in range(i):
        if scores[row, j] == -1.0e30:
            continue
        other = candidates[row, j]
        if (
            wp.length(other.contact_point_center - center) < 1.0e-5
            and wp.dot(other.contact_normal_a_to_b, normal) > 0.9999
        ):
            return False
    return True


@wp.kernel(enable_backward=False)
def _flag_contacts(
    scores: wp.array2d[float],
    candidates: wp.array2d[ContactData],
    flags: wp.array[int],
):
    row, i = wp.tid()
    index = row * MANIFOLD_SIZE + i
    if scores[row, i] == -1.0e30:
        flags[index] = 0
        return
    contact = candidates[row, i]
    n, _a, _b, _separation = prepare_speculative_contact(contact)
    flags[index] = int(
        _unique_contact(scores, candidates, row, i, contact.contact_point_center, n)
    )


@wp.kernel(enable_backward=False)
def _advance_contact_count(offsets: wp.array[int], count: wp.array[int], capacity: int):
    count[0] = wp.min(count[0] + offsets[offsets.shape[0] - 1], capacity)


@wp.kernel(enable_backward=False)
def _emit(
    scores: wp.array2d[float],
    candidates: wp.array2d[ContactData],
    body_q: wp.array[wp.transform],
    bodies: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    offset0: wp.array[wp.vec3],
    offset1: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    count: wp.array[int],
    offsets: wp.array[int],
    flags: wp.array[int],
    tids: wp.array[int],
):
    row, i = wp.tid()
    index = row * MANIFOLD_SIZE + i
    if scores[row, i] == -1.0e30:
        return
    contact = candidates[row, i]
    n, a, b, _separation = prepare_speculative_contact(contact)
    if offsets.shape[0] > 0:
        if flags[index] == 0:
            return
        slot = count[0] + offsets[index] - 1
    else:
        if not _unique_contact(scores, candidates, row, i, contact.contact_point_center, n):
            return
        slot = wp.atomic_add(count, 0, 1)
    if slot >= shape0.shape[0]:
        if offsets.shape[0] == 0:
            wp.atomic_min(count, 0, shape0.shape[0])
        return

    inv = wp.transform_identity()
    if bodies[contact.shape_b] >= 0:
        inv = wp.transform_inverse(body_q[bodies[contact.shape_b]])
    shape0[slot] = contact.shape_a
    shape1[slot] = contact.shape_b
    point0[slot] = a
    point1[slot] = wp.transform_point(inv, b)
    normal[slot] = n
    margin0[slot] = contact.margin_a
    margin1[slot] = contact.margin_b + contact.radius_eff_b
    offset0[slot] = n * contact.margin_a
    offset1[slot] = wp.transform_vector(inv, -n * margin1[slot])
    tids[slot] = -1


class TerrainCollision:
    """Static terrain BVH shared by all worlds, with a conservative face cache per primitive."""

    def __init__(
        self,
        model,
        *,
        ground_label: str = "ground_mesh",
        ground_shape: int | None = None,
        queries: np.ndarray | None = None,
        deterministic: bool = False,
    ):
        self.model = model
        self.device = model.device
        if ground_shape is None:
            matches = [
                i
                for i, label in enumerate(model.shape_label)
                if label == ground_label or label.endswith("/" + ground_label)
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one static terrain mesh named {ground_label!r}")
            ground_shape = matches[0]
        self.ground = ground = ground_shape
        kinds = model.shape_type.numpy()
        if kinds[ground] != GeoType.MESH or model.shape_body.numpy()[ground] >= 0:
            raise ValueError("terrain must be a static triangle mesh")
        mesh = model.shape_source[ground]
        scale = model.shape_scale.numpy()[ground]
        pose = wp.transform(*model.shape_transform.numpy()[ground].tolist())
        vertices = np.asarray(mesh.vertices).reshape(-1, 3) * scale
        rotation = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
        vertices = vertices @ rotation.T + np.asarray(wp.transform_get_translation(pose))
        indices = np.asarray(mesh.indices).reshape(-1, 3)
        if np.prod(scale) < 0:
            indices = indices[:, [0, 2, 1]]
        self.mesh = wp.Mesh(
            wp.array(vertices, dtype=wp.vec3, device=self.device),
            wp.array(indices.flatten(), dtype=int, device=self.device),
        )
        standalone = queries is None
        if standalone:
            pairs = model.shape_contact_pairs.numpy().reshape(-1, 2)
            mesh_pairs = pairs[np.any(pairs == ground, axis=1)]
            queries = mesh_pairs[mesh_pairs != ground]
        if not np.isin(kinds[queries], _SUPPORTED_PRIMITIVES).all():
            raise ValueError("terrain contacts support sphere, capsule, box, and cylinder primitives")
        # Schedule corresponding links together so adjacent GPU lanes take similar paths.
        worlds = model.shape_world.numpy()
        _, starts, world_indices = np.unique(worlds, return_index=True, return_inverse=True)
        ordinal = queries - starts[world_indices[queries]]
        queries = queries[np.lexsort((queries, ordinal, kinds[queries]))]
        self.query_count = len(queries)
        self.queries = wp.array(queries, dtype=int, device=self.device)
        with wp.ScopedDevice(self.device):
            self.poses = wp.empty(self.query_count, dtype=wp.transform)
            self.bounds = wp.empty(self.query_count, dtype=wp.vec3)
            self.cached_lo = wp.empty(self.query_count, dtype=wp.vec3)
            self.cached_hi = wp.empty(self.query_count, dtype=wp.vec3)
            self.counts = wp.full(self.query_count, INVALID_CACHE, dtype=int)
            # Neighboring lanes load neighboring face IDs even for different worlds.
            capacity = min(len(indices), MAX_CACHED_FACES)
            index_dtype = wp.uint16 if len(indices) <= 65536 else wp.int32
            self.faces = wp.empty((capacity, self.query_count), dtype=index_dtype)
            shape = (self.query_count, MANIFOLD_SIZE)
            self.scores = wp.empty(shape, dtype=float)
            self.candidates = wp.empty(shape, dtype=ContactData)
            self.contact_capacity = self.query_count * MANIFOLD_SIZE
            self.deterministic = deterministic
            scratch_size = self.contact_capacity if deterministic else 0
            self.contact_flags = wp.empty(scratch_size, dtype=int)
            self.contact_offsets = wp.empty_like(self.contact_flags)
            self.contacts = (
                newton.Contacts(self.contact_capacity, 0, device=self.device)
                if standalone
                else None
            )

    def invalidate(self):
        self.counts.fill_(INVALID_CACHE)

    def collide(self, state, contacts=None):
        m = self.model
        c = contacts
        if c is None:
            c = self.contacts
            c.clear()
        if self.query_count:
            wp.launch(
                _prepare,
                self.query_count,
                inputs=[
                    state.body_q,
                    self.queries,
                    m.shape_body,
                    m.shape_transform,
                    m.shape_scale,
                    m.shape_type,
                    m.shape_margin,
                    m.shape_gap,
                    self.ground,
                    wp.uint64(self.mesh.id),
                    self.poses,
                    self.bounds,
                    self.cached_lo,
                    self.cached_hi,
                    self.counts,
                    self.faces,
                ],
                device=self.device,
            )
            wp.launch(
                _detect,
                self.query_count,
                inputs=[
                    self.queries,
                    self.poses,
                    self.bounds,
                    m.shape_scale,
                    m.shape_type,
                    m.shape_margin,
                    m.shape_gap,
                    self.ground,
                    wp.uint64(self.mesh.id),
                    self.counts,
                    self.faces,
                    self.scores,
                    self.candidates,
                ],
                device=self.device,
            )
            if self.deterministic:
                wp.launch(
                    _flag_contacts, (self.query_count, MANIFOLD_SIZE),
                    inputs=[self.scores, self.candidates, self.contact_flags],
                    device=self.device,
                )
                wp.utils.array_scan(self.contact_flags, self.contact_offsets)
            wp.launch(
                _emit,
                (self.query_count, MANIFOLD_SIZE),
                inputs=[
                    self.scores,
                    self.candidates,
                    state.body_q,
                    m.shape_body,
                    c.rigid_contact_shape0,
                    c.rigid_contact_shape1,
                    c.rigid_contact_point0,
                    c.rigid_contact_point1,
                    c.rigid_contact_normal,
                    c.rigid_contact_offset0,
                    c.rigid_contact_offset1,
                    c.rigid_contact_margin0,
                    c.rigid_contact_margin1,
                    c.rigid_contact_count,
                    self.contact_offsets,
                    self.contact_flags,
                    c.rigid_contact_tids,
                ],
                device=self.device,
            )
            if self.deterministic:
                wp.launch(
                    _advance_contact_count, 1,
                    inputs=[self.contact_offsets, c.rigid_contact_count, c.rigid_contact_max],
                    device=self.device,
                )


class HybridTerrainCollision:
    """Route supported primitive–mesh pairs to the terrain kernel and the rest to Newton."""

    def __init__(self, model, terrain_shapes, *, deterministic: bool = False):
        self.model = model
        self.mesh_colliders = []
        kinds = model.shape_type.numpy()
        bodies = model.shape_body.numpy()
        mesh_shapes = [
            index for index in terrain_shapes
            if kinds[index] == GeoType.MESH and bodies[index] < 0
        ]
        self.newton = None
        mesh_properties = model._shape_mesh_properties
        has_surface_velocity = mesh_properties is not None and np.any(
            mesh_properties.numpy() & int(MeshProperties.SURFACE_VELOCITY)
        )
        if (
            mesh_shapes
            and not model._has_body_variants
            and not has_surface_velocity
            and not np.any(model.shape_flags.numpy() & int(newton.ShapeFlags.HYDROELASTIC))
        ):
            remaining = model.shape_contact_pairs.numpy().reshape(-1, 2)
            for ground in mesh_shapes:
                ground_first = remaining[:, 0] == ground
                other = np.where(ground_first, remaining[:, 1], remaining[:, 0])
                owned = (ground_first | (remaining[:, 1] == ground)) & np.isin(
                    kinds[other], _SUPPORTED_PRIMITIVES
                )
                if owned.any():
                    self.mesh_colliders.append(
                        TerrainCollision(
                            model,
                            ground_shape=ground,
                            queries=other[owned],
                            deterministic=deterministic,
                        )
                    )
                    remaining = remaining[~owned]

        if not self.mesh_colliders:
            self.newton = newton.CollisionPipeline(model, deterministic=deterministic)
            return

        self.custom_capacity = sum(item.contact_capacity for item in self.mesh_colliders)
        if len(remaining) or model.particle_count:
            newton_capacity = model.rigid_contact_max
            if newton_capacity is None:
                newton_capacity = _estimate_rigid_contact_max(model)
            self.newton = newton.CollisionPipeline(
                model,
                broad_phase="explicit",
                shape_pairs_filtered=wp.array(
                    remaining, dtype=wp.vec2i, device=model.device
                ),
                rigid_contact_max=newton_capacity + self.custom_capacity,
                deterministic=deterministic,
            )

    def contacts(self):
        if self.newton is not None:
            return self.newton.contacts()
        return newton.Contacts(self.custom_capacity, 0, device=self.model.device)

    def collide(self, state, contacts):
        if self.newton is not None:
            self.newton.collide(state, contacts)
        else:
            contacts.clear()
        for collider in self.mesh_colliders:
            collider.collide(state, contacts)
