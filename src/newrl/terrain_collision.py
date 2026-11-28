"""Generate Newton rigid contacts between robot primitives and a static terrain mesh.

TerrainCollision applies the ground pose and scale once and builds a world-space
Warp mesh whose bounding volume hierarchy (BVH) is shared by all worlds. Each
primitive paired with the ground owns one query row. Rows are ordered by shape
type and within-world shape index so adjacent GPU lanes process corresponding
links across replicas. Cached face IDs are stored by candidate slot, then row,
so neighboring lanes can read consecutive entries.

The hybrid router also handles disk-swept rounded wheels on infinite planes.
Their lowest witness is the disk rim in the projected downhill direction plus
the spherical tread offset. A wheel lying on its side uses three rim points.
Both paths write the same Newton contact records.

Each mesh collision update runs three kernels:

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
  convex contact routine. Rounded wheels use exact support witnesses on shallow
  triangle faces and the convex query for edge, vertex, and deep contacts.
  On meshes with substantial coplanar coverage, internal edges do not
  contribute their own wheel contact normals. Predominantly noncoplanar
  meshes use a kernel specialized for the generic convex path.
  The general manifold retains the deepest contact and the
  extreme contact along each of four tetrahedral directions to preserve support
  across a resting face.
* ``_emit`` removes duplicate manifold entries and writes up to five contacts per
  primitive. Each record carries a world-space ground witness and normal, the
  primitive witness in its body's frame, and the offsets and margins Newton
  uses to reconstruct the surface points during solver substeps.
"""

from collections.abc import Sequence

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
    smooth_features: int
    triangle_a: wp.vec3
    triangle_b: wp.vec3
    triangle_c: wp.vec3
    triangle_normal: wp.vec3
    triangle_area: float
    scores: wp.array2d[float]
    candidates: wp.array2d[ContactData]


@wp.func
def _edge_coordinates(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3, n: wp.vec3, area: float) -> wp.vec3:
    # Each coordinate vanishes on the corresponding directed edge AB, BC, CA.
    return wp.vec3(
        wp.dot(wp.cross(b - a, p - a), n),
        wp.dot(wp.cross(c - b, p - b), n),
        wp.dot(wp.cross(a - c, p - c), n),
    ) / area


@wp.func
def _retain(contact: ContactData, manifold: _Manifold, output_index: int) -> None:
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


@wp.func
def _retain_on_coplanar_mesh(contact: ContactData, manifold: _Manifold, output_index: int) -> None:
    if manifold.smooth_features != 0:
        _normal, ground_point, _wheel_point, _separation = prepare_speculative_contact(contact)
        edges = _edge_coordinates(
            ground_point, manifold.triangle_a, manifold.triangle_b, manifold.triangle_c,
            manifold.triangle_normal, manifold.triangle_area,
        )
        side_lengths = wp.vec3(
            wp.length(manifold.triangle_b - manifold.triangle_a),
            wp.length(manifold.triangle_c - manifold.triangle_b),
            wp.length(manifold.triangle_a - manifold.triangle_c),
        )
        # The iterative query may place an edge witness slightly off the edge.
        # Classify in meters so this tolerance does not shrink on dense meshes.
        near = int(0)
        for edge in range(3):
            if wp.abs(edges[edge]) * manifold.triangle_area <= 2.0e-4 * side_lengths[edge]:
                near |= 1 << edge
        for edge in range(3):
            if near == (1 << edge) and (manifold.smooth_features & near) != 0:
                return
            # Vertex i is between the previous edge and edge i. Only ignore
            # a vertex when its entire incident fan is smooth.
            if near == ((1 << edge) | (1 << ((edge + 2) % 3))):
                if (manifold.smooth_features & (1 << (edge + 3))) != 0:
                    return
    _retain(contact, manifold, output_index)


_convex_contacts = create_compute_gjk_mpr_contacts(
    _retain,
    use_precomputed_center=True,
    penetration_refiner=create_triangle_prism_penetration_refiner(support_map),
)


@wp.func
def _wheel_contact(
    wheel_point: wp.vec3, distance: float, normal: wp.vec3,
    ground: int, wheel: int, width: float, gap: float, margin0: float, margin1: float,
) -> ContactData:
    contact = ContactData()
    contact.shape_a = ground
    contact.shape_b = wheel
    contact.contact_normal_a_to_b = normal
    contact.contact_distance = distance
    contact.contact_point_center = wheel_point - 0.5 * distance * normal
    contact.radius_eff_b = width
    contact.margin_a = margin0
    contact.margin_b = margin1
    contact.gap_sum = gap
    return contact


@wp.func
def _extent(kind: int, scale: wp.vec3, axis: wp.vec3) -> float:
    """Upper bound on support distance along a primitive-local unit axis."""
    if kind == int(GeoType.BOX):
        return wp.dot(scale, wp.abs(axis))
    axial = wp.abs(axis[2])
    if kind == int(GeoType.CAPSULE):
        return scale[0] + scale[1] * axial
    radius = scale[0]
    if scale[2] == scale[1] and scale[2] > 0.0:
        # A rounded wheel is a disk swept by a sphere of radius half-width.
        return radius * wp.length(wp.vec2(axis[0], axis[1])) + scale[1]
    if scale[2] > 0.0:
        # Enclose a barrel in the cylinder through its widest cross-section.
        height_sq = scale[1] * scale[1]
        radius += height_sq / (scale[2] + wp.sqrt(wp.max(0.0, scale[2] * scale[2] - height_sq)))
    return radius * wp.length(wp.vec2(axis[0], axis[1])) + scale[1] * axial


def _create_triangle_test(coplanar: bool):
    convex_contacts = _convex_contacts
    if coplanar:
        convex_contacts = create_compute_gjk_mpr_contacts(
            _retain_on_coplanar_mesh,
            use_precomputed_center=True,
            penetration_refiner=create_triangle_prism_penetration_refiner(support_map),
        )

    @wp.func
    def _test_triangle(
        mesh: wp.uint64,
        face: int,
        smooth_features: wp.array[wp.uint8],
        primitive: GenericShapeData,
        rotation: wp.quat,
        ground: int,
        shape: int,
        gap: float,
        margin0: float,
        margin1: float,
        manifold: _Manifold,
    ) -> None:
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
            if wp.static(coplanar):
                wheel = kind == int(GeoType.CYLINDER) and scale[1] == scale[2] and scale[2] > 0.0
                manifold.smooth_features = 0
                if wheel:
                    rim = wp.vec3(-axis[0], -axis[1], 0.0)
                    radial_length = wp.length(rim)
                    if radial_length > 1.0e-5:
                        height = wp.dot(n, manifold.center - a) - scale[0] * radial_length - scale[1]
                        if height >= -scale[1]:
                            smooth = int(smooth_features[face])
                            point = manifold.center + wp.quat_rotate(rotation, rim * (scale[0] / radial_length)) - scale[1] * n
                            edges = _edge_coordinates(point, a, b, c, n, length)
                            inside = True
                            for edge in range(3):
                                if edges[edge] <= 1.0e-5:
                                    shared = (smooth & (1 << edge)) != 0
                                    inside = inside and shared and edges[edge] >= -1.0e-6
                            if inside:
                                contact = _wheel_contact(point, height, n, ground, shape, scale[1], gap, margin0, margin1)
                                contact.sort_sub_key = face << 1
                                _retain(contact, manifold, face << 1)
                                return
                            # With every edge and vertex internal to the planar patch,
                            # this triangle has no other feature to test.
                            if smooth == 0b111111:
                                return
                            # Keep generic contacts for deep overlap and side-face poses.
                            manifold.smooth_features = smooth
                            manifold.triangle_a = a
                            manifold.triangle_b = b
                            manifold.triangle_c = c
                            manifold.triangle_normal = n
                            manifold.triangle_area = length
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
            convex_contacts(
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
    return _test_triangle


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
) -> None:
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


def _create_mesh_detector(coplanar: bool):
    _test_triangle = _create_triangle_test(coplanar)

    @wp.kernel(enable_backward=False, module="unique")
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
        smooth_features: wp.array[wp.uint8],
        counts: wp.array[int],
        faces: wp.array2d[Any],
        scores: wp.array2d[float],
        candidates: wp.array2d[ContactData],
    ) -> None:
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
                    smooth_features,
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
            _test_triangle(mesh, face, smooth_features, primitive, rot, ground, shape, gap, margins[ground], margins[shape], manifold)
    return _detect


_detect_mesh = _create_mesh_detector(coplanar=False)
_detect_coplanar_wheels = _create_mesh_detector(coplanar=True)


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
) -> None:
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
def _advance_contact_count(offsets: wp.array[int], count: wp.array[int], capacity: int) -> None:
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
) -> None:
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


@wp.kernel(enable_backward=False)
def _detect_wheel_plane(
    queries: wp.array[int], ground: int, body_q: wp.array[wp.transform],
    bodies: wp.array[int], local: wp.array[wp.transform], scales: wp.array[wp.vec3],
    gaps: wp.array[float], margins: wp.array[float],
    scores: wp.array2d[float], candidates: wp.array2d[ContactData],
):
    row = wp.tid()
    wheel = queries[row]
    pose = wp.transform_multiply(body_q[bodies[wheel]], local[wheel])
    n = wp.quat_rotate(local[ground].q, wp.vec3(0.0, 0.0, 1.0))
    direction = wp.quat_rotate_inv(pose.q, -n)
    radial = wp.vec3(direction[0], direction[1], 0.0)
    width = scales[wheel][1]
    for i in range(MANIFOLD_SIZE):
        scores[row, i] = -1.0e30
    # A parallel disk face has a support area. Use three rim witnesses there.
    count = int(1)
    if wp.length(radial) < 1.0e-6:
        count = 3
    for i in range(count):
        rim = wp.normalize(radial)
        if count == 3:
            angle = float(i) * 2.094395102
            rim = wp.vec3(wp.cos(angle), wp.sin(angle), 0.0)
        p = wp.transform_point(pose, scales[wheel][0] * rim) - width * n
        d = wp.dot(n, p - local[ground].p)
        contact = _wheel_contact(
            p, d, n, ground, wheel, width, gaps[ground] + gaps[wheel],
            margins[ground], margins[wheel],
        )
        if contact_passes_gap_check(contact):
            scores[row, i] = -d
            candidates[row, i] = contact


def _coplanar_features(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return edge bits 0–2 and vertex bits 3–5 for internal coplanar features."""
    # Weld identical positions for meshes that duplicate vertices at UV seams.
    _, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    faces = vertex_ids.reshape(-1)[indices]
    starts = faces.reshape(-1)
    ends = np.roll(faces, -1, axis=1).reshape(-1)
    order = np.lexsort((np.maximum(starts, ends), np.minimum(starts, ends)))
    bounds = np.stack((np.minimum(starts, ends), np.maximum(starts, ends)), axis=-1)[order]
    changes = np.flatnonzero(np.any(bounds[1:] != bounds[:-1], axis=1)) + 1
    runs = np.concatenate(([0], changes, [len(order)]))
    pairs = runs[:-1][np.diff(runs) == 2]
    left, right = order[pairs], order[pairs + 1]
    reversed_edge = (starts[left] == ends[right]) & (ends[left] == starts[right])
    triangles = vertices[indices]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths > 0, lengths, 1)[:, None]
    face_a, face_b = left // 3, right // 3
    normal_delta = np.linalg.norm(normals[face_a] - normals[face_b], axis=1)
    smooth = (
        reversed_edge & (face_a != face_b) & (lengths[face_a] > 0)
        & (lengths[face_b] > 0) & (normal_delta < 1.0e-5)
    )
    internal = np.concatenate((left[smooth], right[smooth]))
    masks = np.zeros(len(indices), dtype=np.uint8)
    np.bitwise_or.at(masks, internal // 3, (1 << (internal % 3)).astype(np.uint8))
    # A vertex is internal only if all incident edges passed the smooth-edge test.
    sharp = (masks[:, None] & (1 << np.arange(3))) == 0
    internal_vertices = np.ones(vertex_ids.size, dtype=bool)
    internal_vertices[starts[sharp.flatten()]] = False
    internal_vertices[ends[sharp.flatten()]] = False
    vertex_bits = internal_vertices[faces] * (1 << np.arange(3, 6))
    return masks | vertex_bits.sum(axis=1).astype(np.uint8)


class TerrainCollision:
    """Static mesh BVH with per-pair face caches, or analytic wheel contacts on an infinite plane."""

    def __init__(
        self,
        model: newton.Model,
        *,
        ground_label: str = "ground_mesh",
        ground_shape: int | None = None,
        queries: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> None:
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
        if model.shape_body.numpy()[ground] >= 0:
            raise ValueError("terrain must be static")
        self.wheel_plane = kinds[ground] == GeoType.PLANE and queries is not None
        if not self.wheel_plane:
            if kinds[ground] != GeoType.MESH:
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
        if self.wheel_plane:
            scales = model.shape_scale.numpy()
            widths = scales[queries, 1]
            if (
                np.any(scales[ground, :2] != 0)
                or np.any(kinds[queries] != GeoType.CYLINDER)
                or np.any(widths <= 0) or np.any(widths != scales[queries, 2])
                or np.any(model.shape_body.numpy()[queries] < 0)
            ):
                raise ValueError("wheel-plane contacts require moving rounded wheels and an infinite plane")
        elif not np.isin(kinds[queries], _SUPPORTED_PRIMITIVES).all():
            raise ValueError("terrain contacts support sphere, capsule, box, and cylinder primitives")
        if not self.wheel_plane:
            cylinders = queries[kinds[queries] == GeoType.CYLINDER]
            smooth_features = np.zeros(len(indices), dtype=np.uint8)
            if len(cylinders):
                scales = model.shape_scale.numpy()[cylinders]
                if np.any((scales[:, 1] == scales[:, 2]) & (scales[:, 2] > 0)):
                    smooth_features = _coplanar_features(vertices, indices)
            # A few coplanar edges on an otherwise rough mesh do not repay
            # the extra register pressure of the wheel-specific kernel.
            coplanar = np.count_nonzero(smooth_features) >= 0.1 * len(indices)
            self.detect_mesh = _detect_coplanar_wheels if coplanar else _detect_mesh
            self.smooth_features = wp.array(smooth_features, dtype=wp.uint8, device=self.device)
        # Schedule corresponding links together so adjacent GPU lanes take similar paths.
        worlds = model.shape_world.numpy()
        _, starts, world_indices = np.unique(worlds, return_index=True, return_inverse=True)
        ordinal = queries - starts[world_indices[queries]]
        queries = queries[np.lexsort((queries, ordinal, kinds[queries]))]
        self.query_count = len(queries)
        self.queries = wp.array(queries, dtype=int, device=self.device)
        with wp.ScopedDevice(self.device):
            if not self.wheel_plane:
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

    def invalidate(self) -> None:
        if not self.wheel_plane:
            self.counts.fill_(INVALID_CACHE)

    def collide(self, state: newton.State, contacts: newton.Contacts | None  =  None) -> None:
        m = self.model
        c = contacts
        if c is None:
            c = self.contacts
            c.clear()
        if self.query_count:
            if self.wheel_plane:
                wp.launch(
                    _detect_wheel_plane, self.query_count,
                    inputs=[self.queries, self.ground, state.body_q, m.shape_body,
                            m.shape_transform, m.shape_scale, m.shape_gap, m.shape_margin,
                            self.scores, self.candidates], device=self.device,
                )
            else:
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
                    self.detect_mesh,
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
                        self.smooth_features,
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
    """Route supported mesh and wheel–plane pairs to terrain kernels and the rest to Newton."""

    def __init__(
        self,
        model: newton.Model,
        terrain_shapes: Sequence[int],
        *,
        deterministic: bool  =  False,
    ) -> None:
        self.model = model
        self.terrain_colliders = []
        kinds = model.shape_type.numpy()
        bodies = model.shape_body.numpy()
        mesh_shapes = [
            index for index in terrain_shapes
            if kinds[index] == GeoType.MESH and bodies[index] < 0
        ]
        scales = model.shape_scale.numpy()
        wheel = (kinds == GeoType.CYLINDER) & (scales[:, 1] == scales[:, 2]) & (scales[:, 2] > 0)
        planes = [
            i for i in terrain_shapes if kinds[i] == GeoType.PLANE
            and bodies[i] < 0 and np.all(scales[i, :2] == 0)
        ] if wheel.any() else []
        self.newton = None
        mesh_properties = model._shape_mesh_properties
        has_surface_velocity = mesh_properties is not None and np.any(
            mesh_properties.numpy() & int(MeshProperties.SURFACE_VELOCITY)
        )
        if (
            (mesh_shapes or planes)
            and not model._has_body_variants
            and not has_surface_velocity
            and not np.any(model.shape_flags.numpy() & int(newton.ShapeFlags.HYDROELASTIC))
        ):
            remaining = model.shape_contact_pairs.numpy().reshape(-1, 2)
            for ground in [*mesh_shapes, *planes]:
                ground_first = remaining[:, 0] == ground
                other = np.where(ground_first, remaining[:, 1], remaining[:, 0])
                owned = (ground_first | (remaining[:, 1] == ground)) & np.isin(
                    kinds[other], _SUPPORTED_PRIMITIVES
                )
                if kinds[ground] == GeoType.PLANE:
                    owned &= wheel[other] & (bodies[other] >= 0)
                if owned.any():
                    self.terrain_colliders.append(
                        TerrainCollision(
                            model,
                            ground_shape=ground,
                            queries=other[owned],
                            deterministic=deterministic,
                        )
                    )
                    remaining = remaining[~owned]

        if not self.terrain_colliders:
            self.newton = newton.CollisionPipeline(model, deterministic=deterministic)
            return

        self.custom_capacity = sum(item.contact_capacity for item in self.terrain_colliders)
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

    def contacts(self) -> newton.Contacts:
        if self.newton is not None:
            return self.newton.contacts()
        return newton.Contacts(self.custom_capacity, 0, device=self.model.device)

    def collide(self, state: newton.State, contacts: newton.Contacts) -> None:
        if self.newton is not None:
            self.newton.collide(state, contacts)
        else:
            contacts.clear()
        for collider in self.terrain_colliders:
            collider.collide(state, contacts)
