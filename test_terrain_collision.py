import unittest

import newton
import numpy as np
import warp as wp

from newrl.terrain import flat_ground_mesh
from terrain_collision import TerrainCollision


def triangle():
    return newton.Mesh(
        np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=np.float32), np.array([0, 1, 2], dtype=np.int32)
    )


def pose(x, y, z):
    return wp.transform(wp.vec3(x, y, z), wp.quat_identity())


def setup(mesh, xyz, kind="sphere", *, ground_pose=None, scale=(1, 1, 1), margin=0.0):
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(gap=0.0, margin=margin)
    builder.add_shape_mesh(body=-1, mesh=mesh, xform=ground_pose, scale=scale, cfg=cfg, label="ground_mesh")
    body = builder.add_body(xform=pose(*xyz))
    if kind == "sphere":
        builder.add_shape_sphere(body=body, radius=0.1, cfg=cfg)
    elif kind == "box":
        builder.add_shape_box(body=body, hx=0.12, hy=0.09, hz=0.1, cfg=cfg)
    elif kind == "cylinder":
        builder.add_shape_cylinder(body=body, radius=0.12, half_height=0.1, cfg=cfg)
    else:
        builder.add_shape_capsule(body=body, radius=0.06, half_height=0.04, cfg=cfg)
    model = builder.finalize()
    return TerrainCollision(model), model.state()


def contacts(pipeline, state):
    pipeline.collide(state)
    c = pipeline.contacts
    count = c.rigid_contact_count.numpy()[0]
    return c.rigid_contact_point0.numpy()[:count], c.rigid_contact_normal.numpy()[:count]


class TestTerrainCollision(unittest.TestCase):
    def test_edge_vertex_and_teleport(self):
        p, s = setup(triangle(), (-0.06, 0.3, 0.06))
        points, normals = contacts(p, s)
        np.testing.assert_allclose(points, [[0, 0.3, 0]], atol=1e-6)
        np.testing.assert_allclose(normals, [[-(2**-0.5), 0, 2**-0.5]], atol=1e-6)
        cached_lo = p.cached_lo.numpy()
        self.assertEqual(int(p.counts.numpy()[0]), 1)
        for xyz, expect in [((-0.07, 0.3, 0.06), 1), ((5, 5, 5), 0), ((-0.04, -0.04, 0.06), 1)]:
            s.body_q.assign(np.asarray([pose(*xyz)], dtype=np.float32))
            points, normals = contacts(p, s)
            self.assertEqual(len(points), expect)
            if xyz[0] == -0.07:
                np.testing.assert_array_equal(p.cached_lo.numpy(), cached_lo)
        np.testing.assert_allclose(points, [[0, 0, 0]], atol=1e-6)

    def test_sphere_in_corner(self):
        mesh = newton.Mesh(
            np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3]], dtype=np.float32),
            np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        )
        p, s = setup(mesh, (0.05, 0.3, 0.06))
        for _ in range(2):
            points, normals = contacts(p, s)
            self.assertEqual(len(points), 2)
            np.testing.assert_allclose(normals[np.argsort(normals[:, 0])], [[0, 0, 1], [1, 0, 0]], atol=1e-5)

    def test_margin_growth_after_empty_query(self):
        p, s = setup(triangle(), (0.4, 0.3, 0.2))
        self.assertEqual(len(contacts(p, s)[0]), 0)
        p.model.shape_margin.fill_(0.06)
        self.assertEqual(len(contacts(p, s)[0]), 1)

    def test_dense_mesh_traversal(self):
        p, s = setup(flat_ground_mesh(256, 2, 1), (0.013, 0.017, 0.06))
        for _ in range(2):
            points, normals = contacts(p, s)
            self.assertLess(p.counts.numpy()[0], 0)  # Force the uncapped traversal path.
            np.testing.assert_allclose(points, [[0.013, 0.017, 0]], atol=1e-6)
            np.testing.assert_allclose(normals, [[0, 0, 1]], atol=1e-6)

    def test_thin_slab_support(self):
        for kind in ("sphere", "capsule", "box", "cylinder"):
            with self.subTest(kind=kind):
                p, s = setup(flat_ground_mesh(2, 3, 0.02), (0.013, 0.017, 0.095), kind)
                points, normals = contacts(p, s)
                self.assertGreaterEqual(len(points), 1)
                self.assertTrue(np.all(normals[:, 2] > 0.9), normals)
                if kind in ("box", "cylinder"):
                    self.assertGreaterEqual(np.linalg.matrix_rank(points[1:, :2] - points[0, :2], tol=0.001), 2)

    def test_rotated_extent_after_empty_query(self):
        # With the long axis horizontal the box clears the triangle. Rotating
        # about its center brings the end into contact without any translation.
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(gap=0.0)
        builder.add_shape_mesh(body=-1, mesh=triangle(), cfg=cfg, label="ground_mesh")
        body = builder.add_body(xform=pose(0.5, 0.5, 0.19))
        builder.add_shape_box(body=body, hx=0.2, hy=0.03, hz=0.03, cfg=cfg)
        model = builder.finalize()
        pipeline, state = TerrainCollision(model), model.state()
        self.assertEqual(len(contacts(pipeline, state)[0]), 0)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)
        state.body_q.assign(np.array([wp.transform(wp.vec3(0.5, 0.5, 0.19), rotation)], dtype=np.float32))
        points, normals = contacts(pipeline, state)
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(normals[:, 2] > 0.9))

    def test_barrel_widest_cross_section(self):
        # The ends clear the plane. The bulged middle touches it.
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(gap=0.0)
        builder.add_shape_mesh(body=-1, mesh=triangle(), cfg=cfg, label="ground_mesh")
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.5, 0.10), rotation))
        builder.add_shape_cylinder(body=body, radius=0.03, half_height=0.1, barrel_radius=0.1, cfg=cfg)
        model = builder.finalize()
        points, normals = contacts(TerrainCollision(model), model.state())
        self.assertGreater(len(points), 0)
        self.assertTrue(np.all(normals[:, 2] > 0.9))

    def test_ground_pose_scale_and_margin(self):
        # Reflect local X and rotate local Z to world -Y. The witness lies in
        # the scaled area beyond the original triangle, inside the margin shell.
        rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
        p, s = setup(
            triangle(),
            (1, 2.885, 4.2),
            ground_pose=wp.transform(wp.vec3(5, 3, 4), rotation),
            scale=(-2, 1, 1),
            margin=0.01,
        )
        points, normals = contacts(p, s)
        np.testing.assert_allclose(points, [[1, 3, 4.2]], atol=1e-5)
        np.testing.assert_allclose(normals, [[0, -1, 0]], atol=1e-5)
        c = p.contacts
        np.testing.assert_allclose(c.rigid_contact_margin1.numpy()[0], 0.11)
        np.testing.assert_allclose(c.rigid_contact_point1.numpy()[0], 0, atol=1e-5)
