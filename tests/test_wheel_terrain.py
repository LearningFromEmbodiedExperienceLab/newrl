"""Check wheel clearance against the known surface of a heightfield slope."""

import math
import unittest

import newton
import numpy as np
import torch
import warp as wp

from newrl.configs import AS2WLocomotion, M20SLocomotion
from newrl.env import StatefulEnvironment
from newrl.terrain import GeneratedTerrainConfig, PyramidSlopeTileConfig
from newrl.utils.math import quat_mul, quat_rotate, quat_rotate_inverse


def _wheel_clearance(env, normals, offset, kind):
    """Measure signed distance from the wheel collision support to the slope."""
    model = env.model
    wheel_bodies = env.scene.contact_sensors["feet_ground"].body_ids
    body_ids = wheel_bodies.flatten()
    shape_bodies = model.shape_body.numpy()
    collides = model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)
    shape_ids = []
    for body in body_ids.tolist():
        (shape,) = np.flatnonzero((shape_bodies == body) & (collides != 0))
        shape_ids.append(int(shape))
    local_pose = wp.to_torch(model.shape_transform)[shape_ids]
    scales = wp.to_torch(model.shape_scale)[shape_ids]
    kinds = model.shape_type.numpy()[shape_ids]
    np.testing.assert_array_equal(kinds, np.full_like(kinds, kind))
    vertices = None
    if kind == newton.GeoType.CONVEX_MESH:
        vertices = [
            torch.as_tensor(model.shape_source[shape].vertices, device=env.device)
            * scale
            for shape, scale in zip(shape_ids, scales, strict=True)
        ]

    def measure():
        poses = wp.to_torch(env.state.body_q)
        bodies = poses[body_ids]
        center = bodies[:, :3] + quat_rotate(bodies[:, 3:], local_pose[:, :3])
        rotation = quat_mul(bodies[:, 3:], local_pose[:, 3:])
        local_normal = quat_rotate_inverse(rotation, normals)
        if kind == newton.GeoType.SPHERE:
            support = -scales[:, 0]
        elif kind == newton.GeoType.CYLINDER:
            support = -(
                scales[:, 0] * local_normal[:, :2].norm(dim=-1)
                + scales[:, 1] * local_normal[:, 2].abs()
            )
        else:
            support = torch.stack(
                [
                    (points @ normal).amin()
                    for points, normal in zip(vertices, local_normal, strict=True)
                ]
            )
        distance = (center * normals).sum(dim=-1) + support - offset
        return distance.reshape_as(wheel_bodies)

    return measure


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class WheelTerrainTests(unittest.TestCase):
    def test_wheel_penetration_on_heightfield_slopes(self):
        slope = 0.25
        for factory, sphere_wheels, wheel_kind in (
            (M20SLocomotion, False, newton.GeoType.CYLINDER),
            (AS2WLocomotion, False, newton.GeoType.CONVEX_MESH),
            (AS2WLocomotion, True, newton.GeoType.SPHERE),
        ):
            for inverted in (False, True):
                for collision in ("newton", "mujoco"):
                    with self.subTest(
                        robot=factory.__name__, wheels=wheel_kind.name,
                        inverted=inverted, collision=collision,
                    ):
                        terrain = GeneratedTerrainConfig(
                            tiles={
                                "slope": PyramidSlopeTileConfig(
                                    slope=slope, inverted=inverted
                                )
                            },
                            rows=1,
                        )
                        options = {"sphere_wheels": True} if sphere_wheels else {}
                        cfg = factory(
                            num_envs=2,
                            terrain=terrain,
                            collision=collision,
                            deterministic=True,
                            graph_capture=False,
                            output_physical_fields=(("feet_ground", "loaded"),),
                            **options,
                        )
                        cfg.mdp.joint_position_noise = 0.0
                        cfg.mdp.root_xy_noise = 0.0
                        cfg.mdp.root_yaw_noise = 0.0
                        env = StatefulEnvironment(cfg)
                        env.reset()
                        robot = env.scene.entities["robot"]
                        sides = torch.tensor([1.0, -1.0], device=env.device)
                        gradient = sides * (-slope if inverted else slope)
                        length = math.hypot(1.0, slope)
                        normals = torch.zeros((2, 3), device=env.device)
                        normals[:, 0] = gradient / length
                        normals[:, 2] = 1.0 / length
                        wheel_bodies = env.scene.contact_sensors["feet_ground"].body_ids
                        offset = (-1 if inverted else 1) * slope * terrain.tile_size / (2 * length)
                        wheel_normals = normals.repeat_interleave(wheel_bodies.shape[1], dim=0)
                        clearance = _wheel_clearance(env, wheel_normals, offset, wheel_kind)
                        with env._stream():
                            root = wp.to_torch(env.state.joint_q)
                            root[robot.root_q[:, 0]] = sides * 2.5
                            pitch = gradient.atan()
                            root[robot.root_q[:, 4]] = (pitch / 2).sin()
                            root[robot.root_q[:, 6]] = (pitch / 2).cos()
                            newton.eval_fk(
                                env.model, env.state.joint_q, env.state.joint_qd, env.state
                            )
                            root[robot.root_q[:, 2]] += (0.02 - clearance().amin(dim=1)) * length
                            newton.eval_fk(
                                env.model, env.state.joint_q, env.state.joint_qd, env.state
                            )

                        action = torch.zeros((2, env.action_dim), device=env.device)
                        action[:, robot.velocity_mask] = (0.5 * sides)[:, None]
                        gaps = []
                        loads = []
                        wheel_positions = []
                        for _ in range(20):
                            step = env.step(action)
                            gaps.append(clearance())
                            loads.append(step["physical_state", "feet_ground", "loaded"])
                            poses = wp.to_torch(env.state.body_q)[wheel_bodies]
                            wheel_positions.append(poses[..., :2])
                        gaps = torch.stack(gaps)
                        x, y = torch.stack(wheel_positions).abs().unbind(dim=-1)
                        # Leave room for the whole wheel on the planar part of each face.
                        pad_edge = (
                            terrain.spawn_pad_width / 2
                            + terrain.tiles["slope"].sample_spacing
                        )
                        self.assertTrue(
                            ((x > pad_edge + 0.15) & (x > y + 0.15)).all().item()
                        )
                        self.assertTrue((x < terrain.tile_size / 2 - 0.15).all().item())
                        self.assertTrue(
                            torch.stack(loads).any(dim=0).all().item(),
                            "Every wheel must load against the heightfield during the rollout",
                        )
                        # Allow the observed transient penetration of soft contacts.
                        self.assertGreaterEqual(gaps.min().item(), -0.015)
                        self.assertGreaterEqual(
                            gaps[-1].min().item(), -0.006,
                            "Wheel penetration should recede after touchdown",
                        )
