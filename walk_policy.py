"""Batched Anymal C locomotion policy shipped with Newton.

The checkpoint is ``rl_policies/mjw_anymal.onnx`` from ``anybotics_anymal_c``.
It expects the MuJoCo joint order and the observation layout used by
``newton.examples.robot.example_robot_policy``.
"""

from __future__ import annotations

from pathlib import Path

import warp as wp
import yaml
from warp_nn.runtime import OnnxRuntime

_NUM_DOFS = 12
_OBS_DIM = 48


@wp.kernel
def _observe(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_pos_initial: wp.array[float],
    gravity_w: wp.vec3,
    command: wp.array[float],
    prev_act: wp.array2d[float],
    q_stride: int,
    qd_stride: int,
    obs: wp.array2d[float],
):
    world = wp.tid()
    q0 = world * q_stride
    qd0 = world * qd_stride
    rot = wp.quat(joint_q[q0 + 3], joint_q[q0 + 4], joint_q[q0 + 5], joint_q[q0 + 6])
    rot_inv = wp.quat_inverse(rot)
    lin_vel_b = wp.quat_rotate(rot_inv, wp.vec3(joint_qd[qd0 + 0], joint_qd[qd0 + 1], joint_qd[qd0 + 2]))
    ang_vel_b = wp.quat_rotate(rot_inv, wp.vec3(joint_qd[qd0 + 3], joint_qd[qd0 + 4], joint_qd[qd0 + 5]))
    grav = wp.quat_rotate(rot_inv, gravity_w)
    obs[world, 0] = lin_vel_b[0]
    obs[world, 1] = lin_vel_b[1]
    obs[world, 2] = lin_vel_b[2]
    obs[world, 3] = ang_vel_b[0]
    obs[world, 4] = ang_vel_b[1]
    obs[world, 5] = ang_vel_b[2]
    obs[world, 6] = grav[0]
    obs[world, 7] = grav[1]
    obs[world, 8] = grav[2]
    obs[world, 9] = command[0]
    obs[world, 10] = command[1]
    obs[world, 11] = command[2]
    for k in range(_NUM_DOFS):
        obs[world, 12 + k] = joint_q[q0 + 7 + k] - joint_pos_initial[k]
        obs[world, 24 + k] = joint_qd[qd0 + 6 + k]
        obs[world, 36 + k] = prev_act[world, k]


@wp.kernel
def _apply_actions(
    act: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    action_scale: float,
    q_stride: int,
    joint_target_q: wp.array[float],
):
    world, joint = wp.tid()
    joint_target_q[world * q_stride + 7 + joint] = joint_pos_initial[joint] + action_scale * act[world, joint]


class AnymalWalkPolicy:
    """Writes joint position targets for every replicated Anymal C."""

    def __init__(self, model, num_envs: int, command: tuple[float, float, float]):
        self.num_envs = num_envs
        self.q_stride = model.joint_coord_count // num_envs
        self.qd_stride = model.joint_dof_count // num_envs
        if self.q_stride < 7 + _NUM_DOFS or self.qd_stride < 6 + _NUM_DOFS:
            raise RuntimeError(
                f"expected at least 19 coordinates and 18 dofs per world, got {self.q_stride} and {self.qd_stride}"
            )
        config = self.config()
        self.policy = OnnxRuntime(self.policy_path())
        self.input_name = self.policy.input_names[0]
        self.output_name = self.policy.output_names[0]
        self.obs = wp.zeros((num_envs, _OBS_DIM), dtype=float)
        self.prev_act = wp.zeros((num_envs, _NUM_DOFS), dtype=float)
        self.joint_pos_initial = wp.array([float(v) for v in config["mjw_joint_pos"]], dtype=float)
        self.command = wp.array(list(command), dtype=float)
        self.action_scale = float(config["action_scale"])
        self.gravity_w = wp.vec3(0.0, 0.0, -1.0)

    @staticmethod
    def asset_dir() -> Path:
        import newton

        return Path(newton.utils.download_asset("anybotics_anymal_c"))

    @classmethod
    def policy_path(cls) -> str:
        return str(cls.asset_dir() / "rl_policies" / "mjw_anymal.onnx")

    @classmethod
    def config(cls) -> dict:
        path = cls.asset_dir() / "rl_policies" / "anymal.yaml"
        return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)

    def apply(self, state, control) -> None:
        wp.launch(
            _observe,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.joint_pos_initial,
                self.gravity_w,
                self.command,
                self.prev_act,
                self.q_stride,
                self.qd_stride,
                self.obs,
            ],
        )
        act = self.policy({self.input_name: self.obs})[self.output_name]
        wp.launch(
            _apply_actions,
            dim=(self.num_envs, _NUM_DOFS),
            inputs=[act, self.joint_pos_initial, self.action_scale, self.q_stride, control.joint_target_q],
        )
        wp.copy(self.prev_act, act)
