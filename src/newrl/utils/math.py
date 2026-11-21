"""Quaternion helpers. Quaternions are Newton's ``(x, y, z, w)``."""

import torch


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to ``[-pi, pi]``."""
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions ``(x, y, z, w)``."""
    xyz = quat[..., :3]
    t = xyz.cross(vec, dim=-1) * 2
    return vec + quat[..., 3:] * t + xyz.cross(t, dim=-1)


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply the inverse quaternion rotation to vectors."""
    xyz = quat[..., :3]
    t = xyz.cross(vec, dim=-1) * 2
    return vec - quat[..., 3:] * t + xyz.cross(t, dim=-1)


def yaw_rotate(yaw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors about Z by yaw angles in radians."""
    yaw = yaw.reshape(vec.shape[:-1])
    cos = yaw.cos()
    sin = yaw.sin()
    vec = vec.expand(*yaw.shape, 3)
    return torch.stack(
        (
            cos * vec[..., 0] - sin * vec[..., 1],
            sin * vec[..., 0] + cos * vec[..., 1],
            vec[..., 2],
        ),
        dim=-1,
    )


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion ``(x, y, z, w)``."""
    return torch.cat(
        (
            torch.zeros_like(yaw).unsqueeze(-1),
            torch.zeros_like(yaw).unsqueeze(-1),
            (yaw / 2).sin().unsqueeze(-1),
            (yaw / 2).cos().unsqueeze(-1),
        ),
        dim=-1,
    )


def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Drop roll and pitch, keeping the yaw of ``quat``."""
    x, y, z, w = quat.unbind(dim=-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    return quat_from_yaw(yaw)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions ``(x, y, z, w)``."""
    q1, q2 = torch.broadcast_tensors(q1, q2)
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )
