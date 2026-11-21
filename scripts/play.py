"""Run an ANYmal rollout with zero position offsets and optional visualization."""

import argparse

import newton
import torch

from newrl.configs import AnymalLocomotion
from newrl.env import StatefulEnvironment
from newrl.mdp.episode_ends import Timeout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, help="stop after this many steps")
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument(
        "--solver-backend", choices=("mujoco", "steel"), default="mujoco"
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open a Newton GL viewer",
    )
    parser.add_argument(
        "--graph", action="store_true", help="capture the physics loop in a CUDA graph"
    )
    args = parser.parse_args()
    if min(args.num_envs, args.episode_steps) < 1 or (
        args.steps is not None and args.steps < 1
    ):
        parser.error("--num-envs, --steps, and --episode-steps must be positive")

    viewer = newton.viewer.ViewerGL() if args.viewer else None
    try:
        config = AnymalLocomotion(
            num_envs=args.num_envs,
            graph_capture=args.graph,
            solver_backend=args.solver_backend,
        )
        config.mdp.truncations["timeout"] = Timeout(args.episode_steps)
        env = StatefulEnvironment(config)
        actions = torch.zeros((env.num_envs, env.action_dim), device=env.device)
        if viewer is not None:
            viewer.set_model(env.model)
            if env.num_envs > 1:
                viewer.set_world_offsets((2.0, 2.0, 0.0))
        env.reset()
        steps = resets = 0
        while (args.steps is None or steps < args.steps) and (
            viewer is None or viewer.is_running()
        ):
            if viewer is None or not viewer.is_paused():
                transition = env.step(actions)
                steps += 1
                finished = transition["done"].squeeze(-1).nonzero(as_tuple=True)[0]
            else:
                finished = None
            if viewer is not None:
                viewer.begin_frame(steps * env.control_dt)
                viewer.log_state(env.state)
                viewer.end_frame()
            if finished is not None and finished.numel():
                env.reset(finished)
                resets += finished.numel()
        print(f"worlds={env.num_envs} control_steps={steps} world_resets={resets}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
