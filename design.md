This is a new RL codebase based on Newton.

It serves to test if newton's collision pipeline is more flexible and potentially more efficient than Isaac's physx backend.

Compared to mujoco_warp/mjlab, this codebase should support complex, potentially non-convex terrains.

The overall structure will be similar to aa except that we can avoid many ofthe IsaacSim complications.

We will also add custom collision support for a wheeled quadruped, since both cylinder and sphere are poor approximations for the wheels. Possible solutions include analytic collision or SDF with a solid torus.