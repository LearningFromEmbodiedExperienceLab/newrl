"""
A single-file, minimal, hydra-flavor config system. Pure python-based. No yaml.

It should allow the following pattern as hydra does:
python scripts/some_script.py task=AnymalLocoFlat task.solver=steel algo=PPO num_envs=256

Compared to hydra, we want something different:
1. No DictConfig. Everything resolves to a dataclass instance, such that `__post_init__` can be used to validate the config.
2. More flexible dataclass support. Hydra does not support non-primitive fields. We should do. All current dataclasses should be compatible.

We will use a registry (similar to hydra's ConfigStore) to register config classes.
Example:
cs = ConfigStore()
cs.store(name="ppo", node=PPOConfig, group="algo")
cs.store(name="ppo_large", node=PPOConfig(...), group="algo")
cs.store(name="anymal_loco_flat", node=AnymalLocoFlatConfig(terrain=...), group="task")
cs.store(name="anymal_loco_rough", node=AnymalLocoFlatConfig(terrain=...), group="task")

And the (nested) config should be able to be saved to and loaded from disk.
I am not sure what is the format to use. Json and Yaml are not type-flexible. Maybe also python file?
"""