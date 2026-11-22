"""CPU checks for generated terrain geometry and preset resolution."""

import unittest
from dataclasses import replace

import newton
import numpy as np

from newrl.commands.train import TrainConfig
from newrl.terrain import GeneratedTerrainConfig
from newrl.utils.config import cs, dump, load, resolve


class GeneratedTerrainTests(unittest.TestCase):
    def test_presets_choose_native_geometry_and_spawn_anchors(self):
        for name, types in (
            ("flat_tiles", {newton.GeoType.BOX}),
            ("rough_easy", {newton.GeoType.HFIELD}),
            ("rough_hard", {newton.GeoType.HFIELD, newton.GeoType.BOX}),
        ):
            with self.subTest(name=name):
                terrain = cs.get("terrain", name)
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                anchors = terrain.build(builder)
                self.assertEqual(len(anchors), terrain.rows * len(terrain.tiles))
                self.assertEqual(set(builder.shape_type), types)
                self.assertEqual(set(builder.shape_world), {-1})
                if name == "rough_hard":
                    self.assertGreater(anchors[1][2], 0.0)
                    self.assertEqual(anchors[0][2], 0.0)

    def test_heightfield_seed_and_level_spawn_pad_and_edges(self):
        cfg = replace(cs.get("terrain", "rough_easy"), rows=1)
        heights = []
        for terrain in (cfg, cfg, replace(cfg, seed=cfg.seed + 1)):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            terrain.build(builder)
            source = builder.shape_source[0]
            heights.append(source.min_z + source.data * (source.max_z - source.min_z))
        np.testing.assert_array_equal(heights[0], heights[1])
        self.assertFalse(np.array_equal(heights[0], heights[2]))
        grid = heights[0]
        axis = np.linspace(-cfg.tile_size / 2, cfg.tile_size / 2, grid.shape[0])
        center = np.abs(axis) <= cfg.spawn_pad_width / 2
        np.testing.assert_allclose(grid[np.ix_(center, center)], 0.0, atol=1e-7)
        np.testing.assert_allclose(grid[[0, -1], :], 0.0, atol=1e-7)
        np.testing.assert_allclose(grid[:, [0, -1]], 0.0, atol=1e-7)

    def test_cli_selects_and_round_trips_terrain_preset(self):
        cfg = resolve(
            TrainConfig,
            [
                "task=AnymalLocoFlat",
                "task.terrain=rough_hard",
                "task.terrain.rows=1",
                "task.terrain.tiles.stairs.step_height=0.1",
                "algo=ppo",
            ],
        )
        self.assertIsInstance(cfg.task.terrain, GeneratedTerrainConfig)
        self.assertEqual(cfg.task.terrain.tiles["stairs"].step_height, 0.1)
        self.assertEqual(cfg.task.terrain.rows, 1)
        self.assertEqual(load(dump(cfg.task)).terrain, cfg.task.terrain)
