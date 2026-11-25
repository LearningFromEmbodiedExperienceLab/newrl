"""Verify that viewer follow updates yield to browser camera motion."""

import unittest

import numpy as np

from newrl.visualizations.viser import _ClientView


class _Camera:
    def __init__(self):
        self.look_at = np.zeros(3)
        self._position = np.array([5.0, 0.0, 2.0])

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self.look_at += position - self._position
        self._position = position


class ViserFollowTests(unittest.TestCase):
    def test_follow_yields_to_orbit_and_resumes_with_new_angle(self):
        view = _ClientView(None, None, None, None)
        camera = _Camera()
        selection = (0, "robot")
        target = np.array([1.0, 0.0, 0.0])
        view.follow_robot(camera, selection, target, 0.0)
        np.testing.assert_allclose(camera.look_at, target)

        target = target + [0.1, 0.0, 0.0]
        view.follow_robot(camera, selection, target, 0.08)
        np.testing.assert_allclose(camera.look_at, target)

        for now, orbit in ((0.1, [4.0, 3.0, 2.0]), (0.21, [3.0, 4.0, 2.0])):
            target = target + [0.1, 0.0, 0.0]
            camera._position = camera.look_at + orbit
            moved = camera.position.copy()
            view.follow_robot(camera, selection, target, now)
            np.testing.assert_allclose(camera.position, moved)

        view.follow_robot(camera, selection, target, 0.35)
        np.testing.assert_allclose(camera.position, moved)
        view.follow_robot(camera, selection, target, 0.43)
        np.testing.assert_allclose(camera.look_at, target)
        np.testing.assert_allclose(camera.position - camera.look_at, orbit)
