"""Rendering and GUI parameter regressions without opening a Tk window."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from galaxy_gui import GalaxyApp
except ModuleNotFoundError as error:
    if error.name not in ("tkinter", "_tkinter"):
        raise
    GalaxyApp = None


class Value:
    """A minimal stand-in for the get() interface of a Tk variable."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


@unittest.skipIf(GalaxyApp is None, "optional Tkinter is not available")
class TestGalaxyViewer(unittest.TestCase):
    def app(self):
        # __init__ creates the actual Tk root; no display is needed here.
        app = GalaxyApp.__new__(GalaxyApp)
        app.canvas_size = 600
        return app

    def test_start_uses_displayed_timestep_and_softening(self):
        app = self.app()
        for name, value in {"n_var": 120, "dt_var": 0.017, "eps_var": 0.075,
                            "iter_var": 80, "integrator_var": "leapfrog",
                            "initial_var": "two_galaxies", "mode_var": "direct"}.items():
            setattr(app, name, Value(value))
        app.update_simulation = Mock()
        with patch("galaxy_gui.BarnesHutSimulation") as simulation:
            app.start()
        simulation.assert_called_once_with(
            num_particles=120, dt=0.017, eps=0.075,
            integrator="leapfrog", initial="two_galaxies", mode="direct")
        self.assertIs(app.sim, simulation.return_value)
        self.assertEqual(app.current_iter, 0)
        self.assertEqual(app.total_iter, 80)
        app.update_simulation.assert_called_once_with()

    def test_projection_keeps_visible_coordinates_without_clamping(self):
        app = self.app()
        self.assertEqual(app.project(0, 0, 0), (300, 300))
        self.assertEqual(app.project(1, 2, 0), (350, 200))
        # The perspective scale decreases with distance from the camera.
        self.assertEqual(app.project(1, 2, 3), (325, 250))
        for position in ((100, 0, 0), (-100, 0, 0), (0, 100, 0), (0, -100, 0)):
            with self.subTest(position=position):
                self.assertIsNone(app.project(*position))

    def test_projection_culls_camera_plane_behind_camera_and_near_plane(self):
        app = self.app()
        for depth in (-1, 0, app.NEAR_PLANE / 2):
            with self.subTest(depth=depth):
                self.assertIsNone(app.project(0, 0, -app.CAMERA_DISTANCE + depth))
        self.assertEqual(app.project(0, 0, -app.CAMERA_DISTANCE + 2 * app.NEAR_PLANE),
                         (300, 300))
        for position in ((float("nan"), 0, 0), (0, float("inf"), 0), (0, 0, -float("inf"))):
            with self.subTest(position=position):
                self.assertIsNone(app.project(*position))

    def test_draw_skips_invisible_particles(self):
        app = self.app()
        app.canvas = Mock()
        app.color_var = Value(False)
        app.sim = SimpleNamespace(particles=[
            SimpleNamespace(x=0, y=0, z=0),
            SimpleNamespace(x=100, y=0, z=0),
            SimpleNamespace(x=0, y=0, z=-3),
            SimpleNamespace(x=0, y=0, z=-4),
        ])
        app.draw()
        app.canvas.delete.assert_called_once_with("all")
        app.canvas.create_oval.assert_called_once_with(298, 298, 302, 302,
                                                      fill="white", outline="")


if __name__ == "__main__":
    unittest.main()
